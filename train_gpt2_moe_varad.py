import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
import tiktoken
import random
from transformers import GPT2LMHeadModel

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Tuple
from mixture_of_experts import MoE

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    num_experts: int = 4 # number of experts for smoe

# a 3 layered MLP as the experts
class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4))
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim))
        self.act = nn.LeakyReLU(inplace = True)

    def forward(self, x):
        # Use the einstein sum functions to implement a linear layer. Instead, 
        # we could implement w1, w2 and w3 as linear layers, and then call those 
        # on the input x. 

        hidden1 = self.act(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.act(torch.einsum('end,edh->enh', hidden1, self.w2))
        out = torch.einsum('end,edh->enh', hidden2, self.w3)
        
        return out

# This is the main causal attention block. We are going to ensure that
# previous tokens are masked. 

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
      super().__init__()

      # Attention layers start with a simple linear layer, one for three 
      # parameter types --- k, q, and c. An embedding of side C, will be 
      # transformed by the c_attn to an embedding of size 3*C. This will be
      # split later into three embeddings, each of size C. 

      self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
      self.c_proj = nn.Linear(config.n_embd, config.n_embd)
      self.c_proj.NANOGPT_SCALE_INIT = 1

      # We are using multihead attention. 
      self.n_head = config.n_head
      self.n_embd = config.n_embd

    def forward(self, x):

      B, T, C = x.size()
      
      # Get the attention unified tensor, and split it into three parameters.
      qkv = self.c_attn(x) # B, T, 3*C
      q, k, v = qkv.split(self.n_embd, dim=2) # each is of B, T, C

      
      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # B, nh, T, head_size
      y = y.transpose(1,2).contiguous().view(B, T, C) # B, T, C
      y = self.c_proj(y) # B, T, C

      return y

class Block(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.ln_1 = nn.LayerNorm(config.n_embd)
      self.attn = CausalSelfAttention(config)
      self.ln_2 = nn.LayerNorm(config.n_embd)
      # self.smoe = SparseMoE(config)
      self.experts = Experts(config.n_embd, num_experts = 2)
      self.moe = MoE(dim = config.n_embd, num_experts = 2, experts = self.experts)

    def forward(self, x):
        
      x = x + self.attn(self.ln_1(x))

      # Handle MoE output properly
      moe_output = self.moe(self.ln_2(x))
        
      if isinstance(moe_output, tuple):
          moe_out, aux_loss = moe_output
          x = x + moe_out
          return x, aux_loss
      else:
          x = x + moe_output
          return x, None
            
class GPT(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.config = config

      self.transformer = nn.ModuleDict({
          'wte': nn.Embedding(config.vocab_size, config.n_embd),
          'wpe': nn.Embedding(config.block_size, config.n_embd),
          'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
          'ln_f': nn.LayerNorm(config.n_embd),
      })

      self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
      self.transformer.wte.weight = self.lm_head.weight

      self.apply(self._init_weights)

      self.enc = tiktoken.get_encoding("gpt2")


    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          std = 0.02
          if hasattr(module, 'NANOGPT_SCALE_INIT'):
              std *= (2 * self.config.n_layer) ** -0.5
          torch.nn.init.normal_(module.weight, mean=0.0, std=std)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

      B, T = idx.size()

      pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

      tok_emb = self.transformer.wte(idx)
      pos_emb = self.transformer.wpe(pos)
      x = tok_emb + pos_emb

      # Collect auxiliary losses from MoE layers
      total_aux_loss = 0.0
      aux_loss_count = 0
      
      for block in self.transformer.h:
          
          block_output = block(x)
          
          if isinstance(block_output, tuple):
              
              x, aux_loss = block_output
              
              if aux_loss is not None:
                  total_aux_loss += aux_loss
                  aux_loss_count += 1
          else:
              
              x = block_output
          
      x = self.transformer.ln_f(x)

      logits = self.lm_head(x)

      if targets == None:
          return logits, None

      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

      if aux_loss_count > 0:
          
          avg_aux_loss = total_aux_loss / aux_loss_count
          # Weight the auxiliary loss (common values are 0.01 to 0.1)
          loss = loss + 0.01 * avg_aux_loss

      return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['block_size'] = 1024
        config_args['vocab_size'] = 50257

        config = GPTConfig(**config_args)

        # Load my model, and get keys

        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # Load the HF GPT 2 model transformer.
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()
        sd_hf_keys = sd_hf.keys()

        # We remove certain keys fromt the hf model.
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('. c_attn.masked_bias')]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith('.attn.bias')]

        assert len(sd_hf_keys) == len(sd_keys), f"mismatched keys {len(sd_hf_keys)} != {len(sd_keys)}"

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_hf_keys:

            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                sd[k].copy_(sd_hf[k])

        return model

    def generate(self, x, max_length):

        x = self.enc.encode(x)

        x = torch.tensor(x).to(device).view(1,-1)

        for iter in range(max_length):

          logits, loss = self(x)
          out = logits[:, -1, :]

          out_softmax = nn.Softmax(dim=1)(out)
          top_k_probs, top_k_idx = torch.topk(out_softmax, 50)

          select = torch.multinomial(top_k_probs, 1)
          out = torch.gather(top_k_idx, -1, select)
          x = torch.concat((x, out), dim=-1)

        out_list = x.cpu().detach().tolist()


        return enc.decode(out_list[0])

    def configure_optimizers(self, weight_decay, learning_rate, device_type):

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in non_decay_params)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

class DataLoader:

    def __init__(self, B, T, process_rank, num_processes):  
        super().__init__()
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.enc = tiktoken.get_encoding("gpt2")

        torch.manual_seed(1337)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        with open('input.txt', 'r') as f:
            lines = f.readlines()

        self.current_position = self.B * self.T * self.process_rank

        self.input_data = self.enc.encode(''.join(lines))

    def get_next_batch(self):

        B, T = self.B, self.T

        idx = self.current_position
        # print(self.lines[idx: idx + B*T])
        inputs = torch.tensor(self.input_data[idx: idx + B*T]).view(B, T)
        targets = torch.tensor(self.input_data[idx + 1: idx + B * T + 1]).view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position > len(self.input_data) - B*T - 1:
            self.current_position = B * T * self.process_rank

        return inputs, targets

def get_lr(step):

    if step < warm_up_steps:
        return (step + 1) / warm_up_steps * max_lr

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
device = 'cuda:0'

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:

    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
device_type = "cuda" if device.startswith("cuda") else "cpu"

# model = GPT2LMHeadModel.from_pretrained('gpt2')

# sd_hf = model.state_dict()

# for k, v in sd_hf.items():
#  print(k, v.shape)

# torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 3e-4
min_lr = 3e-5
warm_up_steps = 10
max_steps = 50

# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
total_grad_accum = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {total_grad_accum}")

loader = DataLoader(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

for epoch in range(max_steps):

    t0 = time.time()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.zero_grad()

    total_loss_accum = 0.0

    for micro_batch in range(total_grad_accum): 
      
      inputs, targets = loader.get_next_batch()
      inputs, targets = inputs.to(device), targets.to(device)

      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):

          logits, loss = model(inputs, targets)
          loss = loss / (total_grad_accum)

      total_loss_accum += loss
    
      if ddp:
        model.require_backward_grad_sync = (micro_batch == total_grad_accum - 1)

      loss.backward()

    if ddp:
        dist.all_reduce(total_loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0

    optimizer.step()

    tokens_processed = loader.B * loader.T * total_grad_accum * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"Epoch {epoch}, Train loss: {total_loss_accum:.3f}, Time: {dt:0.3f}, Norm: {norm:.3f}, Tokens: {tokens_processed}, Tokens/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
