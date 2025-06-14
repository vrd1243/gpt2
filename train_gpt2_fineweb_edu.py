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
import numpy as np

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class MLP(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
      self.gelu = nn.GELU(approximate='tanh')
      self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
      self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):

      x = self.c_fc(x)
      x = self.gelu(x)
      x = self.c_proj(x)

      return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
      super().__init__()

      self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
      self.c_proj = nn.Linear(config.n_embd, config.n_embd)
      self.c_proj.NANOGPT_SCALE_INIT = 1

      self.n_head = config.n_head
      self.n_embd = config.n_embd

    def forward(self, x):

      B, T, C = x.size()

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
      self.mlp = MLP(config)


    def forward(self, x):
      x = x + self.attn(self.ln_1(x))
      x = x + self.mlp(self.ln_2(x))
      return x

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

      for block in self.transformer.h:
          x = block(x)
      x = self.transformer.ln_f(x)

      logits = self.lm_head(x)

      if targets == None:
          return logits, None

      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

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

        if self.current_position > len(self.input_data) - B * T * self.num_processes - 1:
            self.current_position = B * T * self.process_rank

        return inputs, targets

def load_tokens(filename):

    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:

        def __init__(self, B, T, process_rank, num_processes, split):  

            super().__init__()
            self.B = B
            self.T = T
            self.process_rank = process_rank
            self.num_processes = num_processes
            self.split = split
            assert split in {'train', 'val'}

            self.enc = tiktoken.get_encoding("gpt2")

            torch.manual_seed(1337)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(1337)


            # get the shard filenames
            data_root = "edu_fineweb10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
            self.reset()

        def reset(self):
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        def get_next_batch(self):

            B, T = self.B, self.T
        
            idx = self.current_position
            # print(self.lines[idx: idx + B*T])
            inputs = torch.tensor(self.tokens[idx: idx + B*T]).view(B, T)
            targets = torch.tensor(self.tokens[idx + 1: idx + B * T + 1]).view(B, T)

            self.current_position += B * T * self.num_processes

            if self.current_position > len(self.tokens) - B*T - 1:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
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

model = GPT2LMHeadModel.from_pretrained('gpt2')

sd_hf = model.state_dict()

for k, v in sd_hf.items():
  print(k, v.shape)

torch.set_float32_matmul_precision('high')

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix

# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# This was used for Shakespeare dataset
# max_lr = 3e-4
# min_lr = 3e-5
# warm_up_steps = 10
# max_steps = 50

max_lr = 6e-4
min_lr = max_lr * 0.1
warm_up_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

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

train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for epoch in range(max_steps):

    t0 = time.time()
    
    last_step = (epoch == max_steps - 1)

    # once in a while evaluate our validation loss
    if epoch % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.get_next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{epoch} val {val_loss_accum:.4f}\n")
            if epoch > 0 and (epoch % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{epoch:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': epoch,
                    'val_loss': val_loss_accum
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while generate from the model (except step 0, which is noise)
    if ((epoch > 0 and epoch % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = raw_model.enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = raw_model.enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # Training loop
    model.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.zero_grad()

    total_loss_accum = 0.0

    for micro_batch in range(total_grad_accum): 
      
      inputs, targets = train_loader.get_next_batch()
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

    tokens_processed = train_loader.B * train_loader.T * total_grad_accum * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"Epoch {epoch}, Train loss: {total_loss_accum:.3f}, Time: {dt:0.3f}, Norm: {norm:.3f}, Tokens: {tokens_processed}, Tokens/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()
