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

# a 3 layered MLP as an expert

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
      
      # A linear layer.  
      self.c_proj = nn.Linear(config.n_embd, config.n_embd)

      # This will be used later to decide weight initialization. Find this 
      # later in the code.
      self.c_proj.NANOGPT_SCALE_INIT = 1

      # We are using multihead attention. 
      self.n_head = config.n_head
      self.n_embd = config.n_embd

    def forward(self, x):

      B, T, C = x.size()
      
      # Get the attention unified tensor, and split it into three parameters.
      
      qkv = self.c_attn(x) # B, T, 3*C
      q, k, v = qkv.split(self.n_embd, dim=2) # each is of B, T, C

      # k, q and v are the keys queries and values associated with each 
      # token in the sequence. 
      # keys:    an embedding that each token represents. This is a way for a token
      #          to respond to what other tokens are looking for. 
      # queries: am embedding that represents what a token is looking for in rest 
      #          of the tokens. 
      # values:  the final values. 
      # The attention is calculated as: 
      # attn = Softmax(q x k_trans) / sqrt(dim) * v. 
      # Here, we are going to also flip the dimensions 1,2, because we are going to 
      # do attention on each head. 

      k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, head_size
      y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # B, nh, T, head_size
      y = y.transpose(1,2).contiguous().view(B, T, C) # B, T, C
      
      y = self.c_proj(y) # B, T, C

      return y

# Block layout. Each block consists of attention and expert layers. 
# Experts were traditionally feedforward layers, but have been recently 
# replaced with an array of experts, where each expert is a ffd layer. 
# The rationale is that each token might a require different evaluation 
# pathway. So we can create multiple alternative routes for token 
# evaluation. For each token, a gateway determines which experts to turn
# on during evaluation. 

# Before attention and expert layers, we apply layer norm. 

class Block(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.ln_1 = nn.LayerNorm(config.n_embd)
      self.attn = CausalSelfAttention(config)
      self.ln_2 = nn.LayerNorm(config.n_embd)
      self.experts = Experts(config.n_embd, num_experts = 2)
      self.moe = MoE(dim = config.n_embd, num_experts = 2, experts = self.experts)

    def forward(self, x):
        
      x = x + self.attn(self.ln_1(x))

      # Handle MoE output properly
      moe_output = self.moe(self.ln_2(x))

      # MoE returns the output and a load-balancing loss.  
      # The load-balancing loss determines whether certain experts
      # are preferred or weighted more. Ideally, we want all experts to be 
      # more or less equally used. The MoE loss is added to the total loss later. 

      if isinstance(moe_output, tuple):
          moe_out, aux_loss = moe_output
          x = x + moe_out
          return x, aux_loss
      else:
          x = x + moe_output
          return x, None

# The GPT module 

class GPT(nn.Module):

    def __init__(self, config):
      super().__init__()
      self.config = config
    
      # The transformer contains positional and token embeddings, a sequence of block layers,
      # followed by a feedforward lm_head. The lm_head converts the final embedding into the
      # right vocabulary index, which is the prediction for the next token. 

      self.transformer = nn.ModuleDict({
          'wte': nn.Embedding(config.vocab_size, config.n_embd),
          'wpe': nn.Embedding(config.block_size, config.n_embd),
          'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
          'ln_f': nn.LayerNorm(config.n_embd),
      })

      self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
      
      # We are using weight sharing here between lm_head and the initial token embedding layer. 
      # This is because they perform equal but opposite function: translating between embeddings
      # and tokens. The transpose is handled internally by pytorch.

      self.transformer.wte.weight = self.lm_head.weight

      # Weight initialization 

      self.apply(self._init_weights)
      
      # Use the standard GPT-2 encoding, as provided by the tiktoken library. 
      
      self.enc = tiktoken.get_encoding("gpt2")

    # Weight initializations for the different layers. Weights will be 
    # initialized from a normal distribution with mean 0 and a specified
    # deviation. For most layers, std = 0.02. However, for residual layers,
    # the noise from this into the output signal can compound quickly as we add
    # layers, so we scale them down further based on the total number of layers. 
     
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
      
      # idx is a set of sequence of T tokens for B different batches. 

      B, T = idx.size()

      # Get the token and position embeddings
      pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 0, T

      tok_emb = self.transformer.wte(idx) # B, T, C
      pos_emb = self.transformer.wpe(pos) # B, T, C
      x = tok_emb + pos_emb

      # Collect auxiliary losses from MoE layers
      total_aux_loss = 0.0
      aux_loss_count = 0

      # For each of the blocks, get the output and the load-balancing loss.

      for block in self.transformer.h:
          
          block_output = block(x)
          
          # This is if we use MoE w/ load balancing loss. Always 
          # evaluates to true for this case, but would change if 
          # we switch to simple ffn. 

          if isinstance(block_output, tuple):
              
              x, aux_loss = block_output # B, T, C
              
              if aux_loss is not None:
                  total_aux_loss += aux_loss
                  aux_loss_count += 1
          else:
              
              x = block_output # B, T, C

      x = self.transformer.ln_f(x) # B, T, C

      # .. and then the output from the lm_head.

      logits = self.lm_head(x) # B, T, vocab_size

      if targets == None:
          return logits, None

      # Cross-entropy loss, and loss accumulation.

      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

      if aux_loss_count > 0:
          
          avg_aux_loss = total_aux_loss / aux_loss_count
          # Weight the auxiliary loss (common values are 0.01 to 0.1)
          loss = loss + 0.01 * avg_aux_loss

      return logits, loss
    
    # Generate text from a trained model. The input x needs to be
    # extended by max_length. 

    def generate(self, x, max_length):

        # Tik-token embedding 

        x = self.enc.encode(x)

        # Flatten tensor. 

        x = torch.tensor(x).to(device).view(1,-1) # (B, T)

        for iter in range(max_length):
          
          # Get the logits (next token predictions) and the loss for the input

          logits, loss = self(x) # (B, T, vocab_size)

          # Get the output. The logits are supposed to be of size (B, T, vocab_size); 
          # Within each batch, for the T input tokens, logits contains T next-token 
          # predictions. We only care about the token. 

          out = logits[:, -1, :] # (B, vocab_size)

          # softmax, to normalize the outputs so that they sum to 1. 

          out_softmax = nn.Softmax(dim=1)(out) # (B, vocab_size)

          # For each sequence in the batch, get the top k most likely predictions (the ones with highest probabilities) 

          top_k_probs, top_k_idx = torch.topk(out_softmax, 50) # (B, 50)

          # For each sequence in the batch, run coin tosses to select one from the top-k indicies.

          select = torch.multinomial(top_k_probs, 1) # (B, 1)

          # For each sequence in the batch, get the actual tokens. 

          out = torch.gather(top_k_idx, -1, select) # (B, 50)

          # Append to the input for the next loop. 

          x = torch.concat((x, out), dim=-1) # (B, T + 1)
        
        # Move to CPU from GPU. 

        out_list = x.cpu().detach().tolist()

        # Decode via tiktokenizer. 

        return self.enc.decode(out_list[0])

    # Configure parameters for the optimizer.

    def configure_optimizers(self, weight_decay, learning_rate, device_type):

        # Determine parameters that are named and will undergo backprop updates

        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Layers with >= 2 dimensions usually correspond to linear layers. We 
        # assign a higher regularization to these layers, encouraging them to 
        # remain small. On the other hand, layers < 2 correspond to functions 
        # such as softmax/layernorm. We can don't need any weight decay for them. 

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': non_decay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in non_decay_params)

        # The 'fused' parameter allows AdamW to perform fused CUDA kernel operations during the 
        # weights update, optimizing the process. For example, the fused multiply add operation: 
        # https://docs.nvidia.com/cuda/floating-point/index.html, which optimizes sequential 
        # multiply and add operations.

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer

# Configure dataloader for the Shakespeare Dataset. Ignore if you are using the finewebedu dataset.

class DataLoader:
    
    def __init__(self, B, T, process_rank, num_processes):  
        super().__init__()
        
        # num_processes represents how many total GPU threads are running. 
        # process_rank is the current thread number. These parameters will 
        # be 1 and 0 respectively, if we are doing single thread (or without DDP).
        
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # Tiktoken encoder: https://pypi.org/project/tiktoken/0.3.3/

        self.enc = tiktoken.get_encoding("gpt2")
        
        # We need a constant seed for reproducible random inputs and parameters.

        torch.manual_seed(1337)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

        # Read from the Shakespeare dataset file

        with open('input.txt', 'r') as f:
            lines = f.readlines()

        # Set the intial position in the file for the current thread. We are
        # going to stagger the positions for the different threads, so that
        # they read different parts of the file.

        self.current_position = self.B * self.T * self.process_rank

        # Encode into tokens. 

        self.input_data = self.enc.encode(''.join(lines))

    # Read the next batch 

    def get_next_batch(self):

        B, T = self.B, self.T

        # Get the current position in the file for this DDP thread. 

        idx = self.current_position

        # The next (1, T) tokens will be the inputs, and for each of these, 
        # (2, T+1) tokens will be the output targets for prediction.

        inputs = torch.tensor(self.input_data[idx: idx + B*T]).view(B, T)
        targets = torch.tensor(self.input_data[idx + 1: idx + B * T + 1]).view(B, T)

        # Update the current position to skip all the content that will read by 
        # the remaining threads. To visualize this, set num_processes = 2, and B * T = 1. 

        self.current_position += B * T * self.num_processes

        # If we are at the end of the file and do not have enough data to read into the 
        # batch, just loop back to the starting position of the file. 

        if self.current_position > len(self.input_data) - B*T - 1:
            self.current_position = B * T * self.process_rank

        return inputs, targets

# Choose learning rate based on where we are in the training epoch loop. 

def get_lr(step):

    # First few steps are to warm-up, where we rapidly increase the learning rate
    # up to max_lr.

    if step < warm_up_steps:
        return (step + 1) / warm_up_steps * max_lr

    # After max_steps, we stay at min_lr.  

    if step > max_steps:
        return min_lr

    # In the intermediate phase, perform cosine annealing to slowly reduce the learning rate. 
    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

# Set up Data-distributed parallelism (DDP)

# Is this a ddp run?
ddp = int(os.environ.get('RANK', -1)) != -1 
device = 'cuda:0'

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    
    # With a DDP run, a bumch of environment variables are setup that help us determine 
    # which thread we are running in. RANK tells you the rank of the thread. LOCAL_RANK, 
    # I am not exactly clear about. WORLD_SIZE tells you the total number of GPU threads
    # concurrently running 

    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:

    # Set to defaults if this is not a DDP run. 

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

# Set up model. 

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

# We compile the model, instead of have it run dynamically at run time. This 
# will involve a one-time overhead, but will make subsequent forward/backward
# passes fast. 

model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# Set up the parameteres for learning rate (see get_lr function)

max_lr = 3e-4
min_lr = 3e-5
warm_up_steps = 10
max_steps = 50

# Set up optimizer 

optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# Set up data parameters: batch size, mini-batch size (B), and total sequence length. 

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
total_grad_accum = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {total_grad_accum}")

# There will be a different loader instance per GPU thread as discussed above. 

loader = DataLoader(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)

# Iterate through the entire range of epcohs. This is a pretty standard 
# pytorch loop for feedforward/backprop.

for epoch in range(max_steps):

    t0 = time.time()

    # Get learning rate based on epoch. 

    lr = get_lr(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Zero initialize the optimizer gradients. 
    
    optimizer.zero_grad()

    total_loss_accum = 0.0

    # Since the whole batch wont fit into the GPU, we split it 
    # further into micro-batches. We calculate the loss and gradients for 
    # each micro-batch and accumulate. The weight updates will 
    # happen outside the loop just once. 

    for micro_batch in range(total_grad_accum): 
      
      inputs, targets = loader.get_next_batch()
      inputs, targets = inputs.to(device), targets.to(device)

      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):

          logits, loss = model(inputs, targets)

          # Scale the loss, because we want to report the average loss per micro-batch.

          loss = loss / (total_grad_accum)

      total_loss_accum += loss
    
      if ddp:
        model.require_backward_grad_sync = (micro_batch == total_grad_accum - 1)

      # This calculates the gradients per step. The gradients will be 
      # accumulated throughout the for loop.  

      loss.backward()

    # We will collect all losses, now average them across the processes.
    if ddp:
        dist.all_reduce(total_loss_accum, op=dist.ReduceOp.AVG)
    
    # We are also clipping the gradients and monitoring the norm across all parameters. 
    # Clipping prevents gradients from exploding.

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # All threads synchronize here. 

    if device_type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0

    # Update the weights

    optimizer.step()

    # Logging info.

    tokens_processed = loader.B * loader.T * total_grad_accum * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    if master_process:
        print(f"Epoch {epoch}, Train loss: {total_loss_accum:.3f}, Time: {dt:0.3f}, Norm: {norm:.3f}, Tokens: {tokens_processed}, Tokens/sec: {tokens_per_sec:.2f}")

# Remove everything

if ddp:
    destroy_process_group()
