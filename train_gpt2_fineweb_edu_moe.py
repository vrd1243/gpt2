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
from typing import Tuple
from mixture_of_experts import MoE

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    use_moe: bool = True # Should we use MoE or a simple MLE for the FFN layer? 
    num_experts: int = 2 # number of experts for smoe
    infer_step: int = 250 # Do training loop inference after these steps.

# A traditional MLP. We will use this as a baseline for MoE. 
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

# a 3 layered MLP as an expert
class Experts(nn.Module):
    def __init__(self, dim, num_experts = 16):
        super().__init__()

         # Proper initialization
        self.w1 = nn.Parameter(torch.randn(num_experts, dim, dim * 4) * (0.02 / (dim ** 0.5)))
        self.w2 = nn.Parameter(torch.randn(num_experts, dim * 4, dim * 4) * (0.02 / ((dim * 4) ** 0.5)))
        self.w3 = nn.Parameter(torch.randn(num_experts, dim * 4, dim) * (0.02 / ((dim * 4) ** 0.5)))
        self.gelu = nn.GELU()  # GELU is more common in transformers

    def forward(self, x):
        
        # Use the einstein sum functions to implement a linear layer. Instead, 
        # we could implement w1, w2 and w3 as linear layers, and then call those 
        # on the input x. 

        hidden1 = self.gelu(torch.einsum('end,edh->enh', x, self.w1))
        hidden2 = self.gelu(torch.einsum('end,edh->enh', hidden1, self.w2))
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

      if config.use_moe: 
          # Use MoE
          self.experts = Experts(config.n_embd, num_experts = config.num_experts)
          self.ffn = MoE(dim = config.n_embd, num_experts = config.num_experts, experts = self.experts)
      
      else:
          # Use MLP
          self.ffn = MLP(config)

    def forward(self, x):
        
      x = x + self.attn(self.ln_1(x))

      # Handle MoE output properly
      moe_output = self.ffn(self.ln_2(x))

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
          loss = loss + 0.000 * avg_aux_loss

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

# Helper function that loads data for finewebedu files. 
def load_tokens(filename):

    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Data load code. 
class DataLoaderLite:

        def __init__(self, B, T, process_rank, num_processes, split):  

            super().__init__()
            # Batch size and sequence length
            self.B = B
            self.T = T

            # DDP parameters. Which process we are, and how many processes in total. 
            self.process_rank = process_rank
            self.num_processes = num_processes

            # Split. Train or val. 
            self.split = split
            assert split in {'train', 'val'}

            self.enc = tiktoken.get_encoding("gpt2")

            torch.manual_seed(1337)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(1337)

            # get the shard filenames. You need to run the fineweb.py script to get these files. 
            # The source directory should be edu_fineweb10B. 
            data_root = "edu_fineweb10B"

            # A shard is a single file containing a subset of the data. 
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if master_process:
                print(f"found {len(shards)} shards for split {split}")
            self.reset()

        # Reset the data loader. 
        def reset(self):
            # state, init at shard zero
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        # Get the next batch. 
        def get_next_batch(self):
            
            # Batch size and sequence length.
            B, T = self.B, self.T

            # Get the next batch. 
            idx = self.current_position
            inputs = torch.tensor(self.tokens[idx: idx + B*T]).view(B, T)
            targets = torch.tensor(self.tokens[idx + 1: idx + B * T + 1]).view(B, T)

            # Update the current position. Since its a multi-process setup, we need to 
            # skip over the number of tokens that are processed by the other processes. 
            self.current_position += B * T * self.num_processes

            # If the end of file, load the next shard and reset to start of file. 
            if self.current_position > len(self.tokens) - B*T - 1:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank

            # Return the batch. 
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
    # coeff will go down from 1 to 0. 

    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # lr will go from max_lr to min_lr. 
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

torch.set_float32_matmul_precision('high')

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix

# Set up model. 

config = GPTConfig(vocab_size=50304)
model = GPT(config)
model.to(device)

# We compile the model, instead of have it run dynamically at run time. This 
# will involve a one-time overhead, but will make subsequent forward/backward
# passes fast. 

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

# Setting the optimizer. 
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# total_batch_size represents the batch of the gradient accumulation and descent. 
# However, such a large batch size is not possible to fit in memory. We break it 
# down into smaller chunks B, which are called micro-batches. 

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
total_grad_accum = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {total_grad_accum}")

# Loaders for training and validation data.

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
    if epoch % config.infer_step == 0 or last_step:
        
        # We put our model in eval mode, which disables dropout and other regularization. 

        model.eval()
        val_loader.reset()
        
        # We use torch.no_grad() to disable gradient computation, which saves memory and 
        # speeds up computation. The rest is pretty standard: load batches, evaluate, 
        # accumulate loss, print

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
        
        # accumulate the loss across all processes

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
    
    if ((epoch > 0 and epoch % config.infer_step == 0) or last_step) and (not use_compile):
        
        model.eval()
        num_return_sequences = 4
        max_length = 32
        
        # We use a standard initial prompt to generate from the model. 

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
    
    # Re-emable the training mode. 

    model.train()

    # Similar loop as the validation code, except we do backprop and update the optimizer. 

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

      # Gradients are automatically synchronized across all processes 
      # during the backward pass. Essentially, we want to ensure
      # synchronization of gradients for each parameter are the same across 
      # all processes. We do this only for the last microbatch, when the 
      # below code is evaluated to True.

      if ddp:
        model.require_backward_grad_sync = (micro_batch == total_grad_accum - 1)

      loss.backward()

    # Accumulate the loss across all processes. 

    if ddp:
        dist.all_reduce(total_loss_accum, op=dist.ReduceOp.AVG)

    # We are also clipping the gradients and monitoring the norm across all parameters. 
    # Clipping prevents gradients from exploding.
    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Threads synchronize at this point. 
    
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
