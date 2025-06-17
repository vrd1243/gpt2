# GPT with Mixture of Experts (MoE) Training

This repository contains a PyTorch implementation of GPT with Mixture of Experts (MoE) support, designed for distributed training using Data Distributed Parallel (DDP).

## Features

- **GPT Architecture**: Standard transformer architecture with causal self-attention
- **Mixture of Experts**: Optional MoE layers to replace traditional MLP feedforward networks
- **Distributed Training**: Full support for multi-GPU training using PyTorch DDP
- **FineWeb Dataset**: Optimized for training on the FineWeb educational dataset
- **Mixed Precision**: Automatic mixed precision training with bfloat16
- **Model Compilation**: PyTorch 2.0 compile support for faster training

## Requirements

```bash
pip install torch torchvision torchaudio
pip install tiktoken
pip install transformers
pip install numpy
```

## Project Structure

```
├── train_gpt_moe.py          # Main training script
├── mixture_of_experts.py     # MoE implementation (required)
├── hellaswag.py             # HellaSwag evaluation utilities (required)
├── edu_fineweb10B/          # Dataset directory
│   ├── train_*.bin          # Training data shards
│   └── val_*.bin            # Validation data shards
└── log/                     # Training logs and checkpoints
```

## Dataset Preparation

1. **Download FineWeb Dataset**: You need to obtain the FineWeb educational dataset and tokenize it into binary files.

2. **Expected Format**: The dataset should be tokenized using GPT-2's tiktoken encoding and saved as numpy arrays in `.bin` files.

3. **Directory Structure**: Place tokenized files in `edu_fineweb10B/` with naming convention:
   - Training files: `train_000000.bin`, `train_000001.bin`, etc.
   - Validation files: `val_000000.bin`, `val_000001.bin`, etc.

## Configuration

The model configuration is controlled by the `GPTConfig` dataclass:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024        # Max sequence length
    vocab_size: int = 50257       # Vocabulary size
    n_layer: int = 12            # Number of transformer layers
    n_head: int = 12             # Number of attention heads
    n_embd: int = 768            # Embedding dimension
    use_moe: bool = True         # Enable MoE (set False for standard MLP)
    num_experts: int = 2         # Number of experts in MoE layers
    infer_step: int = 250        # Inference/evaluation frequency
```

## Usage

### Single GPU Training

```bash
python train_gpt_moe.py
```

### Multi-GPU Training with DDP

#### Option 1: Using torchrun (Recommended)

```bash
# For 4 GPUs on a single node
torchrun --nproc_per_node=4 train_gpt_moe.py

# For 8 GPUs on a single node
torchrun --nproc_per_node=8 train_gpt_moe.py

# For multi-node training (2 nodes, 4 GPUs each)
# On first node (master):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=29500 train_gpt_moe.py

# On second node:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=29500 train_gpt_moe.py
```

#### Option 2: Manual Environment Setup

```bash
# Set environment variables manually
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=localhost
export MASTER_PORT=29500

python train_gpt_moe.py
```

### Training Parameters

Key training hyperparameters:

- **Batch Size**: 524,288 tokens total (distributed across GPUs)
- **Micro Batch Size**: 16 sequences per GPU
- **Sequence Length**: 1024 tokens
- **Learning Rate**: 6e-4 (max) with cosine annealing to 6e-5 (min)
- **Warmup Steps**: 715
- **Total Steps**: 19,073 (approximately 1 epoch on 10B tokens)

## Model Architecture Options

### Standard GPT (MLP)
Set `use_moe=False` in the config to use traditional MLP layers:

```python
config = GPTConfig(use_moe=False)
```

### Mixture of Experts (MoE)
Set `use_moe=True` and specify the number of experts:

```python
config = GPTConfig(use_moe=True, num_experts=8)
```

## Monitoring Training

### Console Output
The script provides real-time training metrics:
- Training loss
- Validation loss (every 250 steps)
- Processing speed (tokens/second)
- Gradient norm
- Learning rate

### Log Files
Training logs are saved to `log/log.txt` with format:
```
step_number val validation_loss
```

### Checkpoints
Model checkpoints are automatically saved every 5000 steps to `log/model_XXXXX.pt`

### Text Generation
The model generates sample text every 250 steps using the prompt: "Hello, I'm a language model,"

## Memory Requirements

Approximate GPU memory usage:
- **12-layer model**: ~2-3 GB per GPU (with mixed precision)
- **24-layer model**: ~6-8 GB per GPU
- **48-layer model**: ~12-16 GB per GPU

Memory scales with:
- Model size (layers, embedding dimension)
- Batch size per GPU
- Sequence length
- Number of experts (for MoE)

## Performance Optimization

### Recommended Settings
1. **Enable Compilation**: Set `use_compile=True` for faster training (disable during debugging)
2. **Mixed Precision**: Enabled by default with bfloat16
3. **Gradient Accumulation**: Automatically calculated based on total batch size
4. **CUDA Optimizations**: Fused AdamW optimizer when available

### Multi-GPU Scaling
Expected speedup with DDP:
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.5x speedup  
- 8 GPUs: ~6-7x speedup

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce micro batch size (`B`)
   - Reduce sequence length (`T`)
   - Reduce model size parameters

2. **DDP Initialization Errors**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL backend availability
   - Verify network connectivity between nodes

3. **Dataset Loading Errors**
   - Ensure `edu_fineweb10B/` directory exists
   - Verify `.bin` files are present and readable
   - Check file permissions

4. **Slow Training**
   - Enable model compilation (`use_compile=True`)
   - Verify mixed precision is working
   - Check GPU utilization with `nvidia-smi`

### Debug Mode
For debugging, set:
```python
use_compile = False  # Disable compilation
B = 4               # Smaller batch size
total_batch_size = 32768  # Smaller total batch
```

## Results and Checkpoints

### Loading Trained Models
```python
# Load a checkpoint
checkpoint = torch.load('log/model_19073.pt')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
```

### Text Generation
```python
# Generate text from trained model
model.eval()
generated_text = model.generate("Hello, I'm a language model,", max_length=50)
print(generated_text)
```

## Dependencies

- **mixture_of_experts.py**: Contains the MoE implementation
- **hellaswag.py**: Contains evaluation utilities (imported but not used in main loop)
- **FineWeb Dataset**: Educational web text dataset for training

## Citation

If you use this code, please consider citing the relevant papers:
- GPT architecture papers
- Mixture of Experts papers
- FineWeb dataset paper

## License

[Add your license information here]
