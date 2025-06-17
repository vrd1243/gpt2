# GPT-2 with Mixture of Experts (MoE) Training Manual

## Overview
This manual provides step-by-step instructions for using the GPT-2 implementation with Mixture of Experts support. The process involves two main steps: data preparation using `fineweb.py` and model training using `train_gpt2_fineweb_edu_moe.py`.

## Credits and Acknowledgments
This GPT-2 implementation is based on Andrej Karpathy's excellent educational video tutorial ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=l8pRSuU81PU&t=14379s). The repository follows the core architecture and training methodology demonstrated in the video, with additional enhancements including:

- **Mixture of Experts (MoE)**: Added MoE layers as optional replacements for traditional MLP feedforward networks
- **Distributed Training**: Full PyTorch DDP support for multi-GPU and multi-node training
- **FineWeb Dataset Integration**: Optimized data pipeline for the FineWeb educational dataset
- **Advanced Optimizations**: PyTorch 2.0 compilation, mixed precision training, and fused optimizers

We highly recommend watching Karpathy's video to understand the fundamental concepts before using this implementation.

*This comprehensive instruction manual was created by Claude (Anthropic) to help users navigate the repository and successfully train their own GPT-2 models.*

## Prerequisites

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU(s) with sufficient memory
- At least 50GB of free disk space for the dataset

### Required Dependencies
Install the following packages:

```bash
pip install torch torchvision torchaudio
pip install tiktoken
pip install transformers
pip install numpy
pip install requests
pip install tqdm
```

## Step 1: Data Preparation

### Running fineweb.py
The `fineweb.py` script downloads and preprocesses the FineWeb educational dataset into tokenized binary files suitable for training.

```bash
python fineweb.py
```

### What fineweb.py Does
1. **Downloads FineWeb Dataset**: Automatically downloads the FineWeb educational dataset
2. **Tokenization**: Converts text to tokens using GPT-2's tiktoken encoding
3. **Data Splitting**: Splits data into training and validation sets
4. **Binary Conversion**: Saves tokenized data as numpy arrays in `.bin` files
5. **Directory Structure**: Creates the required `edu_fineweb10B/` directory structure

### Expected Output Structure
After running `fineweb.py`, you should see:
```
edu_fineweb10B/
├── train_000000.bin
├── train_000001.bin
├── train_000002.bin
├── ...
├── val_000000.bin
├── val_000001.bin
└── val_000002.bin
```

### Data Preparation Parameters
The script typically processes approximately 10 billion tokens and creates:
- **Training files**: Multiple shards named `train_XXXXXX.bin`
- **Validation files**: Multiple shards named `val_XXXXXX.bin`
- **Token count**: Each file contains tokenized sequences ready for training

## Step 2: Model Training

### Basic Training Command
Once data preparation is complete, start training:

```bash
python train_gpt2_fineweb_edu_moe.py
```

### Multi-GPU Training (Recommended)
For faster training with multiple GPUs:

```bash
# For 4 GPUs on a single machine
torchrun --nproc_per_node=4 train_gpt2_fineweb_edu_moe.py

# For 8 GPUs on a single machine
torchrun --nproc_per_node=8 train_gpt2_fineweb_edu_moe.py
```

### Multi-Node Training (Advanced)
For training across multiple machines:

```bash
# On the master node (machine 1):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=MASTER_IP --master_port=29500 train_gpt2_fineweb_edu_moe.py

# On the worker node (machine 2):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=MASTER_IP --master_port=29500 train_gpt2_fineweb_edu_moe.py
```

## Configuration Options

### Model Configuration
The model can be configured by modifying the `GPTConfig` class in the training script:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024      # Maximum sequence length
    vocab_size: int = 50257     # Vocabulary size (GPT-2 standard)
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimension
    use_moe: bool = True        # Enable Mixture of Experts
    num_experts: int = 8        # Number of experts in MoE layers
    infer_step: int = 250       # Text generation frequency during training
```

### Training Hyperparameters
Key training settings:
- **Total Batch Size**: 524,288 tokens (distributed across GPUs)
- **Micro Batch Size**: 16 sequences per GPU
- **Learning Rate**: 6e-4 (max) with cosine annealing to 6e-5 (min)
- **Warmup Steps**: 715
- **Total Training Steps**: 19,073 (approximately 1 epoch)
- **Mixed Precision**: Enabled with bfloat16

### MoE vs Standard Training
**With Mixture of Experts (default)**:
```python
config = GPTConfig(use_moe=True, num_experts=8)
```

**Standard GPT-2 (traditional MLP)**:
```python
config = GPTConfig(use_moe=False)
```

## Monitoring Training Progress

### Real-Time Metrics
During training, you'll see:
- **Training Loss**: Current batch loss
- **Validation Loss**: Evaluated every 250 steps
- **Processing Speed**: Tokens processed per second
- **Gradient Norm**: For monitoring training stability
- **Learning Rate**: Current learning rate value

### Log Files
Training progress is saved to:
- **Training Log**: `log/log.txt` - Contains step-by-step metrics
- **Checkpoints**: `log/model_XXXXX.pt` - Model saved every 5000 steps

### Sample Text Generation
Every 250 steps, the model generates sample text using the prompt: "Hello, I'm a language model," to monitor training progress.

## Memory Requirements

### GPU Memory Usage (Approximate)
- **12-layer model**: 2-3 GB per GPU
- **24-layer model**: 6-8 GB per GPU  
- **48-layer model**: 12-16 GB per GPU

Memory usage depends on:
- Model size (layers, embedding dimension)
- Batch size per GPU
- Sequence length
- Number of experts (for MoE models)

### Recommended GPU Configurations
- **Single GPU**: RTX 3080/4080 (12GB+) for 12-layer models
- **Multi-GPU**: RTX 3090/4090 (24GB) or A100 (40GB/80GB) for larger models
- **Distributed**: Multiple nodes with high-speed interconnect (InfiniBand recommended)

## Troubleshooting

### Common Issues and Solutions

**CUDA Out of Memory**
```python
# Reduce these parameters in the training script:
B = 8                    # Reduce micro batch size
T = 512                  # Reduce sequence length
total_batch_size = 262144  # Reduce total batch size
```

**DDP Initialization Errors**
```bash
# Check GPU visibility
nvidia-smi

# Verify NCCL backend
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

**Dataset Loading Errors**
- Ensure `edu_fineweb10B/` directory exists
- Verify `.bin` files are present and readable
- Check file permissions: `chmod 644 edu_fineweb10B/*.bin`

**Slow Training Performance**
```python
# Enable these optimizations in training script:
use_compile = True        # Enable PyTorch 2.0 compilation
# Verify mixed precision is working
# Monitor GPU utilization with nvidia-smi
```

### Debugging Mode
For debugging, use these settings:
```python
use_compile = False       # Disable compilation for easier debugging
B = 4                    # Smaller batch size
total_batch_size = 32768 # Smaller total batch
```

## Model Checkpoints and Resume Training

### Loading Checkpoints
```python
# Load a saved checkpoint
checkpoint = torch.load('log/model_19073.pt')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
```

### Resume Training
The training script automatically resumes from the latest checkpoint if available in the `log/` directory.

### Text Generation from Trained Model
```python
# Load trained model for inference
model.eval()
with torch.no_grad():
    generated_text = model.generate("Hello, I'm a language model,", max_length=100)
    print(generated_text)
```

## Performance Optimization

### Expected Speedup with Multiple GPUs
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.5x speedup
- **8 GPUs**: ~6-7x speedup

### Optimization Tips
1. **Enable Compilation**: Set `use_compile=True` (disable only for debugging)
2. **Mixed Precision**: Enabled by default with bfloat16
3. **Fused Optimizers**: Script uses fused AdamW when available
4. **Gradient Accumulation**: Automatically calculated for optimal batch sizes

## Complete Workflow Summary

1. **Setup Environment**
   ```bash
   pip install torch torchvision torchaudio tiktoken transformers numpy requests tqdm
   ```

2. **Prepare Data**
   ```bash
   python fineweb.py
   ```

3. **Start Training**
   ```bash
   # Single GPU
   python train_gpt2_fineweb_edu_moe.py
   
   # Multi-GPU (recommended)
   torchrun --nproc_per_node=4 train_gpt2_fineweb_edu_moe.py
   ```

4. **Monitor Progress**
   - Watch console output for real-time metrics
   - Check `log/log.txt` for detailed logs
   - Generated text samples appear every 250 steps

5. **Use Trained Model**
   - Load checkpoint from `log/model_XXXXX.pt`
   - Use for inference or fine-tuning

## Additional Notes

- **Training Time**: Full training takes approximately 12-24 hours on 4x RTX 4090 GPUs
- **Dataset Size**: FineWeb educational dataset contains ~10 billion tokens
- **Model Size**: Default 12-layer model has ~117M parameters
- **Checkpoints**: Saved every 5000 steps (approximately every 2-3 hours)

For advanced usage, model architecture modifications, or custom datasets, refer to the source code comments and configuration options in the training script.
