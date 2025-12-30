# BitNet Triton

High-performance Triton kernels for BitNet (1.58-bit) neural networks.

## Features

- **2-bit Weight Packing**: Store ternary weights {-1, 0, 1} in 2 bits (~16x compression)
- **Triton Kernels**: GPU-accelerated matrix multiplication with packed weights
- **PyTorch Integration**: Drop-in replacement for `nn.Linear`
- **Training Support**: STE (Straight-Through Estimator) for gradient computation

## Installation

```bash
pip install bitnet-triton
```

Or from source:

```bash
git clone https://github.com/yukikotani231/bitnet-triton.git
cd bitnet-triton
pip install -e .
```

## Quick Start

```python
import torch
from bitnet_triton import BitLinearTriton

# Create layer
layer = BitLinearTriton(in_features=512, out_features=256).cuda()

# Training mode (uses STE quantization)
x = torch.randn(32, 512).cuda()
output = layer(x)
loss = output.sum()
loss.backward()

# Switch to inference mode (uses Triton kernels)
layer.pack_weights()
output = layer(x)  # Now uses optimized 2-bit kernel
```

## Direct Kernel Usage

For maximum performance, use the kernels directly:

```python
from bitnet_triton.packing import pack_weights
from bitnet_triton.kernels import bitnet_matmul

# Pack weights once
weight = torch.randn(4096, 4096).cuda()
packed, scale = pack_weights(weight)
packed, scale = packed.cuda(), scale.cuda()

# Fast inference
x = torch.randn(128, 4096).cuda()
output = bitnet_matmul(x, packed, scale, 4096)
```

## Benchmarks

### RTX 3070

| Config | nn.Linear | BitNet | Speedup | Memory |
|--------|-----------|--------|---------|--------|
| 128×4096→4096 | 0.64 ms | 0.82 ms | 0.78x | 16x smaller |
| 128×4096→11008 | 2.54 ms | 1.86 ms | **1.37x** | 16x smaller |
| 512×4096→4096 | 7.58 ms | 3.21 ms | **2.36x** | 16x smaller |

### RTX A4000

| Config | nn.Linear | BitNet | Speedup | Memory |
|--------|-----------|--------|---------|--------|
| 128×4096→4096 | 0.68 ms | 0.79 ms | 0.86x | 16x smaller |
| 128×4096→11008 | 2.62 ms | 1.77 ms | **1.48x** | 16x smaller |
| 256×4096→11008 | 4.11 ms | 3.30 ms | **1.25x** | 16x smaller |

**Key Results:**
- **16x memory compression** achieved
- **Up to 2.36x speedup** for large batch workloads
- Best performance with batch size >= 128

## Performance Characteristics

| Batch Size | Performance |
|------------|-------------|
| 1-32 | Slower than cuBLAS (kernel overhead) |
| 64-128 | Comparable to cuBLAS |
| 128+ | **Faster than cuBLAS** |

**Best use cases:**
- High-resolution image generation (DiT with large sequence length)
- Batch inference services
- Training with large batch sizes
- Memory-constrained environments

## How It Works

### Weight Packing

```
FP32 weights: [-0.3, 0.1, 0.8, -0.5, ...]
                ↓ Quantize
Ternary:      [-1, 0, 1, -1, ...]
                ↓ Map to {0, 1, 2}
Mapped:       [0, 1, 2, 0, ...]
                ↓ Pack 16 values into int32
Packed:       [0b...00_10_01_00, ...]
```

### Optimized Kernel

- Tiled matrix multiplication with `tl.dot`
- Aligned 2-bit unpacking (BLOCK_K = multiple of 16)
- Autotune for optimal block sizes
- TF32 acceleration support

## Project Structure

```
bitnet-triton/
├── bitnet_triton/
│   ├── __init__.py
│   ├── kernels.py      # Optimized Triton kernels
│   ├── ops.py          # PyTorch layers
│   └── packing.py      # Weight packing utilities
└── benchmarks/
    ├── benchmark.py    # Layer benchmark
    └── benchmark_v2.py # Kernel benchmark
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU

## Experimental Findings

Comprehensive benchmarks comparing BitNet with other GPU optimization techniques revealed:

### Speed Comparison (MNIST ViT, batch=512)

| Method | Throughput | Relative Speed |
|--------|------------|----------------|
| FP16 + Flash Attention + torch.compile | 13,448 samples/sec | **1.00x** (fastest) |
| FP16 + Flash Attention | 10,462 samples/sec | 0.78x |
| FP32 (naive) | 3,353 samples/sec | 0.25x |
| BitNet (2-bit) | 1,219 samples/sec | 0.09x |

### Key Conclusions

1. **For GPU Speed**: Use FP16 + Flash Attention + `torch.compile`
   - Flash Attention (`F.scaled_dot_product_attention`): 2-3x speedup
   - FP16 precision: ~2x speedup over FP32
   - `torch.compile(mode="max-autotune")`: 1.2-1.5x speedup

2. **BitNet's Value = Memory, Not Speed**
   - 16x weight memory compression
   - Can fit 4x larger batch sizes in same GPU memory
   - Useful when model doesn't fit in VRAM

3. **INT8 Quantization**
   - Native PyTorch INT8 (`torch._int_mm`) is NOT faster than FP16
   - Requires TensorRT for actual speedup

4. **LUT-based computation** (like BitNet.cpp)
   - Effective on CPU with SIMD
   - NOT effective on GPU (Tensor Cores are faster)

### When to Use BitNet

| Use Case | Recommended |
|----------|-------------|
| Maximum inference speed | FP16 + Flash Attention + compile |
| Limited GPU memory | **BitNet** |
| Batch inference at scale | FP16 or TensorRT INT8 |
| Edge deployment (CPU) | BitNet with LUT |

See `examples/` for detailed benchmarks.

## Known Limitations

- Small batch sizes (M < 64) have kernel launch overhead
- Best performance with batch size >= 128
- FP16 input not yet optimized
- **Speed is ~0.1-0.2x of FP16** (trade-off for 16x memory compression)

## License

MIT License

## Acknowledgments

- [Microsoft BitNet](https://github.com/microsoft/BitNet) - Original BitNet implementation
- [Triton](https://github.com/triton-lang/triton) - GPU programming language
