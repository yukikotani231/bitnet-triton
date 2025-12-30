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

## Converting Existing Models

```python
from bitnet_triton import convert_to_bitlinear_triton

# Convert all Linear layers to BitLinearTriton
model = convert_to_bitlinear_triton(model, pack=True)
```

## Benchmarks

Tested on NVIDIA RTX A4000:

| Config | Memory Compression |
|--------|-------------------|
| 4096→4096 | 15.9x (64 MB → 4 MB) |

**Note**: Current speed is slower than nn.Linear due to kernel overhead. Optimization in progress.

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

### Matrix Multiplication

The Triton kernel:
1. Loads packed int32 values
2. Extracts 2-bit weights on-the-fly
3. Converts to ternary {-1, 0, 1}
4. Performs tiled matrix multiplication
5. Applies scale factors

## Project Structure

```
bitnet-triton/
├── bitnet_triton/
│   ├── __init__.py
│   ├── kernels.py    # Triton matmul kernels
│   ├── ops.py        # PyTorch layers
│   └── packing.py    # Weight packing utilities
├── benchmarks/
│   └── benchmark.py
└── examples/
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- CUDA-capable GPU

## TODO

- [ ] Optimize Triton kernels for better speed
- [ ] Add MNIST example
- [ ] Support for different quantization schemes
- [ ] FP16 input support

## License

MIT License

## Acknowledgments

- [Microsoft BitNet](https://github.com/microsoft/BitNet) - Original BitNet implementation
- [Triton](https://github.com/triton-lang/triton) - GPU programming language
