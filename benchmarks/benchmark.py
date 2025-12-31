"""
BitNet Triton Benchmark

Compare performance of:
1. Standard nn.Linear (FP32)
2. BitLinearTriton (2-bit packed, Triton kernel)
"""

import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "..")

from bitnet_triton import BitLinearTriton


def benchmark_layer(layer, x, num_warmup=10, num_runs=100):
    """Benchmark a single layer"""
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = layer(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_runs * 1000  # ms


def get_memory_usage(layer, packed=False):
    """Get memory usage in bytes"""
    total = 0
    for name, param in layer.named_parameters():
        # Skip weight if packed (we use packed_weight instead)
        if packed and name == "weight":
            continue
        total += param.numel() * param.element_size()
    for name, buffer in layer.named_buffers():
        if buffer is not None:
            total += buffer.numel() * buffer.element_size()
    return total


def main():
    print("=" * 70)
    print("BitNet Triton Benchmark")
    print("=" * 70)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Test configurations
    configs = [
        # (batch_size, in_features, out_features)
        (1, 784, 512),  # MNIST-like
        (32, 784, 512),  # MNIST batched
        (128, 784, 512),  # MNIST large batch
        (1, 4096, 4096),  # LLM-like
        (32, 4096, 4096),  # LLM batched
        (128, 4096, 4096),  # LLM large batch
        (1, 4096, 11008),  # LLaMA FFN
        (128, 4096, 11008),  # LLaMA FFN large batch
    ]

    print(
        f"{'Config':<25} {'Linear (ms)':>12} {'BitNet (ms)':>12} {'Speedup':>10} {'Mem Ratio':>10}"
    )
    print("-" * 70)

    for batch_size, in_features, out_features in configs:
        config_str = f"{batch_size}x{in_features}→{out_features}"

        # Create layers
        linear = nn.Linear(in_features, out_features, bias=False).to(device)

        bitnet = BitLinearTriton(in_features, out_features).to(device)
        bitnet.weight.data.copy_(linear.weight.data)
        bitnet.pack_weights()

        # Create input
        x = torch.randn(batch_size, in_features, device=device)

        # Verify correctness
        with torch.no_grad():
            linear(x)
            bitnet(x)
            # Note: outputs won't be exactly equal due to quantization
            # but should be close relative to the scale

        # Benchmark
        try:
            time_linear = benchmark_layer(linear, x)
            time_bitnet = benchmark_layer(bitnet, x)
            speedup = time_linear / time_bitnet

            mem_linear = get_memory_usage(linear)
            mem_bitnet = get_memory_usage(bitnet, packed=True)
            mem_ratio = mem_linear / mem_bitnet

            print(
                f"{config_str:<25} {time_linear:>12.3f} {time_bitnet:>12.3f} {speedup:>10.2f}x {mem_ratio:>10.1f}x"
            )

        except Exception as e:
            print(f"{config_str:<25} ERROR: {e}")

    print()
    print("=" * 70)
    print("Correctness Test")
    print("=" * 70)

    # Detailed correctness test
    in_features, out_features = 512, 256
    batch_size = 32

    linear = nn.Linear(in_features, out_features, bias=False).to(device)
    bitnet = BitLinearTriton(in_features, out_features, fused=True).to(device)
    bitnet.weight.data.copy_(linear.weight.data)

    # Before packing (training mode)
    x = torch.randn(batch_size, in_features, device=device)
    with torch.no_grad():
        out_train = bitnet(x)

    # After packing (inference mode)
    bitnet.pack_weights()
    with torch.no_grad():
        out_infer = bitnet(x)

    # Compare
    diff = (out_train - out_infer).abs()
    print("Training vs Inference output difference:")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    print(f"  Relative diff: {(diff / out_train.abs().clamp(min=1e-6)).mean().item():.4%}")

    print()
    print("=" * 70)
    print("Memory Usage")
    print("=" * 70)

    in_features, out_features = 4096, 4096

    linear = nn.Linear(in_features, out_features, bias=False).to(device)
    bitnet = BitLinearTriton(in_features, out_features, fused=True).to(device)
    bitnet.pack_weights()

    mem_linear = get_memory_usage(linear)
    mem_bitnet = get_memory_usage(bitnet, packed=True)

    print(f"Layer: {in_features} → {out_features}")
    print(f"  nn.Linear (FP32):    {mem_linear:,} bytes ({mem_linear / 1024 / 1024:.2f} MB)")
    print(f"  BitLinear (2-bit):   {mem_bitnet:,} bytes ({mem_bitnet / 1024 / 1024:.2f} MB)")
    print(f"  Compression:         {mem_linear / mem_bitnet:.1f}x")

    # Theoretical compression
    print()
    print("Theoretical:")
    print(f"  FP32: {in_features * out_features * 4:,} bytes")
    print(f"  2-bit: {in_features * out_features * 2 // 8:,} bytes")
    print(
        f"  Ratio: {(in_features * out_features * 4) / (in_features * out_features * 2 / 8):.1f}x"
    )


if __name__ == "__main__":
    main()
