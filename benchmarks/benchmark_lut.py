"""
Benchmark: Dequantize+MatMul vs LUT-based approaches

Compares:
1. Current: unpack -> float -> tl.dot (Tensor Core)
2. LUT: 4-weight groups -> lookup -> accumulate
3. Ternary: separate +1/-1 accumulators -> tl.dot

Hypothesis:
- Current approach wins on large batches (Tensor Core dominance)
- LUT might win on small batches (memory bound scenarios)
"""

import time
import torch
import triton

from bitnet_triton.kernels import bitnet_matmul
from bitnet_triton.kernels_lut import bitnet_matmul_lut, bitnet_matmul_ternary, bitnet_matmul_true_lut
from bitnet_triton.packing import pack_weights


def benchmark_kernel(fn, x, packed, scale, K, num_warmup=20, num_runs=100):
    """Benchmark a kernel function"""
    # Warmup
    for _ in range(num_warmup):
        _ = fn(x, packed, scale, K)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = fn(x, packed, scale, K)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


def verify_correctness(packed, scale, K, device):
    """Verify all kernels produce the same output"""
    print("Verifying correctness...")

    x = torch.randn(16, K, device=device)

    out_current = bitnet_matmul(x, packed, scale, K)
    out_lut = bitnet_matmul_lut(x, packed, scale, K)
    out_ternary = bitnet_matmul_ternary(x, packed, scale, K)
    out_true_lut = bitnet_matmul_true_lut(x, packed, scale, K)

    diff_lut = (out_current - out_lut).abs().max().item()
    diff_ternary = (out_current - out_ternary).abs().max().item()
    diff_true_lut = (out_current - out_true_lut).abs().max().item()

    print(f"  LUT vs Current max diff: {diff_lut:.6f}")
    print(f"  Ternary vs Current max diff: {diff_ternary:.6f}")
    print(f"  TrueLUT vs Current max diff: {diff_true_lut:.6f}")

    max_diff = max(diff_lut, diff_ternary, diff_true_lut)
    if max_diff > 1e-3:
        print("  WARNING: Significant difference detected!")
    else:
        print("  OK: All kernels produce consistent results")

    return max_diff < 1e-3


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Test configurations: (M, K, N)
    configs = [
        # Small (typical for inference)
        (1, 256, 256),
        (1, 512, 512),
        (1, 1024, 1024),
        (1, 2048, 2048),

        # Small batch
        (8, 256, 256),
        (8, 512, 512),
        (8, 1024, 1024),

        # Medium batch
        (32, 256, 256),
        (32, 512, 512),
        (32, 1024, 1024),

        # Large batch
        (128, 256, 256),
        (128, 512, 512),
        (128, 1024, 1024),

        # Large dimensions
        (1, 4096, 4096),
        (8, 4096, 4096),
        (32, 4096, 4096),
    ]

    print("=" * 100)
    print("Kernel Comparison: Current (tl.dot) vs LUT vs Ternary vs TrueLUT (no multiply)")
    print("=" * 100)
    print()

    # Header
    print(f"{'Config':<18} {'Current':>10} {'LUT':>10} {'Ternary':>10} {'TrueLUT':>10} {'Best':>10}")
    print("-" * 78)

    for M, K, N in configs:
        # Create weight and pack
        weight = torch.randn(N, K, device=device)
        packed, scale = pack_weights(weight)
        packed = packed.cuda()
        scale = scale.cuda()

        # Input
        x = torch.randn(M, K, device=device)

        # Verify correctness (only for first config)
        if configs.index((M, K, N)) == 0:
            verify_correctness(packed, scale, K, device)
            print()
            print(f"{'Config':<18} {'Current':>10} {'LUT':>10} {'Ternary':>10} {'TrueLUT':>10} {'Best':>10}")
            print("-" * 78)

        # Benchmark
        try:
            time_current = benchmark_kernel(bitnet_matmul, x, packed, scale, K)
        except Exception as e:
            print(f"Current kernel failed: {e}")
            time_current = float('inf')

        try:
            time_lut = benchmark_kernel(bitnet_matmul_lut, x, packed, scale, K)
        except Exception as e:
            print(f"LUT kernel failed: {e}")
            time_lut = float('inf')

        try:
            time_ternary = benchmark_kernel(bitnet_matmul_ternary, x, packed, scale, K)
        except Exception as e:
            print(f"Ternary kernel failed: {e}")
            time_ternary = float('inf')

        try:
            time_true_lut = benchmark_kernel(bitnet_matmul_true_lut, x, packed, scale, K)
        except Exception as e:
            print(f"TrueLUT kernel failed: {e}")
            time_true_lut = float('inf')

        # Determine best
        times = {'Current': time_current, 'LUT': time_lut, 'Ternary': time_ternary, 'TrueLUT': time_true_lut}
        best = min(times, key=times.get)

        config_str = f"({M}, {K}, {N})"
        print(f"{config_str:<18} {time_current:>10.3f} {time_lut:>10.3f} {time_ternary:>10.3f} {time_true_lut:>10.3f} {best:>10}")

    print()
    print("=" * 90)
    print("Analysis")
    print("=" * 90)
    print()
    print("Key observations:")
    print("1. 'Current' uses tl.dot (Tensor Core) - optimized for matrix multiply")
    print("2. 'LUT' processes 4 weights at a time with scalar ops")
    print("3. 'Ternary' uses two tl.dot calls with masked weights")
    print()
    print("Expected results:")
    print("- Current should win on large matrices (Tensor Core dominance)")
    print("- LUT/Ternary might win on M=1 (batch=1) scenarios")
    print()
    print("Why LUT may not help on GPU:")
    print("- GPU Tensor Cores are extremely efficient for matmul")
    print("- Shared memory LUT access has latency")
    print("- GPU excels at parallel FMA operations")
    print()
    print("Why LUT helps on CPU (BitNet.cpp):")
    print("- CPU has no Tensor Core equivalent")
    print("- CPU SIMD shuffle (vpshufb) is very efficient for LUT")
    print("- Memory hierarchy is different (L1 cache vs shared memory)")


if __name__ == "__main__":
    main()
