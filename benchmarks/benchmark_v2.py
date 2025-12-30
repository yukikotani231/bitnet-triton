"""
Benchmark v2 - Compare kernel versions
"""

import time
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bitnet_triton.packing import pack_weights
from bitnet_triton.kernels import bitnet_matmul
from bitnet_triton.kernels_v2 import bitnet_matmul_v3


def benchmark_fn(fn, *args, num_warmup=10, num_runs=100):
    """Benchmark a function"""
    # Warmup
    for _ in range(num_warmup):
        _ = fn(*args)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = fn(*args)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


def main():
    print("=" * 80)
    print("BitNet Kernel Benchmark - v1 vs v3 (optimized)")
    print("=" * 80)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # (batch_size, in_features, out_features)
        (1, 784, 512),
        (32, 784, 512),
        (128, 784, 512),
        (1, 4096, 4096),
        (32, 4096, 4096),
        (128, 4096, 4096),
        (1, 4096, 11008),
        (32, 4096, 11008),
        (128, 4096, 11008),
    ]

    print(f"{'Config':<22} {'Linear':>10} {'BitNet v1':>10} {'BitNet v3':>10} {'Speedup':>10}")
    print("-" * 72)

    results = []

    for batch_size, in_features, out_features in configs:
        config_str = f"{batch_size}x{in_features}→{out_features}"

        # Create weight and pack
        weight = torch.randn(out_features, in_features, device=device)
        packed, scale = pack_weights(weight)
        packed = packed.to(device)
        scale = scale.to(device)

        # Create input
        x = torch.randn(batch_size, in_features, device=device)

        # nn.Linear baseline
        linear = nn.Linear(in_features, out_features, bias=False).to(device)
        time_linear = benchmark_fn(lambda: linear(x))

        # BitNet v1 (original)
        time_v1 = benchmark_fn(bitnet_matmul, x, packed, scale, in_features)

        # BitNet v3 (optimized)
        time_v3 = benchmark_fn(bitnet_matmul_v3, x, packed, scale, in_features)

        speedup_v1 = time_linear / time_v1
        speedup_v3 = time_linear / time_v3
        improvement = time_v1 / time_v3

        print(f"{config_str:<22} {time_linear:>10.3f} {time_v1:>10.3f} {time_v3:>10.3f} {speedup_v3:>10.2f}x")

        results.append({
            'config': config_str,
            'linear': time_linear,
            'v1': time_v1,
            'v3': time_v3,
            'speedup': speedup_v3,
            'improvement': improvement,
        })

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)

    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    best_speedup = max(r['speedup'] for r in results)
    best_config = [r for r in results if r['speedup'] == best_speedup][0]['config']

    print(f"Average v1 → v3 improvement: {avg_improvement:.2f}x faster")
    print(f"Best speedup vs Linear: {best_speedup:.2f}x ({best_config})")

    # Check if any config beats Linear
    wins = [r for r in results if r['speedup'] >= 1.0]
    if wins:
        print(f"Configs beating Linear: {len(wins)}/{len(results)}")
        for w in wins:
            print(f"  - {w['config']}: {w['speedup']:.2f}x")
    else:
        print("No configs beat Linear yet (more optimization needed)")

    print()
    print("=" * 72)
    print("Correctness Check")
    print("=" * 72)

    in_features, out_features = 512, 256
    batch_size = 32

    weight = torch.randn(out_features, in_features, device=device)
    packed, scale = pack_weights(weight)
    packed = packed.to(device)
    scale = scale.to(device)

    x = torch.randn(batch_size, in_features, device=device)

    out_v1 = bitnet_matmul(x, packed, scale, in_features)
    out_v3 = bitnet_matmul_v3(x, packed, scale, in_features)

    diff = (out_v1 - out_v3).abs()
    print(f"v1 vs v3 max diff: {diff.max().item():.6f}")
    print(f"v1 vs v3 mean diff: {diff.mean().item():.6f}")

    if diff.max().item() < 0.01:
        print("✓ Correctness verified")
    else:
        print("✗ Outputs differ significantly!")


if __name__ == "__main__":
    main()
