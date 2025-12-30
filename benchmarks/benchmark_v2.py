"""
BitNet Kernel Benchmark
"""

import time
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bitnet_triton.packing import pack_weights
from bitnet_triton.kernels_v2 import bitnet_matmul_v3


def benchmark_fn(fn, *args, num_warmup=10, num_runs=100):
    """Benchmark a function"""
    for _ in range(num_warmup):
        _ = fn(*args)

    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = fn(*args)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


def main():
    print("=" * 70)
    print("BitNet Kernel Benchmark")
    print("=" * 70)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # (batch_size, in_features, out_features)
        (1, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1, 4096, 11008),
        (32, 4096, 11008),
        (128, 4096, 11008),
        (256, 4096, 11008),
        (512, 4096, 11008),
    ]

    print(f"{'Config':<22} {'Linear (ms)':>12} {'BitNet (ms)':>12} {'Speedup':>10}")
    print("-" * 60)

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

        # BitNet (optimized)
        time_bitnet = benchmark_fn(bitnet_matmul_v3, x, packed, scale, in_features)

        speedup = time_linear / time_bitnet

        print(f"{config_str:<22} {time_linear:>12.3f} {time_bitnet:>12.3f} {speedup:>10.2f}x")

        results.append({
            'config': config_str,
            'linear': time_linear,
            'bitnet': time_bitnet,
            'speedup': speedup,
        })

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    best_speedup = max(r['speedup'] for r in results)
    best_config = [r for r in results if r['speedup'] == best_speedup][0]['config']

    print(f"Best speedup: {best_speedup:.2f}x ({best_config})")

    wins = [r for r in results if r['speedup'] >= 1.0]
    print(f"Configs beating Linear: {len(wins)}/{len(results)}")
    if wins:
        for w in sorted(wins, key=lambda x: -x['speedup']):
            print(f"  - {w['config']}: {w['speedup']:.2f}x")


if __name__ == "__main__":
    main()
