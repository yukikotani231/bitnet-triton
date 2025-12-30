"""
INT8 vs FP16 vs BitNet Benchmark

Compare different quantization approaches on GPU:
- FP32: Baseline
- FP16: Standard mixed precision
- INT8: TensorRT-style quantization
- BitNet: 2-bit ternary weights
"""

import time
import gc
import torch
import torch.nn as nn

from bitnet_triton import BitLinearTriton


def benchmark_layer(layer, x, num_warmup=10, num_runs=100):
    """Benchmark a layer"""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = layer(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = layer(x)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


class INT8Linear(nn.Module):
    """
    Simulated INT8 Linear layer

    Note: True INT8 acceleration requires TensorRT or CUDA INT8 kernels.
    This demonstrates the concept with manual quantization.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # INT8 weights (stored as int8)
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features))

    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'INT8Linear':
        """Convert FP32 Linear to INT8"""
        layer = cls(linear.in_features, linear.out_features)

        # Per-channel quantization
        weight = linear.weight.data
        scale = weight.abs().max(dim=1).values / 127.0
        scale = scale.clamp(min=1e-5)

        weight_int8 = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)),
            -128, 127
        ).to(torch.int8)

        layer.weight_int8 = weight_int8
        layer.weight_scale = scale

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize and compute (simulated - real INT8 uses Tensor Core directly)
        weight_fp = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
        return torch.nn.functional.linear(x, weight_fp)


class INT8LinearTensorCore(nn.Module):
    """
    INT8 Linear using actual INT8 matmul (requires appropriate GPU support)

    Uses torch's _int_mm for INT8 matrix multiplication when available.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.zeros(out_features))
        self.register_buffer('input_scale', torch.tensor(1.0))

    @classmethod
    def from_float(cls, linear: nn.Linear) -> 'INT8LinearTensorCore':
        """Convert FP32 Linear to INT8"""
        layer = cls(linear.in_features, linear.out_features)

        weight = linear.weight.data.float()
        scale = weight.abs().max(dim=1).values / 127.0
        scale = scale.clamp(min=1e-5)

        weight_int8 = torch.clamp(
            torch.round(weight / scale.unsqueeze(1)),
            -128, 127
        ).to(torch.int8)

        layer.weight_int8 = weight_int8
        layer.weight_scale = scale

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input to INT8
        x_float = x.float()
        x_scale = x_float.abs().max() / 127.0
        x_scale = x_scale.clamp(min=1e-5)
        x_int8 = torch.clamp(torch.round(x_float / x_scale), -128, 127).to(torch.int8)

        # INT8 matmul (if available, otherwise fallback)
        try:
            # Use torch._int_mm for actual INT8 Tensor Core matmul
            # Note: requires contiguous tensors and specific shapes
            out_int32 = torch._int_mm(x_int8, self.weight_int8.t())
            # Dequantize
            out = out_int32.float() * (x_scale * self.weight_scale.unsqueeze(0))
        except Exception:
            # Fallback to simulated
            weight_fp = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
            out = torch.nn.functional.linear(x_float, weight_fp)

        return out


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Check INT8 Tensor Core support
    print("Checking INT8 support...")
    try:
        test_a = torch.randint(-128, 127, (32, 64), dtype=torch.int8, device=device)
        test_b = torch.randint(-128, 127, (64, 32), dtype=torch.int8, device=device)
        _ = torch._int_mm(test_a, test_b)
        int8_native = True
        print("✓ Native INT8 matmul (torch._int_mm) available")
    except Exception as e:
        int8_native = False
        print(f"✗ Native INT8 matmul not available: {e}")
    print()

    # =========================================================================
    # Benchmark configurations
    # =========================================================================
    configs = [
        # (batch, in_features, out_features)
        (1, 1024, 1024),
        (1, 4096, 4096),
        (32, 1024, 1024),
        (32, 4096, 4096),
        (128, 1024, 1024),
        (128, 4096, 4096),
    ]

    print("=" * 100)
    print("Quantization Benchmark: FP32 vs FP16 vs INT8 vs BitNet")
    print("=" * 100)
    print()

    header = f"{'Config':<20} {'FP32 (ms)':>12} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'BitNet (ms)':>12} {'Best':>10}"
    print(header)
    print("-" * 90)

    results = []

    for batch, in_feat, out_feat in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Create base layer
        base_layer = nn.Linear(in_feat, out_feat, bias=False).cuda()

        # FP32
        fp32_layer = base_layer.float()
        x_fp32 = torch.randn(batch, in_feat, device=device, dtype=torch.float32)
        fp32_time = benchmark_layer(fp32_layer, x_fp32)

        # FP16
        fp16_layer = base_layer.half()
        x_fp16 = x_fp32.half()
        fp16_time = benchmark_layer(fp16_layer, x_fp16)

        # INT8
        int8_layer = INT8LinearTensorCore.from_float(base_layer).cuda()
        x_int8 = x_fp32  # INT8 layer handles quantization internally
        int8_time = benchmark_layer(int8_layer, x_int8)

        # BitNet
        bitnet_layer = BitLinearTriton(in_feat, out_feat).cuda()
        bitnet_layer.weight.data.copy_(base_layer.weight.data)
        bitnet_layer.pack_weights()
        bitnet_time = benchmark_layer(bitnet_layer, x_fp32)

        # Find best
        times = {
            'FP32': fp32_time,
            'FP16': fp16_time,
            'INT8': int8_time,
            'BitNet': bitnet_time
        }
        best = min(times, key=times.get)

        config_str = f"({batch}, {in_feat}, {out_feat})"
        print(f"{config_str:<20} {fp32_time:>12.4f} {fp16_time:>12.4f} {int8_time:>12.4f} {bitnet_time:>12.4f} {best:>10}")

        results.append({
            'config': (batch, in_feat, out_feat),
            'fp32': fp32_time,
            'fp16': fp16_time,
            'int8': int8_time,
            'bitnet': bitnet_time,
            'best': best
        })

        del base_layer, fp32_layer, fp16_layer, int8_layer, bitnet_layer

    print()

    # =========================================================================
    # Memory comparison
    # =========================================================================
    print("=" * 100)
    print("Memory Usage Comparison (weight only)")
    print("=" * 100)
    print()

    in_feat, out_feat = 4096, 4096
    params = in_feat * out_feat

    print(f"Layer size: {in_feat} x {out_feat} = {params:,} parameters")
    print()

    print(f"{'Precision':<15} {'Bytes/param':>15} {'Total Memory':>15} {'Compression':>12}")
    print("-" * 60)

    precisions = [
        ("FP32", 4, 1.0),
        ("FP16", 2, 2.0),
        ("INT8", 1, 4.0),
        ("INT4", 0.5, 8.0),
        ("BitNet (2-bit)", 0.125, 16.0),
    ]

    for name, bytes_per_param, compression in precisions:
        total_mb = params * bytes_per_param / 1024 / 1024
        print(f"{name:<15} {bytes_per_param:>15} {total_mb:>12.2f} MB {compression:>11.0f}x")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print()

    # Count wins
    wins = {'FP32': 0, 'FP16': 0, 'INT8': 0, 'BitNet': 0}
    for r in results:
        wins[r['best']] += 1

    print("Performance wins:")
    for method, count in sorted(wins.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}/{len(results)}")

    print()
    print("Recommendations:")
    print()
    print("  Speed priority (GPU):")
    print("    1. FP16 - Simple, fast, widely supported")
    print("    2. INT8 + TensorRT - Best speed with 4x compression")
    print()
    print("  Memory priority:")
    print("    1. INT4 (AWQ/GPTQ) - 8x compression, good speed")
    print("    2. BitNet - 16x compression, slower but fits large models")
    print()

    if not int8_native:
        print("  Note: True INT8 Tensor Core acceleration requires:")
        print("    - TensorRT")
        print("    - torch.compile with inductor")
        print("    - CUDA INT8 kernels (cuBLAS LT)")


if __name__ == "__main__":
    main()
