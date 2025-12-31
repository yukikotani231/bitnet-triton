"""
TensorRT INT8 Benchmark

True INT8 acceleration using TensorRT.
Compares FP16 vs INT8 with proper Tensor Core utilization.
"""

import gc
import time

import torch
import torch.nn as nn

try:
    import torch_tensorrt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    print("torch_tensorrt not installed. Install with: pip install torch-tensorrt")


def benchmark_model(model, x, num_warmup=10, num_runs=100):
    """Benchmark a model"""
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    if HAS_TENSORRT:
        print("TensorRT: Available")
    else:
        print("TensorRT: Not available")
        print()
        print("To install TensorRT:")
        print("  pip install torch-tensorrt")
        print()
        print("Running torch.compile benchmark instead...")

    print()

    # =========================================================================
    # Benchmark with torch.compile (available without TensorRT)
    # =========================================================================
    print("=" * 80)
    print("Benchmark: torch.compile optimization")
    print("=" * 80)
    print()

    configs = [
        (32, 1024),
        (32, 2048),
        (32, 4096),
        (128, 1024),
        (128, 2048),
        (128, 4096),
    ]

    print(f"{'Config (B, H)':<20} {'Eager (ms)':>12} {'Compiled (ms)':>14} {'Speedup':>10}")
    print("-" * 60)

    for batch, hidden in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Input
        x = torch.randn(batch, hidden, device=device, dtype=torch.float16)

        # Eager mode (FP16)
        model_eager = SimpleMLP(hidden).cuda().half()
        eager_time = benchmark_model(model_eager, x)

        # Compiled mode
        model_compiled = torch.compile(SimpleMLP(hidden).cuda().half(), mode="reduce-overhead")
        # Warmup for compilation
        for _ in range(3):
            _ = model_compiled(x)
        torch.cuda.synchronize()
        compiled_time = benchmark_model(model_compiled, x)

        speedup = eager_time / compiled_time

        config_str = f"({batch}, {hidden})"
        print(f"{config_str:<20} {eager_time:>12.4f} {compiled_time:>14.4f} {speedup:>10.2f}x")

        del model_eager, model_compiled

    print()

    # =========================================================================
    # TensorRT benchmark (if available)
    # =========================================================================
    if HAS_TENSORRT:
        print("=" * 80)
        print("Benchmark: TensorRT FP16 vs INT8")
        print("=" * 80)
        print()

        print(f"{'Config (B, H)':<20} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'Speedup':>10}")
        print("-" * 60)

        for batch, hidden in configs:
            gc.collect()
            torch.cuda.empty_cache()

            x = torch.randn(batch, hidden, device=device, dtype=torch.float16)

            # Base model
            model = SimpleMLP(hidden).cuda().half().eval()

            # TensorRT FP16
            try:
                model_fp16 = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(shape=(batch, hidden), dtype=torch.float16)],
                    enabled_precisions={torch.float16},
                    truncate_long_and_double=True,
                )
                fp16_time = benchmark_model(model_fp16, x)
            except Exception as e:
                print(f"FP16 compile failed: {e}")
                fp16_time = float("inf")

            # TensorRT INT8
            try:
                model_int8 = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(shape=(batch, hidden), dtype=torch.float16)],
                    enabled_precisions={torch.float16, torch.int8},
                    truncate_long_and_double=True,
                )
                int8_time = benchmark_model(model_int8, x)
            except Exception as e:
                print(f"INT8 compile failed: {e}")
                int8_time = float("inf")

            speedup = fp16_time / int8_time if int8_time != float("inf") else 0

            config_str = f"({batch}, {hidden})"
            print(f"{config_str:<20} {fp16_time:>12.4f} {int8_time:>12.4f} {speedup:>10.2f}x")

            del model

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("GPU optimization hierarchy:")
    print()
    print("  1. torch.compile (easiest)")
    print("     - Just add: model = torch.compile(model)")
    print("     - Speedup: 1.2-2x")
    print()
    print("  2. TensorRT FP16 (good balance)")
    print("     - Requires torch-tensorrt")
    print("     - Speedup: 2-4x over eager")
    print()
    print("  3. TensorRT INT8 (maximum speed)")
    print("     - Requires calibration for best accuracy")
    print("     - Speedup: 1.5-2x over TensorRT FP16")
    print()
    print("  4. BitNet (maximum memory savings)")
    print("     - 16x weight compression")
    print("     - Speed: ~0.2x of FP16")
    print("     - Use when model doesn't fit in memory")


if __name__ == "__main__":
    main()
