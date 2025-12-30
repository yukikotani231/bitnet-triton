"""
Native INT8 Benchmark using torch._int_mm

Demonstrates actual INT8 Tensor Core performance using PyTorch's native INT8 matmul.
This is the foundation of TensorRT INT8 acceleration.
"""

import time
import gc
import torch
import torch.nn as nn


def benchmark_matmul(fn, num_warmup=20, num_runs=100):
    """Benchmark a matmul function"""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000  # ms


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # =========================================================================
    # Raw matmul benchmark: FP32 vs FP16 vs INT8
    # =========================================================================
    print("=" * 90)
    print("Raw Matmul Benchmark: FP32 vs FP16 vs INT8 (Tensor Core)")
    print("Computing: C = A @ B where A is [M, K], B is [K, N]")
    print("=" * 90)
    print()

    configs = [
        # (M, K, N) - M must be > 16 for INT8 Tensor Core
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (32, 4096, 11008),   # LLaMA MLP up-proj
        (128, 4096, 11008),
    ]

    print(f"{'Config (M,K,N)':<25} {'FP32 (ms)':>12} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'INT8 speedup':>14}")
    print("-" * 80)

    for M, K, N in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # FP32 matmul
        a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
        b_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)
        fp32_time = benchmark_matmul(lambda: torch.mm(a_fp32, b_fp32))

        # FP16 matmul (Tensor Core)
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        fp16_time = benchmark_matmul(lambda: torch.mm(a_fp16, b_fp16))

        # INT8 matmul (Tensor Core)
        # Note: torch._int_mm requires int8 inputs and returns int32
        a_int8 = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
        b_int8 = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
        int8_time = benchmark_matmul(lambda: torch._int_mm(a_int8, b_int8))

        int8_vs_fp16 = fp16_time / int8_time

        config_str = f"({M}, {K}, {N})"
        print(f"{config_str:<25} {fp32_time:>12.4f} {fp16_time:>12.4f} {int8_time:>12.4f} {int8_vs_fp16:>13.2f}x")

    print()

    # =========================================================================
    # Throughput analysis
    # =========================================================================
    print("=" * 90)
    print("Throughput Analysis (TFLOPS)")
    print("=" * 90)
    print()

    M, K, N = 1024, 4096, 4096
    flops = 2 * M * K * N  # matmul FLOPS

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)
    fp32_time = benchmark_matmul(lambda: torch.mm(a_fp32, b_fp32))
    fp32_tflops = flops / (fp32_time / 1000) / 1e12

    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()
    fp16_time = benchmark_matmul(lambda: torch.mm(a_fp16, b_fp16))
    fp16_tflops = flops / (fp16_time / 1000) / 1e12

    a_int8 = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
    b_int8 = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
    int8_time = benchmark_matmul(lambda: torch._int_mm(a_int8, b_int8))
    int8_tops = flops / (int8_time / 1000) / 1e12  # TOPS for INT8

    print(f"Matrix size: ({M}, {K}) @ ({K}, {N})")
    print()
    print(f"  FP32: {fp32_tflops:>6.2f} TFLOPS ({fp32_time:.3f} ms)")
    print(f"  FP16: {fp16_tflops:>6.2f} TFLOPS ({fp16_time:.3f} ms)")
    print(f"  INT8: {int8_tops:>6.2f} TOPS   ({int8_time:.3f} ms)")
    print()

    # RTX A4000 theoretical peaks
    print("RTX A4000 theoretical peaks:")
    print("  FP32: 19.2 TFLOPS")
    print("  FP16: 77 TFLOPS (Tensor Core)")
    print("  INT8: 153 TOPS (Tensor Core)")
    print()

    print(f"Utilization:")
    print(f"  FP32: {fp32_tflops / 19.2 * 100:>5.1f}%")
    print(f"  FP16: {fp16_tflops / 77 * 100:>5.1f}%")
    print(f"  INT8: {int8_tops / 153 * 100:>5.1f}%")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 90)
    print("Summary: INT8 vs FP16 on GPU")
    print("=" * 90)
    print()
    print("Key findings:")
    print()
    print("1. Raw INT8 matmul IS faster than FP16")
    print(f"   - INT8 is ~{fp16_time/int8_time:.1f}x faster for pure matmul")
    print()
    print("2. BUT full INT8 inference has overhead:")
    print("   - Input quantization (FP32/FP16 -> INT8)")
    print("   - Output dequantization (INT32 -> FP32/FP16)")
    print("   - Scale factor management")
    print()
    print("3. When INT8 wins:")
    print("   - Large batch sizes (amortizes overhead)")
    print("   - TensorRT (fuses quant/dequant)")
    print("   - Memory bandwidth limited scenarios")
    print()
    print("4. When FP16 wins:")
    print("   - Small batch / single sample")
    print("   - Mixed precision training")
    print("   - Simple deployment")
    print()
    print("Recommendation:")
    print("  - For maximum speed: TensorRT + INT8")
    print("  - For simplicity: FP16 + torch.compile")
    print("  - For memory: INT4 (AWQ) or BitNet")


if __name__ == "__main__":
    main()
