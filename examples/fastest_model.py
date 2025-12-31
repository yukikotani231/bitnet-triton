"""
Fastest GPU Model - Real optimizations that actually work

No bullshit. Just speed.
"""

import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def benchmark(fn, num_warmup=10, num_runs=100):
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        fn()
    torch.cuda.synchronize()

    return (time.perf_counter() - start) / num_runs * 1000


class NaiveAttention(nn.Module):
    """Slow attention - for comparison"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Naive attention - O(N^2) memory
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class FlashAttention(nn.Module):
    """Fast attention using PyTorch's scaled_dot_product_attention"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Flash Attention - O(N) memory, fused kernel
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class FastTransformerBlock(nn.Module):
    """Optimized transformer block"""

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SlowTransformerBlock(nn.Module):
    """Naive transformer block for comparison"""

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NaiveAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Check Flash Attention support
    print("Flash Attention backend:", end=" ")
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        print("Available")
    else:
        print("Not available")
    print()

    # =========================================================================
    # Test 1: Flash Attention vs Naive
    # =========================================================================
    print("=" * 70)
    print("Test 1: Flash Attention vs Naive Attention")
    print("=" * 70)
    print()

    dim, heads = 1024, 16

    print(f"{'Seq Length':<12} {'Naive (ms)':>12} {'Flash (ms)':>12} {'Speedup':>10}")
    print("-" * 50)

    for seq_len in [128, 256, 512, 1024, 2048]:
        gc.collect()
        torch.cuda.empty_cache()

        x = torch.randn(8, seq_len, dim, device=device, dtype=torch.float16)

        # Naive
        naive = NaiveAttention(dim, heads).cuda().half()
        try:
            naive_time = benchmark(lambda: naive(x))
        except RuntimeError:  # OOM
            naive_time = float("inf")

        # Flash
        flash = FlashAttention(dim, heads).cuda().half()
        flash_time = benchmark(lambda: flash(x))

        speedup = naive_time / flash_time if naive_time != float("inf") else float("inf")
        naive_str = f"{naive_time:.4f}" if naive_time != float("inf") else "OOM"

        print(f"{seq_len:<12} {naive_str:>12} {flash_time:>12.4f} {speedup:>10.2f}x")

        del naive, flash

    print()

    # =========================================================================
    # Test 2: torch.compile
    # =========================================================================
    print("=" * 70)
    print("Test 2: torch.compile optimization")
    print("=" * 70)
    print()

    seq_len = 512
    x = torch.randn(16, seq_len, dim, device=device, dtype=torch.float16)

    # Eager
    model_eager = FastTransformerBlock(dim, heads).cuda().half()
    eager_time = benchmark(lambda: model_eager(x))

    # Compiled
    model_compiled = torch.compile(
        FastTransformerBlock(dim, heads).cuda().half(), mode="max-autotune"
    )
    # Warmup compilation
    for _ in range(5):
        with torch.no_grad():
            _ = model_compiled(x)
    torch.cuda.synchronize()

    compiled_time = benchmark(lambda: model_compiled(x))

    print(f"Eager:    {eager_time:.4f} ms")
    print(f"Compiled: {compiled_time:.4f} ms")
    print(f"Speedup:  {eager_time / compiled_time:.2f}x")
    print()

    del model_eager, model_compiled
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 3: FP32 vs FP16 vs BF16
    # =========================================================================
    print("=" * 70)
    print("Test 3: Precision comparison")
    print("=" * 70)
    print()

    x_fp32 = torch.randn(16, 512, dim, device=device, dtype=torch.float32)
    x_fp16 = x_fp32.half()
    x_bf16 = x_fp32.bfloat16()

    model_fp32 = FastTransformerBlock(dim, heads).cuda().float()
    model_fp16 = FastTransformerBlock(dim, heads).cuda().half()
    model_bf16 = FastTransformerBlock(dim, heads).cuda().bfloat16()

    fp32_time = benchmark(lambda: model_fp32(x_fp32))
    fp16_time = benchmark(lambda: model_fp16(x_fp16))
    bf16_time = benchmark(lambda: model_bf16(x_bf16))

    print(f"FP32: {fp32_time:.4f} ms (1.00x)")
    print(f"FP16: {fp16_time:.4f} ms ({fp32_time / fp16_time:.2f}x)")
    print(f"BF16: {bf16_time:.4f} ms ({fp32_time / bf16_time:.2f}x)")
    print()

    del model_fp32, model_fp16, model_bf16
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 4: Full pipeline - Fastest possible
    # =========================================================================
    print("=" * 70)
    print("Test 4: Fastest possible configuration")
    print("=" * 70)
    print()

    configs = [
        (8, 256, "Small batch, short seq"),
        (8, 1024, "Small batch, long seq"),
        (32, 256, "Large batch, short seq"),
        (32, 1024, "Large batch, long seq"),
    ]

    print(f"{'Config':<30} {'Naive FP32':>12} {'Optimized':>12} {'Speedup':>10}")
    print("-" * 70)

    for batch, seq_len, name in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Naive: FP32, naive attention, eager
        x_fp32 = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float32)
        naive_model = SlowTransformerBlock(dim, heads).cuda().float()
        try:
            naive_time = benchmark(lambda: naive_model(x_fp32), num_runs=50)
        except RuntimeError:
            naive_time = float("inf")

        del naive_model
        gc.collect()
        torch.cuda.empty_cache()

        # Optimized: FP16, flash attention, compiled
        x_fp16 = x_fp32.half()
        fast_model = torch.compile(
            FastTransformerBlock(dim, heads).cuda().half(), mode="max-autotune"
        )
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = fast_model(x_fp16)
        torch.cuda.synchronize()

        fast_time = benchmark(lambda: fast_model(x_fp16), num_runs=50)

        speedup = naive_time / fast_time if naive_time != float("inf") else float("inf")
        naive_str = f"{naive_time:.3f}" if naive_time != float("inf") else "OOM"

        print(f"{name:<30} {naive_str:>12} {fast_time:>12.3f} {speedup:>10.1f}x")

        del fast_model

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("FASTEST GPU MODEL - RECIPE")
    print("=" * 70)
    print()
    print("1. Use FP16 or BF16 (not FP32)")
    print("   model = model.half()  # or .bfloat16()")
    print()
    print("2. Use Flash Attention")
    print("   F.scaled_dot_product_attention(q, k, v)")
    print()
    print("3. Use torch.compile")
    print("   model = torch.compile(model, mode='max-autotune')")
    print()
    print("4. Use larger batch sizes when possible")
    print()
    print("That's it. No fancy quantization needed for speed.")
    print("Quantization (INT8, INT4, BitNet) is for MEMORY, not speed.")


if __name__ == "__main__":
    main()
