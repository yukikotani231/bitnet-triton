"""
MNIST Speed Comparison - Real task benchmark

Comparing:
1. Naive (FP32, no optimization)
2. Optimized (FP16 + Flash Attention + torch.compile)
3. BitNet (for reference)
"""

import time
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bitnet_triton import BitLinearTriton


# =============================================================================
# Models
# =============================================================================

class NaiveViT(nn.Module):
    """Naive Vision Transformer - FP32, no optimization"""

    def __init__(self, img_size=28, patch_size=4, dim=256, depth=6, heads=8, num_classes=10):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            NaiveTransformerBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        # Patchify
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, p * p)

        x = self.patch_embed(x) + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)


class NaiveTransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NaiveAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class NaiveAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FastViT(nn.Module):
    """Fast Vision Transformer - FP16 + Flash Attention"""

    def __init__(self, img_size=28, patch_size=4, dim=256, depth=6, heads=8, num_classes=10):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            FastTransformerBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, p * p)

        x = self.patch_embed(x) + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)


class FastTransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FlashAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class BitNetViT(nn.Module):
    """BitNet Vision Transformer - 2-bit weights"""

    def __init__(self, img_size=28, patch_size=4, dim=256, depth=6, heads=8, num_classes=10):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)  # Keep FP32
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.blocks = nn.ModuleList([
            BitNetTransformerBlock(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)  # Keep FP32
        self.patch_size = patch_size

    def pack_weights(self):
        for block in self.blocks:
            block.pack_weights()

    def forward(self, x):
        B = x.shape[0]
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, p * p)

        x = self.patch_embed(x) + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        return self.head(x)


class BitNetTransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BitNetAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = BitLinearTriton(dim, dim * 4)
        self.fc2 = BitLinearTriton(dim * 4, dim)

    def pack_weights(self):
        self.attn.pack_weights()
        self.fc1.pack_weights()
        self.fc2.pack_weights()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        h = self.fc1(self.norm2(x))
        h = F.gelu(h)
        h = self.fc2(h)
        x = x + h
        return x


class BitNetAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = BitLinearTriton(dim, dim * 3)
        self.proj = BitLinearTriton(dim, dim)

    def pack_weights(self):
        self.qkv.pack_weights()
        self.proj.pack_weights()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_inference(model, dataloader, device, num_batches=50):
    """Benchmark inference throughput"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= 5:
                break
            x = x.to(device)
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    total_samples = 0
    start = time.perf_counter()

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            x = x.to(device)
            _ = model(x)
            total_samples += x.shape[0]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return total_samples / elapsed  # samples/sec


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_memory(model):
    """Get model memory in MB"""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        if b is not None:
            total += b.numel() * b.element_size()
    return total / 1024 / 1024


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    batch_sizes = [64, 128, 256, 512]

    print("=" * 80)
    print("MNIST Vision Transformer Speed Comparison")
    print("Model: ViT (dim=256, depth=6, heads=8)")
    print("=" * 80)
    print()

    # Model specs
    print("Model configurations:")
    print("-" * 50)

    naive = NaiveViT().cuda()
    fast = FastViT().cuda().half()
    bitnet = BitNetViT().cuda()
    bitnet.pack_weights()

    print(f"  Naive (FP32):     {count_parameters(naive):,} params, {get_model_memory(naive):.1f} MB")
    print(f"  Fast (FP16):      {count_parameters(fast):,} params, {get_model_memory(fast):.1f} MB")
    print(f"  BitNet (2-bit):   {count_parameters(bitnet):,} params, {get_model_memory(bitnet):.1f} MB")
    print()

    del naive, fast, bitnet
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark each configuration
    print("Throughput (samples/sec):")
    print("-" * 80)
    print(f"{'Batch Size':<12} {'Naive FP32':>15} {'Fast FP16':>15} {'Compiled':>15} {'BitNet':>15}")
    print("-" * 80)

    for batch_size in batch_sizes:
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Naive FP32
        gc.collect()
        torch.cuda.empty_cache()
        naive = NaiveViT().cuda().float()
        naive_throughput = benchmark_inference(naive, dataloader, device)
        del naive

        # Fast FP16
        gc.collect()
        torch.cuda.empty_cache()
        fast = FastViT().cuda().half()
        # Need to convert input to half
        class HalfWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model(x.half())
        fast_wrapped = HalfWrapper(fast)
        fast_throughput = benchmark_inference(fast_wrapped, dataloader, device)
        del fast, fast_wrapped

        # Compiled FP16
        gc.collect()
        torch.cuda.empty_cache()
        compiled = torch.compile(FastViT().cuda().half(), mode="reduce-overhead")
        compiled_wrapped = HalfWrapper(compiled)
        # Extra warmup for compilation
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= 3:
                    break
                _ = compiled_wrapped(x.cuda())
        torch.cuda.synchronize()
        compiled_throughput = benchmark_inference(compiled_wrapped, dataloader, device)
        del compiled, compiled_wrapped

        # BitNet
        gc.collect()
        torch.cuda.empty_cache()
        bitnet = BitNetViT().cuda()
        bitnet.pack_weights()
        bitnet_throughput = benchmark_inference(bitnet, dataloader, device)
        del bitnet

        print(f"{batch_size:<12} {naive_throughput:>15,.0f} {fast_throughput:>15,.0f} {compiled_throughput:>15,.0f} {bitnet_throughput:>15,.0f}")

    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Speed ranking (fastest to slowest):")
    print("  1. FP16 + Flash Attention + torch.compile")
    print("  2. FP16 + Flash Attention")
    print("  3. Naive FP32")
    print("  4. BitNet (memory優先、速度は犠牲)")
    print()
    print("Memory ranking (smallest to largest):")
    print("  1. BitNet (2-bit) - 最小")
    print("  2. FP16 - 半分")
    print("  3. FP32 - フル")
    print()
    print("結論:")
    print("  速度重視 → FP16 + Flash Attention + torch.compile")
    print("  メモリ重視 → BitNet")


if __name__ == "__main__":
    main()
