"""
BitNet Triton Kernels - Optimized

Key optimizations:
1. Process 16 weights at a time (matching packed int32 structure)
2. Better memory access patterns
3. Optimized autotune configurations
4. Use allow_tf32 for better performance
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bitnet_matmul_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_pn,
    stride_pk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized BitNet matmul: Y = X @ W^T

    BLOCK_K should be a multiple of 16 for efficient unpacking.
    Uses tl.dot for tensor core acceleration.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load weight scales once
    scale_mask = offs_n < N
    scales = tl.load(scale_ptr + offs_n, mask=scale_mask, other=1.0)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X block [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)

        # Compute pack indices for this k block
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        # Load packed weights [BLOCK_N, BLOCK_K]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0,
        )

        # Unpack 2-bit weights
        w_bits = (packed >> (bit_idx[None, :] * 2)) & 0b11
        w = w_bits.to(tl.float32) - 1.0  # [BLOCK_N, BLOCK_K]

        # Matrix multiply: acc += X @ W^T
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)

    # Apply scales and store
    output = acc * scales[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bitnet_matmul_small_batch_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_pn,
    stride_pk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized for small batch sizes (M <= 32)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)

        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0,
        )

        w_bits = (packed >> (bit_idx[None, :] * 2)) & 0b11
        w = w_bits.to(tl.float32) - 1.0

        acc += tl.dot(x, tl.trans(w), allow_tf32=True)

    output = acc * scales[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask,
    )


def bitnet_matmul(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    BitNet matrix multiplication with packed 2-bit weights.

    Args:
        x: Input tensor [..., K]
        packed_weight: Packed 2-bit weights [N, K // 16]
        scale: Weight scales [N]
        original_K: Original input dimension

    Returns:
        output: [..., N]
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1]).contiguous()
    M, K = x.shape
    N = packed_weight.shape[0]

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    # Use small batch kernel for M <= 32
    if M <= 32:
        _bitnet_matmul_small_batch_kernel[grid](
            x,
            packed_weight,
            scale,
            output,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            packed_weight.stride(0),
            packed_weight.stride(1),
            output.stride(0),
            output.stride(1),
        )
    else:
        _bitnet_matmul_kernel[grid](
            x,
            packed_weight,
            scale,
            output,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            packed_weight.stride(0),
            packed_weight.stride(1),
            output.stride(0),
            output.stride(1),
        )

    return output.view(*x_shape[:-1], N)
