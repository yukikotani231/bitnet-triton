"""
BitNet Triton Kernels

Efficient matrix multiplication with 2-bit packed ternary weights
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bitnet_matmul_kernel(
    # Input tensor
    x_ptr,
    # Packed weights
    packed_ptr,
    scale_ptr,
    # Output tensor
    output_ptr,
    # Dimensions
    M, N, K,
    K_packed,
    # Strides
    stride_xm, stride_xk,
    stride_pn, stride_pk,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y = X @ W^T where W is stored in 2-bit packed format

    X: [M, K] input activations (FP16/FP32)
    W: [N, K] weights stored as packed 2-bit values
    Y: [M, N] output

    Each int32 contains 16 weights (2 bits each)
    Weight values: {0, 1, 2} mapped from {-1, 0, 1}
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block start positions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Offsets within blocks
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Load scales for this block of output features
    scale_mask = offs_n < N
    scales = tl.load(scale_ptr + offs_n, mask=scale_mask, other=1.0)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input block: X[m, k]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0
        ).to(tl.float32)

        # Load and unpack weights for this k block
        # packed index = k // 16, bit position = k % 16
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        # Load packed weights: packed[n, pack_idx]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed_vals = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0
        )

        # Extract 2-bit values and convert to ternary
        # {0, 1, 2} -> {-1, 0, 1}
        w_mapped = (packed_vals >> (bit_idx[None, :] * 2)) & 0b11
        w_ternary = w_mapped.to(tl.float32) - 1.0  # [N, K]

        # Matrix multiply: acc += X @ W^T
        # X: [M, K], W^T: [K, N] -> need W: [N, K]
        # acc[m, n] += sum_k(x[m, k] * w[n, k])
        acc += tl.dot(x, tl.trans(w_ternary))

    # Apply scales: output = acc * scales[n]
    output = acc * scales[None, :]

    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask
    )


def bitnet_matmul(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    BitNet matrix multiplication: Y = X @ W^T

    Args:
        x: Input tensor [*, K]
        packed_weight: Packed weights [N, K // 16]
        scale: Weight scales [N]
        original_K: Original K dimension (before padding)

    Returns:
        output: [*, N]
    """
    assert x.is_cuda and packed_weight.is_cuda and scale.is_cuda

    # Handle batched input
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    M, K = x.shape
    N, K_packed = packed_weight.shape

    assert K == original_K, f"Input K ({K}) != original_K ({original_K})"

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    bitnet_matmul_kernel[grid](
        x, packed_weight, scale, output,
        M, N, K, K_packed,
        x.stride(0), x.stride(1),
        packed_weight.stride(0), packed_weight.stride(1),
        output.stride(0), output.stride(1),
    )

    # Restore shape
    output = output.view(*x_shape[:-1], N)
    return output


# =============================================================================
# Fused BitNet matmul with activation quantization
# =============================================================================

@triton.jit
def bitnet_matmul_fused_kernel(
    # Input tensor
    x_ptr,
    # Packed weights
    packed_ptr,
    scale_ptr,
    # Output tensor
    output_ptr,
    # Dimensions
    M, N, K,
    K_packed,
    # Strides
    stride_xm, stride_xk,
    stride_pn, stride_pk,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused BitNet matmul with per-row activation quantization

    Y = (X / max(|X|)) @ W^T * scale * max(|X|)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Load scales
    scale_mask = offs_n < N
    w_scales = tl.load(scale_ptr + offs_n, mask=scale_mask, other=1.0)

    # First pass: compute activation scales (max abs per row)
    x_scales = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0
        ).to(tl.float32)
        block_max = tl.max(tl.abs(x_block), axis=1)
        x_scales = tl.maximum(x_scales, block_max)

    x_scales = tl.maximum(x_scales, 1e-5)

    # Second pass: compute matmul with scaled input
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load and scale input
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0
        ).to(tl.float32)
        x = x / x_scales[:, None]

        # Load and unpack weights
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed_vals = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0
        )

        w_mapped = (packed_vals >> (bit_idx[None, :] * 2)) & 0b11
        w_ternary = w_mapped.to(tl.float32) - 1.0

        acc += tl.dot(x, tl.trans(w_ternary))

    # Apply scales: output = acc * w_scales * x_scales
    output = acc * w_scales[None, :] * x_scales[:, None]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask
    )


def bitnet_matmul_fused(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    Fused BitNet matrix multiplication with activation quantization
    """
    assert x.is_cuda and packed_weight.is_cuda and scale.is_cuda

    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    M, K = x.shape
    N, K_packed = packed_weight.shape

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bitnet_matmul_fused_kernel[grid](
        x, packed_weight, scale, output,
        M, N, K, K_packed,
        x.stride(0), x.stride(1),
        packed_weight.stride(0), packed_weight.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    output = output.view(*x_shape[:-1], N)
    return output
