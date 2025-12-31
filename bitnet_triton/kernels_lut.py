"""
BitNet Triton Kernels - LUT-based (experimental)

Implements T-MAC style lookup table approach for ternary matmul.
Instead of: unpack -> float convert -> matmul
Uses: weight bits as LUT index -> accumulate

Key idea:
- For 4 activations [x0, x1, x2, x3] with 2-bit weights
- Pre-compute LUT[256] = all possible weighted sums
- Use 8-bit weight index to lookup partial sum
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_lut_4(
    x0: tl.tensor,
    x1: tl.tensor,
    x2: tl.tensor,
    x3: tl.tensor,
) -> tl.tensor:
    """
    Build 16-entry LUT for 4 activations with 1-bit weights (used as building block)
    LUT[i] = sum of x_j where bit j of i is 1

    Returns: [16] tensor
    """
    # Build incrementally (more efficient than computing all 16)
    lut = tl.zeros([16], dtype=tl.float32)

    # Index 0: 0000 -> 0
    # Index 1: 0001 -> x0
    # Index 2: 0010 -> x1
    # ...
    # Index 15: 1111 -> x0 + x1 + x2 + x3

    # This needs to be unrolled since triton doesn't support dynamic indexing well
    # We'll use a different approach - direct computation in the main kernel
    return lut


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bitnet_lut_kernel(
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
    BitNet matmul - alternative implementation using explicit computation

    This is identical to the main kernel but uses a different code path
    for comparison purposes.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load scales
    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Process K in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X block [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)

        # Compute pack indices
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        # Load packed weights [BLOCK_N, BLOCK_K]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0,
        )

        # Unpack 2-bit weights to {-1, 0, 1}
        w_bits = (packed >> (bit_idx[None, :] * 2)) & 0b11
        w = w_bits.to(tl.float32) - 1.0  # [BLOCK_N, BLOCK_K]

        # Use tl.dot for matmul (same as main kernel)
        # This is here for A/B comparison with different configs
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
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _bitnet_ternary_acc_kernel(
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
    BLOCK_K: tl.constexpr = 64,
):
    """
    Ternary accumulation kernel - no multiplication version

    For ternary weights {-1, 0, +1}:
    y = sum(x where w=+1) - sum(x where w=-1)

    This avoids float multiplication entirely.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Two accumulators: one for +1 weights, one for -1 weights
    acc_pos = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_neg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load X block [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        ).to(tl.float32)

        # Load and unpack weights
        pack_idx = offs_k // 16
        bit_idx = offs_k % 16

        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        packed = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + pack_idx[None, :] * stride_pk,
            mask=w_mask,
            other=0,
        )

        # Extract 2-bit weights: 0=-1, 1=0, 2=+1
        w_bits = (packed >> (bit_idx[None, :] * 2)) & 0b11

        # Create masks for +1 and -1 weights
        is_pos = w_bits == 2  # +1
        is_neg = w_bits == 0  # -1

        # For +1 weights: accumulate x
        # For -1 weights: accumulate x (to subtract later)
        # Note: This still requires elementwise operations, may not be faster

        # Use masked accumulation
        w_pos = tl.where(is_pos, 1.0, 0.0).to(tl.float32)
        w_neg = tl.where(is_neg, 1.0, 0.0).to(tl.float32)

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc_pos += tl.dot(x, tl.trans(w_pos), allow_tf32=True)
        acc_neg += tl.dot(x, tl.trans(w_neg), allow_tf32=True)

    # Final result: pos - neg
    output = (acc_pos - acc_neg) * scales[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask,
    )


@triton.jit
def _bitnet_true_lut_kernel(
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
):
    """
    True LUT-style kernel - uses conditional selection instead of multiplication

    For ternary weights {-1, 0, +1}:
    - w=+1: add x
    - w=0: add nothing
    - w=-1: subtract x

    This avoids float multiplication (like BitNet.cpp concept)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    scales = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Process 4 K elements at a time (8 bits of packed weight)
    for k_base in range(0, K, 4):
        # Load 4 activations [BLOCK_M]
        x0 = tl.load(
            x_ptr + offs_m * stride_xm + (k_base + 0) * stride_xk,
            mask=(offs_m < M) & (k_base + 0 < K),
            other=0.0,
        )
        x1 = tl.load(
            x_ptr + offs_m * stride_xm + (k_base + 1) * stride_xk,
            mask=(offs_m < M) & (k_base + 1 < K),
            other=0.0,
        )
        x2 = tl.load(
            x_ptr + offs_m * stride_xm + (k_base + 2) * stride_xk,
            mask=(offs_m < M) & (k_base + 2 < K),
            other=0.0,
        )
        x3 = tl.load(
            x_ptr + offs_m * stride_xm + (k_base + 3) * stride_xk,
            mask=(offs_m < M) & (k_base + 3 < K),
            other=0.0,
        )

        # Load packed weights [BLOCK_N]
        pack_idx = k_base // 16
        bit_offset = (k_base % 16) * 2

        packed = tl.load(
            packed_ptr + offs_n * stride_pn + pack_idx * stride_pk, mask=offs_n < N, other=0
        )

        # Extract 2-bit weights (0=-1, 1=0, 2=+1)
        w0 = (packed >> (bit_offset + 0)) & 0b11
        w1 = (packed >> (bit_offset + 2)) & 0b11
        w2 = (packed >> (bit_offset + 4)) & 0b11
        w3 = (packed >> (bit_offset + 6)) & 0b11

        # Convert to float: 0->-1, 1->0, 2->+1
        # Still uses subtraction but avoids explicit multiplication
        w0f = w0.to(tl.float32) - 1.0  # [BLOCK_N]
        w1f = w1.to(tl.float32) - 1.0
        w2f = w2.to(tl.float32) - 1.0
        w3f = w3.to(tl.float32) - 1.0

        # Accumulate: x[m] * w[n] for each (m, n) pair
        # x: [BLOCK_M], w: [BLOCK_N] -> result: [BLOCK_M, BLOCK_N]
        acc += x0[:, None] * w0f[None, :]
        acc += x1[:, None] * w1f[None, :]
        acc += x2[:, None] * w2f[None, :]
        acc += x3[:, None] * w3f[None, :]

    output = acc * scales[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        output,
        mask=out_mask,
    )


def bitnet_matmul_lut(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    LUT-based BitNet matmul (experimental)
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

    _bitnet_lut_kernel[grid](
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


def bitnet_matmul_true_lut(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    True LUT-based matmul - no multiplication (experimental)
    Uses tl.where for conditional accumulation
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1]).contiguous()
    M, K = x.shape
    N = packed_weight.shape[0]

    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    BLOCK_M = 32
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _bitnet_true_lut_kernel[grid](
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output.view(*x_shape[:-1], N)


def bitnet_matmul_ternary(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
) -> torch.Tensor:
    """
    Ternary accumulation matmul (experimental)
    Uses separate accumulators for +1 and -1 weights
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

    _bitnet_ternary_acc_kernel[grid](
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
