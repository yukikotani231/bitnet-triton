"""
Weight packing utilities for BitNet

2-bit packing: {-1, 0, 1} -> {0, 1, 2} stored as 2 bits
16 weights packed into a single int32
"""

import torch
import torch.nn.functional as F


def pack_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack FP32 weights into 2-bit format

    Args:
        weight: FP32 weight tensor [out_features, in_features]

    Returns:
        packed: Packed weights [out_features, in_features // 16] (int32)
        scale: Scale factors [out_features]
    """
    # Move to CPU for packing
    device = weight.device
    weight = weight.float().cpu()

    N, K = weight.shape

    # Calculate scale (mean absolute value per row)
    scale = weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)

    # Quantize to {-1, 0, 1}
    w_scaled = weight / scale
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)

    # Map to {0, 1, 2}
    w_mapped = (w_ternary + 1).to(torch.uint8)

    # Pad to multiple of 16
    K_padded = ((K + 15) // 16) * 16
    if K_padded > K:
        w_mapped = F.pad(w_mapped, (0, K_padded - K), value=1)  # pad with 0 -> mapped to 1

    # Reshape for packing: [N, K_padded] -> [N, K_padded // 16, 16]
    w_reshaped = w_mapped.view(N, -1, 16)

    # Pack 16 x 2-bit values into int32
    packed = torch.zeros(N, w_reshaped.shape[1], dtype=torch.int32)
    for i in range(16):
        packed |= w_reshaped[:, :, i].to(torch.int32) << (i * 2)

    # Move back to original device
    return packed.to(device), scale.squeeze(1).to(device)


def unpack_weights(packed: torch.Tensor, scale: torch.Tensor, original_K: int) -> torch.Tensor:
    """
    Unpack 2-bit weights back to FP32

    Args:
        packed: Packed weights [out_features, in_features // 16]
        scale: Scale factors [out_features]
        original_K: Original in_features dimension

    Returns:
        weight: Unpacked FP32 weights [out_features, in_features]
    """
    device = packed.device
    packed = packed.cpu()
    scale = scale.cpu()

    N, K_packed = packed.shape

    # Unpack
    unpacked = torch.zeros(N, K_packed * 16, dtype=torch.float32)
    for i in range(16):
        # Extract 2-bit values
        vals = ((packed >> (i * 2)) & 0b11).to(torch.float32)
        # Map {0, 1, 2} -> {-1, 0, 1}
        unpacked[:, i::16] = vals - 1

    # Trim to original size
    unpacked = unpacked[:, :original_K]

    # Apply scale
    unpacked = unpacked * scale.unsqueeze(1)

    return unpacked.to(device)


# Alias for backwards compatibility
pack_weights_cpu = pack_weights
