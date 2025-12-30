"""
PyTorch operations and layers for BitNet with Triton backend
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .kernels import bitnet_matmul, bitnet_matmul_fused
from .kernels_v2 import bitnet_matmul_v3 as bitnet_matmul_fast
from .packing import pack_weights, unpack_weights


class BitLinearFunction(torch.autograd.Function):
    """
    Autograd function for BitNet linear with Triton kernel
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        scale: torch.Tensor,
        original_K: int,
        fused: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(packed_weight, scale)
        ctx.original_K = original_K

        if fused:
            return bitnet_matmul_fused(x, packed_weight, scale, original_K)
        else:
            return bitnet_matmul(x, packed_weight, scale, original_K)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        packed_weight, scale = ctx.saved_tensors
        original_K = ctx.original_K

        # Unpack weights for backward pass
        weight = unpack_weights(packed_weight, scale, original_K)

        # Compute gradient w.r.t. input
        grad_input = grad_output @ weight

        # No gradient for packed weights (inference only)
        return grad_input, None, None, None, None


def bitnet_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    original_K: int,
    fused: bool = False,  # Changed default: use optimized kernel
) -> torch.Tensor:
    """
    Functional interface for BitNet linear layer

    Args:
        x: Input tensor [..., K]
        packed_weight: Packed 2-bit weights [N, K // 16]
        scale: Weight scales [N]
        original_K: Original input dimension
        fused: Use fused kernel with activation quantization (slower)

    Returns:
        output: [..., N]
    """
    if not fused:
        # Use optimized v3 kernel
        return bitnet_matmul_fast(x, packed_weight, scale, original_K)
    return BitLinearFunction.apply(x, packed_weight, scale, original_K, fused)


class BitLinearTriton(nn.Module):
    """
    BitNet Linear layer with Triton-accelerated inference

    Drop-in replacement for nn.Linear with 2-bit quantized weights.
    Supports both training (STE) and inference (Triton kernels).

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Not supported (always False)
        fused: Use fused kernel with activation quantization

    Example:
        >>> layer = BitLinearTriton(512, 256)
        >>> layer.pack_weights()  # Convert to 2-bit format
        >>> output = layer(input)  # Uses Triton kernel
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        fused: bool = False,  # Use optimized kernel by default
    ):
        super().__init__()

        if bias:
            raise NotImplementedError("BitLinearTriton does not support bias")

        self.in_features = in_features
        self.out_features = out_features
        self.fused = fused

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)

        # Packed weights for inference (registered as buffers)
        K_packed = (in_features + 15) // 16
        self.register_buffer('packed_weight', None)
        self.register_buffer('weight_scale', None)
        self._packed = False

    @property
    def is_packed(self) -> bool:
        return self._packed

    def pack_weights(self) -> None:
        """
        Pack weights into 2-bit format for inference

        Call this after training to enable Triton-accelerated inference.
        """
        packed, scale = pack_weights(self.weight.data)

        # Ensure on CUDA for Triton kernels
        if not packed.is_cuda:
            packed = packed.cuda()
            scale = scale.cuda()

        self.packed_weight = packed
        self.weight_scale = scale
        self._packed = True

    def unpack_weights(self) -> None:
        """
        Unpack weights back to full precision

        Call this if you need to fine-tune after inference.
        """
        if not self._packed:
            return

        if self.packed_weight.is_cuda:
            weight = unpack_weights(
                self.packed_weight,
                self.weight_scale,
                self.in_features
            )
        else:
            raise RuntimeError("Unpacking requires CUDA")

        self.weight.data.copy_(weight)
        self._packed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._packed and x.is_cuda:
            # Inference with Triton kernel
            return bitnet_linear(
                x,
                self.packed_weight,
                self.weight_scale,
                self.in_features,
                self.fused,
            )
        else:
            # Training with STE quantization
            return self._forward_train(x)

    def _forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training with STE quantization
        """
        # Quantize weights to {-1, 0, 1}
        w_quantized = self._quantize_weights(self.weight)

        # Quantize activations
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x / x_scale

        # Linear operation
        output = F.linear(x_scaled, w_quantized)

        return output * x_scale

    def _quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to {-1, 0, 1} with STE
        """
        scale = w.abs().mean().clamp(min=1e-5)
        w_scaled = w / scale
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)

        # STE: forward uses quantized, backward uses original
        return (w_quantized - w).detach() + w

    @classmethod
    def from_linear(cls, linear: nn.Linear, fused: bool = True) -> 'BitLinearTriton':
        """
        Create BitLinearTriton from an existing nn.Linear layer

        Args:
            linear: Source linear layer
            fused: Use fused kernel

        Returns:
            BitLinearTriton layer with copied weights
        """
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=False,
            fused=fused,
        )
        layer.weight.data.copy_(linear.weight.data)
        return layer

    @classmethod
    def from_bitlinear(cls, bitlinear: nn.Module, fused: bool = True) -> 'BitLinearTriton':
        """
        Create BitLinearTriton from a BitLinear layer

        Args:
            bitlinear: Source BitLinear layer
            fused: Use fused kernel

        Returns:
            BitLinearTriton layer with copied weights
        """
        layer = cls(
            bitlinear.in_features,
            bitlinear.out_features,
            bias=False,
            fused=fused,
        )
        layer.weight.data.copy_(bitlinear.weight.data)
        return layer

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'packed={self._packed}, '
            f'fused={self.fused}'
        )


def convert_to_bitlinear_triton(
    model: nn.Module,
    fused: bool = True,
    pack: bool = True,
) -> nn.Module:
    """
    Convert all Linear layers in a model to BitLinearTriton

    Args:
        model: PyTorch model
        fused: Use fused kernels
        pack: Immediately pack weights for inference

    Returns:
        Model with converted layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = BitLinearTriton.from_linear(module, fused=fused)
            if pack:
                new_layer.cuda()
                new_layer.pack_weights()
            setattr(model, name, new_layer)
        else:
            convert_to_bitlinear_triton(module, fused=fused, pack=pack)

    return model
