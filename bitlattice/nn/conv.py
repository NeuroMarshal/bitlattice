"""Causal depthwise convolution along time (the short convolution of linear-attention and state-space layers).

    y = act(bias + causal_conv(x, weight)),  weight [C, W],  act None or SiLU

:func:`causal_conv1d` takes x as [B, C, T] like the ``causal_conv1d`` package and the reference implementations in
transformers; when x is a transposed channels-last tensor (a [B, T, C] tensor viewed as [B, C, T], as these layers
produce it) no copy is made, and the result is returned in the same layout. Backward keeps only x (the reference
keeps x and the pre-activation) and recomputes the pre-activation.

The ``gated_delta_net`` conversion uses it for the linear-attention layers of Qwen3.5 / Qwen3-Next.
"""
from __future__ import annotations

import torch

from .. import _ops

__all__ = ["causal_conv1d"]


class _CausalConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, silu: bool):
        ctx.save_for_backward(x, weight, bias)
        ctx.silu = silu
        return torch.ops.bitlattice.causal_conv1d_fwd(x, weight, bias, silu)

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias = ctx.saved_tensors
        dx, dw, db = torch.ops.bitlattice.causal_conv1d_bwd(x, weight, bias, dy.contiguous(), ctx.silu)
        return (dx, dw.to(weight.dtype) if ctx.needs_input_grad[1] else None,
                db.to(bias.dtype) if bias is not None and ctx.needs_input_grad[2] else None, None)


def _reference(x, weight, bias, activation):
    W = weight.shape[-1]
    y = torch.nn.functional.conv1d(x.to(weight.dtype), weight.unsqueeze(1), bias, padding=W - 1,
                                   groups=x.shape[1])[..., :x.shape[-1]]
    if activation in ("silu", "swish"):
        y = torch.nn.functional.silu(y)
    return y.to(x.dtype)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                  activation: str | None = None, **kwargs) -> torch.Tensor:
    """x [B, C, T] -> [B, C, T]; ``weight`` [C, W] (or [C, 1, W]); ``activation`` None, 'silu' or 'swish'."""
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    fast = (_ops.HAVE_EXT and x.is_cuda and x.dim() == 3 and activation in (None, "silu", "swish")
            and 2 <= weight.shape[-1] <= 8 and x.shape[1] % 2 == 0 and not kwargs.get("seq_idx")
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32))
    if not fast:
        return _reference(x, weight, bias, activation)
    xt = x.transpose(1, 2)                                     # [B, T, C]
    if not xt.is_contiguous():
        xt = xt.contiguous()
    y = _CausalConv1d.apply(xt, weight, bias, activation is not None)
    return y.transpose(1, 2)
