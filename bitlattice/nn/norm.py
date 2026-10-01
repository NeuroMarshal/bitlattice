"""Memory-lean RMSNorm for existing models.

Reference implementations normalize in float32 and keep several float32 copies of the input for backward (12 bytes
per element). :class:`_RMSNorm` keeps the input in its own dtype plus one float32 value per row and recomputes the
rest in backward, in row chunks so that the float32 temporaries stay bounded.

The ``rms_norm`` conversion (``bl.surgery.convert_``) puts a lean norm in place of a model's RMSNorm modules
(parameters and names unchanged). Two conventions are recognized by comparing against the original on a random
input:
    'llama'  y = w * cast(x * rstd)              (Llama, Qwen2/3, Mistral, ...)
    'offset' y = cast(x * rstd * (1 + w))        (Gemma, Qwen3.5, ...)
Modules whose output matches neither are left alone.
"""
from __future__ import annotations

import torch

from .. import _ops

__all__ = ["rms_norm", "rms_norm_gated"]

_ROWS = 4096


def _kernel(x: torch.Tensor, w: torch.Tensor) -> bool:
    """The CUDA kernels (one pass, the same per-element arithmetic; only the order of row sums differs) unless
    switched off (``bitlattice.kernels(fused_norm=False)``)."""
    from ..kernel_choice import active
    return (_ops.HAVE_EXT and x.is_cuda and w.dtype == x.dtype and active().fused_norm
            and x.dtype in (torch.bfloat16, torch.float16, torch.float32))


class _NormKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate, w, eps: float, offset: float, cast_first: bool):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        g2 = None if gate is None else gate.reshape(-1, shape[-1]).contiguous()
        y, rstd = torch.ops.bitlattice.rms_norm_fwd(x2, g2, w.contiguous(), eps, offset, cast_first)
        ctx.save_for_backward(x2, g2, w, rstd)
        ctx.offset, ctx.shape = offset, shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, gy):
        x2, g2, w, rstd = ctx.saved_tensors
        need_x, need_g, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        dx, dg, dw = torch.ops.bitlattice.rms_norm_bwd(x2, g2, w.contiguous(), rstd,
                                                        gy.reshape(x2.shape).contiguous(), ctx.offset, need_x,
                                                        need_g, need_w)
        return (dx.view(ctx.shape) if need_x else None, dg.view(ctx.shape) if need_g and g2 is not None else None,
                dw.to(w.dtype) if need_w else None, None, None, None)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps: float, offset: float, cast_first: bool):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        wf = w.float() + offset
        for s in range(0, x2.shape[0], _ROWS):
            xf = x2[s:s + _ROWS].float()
            r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            rstd[s:s + _ROWS] = r.squeeze(-1)
            if cast_first:
                out[s:s + _ROWS] = (w * (xf * r).to(x.dtype)).to(x.dtype)
            else:
                out[s:s + _ROWS] = (xf * r * wf).to(x.dtype)
        ctx.save_for_backward(x, w, rstd)
        ctx.offset, ctx.cast_first = offset, cast_first
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, gy):
        x, w, rstd = ctx.saved_tensors
        d = x.shape[-1]
        x2, g2 = x.reshape(-1, d), gy.reshape(-1, d)
        gx = torch.empty_like(x2) if ctx.needs_input_grad[0] else None
        gw = torch.zeros(d, dtype=torch.float32, device=x.device) if ctx.needs_input_grad[1] else None
        wf = w.float() + ctx.offset
        for s in range(0, x2.shape[0], _ROWS):
            r = rstd[s:s + _ROWS, None]
            xh = x2[s:s + _ROWS].float() * r
            g = g2[s:s + _ROWS].float()
            if gw is not None:
                gw += (g * xh).sum(0)
            if gx is not None:
                gxh = g * wf
                gx[s:s + _ROWS] = (r * (gxh - xh * (gxh * xh).mean(-1, keepdim=True))).to(x.dtype)
        return (gx.reshape(x.shape) if gx is not None else None,
                gw.to(w.dtype) if gw is not None else None, None, None, None)


class _RMSNormGated(torch.autograd.Function):
    """y = cast(cast(w * cast(x * rstd)) * silu(gate)) (normalize, then gate; as in Qwen3.5 / Qwen3-Next linear
    attention). Reference implementations keep several float32 copies of x and gate (normalized input, its product
    with the weight, silu(gate), the product); here only x, gate (their own dtype) and rstd are kept."""

    @staticmethod
    def forward(ctx, x, gate, w, eps: float):
        shape = x.shape
        x2, g2 = x.reshape(-1, shape[-1]), gate.reshape(-1, shape[-1])
        out = torch.empty_like(x2)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        for s in range(0, x2.shape[0], _ROWS):
            xf = x2[s:s + _ROWS].float()
            r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
            rstd[s:s + _ROWS] = r.squeeze(-1)
            b = w * (xf * r).to(x.dtype)
            out[s:s + _ROWS] = (b * torch.nn.functional.silu(g2[s:s + _ROWS].float())).to(x.dtype)
        ctx.save_for_backward(x, gate, w, rstd)
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, gy):
        x, gate, w, rstd = ctx.saved_tensors
        d = x.shape[-1]
        x2, g2, gy2 = x.reshape(-1, d), gate.reshape(-1, d), gy.reshape(-1, d)
        gx = torch.empty_like(x2) if ctx.needs_input_grad[0] else None
        gg = torch.empty_like(g2) if ctx.needs_input_grad[1] else None
        gw = torch.zeros(d, dtype=torch.float32, device=x.device) if ctx.needs_input_grad[2] else None
        wf = w.float()
        for s in range(0, x2.shape[0], _ROWS):
            r = rstd[s:s + _ROWS, None]
            n = x2[s:s + _ROWS].float() * r                                   # normalized input
            a = n.to(x.dtype).float()
            z = g2[s:s + _ROWS].float()
            sg = torch.sigmoid(z)
            dy = gy2[s:s + _ROWS].float()
            if gg is not None:
                b = (w * a.to(x.dtype)).float()
                gg[s:s + _ROWS] = (dy * b * sg * (1.0 + z * (1.0 - sg))).to(gate.dtype)
            db = dy * (z * sg)                                                # d(w * a)
            if gw is not None:
                gw += (db * a).sum(0)
            if gx is not None:
                dn = db * wf
                gx[s:s + _ROWS] = (r * (dn - n * (dn * n).mean(-1, keepdim=True))).to(x.dtype)
        return (gx.reshape(x.shape) if gx is not None else None, gg.reshape(gate.shape) if gg is not None else None,
                gw.to(w.dtype) if gw is not None else None, None)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    if _kernel(x, w) and gate.dtype == x.dtype:
        return _NormKernel.apply(x, gate, w, eps, 0.0, True)
    return _RMSNormGated.apply(x, gate, w, eps)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, offset: float = 0.0,
             cast_first: bool = True) -> torch.Tensor:
    if _kernel(x, w):
        return _NormKernel.apply(x, None, w, eps, offset, cast_first)
    return _RMSNorm.apply(x, w, eps, offset, cast_first)


def _eps(mod) -> float | None:
    for name in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(mod, name, None)
        if isinstance(v, float):
            return v
    return None


def _convention(mod, eps: float):
    """'llama' / 'offset' if a lean norm reproduces ``mod`` on a random input, else None."""
    w = mod.weight
    if w.device.type == "meta":
        return None
    x = torch.randn(4, w.shape[0], device=w.device, dtype=w.dtype if w.is_floating_point() else torch.bfloat16)
    with torch.no_grad():
        ref = mod(x)
        for name, offset, cast_first in (("llama", 0.0, True), ("offset", 1.0, False)):
            out = rms_norm(x, w, eps, offset, cast_first)
            if out.shape == ref.shape and torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2):
                return name, offset, cast_first
    return None
