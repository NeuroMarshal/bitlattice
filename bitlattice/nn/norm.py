"""Memory-lean RMSNorm for existing models.

Reference implementations normalize in float32 and keep several float32 copies of the input for backward (12 bytes
per element). :class:`_RMSNorm` keeps the input in its own dtype plus one float32 value per row and recomputes the
rest in backward, in row chunks so that the float32 temporaries stay bounded.

:func:`fuse_rmsnorm_` swaps the forward of the RMSNorm modules of a model in place (parameters and names unchanged).
Two conventions are recognized by comparing against the original on a random input:
    'llama'  y = w * cast(x * rstd)              (Llama, Qwen2/3, Mistral, ...)
    'offset' y = cast(x * rstd * (1 + w))        (Gemma, Qwen3.5, ...)
Modules whose output matches neither are left alone.
"""
from __future__ import annotations

import torch

__all__ = ["rms_norm", "fuse_rmsnorm_"]

_ROWS = 4096


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


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, offset: float = 0.0,
             cast_first: bool = True) -> torch.Tensor:
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


def fuse_rmsnorm_(model: torch.nn.Module) -> dict[str, int]:
    """Use the lean RMSNorm in every module named ``*RMSNorm`` (not gated variants) whose output it reproduces;
    the convention is decided once per class on a materialized instance. Returns {class name: modules changed}."""
    by_class: dict[type, object] = {}
    changed: dict[str, int] = {}
    for mod in model.modules():
        cls = type(mod)
        if not cls.__name__.endswith("RMSNorm") or not isinstance(getattr(mod, "weight", None), torch.Tensor):
            continue
        if mod.weight.dim() != 1:
            continue
        eps = _eps(mod)
        if eps is None:
            continue
        if cls not in by_class or by_class[cls] is None:
            conv = _convention(mod, eps)
            if conv is None and mod.weight.device.type == "meta":
                continue                                   # decide on a materialized instance
            by_class[cls] = conv
        conv = by_class[cls]
        if conv is None:
            continue
        _, offset, cast_first = conv

        def forward(x, _m=mod, _eps=eps, _o=offset, _c=cast_first):
            return rms_norm(x, _m.weight, _eps, _o, _c)

        mod.forward = forward
        changed[cls.__name__] = changed.get(cls.__name__, 0) + 1
    return changed
