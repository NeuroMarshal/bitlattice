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

__all__ = ["rms_norm", "rms_norm_gated", "fuse_rmsnorm_"]

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
    return _RMSNormGated.apply(x, gate, w, eps)


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
    """Use the lean RMSNorm in every module named ``*RMSNorm`` whose output it reproduces, and the lean gated
    RMSNorm in every ``*RMSNormGated`` (normalize, then gate with SiLU); the convention is decided once per class on
    a materialized instance. Returns {class name: modules changed}."""
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
    for mod in model.modules():
        cls = type(mod)
        if not cls.__name__.endswith("RMSNormGated") or getattr(mod, "_bl_gated", False):
            continue
        w, eps = getattr(mod, "weight", None), _eps(mod)
        if not isinstance(w, torch.Tensor) or w.dim() != 1 or eps is None or w.device.type == "meta":
            continue
        if cls not in by_class:
            by_class[cls] = _gated_matches(mod, eps)
        if not by_class[cls]:
            continue

        def gated(x, gate=None, _m=mod, _eps=eps, _orig=mod.forward):
            if gate is None:
                return _orig(x)
            return rms_norm_gated(x, gate, _m.weight, _eps)

        mod.forward = gated
        mod._bl_gated = True
        changed[cls.__name__] = changed.get(cls.__name__, 0) + 1
    return changed


def _gated_matches(mod, eps: float) -> bool:
    """Whether ``mod(x, gate)`` is normalize-then-gate with SiLU (the convention :func:`rms_norm_gated` implements)."""
    w = mod.weight
    dt = w.dtype if w.is_floating_point() else torch.bfloat16
    x = torch.randn(4, w.shape[0], device=w.device, dtype=dt)
    g = torch.randn(4, w.shape[0], device=w.device, dtype=dt)
    try:
        with torch.no_grad():
            ref = mod(x, g)
            out = rms_norm_gated(x, g, w, eps)
    except TypeError:
        return False
    return out.shape == ref.shape and torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
