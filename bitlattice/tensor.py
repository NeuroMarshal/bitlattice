"""LatticeTensor: a torch.Tensor subclass holding a weight in a lattice format.

The logical tensor (shape, dtype) is what PyTorch code sees; the storage is packed codes + group scales (+ global
scale, zero points) and an optional set of input columns kept in high precision. ``F.linear`` and ``F.embedding``
use dedicated paths; every other operator falls back to the dequantized weight, so arbitrary modules work and
autograd produces a dense gradient of the logical shape.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils._pytree import tree_map

from . import _ops
from . import formats as F
from . import reference as R

__all__ = ["LatticeTensor", "is_lattice"]

_PARTS = ("packed", "scales", "global_scale", "zeros", "hi_idx", "hi_val")


def is_lattice(t) -> bool:
    return isinstance(t, LatticeTensor)


class LatticeTensor(torch.Tensor):
    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor | None
    zeros: torch.Tensor | None
    hi_idx: torch.Tensor | None
    hi_val: torch.Tensor | None
    fmt: F.Format

    @staticmethod
    def __new__(cls, packed, scales, global_scale, zeros, fmt, shape, dtype=torch.bfloat16, hi_idx=None, hi_val=None,
                requires_grad=False):
        t = torch.Tensor._make_wrapper_subclass(cls, torch.Size(shape), dtype=dtype, device=packed.device,
                                                requires_grad=requires_grad)
        t.packed, t.scales, t.global_scale, t.zeros = packed, scales, global_scale, zeros
        t.hi_idx, t.hi_val = hi_idx, hi_val
        t.fmt = F.get(fmt)
        t.compute = None          # bitlattice.Compute set by surgery.set_compute_; None: exact (see bitlattice.compute)
        return t

    def __init__(self, *args, **kwargs):  # state is set in __new__
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------
    # construction

    @classmethod
    def from_float(cls, w: torch.Tensor, fmt, method: str = "mse", keep: float = 0.0,
                   keep_dtype: torch.dtype = torch.bfloat16, saliency: torch.Tensor | None = None,
                   dtype: torch.dtype | None = None) -> "LatticeTensor":
        """Quantize a dense tensor. ``keep``: fraction of input columns (of the 2-D view) stored in ``keep_dtype``,
        chosen by ``saliency`` (per column, larger = more important; default: column L2 norm)."""
        fmt = F.get(fmt)
        w2 = R.view2d(w.detach()).to(torch.float32)
        rows, cols = w2.shape
        hi_idx = hi_val = None
        k = int(round(max(0.0, min(1.0, keep)) * cols))
        if k > 0:
            score = saliency.to(w2.device, torch.float32) if saliency is not None else w2.square().sum(0)
            hi_idx = torch.topk(score, k).indices.sort().values
            hi_val = w2[:, hi_idx].to(keep_dtype)
            w2 = w2.clone()
            w2[:, hi_idx] = 0.0
        q = R.quantize(w2, fmt, method=method)
        logical = dtype or (w.dtype if w.dtype.is_floating_point else torch.bfloat16)
        return cls(q.packed, q.scales, q.global_scale, q.zeros, fmt, w.shape, logical, hi_idx, hi_val)

    # ------------------------------------------------------------------------------------------------------------
    # views of the storage

    @property
    def rows(self) -> int:
        return 1 if self.dim() == 1 else self.shape[0]

    @property
    def cols(self) -> int:
        return self.numel() // self.rows

    def parts(self) -> dict[str, torch.Tensor | None]:
        return {k: getattr(self, k) for k in _PARTS}

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dense tensor of the logical shape in ``dtype`` (default: the logical dtype)."""
        dtype = dtype or self.dtype
        kd = dtype if dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.float32
        w = _ops.dequantize(self.packed, self.scales, self.global_scale, self.zeros, self.fmt, self.rows, self.cols, kd)
        if self.hi_idx is not None:
            w[:, self.hi_idx] = self.hi_val.to(kd)
        return w.to(dtype).reshape(self.shape)

    def dequantize_rows(self, idx: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Dequantize selected rows only ([len(idx), cols]); requires per-row or finer groups."""
        dtype = dtype or self.dtype
        spec = self.fmt.scale
        if spec.group == 0:
            return self.dequantize(dtype).reshape(self.rows, self.cols)[idx]
        kd = dtype if dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.float32
        z = self.zeros[idx] if self.zeros is not None else None
        w = _ops.dequantize(self.packed[idx].contiguous(), self.scales[idx].contiguous(), self.global_scale, z,
                            self.fmt, idx.numel(), self.cols, kd)
        if self.hi_idx is not None:
            w[:, self.hi_idx] = self.hi_val[idx].to(kd)
        return w.to(dtype)

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.parts().values() if t is not None)

    def __repr__(self, *, tensor_contents=None) -> str:
        keep = f", keep={self.hi_idx.numel()}/{self.cols}" if self.hi_idx is not None else ""
        return (f"LatticeTensor(shape={tuple(self.shape)}, format={self.fmt.name}, dtype={self.dtype}, "
                f"device={self.device}{keep})")

    # ------------------------------------------------------------------------------------------------------------
    # torch.compile / FX support

    def __tensor_flatten__(self):
        names = [k for k in _PARTS if getattr(self, k) is not None]
        return names, (self.fmt, tuple(self.shape), self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner, ctx, outer_size, outer_stride):
        fmt, shape, dtype = ctx
        return LatticeTensor(inner["packed"], inner["scales"], inner.get("global_scale"), inner.get("zeros"), fmt,
                             shape, dtype, inner.get("hi_idx"), inner.get("hi_val"))

    def _map(self, fn, **kw) -> "LatticeTensor":
        p = {k: (fn(v) if v is not None else None) for k, v in self.parts().items()}
        t = LatticeTensor(p["packed"], p["scales"], p["global_scale"], p["zeros"], self.fmt, kw.get("shape", self.shape),
                          kw.get("dtype", self.dtype), p["hi_idx"], p["hi_val"])
        t.compute = self.compute
        return t

    # ------------------------------------------------------------------------------------------------------------
    # operator interception

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            if isinstance(w, LatticeTensor) and not isinstance(x, LatticeTensor):
                return _LatticeLinear.apply(x, w, b)
        if func is torch.nn.functional.embedding:
            idx, w = args[0], args[1]
            if isinstance(w, LatticeTensor) and not kwargs.get("max_norm") and len(args) <= 2:
                return _LatticeEmbedding.apply(idx, w, kwargs.get("padding_idx"))
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten
        if func in (aten.detach.default, aten.alias.default):
            return args[0]._map(lambda t: t)
        if func is aten.clone.default:
            return args[0]._map(lambda t: t.clone())
        if func is aten._to_copy.default:
            src = args[0]
            device = kwargs.get("device", src.device)
            dtype = kwargs.get("dtype", src.dtype)
            moved = src._map(lambda t: t.to(device))
            moved_dtype = dtype if dtype is not None and dtype.is_floating_point else src.dtype
            return moved._map(lambda t: t, dtype=moved_dtype)
        if func is aten.copy_.default:
            dst, src = args[0], args[1]
            dst._bl_version = getattr(dst, "_bl_version", 0) + 1        # invalidates cached derived operands
            if isinstance(src, LatticeTensor) and src.fmt == dst.fmt:
                for k in _PARTS:
                    d, s = getattr(dst, k), getattr(src, k)
                    if d is not None and s is not None and d.shape == s.shape:
                        d.copy_(s)
                    else:
                        setattr(dst, k, s.to(dst.device).clone() if s is not None else None)
                return dst
            dense = src.dequantize(torch.float32) if isinstance(src, LatticeTensor) else src
            q = LatticeTensor.from_float(dense.to(dst.device), dst.fmt, method="rtn", dtype=dst.dtype)
            for k in _PARTS:
                setattr(dst, k, getattr(q, k))
            return dst
        if func in (aten.is_same_size.default,):
            return args[0].shape == args[1].shape

        def unwrap(t):
            return t.dequantize() if isinstance(t, LatticeTensor) else t

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


# --------------------------------------------------------------------------------------------------------------
# autograd paths

def _kernels(ctx):
    from .backend import reference_kernels
    return reference_kernels(ctx.reference)


class _LatticeLinear(torch.autograd.Function):
    """y = x W^T + b with W dequantized on the fly (forward and backward); no dense W is kept between them."""

    @staticmethod
    def forward(ctx, x, w: LatticeTensor, bias):
        from .compute_policy import route
        ctx.w = w
        ctx.has_bias = bias is not None
        ctx.policy, comp = route(x, w)
        ctx.recipe = comp.nvfp4                         # the backward runs on other threads: keep the forward's
        ctx.reference = comp.kernels == "reference"
        with _kernels(ctx):
            return _LatticeLinear._forward(ctx, x, w, bias)

    @staticmethod
    def _forward(ctx, x, w, bias):
        from . import backend
        # the input is kept only for the weight gradient: not when the weight is frozen, and not in the no-grad
        # first pass of a checkpointed block (its recomputation saves it)
        need_w = ctx.needs_input_grad[1]
        if ctx.policy == "fp4_all":
            ctx.x_meta = (x.shape, x.dtype)
            if need_w:
                y, xt = backend.fp4_forward_saving(x, w, bias, ctx.recipe)
                # the rotation seed is saved as a tensor, not a ctx attribute: checkpointed recomputation replaces
                # the saved tensors (a new x^T with a new rotation), and the seed must come with them
                meta = torch.tensor([int(xt.blocked), xt.rotate_seed], dtype=torch.int64)
                ctx.save_for_backward(xt.packed, xt.scales, xt.global_scale, meta)
                return y
            ctx.save_for_backward()
            return backend.nvfp4_linear(x, w, bias, ctx.recipe)
        if ctx.policy == "mxfp8_all":
            ctx.x_meta = (x.shape, x.dtype)
            if need_w:
                y, xt = backend.mx_forward_saving(x, w, bias)
                ctx.save_for_backward(xt.codes, xt.scales)
                return y
            ctx.save_for_backward()
            return backend.mx_linear(x, w, bias)
        ctx.save_for_backward(x if need_w else None)
        if ctx.policy == "fp4":
            return backend.nvfp4_linear(x, w, bias, ctx.recipe)
        if ctx.policy == "mxfp8":
            return backend.mx_linear(x, w, bias)
        return torch.nn.functional.linear(x, w.dequantize(x.dtype), bias)

    @staticmethod
    def backward(ctx, gy):
        with _kernels(ctx):
            return _LatticeLinear._backward(ctx, gy)

    @staticmethod
    def _backward(ctx, gy):
        w = ctx.w
        gb = gy.reshape(-1, gy.shape[-1]).sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        if ctx.policy == "fp4_all":
            from . import backend
            shape, dtype = ctx.x_meta
            saved = ctx.saved_tensors
            xt = None
            if saved:
                blocked, seed = saved[3].tolist()
                xt = backend.Fp4Operand(*saved[:3], bool(blocked), seed)
            gx, gw = backend.fp4_backward(gy, w, xt, ctx.needs_input_grad[0],
                                          ctx.needs_input_grad[1] and xt is not None, ctx.recipe)
            gx = gx.to(dtype).reshape(shape) if gx is not None else None
            gw = gw.to(w.dtype) if gw is not None else None          # None: deferred (backend.defer_wgrad)
            return gx, gw, gb
        if ctx.policy == "mxfp8_all":
            from . import backend
            shape, dtype = ctx.x_meta
            saved = ctx.saved_tensors
            xt = backend.Mx8Operand(*saved) if saved else None
            gx, gw = backend.mx_backward(gy, w, xt, ctx.needs_input_grad[0],
                                         ctx.needs_input_grad[1] and xt is not None)
            gx = gx.to(dtype).reshape(shape) if gx is not None else None
            gw = gw.to(w.dtype) if gw is not None else None
            return gx, gw, gb
        (x,) = ctx.saved_tensors
        W = w.dequantize(gy.dtype)
        gx = gy @ W if ctx.needs_input_grad[0] else None
        gw = None
        if ctx.needs_input_grad[1] and x is not None:
            gw = (gy.reshape(-1, gy.shape[-1]).t() @ x.reshape(-1, x.shape[-1])).to(w.dtype)
        return gx, gw, gb


class _LatticeEmbedding(torch.autograd.Function):
    """Row lookup dequantizing only the referenced rows."""

    @staticmethod
    def forward(ctx, idx, w: LatticeTensor, padding_idx):
        flat = idx.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = w.dequantize_rows(uniq)
        out = rows[inv].reshape(*idx.shape, w.cols)
        ctx.save_for_backward(idx)
        ctx.shape, ctx.dtype, ctx.padding_idx = w.shape, w.dtype, padding_idx
        return out

    @staticmethod
    def backward(ctx, gy):
        """Row-sparse gradient (COO over rows): only referenced rows are materialized."""
        (idx,) = ctx.saved_tensors
        flat = idx.reshape(-1)
        rows, inv = torch.unique(flat, return_inverse=True)
        vals = torch.zeros(rows.numel(), gy.shape[-1], dtype=ctx.dtype, device=gy.device)
        vals.index_add_(0, inv, gy.reshape(-1, gy.shape[-1]).to(ctx.dtype))
        if ctx.padding_idx is not None:
            vals[rows == ctx.padding_idx] = 0
        gw = torch.sparse_coo_tensor(rows.unsqueeze(0), vals, (ctx.shape[0], gy.shape[-1]), is_coalesced=True,
                                     check_invariants=False)
        return None, gw.reshape(ctx.shape) if len(ctx.shape) != 2 else gw, None


def to_parameter(t: LatticeTensor, requires_grad: bool = True) -> torch.nn.Parameter:
    return torch.nn.Parameter(t, requires_grad=requires_grad)


def summary(t: Any) -> str:
    return repr(t)


def _register_safe_globals() -> None:
    """Allow torch.load(weights_only=True) of state dicts containing LatticeTensor."""
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is not None:
        add([LatticeTensor, F.TableFormat, F.UniformFormat, F.NativeFormat, F.ScaleSpec])


_register_safe_globals()
