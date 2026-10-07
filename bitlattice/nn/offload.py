"""Weights of modules kept in host memory and brought to the GPU around their use (``checkpoint`` conversion with
``weights="ram"``).

A recomputed module needs its weights twice per step: in its forward (computed without a graph) and in backward
(recomputation, backward, and the fused optimizer update). Between these uses they live in exact-size pinned host
memory. A module's weights stay on the GPU until another streamed module of the same device is used (so all
micro-batches of a pipeline unit share one copy); the weights of the next module in the order of use -- the next one
in forward, the previous one in backward -- are copied to the GPU on a side stream while the current one computes, so
at most two modules' weights are on a GPU at once. Weights that went through backward (updated by a fused optimizer)
are copied back to host memory when they leave the GPU.

Large parameters are streamed (dense parameters and the parts of quantized ones: codes, scales, kept columns); small
ones (norm weights, biases, convolution kernels) stay on the GPU.
"""
from __future__ import annotations

import torch

__all__ = ["STREAM_MIN_NUMEL", "Residency"]

STREAM_MIN_NUMEL = 1 << 16          # parameters (or parts) with fewer elements stay on the GPU

_ORDER: dict = {}                    # device -> [Residency] in model order (the order of use in forward)
_STREAMS: dict = {}                  # device -> side stream for weight transfers
_LOADED: dict = {}                   # device -> the Residency whose weights are in use on it


def _side(device) -> torch.cuda.Stream:
    if device not in _STREAMS:
        _STREAMS[device] = torch.cuda.Stream(device=device)
    return _STREAMS[device]


class Residency:
    """The streamed weights of one module: their host copies, and their GPU copies while the module is in use."""

    def __init__(self, module: torch.nn.Module):
        from ..hostmem import to_pinned
        from ..tensor import LatticeTensor
        self.items = []                                   # (parameter, attribute) holding a streamed tensor
        seen = set()
        for p in module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            names = [k for k, v in p.parts().items() if v is not None] if isinstance(p, LatticeTensor) else ["data"]
            for k in names:
                t = getattr(p, k)
                if t.is_cuda and t.numel() >= STREAM_MIN_NUMEL:
                    self.items.append((p, k))
        if not self.items:
            raise ValueError("no parameter large enough to stream")
        self.device = getattr(self.items[0][0], self.items[0][1]).device
        self.host = [to_pinned(getattr(p, k)) for p, k in self.items]
        self.trainable = any(p.requires_grad for p, _ in self.items)
        self.gpu = None                                   # GPU copies while loaded or being loaded
        self.ready = None                                 # event: the GPU copies are complete
        self.loaded = False
        self.dirty = False                                # updated on the GPU since loaded
        for (p, k), h in zip(self.items, self.host):      # the GPU copies are released now
            self._set(p, k, h)
        order = _ORDER.setdefault(self.device, [])
        self.index = len(order)
        order.append(self)

    @staticmethod
    def _set(p, k, t):
        if k == "data":
            p.data = t
        else:
            setattr(p, k, t)

    def prefetch(self) -> None:
        """Start copying the weights to the GPU (no-op if they are there or on their way)."""
        if self.gpu is not None:
            return
        side = _side(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))   # after any earlier write-back of these weights
        with torch.cuda.stream(side):
            self.gpu = [h.to(self.device, non_blocking=True) for h in self.host]
            self.ready = torch.cuda.Event()
            self.ready.record(side)

    def acquire(self) -> None:
        """Make the weights usable on the current stream (the module in use before on this device leaves)."""
        if self.loaded:
            return
        other = _LOADED.get(self.device)
        if other is not None and other is not self:
            other.release()
        self.prefetch()
        cur = torch.cuda.current_stream(self.device)
        cur.wait_event(self.ready)
        for (p, k), g in zip(self.items, self.gpu):
            g.record_stream(cur)
            self._set(p, k, g)
        self.loaded = True
        _LOADED[self.device] = self

    def neighbour(self, step: int):
        order = _ORDER.get(self.device, [])
        j = self.index + step
        return order[j] if 0 <= j < len(order) and order[j] is not None else None

    def release(self) -> None:
        """Hand the weights back to host memory (copying them first if they were updated) and free the GPU copies."""
        from ..backend import drop_cached
        if not self.loaded:
            return
        cur = torch.cuda.current_stream(self.device)
        side = _side(self.device)
        if self.dirty and self.trainable:
            side.wait_stream(cur)                          # after the update
            with torch.cuda.stream(side):
                for (p, k), h in zip(self.items, self.host):
                    g = getattr(p, k)
                    h.copy_(g, non_blocking=True)
                    g.record_stream(side)
        for (p, k), h in zip(self.items, self.host):
            if k != "data":
                drop_cached(p)                             # derived operands (e.g. transposed FP4 copies)
            self._set(p, k, h)
        self.gpu, self.ready, self.loaded, self.dirty = None, None, False, False
        if _LOADED.get(self.device) is self:
            del _LOADED[self.device]

    def make_resident(self) -> None:
        """Keep the weights on the GPU for good (when the conversion is reverted)."""
        self.acquire()
        self.dirty = False
        _LOADED.pop(self.device, None)
        order = _ORDER.get(self.device, [])
        if self.index < len(order) and order[self.index] is self:
            order[self.index] = None
        self.gpu, self.ready, self.loaded = None, None, False

    def nbytes(self) -> int:
        return sum(h.numel() * h.element_size() for h in self.host)


class StreamedCheckpoint(torch.autograd.Function):
    """Reentrant checkpoint of ``module.original`` whose weights are streamed by ``module.residency``."""

    @staticmethod
    def forward(ctx, module, n_args: int, names, rest: dict, *flat):
        res = module.residency
        ctx.module, ctx.n_args, ctx.names, ctx.rest = module, n_args, names, rest
        ctx.tensor_at = [i for i, a in enumerate(flat) if isinstance(a, torch.Tensor)]
        ctx.others = [a if not isinstance(a, torch.Tensor) else None for a in flat]
        ctx.save_for_backward(*[flat[i] for i in ctx.tensor_at])
        res.acquire()
        nxt = res.neighbour(+1)
        if nxt is not None:
            nxt.prefetch()
        with torch.no_grad():
            out = module.original(*flat[:n_args], **{**rest, **dict(zip(names, flat[n_args:]))})
        return out

    @staticmethod
    def backward(ctx, *grads):
        from .. import backend
        module, res = ctx.module, ctx.module.residency
        saved = ctx.saved_tensors
        flat = list(ctx.others)
        for i, t in zip(ctx.tensor_at, saved):
            flat[i] = t.detach().requires_grad_(t.requires_grad)
        res.acquire()
        prv = res.neighbour(-1)
        if prv is not None:
            prv.prefetch()
        args, kwargs = flat[:ctx.n_args], {**ctx.rest, **dict(zip(ctx.names, flat[ctx.n_args:]))}
        with torch.enable_grad():
            out = module._recompute(args, kwargs)
        outs = out if isinstance(out, tuple) else (out,)
        pairs = [(o, g) for o, g in zip(outs, grads) if isinstance(o, torch.Tensor) and o.requires_grad
                 and g is not None]
        if pairs:
            torch.autograd.backward([o for o, _ in pairs], [g for _, g in pairs])
        mine = {id(p) for p, _ in res.items}
        if any(id(p) in mine for p in backend.pending_wgrad()):
            raise RuntimeError("deferred weight gradients cannot be used with streamed weights (the weights leave "
                               "the GPU before the deferred gradient would be computed)")
        res.dirty = True
        inputs = [flat[i].grad if flat[i].requires_grad else None for i in ctx.tensor_at]
        result = [None] * len(flat)
        for i, g in zip(ctx.tensor_at, inputs):
            result[i] = g
        return (None, None, None, None, *result)
