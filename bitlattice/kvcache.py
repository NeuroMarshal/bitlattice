"""Quantized key/value cache for generation, in any lattice format, for keys and values separately.

Layout of one attention layer's cache (per K and V):
    sink   the first ``sink`` tokens, kept in the model dtype (attention sinks dominate many heads)
    store  quantized chunks of ``chunk`` tokens, appended and never re-quantized
    tail   the most recent tokens (``residual`` to ``residual + chunk``), kept in the model dtype

Each chunk is quantized along the head dimension per token ('token' axis) or along the tokens per channel ('channel'
axis; keys have outlier channels, KIVI). The store is one growing buffer per tensor, so a read dequantizes the whole
store with one kernel call. A random Hadamard rotation along the head dimension can be applied before quantizing
(token axis), spreading outliers; reads rotate back. Formats with a per-tensor scale (nvfp4) fix it from the first
chunk with headroom and open a new segment if a later chunk exceeds it.

    cache = bl.kvcache.quantized_cache(model, k="nvfp4", v="nvfp4")   # a transformers DynamicCache
    model.generate(ids, past_key_values=cache, ...)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import _ops
from . import formats as F
from . import reference as R

__all__ = ["KVSpec", "QuantizedKVLayer", "quantized_cache", "cache_bytes"]

HEADROOM = 4.0


@dataclass(frozen=True)
class KVSpec:
    """How one of K / V is stored: ``format`` (any lattice format name or Format; a native dtype keeps it dense),
    ``axis`` 'token' or 'channel', ``rotate`` (Hadamard along the head dimension, token axis only)."""

    format: object = "nvfp4"
    axis: str = "token"
    rotate: bool = False

    def __post_init__(self):
        if self.axis not in ("token", "channel"):
            raise ValueError("axis must be 'token' or 'channel'")
        if self.rotate and self.axis != "token":
            raise ValueError("rotation applies to the token axis")


def _hadamard(n: int, device, dtype) -> torch.Tensor:
    """Orthonormal Hadamard matrix of size n (power of two) with a fixed random sign per row."""
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    s = torch.randint(0, 2, (n,), generator=torch.Generator().manual_seed(0x4B56)).float() * 2 - 1
    return (s[:, None] * h / n ** 0.5).to(device, dtype)


class _Segment:
    """Growing quantized rows of one format (and one global scale, if the format has one)."""

    def __init__(self, fmt: F.Format, cols: int, device, global_scale: torch.Tensor | None):
        self.fmt, self.cols, self.device, self.g = fmt, cols, device, global_scale
        self.rows = 0
        self.packed = self.scales = self.zeros = None

    def _grow(self, need: int, q) -> None:
        cap = 0 if self.packed is None else self.packed.shape[0]
        if self.rows + need <= cap:
            return
        new = max(self.rows + need, 2 * cap, 256)

        def grown(old, part):
            if part is None:
                return None
            t = torch.empty((new, *part.shape[1:]), dtype=part.dtype, device=part.device)
            if old is not None:
                t[:self.rows] = old[:self.rows]
            return t
        self.packed, self.scales = grown(self.packed, q.packed), grown(self.scales, q.scales)
        self.zeros = grown(self.zeros, q.zeros)

    def fits(self, x2: torch.Tensor) -> bool:
        if self.g is None:
            return True
        return bool(x2.abs().amax() <= self.g * self.fmt.max_abs * F.E4M3_MAX)

    def append(self, x2: torch.Tensor) -> None:
        q = _quantize(x2, self.fmt, self.g)
        self._grow(x2.shape[0], q)
        n = x2.shape[0]
        self.packed[self.rows:self.rows + n] = q.packed
        self.scales[self.rows:self.rows + n] = q.scales
        if q.zeros is not None:
            self.zeros[self.rows:self.rows + n] = q.zeros
        self.rows += n

    def read(self, dtype) -> torch.Tensor:
        z = self.zeros[:self.rows] if self.zeros is not None else None
        return _ops.dequantize(self.packed[:self.rows], self.scales[:self.rows], self.g, z, self.fmt, self.rows,
                               self.cols, dtype)

    def select(self, view_fn) -> None:
        """Re-index rows through ``view_fn(tensor[rows, ...]) -> tensor`` (batch selection / reordering)."""
        for name in ("packed", "scales", "zeros"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, view_fn(t[:self.rows]).contiguous())
        self.rows = self.packed.shape[0]

    def nbytes(self) -> int:
        parts = [self.packed, self.scales, self.zeros]
        return sum(t[:self.rows].numel() * t.element_size() for t in parts if t is not None)


def _four_over_six(fmt: F.Format) -> bool:
    return getattr(fmt, "init", "") == "4over6"


def _quantize(x2: torch.Tensor, fmt: F.Format, g: torch.Tensor | None):
    """Rows of x2 quantized along the last dimension (CUDA kernel for NVFP4-layout formats, with four-over-six block
    scales for e.g. 'nvfp4_4o6')."""
    from .backend import is_nvfp4
    if is_nvfp4(fmt) and x2.is_cuda and x2.shape[-1] % 16 == 0 and _ops.HAVE_EXT:
        packed, scales, _ = torch.ops.bitlattice.nvfp4_quantize(x2.contiguous(), False, 0, _four_over_six(fmt), False,
                                                                False, 0, g.float().reshape(1))
        return R.Quantized(packed, scales, g, None, x2.shape[0], x2.shape[1])
    return R.quantize(x2, fmt, method="4over6" if _four_over_six(fmt) else "rtn", global_scale=g)


class _Store:
    """Quantized storage of K or V of one layer: chunks [B, H, chunk, D] appended along the token axis."""

    def __init__(self, spec: KVSpec):
        self.spec = spec
        self.fmt = F.get(spec.format)
        self.segments: list[tuple[_Segment, int]] = []            # (segment, tokens it holds)
        self.tokens = 0
        self.shape = None                                         # (B, H, D)
        self.H = None

    def _layout(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        if self.spec.rotate:
            if self.H is None:
                self.H = _hadamard(D, x.device, torch.float32)
            x = (x.float() @ self.H).to(x.dtype)
        if self.spec.axis == "token":
            return x.permute(2, 0, 1, 3).reshape(T * B * H, D)            # rows: (t, b, h)
        return x.permute(0, 1, 3, 2).reshape(B * H * D, T)                # rows: (b, h, d); one chunk

    def append(self, x: torch.Tensor) -> None:
        B, H, T, D = x.shape
        self.shape = (B, H, D)
        x2 = self._layout(x).float()
        cols = x2.shape[1]
        cols_fmt = F.fit_group(self.fmt, cols) or self.fmt
        if not self.segments or not self.segments[-1][0].fits(x2):
            g = None
            if cols_fmt.scale is not None and cols_fmt.scale.global_scale:
                # four over six: E4M3 scales up to 448 must still fit 6/4 times the block scale of the top level
                top = 256.0 if _four_over_six(cols_fmt) else F.E4M3_MAX
                g = (x2.abs().amax() * HEADROOM / (cols_fmt.max_abs * top)).clamp_min(1e-12).reshape(1)
            self.segments.append((_Segment(cols_fmt, cols, x.device, g), 0))
        seg, n = self.segments[-1]
        seg.append(x2)
        self.segments[-1] = (seg, n + T)
        self.tokens += T

    def read(self, dtype) -> torch.Tensor:
        B, H, D = self.shape
        outs = []
        for seg, n in self.segments:
            y = seg.read(dtype if dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.float32)
            if self.spec.axis == "token":
                y = y.view(n, B, H, D).permute(1, 2, 0, 3)
            else:
                chunks = seg.rows // (B * H * D)
                y = y.view(chunks, B, H, D, n // chunks).permute(1, 2, 0, 4, 3).reshape(B, H, n, D)
            outs.append(y)
        y = torch.cat(outs, 2) if len(outs) > 1 else outs[0]
        if self.spec.rotate:
            y = (y.float() @ self.H.t())
        return y.to(dtype).contiguous()

    def select_batch(self, idx: torch.Tensor) -> None:
        B, H, D = self.shape
        for seg, n in self.segments:
            if self.spec.axis == "token":
                seg.select(lambda t: t.view(n, B, H, *t.shape[1:])[:, idx].reshape(-1, *t.shape[1:]))
            else:
                chunks = seg.rows // (B * H * D)
                seg.select(lambda t: t.view(chunks, B, H * D, *t.shape[1:])[:, idx].reshape(-1, *t.shape[1:]))
        self.shape = (idx.numel(), H, D)

    def nbytes(self) -> int:
        return sum(seg.nbytes() for seg, _ in self.segments)


try:
    from transformers.cache_utils import DynamicLayer as _Base
except Exception:                                                # pragma: no cover - transformers without layers
    _Base = object


class QuantizedKVLayer(_Base):
    """A transformers cache layer storing K and V as described in the module docstring."""

    is_sliding = False
    is_croppable = True

    def __init__(self, k: KVSpec, v: KVSpec, residual: int = 128, chunk: int = 128, sink: int = 0):
        super().__init__()
        if chunk <= 0 or residual < 0 or sink < 0:
            raise ValueError("chunk > 0, residual >= 0, sink >= 0")
        self.k_spec, self.v_spec = k, v
        self.residual, self.chunk, self.n_sink = residual, chunk, sink
        self.k_store, self.v_store = _Store(k), _Store(v)
        self.sink_k = self.sink_v = None
        self.cumulative_length = 0

    def lazy_initialization(self, key_states, value_states) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = key_states.new_zeros((*key_states.shape[:2], 0, key_states.shape[-1]))
        self.values = value_states.new_zeros((*value_states.shape[:2], 0, value_states.shape[-1]))
        self.sink_k, self.sink_v = self.keys, self.values
        self.is_initialized = True

    def _dense(self, spec: KVSpec) -> bool:
        return F.get(spec.format).kind == "native"

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        k, v = key_states, value_states
        if self.sink_k.shape[-2] < self.n_sink:                   # fill the sink first
            take = min(self.n_sink - self.sink_k.shape[-2], k.shape[-2])
            self.sink_k = torch.cat([self.sink_k, k[..., :take, :]], -2)
            self.sink_v = torch.cat([self.sink_v, v[..., :take, :]], -2)
            k, v = k[..., take:, :], v[..., take:, :]
        self.keys = torch.cat([self.keys, k], -2)
        self.values = torch.cat([self.values, v], -2)
        # move whole chunks out of the tail while it is longer than residual + chunk
        excess = self.keys.shape[-2] - self.residual
        n = (excess // self.chunk) * self.chunk if excess >= self.chunk else 0
        if n:
            for c in range(0, n, self.chunk):
                self._store(self.keys[..., c:c + self.chunk, :], self.values[..., c:c + self.chunk, :])
            self.keys, self.values = self.keys[..., n:, :], self.values[..., n:, :]
        return self._full()

    def _store(self, k, v) -> None:
        for spec, store, x in ((self.k_spec, self.k_store, k), (self.v_spec, self.v_store, v)):
            if self._dense(spec):
                store.dense = torch.cat([getattr(store, "dense", x[..., :0, :]), x.to(F.get(spec.format).dtype)], -2)
                store.tokens += x.shape[-2]
            else:
                store.append(x)

    def _read(self, spec, store):
        if store.tokens == 0:
            return None
        if self._dense(spec):
            return store.dense.to(self.dtype)
        return store.read(self.dtype)

    def _full(self):
        parts_k = [p for p in (self.sink_k, self._read(self.k_spec, self.k_store), self.keys)
                   if p is not None and p.shape[-2]]
        parts_v = [p for p in (self.sink_v, self._read(self.v_spec, self.v_store), self.values)
                   if p is not None and p.shape[-2]]
        return torch.cat(parts_k, -2), torch.cat(parts_v, -2)

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return -1

    get_max_cache_shape = get_max_length                          # name in older transformers

    def get_mask_sizes(self, query) -> tuple[int, int]:
        """(kv length, kv offset); ``query`` is the query length (newer transformers) or the cache positions."""
        q = query.shape[0] if isinstance(query, torch.Tensor) else int(query)
        return self.get_seq_length() + q, 0

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:                                    # a target length (older transformers API)
            tokens_to_remove = tokens_to_remove - self.get_seq_length()
        n = -tokens_to_remove if tokens_to_remove < 0 else 0
        if n == 0:
            return
        if n > self.keys.shape[-2]:
            raise NotImplementedError("cropping into the quantized part of the cache")
        self.keys, self.values = self.keys[..., :-n, :], self.values[..., :-n, :]
        self.cumulative_length -= n

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if not self.is_initialized:
            return
        self.keys, self.values = self.keys[indices], self.values[indices]
        self.sink_k, self.sink_v = self.sink_k[indices], self.sink_v[indices]
        for spec, store in ((self.k_spec, self.k_store), (self.v_spec, self.v_store)):
            if store.tokens == 0:
                continue
            if self._dense(spec):
                store.dense = store.dense[indices]
            else:
                store.select_batch(indices.to(self.device))

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.is_initialized:
            b = self.keys.shape[0]
            self.batch_select_indices(torch.arange(b, device=self.device).repeat_interleave(repeats))

    def nbytes(self) -> dict[str, int]:
        dense = lambda t: 0 if t is None else t.numel() * t.element_size()  # noqa: E731
        q = 0
        for spec, store in ((self.k_spec, self.k_store), (self.v_spec, self.v_store)):
            q += dense(getattr(store, "dense", None)) if self._dense(spec) else store.nbytes()
        return {"quantized": q, "dense": dense(self.keys) + dense(self.values) + dense(self.sink_k) +
                dense(self.sink_v)}

    def offload(self):                                              # host offload is not supported for this layer
        pass

    def prefetch(self):
        pass


def _spec(x, axis: str) -> KVSpec:
    if isinstance(x, KVSpec):
        return x
    return KVSpec(format=x, axis=axis)


def quantized_cache(model_or_config, k="nvfp4", v=None, *, k_axis: str = "channel", v_axis: str = "token",
                    residual: int = 128, chunk: int = 128, sink: int = 0, rotate: bool = False):
    """A transformers DynamicCache whose full-attention layers store K / V quantized (other layer kinds, e.g. the
    recurrent states of linear-attention layers or sliding windows, are kept as the model defines them).

    k, v      -- formats (or KVSpec); ``v=None`` uses the format of ``k``
    k_axis    -- 'channel' (default, per-channel keys) or 'token'
    v_axis    -- 'token' (default) or 'channel'
    rotate    -- Hadamard rotation along the head dimension for token-axis tensors
    """
    from transformers.cache_utils import DynamicCache, DynamicLayer
    config = getattr(model_or_config, "config", model_or_config)
    cache = DynamicCache(config=config)
    ks = _spec(k, k_axis)
    vs = _spec(v if v is not None else ks.format, v_axis)
    if rotate:
        ks = KVSpec(ks.format, ks.axis, ks.axis == "token")
        vs = KVSpec(vs.format, vs.axis, vs.axis == "token")
    for i, layer in enumerate(cache.layers):
        if type(layer) is DynamicLayer:
            cache.layers[i] = QuantizedKVLayer(ks, vs, residual=residual, chunk=chunk, sink=sink)
    return cache


def cache_bytes(cache) -> dict[str, int]:
    """Bytes held by the quantized and dense parts of a cache built by :func:`quantized_cache`."""
    out = {"quantized": 0, "dense": 0}
    for layer in cache.layers:
        if isinstance(layer, QuantizedKVLayer):
            b = layer.nbytes()
            out["quantized"] += b["quantized"]
            out["dense"] += b["dense"]
        elif getattr(layer, "keys", None) is not None and isinstance(layer.keys, torch.Tensor):
            out["dense"] += layer.keys.numel() * layer.keys.element_size() * 2
    return out
