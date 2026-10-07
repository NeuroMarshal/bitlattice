"""Explicit layer conversion: replace chosen modules of a model by Bitlattice modules and back.

A conversion is always asked for by name and target; nothing is converted implicitly. The new module adopts the
original's parameters, buffers and child modules -- the same objects under the same names -- so the state dict, the
optimizer and any tied weights are unaffected and no data is lost; :func:`revert_` puts the original module back
(with the current parameters). Before a module is replaced, the two are run on a random input and must agree,
otherwise the conversion is refused.

    bl.surgery.conversions()                                         # name -> what it does
    bl.surgery.convert_(model, "*.input_layernorm", "rms_norm")
    bl.surgery.convert_(model, "*.linear_attn", "gated_delta_net", seq_chunk=4096)
    bl.surgery.convert_(model, "*.mlp", "token_chunks", chunk=2048)
    bl.surgery.convert_(model, "*.linear_attn", "compressed_saves", fmt="fp8")   # stacks on the previous one
    bl.surgery.revert_(model, "*.linear_attn")                       # one level back

``target`` is a selector (see :mod:`bitlattice.select`). Further conversions are added with
:func:`register_conversion` (e.g. by a model integration for its own layer types).

Conversions:
    rms_norm          lean RMSNorm: keeps the input and one fp32 value per row (reference norms keep fp32 copies)
    gated_rms_norm    lean normalize-then-gate RMSNorm (linear-attention output norms)
    gated_delta_net   GatedDeltaNet layer (Qwen3.5 / Qwen3-Next): fused causal convolution, q / k in their own
                      heads (no repeated copies); ``seq_chunk``: the delta rule over pieces of that many tokens;
                      ``chunk``: the whole layer over pieces of that many tokens (convolution inputs and recurrent
                      state carried between pieces), each piece recomputed in backward -- the layer keeps its input
                      only; ``precision``: 'bf16' runs fla's kernel on the activations' dtype (as fla / vLLM do),
                      'fp32' on float32 inputs (as transformers' own implementation without the hub kernel; ~1-2 %
                      lower perplexity on a 27B model, slower); calls with a cache (generation) go to the original
    token_chunks      a position-wise module (MLP) over pieces of ``chunk`` tokens, each recomputed in backward
    attention_chunks  a causal self-attention layer (q/k/v/o projections, optional q/k norms and output gate, e.g.
                      Qwen3 / Qwen3.5) over pieces of ``chunk`` queries: keys and values are projected once for
                      all tokens, every query piece (projection, rotary, attention, gate, output projection) is
                      recomputed in backward -- the layer keeps its input, K and V instead of all its internals;
                      runs on fused attention kernels only (never a materialized score matrix); calls with a
                      cache or unknown attention arguments go to the original module
    checkpoint        activation checkpointing of the module: the forward keeps only its inputs, backward
                      recomputes it; ``saved`` ('fp8' / 'nvfp4', lossy) stores the recomputed module's saved
                      tensors compressed, which lowers the per-layer peak (use it instead of the model's own
                      gradient checkpointing for these modules); ``weights='ram'`` keeps the module's weights in
                      pinned host memory between its uses and streams them in (bitlattice.nn.offload): about two
                      modules' weights on the GPU at a time
    host_embedding    a frozen embedding table kept in pinned host memory: lookups gather the rows on the CPU and copy
                      only them to the device of the indices (frees the table's GPU memory, e.g. 2.5 GB for a
                      250k x 5120 vocabulary); refused for trainable or tied tables
    host_linear       a frozen linear layer (e.g. the output head) whose weight stays in pinned host memory and is
                      streamed to the GPU in row tiles when used; :func:`linear_cross_entropy` streams it once per
                      group of tokens (2.5 GB freed for a 250k x 5120 head); refused for trainable or tied weights
    compressed_saves  tensors saved for backward inside the module stored as ``fmt`` ('fp8' or 'nvfp4'; lossy) --
                      for modules that are not recomputed: inside a checkpointed region the hooks would also act in
                      its first pass and keep what checkpointing discards; there use ``checkpoint`` with ``saved``
"""
from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..select import Selector, select
from ..tensor import LatticeTensor

log = logging.getLogger("bitlattice")

__all__ = [
    "AttentionChunks",
    "Checkpointed",
    "CompressedSaves",
    "Conversion",
    "Converted",
    "GatedDeltaNet",
    "GatedRMSNorm",
    "HostEmbedding",
    "HostLinear",
    "RMSNorm",
    "TokenChunks",
    "Unfit",
    "conversions",
    "convert_",
    "register_conversion",
    "revert_",
]


class Converted(torch.nn.Module):
    """Base of converted modules: shares the original's parameters, buffers and children (the same dictionaries, so
    an edit through either -- a nested conversion, a replaced child, a quantized parameter -- is seen by both) and
    remembers the original module for :func:`revert_`."""

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        self.__dict__["_bl_original"] = original                 # not a child: its parameters appear once
        self._parameters = original._parameters
        self._buffers = original._buffers
        self._non_persistent_buffers_set = original._non_persistent_buffers_set
        self._modules = original._modules
        self.training = original.training

    def train(self, mode: bool = True):
        super().train(mode)
        self.original.training = mode                           # the original runs the computation it delegates
        return self

    def _delegate(self, why: str, *args, **kwargs):
        """Run the original module for a call this conversion does not handle: fine without gradients (evaluation,
        generation), an error in training, where the conversion was asked for."""
        if torch.is_grad_enabled() and self.training:
            raise ValueError(f"{type(self).__name__} conversion of {type(self.original).__name__}: {why} in a "
                             "training call; the conversion does not run it (revert it or change the call)")
        return self.original(*args, **kwargs)

    @property
    def original(self) -> torch.nn.Module:
        return self.__dict__["_bl_original"]

    def installed(self) -> None:
        """Called once the conversion is applied (after its check): conversions that move data do it here."""

    def __getattr__(self, name: str):
        # parameters, buffers and children are this module's own; the original's plain instance attributes
        # (configuration such as a decoder layer's attention_type, which the model reads) stay readable
        try:
            return super().__getattr__(name)
        except AttributeError:
            attrs = self.__dict__["_bl_original"].__dict__ if "_bl_original" in self.__dict__ else {}
            if name.startswith("_") or name not in attrs:
                raise
            return attrs[name]

    def restored(self) -> torch.nn.Module:
        """The original module (it shares this module's parameters, buffers and children)."""
        orig = self.original
        orig.train(self.training)
        return orig

    def settings(self) -> str:
        """The conversion's options, for printing."""
        return ""

    def extra_repr(self) -> str:
        return ", ".join(x for x in (self.settings(), f"from={type(self.original).__name__}") if x)


def _tensors(obj):
    """Tensors nested in tuples, lists and dicts."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _tensors(x)


# ---- module implementations ---------------------------------------------------------------------------------------

class RMSNorm(Converted):
    def __init__(self, original, eps: float, offset: float, cast_first: bool):
        super().__init__(original)
        self.eps, self.offset, self.cast_first = eps, offset, cast_first

    def forward(self, x):
        from .norm import rms_norm
        return rms_norm(x, self.weight, self.eps, self.offset, self.cast_first)

    def settings(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}, offset={self.offset}"


class GatedRMSNorm(Converted):
    def __init__(self, original, eps: float):
        super().__init__(original)
        self.eps = eps

    def forward(self, x, gate=None):
        if gate is None:
            return self.original(x)
        from .norm import rms_norm_gated
        return rms_norm_gated(x, gate, self.weight, self.eps)


class GatedDeltaNet(Converted):
    _COPY = ("hidden_size", "num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim", "key_dim", "value_dim",
             "conv_kernel_size", "layer_idx", "activation", "layer_norm_epsilon", "conv_dim", "layer_type")

    def __init__(self, original, seq_chunk: int = 0, chunk: int = 0, precision: str = "bf16"):
        super().__init__(original)
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be 'bf16' or 'fp32'")
        self.precision = precision
        for k in self._COPY:
            if hasattr(original, k):
                setattr(self, k, getattr(original, k))
        self.seq_chunk, self.chunk = int(seq_chunk), int(chunk)
        if self.chunk < 0 or self.chunk % 64:
            raise ValueError("chunk must be 0 or a positive multiple of 64")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        from .sequence import sequence_chunked
        self._rule = sequence_chunked(chunk_gated_delta_rule, self.seq_chunk) if self.seq_chunk else \
            chunk_gated_delta_rule

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None or kwargs.get("cu_seq_lens_q") is not None:
            why = "a cache" if cache_params is not None else "packed variable-length sequences"
            return self._delegate(f"called with {why}", hidden_states, cache_params=cache_params,
                                  attention_mask=attention_mask, **kwargs)
        if attention_mask is not None and attention_mask.dim() == 2 and hidden_states.shape[1] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        T = hidden_states.shape[1]
        if not self.chunk or T <= self.chunk:
            return self._piece(hidden_states, None, None, last=True)[0]
        # the layer over pieces of `chunk` tokens: the convolution continues from the previous piece's last inputs,
        # the recurrence from its final state; with gradients each piece is recomputed in backward
        from .chunk import place
        ckpt = torch.is_grad_enabled() and self.training
        out, tail, state = None, None, None
        for s in range(0, T, self.chunk):
            x, last = hidden_states[:, s:s + self.chunk], s + self.chunk >= T
            args = (x, tail, state, last)
            o, tail, state = _checkpoint(self._piece, *args, use_reentrant=False) if ckpt else self._piece(*args)
            out = place(out, o, 1, s, T)
        return out

    def _piece(self, x, tail, state, last: bool):
        """(output, convolution inputs of the last tokens, final recurrent state) of tokens x [B, L, D] that follow
        the tokens whose convolution inputs are ``tail`` and recurrent state is ``state`` (None: sequence start)."""
        from .conv import causal_conv1d
        B, L, _ = x.shape
        mixed = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        ext = mixed if tail is None else torch.cat([tail, mixed], 1)
        conv = causal_conv1d(ext.transpose(1, 2), self.conv1d.weight.squeeze(1), self.conv1d.bias,
                             self.activation).transpose(1, 2)[:, ext.shape[1] - L:]
        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, L, -1, self.head_k_dim)
        k = k.reshape(B, L, -1, self.head_k_dim)
        v = v.reshape(B, L, -1, self.head_v_dim)
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
        beta = b.sigmoid()
        if self.precision == "fp32":           # the delta rule on float32 inputs, as transformers' own path
            q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        o, state = self._rule(q, k, v, g=g, beta=beta, initial_state=state, output_final_state=not last,
                              use_qk_l2norm_in_kernel=True)
        o = o.to(x.dtype)
        o = self.norm(o.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(B, L, -1)
        keep = self.conv1d.weight.shape[-1] - 1
        return self.out_proj(o), (None if last else ext[:, ext.shape[1] - keep:]), state

    def settings(self):
        return f"seq_chunk={self.seq_chunk}, chunk={self.chunk}, precision={self.precision}"


_PLAIN_KWARGS = {"position_ids", "cache_position", "use_cache", "output_attentions", "past_key_values",
                 "past_key_value"}


class AttentionChunks(Converted):
    def __init__(self, original, chunk: int):
        super().__init__(original)
        import sys
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise ValueError("chunk must be >= 1")
        self._rotary = sys.modules[type(original).__module__].apply_rotary_pos_emb
        self.n_heads = original.o_proj.in_features // original.head_dim
        self.gated = original.q_proj.out_features == 2 * original.o_proj.in_features

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        m = self.original
        unknown = sorted(set(kwargs) - _PLAIN_KWARGS)
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        mask_ok = attention_mask is None or (attention_mask.dim() == 4 and attention_mask.dtype == torch.bool)
        why = (f"arguments {unknown}" if unknown else "a cache" if cache is not None else
               "output_attentions" if kwargs.get("output_attentions") else
               "an attention mask that is not a boolean [B, 1|H, T, T] mask" if not mask_ok else None)
        if why is not None:
            return self._delegate(f"called with {why}", hidden_states, position_embeddings, attention_mask, **kwargs)
        B, T, _ = hidden_states.shape
        hd = m.head_dim
        cos, sin = position_embeddings
        k = self._norm("k_norm", m.k_proj(hidden_states).view(B, T, -1, hd)).transpose(1, 2)
        v = m.v_proj(hidden_states).view(B, T, -1, hd).transpose(1, 2)
        _, k = self._rotary(k[:, :1], k, cos, sin)
        from .chunk import place
        ckpt = torch.is_grad_enabled() and self.training
        out = None
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk)
            mask = None if attention_mask is None else attention_mask[:, :, s:e, :e]
            args = (hidden_states[:, s:e], k[:, :, :e], v[:, :, :e], cos[:, s:e], sin[:, s:e], mask)
            out = place(out, _checkpoint(self._piece, *args, use_reentrant=False) if ckpt else self._piece(*args),
                        1, s, T)
        return out, None

    def _norm(self, name: str, x):
        norm = getattr(self.original, name, None)
        return x if norm is None else norm(x)

    def _piece(self, x, k, v, cos, sin, mask):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.nn.attention.bias import causal_lower_right
        m = self.original
        B, L, _ = x.shape
        hd = m.head_dim
        qp = m.q_proj(x)
        if self.gated:
            q, gate = torch.chunk(qp.view(B, L, -1, 2 * hd), 2, dim=-1)
            gate = gate.reshape(B, L, -1)
        else:
            q, gate = qp.view(B, L, -1, hd), None
        q = self._norm("q_norm", q).transpose(1, 2)
        q, _ = self._rotary(q, q[:, :1], cos, sin)
        bias = causal_lower_right(L, k.shape[2]) if mask is None else mask
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                              SDPBackend.CUDNN_ATTENTION]):
                o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=m.scaling,
                                                                     enable_gqa=q.shape[1] != k.shape[1])
        except RuntimeError as e:
            raise RuntimeError(f"attention_chunks: no fused attention kernel for queries {tuple(q.shape)}, keys "
                               f"{tuple(k.shape)}, {'a padding mask' if mask is not None else 'causal'} on this "
                               f"device ({str(e).splitlines()[0][:120]})") from None
        o = o.transpose(1, 2).reshape(B, L, -1)
        if gate is not None:
            o = o * torch.sigmoid(gate)
        return m.o_proj(o)

    def settings(self):
        return f"chunk={self.chunk}"


def _make_attention(m, *, chunk):
    import sys
    if not hasattr(sys.modules[type(m).__module__], "apply_rotary_pos_emb"):
        raise Unfit("its module defines no apply_rotary_pos_emb")
    if getattr(m, "sliding_window", None):
        raise Unfit("sliding-window attention")
    if getattr(m, "attention_dropout", 0.0):
        raise Unfit("attention dropout")
    if not getattr(m, "is_causal", True):
        raise Unfit("not causal")
    heads = m.o_proj.in_features // m.head_dim
    if m.q_proj.out_features not in (heads * m.head_dim, 2 * heads * m.head_dim):
        raise Unfit("q_proj is neither the query nor query + gate projection")
    return AttentionChunks(m, chunk)


def _attention_probe(m, new):
    T = 2 * new.chunk + 7                                            # several pieces and a ragged last one
    x = torch.randn(1, T, m.q_proj.in_features, device=_device(m), dtype=_wdtype(m))
    angle = torch.rand(1, T, m.head_dim, device=_device(m)) * 6.3
    return x, (angle.cos().to(x.dtype), angle.sin().to(x.dtype)), None


class HostEmbedding(Converted):
    def __init__(self, original):
        super().__init__(original)
        w = self.weight
        self._device = w.device
        # the host copy is made now and checked; the parameter gives up its device storage once the conversion is
        # applied (installed), so a refused conversion changes nothing
        from ..hostmem import to_pinned
        self.__dict__["_host"] = to_pinned(w)

    def installed(self):
        self.weight.data = self.__dict__.pop("_host")

    @property
    def output_device(self) -> torch.device:
        """Where the looked-up rows go (the table's device before the conversion), whatever the indices' device."""
        return self._device

    def forward(self, ids):
        w = self.__dict__.get("_host", self.weight)
        idx = ids.reshape(-1).to("cpu")
        out = torch.empty(idx.numel(), w.shape[1], dtype=w.dtype, pin_memory=w.is_pinned())
        torch.index_select(w, 0, idx, out=out)
        return out.view(*ids.shape, w.shape[1]).to(self._device, non_blocking=True)

    def restored(self):
        self.weight.data = self.weight.data.to(self._device)
        return super().restored()

    def settings(self):
        return f"{tuple(self.weight.shape)}, pinned={self.weight.is_pinned()}"


class HostLinear(Converted):
    _TILE_BYTES = 64 << 20

    def __init__(self, original):
        super().__init__(original)
        w = self.weight
        self._device = w.device
        from ..hostmem import to_pinned
        self.__dict__["_host"] = to_pinned(w)

    def installed(self):
        self.weight.data = self.__dict__.pop("_host")

    def forward(self, x):
        from .stream import stream_rows
        w = self.__dict__.get("_host", self.weight)
        rows = max(128, self._TILE_BYTES // (w.shape[1] * w.element_size()) // 128 * 128)
        b = None if self.original.bias is None else self.original.bias.to(x.device)
        out = torch.empty(*x.shape[:-1], w.shape[0], dtype=x.dtype, device=x.device)
        for v0, v1, tile in stream_rows(w, rows, x.device):
            out[..., v0:v1] = torch.nn.functional.linear(x, tile.to(x.dtype), None if b is None else b[v0:v1])
        return out

    def restored(self):
        self.weight.data = self.weight.data.to(self._device)
        return super().restored()

    def settings(self):
        return f"{tuple(self.weight.shape)}, pinned={self.weight.is_pinned()}"


def _make_checkpoint(m, saved=None, min_numel=1 << 16, weights="gpu"):
    if weights == "ram":
        from ..tensor import LatticeTensor
        from .offload import STREAM_MIN_NUMEL
        on_gpu = [p for p in m.parameters() if p.is_cuda]
        big = [p for p in on_gpu if (max(v.numel() for v in p.parts().values() if v is not None)
                                     if isinstance(p, LatticeTensor) else p.numel()) >= STREAM_MIN_NUMEL]
        if on_gpu and not big:
            raise Unfit("no parameter large enough to stream")
    return Checkpointed(m, saved, min_numel, weights)


def _make_host_linear(m):
    w = m.weight
    if w.device.type == "meta":
        raise Unfit("parameters not materialized (meta)")
    if w.device.type != "cuda":
        raise Unfit("the weight is not on a GPU")
    if isinstance(w, LatticeTensor):
        raise Unfit("quantized weights are not streamed (dequantize it or keep it on the GPU)")
    if w.requires_grad or (m.bias is not None and m.bias.requires_grad):
        raise Unfit("the layer is trainable (freeze it first: a streamed weight gets no gradient)")
    return HostLinear(m)


def _make_host_embedding(m):
    w = m.weight
    if w.device.type == "meta":
        raise Unfit("parameters not materialized (meta)")
    if w.requires_grad:
        raise Unfit("the table is trainable (freeze it first: host lookups have no gradient)")
    if getattr(m, "max_norm", None) is not None:
        raise Unfit("max_norm renormalizes the table in place")
    return HostEmbedding(m)


class TokenChunks(Converted):
    def __init__(self, original, chunk: int = 2048, dim: int = -2, checkpoint: bool = True):
        super().__init__(original)
        self.chunk, self.dim, self.use_checkpoint = int(chunk), dim, checkpoint

    def forward(self, x, *args, **kwargs):
        if args or kwargs:
            return self._delegate("called with more than one input", x, *args, **kwargs)
        from .chunk import run_chunked
        ckpt = self.use_checkpoint and torch.is_grad_enabled() and self.training
        return run_chunked(self.original, x, self.chunk, self.dim, ckpt)

    def settings(self):
        return f"chunk={self.chunk}"


class Checkpointed(Converted):
    """Recomputes the module in backward. The first pass runs without a graph (nothing is saved), the recomputation
    with one; with ``saved`` its saved tensors are packed (the per-layer peak of long sequences). Tensor arguments,
    positional or keyword, receive gradients; tensors nested in other arguments (tuples of rotary tables, caches)
    are constants and must not require gradients."""

    def __init__(self, original, saved: str | None = None, min_numel: int = 1 << 16, weights: str = "gpu"):
        super().__init__(original)
        from ..memory import SavedSpec
        if weights not in ("gpu", "ram"):
            raise ValueError("weights must be 'gpu' or 'ram'")
        self.spec = SavedSpec(saved, min_numel) if saved else None
        self.weights = weights
        self.__dict__["residency"] = None

    def installed(self):
        if self.weights == "ram" and any(p.is_cuda for p in self.parameters()):   # modules of other processes: meta
            from .offload import Residency
            self.__dict__["residency"] = Residency(self)

    def restored(self):
        if self.residency is not None:
            self.residency.make_resident()
            self.__dict__["residency"] = None
        return super().restored()

    def _recompute(self, args, kwargs):
        if self.spec is not None:
            from ..memory import compress_saved
            with compress_saved(self.spec):
                return self.original(*args, **kwargs)
        return self.original(*args, **kwargs)

    def forward(self, *args, **kwargs):
        res = self.residency
        if not (torch.is_grad_enabled() and self.training):
            if res is not None:
                res.acquire()
                nxt = res.neighbour(+1)
                if nxt is not None:
                    nxt.prefetch()
            return self.original(*args, **kwargs)
        names = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]
        rest = {k: v for k, v in kwargs.items() if k not in names}
        nested = [k for k, v in rest.items() if any(t.requires_grad for t in _tensors(v))]
        if nested:
            raise ValueError(f"checkpoint: tensors inside argument(s) {nested} require gradients; the recomputation "
                             "can only pass gradients to tensor arguments")
        flat = (*args, *(kwargs[k] for k in names))
        if not any(isinstance(a, torch.Tensor) and a.requires_grad for a in flat):
            # nothing upstream needs a gradient (e.g. frozen embeddings), but the module's own parameters do: the
            # recomputation needs an input that is part of the graph
            i = next((k for k, a in enumerate(flat) if isinstance(a, torch.Tensor) and a.is_floating_point()), None)
            if i is not None:
                flat = (*flat[:i], flat[i].detach().requires_grad_(True), *flat[i + 1:])
        if res is not None:
            from .offload import StreamedCheckpoint
            return StreamedCheckpoint.apply(self, len(args), names, rest, *flat)
        return _checkpoint(functools.partial(self._run, len(args), names, rest), *flat, use_reentrant=True)

    def _run(self, n_args, names, rest, *flat):
        args, kwargs = flat[:n_args], {**rest, **dict(zip(names, flat[n_args:]))}
        if torch.is_grad_enabled():                                      # the recomputation in backward
            return self._recompute(args, kwargs)
        return self.original(*args, **kwargs)

    def settings(self):
        out = f"saved={self.spec.fmt if self.spec else 'as is'}"
        if self.residency is not None:
            out += f", weights in RAM ({self.residency.nbytes() / 2 ** 30:.2f} GiB streamed)"
        return out


class CompressedSaves(Converted):
    def __init__(self, original, fmt: str = "fp8", min_numel: int = 1 << 16):
        super().__init__(original)
        from ..memory import SavedSpec
        self.spec = SavedSpec(fmt, min_numel)

    def forward(self, *args, **kwargs):
        from ..memory import compress_saved
        with compress_saved(self.spec):
            return self.original(*args, **kwargs)

    def settings(self):
        return f"fmt={self.spec.fmt}"


# ---- conversions --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Conversion:
    """``applies(module)``: whether the module can be converted; ``make(module, **options)``: the new module (raises
    :class:`Unfit` with the reason when it finds that the module does not fit); ``probe(module, converted)``: inputs
    for the agreement check, sized to exercise the converted module's options (None for lossy conversions, which
    are not checked)."""

    describe: str
    applies: Callable[[torch.nn.Module], bool]
    make: Callable[..., torch.nn.Module]
    probe: Callable[[torch.nn.Module, torch.nn.Module], tuple] | None
    tol: float = 2e-2
    # the module's parameters must not be shared with other modules (e.g. tied heads): a bool, or a function of the
    # conversion's options
    exclusive: bool | Callable[[dict], bool] = False


class Unfit(ValueError):
    """Raised by a conversion's ``make``: the module cannot be converted; the message says why."""


def _eps(m) -> float | None:
    for name in ("variance_epsilon", "eps", "epsilon", "layer_norm_epsilon"):
        v = getattr(m, name, None)
        if isinstance(v, float):
            return v
    return None


def _wdtype(m) -> torch.dtype:
    p = next(m.parameters(), None)
    return p.dtype if p is not None and p.is_floating_point() else torch.bfloat16


def _device(m) -> torch.device:
    p = next(m.parameters(), None)
    return p.device if p is not None else torch.device("cuda")


def _make_rms(m):
    from .norm import _convention
    eps = _eps(m)
    if eps is None:
        raise Unfit("no eps / variance_epsilon attribute")
    if m.weight.device.type == "meta":
        raise Unfit("parameters not materialized (meta): the normalization convention cannot be checked")
    conv = _convention(m, eps)
    if conv is None:
        raise Unfit("not an RMSNorm convention this library implements")
    return RMSNorm(m, eps, conv[1], conv[2])


def _width(m) -> int | None:
    for name in ("hidden_size", "in_features", "embed_dim"):
        v = getattr(m, name, None)
        if isinstance(v, int):
            return v
    lin = next((c for c in m.modules() if isinstance(c, torch.nn.Linear) or hasattr(c, "in_features")), None)
    w = getattr(lin, "weight", None)
    return int(w.shape[-1]) if w is not None else None


_CONVERSIONS: dict[str, Conversion] = {
    "rms_norm": Conversion(
        "lean RMSNorm (input + one fp32 value per row kept for backward)",
        lambda m: type(m).__name__.endswith("RMSNorm") and getattr(getattr(m, "weight", None), "dim", lambda: 0)() == 1,
        _make_rms,
        lambda m, new: (torch.randn(4, m.weight.shape[0], device=_device(m), dtype=_wdtype(m)),)),
    "gated_rms_norm": Conversion(
        "lean normalize-then-gate (SiLU) RMSNorm",
        lambda m: type(m).__name__.endswith("RMSNormGated") and _eps(m) is not None,
        lambda m: GatedRMSNorm(m, _eps(m)),
        lambda m, new: (torch.randn(4, m.weight.shape[0], device=_device(m), dtype=_wdtype(m)),
                        torch.randn(4, m.weight.shape[0], device=_device(m), dtype=_wdtype(m)))),
    "gated_delta_net": Conversion(
        "GatedDeltaNet: fused convolution, q / k in their own heads; seq_chunk: rule over pieces; chunk: whole layer "
        "over pieces recomputed in backward",
        lambda m: all(hasattr(m, k) for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d",
                                              "A_log", "dt_bias", "norm", "out_proj")),
        lambda m, seq_chunk=0, chunk=0, precision="bf16": GatedDeltaNet(m, seq_chunk, chunk, precision),
        lambda m, new: (torch.randn(1, max(128, 2 * max(new.seq_chunk, new.chunk) + 64), m.hidden_size,
                                    device=_device(m), dtype=_wdtype(m)) * 0.5,)),
    "token_chunks": Conversion(
        "position-wise module over pieces of `chunk` tokens, each recomputed in backward",
        lambda m: _width(m) is not None,
        lambda m, *, chunk, dim=-2, checkpoint=True: TokenChunks(m, chunk, dim, checkpoint),
        lambda m, new: (torch.randn(2, 2 * new.chunk + 7, _width(m), device=_device(m), dtype=_wdtype(m)),)),
    "attention_chunks": Conversion(
        "causal self-attention over pieces of `chunk` queries (K / V projected once), each piece recomputed in "
        "backward; fused attention kernels only",
        lambda m: all(hasattr(m, k) for k in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim", "scaling")),
        _make_attention,
        _attention_probe),
    "host_embedding": Conversion(
        "frozen embedding table in pinned host memory; rows gathered on the CPU, only they are copied to the GPU",
        lambda m: isinstance(m, torch.nn.Embedding),
        _make_host_embedding,
        lambda m, new: (torch.randint(0, m.num_embeddings, (2, 33), device=new._device),),
        tol=0.0, exclusive=True),
    "host_linear": Conversion(
        "frozen linear layer with its weight in pinned host memory, streamed to the GPU in row tiles",
        lambda m: isinstance(m, torch.nn.Linear),
        _make_host_linear,
        lambda m, new: (torch.randn(2, 33, m.in_features, device=new._device, dtype=_wdtype(m)),),
        exclusive=True),
    "checkpoint": Conversion(
        "recompute in backward (inputs kept); `saved` packs the recomputed saved tensors ('fp8' / 'nvfp4'); "
        "`weights='ram'` keeps the module's weights in host memory, streamed to the GPU around its uses",
        lambda m: True,
        _make_checkpoint,
        None, exclusive=lambda options: options.get("weights") == "ram"),
    "compressed_saves": Conversion(
        "tensors saved for backward inside the module stored as `fmt` ('fp8' / 'nvfp4'; lossy, not checked)",
        lambda m: True,
        lambda m, fmt="fp8", min_numel=1 << 16: CompressedSaves(m, fmt, min_numel),
        None),
}


def _options(conv: Conversion) -> str:
    params = list(inspect.signature(conv.make).parameters.values())[1:]
    return ", ".join(str(p) for p in params) or "none"


def conversions() -> dict[str, str]:
    """Available conversions: what each does and its options."""
    return {k: f"{c.describe}; options: {_options(c)}" for k, c in _CONVERSIONS.items()}


def register_conversion(name: str, conversion: Conversion) -> None:
    """Make ``conversion`` available to :func:`convert_` under ``name``."""
    if name in _CONVERSIONS:
        raise ValueError(f"conversion {name!r} exists")
    _CONVERSIONS[name] = conversion


def _set(model: torch.nn.Module, name: str, new: torch.nn.Module) -> None:
    parent, _, attr = name.rpartition(".")
    owner = model.get_submodule(parent) if parent else model
    if isinstance(owner, torch.nn.ModuleList):
        owner[int(attr)] = new
    else:
        setattr(owner, attr, new)


def _agree(a: torch.Tensor, b: torch.Tensor, tol: float) -> bool:
    a, b = a.float(), b.float()
    return a.shape == b.shape and bool(torch.isfinite(a).all()) and ((a - b).norm() <= tol * b.norm() + 1e-6)


def convert_(model: torch.nn.Module, target: Selector, to: str, *, check: bool = True, skip_unfit: bool = False,
             **options) -> list[tuple[str, str, str]]:
    """Replace the modules selected by ``target`` by the conversion ``to`` (see :func:`conversions`) with
    ``options`` for it. Every selected module must qualify and, with ``check``, agree with its conversion on a
    random input; otherwise a ValueError names it and nothing is replaced -- or, with ``skip_unfit``, such modules
    stay as they are and are logged. Returns (name, old class, new class) per converted module."""
    if to not in _CONVERSIONS:
        raise ValueError(f"unknown conversion {to!r}; available: {sorted(_CONVERSIONS)}")
    conv = _CONVERSIONS[to]
    try:
        inspect.signature(conv.make).bind(None, **options)
    except TypeError as e:
        raise TypeError(f"conversion {to!r}: {e}; its options: {_options(conv)}") from None
    chosen = select(model, target, outermost=True)
    if not chosen:
        raise ValueError(f"no module matches {target!r}")
    plan = []
    for name, m in chosen:
        exclusive = conv.exclusive(options) if callable(conv.exclusive) else conv.exclusive
        shared = _shared_with(model, name, m) if exclusive else None
        why = f"its parameters are shared with {shared} (tied)" if shared else _refusal(conv, to, m, options, check)
        if isinstance(why, str):
            if not skip_unfit:
                raise ValueError(f"{name} ({type(m).__name__}): {why}")
            log.info("%s (%s) not converted to %s: %s", name, type(m).__name__, to, why)
            continue
        plan.append((name, m, why))
    for name, m, new in plan:
        _set(model, name, new)
        new.installed()
    return [(name, type(m).__name__, type(new).__name__) for name, m, new in plan]


def _shared_with(model: torch.nn.Module, name: str, m: torch.nn.Module) -> str | None:
    """Name of a module outside ``m`` that holds one of ``m``'s parameters, if any."""
    mine = {id(p) for p in m.parameters()}
    inside = name + "."
    for other_name, other in model.named_modules(remove_duplicate=False):
        if other_name == name or other_name.startswith(inside) or other is m:
            continue
        if any(id(p) in mine for p in other.parameters(recurse=False)):
            return other_name or "the model"
    return None


def _refusal(conv: Conversion, to: str, m: torch.nn.Module, options: dict, check: bool):
    """The converted module, or the reason why ``m`` cannot be converted."""
    if not conv.applies(m):
        return f"cannot be converted to {to!r}"
    try:
        new = conv.make(m, **options)
    except Unfit as e:
        return str(e)
    if check and conv.probe is not None and next(m.parameters(), torch.empty(0)).device.type != "meta":
        inputs = conv.probe(m, new)
        with torch.no_grad():
            ref = m(*inputs)
            try:
                out = new(*inputs)
            except RuntimeError as e:                    # e.g. no fused kernel for these shapes on this device
                return f"the {to!r} conversion fails on this module: {str(e).splitlines()[0][:200]}"
        ref = ref[0] if isinstance(ref, tuple) else ref
        out = out[0] if isinstance(out, tuple) else out
        if not _agree(out, ref, conv.tol):
            return f"the {to!r} conversion does not reproduce it"
    return new


def revert_(model: torch.nn.Module, target: Selector = "*") -> list[tuple[str, str, str]]:
    """Put back the original module of every converted module among the modules selected by ``target`` ("*": all;
    one conversion level, call again for stacked conversions). Returns (name, converted class, restored class)."""
    out = []
    for name, m in select(model, target):
        if isinstance(m, Converted):
            orig = m.restored()
            _set(model, name, orig)
            out.append((name, type(m).__name__, type(orig).__name__))
    return out
