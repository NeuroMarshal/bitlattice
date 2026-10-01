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
                      heads (no repeated copies), optional pieces of ``seq_chunk`` tokens for the delta rule; calls
                      with a cache (generation) go to the original module
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
                      gradient checkpointing for these modules)
    compressed_saves  tensors saved for backward inside the module stored as ``fmt`` ('fp8' or 'nvfp4'; lossy) --
                      for modules that are not recomputed: inside a checkpointed region the hooks would also act in
                      its first pass and keep what checkpointing discards; there use ``checkpoint`` with ``saved``
"""
from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint

from ..select import Selector, select

log = logging.getLogger("bitlattice")

__all__ = ["convert_", "revert_", "conversions", "register_conversion", "Conversion", "Converted", "Unfit",
           "AttentionChunks", "RMSNorm",
           "GatedRMSNorm", "GatedDeltaNet", "TokenChunks", "Checkpointed", "CompressedSaves"]


class Converted(torch.nn.Module):
    """Base of converted modules: adopts the original's parameters, buffers and children (same objects, same
    names) and remembers the original module for :func:`revert_`."""

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        self.__dict__["_bl_original"] = original                 # not a child: its parameters appear once
        for name, p in original.named_parameters(recurse=False):
            self.register_parameter(name, p)
        for name, b in original.named_buffers(recurse=False):
            self.register_buffer(name, b, persistent=name not in original._non_persistent_buffers_set)
        for name, child in original.named_children():
            self.add_module(name, child)
        self.training = original.training

    @property
    def original(self) -> torch.nn.Module:
        return self.__dict__["_bl_original"]

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
        """The original module with this module's current parameters, buffers and children."""
        orig = self.original
        for name, p in self.named_parameters(recurse=False):
            orig._parameters[name] = p
        for name, b in self.named_buffers(recurse=False):
            orig._buffers[name] = b
        for name, child in self.named_children():
            orig._modules[name] = child
        orig.train(self.training)
        return orig

    def settings(self) -> str:
        """The conversion's options, for printing."""
        return ""

    def extra_repr(self) -> str:
        return ", ".join(x for x in (self.settings(), f"from={type(self.original).__name__}") if x)


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

    def __init__(self, original, seq_chunk: int = 0):
        super().__init__(original)
        for k in self._COPY:
            if hasattr(original, k):
                setattr(self, k, getattr(original, k))
        self.seq_chunk = int(seq_chunk)
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from .sequence import sequence_chunked
        self._rule = sequence_chunked(chunk_gated_delta_rule, self.seq_chunk) if self.seq_chunk else \
            chunk_gated_delta_rule

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if cache_params is not None or kwargs.get("cu_seq_lens_q") is not None:
            return self.original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        from .conv import causal_conv1d
        if attention_mask is not None and attention_mask.dim() == 2 and hidden_states.shape[1] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        B, T, _ = hidden_states.shape
        mixed = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed = causal_conv1d(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias,
                              self.activation).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(B, T, -1, self.head_k_dim)
        k = k.reshape(B, T, -1, self.head_k_dim)
        v = v.reshape(B, T, -1, self.head_v_dim)
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)
        o, _ = self._rule(q, k, v, g=g, beta=b.sigmoid(), initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=True)
        o = self.norm(o.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)).reshape(B, T, -1)
        return self.out_proj(o)

    def settings(self):
        return f"seq_chunk={self.seq_chunk}"


_PLAIN_KWARGS = {"position_ids", "cache_position", "use_cache", "output_attentions", "past_key_values",
                 "past_key_value"}


class AttentionChunks(Converted):
    def __init__(self, original, chunk: int):
        super().__init__(original)
        import sys
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise ValueError("chunk must be >= 1")
        self._rotary = getattr(sys.modules[type(original).__module__], "apply_rotary_pos_emb")
        self.n_heads = original.o_proj.in_features // original.head_dim
        self.gated = original.q_proj.out_features == 2 * original.o_proj.in_features

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        m = self.original
        unknown = set(kwargs) - _PLAIN_KWARGS
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        mask_ok = attention_mask is None or (attention_mask.dim() == 4 and attention_mask.dtype == torch.bool)
        if unknown or cache is not None or kwargs.get("output_attentions") or not mask_ok:
            return m(hidden_states, position_embeddings, attention_mask, **kwargs)
        B, T, _ = hidden_states.shape
        hd = m.head_dim
        cos, sin = position_embeddings
        k = self._norm("k_norm", m.k_proj(hidden_states).view(B, T, -1, hd)).transpose(1, 2)
        v = m.v_proj(hidden_states).view(B, T, -1, hd).transpose(1, 2)
        _, k = self._rotary(k[:, :1], k, cos, sin)
        ckpt = torch.is_grad_enabled() and self.training
        outs = []
        for s in range(0, T, self.chunk):
            e = min(T, s + self.chunk)
            mask = None if attention_mask is None else attention_mask[:, :, s:e, :e]
            args = (hidden_states[:, s:e], k[:, :, :e], v[:, :, :e], cos[:, s:e], sin[:, s:e], mask)
            outs.append(_checkpoint(self._piece, *args, use_reentrant=False) if ckpt else self._piece(*args))
        return torch.cat(outs, 1), None

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
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=m.scaling,
                                                                 enable_gqa=q.shape[1] != k.shape[1])
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


class TokenChunks(Converted):
    def __init__(self, original, chunk: int = 2048, dim: int = -2, checkpoint: bool = True):
        super().__init__(original)
        self.chunk, self.dim, self.use_checkpoint = int(chunk), dim, checkpoint

    def forward(self, x, *args, **kwargs):
        if args or kwargs:
            return self.original(x, *args, **kwargs)
        from .chunk import run_chunked
        ckpt = self.use_checkpoint and torch.is_grad_enabled() and self.training
        return run_chunked(self.original, x, self.chunk, self.dim, ckpt)

    def settings(self):
        return f"chunk={self.chunk}"


class Checkpointed(Converted):
    """Recomputes the module in backward. The first pass runs without a graph (nothing is saved), the recomputation
    with one; with ``saved`` its saved tensors are packed (the per-layer peak of long sequences). Positional
    arguments that require gradients receive them; keyword arguments are passed through as constants."""

    def __init__(self, original, saved: str | None = None, min_numel: int = 1 << 16):
        super().__init__(original)
        from ..memory import SavedSpec
        self.spec = SavedSpec(saved, min_numel) if saved else None

    def forward(self, *args, **kwargs):
        if not (torch.is_grad_enabled() and self.training):
            return self.original(*args, **kwargs)
        if not any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
            # nothing upstream needs a gradient (e.g. frozen embeddings), but the module's own parameters do: the
            # recomputation needs an input that is part of the graph
            i = next((k for k, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()), None)
            if i is not None:
                args = (*args[:i], args[i].detach().requires_grad_(True), *args[i + 1:])
        return _checkpoint(functools.partial(self._run, kwargs), *args, use_reentrant=True)

    def _run(self, kwargs, *args):
        if self.spec is not None and torch.is_grad_enabled():            # the recomputation in backward
            from ..memory import compress_saved
            with compress_saved(self.spec):
                return self.original(*args, **kwargs)
        return self.original(*args, **kwargs)

    def settings(self):
        return f"saved={self.spec.fmt if self.spec else 'as is'}"


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
        "GatedDeltaNet: fused convolution, q / k in their own heads, optional seq_chunk pieces",
        lambda m: all(hasattr(m, k) for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d",
                                              "A_log", "dt_bias", "norm", "out_proj")),
        lambda m, seq_chunk=0: GatedDeltaNet(m, seq_chunk),
        lambda m, new: (torch.randn(1, max(128, 2 * new.seq_chunk + 64), m.hidden_size, device=_device(m),
                                    dtype=_wdtype(m)) * 0.5,)),
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
    "checkpoint": Conversion(
        "recompute in backward (inputs kept); `saved` packs the recomputed saved tensors ('fp8' / 'nvfp4')",
        lambda m: True,
        lambda m, saved=None, min_numel=1 << 16: Checkpointed(m, saved, min_numel),
        None),
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
        why = _refusal(conv, to, m, options, check)
        if isinstance(why, str):
            if not skip_unfit:
                raise ValueError(f"{name} ({type(m).__name__}): {why}")
            log.info("%s (%s) not converted to %s: %s", name, type(m).__name__, to, why)
            continue
        plan.append((name, m, why))
    for name, m, new in plan:
        _set(model, name, new)
    return [(name, type(m).__name__, type(new).__name__) for name, m, new in plan]


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
