"""Linear-attention recurrences over sequence chunks.

Chunked kernels for gated delta rules and similar recurrences (fla's ``chunk_*`` functions) allocate temporaries that
grow with the sequence: the per-chunk states alone are heads × T/64 × K × V fp32 values (0.8 GB for 48 heads of
128 × 128 at 16k tokens), in forward and again in backward. :func:`sequence_chunked` runs such a function over
pieces of the sequence with the recurrent state carried from one piece to the next, so the temporaries scale with the
piece. The result is the same recurrence (pieces are multiples of the kernel's 64-token chunks); gradients flow
through the carried state. Variable-length batches (``cu_seqlens``) are passed through unchanged.

:func:`sequence_chunked_` switches it in for the modules of a model that call such a function by module-level name
(``torch_chunk_gated_delta_rule`` / ``chunk_gated_delta_rule`` in transformers' Qwen3.5 / Qwen3-Next).
"""
from __future__ import annotations

import functools
import sys

import torch

__all__ = ["sequence_chunked", "sequence_chunked_"]

_SPLIT = ("g", "beta", "gk", "gv", "w", "u", "a", "b")
NAMES = ("torch_chunk_gated_delta_rule", "chunk_gated_delta_rule")


def sequence_chunked(fn, chunk: int = 4096):
    """Wrap ``fn(q, k, v, *, initial_state=None, output_final_state=False, **kw) -> (o, final_state)`` (time on
    dimension 1 of q / k / v and of the per-token keyword tensors) to run over pieces of ``chunk`` tokens."""
    if chunk <= 0 or chunk % 64:
        raise ValueError("chunk must be a positive multiple of 64")
    if getattr(fn, "_bl_sequence_chunk", None):
        fn = fn.__wrapped__

    @functools.wraps(fn)
    def wrapped(q, k, v, *args, initial_state=None, output_final_state=False, **kw):
        T = q.shape[1]
        if T <= chunk or kw.get("cu_seqlens") is not None or args:
            return fn(q, k, v, *args, initial_state=initial_state, output_final_state=output_final_state, **kw)
        per_token = {n: t for n, t in kw.items() if n in _SPLIT and isinstance(t, torch.Tensor) and t.dim() >= 2
                     and t.shape[1] == T}
        outs, state = [], initial_state
        for s in range(0, T, chunk):
            sl = slice(s, min(T, s + chunk))
            piece = {**kw, **{n: t[:, sl] for n, t in per_token.items()}}
            o, state = fn(q[:, sl], k[:, sl], v[:, sl], initial_state=state, output_final_state=True, **piece)
            outs.append(o)
        return torch.cat(outs, 1), (state if output_final_state else None)

    wrapped._bl_sequence_chunk = chunk
    return wrapped


def sequence_chunked_(model: torch.nn.Module, chunk: int = 4096, names=NAMES) -> list[str]:
    """Wrap the module-level recurrence functions ``names`` used by ``model``'s layers; returns 'module.name' of the
    wrapped ones."""
    done = []
    for mod in {type(m).__module__ for m in model.modules()}:
        module = sys.modules.get(mod)
        for name in names:
            fn = getattr(module, name, None)
            if callable(fn):
                setattr(module, name, sequence_chunked(fn, chunk))
                done.append(f"{mod}.{name}")
    return done
