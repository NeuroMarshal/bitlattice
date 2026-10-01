"""Linear-attention recurrences over sequence chunks.

Chunked kernels for gated delta rules and similar recurrences (fla's ``chunk_*`` functions) allocate temporaries that
grow with the sequence: the per-chunk states alone are heads × T/64 × K × V fp32 values (0.8 GB for 48 heads of
128 × 128 at 16k tokens), in forward and again in backward. :func:`sequence_chunked` runs such a function over
pieces of the sequence with the recurrent state carried from one piece to the next, so the temporaries scale with the
piece. The result is the same recurrence (pieces are multiples of the kernel's 64-token chunks); gradients flow
through the carried state. Variable-length batches (``cu_seqlens``) are passed through unchanged.

The ``gated_delta_net`` conversion takes the piece length as its ``seq_chunk`` option.
"""
from __future__ import annotations

import functools

import torch

__all__ = ["sequence_chunked"]

_SPLIT = ("g", "beta", "gk", "gv", "w", "u", "a", "b")


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
        from .chunk import place
        out, state = None, initial_state
        for s in range(0, T, chunk):
            sl = slice(s, min(T, s + chunk))
            piece = {**kw, **{n: t[:, sl] for n, t in per_token.items()}}
            o, state = fn(q[:, sl], k[:, sl], v[:, sl], initial_state=state, output_final_state=True, **piece)
            out = place(out, o, 1, s, T)
        return out, (state if output_final_state else None)

    wrapped._bl_sequence_chunk = chunk
    return wrapped
