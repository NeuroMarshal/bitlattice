"""Token-chunked execution of position-wise modules (MLPs, MoE feed-forward blocks, output norms); the
``token_chunks`` conversion applies it to modules of a model.

A position-wise module maps every token independently, so it can run on slices of the sequence. With
checkpointed pieces each slice is recomputed in the backward pass, and the tensors saved for backward shrink from
the full [tokens, intermediate] activations to the [tokens, hidden] input: for an MLP with a 3.4x intermediate
width and gated activation this removes ~90% of its activation memory, at the cost of recomputing it once.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

__all__ = ["run_chunked", "place"]


def run_chunked(fn, x: torch.Tensor, chunk: int, dim: int = -2, checkpoint_pieces: bool = True) -> torch.Tensor:
    """fn(x) computed over pieces of ``chunk`` positions along ``dim`` (each recomputed in backward with
    ``checkpoint_pieces``); for position-wise fn the result equals fn(x)."""
    n = x.shape[dim]
    if n <= chunk and not checkpoint_pieces:
        return fn(x)
    out = None
    for s in range(0, n, chunk):
        xs = x.narrow(dim, s, min(chunk, n - s))
        y = checkpoint(fn, xs, use_reentrant=False) if checkpoint_pieces else fn(xs)
        out = place(out, y, dim, s, n)
    return out


def place(out, y: torch.Tensor, dim: int, start: int, total: int) -> torch.Tensor:
    """Write piece ``y`` at ``start`` along ``dim`` of the output (allocated on the first piece): each piece is freed
    once copied, instead of all pieces and their concatenation existing at once."""
    if out is None:
        shape = list(y.shape)
        shape[dim] = total
        out = torch.empty(shape, dtype=y.dtype, device=y.device)
    out.narrow(dim, start, y.shape[dim]).copy_(y)
    return out
