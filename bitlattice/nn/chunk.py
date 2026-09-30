"""Token-chunked execution of position-wise modules (MLPs, MoE feed-forward blocks, output norms).

A position-wise module maps every token independently, so it can run on slices of the sequence. With
``checkpoint=True`` each slice is recomputed in the backward pass, and the tensors saved for backward shrink from
the full [tokens, intermediate] activations to the [tokens, hidden] input: for an MLP with a 3.4x intermediate
width and gated activation this removes ~90% of its activation memory, at the cost of recomputing it once.
"""
from __future__ import annotations

import fnmatch

import torch
from torch.utils.checkpoint import checkpoint

__all__ = ["TokenChunked", "token_chunked_"]


def _run_chunked(fn, x: torch.Tensor, chunk: int, dim: int, use_ckpt: bool) -> torch.Tensor:
    n = x.shape[dim]
    if n <= chunk and not use_ckpt:
        return fn(x)
    outs = []
    for s in range(0, n, chunk):
        xs = x.narrow(dim, s, min(chunk, n - s))
        outs.append(checkpoint(fn, xs, use_reentrant=False) if use_ckpt else fn(xs))
    return torch.cat(outs, dim)


class TokenChunked(torch.nn.Module):
    """Wraps a position-wise module; ``forward(x)`` runs it on slices of ``chunk`` positions along ``dim``."""

    def __init__(self, module: torch.nn.Module, chunk: int = 2048, dim: int = -2, checkpoint: bool = True):
        super().__init__()
        self.module, self.chunk, self.dim, self.use_checkpoint = module, int(chunk), dim, checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ckpt = self.use_checkpoint and torch.is_grad_enabled() and self.training
        return _run_chunked(self.module, x, self.chunk, self.dim, ckpt)


def token_chunked_(model: torch.nn.Module, pattern: str = "*mlp", chunk: int = 2048, dim: int = -2,
                   checkpoint: bool = True) -> list[str]:
    """Run every module whose name matches ``pattern`` (glob, or ``re:`` regex) token-chunked, in place.

    The module object is kept (parameter names and the optimizer are unaffected); its ``forward`` is replaced.
    Only single-tensor-input position-wise modules qualify. Returns the names of the wrapped modules."""
    import re
    match = (lambda n: re.search(pattern[3:], n) is not None) if pattern.startswith("re:") else \
        (lambda n: fnmatch.fnmatchcase(n, pattern))
    done = []
    for name, mod in model.named_modules():
        if not name or not match(name) or getattr(mod, "_bl_chunked", False):
            continue
        inner = mod.forward

        def forward(x, *args, _inner=inner, _mod=mod, **kwargs):
            if args or kwargs:
                return _inner(x, *args, **kwargs)
            ckpt = _mod._bl_checkpoint and torch.is_grad_enabled() and _mod.training
            return _run_chunked(_inner, x, _mod._bl_chunk, dim, ckpt)

        mod.forward = forward
        mod._bl_chunked, mod._bl_chunk, mod._bl_checkpoint = True, int(chunk), checkpoint
        done.append(name)
    return done
