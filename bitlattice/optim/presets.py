"""Presets and automatic parameter grouping."""
from __future__ import annotations

from typing import Any

import torch

from .lattice import Lattice

__all__ = ["PRESETS", "param_groups", "build"]

PRESETS: dict[str, dict[str, Any]] = {
    # orthogonalized (Newton-Schulz) direction, 2-bit phase: best quality on 4..16-bit formats
    "muon": dict(direction="muon_block", phase_bits=2),
    # same with an 8-bit phase: ternary / binary / 2-bit weights
    "muon_precise": dict(direction="muon_block", phase_bits=8),
    # factored (Adafactor-like, no momentum) direction with a 1 / 2 / 8-bit phase
    "light": dict(direction="factored", phase_bits=1),
    "balanced": dict(direction="factored", phase_bits=2, head_momentum="ternary"),
    "precise": dict(direction="factored", phase_bits=8, head_momentum="ternary"),
    # no optimizer state at all: memory equals inference; for formats of 8 bits and more
    "stateless": dict(direction="sinkhorn", vector_direction="sinkhorn", phase_bits=0, rounding="sr"),
}

_HEAD_NAMES = ("lm_head", "embed_out", "output", "classifier", "head")


def param_groups(model: torch.nn.Module, head_direction: str = "factored", head_momentum: str = "none",
                 no_decay_vectors: bool = True, weight_decay: float = 0.0) -> list[dict]:
    """Split trainable parameters: embeddings and output heads get ``head_direction`` (Newton-Schulz on a vocabulary
    matrix is prohibitively expensive), vectors get no weight decay, other matrices use the optimizer default."""
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    heads, matrices, vectors, seen = [], [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        module = name.rsplit(".", 1)[0] if "." in name else ""
        if id(p) in emb or module.rsplit(".", 1)[-1] in _HEAD_NAMES:
            heads.append(p)
        elif p.dim() >= 2:
            matrices.append(p)
        else:
            vectors.append(p)
    groups = []
    if matrices:
        groups.append({"params": matrices, "weight_decay": weight_decay})
    if heads:
        groups.append({"params": heads, "direction": head_direction, "momentum": head_momentum,
                       "weight_decay": weight_decay})
    if vectors:
        groups.append({"params": vectors, "weight_decay": 0.0 if no_decay_vectors else weight_decay})
    return groups


def build(model: torch.nn.Module, preset: str = "muon", lr: float = 3e-3, *, fused_backward: bool = False,
          seed: int = 0, weight_decay: float = 0.0, **overrides) -> Lattice:
    """Lattice optimizer for ``model`` from a preset; ``overrides`` replace preset options."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    opts = {**PRESETS[preset], **overrides}
    head_momentum = opts.pop("head_momentum", "none")
    groups = param_groups(model, head_direction="factored" if opts.get("direction", "").startswith("muon")
                          else opts.get("direction", "factored"), head_momentum=head_momentum,
                          weight_decay=weight_decay)
    return Lattice(groups, lr=lr, seed=seed, fused_backward=fused_backward, **opts)
