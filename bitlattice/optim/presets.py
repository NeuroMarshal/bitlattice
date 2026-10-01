"""Optimizer presets (named option sets for :class:`bitlattice.optim.Lattice`) and the default parameter grouping of
a model."""
from __future__ import annotations

from typing import Any

import torch


__all__ = ["PRESETS", "param_groups"]

PRESETS: dict[str, dict[str, Any]] = {
    # orthogonalized (Newton-Schulz) direction, 2-bit phase: best quality on 4..16-bit formats
    "muon": dict(direction="muon_block", phase_bits=2),
    # bands of 256 rows over the full width, Gram Newton-Schulz in fp16 with the Polar Express schedule:
    # 7x cheaper orthogonalization on 8192x8192 (RTX 5070 Ti), held-out loss +0.9 % (within seed spread) in the
    # optimizer ablation; for large models, where the orthogonalization would dominate the step
    "muon_fast": dict(direction="muon_block", phase_bits=2, block=(256, None), ns_dtype="fp16",
                      ns_coeffs="polar_express"),
    # neuron-wise normalized Muon (NorMuon, arXiv 2510.05491): every row of the orthogonalized direction divided by the
    # root of an exponential average of its mean square (one fp32 value per row), evening out per-neuron step sizes
    "normuon": dict(direction="muon_block", phase_bits=2, normalize="row"),
    "normuon_fast": dict(direction="muon_block", phase_bits=2, block=(256, None), ns_dtype="fp16",
                         ns_coeffs="polar_express", normalize="row"),
    # same with an 8-bit phase: ternary / binary / 2-bit weights
    "muon_precise": dict(direction="muon_block", phase_bits=8),
    # factored (Adafactor-like, no momentum) direction with a 1 / 2 / 8-bit phase
    "light": dict(direction="factored", phase_bits=1),
    "balanced": dict(direction="factored", phase_bits=2),
    "precise": dict(direction="factored", phase_bits=8),
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
        if not p.requires_grad or id(p) in seen or p.device.type == "meta":      # meta: owned by another process
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
