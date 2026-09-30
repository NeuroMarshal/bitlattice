"""Presets and automatic parameter grouping."""
from __future__ import annotations

from typing import Any

import torch

from .lattice import Lattice

__all__ = ["PRESETS", "param_groups", "build"]

PRESETS: dict[str, dict[str, Any]] = {
    # orthogonalized (Newton-Schulz) direction, 2-bit phase: best quality on 4..16-bit formats
    "muon": dict(direction="muon_block", phase_bits=2),
    # bands of 256 rows over the full width, Gram Newton-Schulz in fp16 with the Polar Express schedule:
    # 7x cheaper orthogonalization on 8192x8192 (RTX 5070 Ti), held-out loss +0.9 % (within seed spread) in the
    # optimizer ablation; for large models, where the orthogonalization would dominate the step
    "muon_fast": dict(direction="muon_block", phase_bits=2, block=(256, None), ns_dtype="fp16",
                      ns_coeffs="polar_express"),
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
          seed: int = 0, weight_decay: float = 0.0, storage: str = "gpu", **overrides) -> Lattice:
    """Lattice optimizer for ``model`` from a preset; ``overrides`` replace preset options. ``storage`` selects
    where optimizer state lives ('gpu', 'state_in_ram', 'state_and_acts_in_ram', or 'auto': on the GPU when it
    takes at most ``bitlattice.storage.AUTO_STATE_FRACTION`` of the free memory of every device, else in host
    memory); for 'state_and_acts_in_ram' also run the forward pass inside ``bitlattice.storage.activations_in_ram()``.
    The chosen policy is ``optimizer.storage``."""
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    opts = {**PRESETS[preset], **overrides}
    head_momentum = opts.pop("head_momentum", "none")
    groups = param_groups(model, head_direction="factored" if opts.get("direction", "").startswith("muon")
                          else opts.get("direction", "factored"), head_momentum=head_momentum,
                          weight_decay=weight_decay)
    from ..storage import POLICIES, choose_policy
    if storage == "auto":
        storage = choose_policy(groups, opts)
    if storage not in POLICIES:
        raise ValueError(f"storage must be one of {POLICIES} or 'auto'")
    state_device = "cpu" if storage in ("state_in_ram", "state_and_acts_in_ram") else "cuda"
    opt = Lattice(groups, lr=lr, seed=seed, fused_backward=fused_backward, state_device=state_device, **opts)
    opt.storage = storage
    return opt
