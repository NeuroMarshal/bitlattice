"""Per-layer quantization presets.

Names follow the Hugging Face conventions (``*.layers.N.*``, ``lm_head``, ``embed_tokens``); every preset is a
plain QuantConfig and can be extended with further rules.
"""
from __future__ import annotations

import re

import torch

from .config import QuantConfig, Rule

__all__ = ["default", "accuracy_critical", "aggressive", "skip_blocks", "layer_indices"]

_HEAD_EMBED = (Rule("*lm_head*"), Rule("*embed_tokens*"), Rule("*embed_positions*"), Rule("*wte*"), Rule("*wpe*"))


def default(fmt="nvfp4", **kw) -> QuantConfig:
    """All linear layers except the output head and embeddings."""
    return QuantConfig(format=fmt, rules=_HEAD_EMBED, **kw)


def aggressive(fmt="nvfp4", **kw) -> QuantConfig:
    """Every linear layer including the output head."""
    return QuantConfig(format=fmt, **kw)


def layer_indices(model: torch.nn.Module) -> list[int]:
    idx = set()
    for name, _ in model.named_modules():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name + ".")
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def skip_blocks(model: torch.nn.Module, first: int = 1, last: int = 1) -> tuple[Rule, ...]:
    """Rules keeping the first ``first`` and last ``last`` transformer blocks dense."""
    idx = layer_indices(model)
    keep = idx[:first] + (idx[-last:] if last else [])
    return tuple(Rule(f"re:(?:^|\\.)layers\\.{i}\\.") for i in sorted(set(keep)))


def accuracy_critical(model: torch.nn.Module, fmt="nvfp4", keep: float = 0.1, **kw) -> QuantConfig:
    """For tasks where precision dominates (speech recognition, retrieval, small models): dense head, embeddings,
    first and last blocks and cross-attention; 8-bit attention outputs; ``keep`` of the input channels in BF16."""
    rules = (skip_blocks(model, 1, 1) + _HEAD_EMBED +
             (Rule("*encoder_attn*"), Rule("*proj_out*"), Rule("*conv*")) +
             (Rule("*o_proj", {"format": "int8", "keep": 0.0}), Rule("*out_proj", {"format": "int8", "keep": 0.0})))
    return QuantConfig(format=fmt, keep=keep, method="mse", rules=rules, **kw)
