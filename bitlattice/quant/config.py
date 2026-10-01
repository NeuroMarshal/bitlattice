"""Quantization configuration: target format, precision budget and per-layer rules."""
from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from .. import formats as F

__all__ = ["QuantConfig", "Rule", "SKIP"]

SKIP = "skip"


@dataclass(frozen=True)
class Rule:
    """``pattern`` matches module names (glob by default, ``re:...`` for a regular expression).
    ``action`` is ``SKIP`` (keep the layer dense) or a dict of QuantConfig field overrides."""

    pattern: str
    action: Any = SKIP

    def matches(self, name: str) -> bool:
        if self.pattern.startswith("re:"):
            return re.search(self.pattern[3:], name) is not None
        return fnmatch.fnmatchcase(name, self.pattern)


@dataclass(frozen=True)
class QuantConfig:
    """Target format and budget.

    format       -- any format accepted by ``bitlattice.formats.get`` ('nvfp4', 'int4', 'fp6_e3m2', ...)
    method       -- 'rtn', 'mse' (per-group clipping search) or '4over6' (table formats: each group maps its maximum
                    to the top or the second-highest level, whichever has the lower error)
    keep         -- fraction in [0, 1] of input channels of every quantized layer kept in ``keep_dtype``
    saliency     -- 'weight' (column norm) or 'activation' (mean square input x column norm; needs calibration)
    group, scale -- override the scale group size / scale dtype of the format
    targets      -- module types that are quantized
    min_features -- layers with fewer input features stay dense
    compute      -- per-layer compute policy ('exact', 'fp4'); None follows bitlattice.set_compute
    rules        -- ordered list of Rule; the first match decides
    """

    format: Any = "nvfp4"
    method: str = "mse"
    keep: float = 0.0
    keep_dtype: torch.dtype = torch.bfloat16
    saliency: str = "weight"
    group: int | None = None
    scale: str | None = None
    targets: tuple[type, ...] = (torch.nn.Linear,)
    min_features: int = 32
    compute: str | None = None
    rules: tuple[Rule, ...] = ()

    def __post_init__(self):
        if not 0.0 <= self.keep <= 1.0:
            raise ValueError("keep must be in [0, 1]")
        if self.method not in ("rtn", "mse", "4over6"):
            raise ValueError("method must be 'rtn', 'mse' or '4over6'")
        if self.saliency not in ("weight", "activation"):
            raise ValueError("saliency must be 'weight' or 'activation'")
        object.__setattr__(self, "rules", tuple(r if isinstance(r, Rule) else Rule(*r) for r in self.rules))

    def resolved_format(self) -> F.Format:
        fmt = F.get(self.format)
        if fmt.kind != "native" and (self.group is not None or self.scale is not None):
            kw = {}
            if self.group is not None:
                kw["group"] = self.group
            if self.scale is not None:
                kw["dtype"] = self.scale
            fmt = fmt.with_scale(**kw)
        return fmt

    def for_module(self, name: str) -> "QuantConfig | None":
        """Effective config for a module name, or None if the layer stays dense."""
        for rule in self.rules:
            if rule.matches(name):
                if rule.action == SKIP or rule.action is None:
                    return None
                return replace(self, rules=(), **rule.action)
        return self

    def with_rules(self, *rules) -> "QuantConfig":
        return replace(self, rules=tuple(rules) + self.rules)
