"""quantize_ / dequantize_: convert the weights of a model in place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from .. import formats as F
from ..tensor import LatticeTensor
from .config import QuantConfig

__all__ = ["quantize_", "dequantize_", "calibrate", "QuantReport"]


@dataclass
class LayerReport:
    name: str
    format: str
    shape: tuple
    keep: int
    rel_error: float
    bits_per_weight: float


@dataclass
class QuantReport:
    layers: list[LayerReport] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def bits_per_weight(self) -> float:
        n = sum(torch.Size(l.shape).numel() for l in self.layers)
        return sum(l.bits_per_weight * torch.Size(l.shape).numel() for l in self.layers) / max(n, 1)

    def __str__(self) -> str:
        lines = [f"{len(self.layers)} quantized, {len(self.skipped)} kept dense, "
                 f"{self.bits_per_weight:.3f} bits/weight over quantized layers"]
        for l in self.layers:
            lines.append(f"  {l.name:60s} {l.format:10s} {str(l.shape):18s} keep={l.keep:<5d} "
                         f"err={l.rel_error:.4f} bits={l.bits_per_weight:.3f}")
        return "\n".join(lines)


@torch.no_grad()
def calibrate(model: torch.nn.Module, batches: Iterable, targets=(torch.nn.Linear,), max_batches: int = 32,
              forward=None) -> dict[str, torch.Tensor]:
    """Mean square of the inputs of every target module (per input feature) over ``batches``."""
    stats: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    def make(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            s = x.square().sum(0)
            stats[name] = stats[name] + s if name in stats else s
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, targets):
            hooks.append(mod.register_forward_hook(make(name)))
    try:
        for i, batch in enumerate(batches):
            if i >= max_batches:
                break
            if forward is not None:
                forward(model, batch)
            elif isinstance(batch, dict):
                model(**batch)
            elif isinstance(batch, (tuple, list)):
                model(*batch)
            else:
                model(batch)
    finally:
        for h in hooks:
            h.remove()
    return {k: v / counts[k] for k, v in stats.items()}


@torch.no_grad()
def quantize_(model: torch.nn.Module, config: QuantConfig | None = None, *, calibration: dict | None = None,
              report: bool = False, **overrides):
    """Replace the weights of target modules by LatticeTensor parameters in place.

    ``calibration`` -- output of :func:`calibrate`, required for ``saliency='activation'``.
    Keyword overrides are applied to ``config`` (e.g. ``quantize_(model, format='int4', keep=0.05)``).
    """
    from dataclasses import replace
    config = replace(config or QuantConfig(), **overrides) if overrides else (config or QuantConfig())
    rep = QuantReport()
    for name, mod in model.named_modules():
        if not isinstance(mod, config.targets):
            continue
        w = getattr(mod, "weight", None)
        if w is None or isinstance(w, LatticeTensor) or w.dim() < 2:
            continue
        cfg = config.for_module(name)
        cols = w.numel() // w.shape[0]
        if cfg is None or cols < cfg.min_features:
            rep.skipped.append(name)
            continue
        fmt = cfg.resolved_format()
        if fmt.kind != "native":
            fmt = F.fit_group(fmt, cols)
            if fmt is None:
                rep.skipped.append(name)
                continue
        if fmt.kind == "native":
            mod.weight = torch.nn.Parameter(w.detach().to(fmt.dtype), requires_grad=w.requires_grad)
            rep.skipped.append(name)
            continue
        sal = None
        if cfg.keep > 0 and cfg.saliency == "activation":
            if calibration is None or name not in calibration:
                raise ValueError(f"saliency='activation' needs calibration statistics for {name}")
            sal = calibration[name].to(w.device) * w.detach().float().reshape(w.shape[0], -1).square().sum(0)
        lt = LatticeTensor.from_float(w.detach(), fmt, method=cfg.method, keep=cfg.keep, keep_dtype=cfg.keep_dtype,
                                      saliency=sal, dtype=w.dtype)
        if report:
            ref = w.detach().float().reshape(w.shape[0], -1)
            err = ((lt.dequantize(torch.float32).reshape_as(ref) - ref).norm() / ref.norm().clamp_min(1e-30)).item()
            k = 0 if lt.hi_idx is None else lt.hi_idx.numel()
            bits = 8.0 * lt.storage_bytes() / w.numel()
            rep.layers.append(LayerReport(name, fmt.name, tuple(w.shape), k, err, bits))
        lt.compute = cfg.compute
        mod.weight = torch.nn.Parameter(lt, requires_grad=w.requires_grad)
    return rep if report else model


@torch.no_grad()
def dequantize_(model: torch.nn.Module, dtype: torch.dtype | None = None) -> torch.nn.Module:
    """Replace every LatticeTensor parameter by its dense value."""
    for mod in model.modules():
        for pname, p in list(mod.named_parameters(recurse=False)):
            if isinstance(p, LatticeTensor):
                setattr(mod, pname, torch.nn.Parameter(p.dequantize(dtype or p.dtype), requires_grad=p.requires_grad))
    return model
