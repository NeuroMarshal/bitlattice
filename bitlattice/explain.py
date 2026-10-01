"""What the library did to a model: one line per relevant module."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .nn.convert import Converted
from .tensor import LatticeTensor

__all__ = ["explain", "Explanation"]


@dataclass
class Row:
    name: str
    kind: str          # 'converted' | 'lattice'
    detail: str
    trainable: bool
    bytes: int


class Explanation(list):
    """Rows (name, kind, detail, trainable, bytes); printing gives a table. ``kernels``: the automatic kernel choices
    in effect and which fast paths this machine has."""

    kernels: str = ""

    def __str__(self) -> str:
        head = [self.kernels] if self.kernels else []
        if not self:
            return "\n".join(head + ["no converted modules and no lattice weights"])
        w = max(len(r.name) for r in self)
        lines = head + [f"{'module / weight':{w}s}  {'what':10s} {'trains':6s} {'MiB':>9s}  details"]
        for r in self:
            lines.append(f"{r.name:{w}s}  {r.kind:10s} {'yes' if r.trainable else 'no':6s} "
                         f"{r.bytes / 2 ** 20:9.2f}  {r.detail}")
        return "\n".join(lines)


def explain(model: torch.nn.Module) -> Explanation:
    """Converted modules (what they replaced) and lattice weights (format, how they compute) of ``model``."""
    from .compute_policy import _scope
    from .kernel_choice import active, kernels_available
    out = Explanation()
    dev = next((p.device.index for p in model.parameters() if p.is_cuda), None)
    if dev is not None:
        have = kernels_available(dev)
        missing = [k for k, v in have.items() if not v]
        out.kernels = (f"kernels: {active().describe()}"
                       + (f"; not available here: {', '.join(missing)}" if missing else "; all fast paths available"))
    for name, mod in model.named_modules():
        if isinstance(mod, Converted):
            nbytes = sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False))
            out.append(Row(name, "converted", f"{type(mod).__name__} from {type(mod.original).__name__}"
                           + (f" ({mod.settings()})" if mod.settings() else ""),
                           any(p.requires_grad for p in mod.parameters()), nbytes))
        for pname, p in mod.named_parameters(recurse=False):
            if isinstance(p, LatticeTensor):
                c = _scope[-1] if _scope else p.compute
                how = "exact (default)" if c is None else c.describe() + (" (temporary)" if _scope else "")
                keep = "" if p.hi_idx is None else f", {p.hi_idx.numel()} columns kept"
                where = "; in host memory, streamed" if not p.packed.is_cuda and p.is_cuda else ""
                out.append(Row(".".join(x for x in (name, pname) if x), "lattice",
                               f"{p.fmt.name} {tuple(p.shape)}{keep}; computes {how}{where}", p.requires_grad,
                               p.storage_bytes()))
    return out
