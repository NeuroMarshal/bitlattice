"""Memory placement and accounting.

Optimizer state: ``Lattice(..., state='gpu' | 'ram' | 'auto')``. Activations: :class:`bitlattice.Memory` given to a
:class:`bitlattice.Runner` ('ram': layer inputs kept in host memory between forward and backward). :func:`plan`
reports where the memory goes before training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor import LatticeTensor

__all__ = ["activations_in_ram", "plan", "MemoryPlan", "choose_placement"]

AUTO_STATE_FRACTION = 0.25


def activations_in_ram(pin_memory: bool = True):
    """Context manager: CUDA tensors saved for backward are moved to host memory and brought back on use, with their
    layout (some kernels require the strides they were saved with, e.g. padded attention masks); broadcast
    dimensions (stride 0) are stored once and expanded again. A view whose layout spans much more memory than its
    elements (a slice of a larger tensor) is stored compactly instead."""

    def pack(t: torch.Tensor):
        if not t.is_cuda or type(t) is not torch.Tensor:
            return t
        broadcast = [d for d in range(t.dim()) if t.stride(d) == 0 and t.size(d) > 1]
        base = t
        for d in broadcast:
            base = base.narrow(d, 0, 1)
        extent = 1 + sum((n - 1) * s for n, s in zip(base.shape, base.stride()) if n > 0)
        if base.numel() and extent > base.numel() * 5 // 4 + 64:
            base = base.contiguous()
        pin = pin_memory and torch.cuda.is_available()
        host = torch.empty_strided(base.size(), base.stride(), dtype=t.dtype, device="cpu", pin_memory=pin)
        host.copy_(base, non_blocking=pin)
        return t.device, host, tuple(t.size()) if broadcast else None

    def unpack(p):
        if isinstance(p, torch.Tensor):
            return p
        device, host, size = p
        g = torch.empty_strided(host.size(), host.stride(), dtype=host.dtype, device=device)  # .to() would densify
        g.copy_(host, non_blocking=pin_memory)
        return g.expand(size) if size is not None else g

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _state_estimate(p: torch.Tensor, g: dict) -> int:
    """Bytes of per-parameter optimizer state: phase, momentum (in its stored format), factored second moments and
    the per-row statistics of the neuron-wise normalization."""
    n = p.numel()
    rows = p.shape[0] if p.dim() >= 2 else 1
    cols = n // max(rows, 1)
    b = (n * g.get("phase_bits", 0) + 7) // 8
    mom = g.get("momentum", "none")
    if mom == "nvfp4" and cols % 16:
        mom = "int8"
    b += {"none": 0, "fp32": 4 * n, "bf16": 2 * n, "ternary": n + 4 * rows,
          "int8": n + 4 * -(-n // 256), "fp8": n + 4 * -(-n // 256),
          "nvfp4": n // 2 + n // 16 + 4}[mom]
    matrix = p.dim() >= 2 and min(rows, cols) > 1
    if (g.get("direction") if matrix else g.get("vector_direction")) == "factored":
        b += 4 * (rows + cols)
    if matrix and g.get("normalize") == "row":
        b += 4 * rows
    return b


def choose_placement(groups) -> str:
    """'gpu' if the per-element optimizer state of every device fits in AUTO_STATE_FRACTION of its free memory,
    else 'ram'."""
    need: dict[torch.device, int] = {}
    for g in groups:
        for p in g["params"]:
            if p.is_cuda:
                need[p.device] = need.get(p.device, 0) + _state_estimate(p, g)
    for dev, b in need.items():
        free, _ = torch.cuda.mem_get_info(dev)
        if b > AUTO_STATE_FRACTION * free:
            return "ram"
    return "gpu"


@dataclass
class MemoryPlan:
    weights_bytes: int
    dense_weights_bytes: int
    state_gpu_bytes: int
    state_host_bytes: int
    grad_peak_bytes: int
    params: int
    weights_by_device: dict | None = None

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.weights_bytes / max(self.params, 1)

    def __str__(self) -> str:
        gib = 2 ** 30
        return (f"parameters {self.params / 1e6:.1f}M | weights {self.weights_bytes / gib:.3f} GiB "
                f"({self.bits_per_weight:.2f} bits/weight; dense bf16 would be {self.dense_weights_bytes / gib:.3f}) | "
                f"optimizer state GPU {self.state_gpu_bytes / gib:.3f} GiB, host {self.state_host_bytes / gib:.3f} GiB"
                f" | gradient peak {self.grad_peak_bytes / gib:.3f} GiB"
                + ("" if not self.weights_by_device or len(self.weights_by_device) < 2 else
                   " | weights by device " + ", ".join(f"{d} {b / gib:.2f} GiB"
                                                       for d, b in self.weights_by_device.items())))


def plan(model: torch.nn.Module, optimizer=None) -> MemoryPlan:
    """Static memory accounting. Optimizer state is estimated from the options before the first step."""
    weights = dense = n = 0
    largest = total_grad = 0
    seen = set()
    by_dev: dict[str, int] = {}
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n += p.numel()
        dense += p.numel() * 2
        wb = p.storage_bytes() if isinstance(p, LatticeTensor) else p.numel() * p.element_size()
        weights += wb
        by_dev[str(p.device)] = by_dev.get(str(p.device), 0) + wb
        if p.requires_grad:
            gbytes = p.numel() * (2 if isinstance(p, LatticeTensor) else p.element_size())
            largest = max(largest, gbytes + p.numel() * 4)      # gradient + float32 direction
            total_grad += gbytes
    sg = sh = 0
    if optimizer is not None:
        on_host = getattr(optimizer, "state_device", "cuda") == "cpu"
        if optimizer.state:
            for st in optimizer.state.values():
                for v in st.values():
                    if torch.is_tensor(v):
                        b = v.numel() * v.element_size()
                        if v.is_cuda:
                            sg += b
                        else:
                            sh += b
        else:
            for g in optimizer.param_groups:
                for p in g["params"]:
                    b = _state_estimate(p, g)
                    if on_host:
                        sh += b
                    else:
                        sg += b
    fused = getattr(optimizer, "fused_backward", False)
    return MemoryPlan(weights, dense, sg, sh, largest if fused else total_grad + largest, n, by_dev)
