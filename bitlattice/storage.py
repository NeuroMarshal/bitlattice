"""Storage policies and memory accounting.

Policies:
    'gpu'                    weights, optimizer state and saved activations on the GPU
    'state_in_ram'           per-element optimizer state in pinned host memory (``Lattice(state_device='cpu')``)
    'state_and_acts_in_ram'  additionally, tensors saved for backward are kept in pinned host memory
                             (``activations_in_ram()``); combine with gradient checkpointing so that only layer
                             inputs are saved
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .tensor import LatticeTensor

__all__ = ["POLICIES", "activations_in_ram", "maybe_activations_in_ram", "plan", "MemoryPlan", "choose_policy"]

POLICIES = ("gpu", "state_in_ram", "state_and_acts_in_ram")
AUTO_STATE_FRACTION = 0.25


def activations_in_ram(pin_memory: bool = True):
    """Context manager: tensors saved for backward are moved to host memory and brought back on use."""
    return torch.autograd.graph.save_on_cpu(pin_memory=pin_memory)


def maybe_activations_in_ram(policy: str):
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    return activations_in_ram() if policy == "state_and_acts_in_ram" else contextlib.nullcontext()


def _state_estimate(p: torch.Tensor, g: dict) -> int:
    b = (p.numel() * g.get("phase_bits", 0) + 7) // 8
    if g.get("momentum") == "fp32":
        b += p.numel() * 4
    elif g.get("momentum") == "ternary":
        b += p.numel()
    return b


def choose_policy(groups, options: dict) -> str:
    """'gpu' if the per-element optimizer state of every device fits in AUTO_STATE_FRACTION of its free memory,
    else 'state_in_ram'."""
    need: dict[torch.device, int] = {}
    for g in groups:
        g = {**options, **g}
        for p in g["params"]:
            if p.is_cuda:
                need[p.device] = need.get(p.device, 0) + _state_estimate(p, g)
    for dev, b in need.items():
        free, _ = torch.cuda.mem_get_info(dev)
        if b > AUTO_STATE_FRACTION * free:
            return "state_in_ram"
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
