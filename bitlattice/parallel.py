"""Layer-wise model parallelism: a model is split into contiguous runs of blocks, one run per GPU.

``infer_device_map`` assigns the repeated blocks (the largest ``nn.ModuleList``) and the modules around them to
devices so that the largest ratio of assigned bytes to the device budget is minimal; parameters shared between
modules (tied embeddings) stay on one device. ``dispatch`` moves every module to its device and installs pre-hooks
that move tensor inputs to the device of the module, so the forward pass, autograd and generation work unchanged.
The optimizer updates every parameter on its own device; there is no cross-device synchronization besides the
activations crossing the split points.
"""
from __future__ import annotations

import re
from typing import Callable, Mapping

import torch

from .tensor import LatticeTensor

__all__ = ["infer_device_map", "dispatch", "device_of", "block_list", "parse_bytes"]

_UNITS = {"b": 1, "kb": 10 ** 3, "mb": 10 ** 6, "gb": 10 ** 9, "kib": 2 ** 10, "mib": 2 ** 20, "gib": 2 ** 30}


def parse_bytes(v) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(v))
    if not m or m.group(2).lower() not in _UNITS | {"": 1}:
        raise ValueError(f"cannot parse memory size {v!r}")
    return int(float(m.group(1)) * _UNITS.get(m.group(2).lower(), 1))


def block_list(model: torch.nn.Module) -> str | None:
    """Name of the ModuleList holding the most parameters (the repeated layers), or None."""
    best, best_n = None, 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 1:
            n = sum(p.numel() for p in m.parameters())
            if n > best_n:
                best, best_n = name, n
    return best


def _param_bytes(p: torch.Tensor) -> int:
    if isinstance(p, LatticeTensor):
        return p.storage_bytes()
    return p.numel() * p.element_size()


def _units(model: torch.nn.Module) -> list[str]:
    """Placement units in execution order: every block of the block list, and every other module that directly
    owns parameters or buffers."""
    blocks = block_list(model)
    units, inside = [], (blocks + ".") if blocks else None
    for name, m in model.named_modules():
        if blocks is not None and name == blocks:
            units.extend(f"{name}.{i}" for i in range(len(m)))
            continue
        if inside and name.startswith(inside):
            continue
        if any(True for _ in m.parameters(recurse=False)) or any(True for _ in m.buffers(recurse=False)):
            units.append(name)
    return units


def device_of(name: str, device_map: Mapping[str, torch.device], default=None) -> torch.device:
    """Device of a parameter / module name: the entry with the longest matching prefix."""
    best, best_len = default, -1
    for key, dev in device_map.items():
        if (key == "" or name == key or name.startswith(key + ".")) and len(key) > best_len:
            best, best_len = dev, len(key)
    if best is None:
        raise KeyError(f"no device for {name!r}")
    return torch.device(best)


def _default_budgets(reserve: int) -> dict[torch.device, int]:
    out = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        out[torch.device("cuda", i)] = max(0, free - reserve)
    return out


def infer_device_map(model: torch.nn.Module, max_memory: Mapping | None = None, *,
                     param_bytes: Callable[[str, torch.Tensor], int] | None = None,
                     reserve: int | str = "2GiB") -> dict[str, torch.device]:
    """Contiguous split of ``model`` over devices.

    max_memory  -- {device: bytes or '12GiB'} budget for weights on each device (default: free memory of every
                   visible GPU minus ``reserve`` for activations, gradients and the optimizer workspace)
    param_bytes -- size estimate per parameter (default: its storage size); used when loading into a skeleton
    """
    reserve = parse_bytes(reserve)
    if max_memory is None:
        budgets = _default_budgets(reserve)
    else:
        budgets = {torch.device(f"cuda:{k}" if isinstance(k, int) else k): parse_bytes(v)
                   for k, v in max_memory.items()}
    devices = list(budgets)
    if not devices:
        raise RuntimeError("no devices to place the model on")
    size = param_bytes or (lambda n, p: _param_bytes(p))

    units = _units(model)
    blocks = block_list(model)
    owner: dict[int, str] = {}            # parameter id -> first unit that holds it (tied weights)
    cost, ties = {}, {}
    for u in units:
        m = model.get_submodule(u) if u else model
        recurse = blocks is not None and u.startswith(blocks + ".")
        c = 0
        for pn, p in m.named_parameters(recurse=recurse, remove_duplicate=False):
            full = f"{u}.{pn}" if u else pn
            if id(p) in owner:
                ties[u] = owner[id(p)]
                continue
            owner[id(p)] = u
            c += size(full, p)
        for b in m.buffers(recurse=recurse):
            c += b.numel() * b.element_size()
        cost[u] = c

    total = sum(cost.values())
    if len(devices) == 1:
        placement = {u: devices[0] for u in units}
    else:
        def fill(r: float):
            out, j, load = {}, 0, 0
            for u in units:
                c = cost[u] if u not in ties else 0
                while load + c > r * budgets[devices[j]]:
                    j, load = j + 1, 0
                    if j == len(devices):
                        return None
                out[u] = devices[j]
                load += c
            return out

        lo, hi = 0.0, max(1.0, 2.0 * total / max(1, min(budgets.values())))
        placement = fill(hi)
        if placement is None:
            raise RuntimeError("model does not fit the given devices")
        for _ in range(40):
            mid = (lo + hi) / 2
            p = fill(mid)
            if p is None:
                lo = mid
            else:
                hi, placement = mid, p
        if hi > 1.0 + 1e-6:
            raise RuntimeError(f"model needs {hi:.2f}x the memory budget of the devices")
    for u, src in ties.items():
        placement[u] = placement[src]
    return {u: placement[u] for u in units}


def _move(obj, device: torch.device):
    if torch.is_tensor(obj):
        if obj.device == device:
            return obj
        # asynchronous only towards a GPU: a non-blocking device-to-host copy is not complete on return
        return obj.to(device, non_blocking=device.type == "cuda")
    if isinstance(obj, tuple):
        return tuple(_move(o, device) for o in obj)
    if isinstance(obj, list):
        return [_move(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


def _place(model: torch.nn.Module, device_map: Mapping[str, torch.device]) -> None:
    default = next(iter(device_map.values()))
    done: dict[int, torch.Tensor] = {}
    for mod_name, mod in model.named_modules():
        for attr, p in list(mod._parameters.items()):
            if p is None:
                continue
            full = f"{mod_name}.{attr}" if mod_name else attr
            if id(p) in done:
                mod._parameters[attr] = done[id(p)]
                continue
            dev = device_of(full, device_map, default)
            new = p
            if p.device != dev:
                new = torch.nn.Parameter(p.detach().to(dev), requires_grad=p.requires_grad)
            done[id(p)] = new
            mod._parameters[attr] = new
        for attr, b in list(mod._buffers.items()):
            if b is None:
                continue
            full = f"{mod_name}.{attr}" if mod_name else attr
            dev = device_of(full, device_map, default)
            if b.device != dev:
                mod._buffers[attr] = b.to(dev)


def dispatch(model: torch.nn.Module, device_map: Mapping[str, torch.device | str | int] | None = None,
             **infer_kw) -> torch.nn.Module:
    """Place ``model`` according to ``device_map`` (default: ``infer_device_map(model, **infer_kw)``) and make
    every placed module move its tensor inputs to its device. Returns the model."""
    if device_map is None:
        device_map = infer_device_map(model, **infer_kw)
    device_map = {k: torch.device(f"cuda:{v}" if isinstance(v, int) else v) for k, v in device_map.items()}
    _place(model, device_map)
    for h in getattr(model, "_bl_dispatch_hooks", []):
        h.remove()
    # hook the outermost callable modules whose state lives on a single device: their inputs are moved there, and
    # everything inside runs on that device
    devs: dict[str, set] = {}
    for name, mod in model.named_modules():
        devs[name] = {t.device for t in list(mod.parameters()) + list(mod.buffers())}
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)) or len(devs[name]) != 1:
            continue
        parent = name.rpartition(".")[0] if name else None
        while parent is not None and isinstance(model.get_submodule(parent), (torch.nn.ModuleList,
                                                                              torch.nn.ModuleDict)):
            parent = parent.rpartition(".")[0] if parent else None
        if name and parent is not None and len(devs[parent]) == 1:
            continue                                                     # covered by an enclosing module
        targets[name] = (mod, next(iter(devs[name])))
    hooks = []
    for name, (mod, dev) in targets.items():

        def pre(module, args, kwargs, _dev=dev):
            return _move(args, _dev), _move(kwargs, _dev)

        hooks.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
    model._bl_dispatch_hooks = hooks
    model.bl_device_map = dict(device_map)
    return model
