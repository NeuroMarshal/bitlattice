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

__all__ = ["infer_device_map", "interleaved_device_map", "dispatch", "device_of", "block_list", "parse_bytes",
           "to_device", "pipelined_causal_lm_loss"]

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


def _peer(a: torch.device, b: torch.device) -> bool:
    key = (a.index, b.index)
    if key not in _PEER:
        _PEER[key] = torch.cuda.can_device_access_peer(a.index, b.index)
    return _PEER[key]


_PEER: dict = {}


_COPY_STREAMS: dict = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    s = _COPY_STREAMS.get(device)
    if s is None:
        s = _COPY_STREAMS[device] = torch.cuda.Stream(device=device)
    return s


def _staged_copy(x: torch.Tensor, dst: torch.device) -> torch.Tensor:
    """GPU-to-GPU copy that never blocks the host or the compute streams. Without peer access a plain copy is
    staged through pageable memory synchronously. Here a copy stream on the source waits for the work issued so far
    and writes a pinned buffer, a copy stream on the destination waits for that and reads it, and the destination
    compute stream waits only for the second copy: both GPUs keep computing while the tensor is in flight."""
    src = x.device
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(src))
    cs, cd = _copy_stream(src), _copy_stream(dst)
    cs.wait_event(ready)
    host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    with torch.cuda.stream(cs):
        host.copy_(x, non_blocking=True)
        x.record_stream(cs)
        sent = torch.cuda.Event()
        sent.record(cs)
    cd.wait_event(sent)
    with torch.cuda.stream(cd):
        out = host.to(dst, non_blocking=True)
        done = torch.cuda.Event()
        done.record(cd)
    compute = torch.cuda.current_stream(dst)
    compute.wait_event(done)
    out.record_stream(compute)
    return out


def _copy(x: torch.Tensor, dst: torch.device) -> torch.Tensor:
    if x.device.type == "cuda" and dst.type == "cuda" and not _peer(x.device, dst):
        return _staged_copy(x, dst)
    # asynchronous only towards a GPU: a non-blocking device-to-host copy is not complete on return
    return x.to(dst, non_blocking=dst.type == "cuda")


class _ToDevice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dst):
        ctx.src = x.device
        return _copy(x, dst)

    @staticmethod
    def backward(ctx, g):
        return _copy(g, ctx.src), None


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Differentiable device move that does not block the host (also for GPUs without peer access)."""
    device = torch.device(device)
    if x.device == device:
        return x
    if x.requires_grad and torch.is_grad_enabled():
        return _ToDevice.apply(x, device)
    return _copy(x, device)


def _move(obj, device: torch.device):
    if torch.is_tensor(obj):
        return to_device(obj, device)
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


# --------------------------------------------------------------------------------------------------------------
# interleaved placement + layer-major micro-batch schedule

def _compute_weight(name: str, p: torch.Tensor, embeddings: set) -> int:
    """Relative compute of a parameter per token: matrix-vector work ~ numel, embeddings (lookups) ~ 0."""
    return 0 if id(p) in embeddings or p.dim() < 2 else p.numel()


def interleaved_device_map(model: torch.nn.Module, devices=None, chunk: int = 1,
                           tail_cost: float = 1.0) -> dict[str, torch.device]:
    """Place the repeated blocks round-robin in runs of ``chunk`` over ``devices`` (default: all GPUs), with the
    per-device compute balanced: the output head counts as compute, embeddings do not, and the head goes to the last
    device, which receives correspondingly fewer blocks. Modules before the blocks go to the first device, modules
    after them to the last. Compute is estimated by parameter count; ``tail_cost`` scales the estimate of the modules
    after the blocks (the head, extra heads) relative to a block parameter -- e.g. ~3 for a BF16 head next to FP4
    blocks. Use with :func:`pipelined_causal_lm_loss`."""
    devices = [torch.device(f"cuda:{d}" if isinstance(d, int) else d)
               for d in (devices if devices is not None else range(torch.cuda.device_count()))]
    blocks = block_list(model)
    if blocks is None:
        raise ValueError("model has no repeated block list")
    layers = model.get_submodule(blocks)
    emb = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
    per_layer = [sum(_compute_weight(n, p, emb) for n, p in l.named_parameters()) for l in layers]
    units = _units(model)
    first = units.index(f"{blocks}.0")
    before, after = units[:first], units[first + len(layers):]
    tail = 0
    seen = {id(p) for l in layers for p in l.parameters()}
    for u in after:
        for p in model.get_submodule(u).parameters(recurse=False):
            if id(p) not in seen:
                tail += _compute_weight(u, p, set()) * tail_cost      # a tied head still costs compute
    D, L = len(devices), len(layers)
    total = sum(per_layer) + tail
    quota = [total / D] * D
    quota[-1] -= tail
    # round-robin runs of `chunk` blocks; a device whose compute quota is used up is skipped
    load, out, d = [0.0] * D, {}, 0
    i = 0
    while i < L:
        tries = 0
        while load[d] + per_layer[i] / 2 > quota[d] and tries < D:
            d, tries = (d + 1) % D, tries + 1
        for _ in range(chunk):
            if i >= L:
                break
            out[f"{blocks}.{i}"] = devices[d]
            load[d] += per_layer[i]
            i += 1
        d = (d + 1) % D
    for u in before:
        out[u] = devices[0]
    for u in after:
        out[u] = devices[-1]
    return {u: out[u] for u in units}


class _Stop(Exception):
    pass


def _block_inputs(backbone: torch.nn.Module, blocks, input_ids: torch.Tensor | None, kwargs: dict,
                  inputs_embeds: torch.Tensor | None = None):
    """(hidden entering the first block, [kwargs of every block]) exactly as the backbone passes them: embedding,
    per-block masks (hybrid models mix attention types), positions and rotary tables are prepared by the model
    itself. The blocks are replaced by pass-throughs while recording, so nothing but the preparation runs."""
    box, saved = {"kwargs": []}, []

    def recorder(i):
        def fwd(*args, **kw):
            h = args[0] if args else kw.pop("hidden_states")
            if i == 0:
                box["hidden"] = h
            kw.pop("hidden_states", None)
            box["kwargs"].append(dict(kw))
            if i == len(blocks) - 1:
                raise _Stop                     # skip the final norm: it may live on another process
            return h
        return fwd

    for i, b in enumerate(blocks):
        saved.append(b.__dict__.get("forward"))
        b.forward = recorder(i)
    try:
        if inputs_embeds is not None:
            backbone(inputs_embeds=inputs_embeds, use_cache=False, **kwargs)
        else:
            backbone(input_ids=input_ids, use_cache=False, **kwargs)
    except _Stop:
        pass
    finally:
        for b, f in zip(blocks, saved):
            if f is None:
                del b.forward
            else:
                b.forward = f
    if len(box["kwargs"]) != len(blocks):
        raise RuntimeError("the backbone did not call every block once")
    return box.get("hidden"), box["kwargs"]


_NORM_NAMES = ("norm", "final_layernorm", "ln_f", "final_norm", "norm_f")


def pipelined_causal_lm_loss(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                             micro: int = 4, chunk: int = 1024, **kwargs) -> torch.Tensor:
    """Next-token loss of a Hugging Face decoder-only LM, computed block-major over ``micro`` micro-batches.

    Every block processes all micro-batches before the next block starts, so with blocks interleaved over GPUs
    (:func:`interleaved_device_map`) one GPU works on block i for micro-batch m while the other works on block i+1
    for micro-batch m-1, forward and backward. Because the backward also runs block-major, a block's gradient is
    complete after its last micro-batch and the fused optimizer update applies it at once: one block of gradients
    in memory, no stale weights, the same result as the full batch."""
    from .nn.loss import linear_cross_entropy
    labels = input_ids if labels is None else labels
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    blocks = backbone.layers
    norm = next((getattr(backbone, n) for n in _NORM_NAMES if hasattr(backbone, n)), None)
    head = model.get_output_embeddings()
    ids_mb, lab_mb = input_ids.chunk(micro, 0), labels.chunk(micro, 0)
    states, per_dev = [], []
    for ids in ids_mb:
        h, kws = _block_inputs(backbone, blocks, ids, kwargs)
        states.append(h)
        per_dev.append(kws)
    devs = [next(b.parameters()).device for b in blocks]
    tail_dev = head.weight.device
    for i, block in enumerate(blocks):
        dev = devs[i]
        nxt = devs[i + 1] if i + 1 < len(blocks) else tail_dev
        for m in range(len(states)):
            kw = per_dev[m][i]
            kw = per_dev[m][i] = _move(kw, dev)
            out = block(states[m], **kw)
            out = out[0] if isinstance(out, tuple) else out
            # send each micro-batch on as soon as it is computed, so the next device starts on it while this one
            # works on the following micro-batch
            states[m] = to_device(out, nxt)
    n_tok = [int(lb[:, 1:].numel()) for lb in lab_mb]
    loss = 0.0
    for h, lb, n in zip(states, lab_mb, n_tok):
        if norm is not None:
            h = norm(h)
        loss = loss + linear_cross_entropy(h[:, :-1], head.weight, lb[:, 1:], getattr(head, "bias", None),
                                           chunk) * (n / sum(n_tok))
    return loss


from .pipeline import init, is_main, train_step  # noqa: E402,F401  (re-exported: one process per GPU; see bitlattice.pipeline)
