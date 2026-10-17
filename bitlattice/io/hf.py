"""Loading and saving Hugging Face style checkpoints.

``load`` builds the model skeleton without allocating parameters, then streams tensors from safetensors shards
one module at a time: pre-quantized layouts are converted losslessly, dense weights are quantized on the target
device according to a QuantConfig. Peak memory is the final model plus one dense tensor.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any

import torch

from .. import formats as F
from ..quant.config import QuantConfig
from ..tensor import LatticeTensor
from . import convert as C

__all__ = ["detect_layout", "export_compressed_tensors", "load", "save"]

_PART_SUFFIXES = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale", "weight_scale_2",
                  "input_scale", "weight_scale_inv", "weight_zero_point", "weight_shape", "qweight", "qzeros",
                  "scales", "g_idx")


# checkpoint discovery

def _resolve(path_or_id: str, revision: str | None) -> str:
    if os.path.isdir(path_or_id):
        return path_or_id
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py"])


def _shards(path: str) -> list[str]:
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            return sorted({os.path.join(path, v) for v in json.load(f)["weight_map"].values()})
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {path}")
    return files


def detect_layout(path: str, config) -> dict[str, Any]:
    """{'kind': 'dense'|'ct_nvfp4'|'mo_nvfp4'|'fp8'|'fp8_block'|'ct_int4'|'gptq'|'awq'|'bitlattice', ...}"""
    if os.path.exists(os.path.join(path, "bitlattice.json")):
        return {"kind": "bitlattice"}
    mo = os.path.join(path, "hf_quant_config.json")
    if os.path.exists(mo):
        with open(mo) as f:
            q = json.load(f).get("quantization", {})
        if str(q.get("quant_algo", "")).upper() == "NVFP4":
            return {"kind": "mo_nvfp4", "ignore": q.get("exclude_modules", [])}
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return {"kind": "dense"}
    qc = qc if isinstance(qc, dict) else qc.to_dict()
    method = str(qc.get("quant_method", "")).lower()
    if method == "compressed-tensors":
        groups = qc.get("config_groups") or {}
        group_formats = {g.get("format") for g in groups.values() if isinstance(g, dict) and g.get("format")}
        fmt = str(qc.get("format") or "")
        if fmt in ("", "mixed-precision") and len(group_formats) == 1:
            fmt = group_formats.pop()
        ignore = qc.get("ignore", [])
        if "nvfp4" in fmt:
            return {"kind": "ct_nvfp4", "ignore": ignore}
        if fmt == "float-quantized":
            return {"kind": "fp8", "ignore": ignore}
        if fmt == "pack-quantized":
            groups = qc.get("config_groups", {})
            w = next(iter(groups.values()), {}).get("weights", {}) if groups else {}
            return {"kind": "ct_int4", "group": int(w.get("group_size") or 128), "ignore": ignore}
        raise NotImplementedError(f"compressed-tensors format {fmt!r}")
    if method == "modelopt" and "NVFP4" in json.dumps(qc).upper():
        return {"kind": "mo_nvfp4", "ignore": qc.get("exclude_modules", qc.get("ignore", []))}
    if method == "fp8":
        blk = qc.get("weight_block_size")
        return {"kind": "fp8_block", "block": tuple(blk)} if blk else {"kind": "fp8"}
    if method == "gptq":
        return {"kind": "gptq", "group": int(qc.get("group_size", 128)), "bits": int(qc.get("bits", 4)),
                "v1": qc.get("checkpoint_format", "gptq") != "gptq_v2"}
    if method == "awq":
        return {"kind": "awq", "group": int(qc.get("group_size", 128)), "bits": int(qc.get("bits", 4))}
    raise NotImplementedError(f"quantization method {method!r}")


# model skeleton

def _model_class(config, model_cls):
    if model_cls is not None:
        return model_cls
    import transformers
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    return transformers.AutoModelForCausalLM


def _skeleton(config, model_cls, dtype, trust_remote_code):
    from accelerate import init_empty_weights
    cls = _model_class(config, model_cls)
    with init_empty_weights(include_buffers=False):
        if hasattr(cls, "from_config"):
            model = cls.from_config(config, torch_dtype=dtype, trust_remote_code=trust_remote_code)
        else:
            model = cls._from_config(config, torch_dtype=dtype)
    return model


def _rename(model, name: str) -> str | None:
    """Checkpoint key -> parameter name: user key map (``load(key_map=...)``; a None replacement drops the key),
    then the model's own conversion mapping."""
    for pat, rep in (getattr(model, "_bl_key_map", None) or {}).items():
        if re.search(pat, name):
            if rep is None:
                return None
            name = re.sub(pat, rep, name)
            break
    mapping = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    for pat, rep in mapping.items():
        new, n = re.subn(pat, rep, name)
        if n:
            return new
    return name


def _set(model: torch.nn.Module, name: str, value: torch.Tensor, is_param: bool) -> None:
    mod_name, _, attr = name.rpartition(".")
    mod = model.get_submodule(mod_name) if mod_name else model
    if is_param:
        old = getattr(mod, attr, None)
        rg = old.requires_grad if isinstance(old, torch.nn.Parameter) else True
        setattr(mod, attr, torch.nn.Parameter(value, requires_grad=rg and value.is_floating_point()))
    else:
        mod.register_buffer(attr, value, persistent=True)


def _ignored(name: str, patterns) -> bool:
    for p in patterns or []:
        if p.startswith("re:") and re.match(p[3:], name):
            return True
        if name == p or name.startswith(p + ".") or name.endswith("." + p) or p in name.split("."):
            return True
    return False


# loading

_DT_BYTES = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
             "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BOOL": 1, "F4": 0.5}


def _estimate_sizes(path: str, model, layout: dict, quant: QuantConfig | None) -> dict[str, int]:
    """Bytes of every parameter after loading, from the safetensors headers (no tensor data is read)."""
    from safetensors import safe_open
    modules = dict(model.named_modules())
    params = dict(model.named_parameters(remove_duplicate=False))
    sizes: dict[str, int] = defaultdict(int)
    for shard in _shards(path):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                shape, dt = sl.get_shape(), sl.get_dtype()
                n = 1
                for d in shape:
                    n *= d
                if ".__" in k:                                     # bitlattice checkpoint part
                    sizes[k.split(".__", 1)[0]] += n * _DT_BYTES.get(dt, 2)
                    continue
                renamed = _rename(model, k)
                if renamed is None:
                    continue
                prefix, _, suffix = renamed.rpartition(".")
                name = f"{prefix}.weight" if suffix in _PART_SUFFIXES and prefix in modules else \
                    (f"{prefix}.{suffix}" if prefix else suffix)
                b = n * _DT_BYTES.get(dt, 2)
                p = params.get(name)
                if suffix == "weight" and p is not None and quant is not None and dt in ("F32", "F16", "BF16"):
                    b = _quantized_bytes(name, p, modules.get(prefix), quant, b)
                elif dt == "F32" and p is not None and p.dtype != torch.float32 and p.is_floating_point():
                    b = n * p.element_size()
                sizes[name] += int(b)
    return dict(sizes)


def _quantized_bytes(name: str, p, module, quant: QuantConfig, dense_bytes: int) -> int:
    if not isinstance(module, quant.targets) or p.dim() < 2:
        return dense_bytes
    cfg = quant.for_module(name.rpartition(".")[0])
    rows, cols = p.shape[0], p.numel() // p.shape[0]
    if cfg is None or cols < cfg.min_features:
        return dense_bytes
    fmt = cfg.resolved_format()
    if fmt.kind == "native":
        return p.numel() * fmt.dtype.itemsize
    fmt = F.fit_group(fmt, cols)
    if fmt is None:
        return dense_bytes
    keep = int(round(cfg.keep * cols))
    return fmt.storage_bits(rows, cols - keep) // 8 + keep * rows * cfg.keep_dtype.itemsize


def _device_plan(path, model, layout, quant, device, device_map, max_memory):
    """(tensor name -> device, device map or None) for the weights being loaded."""
    if device_map is None:
        dev = torch.device(device)
        return (lambda name: dev), None
    from .. import parallel
    if isinstance(device_map, str) and device_map.startswith("interleaved"):
        # "interleaved[:chunk][:tail=<cost>]"
        parts = device_map.split(":")[1:]
        chunk, opts = 1, {}
        for x in parts:
            if "=" in x:
                k, v = x.split("=", 1)
                if k != "tail":
                    raise ValueError(f"unknown interleaved option {k!r} (known: tail)")
                opts["tail_cost"] = float(v)
            elif x:
                chunk = int(x)
        devs = list(max_memory) if max_memory else None
        device_map = parallel.interleaved_device_map(model, devs, chunk, **opts)
    if isinstance(device_map, str):
        if device_map not in ("auto", "balanced"):
            raise ValueError("device_map must be 'auto', 'balanced', 'interleaved[:chunk]' or a dict")
        sizes = _estimate_sizes(path, model, layout, quant)
        device_map = parallel.infer_device_map(
            model, max_memory, param_bytes=lambda n, p: sizes.get(n, p.numel() * p.element_size()))
    device_map = {k: torch.device(f"cuda:{v}" if isinstance(v, int) else v) for k, v in device_map.items()}
    first = next(iter(device_map.values()))
    return (lambda name: parallel.device_of(name, device_map, first)), device_map


@torch.no_grad()
def load(path_or_id: str, quant: QuantConfig | None = None, *, device: str | torch.device = "cuda",
         device_map: str | dict | None = None, max_memory: dict | None = None, dtype: torch.dtype = torch.bfloat16,
         model_cls=None, trust_remote_code: bool = False, revision: str | None = None,
         requantize: bool = False, config=None, key_map: dict | None = None) -> torch.nn.Module:
    r"""Load a model (local directory or hub id) directly into lattice formats.

    quant       -- QuantConfig for dense weights (None: keep dense in ``dtype``). Pre-quantized checkpoints keep
                   their stored format unless ``requantize`` (dequantize then apply ``quant``).
    device_map  -- None (everything on ``device``), 'auto' / 'balanced' (contiguous split over the GPUs, sized from
                   the checkpoint headers and ``max_memory``), 'interleaved[:chunk][:tail=cost]' (blocks round-robin over the GPUs
                   for ``bitlattice.parallel.pipelined_causal_lm_loss``) or {module name: device}
    model_cls   -- transformers class to instantiate (default: from config.architectures).
    config      -- model config to use instead of the checkpoint's (e.g. the text part of a multimodal model).
    key_map     -- {regex: replacement or None} applied to checkpoint keys; None drops the tensor (e.g.
                   ``{r"^model\.visual\.": None, r"^model\.language_model\.": "model."}``).
    """
    from safetensors import safe_open
    from transformers import AutoConfig

    path = _resolve(path_or_id, revision)
    full_config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    layout = detect_layout(path, full_config)
    config = config or full_config
    if layout["kind"] == "bitlattice":
        return _load_bitlattice(path, config, model_cls, dtype, device, trust_remote_code, device_map, max_memory)
    model = _skeleton(config, model_cls, dtype, trust_remote_code)
    model._bl_key_map = dict(key_map or {})
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())
    modules = dict(model.named_modules())
    device, dmap = _device_plan(path, model, layout, quant, device, device_map, max_memory)
    local = _local_device(dmap)
    mine = (lambda name: True) if local is None else (lambda name: device(name) == local)

    for shard in _shards(path):
        with safe_open(shard, framework="pt", device="cpu") as f:
            by_prefix: dict[str, dict[str, str]] = defaultdict(dict)
            for k in f.keys():
                renamed = _rename(model, k)
                if renamed is None:
                    continue
                prefix, _, suffix = renamed.rpartition(".")
                by_prefix[prefix][suffix] = k
            for prefix, sfx in by_prefix.items():
                packed = _packed_parts(f, sfx, layout["kind"]) if prefix in modules else None
                if packed and mine(f"{prefix}.weight"):
                    _load_packed(model, f, prefix, {s: sfx[s] for s in packed}, layout, quant, requantize, device,
                                 dtype)
                rest = {s: k for s, k in sfx.items() if not packed or s not in packed}
                for suffix, k in rest.items():
                    if suffix in _PART_SUFFIXES and prefix in modules and packed:
                        continue
                    target = f"{prefix}.{suffix}" if prefix else suffix
                    if target in params and not mine(target):
                        continue
                    _load_dense(model, f, target, k, params, buffers, modules, quant, layout, device, dtype)
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta" and mine(n)]
    if missing and getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
        missing = [n for n, p in model.named_parameters() if p.device.type == "meta" and mine(n)]
    if missing:
        raise RuntimeError(f"weights missing from the checkpoint: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    _finish(model, device, dmap, local)
    return model


def _local_device(dmap):
    """In a multi-process run (torch.distributed initialized) with a device map, each process loads only the
    parameters placed on its own GPU; the rest stay on the meta device."""
    import torch.distributed as dist
    if dmap is None or not (dist.is_available() and dist.is_initialized()):
        return None
    return torch.device("cuda", torch.cuda.current_device())


def _finish(model, device, dmap, local=None):
    for name, b in list(model.named_buffers()):
        target = local if local is not None else device(name)            # small buffers live on every process
        if b.device != target:
            mod_name, _, attr = name.rpartition(".")
            mod = model.get_submodule(mod_name) if mod_name else model
            mod._buffers[attr] = b.to(target)
    if dmap is not None:
        from .. import parallel
        if local is None:
            parallel.dispatch(model, dmap)
        else:
            model.bl_device_map = dict(dmap)
            model.bl_local_device = local
    model.eval()


def _quantize_dense(name: str, w: torch.Tensor, module, quant: QuantConfig | None, dtype):
    if quant is None or not isinstance(module, quant.targets) or w.dim() < 2:
        return None
    cfg = quant.for_module(name.rpartition(".")[0])
    cols = w.numel() // w.shape[0]
    if cfg is None or cols < cfg.min_features:
        return None
    fmt = cfg.resolved_format()
    if fmt.kind == "native":
        return w.to(fmt.dtype)
    fmt = F.fit_group(fmt, cols)
    if fmt is None:
        return None
    lt = LatticeTensor.from_float(w, fmt, method=cfg.method, keep=cfg.keep, keep_dtype=cfg.keep_dtype, dtype=dtype)
    return lt


_QUANT_DTYPES = ("U8", "I8", "I32", "F8_E4M3", "F8_E5M2")


def _packed_parts(f, sfx: dict[str, str], kind: str) -> tuple[str, ...] | None:
    """Suffixes of one module that form a pre-quantized weight, or None."""
    def dt(s):
        return f.get_slice(sfx[s]).get_dtype()
    if kind in ("ct_nvfp4", "ct_int4") and "weight_packed" in sfx:
        return tuple(s for s in ("weight_packed", "weight_scale", "weight_global_scale", "weight_zero_point",
                                 "weight_shape") if s in sfx)
    if kind == "mo_nvfp4" and "weight" in sfx and "weight_scale" in sfx and dt("weight") in _QUANT_DTYPES:
        return tuple(s for s in ("weight", "weight_scale", "weight_scale_2") if s in sfx)
    if kind == "fp8" and "weight" in sfx and dt("weight").startswith("F8"):
        return tuple(s for s in ("weight", "weight_scale") if s in sfx)
    if kind == "fp8_block" and "weight" in sfx and "weight_scale_inv" in sfx:
        return ("weight", "weight_scale_inv")
    if kind in ("gptq", "awq") and "qweight" in sfx:
        return tuple(s for s in ("qweight", "qzeros", "scales", "g_idx") if s in sfx)
    return None


def _load_dense(model, f, target, key, params, buffers, modules, quant, layout, device, dtype):
    t = f.get_tensor(key)
    if target in params:
        mod = modules.get(target.rpartition(".")[0])
        if target.endswith(".weight") and t.dtype in (torch.uint8, torch.int32, torch.float8_e4m3fn):
            raise RuntimeError(f"packed tensor {key} without the scales of layout {layout['kind']!r}")
        w = t.to(device=device(target), dtype=dtype if t.is_floating_point() else t.dtype)
        q = _quantize_dense(target, w, mod, quant, dtype) if target.endswith(".weight") else None
        _set(model, target, q if q is not None else w, True)
    elif target in buffers:
        _set(model, target, t.to(device(target)), False)


def _load_packed(model, f, prefix, parts, layout, quant, requantize, device, dtype):
    tensors = {suffix: f.get_tensor(k) for suffix, k in parts.items()}
    kind = layout["kind"]
    device = device(f"{prefix}.weight")
    if kind == "ct_nvfp4":
        lt = C.from_ct_nvfp4(tensors, dtype, device)
    elif kind == "mo_nvfp4":
        lt = C.from_mo_nvfp4(tensors, dtype, device)
    elif kind == "fp8":
        lt = C.from_fp8(tensors, dtype, device)
    elif kind == "fp8_block":
        lt = C.from_fp8_block(tensors, dtype, device, layout["block"])
    elif kind == "ct_int4":
        lt = C.from_ct_int4(tensors, layout["group"], dtype, device)
    elif kind == "gptq":
        try:
            lt = C.from_gptq(tensors, layout["group"], dtype, device, layout["v1"])
        except NotImplementedError:
            lt = _gptq_dense(tensors, layout, dtype, device)
    elif kind == "awq":
        lt = C.from_awq(tensors, layout["group"], dtype, device)
    else:
        raise NotImplementedError(kind)
    if requantize and quant is not None:
        dense = lt.dequantize(dtype) if isinstance(lt, LatticeTensor) else lt
        q = _quantize_dense(f"{prefix}.weight", dense, model.get_submodule(prefix), quant, dtype)
        lt = q if q is not None else dense
    _set(model, f"{prefix}.weight", lt, True)


def _gptq_dense(t, layout, dtype, device):
    """Act-order GPTQ: dequantize with the per-row group index."""
    qw = t["qweight"].to(device)
    u = C.unpack_int32_nibbles(qw, dim=0).float()                          # [in, out]
    out = qw.shape[1]
    z = C.unpack_int32_nibbles(t["qzeros"].to(device), dim=1)[:, :out].float()
    if layout["v1"]:
        z = (z + 1) % 16
    g = t["g_idx"].to(device).long()
    w = (u - z[g]) * t["scales"].to(device).float()[g]
    return w.t().contiguous().to(dtype)


# bitlattice checkpoints

def _format_spec(fmt: F.Format) -> dict:
    d = {"kind": fmt.kind, "name": fmt.name, "bits": fmt.bits}
    if fmt.kind == "table":
        d["values"] = list(fmt.values)
        d["init"] = fmt.init
    if fmt.kind == "uniform":
        d.update(qmin=fmt.qmin, qmax=fmt.qmax, zero_point=fmt.zero_point, init=fmt.init)
    if fmt.scale is not None:
        d["scale"] = dict(group=fmt.scale.group, dtype=fmt.scale.dtype, global_scale=fmt.scale.global_scale,
                          strict=fmt.scale.strict)
    return d


def _format_from_spec(d: dict) -> F.Format:
    sc = F.ScaleSpec(**d["scale"])
    if d["kind"] == "table":
        return F.TableFormat(d["name"], d["bits"], tuple(float(v) for v in d["values"]), sc, d.get("init", "absmax"))
    return F.UniformFormat(d["name"], d["bits"], d["qmin"], d["qmax"], sc, d["zero_point"], d.get("init", "absmax"))


def save(model: torch.nn.Module, path: str, max_shard_bytes: int = 4 << 30) -> None:
    """Save weights (lattice parts included) as safetensors + bitlattice.json + the model config.

    In a multi-process run (``torch.distributed`` initialized) every process calls it: each writes the parameters
    it holds (meta parameters are another process's), and process 0 merges the indexes into one checkpoint that
    loads like a single-process one."""
    import torch.distributed as dist
    from safetensors.torch import save_file
    multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if multi else 0
    tag = f"r{rank}-" if multi else ""
    if rank == 0:
        os.makedirs(path, exist_ok=True)
    if multi:
        dist.barrier()
    meta: dict[str, Any] = {"version": 1, "tensors": {}}
    flat: dict[str, torch.Tensor] = {}
    seen = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            meta.setdefault("tied", {})[name] = seen[id(p)]
            continue
        seen[id(p)] = name
        if p.device.type == "meta":
            continue
        if isinstance(p, LatticeTensor):
            meta["tensors"][name] = {"format": _format_spec(p.fmt), "shape": list(p.shape),
                                     "dtype": str(p.dtype).replace("torch.", "")}
            for k, v in p.parts().items():
                if v is not None:
                    flat[f"{name}.__{k}"] = v.detach().contiguous().cpu()
        else:
            flat[name] = p.detach().contiguous().cpu()
    if rank == 0:
        for name, b in model.named_buffers():
            if b.device.type != "meta":
                flat.setdefault(name, b.detach().contiguous().cpu())
    shards, cur, size = [], {}, 0
    for k, v in flat.items():
        n = v.numel() * v.element_size()
        if cur and size + n > max_shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += n
    shards.append(cur)
    weight_map = {}
    for i, sh in enumerate(shards):
        fname = f"model-{tag}{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(path, fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in sh})
    part = (os.path.join(path, f"bitlattice.{tag}json"), os.path.join(path, f"index.{tag}json"))
    if multi:
        with open(part[0], "w") as f:
            json.dump(meta, f)
        with open(part[1], "w") as f:
            json.dump(weight_map, f)
        dist.barrier()
        if rank != 0:
            return
        meta, weight_map = {"version": 1, "tensors": {}}, {}
        for r in range(dist.get_world_size()):
            with open(os.path.join(path, f"bitlattice.r{r}-json")) as f:
                m = json.load(f)
            meta["tensors"].update(m["tensors"])
            meta.setdefault("tied", {}).update(m.get("tied", {}))
            with open(os.path.join(path, f"index.r{r}-json")) as f:
                weight_map.update(json.load(f))
            os.remove(os.path.join(path, f"bitlattice.r{r}-json"))
            os.remove(os.path.join(path, f"index.r{r}-json"))
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(path, "bitlattice.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if hasattr(model, "config") and hasattr(model.config, "to_json_file"):
        cfg = model.config.to_dict()
        cfg.pop("quantization_config", None)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)


@torch.no_grad()
def _load_bitlattice(path, config, model_cls, dtype, device, trust_remote_code, device_map=None, max_memory=None):
    from safetensors import safe_open
    with open(os.path.join(path, "bitlattice.json")) as f:
        meta = json.load(f)
    model = _skeleton(config, model_cls, dtype, trust_remote_code)
    device, dmap = _device_plan(path, model, {"kind": "bitlattice"}, None, device, device_map, max_memory)
    local = _local_device(dmap)
    mine = (lambda name: True) if local is None else (lambda name: device(name) == local)
    parts: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    buffers = dict(model.named_buffers())
    for shard in _shards(path):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if ".__" in k:
                    name, part = k.split(".__", 1)
                    if mine(name):
                        parts[name][part] = f.get_tensor(k).to(device(name))
                elif k in buffers:
                    _set(model, k, f.get_tensor(k).to(local or device(k)), False)
                elif mine(k):
                    _set(model, k, f.get_tensor(k).to(device(k)), True)
    for name, info in meta["tensors"].items():
        if not mine(name):
            continue
        p = parts[name]
        lt = LatticeTensor(p["packed"], p["scales"], p.get("global_scale"), p.get("zeros"),
                           _format_from_spec(info["format"]), info["shape"], getattr(torch, info["dtype"]),
                           p.get("hi_idx"), p.get("hi_val"))
        _set(model, name, lt, True)
    for name, src in meta.get("tied", {}).items():
        if not mine(src):
            continue
        mod_name, _, attr = name.rpartition(".")
        setattr(model.get_submodule(mod_name) if mod_name else model, attr, model.get_parameter(src))
    _finish(model, device, dmap, local)
    return model


def export_compressed_tensors(model: torch.nn.Module, path: str) -> None:
    """Write nvfp4 layers in the compressed-tensors 'nvfp4-pack-quantized' layout (weights only), other tensors
    dense in their dtype. Experimental: verify with the consuming runtime."""
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    flat, ignore = {}, []
    for name, p in model.named_parameters():
        if isinstance(p, LatticeTensor) and p.fmt.name == "nvfp4" and p.hi_idx is None:
            prefix = name.rpartition(".")[0]
            for k, v in C.to_ct_nvfp4(p).items():
                flat[f"{prefix}.{k}"] = v.cpu()
        else:
            if isinstance(p, LatticeTensor):
                p = p.dequantize()
            flat[name] = p.detach().contiguous().cpu()
            if name.endswith(".weight"):
                ignore.append(name.rpartition(".")[0])
    save_file(flat, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    if hasattr(model, "config"):
        cfg = model.config.to_dict()
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "input_activations": None, "output_activations": None,
                                          "weights": {"num_bits": 4, "type": "float", "strategy": "tensor_group",
                                                      "group_size": 16, "symmetric": True, "dynamic": False}}},
            "ignore": ignore, "quantization_status": "compressed"}
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
