"""Lattice optimizer: updates every weight directly in its storage format, without master weights.

One step of a parameter W with gradient G:
    m = momentum(G)                        none | fp32 | ternary
    D = direction(m), RMS-clipped          muon | muon_block | factored | sinkhorn | colnorm | sign | raw
    t = latent(W, phase) * (1 - lr*wd) - lr*D
    refit group scales (quantized formats), then project t back to (code, phase) with unbiased rounding.
Every option can be overridden per parameter group.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from .. import _ops
from .. import formats as F
from .. import reference as R
from ..packing import pack, packed_cols, sign_extend, unpack
from ..tensor import LatticeTensor

__all__ = ["Lattice", "DEFAULTS", "DIRECTIONS", "MOMENTA", "ROUNDINGS", "SCALE_RULES"]

DIRECTIONS = ("muon_block", "muon", "factored", "sinkhorn", "colnorm", "sign", "raw")
MOMENTA = ("none", "fp32", "ternary")
ROUNDINGS = ("sr", "strat", "rtn")
SCALE_RULES = ("fit", "fit_latent", "fixed")

DEFAULTS: dict[str, Any] = dict(
    lr=3e-3, direction="muon_block", vector_direction="factored", phase_bits=2, rounding="sr", scale_rule="fit",
    momentum="none", beta1=0.9, beta2=0.95, weight_decay=0.0, clip=1.0, block=512, ns_steps=5, ns_dtype="bf16",
    sinkhorn_iters=2, trust=0.01, trust_lr=1.0, strata=16, tile_elems=1 << 24,
)


def _validate(g: dict) -> None:
    for key, allowed in (("direction", DIRECTIONS), ("vector_direction", DIRECTIONS), ("momentum", MOMENTA),
                         ("rounding", ROUNDINGS), ("scale_rule", SCALE_RULES)):
        if g[key] not in allowed:
            raise ValueError(f"{key}={g[key]!r}; expected one of {allowed}")
    if not 0 <= g["phase_bits"] <= 8:
        raise ValueError("phase_bits must be in [0, 8]")
    if g["vector_direction"] in ("muon", "muon_block"):
        raise ValueError("vector_direction cannot be a matrix direction")


class Lattice(torch.optim.Optimizer):
    """``params``: iterable of tensors or param-group dicts (any key of ``DEFAULTS`` can be set per group).

    ``fused_backward=True`` applies each parameter's update as soon as its gradient is accumulated and frees the
    gradient (the optimizer step then only advances the step counter); peak memory holds one gradient at a time.

    ``state_device='cpu'`` keeps per-element state (phase, momentum) in pinned host memory. The state of a parameter
    is copied to the GPU on a side stream before its update (prefetched in the update order recorded on the first
    step) and copied back afterwards; at most two parameters' state reside on the GPU.
    """

    def __init__(self, params: Iterable, lr: float = DEFAULTS["lr"], *, seed: int = 0, fused_backward: bool = False,
                 state_device: str = "cuda", **options):
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown options {sorted(unknown)}")
        defaults = {**DEFAULTS, **options, "lr": lr}
        super().__init__(params, defaults)
        for g in self.param_groups:
            _validate(g)
        self.seed = int(seed)
        self.fused_backward = fused_backward
        if state_device not in ("cuda", "cpu"):
            raise ValueError("state_device must be 'cuda' or 'cpu'")
        self.state_device = state_device
        self._copy_stream = torch.cuda.Stream() if state_device == "cpu" and torch.cuda.is_available() else None
        self._last_copy = None
        self._order: list = []
        self._next: dict = {}
        self._recording = True
        self._prefetched: dict = {}
        self._pid = {}
        for g in self.param_groups:
            for p in g["params"]:
                self._pid[p] = len(self._pid)
        self._group_of = {p: g for g in self.param_groups for p in g["params"]}
        self._done: set = set()
        self._hooks = []
        if fused_backward:
            for p in self._pid:
                if p.requires_grad:
                    self._hooks.append(p.register_post_accumulate_grad_hook(self._on_grad))

    # ------------------------------------------------------------------------------------------------------------
    def _on_grad(self, p: torch.Tensor) -> None:
        if p.grad is None:
            return
        with torch.no_grad():
            self._update(p, self._group_of[p])
        p.grad = None
        self._done.add(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p in self._done or p.grad is None:
                    continue
                self._update(p, g)
                if self.fused_backward:
                    p.grad = None
        self._done.clear()
        if self._recording and self._order:
            self._recording = False
            self._next = {a: b for a, b in zip(self._order, self._order[1:])}
            self._next[self._order[-1]] = self._order[0]
        return loss

    # ------------------------------------------------------------------------------------------------------------
    # host-memory state

    _OFFLOAD = ("phase", "hi_phase", "m", "mq")

    def _prefetch(self, q: torch.Tensor) -> None:
        st = self.state.get(q)
        if not st or q in self._prefetched:
            return
        host = {k: st[k] for k in self._OFFLOAD if k in st and not st[k].is_cuda}
        if not host:
            return
        with torch.cuda.stream(self._copy_stream):
            gpu = {k: v.to(q.device, non_blocking=True) for k, v in host.items()}
            ev = torch.cuda.Event()
            ev.record(self._copy_stream)
        self._prefetched[q] = (gpu, ev)

    def _bring(self, p: torch.Tensor, st: dict) -> None:
        if self._copy_stream is None:
            return
        if self._recording:
            self._order.append(p)
        pre = self._prefetched.pop(p, None)
        if pre is not None:
            gpu, ev = pre
            torch.cuda.current_stream().wait_event(ev)
            for k, t in gpu.items():
                t.record_stream(torch.cuda.current_stream())
                st[k] = t
        else:
            if self._last_copy is not None:
                torch.cuda.current_stream().wait_event(self._last_copy)
            for k in self._OFFLOAD:
                if k in st and not st[k].is_cuda:
                    st[k] = st[k].to(p.device, non_blocking=True)
        nxt = self._next.get(p)
        if nxt is not None:
            self._prefetch(nxt)

    def _send_back(self, p: torch.Tensor, st: dict) -> None:
        if self._copy_stream is None:
            return
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(done)
            for k in self._OFFLOAD:
                if k in st and st[k].is_cuda:
                    gpu = st[k]
                    host = torch.empty(gpu.shape, dtype=gpu.dtype, device="cpu", pin_memory=True)
                    host.copy_(gpu, non_blocking=True)
                    gpu.record_stream(self._copy_stream)
                    st[k] = host
            self._last_copy = torch.cuda.Event()
            self._last_copy.record(self._copy_stream)

    def synchronize(self) -> None:
        """Wait for pending host-memory state transfers."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()

    def state_dict(self):
        self.synchronize()
        return super().state_dict()

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=True)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------------------------------------------------
    def _momentum(self, G: torch.Tensor, st: dict, g: dict, pid: int) -> torch.Tensor:
        mode = g["momentum"]
        if mode == "none":
            return G
        X = R.view2d(G).float()
        if mode == "fp32":
            if "m" not in st:
                st["m"] = torch.zeros_like(X)
            st["m"].mul_(g["beta1"]).add_(X, alpha=1 - g["beta1"])
            return st["m"]
        pk = X.abs().amax()
        X = X / torch.where(pk > 0, pk, torch.ones_like(pk))
        if "mq" not in st:
            st["mq"] = torch.zeros_like(X, dtype=torch.int8)
            st["ms"] = torch.zeros(X.shape[0], device=X.device)
        m = g["beta1"] * st["mq"].float() * st["ms"][:, None] + (1 - g["beta1"]) * X
        s = m.abs().amax(1).clamp_min(1e-30)
        q = m / s[:, None]
        fl = torch.floor(q)
        u = R.hash_uniform(q.numel(), self.seed + 7919 * pid + 1, st["step"], q.device).view_as(q)
        st["mq"] = (fl + (u < q - fl).float()).clamp(-1, 1).to(torch.int8)
        st["ms"] = s
        return m

    def _direction(self, G: torch.Tensor, st: dict, g: dict, matrix: bool) -> torch.Tensor:
        kind = g["direction"] if matrix else g["vector_direction"]
        r = c = None
        if kind == "factored":
            rows, cols = R.view2d(G).shape
            if "r" not in st:
                st["r"] = torch.zeros(rows, device=G.device)
                st["c"] = torch.zeros(cols, device=G.device)
                st["k"] = 0
            st["k"] += 1
            r, c = st["r"], st["c"]
        return _ops.direction(G, kind, r, c, st.get("k", 1), g["beta2"], g["block"], g["ns_steps"], g["ns_dtype"],
                              g["sinkhorn_iters"], g["clip"])

    def _update(self, p: torch.Tensor, g: dict) -> None:
        st = self.state[p]
        self._bring(p, st)
        st["step"] = st.get("step", 0) + 1
        pid = self._pid[p]
        seed = (self.seed + 7919 * pid) & 0xFFFFFFFF
        G = p.grad
        matrix = p.dim() >= 2 and min(p.shape[0], p.numel() // p.shape[0]) > 1
        if matrix and self._tileable(p, g):
            self._update_tiled(p, G, st, g, seed)
            self._send_back(p, st)
            return
        G = self._momentum(G, st, g, pid)
        D = self._direction(G, st, g, matrix)
        kw = dict(lr=g["lr"], weight_decay=g["weight_decay"], phase_bits=g["phase_bits"], rounding=g["rounding"],
                  seed=seed, step=st["step"], strata=g["strata"])
        if isinstance(p, LatticeTensor):
            self._update_lattice(p, D, st, g, kw)
        else:
            self._update_native(p, p.data, D, st, "phase", kw)
        self._send_back(p, st)

    # ------------------------------------------------------------------------------------------------------------
    # row tiles: bounded transient memory for large matrices (same result as the untiled update)

    def _tileable(self, p: torch.Tensor, g: dict) -> bool:
        if p.numel() <= g["tile_elems"] or g["momentum"] != "none" or not (p.is_cuda and _ops.HAVE_EXT):
            return False
        if g["direction"] not in ("factored", "muon_block", "sign", "raw"):
            return False
        cols = p.numel() // p.shape[0]
        if isinstance(p, LatticeTensor):
            return p.hi_idx is None and p.fmt.scale.group != 0 and _ops.cuda_ok(p.fmt, p.rows, cols, p.packed.device)
        return p.dtype in (torch.bfloat16, torch.float16, torch.float32) and p.is_contiguous() and cols % 8 == 0

    def _update_tiled(self, p: torch.Tensor, G: torch.Tensor, st: dict, g: dict, seed: int) -> None:
        rows = p.shape[0]
        cols = p.numel() // rows
        G2 = G.reshape(rows, cols)
        kind = g["direction"]
        step_rows = max(1, g["tile_elems"] // cols)
        if kind == "muon_block":
            step_rows = max(g["block"], step_rows - step_rows % g["block"])
        if kind == "factored":
            if "r" not in st:
                st["r"] = torch.zeros(rows, device=G.device)
                st["c"] = torch.zeros(cols, device=G.device)
                st["k"] = 0
            st["k"] += 1
            a, b = torch.ops.bitlattice.factored_scales(G2, st["r"], st["c"], st["k"], g["beta2"], g["clip"])
        bits = g["phase_bits"]
        cb = _ops.phase_container(bits)
        kw = dict(lr=g["lr"], weight_decay=g["weight_decay"], phase_bits=bits, rounding=g["rounding"], seed=seed,
                  step=st["step"], strata=g["strata"])
        quant = isinstance(p, LatticeTensor)
        if bits and "phase" not in st:
            st["phase"] = (torch.zeros(rows, packed_cols(cols, cb), dtype=torch.uint8, device=p.device) if quant else
                           torch.zeros(_ops.phase_numel(p.numel()) * cb // 8, dtype=torch.uint8, device=p.device))
        new_scales = []
        for r0 in range(0, rows, step_rows):
            n = min(step_rows, rows - r0)
            if kind == "factored":
                D = torch.ops.bitlattice.factored_apply(G2, a, b, r0, n)
            else:
                D = _ops.direction(G2[r0:r0 + n], kind, block=g["block"], ns_steps=g["ns_steps"],
                                   ns_dtype=g["ns_dtype"], clip=g["clip"])
            if quant:
                zeros = p.zeros[r0:r0 + n] if p.zeros is not None else None
                sc = _ops.step_quantized(p.packed[r0:r0 + n], st["phase"][r0:r0 + n] if bits else None,
                                         p.scales[r0:r0 + n], p.global_scale, zeros, D, p.fmt, n, cols, phase_cb=cb,
                                         scale_rule=g["scale_rule"], trust=g["trust"], trust_lr=g["trust_lr"],
                                         offset=r0 * cols, **kw)
                new_scales.append(sc)
            else:
                w = p.data.view(rows, cols)[r0:r0 + n]
                ph = st["phase"][r0 * cols * cb // 8:(r0 + n) * cols * cb // 8] if bits else None
                _ops.step_native(w, ph, D, phase_cb=cb, offset=r0 * cols, **kw)
        if quant:
            p.scales.copy_(torch.cat(new_scales, 0))

    def _update_native(self, p, w: torch.Tensor, D: torch.Tensor, st: dict, key: str, kw: dict) -> None:
        bits = kw["phase_bits"]
        cb = _ops.phase_container(bits)
        n = w.numel()
        if bits and key not in st:
            st[key] = torch.zeros(_ops.phase_numel(n) * cb // 8, dtype=torch.uint8, device=w.device)
        if w.is_cuda and _ops.HAVE_EXT and w.dtype in (torch.bfloat16, torch.float16, torch.float32) \
                and w.is_contiguous():
            _ops.step_native(w, st.get(key) if bits else None, D, phase_cb=cb, **kw)
            return
        phase = None
        if bits:
            phase = sign_extend(unpack(st[key].view(1, -1), cb, _ops.phase_numel(n)), bits)[0, :n].to(torch.int8)
        new, code = R.lattice_step_native(w, phase.view_as(w) if phase is not None else None, D, **kw)
        w.copy_(new)
        if bits:
            flat = torch.nn.functional.pad(code.reshape(-1).to(torch.int32) & ((1 << bits) - 1),
                                           (0, _ops.phase_numel(n) - n))
            st[key].copy_(pack(flat.view(1, -1), cb).view(-1))

    def _update_lattice(self, p: LatticeTensor, D: torch.Tensor, st: dict, g: dict, kw: dict) -> None:
        fmt, rows, cols = p.fmt, p.rows, p.cols
        bits = kw["phase_bits"]
        cb = _ops.phase_container(bits)
        if p.hi_idx is not None:
            D_hi = D[:, p.hi_idx].contiguous()
            D = D.clone()
            D[:, p.hi_idx] = 0.0
            self._update_native(p, p.hi_val, D_hi, st, "hi_phase", kw)
        if bits and "phase" not in st:
            st["phase"] = torch.zeros(rows, packed_cols(cols, cb), dtype=torch.uint8, device=p.packed.device)
        common = dict(scale_rule=g["scale_rule"], trust=g["trust"], trust_lr=g["trust_lr"])
        if _ops.cuda_ok(fmt, rows, cols, p.packed.device):
            new_scales = _ops.step_quantized(p.packed, st.get("phase") if bits else None, p.scales, p.global_scale,
                                             p.zeros, D, fmt, rows, cols, phase_cb=cb, **kw, **common)
            p.scales.copy_(new_scales)
            return
        q = R.Quantized(p.packed, p.scales, p.global_scale, p.zeros, rows, cols)
        phase = sign_extend(unpack(st["phase"], cb, cols), bits).to(torch.int8) if bits else None
        packed, scales, code = R.lattice_step_quantized(q, fmt, phase, D, **kw, **common)
        p.packed.copy_(packed)
        p.scales.copy_(scales)
        if bits:
            st["phase"].copy_(pack(code.to(torch.int32) & ((1 << bits) - 1), cb))

    # ------------------------------------------------------------------------------------------------------------
    def state_bits(self) -> dict[str, float]:
        """Logical optimizer state in bits, by component (phase, factors, momentum)."""
        out = {"phase": 0.0, "factors": 0.0, "momentum": 0.0}
        for g in self.param_groups:
            for p in g["params"]:
                st = self.state.get(p, {})
                n = p.numel()
                out["phase"] += n * g["phase_bits"]
                if "r" in st:
                    out["factors"] += 32 * (st["r"].numel() + st["c"].numel())
                if "m" in st:
                    out["momentum"] += 32 * n
                if "mq" in st:
                    out["momentum"] += 2 * n + 32 * st["ms"].numel()
        out["total"] = sum(out.values())
        return out

    def state_bytes(self) -> int:
        """Bytes actually allocated for optimizer state tensors."""
        return sum(v.numel() * v.element_size() for st in self.state.values() for v in st.values()
                   if torch.is_tensor(v))
