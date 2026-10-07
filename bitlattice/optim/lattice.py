"""Lattice optimizer: updates every weight directly in its storage format, without master weights.

One step of a parameter W with gradient G:
    m = momentum(G)                        none | fp32 | ternary | bf16 | int8 | fp8 | nvfp4 (stored formats of
                                           m; bf16 / int8 / nvfp4 with stochastic rounding, fp8 to nearest; nvfp4
                                           for rows of 16-element blocks, int8 for other tensors)
    D = direction(m), RMS-clipped          muon | muon_block | factored | sinkhorn | colnorm | sign | raw
        optional neuron-wise normalization of matrix directions (normalize='row', NorMuon): every row divided by
        the root of an exponential average of its mean square (one fp32 value per row of state)
    t = latent(W, phase) * (1 - lr*wd) - lr*D
    refit group scales (quantized formats), then project t back to (code, phase) with unbiased rounding.
Every option can be overridden per parameter group.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

from .. import _ops
from .. import formats as F
from .. import reference as R
from ..packing import pack, packed_cols, sign_extend, unpack
from ..tensor import LatticeTensor

log = logging.getLogger("bitlattice")

__all__ = ["DEFAULTS", "DIRECTIONS", "MOMENTA", "ROUNDINGS", "SCALE_RULES", "Lattice"]

DIRECTIONS = ("muon_block", "muon", "factored", "sinkhorn", "colnorm", "sign", "raw")
MOMENTA = ("none", "fp32", "ternary", "bf16", "int8", "fp8", "nvfp4")
_MOM_BLOCK = 256                # elements per scale of int8 / fp8 momentum
ROUNDINGS = ("sr", "strat", "rtn")
SCALE_RULES = ("fit", "fit_latent", "fixed")

DEFAULTS: dict[str, Any] = dict(
    lr=3e-3, direction="muon_block", vector_direction="factored", phase_bits=2, rounding="sr", scale_rule="fit",
    momentum="none", beta1=0.9, beta2=0.95, weight_decay=0.0, clip=1.0, block=512, ns_steps=5, ns_dtype="bf16",
    ns_coeffs="muon", ns_algo="auto", sinkhorn_iters=2, trust=0.01, trust_lr=1.0, strata=16, tile_elems=1 << 24,
    normalize="none", row_beta=0.95, row_eps=1e-8,
)


def _validate(g: dict) -> None:
    for key, allowed in (("direction", DIRECTIONS), ("vector_direction", DIRECTIONS), ("momentum", MOMENTA),
                         ("rounding", ROUNDINGS), ("scale_rule", SCALE_RULES)):
        if g[key] not in allowed:
            raise ValueError(f"{key}={g[key]!r}; expected one of {allowed}")
    for key, allowed in (("ns_dtype", ("bf16", "fp16", "fp32")), ("ns_coeffs", ("muon", "polar_express")),
                         ("ns_algo", ("auto", "standard", "gram"))):
        if g[key] not in allowed:
            raise ValueError(f"{key}={g[key]!r}; expected one of {allowed}")
    if not 0 <= g["phase_bits"] <= 8:
        raise ValueError("phase_bits must be in [0, 8]")
    if g["vector_direction"] in ("muon", "muon_block"):
        raise ValueError("vector_direction cannot be a matrix direction")
    if g["normalize"] not in ("none", "row"):
        raise ValueError("normalize must be 'none' or 'row'")



def _row_length(p: torch.Tensor) -> int:
    return p.numel() // p.shape[0] if p.dim() >= 2 else p.numel()


def _check_momentum(g: dict) -> None:
    """nvfp4 momentum needs the extension and CUDA parameters; tensors whose rows are not made of 16-element blocks
    (e.g. short convolution kernels) store int8 momentum instead -- listed in the log."""
    if g["momentum"] != "nvfp4":
        return
    if not _ops.HAVE_EXT or any(not p.is_cuda for p in g["params"]):
        raise ValueError("momentum='nvfp4' needs the bitlattice CUDA extension and parameters on the GPU")
    odd = sorted({tuple(p.shape) for p in g["params"] if _row_length(p) % 16})
    if odd:
        log.info("momentum nvfp4: %d parameter shape(s) with rows not made of 16-element blocks keep int8 "
                 "momentum: %s", len(odd), odd[:8])


def _restrict_rows(G: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Row-sparse gradient holding only ``rows`` (parameters marked with ``bitlattice.surgery.train_rows_``)."""
    rows = rows.to(G.device)
    if G.is_sparse:
        G = G.coalesce()
        idx, val = G.indices()[0], G.values()
        keep = torch.isin(idx, rows)
        idx, val = idx[keep], val[keep]
    else:
        idx, val = rows, G[rows]
    return torch.sparse_coo_tensor(idx.unsqueeze(0), val, G.shape, is_coalesced=True, check_invariants=False)


def _ns(g: dict) -> dict:
    return {"ns_coeffs": g["ns_coeffs"], "ns_algo": g["ns_algo"]}


def _row_norm(g: dict, st: dict, rows: int, device, sl=None) -> dict:
    """Keyword arguments of ``_ops.direction`` for the neuron-wise normalization (state created on first use);
    ``sl``: a slice or index tensor of the rows being updated."""
    if g["normalize"] != "row" or g["direction"] == "factored":     # factored directions are already adaptive
        return {}
    if "nv" not in st:
        st["nv"] = torch.zeros(rows, device=device)
    v = st["nv"] if sl is None else st["nv"][sl]
    return {"row_state": v, "row_k": st["step"], "row_beta": g["row_beta"], "row_eps": g["row_eps"]}


class Lattice(torch.optim.Optimizer):
    """The Lattice optimizer.

    ``params``     a model (its trainable parameters, grouped by :func:`bitlattice.optim.param_groups`: embeddings
                   and output heads get a factored direction, vectors no weight decay) or an iterable of tensors /
                   param-group dicts (any key of ``DEFAULTS`` can be set per group)
    ``preset``     a name in ``bitlattice.optim.PRESETS``: a set of options; options given here override it
    ``state``      where per-element state (phase, momentum) lives: 'gpu', 'ram' (pinned host memory: copied to the
                   GPU on a side stream before a parameter's update, prefetched in the update order recorded on
                   the first step, copied back afterwards; at most two parameters' state on the GPU) or 'auto'
                   ('ram' when the state would take more than ``bitlattice.storage.AUTO_STATE_FRACTION`` of the free
                   memory of a device; the choice is logged and kept in ``self.state_placement``)
    ``fused_backward``  apply each parameter's update as soon as its gradient is accumulated and free the gradient
                   (the optimizer step then only advances the step counter); peak memory holds one gradient
    """

    def __init__(self, params, preset: str | None = None, *, lr: float = DEFAULTS["lr"], state: str = "gpu",
                 fused_backward: bool = False, seed: int = 0, weight_decay: float = 0.0, **options):
        from .presets import PRESETS, param_groups
        if preset is not None:
            if preset not in PRESETS:
                raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
            options = {**PRESETS[preset], **options}
        head_momentum = options.pop("head_momentum", "none")
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown options {sorted(unknown)}")
        if isinstance(params, torch.nn.Module):
            direction = options.get("direction", DEFAULTS["direction"])
            params = param_groups(params, head_direction="factored" if direction.startswith("muon") else direction,
                                  head_momentum=head_momentum, weight_decay=weight_decay)
        defaults = {**DEFAULTS, **options, "lr": lr, "weight_decay": options.get("weight_decay", weight_decay)}
        super().__init__(params, defaults)
        for g in self.param_groups:
            _validate(g)
            _check_momentum(g)
        if state not in ("gpu", "ram", "auto"):
            raise ValueError("state must be 'gpu', 'ram' or 'auto'")
        if state == "auto":
            from ..storage import choose_placement
            state = choose_placement(self.param_groups)
            log.info("optimizer state placed in %s memory", "host" if state == "ram" else "GPU")
        self.state_placement = state
        state_device = "cpu" if state == "ram" else "cuda"
        self.seed = int(seed)
        self.fused_backward = fused_backward
        self.state_device = state_device
        self._copy_streams: dict = {}     # device -> side stream for host-memory state transfers
        self._host_bufs: dict = {}        # (parameter, state key) -> its exact-size pinned host buffer
        self._last_copy: dict = {}        # device -> event of the last write-back
        self._order: dict = {}            # device -> parameters in update order (recorded on the first step)
        self._next: dict = {}
        self._recording = True
        self._prefetched: dict = {}
        self._pid = {}
        for g in self.param_groups:
            for p in g["params"]:
                self._pid[p] = len(self._pid)
        self._group_of = {p: g for g in self.param_groups for p in g["params"]}
        self._done: set = set()
        self._defer = 0
        self._hooks = []
        if fused_backward:
            for p in self._pid:
                if p.requires_grad:
                    self._hooks.append(p.register_post_accumulate_grad_hook(self._on_grad))

    # ------------------------------------------------------------------------------------------------------------
    def _on_grad(self, p: torch.Tensor) -> None:
        if p.grad is None or self._defer:
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
                with torch.cuda.device(p.device) if p.is_cuda else contextlib.nullcontext():
                    self._update(p, g)
                if self.fused_backward:
                    p.grad = None
        self._done.clear()
        if self._recording and self._order:
            self._recording = False
            for order in self._order.values():
                self._next.update(zip(order, order[1:]))
                self._next[order[-1]] = order[0]
        return loss

    @contextlib.contextmanager
    def accumulate(self):
        """Gradient accumulation with ``fused_backward``: inside the context gradients are summed in ``.grad`` and
        no update is applied; the next backward pass outside the context applies the update with the total."""
        self._defer += 1
        try:
            yield
        finally:
            self._defer -= 1

    # ------------------------------------------------------------------------------------------------------------
    # host-memory state

    _OFFLOAD = ("phase", "hi_phase", "m", "mq", "mb", "m8", "m8s", "m4p", "m4s", "m4g")

    def _offloading(self, p: torch.Tensor) -> bool:
        return self.state_device == "cpu" and p.is_cuda

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        s = self._copy_streams.get(device)
        if s is None:
            s = self._copy_streams[device] = torch.cuda.Stream(device=device)
        return s

    def _prefetch(self, q: torch.Tensor) -> None:
        st = self.state.get(q)
        if not st or q in self._prefetched:
            return
        host = {k: st[k] for k in self._OFFLOAD if k in st and not st[k].is_cuda}
        if not host:
            return
        stream = self._stream(q.device)
        with torch.cuda.stream(stream):
            gpu = {k: v.to(q.device, non_blocking=True) for k, v in host.items()}
            ev = torch.cuda.Event()
            ev.record(stream)
        self._prefetched[q] = (gpu, ev)

    def _bring(self, p: torch.Tensor, st: dict) -> None:
        if not self._offloading(p):
            return
        if self._recording:
            self._order.setdefault(p.device, []).append(p)
        cur = torch.cuda.current_stream(p.device)
        pre = self._prefetched.pop(p, None)
        if pre is not None:
            gpu, ev = pre
            cur.wait_event(ev)
            for k, t in gpu.items():
                t.record_stream(cur)
                st[k] = t
        else:
            last = self._last_copy.get(p.device)
            if last is not None:
                cur.wait_event(last)
            for k in self._OFFLOAD:
                if k in st and not st[k].is_cuda:
                    st[k] = st[k].to(p.device, non_blocking=True)
        nxt = self._next.get(p)
        if nxt is not None:
            self._prefetch(nxt)

    def _send_back(self, p: torch.Tensor, st: dict) -> None:
        if not self._offloading(p):
            return
        stream = self._stream(p.device)
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(p.device))
        with torch.cuda.stream(stream):
            stream.wait_event(done)
            for k in self._OFFLOAD:
                if k in st and st[k].is_cuda:
                    gpu = st[k]
                    # the same exact-size buffer every step: its previous contents were read by the copy to the
                    # GPU, which the update (and so this write-back) follows
                    host = self._host_bufs.get((p, k))
                    if host is None or host.shape != gpu.shape or host.dtype != gpu.dtype:
                        from ..hostmem import pinned_empty
                        host = self._host_bufs[(p, k)] = pinned_empty(gpu.shape, gpu.dtype)
                    host.copy_(gpu, non_blocking=True)
                    gpu.record_stream(stream)
                    st[k] = host
            ev = torch.cuda.Event()
            ev.record(stream)
            self._last_copy[p.device] = ev

    def synchronize(self) -> None:
        """Wait for pending host-memory state transfers."""
        for s in self._copy_streams.values():
            s.synchronize()

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
        if mode in ("bf16", "int8", "fp8", "nvfp4"):
            if mode == "nvfp4" and X.shape[1] % 16:
                mode = "int8"                              # rows not made of 16-blocks (see _check_momentum)
            m = self._load_momentum(st, mode, X)
            m.mul_(g["beta1"]).add_(X, alpha=1 - g["beta1"])
            self._store_momentum(st, mode, m, (self.seed + 7919 * pid + 104729 * st["step"]) & 0x7FFFFFFF)
            return m
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

    # momentum stored in a low-precision format: dequantized to fp32 for the update, re-quantized after it
    def _load_momentum(self, st: dict, mode: str, X: torch.Tensor) -> torch.Tensor:
        rows, cols = X.shape
        if mode == "bf16":
            return st["mb"].float() if "mb" in st else torch.zeros_like(X)
        if mode in ("int8", "fp8"):
            if "m8" not in st:
                return torch.zeros_like(X)
            return (st["m8"].float() * st["m8s"][:, None]).reshape(-1)[:X.numel()].view(rows, cols)
        if "m4p" not in st:
            return torch.zeros_like(X)
        return _ops.dequantize(st["m4p"], st["m4s"], st["m4g"], None, F.get("nvfp4"), rows, cols, torch.float32)

    def _store_momentum(self, st: dict, mode: str, m: torch.Tensor, seed: int) -> None:
        gen = self._generator(m.device, seed)
        if mode == "bf16":                                   # stochastic rounding: random low 16 bits, truncate
            bits = m.contiguous().view(torch.int32)
            r = torch.randint(0, 1 << 16, bits.shape, device=m.device, dtype=torch.int32, generator=gen)
            st["mb"] = ((bits + r) & -65536).view(torch.float32).to(torch.bfloat16)
            return
        if mode in ("int8", "fp8"):
            flat = m.reshape(-1)
            fb = torch.nn.functional.pad(flat, (0, (-flat.numel()) % _MOM_BLOCK)).view(-1, _MOM_BLOCK)
            top = 127.0 if mode == "int8" else 448.0
            sc = (fb.abs().amax(1) / top).clamp_min(1e-30)
            q = fb / sc[:, None]
            if mode == "int8":
                u = torch.rand(q.shape, device=m.device, generator=gen)
                st["m8"] = torch.floor(q + u).clamp_(-127, 127).to(torch.int8)
            else:
                st["m8"] = q.to(torch.float8_e4m3fn)
            st["m8s"] = sc
            return
        rows, cols = m.shape
        packed, scales, gs = torch.ops.bitlattice.nvfp4_quantize(m.contiguous(), False, 1, True, False, False, seed)
        st["m4p"], st["m4s"], st["m4g"] = packed, scales.view(torch.float8_e4m3fn), gs

    def _generator(self, device: torch.device, seed: int) -> torch.Generator:
        gens = self.__dict__.setdefault("_gens", {})
        gen = gens.get(device)
        if gen is None:
            gen = gens[device] = torch.Generator(device=device)
        gen.manual_seed(seed)
        return gen

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
        rn = _row_norm(g, st, R.view2d(G).shape[0], G.device) if matrix else {}
        return _ops.direction(G, kind, r, c, st.get("k", 1), g["beta2"], g["block"], g["ns_steps"], g["ns_dtype"],
                              g["sinkhorn_iters"], g["clip"], **_ns(g), **rn)

    def _update(self, p: torch.Tensor, g: dict) -> None:
        p._bl_version = getattr(p, "_bl_version", 0) + 1          # invalidates cached derived operands
        from ..backend import drop_cached
        drop_cached(p)                                            # and frees them now
        st = self.state[p]
        self._bring(p, st)
        st["step"] = st.get("step", 0) + 1
        pid = self._pid[p]
        seed = (self.seed + 7919 * pid) & 0xFFFFFFFF
        G = p.grad
        rows = getattr(p, "_bl_rows", None)
        if rows is not None:
            G = _restrict_rows(G, rows)
        if G.is_sparse:
            if self._sparse_ok(p, g):
                self._update_sparse(p, G, st, g, seed)
                self._send_back(p, st)
                return
            G = G.to_dense()
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
            br = _ops.block_shape(g["block"])[0]
            step_rows = max(br, step_rows - step_rows % br)
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
                                   ns_dtype=g["ns_dtype"], clip=g["clip"], **_ns(g),
                                   **_row_norm(g, st, rows, p.device, slice(r0, r0 + n)))
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

    # ------------------------------------------------------------------------------------------------------------
    # row-sparse gradients (embeddings): only the referenced rows are updated

    def _sparse_ok(self, p: torch.Tensor, g: dict) -> bool:
        if p.dim() != 2 or g["momentum"] != "none" or not (p.is_cuda and _ops.HAVE_EXT):
            return False
        if isinstance(p, LatticeTensor):
            return p.hi_idx is None and p.fmt.scale.group != 0 and _ops.cuda_ok(p.fmt, p.rows, p.cols, p.packed.device)
        return p.dtype in (torch.bfloat16, torch.float16, torch.float32) and p.is_contiguous() and p.shape[1] % 8 == 0

    def _update_sparse(self, p: torch.Tensor, G: torch.Tensor, st: dict, g: dict, seed: int) -> None:
        G = G.coalesce()
        rows = G.indices()[0]
        vals = G.values()
        if rows.numel() == 0:
            return
        n_rows, cols = p.shape
        kind = g["direction"]
        if kind == "factored":
            if "r" not in st:
                st["r"] = torch.zeros(n_rows, device=p.device)
                st["c"] = torch.zeros(cols, device=p.device)
                st["k"] = 0
            st["k"] += 1
            r_sub = st["r"][rows].contiguous()
            D = _ops.direction(vals, "factored", r_sub, st["c"], st["k"], g["beta2"], clip=g["clip"])
            st["r"][rows] = r_sub
        else:
            rn = _row_norm(g, st, n_rows, p.device)
            if rn:
                v_sub = st["nv"][rows].contiguous()
                rn["row_state"] = v_sub
            D = _ops.direction(vals, kind, block=g["block"], ns_steps=g["ns_steps"], ns_dtype=g["ns_dtype"], **_ns(g),
                               sinkhorn_iters=g["sinkhorn_iters"], clip=g["clip"], **rn)
            if rn:
                st["nv"][rows] = v_sub
        bits = g["phase_bits"]
        cb = _ops.phase_container(bits)
        kw = dict(lr=g["lr"], weight_decay=g["weight_decay"], phase_bits=bits, rounding=g["rounding"],
                  seed=(seed + 104729 * st["step"]) & 0xFFFFFFFF, step=st["step"], strata=g["strata"])
        if bits and "phase" not in st:
            st["phase"] = torch.zeros(n_rows, cols * cb // 8, dtype=torch.uint8, device=p.device)
        ph = st["phase"][rows].contiguous() if bits else None
        k = rows.numel()
        if isinstance(p, LatticeTensor):
            packed = p.packed[rows].contiguous()
            zeros = p.zeros[rows].contiguous() if p.zeros is not None else None
            sc = _ops.step_quantized(packed, ph, p.scales[rows].contiguous(), p.global_scale, zeros, D, p.fmt, k, cols,
                                     phase_cb=cb, scale_rule=g["scale_rule"], trust=g["trust"],
                                     trust_lr=g["trust_lr"], **kw)
            p.packed[rows] = packed
            p.scales[rows] = sc
        else:
            w = p.data[rows].contiguous()
            _ops.step_native(w, ph.reshape(-1) if bits else None, D, phase_cb=cb, **kw)
            p.data[rows] = w
        if bits:
            st["phase"][rows] = ph

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
                if "nv" in st:
                    out["factors"] += 32 * st["nv"].numel()
                if "m" in st:
                    out["momentum"] += 32 * n
                if "mq" in st:
                    out["momentum"] += 2 * n + 32 * st["ms"].numel()
                if "mb" in st:
                    out["momentum"] += 16 * n
                if "m8" in st:
                    out["momentum"] += 8 * n + 32 * st["m8s"].numel()
                if "m4p" in st:
                    out["momentum"] += 4 * n + 8 * st["m4s"].numel() + 32
        out["total"] = sum(out.values())
        return out

    def state_bytes(self) -> int:
        """Bytes actually allocated for optimizer state tensors."""
        return sum(v.numel() * v.element_size() for st in self.state.values() for v in st.values()
                   if torch.is_tensor(v))
