"""Fused decoder layers: a pre-norm transformer layer computed by Bitlattice code instead of its module tree.

A decoder layer of Qwen3.5 (and models of the same shape)

    x1 = x + mixer(norm1(x))          mixer: gated delta net (linear attention) or gated softmax attention
    y  = x1 + down(silu(gate(h)) * up(h)),  h = norm2(x1)

normally runs as dozens of modules, each a few PyTorch operators, plus autograd and activation checkpointing around
them: on a slow CPU core the launches, not the GPU, set the pace. The ``fused_block`` conversion replaces the layer by
one autograd node:

* the forward keeps only the layer input; the backward recomputes the layer and differentiates it by hand;
* projections that read the same input (q / k / v / z, gate / up) share one quantization of it, and the operands
  of the weight gradients (NVFP4 and MXFP8 routes) are made once per input; with deferred weight gradients they are
  kept for the end of the unit without passing through autograd;
* the norms run on the CUDA kernels of :mod:`bitlattice.nn.norm`, the short convolution on
  :func:`bitlattice.nn.causal_conv1d`, the delta rule on fla's chunked kernels, attention on the fused SDPA kernels;
  only these few operators keep an autograd graph, local to the layer's backward.

The layer's weights keep their compute settings (``bitlattice.surgery.set_compute_``): each projection computes on
its route ('exact', 'nvfp4' with its FP4 recipe, 'mxfp8'); forward-only routes are not supported here. The
conversion adopts the layer's parameters (same objects and names) and is checked against the layer on a random
input. Calls with a cache, an attention mask or packed sequences go to the original layer (generation, evaluation);
in training they are an error.
"""
from __future__ import annotations

import contextlib
import os

import torch

from .. import backend
from ..tensor import LatticeTensor

__all__ = ["FusedBlock", "make_fused_block"]


# projections


class _Proj:
    """A linear layer inside a fused block: computes on its weight's route, by hand (no autograd)."""

    __slots__ = ("mod", "name")

    def __init__(self, mod: torch.nn.Module, name: str):
        self.mod, self.name = mod, name

    @property
    def weight(self) -> torch.Tensor:
        return self.mod.weight

    def route(self, x: torch.Tensor):
        """(route, Compute) for input ``x``: 'exact', 'fp4_all' or 'mxfp8_all'."""
        w = self.weight
        if not isinstance(w, LatticeTensor):
            return "dense", None
        from ..compute_policy import route
        r, comp = route(x, w)
        if r not in ("exact", "fp4_all", "mxfp8_all"):
            raise RuntimeError(f"{self.name}: fused_block computes the backward of every projection; the forward-only "
                               f"route {r!r} is not supported (use the same setting with backward=True)")
        return r, comp


class _Env:
    """What a run replayed from a CUDA graph takes from outside the graph: the weights' derived operands (made
    eagerly into fixed buffers, see :class:`_Scratch`), a device counter for the stochastic seeds, and a list that
    receives the weight-gradient operands (copied out after the replay)."""

    __slots__ = ("epoch", "w_ops", "wt_ops", "keep")

    def __init__(self, epoch, w_ops: dict, wt_ops: dict):
        self.epoch, self.w_ops, self.wt_ops = epoch, w_ops, wt_ops
        self.keep: list = []


class _Input:
    """One input [M, K] of several projections: its quantized operands are made once per (route, recipe) and
    shared, so is the transposed operand of the weight gradients."""

    def __init__(self, x2: torch.Tensor, train: bool, env: _Env | None = None):
        self.x2, self.train, self.env = x2, train, env
        self.ops: dict = {}

    def operands(self, route: str, comp, site):
        key = (route, comp.nvfp4 if route == "fp4_all" else None)
        hit = self.ops.get(key)
        if hit is not None:
            return hit
        x = self.x2
        if route == "fp4_all":
            recipe = comp.nvfp4
            if self.train:
                xq, xt = backend._input_operands(x, recipe, site, self.env.epoch if self.env else None)
            else:
                xq = backend._act_operand(backend._pad_rows(x, 16).contiguous(), recipe)
                xt = None
        else:                                                                    # mxfp8_all
            xq, xt = backend._mx2(backend._pad_rows(x, 128).contiguous(), True, self.train)
        self.ops[key] = (xq, xt)
        return xq, xt


def _kernel_scope(comp):
    if comp is None or comp.kernels is None:
        return contextlib.nullcontext()
    from ..kernel_choice import kernels
    return kernels(comp.kernels)


def _project(p: _Proj, inp: _Input, saved: dict | None) -> torch.Tensor:
    """y = x W^T (+ b) for x = inp.x2; ``saved`` (training) receives what the backward of this projection needs."""
    x = inp.x2
    M = x.shape[0]
    route, comp = p.route(x)
    w, b = p.weight, getattr(p.mod, "bias", None)
    with _kernel_scope(comp):
        if route == "fp4_all":
            xq, xt = inp.operands(route, comp, backend._site(w, "x"))
            wq = inp.env.w_ops[p.name] if inp.env is not None else backend.weight_operand(w)
            y = backend.fp4_mm(xq, wq, x.dtype)[:M]
        elif route == "mxfp8_all":
            if inp.env is not None:
                raise RuntimeError(f"{p.name}: CUDA graphs of fused blocks support the 'nvfp4' and exact routes")
            xq, xt = inp.operands(route, comp, None)
            wq, _ = backend.mx_weight_operands(w)
            y = backend.mx_mm(xq, wq, x.dtype)[:M]
        else:
            xt = None
            W = w.dequantize(x.dtype) if route == "exact" else w
            y = torch.nn.functional.linear(x, W)
    if b is not None:
        y = y + b.to(y.dtype)
    if saved is not None:
        # quantized routes keep only x^T; the exact route keeps the input itself for its weight gradient
        saved[p.name] = (route, comp, xt, inp.x2 if route not in ("fp4_all", "mxfp8_all") else None, inp.env)
    return y


def _project_back(p: _Proj, gy: torch.Tensor, saved: dict, need_x: bool, grads: dict) -> torch.Tensor | None:
    """Input gradient of a projection (None if not needed); its weight / bias gradients go to ``grads`` (a deferred
    weight gradient is kept by the back end instead)."""
    route, comp, xt, x2, env = saved.pop(p.name)           # nothing of it is read again: freed as we go
    w = p.weight
    need_w = w.requires_grad
    gx = gw = None
    with _kernel_scope(comp):
        if route == "fp4_all" and env is not None:
            keep = []
            gx, _ = backend.fp4_backward(gy, w, xt, need_x, need_w, comp.nvfp4, wt=env.wt_ops.get(p.name),
                                         keep=keep, epoch=env.epoch)
            env.keep += [(p, gt, x_t) for gt, x_t in keep]
        elif route == "fp4_all":
            gx, gw = backend.fp4_backward(gy, w, xt, need_x, need_w, comp.nvfp4)
        elif route == "mxfp8_all":
            gx, gw = backend.mx_backward(gy, w, xt, need_x, need_w)
        else:
            W = w.dequantize(gy.dtype) if route == "exact" else w
            if need_x:
                gx = gy @ W
            if need_w:
                gw = gy.t() @ x2
    if gx is not None:
        gx = gx[:gy.shape[0]].to(gy.dtype)
    if gw is not None:
        grads[w] = gw.to(w.dtype)
    b = getattr(p.mod, "bias", None)
    if b is not None and b.requires_grad:
        grads[b] = gy.sum(0).to(b.dtype)
    return gx


def _acc(a: torch.Tensor | None, b: torch.Tensor | None) -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    return a.add_(b)


# norms


class _Norm:
    """An RMSNorm of the layer in one of the conventions of :mod:`bitlattice.nn.norm` (CUDA kernels)."""

    __slots__ = ("mod", "eps", "offset", "cast_first")

    def __init__(self, mod, eps: float, offset: float, cast_first: bool):
        self.mod, self.eps, self.offset, self.cast_first = mod, eps, offset, cast_first

    def forward(self, x2: torch.Tensor):
        return torch.ops.bitlattice.rms_norm_fwd(x2, None, self.mod.weight, self.eps, self.offset, self.cast_first)

    def backward(self, x2, rstd, dy, grads: dict) -> torch.Tensor:
        w = self.mod.weight
        dx, _, dw = torch.ops.bitlattice.rms_norm_bwd(x2, None, w, rstd, dy.contiguous(), self.offset, True, False,
                                                      w.requires_grad)
        if w.requires_grad:
            grads[w] = dw.to(w.dtype)
        return dx


def _norm_of(mod) -> _Norm:
    from .convert import Unfit, _eps
    from .norm import _convention
    eps = _eps(mod)
    if eps is None:
        raise Unfit(f"{type(mod).__name__} has no eps attribute")
    conv = _convention(mod, eps)
    if conv is None:
        raise Unfit(f"{type(mod).__name__} is not an RMSNorm convention this library implements")
    return _Norm(mod, eps, conv[1], conv[2])


# mixers: projections by hand, the rest ("core") through a small local graph


def _leaves(params: list) -> list:
    """Detached stand-ins of the layer's small parameters for the core's local graph (its gradients are taken at
    them; the parameters' own autograd nodes are never part of it)."""
    return [None if t is None else t.detach().requires_grad_(t.requires_grad) for t in params]


def _core_grads(o, leaves: list, small: list, params: list, do, grads: dict) -> list:
    """Gradients of the core output ``o`` (with ``do``) at the input leaves; the small parameters' gradients go to
    ``grads``."""
    want = [(t, p) for t, p in zip(small, params) if t is not None and t.requires_grad]
    got = torch.autograd.grad(o, leaves + [t for t, _ in want], do, allow_unused=True)
    for (t, p), g in zip(want, got[len(leaves):]):
        grads[p] = g.to(p.dtype) if g is not None else torch.zeros_like(p)
    return list(got[:len(leaves)])


class _DeltaNetMixer:
    """Gated delta net (Qwen3.5 / Qwen3-Next linear attention): in_proj_qkv / z / b / a, causal convolution + SiLU,
    chunked gated delta rule (q / k l2-normalized, decay -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)),
    normalize-then-gate RMSNorm with z, out_proj."""

    def __init__(self, m):
        from .convert import Unfit
        need = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "A_log", "dt_bias", "norm", "out_proj")
        if not all(hasattr(m, k) for k in need):
            raise Unfit("not a gated delta net layer")
        if getattr(m, "activation", "silu") not in ("silu", "swish"):
            raise Unfit(f"convolution activation {m.activation!r}")
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        except ImportError as e:
            raise Unfit(f"the delta rule kernels need flash-linear-attention ({e})") from None
        self.rule = chunk_gated_delta_rule
        self.m = m
        self.qkv, self.z = _Proj(m.in_proj_qkv, "in_proj_qkv"), _Proj(m.in_proj_z, "in_proj_z")
        self.b, self.a = _Proj(m.in_proj_b, "in_proj_b"), _Proj(m.in_proj_a, "in_proj_a")
        self.out = _Proj(m.out_proj, "out_proj")
        self.kd, self.vd = m.key_dim, m.value_dim
        self.hk, self.hv = m.head_k_dim, m.head_v_dim
        self.norm_eps = m.norm.variance_epsilon

    def projections(self):
        return [self.qkv, self.z, self.b, self.a, self.out]

    def small(self) -> list:
        """The layer's own small parameters used by the core (None where absent)."""
        m = self.m
        return [m.conv1d.weight, m.conv1d.bias, m.A_log, m.dt_bias, m.norm.weight]

    def _core(self, qkv, z, b, a, B: int, T: int, small: list):
        """Everything between the input projections and out_proj; [M, value_dim]. ``small``: the tensors to use for
        the conv weight / bias, A_log, dt_bias and the norm weight (leaves of a local graph in backward)."""
        from .conv import _CausalConv1d
        from .norm import rms_norm_gated
        conv_w, conv_b, A_log, dt_bias, norm_w = small
        conv = _CausalConv1d.apply(qkv.view(B, T, -1), conv_w.squeeze(1), conv_b, True)
        q, k, v = torch.split(conv, [self.kd, self.kd, self.vd], dim=-1)
        # the decay and beta in float32 / the activation dtype outside the kernel, as the gated_delta_net conversion
        # (fla's in-kernel gate differs by a few percent in the A_log / dt_bias gradients)
        g = -A_log.float().exp() * torch.nn.functional.softplus(a.view(B, T, -1).float() + dt_bias)
        o, _ = self.rule(q.reshape(B, T, -1, self.hk), k.reshape(B, T, -1, self.hk), v.reshape(B, T, -1, self.hv),
                         g=g, beta=b.view(B, T, -1).sigmoid(), use_qk_l2norm_in_kernel=True)
        o = rms_norm_gated(o.reshape(-1, self.hv).to(z.dtype), z.reshape(-1, self.hv), norm_w, self.norm_eps)
        return o.reshape(B * T, -1)

    def forward(self, h2: torch.Tensor, B: int, T: int, ctx, saved: dict | None, env=None):
        inp = _Input(h2, saved is not None, env)
        qkv, z = _project(self.qkv, inp, saved), _project(self.z, inp, saved)
        b, a = _project(self.b, inp, saved), _project(self.a, inp, saved)
        if saved is None:
            o = self._core(qkv, z, b, a, B, T, self.small())
            return _project(self.out, _Input(o, False, env), None)
        leaves = [t.detach().requires_grad_(True) for t in (qkv, z, b, a)]
        small = _leaves(self.small())
        with torch.enable_grad():
            o = self._core(*leaves, B, T, small)
        saved["core"] = (o, leaves, small)
        return _project(self.out, _Input(o.detach(), True, env), saved)

    def backward(self, dy, saved: dict, grads: dict) -> torch.Tensor:
        do = _project_back(self.out, dy, saved, True, grads)
        o, leaves, small = saved.pop("core")
        dqkv, dz, db, da = _core_grads(o, leaves, small, self.small(), do, grads)
        dh = _project_back(self.qkv, dqkv.contiguous(), saved, True, grads)
        dh = _acc(dh, _project_back(self.z, dz.contiguous(), saved, True, grads))
        dh = _acc(dh, _project_back(self.b, db.contiguous(), saved, True, grads))
        return _acc(dh, _project_back(self.a, da.contiguous(), saved, True, grads))


class _AttentionMixer:
    """Gated softmax attention (Qwen3.5 full-attention layers): q_proj gives queries and an output gate per head,
    q / k RMS-normalized per head, partial rotary embedding, causal SDPA with grouped KV heads, output gated by
    sigmoid(gate), o_proj."""

    def __init__(self, m):
        import sys
        from .convert import Unfit
        need = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm", "head_dim", "scaling")
        if not all(hasattr(m, k) for k in need):
            raise Unfit("not a q/k-normed attention layer")
        heads = m.o_proj.in_features // m.head_dim
        if m.q_proj.out_features != 2 * heads * m.head_dim:
            raise Unfit("q_proj does not hold a query and an output gate per head")
        if getattr(m, "attention_dropout", 0.0) or getattr(m, "sliding_window", None):
            raise Unfit("attention dropout / sliding window")
        self.rotary = getattr(sys.modules[type(m).__module__], "apply_rotary_pos_emb", None)
        if self.rotary is None:
            raise Unfit("its module defines no apply_rotary_pos_emb")
        self.m, self.heads, self.hd = m, heads, m.head_dim
        self.qn, self.kn = _norm_of(m.q_norm), _norm_of(m.k_norm)
        self.q, self.k = _Proj(m.q_proj, "q_proj"), _Proj(m.k_proj, "k_proj")
        self.v, self.o = _Proj(m.v_proj, "v_proj"), _Proj(m.o_proj, "o_proj")

    def projections(self):
        return [self.q, self.k, self.v, self.o]

    def small(self) -> list:
        return [self.m.q_norm.weight, self.m.k_norm.weight]

    def _core(self, qg, k, v, B: int, T: int, cos, sin, small: list):
        from .norm import rms_norm
        hd, m = self.hd, self.m
        qn_w, kn_w = small
        q, gate = torch.chunk(qg.view(B, T, -1, 2 * hd), 2, dim=-1)
        q = rms_norm(q, qn_w, self.qn.eps, self.qn.offset, self.qn.cast_first).transpose(1, 2)
        k = rms_norm(k.view(B, T, -1, hd), kn_w, self.kn.eps, self.kn.offset, self.kn.cast_first).transpose(1, 2)
        v = v.view(B, T, -1, hd).transpose(1, 2)
        q, k = self.rotary(q, k, cos, sin)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=m.scaling,
                                                             enable_gqa=q.shape[1] != k.shape[1])
        o = o.transpose(1, 2).reshape(B * T, -1)
        return o * torch.sigmoid(gate.reshape(B * T, -1))

    def forward(self, h2, B: int, T: int, ctx, saved: dict | None, env=None):
        cos, sin = ctx
        inp = _Input(h2, saved is not None, env)
        qg, k, v = _project(self.q, inp, saved), _project(self.k, inp, saved), _project(self.v, inp, saved)
        if saved is None:
            return _project(self.o, _Input(self._core(qg, k, v, B, T, cos, sin, self.small()), False, env), None)
        leaves = [t.detach().requires_grad_(True) for t in (qg, k, v)]
        small = _leaves(self.small())
        with torch.enable_grad():
            o = self._core(*leaves, B, T, cos, sin, small)
        saved["core"] = (o, leaves, small)
        return _project(self.o, _Input(o.detach(), True, env), saved)

    def backward(self, dy, saved: dict, grads: dict) -> torch.Tensor:
        do = _project_back(self.o, dy, saved, True, grads)
        o, leaves, small = saved.pop("core")
        dqg, dk, dv = _core_grads(o, leaves, small, self.small(), do, grads)
        dh = _project_back(self.q, dqg.contiguous(), saved, True, grads)
        dh = _acc(dh, _project_back(self.k, dk.contiguous(), saved, True, grads))
        return _acc(dh, _project_back(self.v, dv.contiguous(), saved, True, grads))


def _swiglu(g, u):
    if g.is_cuda and g.dtype in (torch.bfloat16, torch.float16):
        return torch.ops.bitlattice.swiglu_fwd(g, u)
    return torch.nn.functional.silu(g) * u


def _swiglu_bwd(g, u, da):
    """(dg, du) of silu(g) * u with the roundings of its autograd graph."""
    if g.is_cuda and g.dtype in (torch.bfloat16, torch.float16):
        return torch.ops.bitlattice.swiglu_bwd(g, u, da.contiguous())
    s = torch.nn.functional.silu(g)
    return torch.ops.aten.silu_backward(da * u, g), da * s


class _SwiGLU:
    """down(silu(gate(h)) * up(h)); the activation rounds as the reference does (silu to the activation dtype, then
    the product)."""

    def __init__(self, mlp):
        from .convert import Unfit
        if not all(hasattr(mlp, k) for k in ("gate_proj", "up_proj", "down_proj")):
            raise Unfit("the MLP is not gate_proj / up_proj / down_proj")
        act = getattr(mlp, "act_fn", None)
        if act is not None and type(act).__name__ not in ("SiLU", "SiLUActivation"):
            raise Unfit(f"MLP activation {type(act).__name__}")
        self.gate, self.up = _Proj(mlp.gate_proj, "gate_proj"), _Proj(mlp.up_proj, "up_proj")
        self.down = _Proj(mlp.down_proj, "down_proj")

    def projections(self):
        return [self.gate, self.up, self.down]

    def forward(self, h2, saved: dict | None, env=None):
        inp = _Input(h2, saved is not None, env)
        g, u = _project(self.gate, inp, saved), _project(self.up, inp, saved)
        act = _swiglu(g, u)
        if saved is not None:
            saved["gu"] = (g, u)
        return _project(self.down, _Input(act, saved is not None, env), saved)

    def backward(self, dy, saved: dict, grads: dict) -> torch.Tensor:
        da = _project_back(self.down, dy, saved, True, grads)
        g, u = saved.pop("gu")
        dg, du = _swiglu_bwd(g, u, da)
        dh = _project_back(self.gate, dg, saved, True, grads)
        return _acc(dh, _project_back(self.up, du, saved, True, grads))


# the layer


class _Plan:
    """Forward and hand-written backward of one fused layer."""

    def __init__(self, layer):
        from .convert import Unfit
        self.layer = layer
        self.norm1 = _norm_of(layer.input_layernorm)
        self.norm2 = _norm_of(layer.post_attention_layernorm)
        if hasattr(layer, "linear_attn"):
            self.mixer = _DeltaNetMixer(layer.linear_attn)
        elif hasattr(layer, "self_attn"):
            self.mixer = _AttentionMixer(layer.self_attn)
        else:
            raise Unfit("no linear_attn / self_attn mixer")
        if not hasattr(layer, "mlp"):
            raise Unfit("no mlp")
        self.mlp = _SwiGLU(layer.mlp)

    def params(self) -> list[torch.Tensor]:
        seen, out = set(), []
        for p in self.layer.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def run(self, x: torch.Tensor, ctx, saved: dict | None, env: _Env | None = None) -> torch.Tensor:
        B, T, D = x.shape
        x2 = x.reshape(B * T, D)
        h, r1 = self.norm1.forward(x2)
        x1 = x2 + self.mixer.forward(h, B, T, ctx, saved, env)
        h2, r2 = self.norm2.forward(x1)
        out = x1 + self.mlp.forward(h2, saved, env)
        if saved is not None:
            saved["norms"] = (x2, r1, x1, r2)
        return out.view(B, T, D)

    def projections(self) -> list[_Proj]:
        return self.mixer.projections() + self.mlp.projections()

    def backward(self, x, ctx, dy, env: _Env | None = None) -> tuple[torch.Tensor, dict]:
        saved, grads = {}, {}
        with torch.no_grad():
            self.run(x, ctx, saved, env)
            x2, r1, x1, r2 = saved.pop("norms")
            d = dy.reshape(x2.shape)
            dh2 = self.mlp.backward(d, saved, grads)
            dx1 = d + self.norm2.backward(x1, r2, dh2, grads)
            del x1, dh2
            dh = self.mixer.backward(dx1, saved, grads)
            dx = dx1.add_(self.norm1.backward(x2, r1, dh, grads))
        return dx.view(dy.shape), grads


_NVTX = os.environ.get("BL_NVTX", "") == "1"          # NVTX ranges for Nsight Systems timelines


def _range(name: str):
    return torch.cuda.nvtx.range(name) if _NVTX else contextlib.nullcontext()


# CUDA graphs
#
# A fused layer is a fixed sequence of kernels for a given input shape, so it can be recorded once and replayed: one
# host call per layer and micro-batch instead of a few hundred kernel launches. What a replay needs from outside is
# put in fixed places: the input (copied into a buffer per shape), the weights' derived operands (made eagerly into
# per-slot buffers when the slot changes layer or the weights change), a device counter mixed into the stochastic
# seeds (incremented by the graph itself), and the outputs (copied out of fixed buffers after the replay). All
# fused-layer graphs of a device share one memory pool: replays run one after another on one stream and no graph's
# memory is read after its replay.


class _Device:
    """Graph resources of one device."""

    _ALL: dict = {}

    def __init__(self, device: torch.device):
        self.device = device
        self.pool = torch.cuda.graph_pool_handle()
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.stream = torch.cuda.Stream(device=device)
        # a graph that is never replayed keeps the pool registered while the layers' graphs come and go
        self.anchor = torch.cuda.CUDAGraph()
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.anchor.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            self.epoch.add_(0)
            self.anchor.capture_end()
        self.bufs: dict = {}
        self.slots: dict = {}                  # slot -> (holder key, buffers, W operands, W^T operands)

    @classmethod
    def of(cls, device: torch.device) -> _Device:
        d = cls._ALL.get(device)
        if d is None:
            d = cls._ALL[device] = cls(device)
        return d

    def buf(self, key, shape, dtype) -> torch.Tensor:
        t = self.bufs.get(key)
        if t is None or tuple(t.shape) != tuple(shape) or t.dtype != dtype:
            t = self.bufs[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return t


def _slot_seed(slot: int, name: str) -> int:
    """Rotation seed of a slot's W^T operand: fixed, so the graphs that rotate gradients to match it stay valid."""
    h = 0x2545F491 + 7919 * slot
    for ch in name:
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return h


def _fp4_projs(plan, x):
    out = []
    for p in plan.projections():
        route, comp = p.route(x)
        if route == "fp4_all":
            out.append((p, comp))
    return out


def _load_slot(dev: _Device, plan, x, with_t: bool) -> tuple[dict, dict]:
    """Weight operands of the layer's FP4 projections in the slot's fixed buffers: (W operands, W^T operands). They
    are remade only when the slot held another layer or the weights changed since; W^T is added when first needed."""
    projs = _fp4_projs(plan, x)
    key = (id(plan), tuple(getattr(p.weight, "_bl_version", 0) for p, _ in projs))
    held, bufs, w_ops, wt_ops = dev.slots.get(plan.slot, (None, {}, {}, {}))
    if held != key:
        w_ops, wt_ops = {}, {}
        for p, _ in projs:
            w = p.weight
            blocked = backend.to_blocked(w.scales)
            sc = bufs.get((p.name, "s"))
            if sc is None or sc.shape != blocked.shape:
                sc = bufs[(p.name, "s")] = torch.empty_like(blocked)
                bufs[(p.name, "g")] = torch.empty(1, dtype=torch.float32, device=w.device)
            g = bufs[(p.name, "g")]
            sc.copy_(blocked)
            g.copy_(w.global_scale.float().reshape(1))
            w_ops[p.name] = backend.Fp4Operand(w.packed, sc, g, True)
    if with_t and len(wt_ops) < len(projs):
        for p, comp in projs:
            w = p.weight
            seed = _slot_seed(plan.slot, p.name)
            wt = backend.weight_t_rotated_operand(w, comp.nvfp4.backward_4over6, seed=seed)
            for k, t in (("tp", wt.packed), ("ts", wt.scales), ("tg", wt.global_scale.float().reshape(1))):
                b = bufs.get((p.name, k))
                if b is None or b.shape != t.shape:
                    b = bufs[(p.name, k)] = torch.empty_like(t)
                b.copy_(t)
            wt_ops[p.name] = backend.Fp4Operand(bufs[(p.name, "tp")], bufs[(p.name, "ts")], bufs[(p.name, "tg")],
                                                True, seed)
    dev.slots[plan.slot] = (key, bufs, w_ops, wt_ops)
    return w_ops, wt_ops


class _Graphs:
    """The forward and backward graphs of one fused layer, per input shape. A shape is run eagerly ``warmup`` times
    first (kernel autotuning must not happen during a capture)."""

    def __init__(self, plan, warmup: int = 1):
        self.plan, self.warmup = plan, warmup
        self.seen: dict = {}
        self.fwd: dict = {}
        self.bwd: dict = {}

    def _signature(self, x):
        out = []
        for p in self.plan.projections():
            route, comp = p.route(x)
            w = p.weight
            out.append((p.name, route, getattr(comp, "nvfp4", None),
                        (w.packed if isinstance(w, LatticeTensor) else w).data_ptr()))
        return tuple(out)

    def _key(self, kind, x, cos):
        return (kind, tuple(x.shape), x.dtype, None if cos is None else tuple(cos.shape), self._signature(x))

    def _ready(self, key) -> bool:
        n = self.seen.get(key, 0)
        self.seen[key] = n + 1
        return n >= self.warmup

    def _inputs(self, dev, key, tensors):
        out = []
        for i, t in enumerate(tensors):
            if t is None:
                out.append(None)
                continue
            b = dev.buf(("in", key[0], i, key[1:4]), t.shape, t.dtype)
            out.append(b)
        return out

    def _capture(self, dev, fn):
        g = torch.cuda.CUDAGraph()
        cur = torch.cuda.current_stream(dev.device)
        dev.stream.wait_stream(cur)
        with torch.cuda.stream(dev.stream):
            g.capture_begin(pool=dev.pool, capture_error_mode="thread_local")
            try:
                out = fn()
            finally:
                g.capture_end()
        cur.wait_stream(dev.stream)
        return g, out

    def forward(self, x, cos, sin):
        plan = self.plan
        key = self._key("f", x, cos)
        dev = _Device.of(x.device)
        if key not in self.fwd and not self._ready(key):
            return plan.run(x, (cos, sin), None)
        xin, cin, sinb = self._inputs(dev, key, (x, cos, sin))
        w_ops, _ = _load_slot(dev, plan, x, False)
        for b, t in ((xin, x), (cin, cos), (sinb, sin)):
            if b is not None:
                b.copy_(t)
        if key not in self.fwd:
            yout = dev.buf(("out", key[1:3]), x.shape, x.dtype)

            def run():
                y = plan.run(xin, (cin, sinb), None, _Env(None, w_ops, {}))
                yout.copy_(y)
            graph, _ = self._capture(dev, run)
            self.fwd[key] = (graph, yout)
        graph, yout = self.fwd[key]
        with _range("replay F"):
            graph.replay()
        return yout.clone()

    def backward(self, x, cos, sin, dy):
        plan = self.plan
        key = self._key("b", x, cos)
        dev = _Device.of(x.device)
        if key not in self.bwd and not self._ready(key):
            return plan.backward(x, (cos, sin), dy)
        xin, cin, sinb, dyin = self._inputs(dev, key, (x, cos, sin, dy))
        w_ops, wt_ops = _load_slot(dev, plan, x, True)
        for b, t in ((xin, x), (cin, cos), (sinb, sin), (dyin, dy)):
            if b is not None:
                b.copy_(t)
        if key not in self.bwd:
            self.bwd[key] = self._capture_backward(dev, key, xin, cin, sinb, dyin, w_ops, wt_ops)
        graph, dxout, small, staged = self.bwd[key]
        with _range("replay B"):
            graph.replay()
        grads = {t: b.clone() for t, b in small.items()}
        ops = {}
        for p, gt_b, xt_b in staged:
            if id(xt_b) not in ops:
                ops[id(xt_b)] = xt_b.copy()
            gt, xt = gt_b.copy(), ops[id(xt_b)]
            w = p.weight
            gw = backend._weight_grad(w, gt, xt)
            if gw is not None:
                grads[w] = gw.to(w.dtype)
        return dxout.clone(), grads

    def _capture_backward(self, dev, key, xin, cin, sinb, dyin, w_ops, wt_ops):
        plan = self.plan
        env = _Env(dev.epoch, w_ops, wt_ops)
        holder = {}

        def run():
            dev.epoch.add_(1)
            dx, grads = plan.backward(xin, (cin, sinb), dyin, env)
            dxout = dev.buf(("dx", key[1:3]), dx.shape, dx.dtype)
            dxout.copy_(dx)
            small = {}
            for t, g in grads.items():
                b = torch.empty(g.shape, dtype=g.dtype, device=g.device)       # graph pool: kept by the graph
                b.copy_(g)
                small[t] = b
            staged, by_id = [], {}
            for p, gt, xt in env.keep:
                gt_b = _Staged.of(dev, ("gt", p.name, key[1:3]), gt)
                if id(xt) not in by_id:
                    by_id[id(xt)] = _Staged.of(dev, ("xt", p.name, key[1:3]), xt)
                staged.append((p, gt_b, by_id[id(xt)]))
            holder.update(dxout=dxout, small=small, staged=staged)
        graph, _ = self._capture(dev, run)
        return graph, holder["dxout"], holder["small"], holder["staged"]


class _Staged:
    """An NVFP4 operand produced inside a graph, copied into fixed buffers there; ``copy`` gives a fresh operand."""

    __slots__ = ("packed", "scales", "g", "blocked", "seed")

    @classmethod
    def of(cls, dev, key, op) -> _Staged:
        st = cls()
        st.packed = dev.buf(key + ("p",), op.packed.shape, op.packed.dtype)
        st.scales = dev.buf(key + ("s",), op.scales.shape, op.scales.dtype)
        st.g = dev.buf(key + ("g",), op.global_scale.shape, op.global_scale.dtype)
        st.packed.copy_(op.packed)
        st.scales.copy_(op.scales)
        st.g.copy_(op.global_scale)
        st.blocked, st.seed = op.blocked, op.rotate_seed
        return st

    def copy(self) -> backend.Fp4Operand:
        return backend.Fp4Operand(self.packed.clone(), self.scales.clone(), self.g.clone(), self.blocked, self.seed)


class _FusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, graphs, cos, sin, *params):
        ctx.plan, ctx.graphs = plan, graphs
        ctx.save_for_backward(x, cos, sin)
        with torch.no_grad():
            if graphs is not None:
                return graphs.forward(x, cos, sin)
            return plan.run(x, (cos, sin), None)

    @staticmethod
    def backward(ctx, dy):
        x, cos, sin = ctx.saved_tensors
        plan, graphs = ctx.plan, ctx.graphs
        if graphs is not None:
            dx, grads = graphs.backward(x, cos, sin, dy.contiguous())
        else:
            dx, grads = plan.backward(x, (cos, sin), dy.contiguous())
        out = []
        for p, need in zip(plan.params(), ctx.needs_input_grad[5:]):
            out.append(grads.get(p) if need else None)
        return (dx if ctx.needs_input_grad[0] else None, None, None, None, None, *out)


_SLOTS = [0]


def make_fused_block(layer, graphs: bool = False, slots: int = 2):
    from .convert import Unfit
    if not all(hasattr(layer, k) for k in ("input_layernorm", "post_attention_layernorm")):
        raise Unfit("not a pre-norm decoder layer (input_layernorm, post_attention_layernorm)")
    if any(p.device.type == "meta" for p in layer.parameters()):
        raise Unfit("parameters not materialized (meta): the layer lives on another process")
    if graphs and not next(layer.parameters()).is_cuda:
        raise Unfit("CUDA graphs need the layer on a CUDA device")
    plan = _Plan(layer)
    plan.slot = _SLOTS[0] % max(1, slots)                 # consecutive layers use different operand slots
    _SLOTS[0] += 1
    return FusedBlock(layer, plan, _Graphs(plan) if graphs else None)


from .convert import Converted  # noqa: E402  (convert imports this module lazily)


class FusedBlock(Converted):
    def __init__(self, original, plan: _Plan, graphs: _Graphs | None = None):
        super().__init__(original)
        self.__dict__["plan"] = plan
        self.__dict__["graphs"] = graphs

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None,
                past_key_values=None, **kwargs):
        unknown = sorted(k for k, v in kwargs.items() if v is not None and k not in ("use_cache", "cache_position"))
        why = ("a cache" if past_key_values is not None or kwargs.get("use_cache") else
               "an attention mask" if attention_mask is not None else f"arguments {unknown}" if unknown else None)
        if why is not None:
            return self._delegate(f"called with {why}", hidden_states, position_embeddings=position_embeddings,
                                  attention_mask=attention_mask, position_ids=position_ids,
                                  past_key_values=past_key_values, **kwargs)
        x = hidden_states.contiguous()
        cos, sin = position_embeddings if position_embeddings is not None else (None, None)
        if isinstance(self.plan.mixer, _AttentionMixer) and cos is None:
            raise ValueError("fused_block: an attention layer needs position_embeddings")
        params = self.plan.params()
        if not (torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in params))):
            with torch.no_grad():
                if self.graphs is not None:
                    return self.graphs.forward(x, cos, sin)
                return self.plan.run(x, (cos, sin), None)
        return _FusedFn.apply(x, self.plan, self.graphs, cos, sin, *params)

    def settings(self):
        mixer = type(self.plan.mixer).__name__.strip("_").replace("Mixer", "").lower()
        return f"{mixer}, graphs={'on' if self.graphs is not None else 'off'}"
