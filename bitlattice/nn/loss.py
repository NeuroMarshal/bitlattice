"""Memory-efficient output layer + cross-entropy.

``linear_cross_entropy(hidden, weight, labels)`` never materializes the full [tokens, vocab] logits: tokens are
processed in chunks, the forward keeps only per-token log-sum-exp values, and the backward recomputes each chunk's
logits to form d(hidden) and d(weight). The weight gradient is accumulated in one dense buffer (or returned per
chunk to a callback for fused updates).

A frozen weight may live in pinned host memory (the ``host_linear`` conversion): it is then streamed to the GPU in row
tiles, once per group of tokens, overlapped with the computation.

With a frozen output layer (no gradient for the weight and bias) the gradient for ``hidden`` is formed in the forward
pass instead: one pass over vocabulary tiles with an online softmax (running maximum and sum, as in flash
attention) accumulates both the log-sum-exp and softmax-weighted rows of the weight, so each tile costs two GEMMs,
nothing is recomputed in backward and only the [tokens, hidden] gradient is kept. Checkpointing such a head is
unnecessary (and would repeat the work).
"""
from __future__ import annotations

import torch

from ..tensor import LatticeTensor

__all__ = ["linear_cross_entropy", "causal_lm_loss"]


def _dense(w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if w.device.type == "cpu" and not isinstance(w, LatticeTensor):
        return w                                   # host-memory weight: streamed, tiles are cast on the device
    return w.dequantize(dtype) if isinstance(w, LatticeTensor) else w.to(dtype)


_VOCAB_TILE_BYTES = 128 << 20          # fp32 logits per vocabulary tile
_HOST_TILE_BYTES = 64 << 20            # weight rows per tile streamed from host memory
_HOST_GROUP_BYTES = 512 << 20          # fp32 [tokens, hidden] accumulator per pass over a host-memory weight


def _vocab_pass(h, W, bias, y, valid, chunk, dtype, want_grad: bool):
    """(lse, target logit, d loss_sum / d h or None) per token with passes over vocabulary tiles and an online
    softmax. A weight on the GPU is read per chunk of tokens; a weight in host memory (a frozen layer converted with
    ``host_linear``) is streamed to the GPU once per group of tokens (as many as an accumulator of
    ``_HOST_GROUP_BYTES`` holds; all tokens without a gradient)."""
    from .stream import stream_rows
    n, d = h.shape
    V = W.shape[0]
    host = W.device != h.device
    lse = torch.empty(n, dtype=torch.float32, device=h.device)
    tgt = torch.empty(n, dtype=torch.float32, device=h.device)
    gh = torch.empty(n, d, dtype=h.dtype if h.dtype in (torch.bfloat16, torch.float16) else torch.float32,
                     device=h.device) if want_grad else None
    if not host:
        group = chunk
    elif want_grad:
        group = max(chunk, _HOST_GROUP_BYTES // (4 * d) // chunk * chunk)
    else:
        group = max(n, 1)
    for g0 in range(0, n, group):
        subs = [(s, min(n, g0 + group, s + chunk)) for s in range(g0, min(n, g0 + group), chunk)]
        c = max(e - s for s, e in subs)
        vt = max(128, min(V, _VOCAB_TILE_BYTES // (4 * c)))
        vt -= vt % 128 if vt < V else 0
        if host:
            vt = max(128, min(vt, _HOST_TILE_BYTES // (d * W.element_size()) // 128 * 128))
        st = []
        for s, e in subs:
            k = e - s
            st.append(dict(hc=h[s:e].to(dtype), yc=y[s:e], m=torch.full((k,), float("-inf"), device=h.device),
                           z=torch.zeros(k, device=h.device), tg=torch.zeros(k, device=h.device),
                           acc=torch.zeros(k, d, device=h.device) if want_grad else None,
                           trow=torch.zeros(k, d, dtype=W.dtype, device=h.device) if want_grad and host else None))
        tiles = (stream_rows(W, vt, h.device) if host else
                 ((v0, min(V, v0 + vt), W[v0:min(V, v0 + vt)]) for v0 in range(0, V, vt)))
        for v0, v1, Wt in tiles:
            for t in st:
                yc = t["yc"]
                L = (t["hc"] @ Wt.t()).float()
                if bias is not None:
                    L += bias[v0:v1].float()
                inside = (yc >= v0) & (yc < v1)
                col = (yc - v0).clamp(0, v1 - v0 - 1)
                t["tg"] += torch.where(inside, L.gather(1, col[:, None]).squeeze(1), 0.0)
                if t["trow"] is not None:                       # target rows: exactly one tile holds each
                    t["trow"] = torch.where(inside[:, None], Wt[col], t["trow"])
                m_new = torch.maximum(t["m"], L.amax(1))
                r = torch.exp(t["m"] - m_new)                   # rescale of the previous tiles
                P = L.sub_(m_new[:, None]).exp_()
                t["z"] = t["z"] * r + P.sum(1)
                if want_grad:
                    t["acc"].mul_(r[:, None]).add_((P.to(dtype) @ Wt.to(dtype)).float())
                t["m"] = m_new
        for (s, e), t in zip(subs, st):
            lse[s:e] = t["m"] + t["z"].log()
            tgt[s:e] = t["tg"]
            if want_grad:
                rows = t["trow"] if host else W[t["yc"]]
                g = t["acc"] / t["z"][:, None] - rows.float()       # softmax rows minus the target row
                gh[s:e] = (g * valid[s:e, None]).to(gh.dtype)
    return lse, tgt, gh


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, chunk: int, ignore_index: int, dtype):
        from ..kernel_choice import active
        ctx.in_forward = (ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]
                          and not (bias is not None and ctx.needs_input_grad[2]) and active().grad_in_forward)
        if ctx.in_forward:
            h = hidden.reshape(-1, hidden.shape[-1])
            y = labels.reshape(-1)
            valid = y != ignore_index
            ysafe = torch.where(valid, y, torch.zeros_like(y))
            lse, tgt, gh = _vocab_pass(h, _dense(weight, dtype), bias, ysafe, valid.float(), chunk, dtype, True)
            count = valid.sum().clamp_min(1)
            loss = ((lse - tgt) * valid).sum() / count
            ctx.save_for_backward(gh, count)
            ctx.shape, ctx.hdtype = hidden.shape, hidden.dtype
            return loss
        h = hidden.reshape(-1, hidden.shape[-1])
        y = labels.reshape(-1)
        valid = y != ignore_index
        ysafe = torch.where(valid, y, torch.zeros_like(y))
        lse, tgt, _ = _vocab_pass(h, _dense(weight, dtype), bias, ysafe, valid.float(), chunk, dtype, False)
        count = valid.sum().clamp_min(1)
        loss = ((lse - tgt) * valid).sum() / count
        ctx.save_for_backward(hidden, lse, ysafe, valid, count)
        ctx.weight, ctx.bias, ctx.chunk, ctx.dtype = weight, bias, chunk, dtype
        return loss

    @staticmethod
    def backward(ctx, g):
        """Vocabulary tiles outer, token chunks inner: each weight-gradient tile is accumulated in fp32 once and
        written in the weight dtype, so no fp32 buffer of the full [vocab, hidden] gradient is needed."""
        if ctx.in_forward:
            gh, count = ctx.saved_tensors
            out = (gh.float() * (g / count).float()).to(ctx.hdtype).reshape(ctx.shape)
            return out, None, None, None, None, None, None
        hidden, lse, ysafe, valid, count = ctx.saved_tensors
        weight, bias, chunk, dtype = ctx.weight, ctx.bias, ctx.chunk, ctx.dtype
        h = hidden.reshape(-1, hidden.shape[-1])
        W = _dense(weight, dtype)
        V, d = W.shape
        n = h.shape[0]
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_b = bias is not None and ctx.needs_input_grad[2]
        rows = getattr(weight, "_bl_rows", None) if need_w else None
        if rows is not None:                            # gradient for the marked rows only (row-sparse)
            need_w = False
        gh = torch.zeros(n, d, dtype=torch.float32, device=h.device) if need_h else None
        gw = torch.empty(V, d, dtype=weight.dtype, device=h.device) if need_w else None
        gb = torch.empty(V, dtype=bias.dtype, device=h.device) if need_b else None
        scale = (g / count).float()
        coef = valid.float() * scale                                    # [n]
        vt = max(chunk, min(V, (64 << 20) // max(d, 1)))                # ~256 MB fp32 weight-gradient tile
        vt = max(128, vt - vt % 128)                                    # aligned tiles keep the fast GEMM kernels
        for v0 in range(0, V, vt):
            v1 = min(V, v0 + vt)
            Wt = W[v0:v1]
            gwt = torch.zeros(v1 - v0, d, dtype=torch.float32, device=h.device) if need_w else None
            gbt = torch.zeros(v1 - v0, dtype=torch.float32, device=h.device) if need_b else None
            for s in range(0, n, chunk):
                hc = h[s:s + chunk].to(dtype)
                logits = (hc @ Wt.t()).float()
                if bias is not None:
                    logits += bias[v0:v1].float()
                p = logits.sub_(lse[s:s + chunk, None]).exp_()                 # softmax restricted to the tile
                y = ysafe[s:s + chunk]
                inside = ((y >= v0) & (y < v1)).to(p.dtype)                    # no host sync (no nonzero)
                col = (y - v0).clamp_(0, v1 - v0 - 1)
                p.scatter_add_(1, col[:, None], -inside[:, None])
                p.mul_(coef[s:s + chunk, None])
                pd = p.to(dtype)
                if need_h:
                    gh[s:s + chunk] += pd @ Wt
                if need_w:
                    gwt += pd.t() @ hc
                if need_b:
                    gbt += p.sum(0)
            if need_w:
                gw[v0:v1] = gwt.to(gw.dtype)
            if need_b:
                gb[v0:v1] = gbt.to(gb.dtype)
        if rows is not None:
            rows = rows.to(h.device)
            Wr = W[rows]
            gwr = torch.zeros(rows.numel(), d, dtype=torch.float32, device=h.device)
            for s in range(0, n, chunk):
                hc = h[s:s + chunk].to(dtype)
                p = (hc @ Wr.t()).float()
                if bias is not None:
                    p += bias[rows].float()
                p = p.sub_(lse[s:s + chunk, None]).exp_()
                p -= (ysafe[s:s + chunk, None] == rows[None, :]).float()
                p.mul_(coef[s:s + chunk, None])
                gwr += p.to(dtype).t() @ hc
            gw = torch.sparse_coo_tensor(rows.unsqueeze(0), gwr.to(weight.dtype), (V, d), check_invariants=False)
        gh = gh.to(hidden.dtype).reshape(hidden.shape) if need_h else None
        return gh, gw, gb, None, None, None, None


def linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                         bias: torch.Tensor | None = None, chunk: int = 1024, ignore_index: int = -100,
                         dtype: torch.dtype | None = None) -> torch.Tensor:
    """Mean cross-entropy of ``hidden @ weight.T (+ bias)`` against ``labels`` without full logits. Inputs on
    another device than ``weight`` are moved to it (layer-split models)."""
    from ..parallel import to_device
    if weight.device.type == "cpu" and hidden.is_cuda:            # a weight kept in host memory (host_linear)
        if weight.requires_grad or (bias is not None and bias.requires_grad):
            raise ValueError("a weight kept in host memory must be frozen: it is streamed to the GPU in tiles and "
                             "gets no gradient")
        labels = to_device(labels, hidden.device)
        bias = None if bias is None else bias.to(hidden.device)
    else:
        hidden, labels = to_device(hidden, weight.device), to_device(labels, weight.device)
    dtype = dtype or (hidden.dtype if hidden.dtype in (torch.bfloat16, torch.float16) else torch.float32)
    return _LinearCE.apply(hidden, weight, bias, labels, chunk, ignore_index, dtype)


def causal_lm_loss(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                   chunk: int = 1024, **kwargs) -> torch.Tensor:
    """Next-token loss of a Hugging Face causal LM computed with :func:`linear_cross_entropy`.

    ``labels`` defaults to ``input_ids``; positions equal to -100 are ignored. Uses ``model.get_decoder()`` (or
    ``model.model``) for hidden states (without a cache unless ``use_cache`` is given) and
    ``model.get_output_embeddings()`` for the head."""
    labels = input_ids if labels is None else labels
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    hidden = backbone(input_ids=input_ids, **{"use_cache": False, **kwargs})[0]
    head = model.get_output_embeddings()
    return linear_cross_entropy(hidden[:, :-1], head.weight, labels[:, 1:], getattr(head, "bias", None), chunk)
