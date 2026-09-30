"""Memory-efficient output layer + cross-entropy.

``linear_cross_entropy(hidden, weight, labels)`` never materializes the full [tokens, vocab] logits: tokens are
processed in chunks, the forward keeps only per-token log-sum-exp values, and the backward recomputes each chunk's
logits to form d(hidden) and d(weight). The weight gradient is accumulated in one dense buffer (or returned per
chunk to a callback for fused updates).
"""
from __future__ import annotations

import torch

from ..tensor import LatticeTensor

__all__ = ["linear_cross_entropy", "causal_lm_loss"]


def _dense(w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return w.dequantize(dtype) if isinstance(w, LatticeTensor) else w.to(dtype)


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, labels, chunk: int, ignore_index: int, dtype):
        h = hidden.reshape(-1, hidden.shape[-1])
        y = labels.reshape(-1)
        W = _dense(weight, dtype)
        n = h.shape[0]
        lse = torch.empty(n, dtype=torch.float32, device=h.device)
        tgt = torch.zeros(n, dtype=torch.float32, device=h.device)
        valid = y != ignore_index
        ysafe = torch.where(valid, y, torch.zeros_like(y))
        for s in range(0, n, chunk):
            logits = (h[s:s + chunk].to(dtype) @ W.t()).float()
            if bias is not None:
                logits += bias.float()
            lse[s:s + chunk] = torch.logsumexp(logits, -1)
            tgt[s:s + chunk] = logits.gather(1, ysafe[s:s + chunk, None]).squeeze(1)
        count = valid.sum().clamp_min(1)
        loss = ((lse - tgt) * valid).sum() / count
        ctx.save_for_backward(hidden, lse, ysafe, valid, count)
        ctx.weight, ctx.bias, ctx.chunk, ctx.dtype = weight, bias, chunk, dtype
        return loss

    @staticmethod
    def backward(ctx, g):
        hidden, lse, ysafe, valid, count = ctx.saved_tensors
        weight, bias, chunk, dtype = ctx.weight, ctx.bias, ctx.chunk, ctx.dtype
        h = hidden.reshape(-1, hidden.shape[-1])
        W = _dense(weight, dtype)
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gh = torch.empty_like(h) if need_h else None
        gw = torch.zeros(W.shape, dtype=torch.float32, device=h.device) if need_w else None
        gb = torch.zeros(W.shape[0], dtype=torch.float32, device=h.device) if bias is not None and \
            ctx.needs_input_grad[2] else None
        scale = (g / count).float()
        for s in range(0, h.shape[0], chunk):
            hc = h[s:s + chunk].to(dtype)
            logits = (hc @ W.t()).float()
            if bias is not None:
                logits += bias.float()
            p = logits.sub_(lse[s:s + chunk, None]).exp_()                     # softmax in place
            p[torch.arange(p.shape[0], device=p.device), ysafe[s:s + chunk]] -= 1.0
            p.mul_(valid[s:s + chunk, None] * scale)
            pd = p.to(dtype)
            if need_h:
                gh[s:s + chunk] = (pd @ W).to(gh.dtype)
            if need_w:
                gw.addmm_(pd.t().float(), hc.float())
            if gb is not None:
                gb += p.sum(0)
        gh = gh.reshape(hidden.shape) if need_h else None
        gw = gw.to(weight.dtype) if need_w else None
        gb = gb.to(bias.dtype) if gb is not None else None
        return gh, gw, gb, None, None, None, None


def linear_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                         bias: torch.Tensor | None = None, chunk: int = 1024, ignore_index: int = -100,
                         dtype: torch.dtype | None = None) -> torch.Tensor:
    """Mean cross-entropy of ``hidden @ weight.T (+ bias)`` against ``labels`` without full logits."""
    dtype = dtype or (hidden.dtype if hidden.dtype in (torch.bfloat16, torch.float16) else torch.float32)
    return _LinearCE.apply(hidden, weight, bias, labels, chunk, ignore_index, dtype)


def causal_lm_loss(model: torch.nn.Module, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                   chunk: int = 1024, **kwargs) -> torch.Tensor:
    """Next-token loss of a Hugging Face causal LM computed with :func:`linear_cross_entropy`.

    ``labels`` defaults to ``input_ids``; positions equal to -100 are ignored. Uses ``model.get_decoder()`` (or
    ``model.model``) for hidden states and ``model.get_output_embeddings()`` for the head."""
    labels = input_ids if labels is None else labels
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    hidden = backbone(input_ids=input_ids, **kwargs)[0]
    head = model.get_output_embeddings()
    return linear_cross_entropy(hidden[:, :-1], head.weight, labels[:, 1:], getattr(head, "bias", None), chunk)
