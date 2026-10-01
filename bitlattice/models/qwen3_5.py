"""Qwen3.5 / Qwen3.8 (hybrid Gated DeltaNet + attention) text models from multimodal checkpoints, with the
multi-token-prediction (MTP) head that transformers does not implement.

MTP head (one extra token ahead): for position t, the embedding of token t+1 and the backbone's last hidden state
at t are normalized separately, concatenated, projected back to the model width (``fc``), passed through one
full-attention decoder layer and the final norm, and decoded by the shared output head as token t+2.
"""
from __future__ import annotations

import copy

import torch

from .. import io as bl_io

__all__ = ["load", "MTPHead", "TEXT_KEY_MAP"]

TEXT_KEY_MAP = {r"^model\.visual\.": None, r"^model\.language_model\.": "model."}


class MTPHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
        cfg = copy.deepcopy(config)
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
        d = cfg.hidden_size
        self.fc = torch.nn.Linear(2 * d, d, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(d, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(d, eps=cfg.rms_norm_eps)
        self.layers = torch.nn.ModuleList([Qwen3_5DecoderLayer(cfg, cfg.num_hidden_layers - 1)])
        self.norm = Qwen3_5RMSNorm(d, eps=cfg.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, next_embeds: torch.Tensor, position_embeddings, **kwargs) -> torch.Tensor:
        """hidden [B, T, d]: backbone states at t; next_embeds [B, T, d]: embeddings of tokens t+1."""
        x = torch.cat([self.pre_fc_norm_embedding(next_embeds), self.pre_fc_norm_hidden(hidden)], -1)
        x = self.fc(x)
        for layer in self.layers:
            out = layer(x, position_embeddings=position_embeddings, **kwargs)
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)


def load(path: str, quant=None, *, mtp: bool = True, **kw):
    """Text model (Qwen3_5ForCausalLM) from a Qwen3.5-family checkpoint, visual tower dropped; with ``mtp`` the
    MTP head is attached as ``model.mtp`` (loaded from ``mtp.*``)."""
    from transformers import AutoConfig, Qwen3_5ForCausalLM
    cfg = AutoConfig.from_pretrained(path)
    text_cfg = getattr(cfg, "text_config", cfg)                 # multimodal checkpoint or a saved text model
    key_map = dict(TEXT_KEY_MAP)
    if not mtp:
        key_map[r"^mtp\."] = None

    class _WithMTP(Qwen3_5ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            if mtp:
                self.mtp = MTPHead(config)

    _WithMTP.__name__ = "Qwen3_5ForCausalLMWithMTP"
    model = bl_io.load(path, quant, config=text_cfg, model_cls=_WithMTP, key_map=key_map, **kw)
    return model


def _positions(B: int, T: int, device, offset: int = 0) -> torch.Tensor:
    """Multimodal-rope position ids [3, B, T] for plain text."""
    return (torch.arange(T, device=device) + offset).view(1, 1, -1).expand(3, B, -1)


def backbone_hidden(model, input_ids: torch.Tensor, grad: bool = False) -> torch.Tensor:
    """Last hidden states of the backbone before the final norm (what the MTP head consumes)."""
    box = {}
    h = model.model.norm.register_forward_pre_hook(lambda m, a: box.__setitem__("h", a[0]))
    try:
        with torch.set_grad_enabled(grad):
            model.model(input_ids=input_ids, use_cache=False)
    finally:
        h.remove()
    return box["h"]


def mtp_hidden(model, hidden: torch.Tensor, input_ids: torch.Tensor, pos_offset: int = 1,
               next_embeds: torch.Tensor | None = None) -> torch.Tensor:
    """MTP output states for positions 0..T-2 (predicting tokens 2..T): hidden[:, t] with embedding(token t+1)
    (``next_embeds`` when the embedding lives elsewhere, e.g. on another pipeline process)."""
    mtp = model.mtp
    dev = mtp.fc.weight.device
    h = hidden[:, :-1].to(dev)
    if next_embeds is None:
        next_embeds = model.model.embed_tokens(input_ids[:, 1:].to(model.model.embed_tokens.weight.device))
    e = next_embeds.to(dev)
    B, T = h.shape[:2]
    pos = _positions(B, T, dev, pos_offset)
    rot = model.model.rotary_emb
    cos, sin = rot(h, pos.to(next(rot.buffers()).device))
    return mtp(h, e, (cos.to(dev), sin.to(dev)), attention_mask=None, position_ids=pos[0])


def mtp_loss(model, input_ids: torch.Tensor, hidden: torch.Tensor | None = None, pos_offset: int = 1,
             chunk: int = 1024) -> torch.Tensor:
    """Cross-entropy of the MTP head predicting token t+2 (backbone frozen when ``hidden`` is given)."""
    from ..nn.loss import linear_cross_entropy
    if hidden is None:
        hidden = backbone_hidden(model, input_ids)
    m = mtp_hidden(model, hidden, input_ids, pos_offset)
    head = model.get_output_embeddings()
    return linear_cross_entropy(m[:, :-1], head.weight, input_ids[:, 2:].to(head.weight.device), None, chunk)


@torch.no_grad()
def accuracy(model, input_ids: torch.Tensor, hidden: torch.Tensor | None = None, pos_offset: int = 1) -> dict:
    """Top-1 accuracy of the main head (token t+1) and of the MTP head (token t+2), and the MTP accuracy on the
    positions where the main head is right (the speculative-decoding acceptance proxy)."""
    if hidden is None:
        hidden = backbone_hidden(model, input_ids)
    head = model.get_output_embeddings()
    hd = head.weight.device
    main = model.model.norm(hidden.to(model.model.norm.weight.device)).to(hd)
    ids = input_ids.to(hd)
    out = {}
    pm = torch.cat([(main[:, i:i + 512] @ head.weight.t()).argmax(-1) for i in range(0, main.shape[1], 512)], 1)
    ok_main = pm[:, :-1] == ids[:, 1:]
    out["main_acc"] = ok_main.float().mean().item()
    if hasattr(model, "mtp"):
        m = mtp_hidden(model, hidden, input_ids, pos_offset).to(hd)
        pt = torch.cat([(m[:, i:i + 512] @ head.weight.t()).argmax(-1) for i in range(0, m.shape[1], 512)], 1)
        ok_mtp = pt[:, :-1] == ids[:, 2:]
        out["mtp_acc"] = ok_mtp.float().mean().item()
        both = ok_main[:, :-1] & ok_mtp
        out["mtp_acc_given_main"] = (both.float().sum() / ok_main[:, :-1].float().sum().clamp_min(1)).item()
    return out


def mtp_extra_loss(weight: float = 0.3, pos_offset: int = 1, chunk: int = 1024, checkpoint: bool = True):
    """``extra_loss`` for :func:`bitlattice.parallel.train_step`: the MTP head trained jointly with the model.
    ``checkpoint``: recompute the MTP block in backward (its loss over the vocabulary is not recomputed)."""
    from torch.utils.checkpoint import checkpoint as ckpt

    from ..nn.loss import linear_cross_entropy

    def fn(model, hidden, input_ids, labels, next_embeds):
        if checkpoint and torch.is_grad_enabled():
            m = ckpt(lambda h, e: mtp_hidden(model, h, input_ids, pos_offset, e), hidden, next_embeds,
                     use_reentrant=False)
        else:
            m = mtp_hidden(model, hidden, input_ids, pos_offset, next_embeds)
        head = model.get_output_embeddings()
        return weight * linear_cross_entropy(m[:, :-1], head.weight, labels[:, 2:].to(head.weight.device), None,
                                             chunk)
    return fn
