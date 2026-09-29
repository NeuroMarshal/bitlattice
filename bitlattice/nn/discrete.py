"""Discrete activations trained without differentiating the discrete operator.

kinds: 'sign' ({-1, 1}), 'step' ({0, 1}), 'ternary' ({-1, 0, 1}), 'int<k>' (2^k uniform levels in [-1, 1]).
modes:
    'expect'      stochastic forward (uniform dither of width 2r for thresholds, stochastic rounding between levels);
                  backward = exact derivative of the expectation (secant slope, bounded by 1/(2r))
    'expect_det'  deterministic forward, expectation derivative in backward
    'ste'         deterministic forward, clipped straight-through estimator
    'true'        deterministic forward, true derivative (zero almost everywhere); for comparison only
Evaluation mode is always deterministic.
"""
from __future__ import annotations

import torch

__all__ = ["DiscreteAct", "discrete"]

KINDS = ("sign", "step", "ternary")
MODES = ("expect", "expect_det", "ste", "true")


def _levels(kind: str, device) -> torch.Tensor:
    if kind == "sign":
        return torch.tensor([-1.0, 1.0], device=device)
    if kind == "step":
        return torch.tensor([0.0, 1.0], device=device)
    if kind == "ternary":
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    if kind.startswith("int") and kind[3:].isdigit():
        return torch.linspace(-1.0, 1.0, 1 << int(kind[3:]), device=device)
    raise ValueError(f"unknown discrete kind {kind!r}")


def _threshold(kind: str) -> bool:
    return kind in ("sign", "step")


class _Discrete(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kind: str, mode: str, r: float, training: bool):
        lv = _levels(kind, x.device).to(x.dtype)
        if training and mode == "expect":
            if _threshold(kind):
                u = (torch.rand_like(x) * 2 - 1) * r
                y = torch.where(x + u > 0, lv[1], lv[0])
            else:
                xc = x.clamp(lv[0], lv[-1])
                idx = torch.bucketize(xc, lv, right=True).clamp(1, lv.numel() - 1)
                a, b = lv[idx - 1], lv[idx]
                y = torch.where(torch.rand_like(x) < (xc - a) / (b - a), b, a)
        elif _threshold(kind):
            y = torch.where(x > 0, lv[1], lv[0])
        else:
            mid = (lv[1:] + lv[:-1]) / 2
            y = lv[torch.bucketize(x.clamp(lv[0], lv[-1]), mid)]
        ctx.save_for_backward(x)
        ctx.kind, ctx.mode, ctx.r = kind, mode, r
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        lv = _levels(ctx.kind, x.device).to(x.dtype)
        if ctx.mode == "true":
            return torch.zeros_like(x), None, None, None, None
        if _threshold(ctx.kind):
            if ctx.mode == "ste":
                slope = (x.abs() < ctx.r).to(gy.dtype)
            else:
                slope = (x.abs() < ctx.r).to(gy.dtype) * float(lv[1] - lv[0]) / (2 * ctx.r)
        else:
            slope = ((x > lv[0]) & (x < lv[-1])).to(gy.dtype)
        return gy * slope, None, None, None, None


def discrete(x: torch.Tensor, kind: str = "sign", mode: str = "expect", r: float = 1.0,
             training: bool = True) -> torch.Tensor:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return _Discrete.apply(x, kind, mode, r, training)


class DiscreteAct(torch.nn.Module):
    def __init__(self, kind: str = "sign", mode: str = "expect", r: float = 1.0):
        super().__init__()
        _levels(kind, "cpu")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.kind, self.mode, self.r = kind, mode, float(r)

    def forward(self, x):
        return _Discrete.apply(x, self.kind, self.mode, self.r, self.training)

    def extra_repr(self) -> str:
        return f"kind={self.kind}, mode={self.mode}, r={self.r}"
