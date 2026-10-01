"""Compressed saved activations: gradients close to the uncompressed ones, fp32 and parameters untouched, works
inside activation checkpointing."""
import pytest
import torch
from torch.utils.checkpoint import checkpoint

import bitlattice as bl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


class Block(torch.nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.up = torch.nn.Linear(d, 4 * d)
        self.down = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.up(self.norm(x))
        return x + self.down(torch.nn.functional.silu(h) * torch.sigmoid(h.float()).to(h.dtype))


def _grads(model, x, ctx=None, ckpt=False):
    model.zero_grad()
    xi = x.clone().requires_grad_(True)
    f = (lambda t: checkpoint(model, t, use_reentrant=False)) if ckpt else model
    if ctx is not None:
        with ctx:
            y = f(xi)
    else:
        y = f(xi)
    y.float().square().mean().backward()
    return [xi.grad.float()] + [p.grad.float() for p in model.parameters()]


@pytest.mark.parametrize("fmt,tol", [("fp8", 0.03), ("nvfp4", 0.15)])
@pytest.mark.parametrize("ckpt", [False, True])
def test_gradients_close(fmt, tol, ckpt):
    torch.manual_seed(0)
    model = Block().to(DEV).to(torch.bfloat16)
    x = torch.randn(4, 512, 256, device=DEV, dtype=torch.bfloat16)
    ref = _grads(model, x)
    if ckpt:                                           # hooks inside the checkpointed forward
        inner = model.forward
        model.forward = lambda t: _with(fmt, inner, t)
        got = _grads(model, x, ckpt=True)
        model.forward = inner
    else:
        bl.memory.stats(reset=True)
        got = _grads(model, x, bl.memory.compress_saved(bl.memory.SavedSpec(fmt, min_numel=1024)))
        st = bl.memory.stats()
        assert st["packed"] > 0 and st["bytes_out"] < 0.6 * st["bytes_in"]
    for r, g in zip(ref, got):
        assert ((g - r).norm() / r.norm()).item() < tol


def _with(fmt, fn, t):
    with bl.memory.compress_saved(bl.memory.SavedSpec(fmt, min_numel=1024)):
        return fn(t)


def test_fp32_and_parameters_kept():
    w = torch.nn.Parameter(torch.randn(64, 64, device=DEV, dtype=torch.bfloat16))
    a = torch.randn(4096, 64, device=DEV, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4096, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    bl.memory.stats(reset=True)
    with bl.memory.compress_saved(bl.memory.SavedSpec("fp8", min_numel=1024)):
        y = (a.exp().sum() + (b.exp() @ w).float().sum())
    st = bl.memory.stats()
    y.backward()
    assert st["packed"] == 1                           # only the bf16 activation exp(b); a.exp() (fp32) and w kept
    assert torch.isfinite(w.grad).all() and torch.isfinite(a.grad).all()


def test_compress_saved_patches_layers():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Sequential(torch.nn.Linear(8, 8)))
    names = bl.memory.compress_saved_(model, "1")
    assert names == ["1"]
    assert bl.memory.compress_saved_(model, "1") == []          # already wrapped
