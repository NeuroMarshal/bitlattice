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


def test_compressed_saves_conversion():
    model = torch.nn.Sequential(Block(), Block()).to(DEV).to(torch.bfloat16)
    x = torch.randn(2, 256, 256, device=DEV, dtype=torch.bfloat16)
    ref = _grads(model, x)
    rep = bl.surgery.convert_(model, "1", "compressed_saves", fmt="fp8", min_numel=1024)
    assert rep == [("1", "Block", "CompressedSaves")]
    bl.memory.stats(reset=True)
    got = _grads(model, x)
    assert bl.memory.stats()["packed"] > 0
    for r, g in zip(ref, got):
        assert ((g - r).norm() / r.norm()).item() < 0.03
    assert bl.surgery.revert_(model, "1") == [("1", "CompressedSaves", "Block")]


def test_checkpoint_conversion_with_compressed_recomputation():
    torch.manual_seed(2)
    model = torch.nn.Sequential(*[Block(512) for _ in range(4)]).to(DEV).to(torch.bfloat16)
    x = torch.randn(4, 1024, 512, device=DEV, dtype=torch.bfloat16)

    def run(grad_input=True):
        model.zero_grad()
        xi = x.clone().requires_grad_(grad_input)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        y = model(xi)
        kept = torch.cuda.memory_allocated() - base
        y.float().square().mean().backward()
        return kept, ([xi.grad.float()] if grad_input else []) + [p.grad.float() for p in model.parameters()]

    k0, ref = run()
    bl.surgery.convert_(model, "re:^[0-9]+$", "checkpoint")
    k1, g1 = run()
    assert k1 < 0.2 * k0                                       # the forward keeps only the layer inputs
    for r, a in zip(ref, g1):
        assert ((a - r).norm() / r.norm()).item() < 1e-2
    _, g_frozen_input = run(grad_input=False)                  # parameters still get gradients
    for r, a in zip(ref[1:], g_frozen_input):
        assert ((a - r).norm() / r.norm()).item() < 1e-2
    bl.surgery.revert_(model, "*")
    bl.memory.stats(reset=True)
    bl.surgery.convert_(model, "re:^[0-9]+$", "checkpoint", saved="fp8", min_numel=1024)
    _, g2 = run()
    st = bl.memory.stats()
    assert st["packed"] > 0 and st["bytes_out"] < 0.55 * st["bytes_in"]   # recomputed saves packed
    for r, b in zip(ref, g2):
        assert ((b - r).norm() / r.norm()).item() < 0.05


def test_skip_unfit_conversions():
    model = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64, 64)).to(DEV)
    with pytest.raises(ValueError, match="cannot be converted"):
        bl.surgery.convert_(model, "*", "rms_norm")
    assert bl.surgery.convert_(model, "re:^[0-9]$", "rms_norm", skip_unfit=True) == []
