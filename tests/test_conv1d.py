"""Causal depthwise convolution kernel against the PyTorch reference."""
import pytest
import torch

import bitlattice as bl
from bitlattice.nn.conv import _reference

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and bl.HAVE_EXT), reason="needs the CUDA extension")
DEV = "cuda"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("W", [2, 4, 7])
@pytest.mark.parametrize("act", [None, "silu"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("T", [1, 37, 300])
def test_matches_reference(dtype, W, act, bias, T):
    torch.manual_seed(0)
    B, C = 2, 96
    base = torch.randn(B, T, C, device=DEV, dtype=dtype)               # channels last, as the layers produce it
    w = (torch.randn(C, W, device=DEV) * 0.3).requires_grad_(True)
    b = (torch.randn(C, device=DEV) * 0.1).requires_grad_(True) if bias else None
    dy = torch.randn(B, C, T, device=DEV, dtype=dtype)
    outs = []
    for fn in (bl.nn.causal_conv1d, _reference):
        x = base.clone().requires_grad_(True)
        y = fn(x.transpose(1, 2), w, b, act)
        y.backward(dy)
        outs.append((y.float(), x.grad.float(), w.grad.clone(), None if b is None else b.grad.clone()))
        w.grad = None
        if b is not None:
            b.grad = None
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    for got, ref in zip(outs[0], outs[1]):
        if ref is None:
            continue
        assert ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item() < tol


def test_no_copy_for_channels_last_input():
    x = torch.randn(1, 64, 32, device=DEV, dtype=torch.bfloat16).transpose(1, 2)   # [B, C, T] view
    y = bl.nn.causal_conv1d(x, torch.randn(32, 4, device=DEV), None, "silu")
    assert y.shape == x.shape and y.transpose(1, 2).is_contiguous()
