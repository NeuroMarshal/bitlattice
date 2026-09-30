"""Chunked linear + cross-entropy against the dense computation."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_matches_dense(quant, bias):
    torch.manual_seed(0)
    h = torch.randn(2, 37, 64, device=DEV, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(500, 64, device=DEV) * 0.1)
    b = torch.nn.Parameter(torch.randn(500, device=DEV) * 0.1) if bias else None
    if quant:
        w = torch.nn.Parameter(LatticeTensor.from_float(w.detach(), "int8", dtype=torch.float32))
    y = torch.randint(0, 500, (2, 37), device=DEV)
    y[0, :5] = -100
    loss = bl.nn.linear_cross_entropy(h, w, y, b, chunk=16)
    loss.backward()
    h2 = h.detach().clone().requires_grad_(True)
    W = (w.dequantize(torch.float32) if quant else w.detach()).clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True) if bias else None
    ref = torch.nn.functional.cross_entropy(torch.nn.functional.linear(h2, W, b2).reshape(-1, 500), y.reshape(-1))
    ref.backward()
    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.allclose(h.grad, h2.grad, atol=1e-5)
    assert torch.allclose(w.grad.float(), W.grad, atol=1e-5)
    if bias:
        assert torch.allclose(b.grad, b2.grad, atol=1e-5)


def test_peak_memory_below_full_logits():
    torch.manual_seed(0)
    n, d, v = 4096, 256, 32000
    h = torch.randn(n, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(v, d, device=DEV, dtype=torch.bfloat16) * 0.02)
    y = torch.randint(0, v, (n,), device=DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    bl.nn.linear_cross_entropy(h, w, y, chunk=512).backward()
    peak = torch.cuda.max_memory_allocated() - base
    h.grad = w.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    torch.nn.functional.cross_entropy(torch.nn.functional.linear(h, w).float(), y).backward()
    dense = torch.cuda.max_memory_allocated() - base
    assert peak < 0.35 * dense, (peak, dense)
