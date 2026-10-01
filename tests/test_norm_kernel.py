"""CUDA RMSNorm kernels against the reference implementation (same per-element arithmetic; row sums may differ in
order)."""
import pytest
import torch

import bitlattice as bl
from bitlattice.nn import norm as N

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and bl.HAVE_EXT), reason="needs the CUDA extension")


def _run(fn, *args):
    args = [a.clone().requires_grad_(a.is_floating_point()) for a in args]
    y = fn(*args)
    y.float().square().mean().backward()
    return [y.float()] + [a.grad.float() for a in args]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,d", [(1, 64), (37, 128), (300, 5120)])
@pytest.mark.parametrize("conv", [(0.0, True), (1.0, False)])
def test_rms_norm_kernel_matches_reference(dtype, rows, d, conv):
    torch.manual_seed(0)
    x = torch.randn(rows, d, device="cuda", dtype=dtype) * 3
    w = (torch.rand(d, device="cuda") + 0.5).to(dtype)
    ref = _run(lambda a, b: N._RMSNorm.apply(a, b, 1e-6, *conv), x, w)
    out = _run(lambda a, b: N.rms_norm(a, b, 1e-6, *conv), x, w)
    for a, b in zip(out, ref):
        assert ((a - b).norm() / b.norm().clamp_min(1e-12)).item() < 2e-3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,d", [(5, 128), (999, 128), (64, 256)])
def test_gated_rms_norm_kernel_matches_reference(dtype, rows, d):
    torch.manual_seed(1)
    x = torch.randn(rows, d, device="cuda", dtype=dtype)
    g = torch.randn(rows, d, device="cuda", dtype=dtype) * 2
    w = (torch.rand(d, device="cuda") + 0.5).to(dtype)
    ref = _run(lambda a, b, c: N._RMSNormGated.apply(a, b, c, 1e-6), x, g, w)
    out = _run(lambda a, b, c: N.rms_norm_gated(a, b, c, 1e-6), x, g, w)
    for a, b in zip(out, ref):
        assert ((a - b).norm() / b.norm().clamp_min(1e-12)).item() < 2e-3


def test_kernel_choice_switches_the_norm_kernels_off():
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    with bl.kernels(fused_norm=False):
        assert not N._kernel(x, w)
    with bl.kernels("reference"):
        assert not N._kernel(x, w)
        with bl.kernels(fused_norm=True):                          # inner settings win
            assert N._kernel(x, w)
    assert N._kernel(x, w)
