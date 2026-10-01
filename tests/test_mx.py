"""MX (E8M0-scaled FP8) quantization kernel against the reference quantizer."""
import pytest
import torch

import bitlattice as bl
from bitlattice import formats as F, reference as R

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and bl.HAVE_EXT), reason="needs the CUDA extension")
DEV = "cuda"


def _dequant(codes, scales, e5m2=False):
    v = codes.float()
    s = torch.exp2(scales.view(torch.uint8).float() - 127)
    rows, cols = v.shape
    return (v.view(rows, cols // 32, 32) * s.view(rows, cols // 32, 1)).view(rows, cols)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("e5m2", [False, True])
def test_mx_dual_matches_reference(dtype, e5m2):
    torch.manual_seed(0)
    x = (torch.randn(256, 384, device=DEV) * torch.exp(torch.randn(256, 1, device=DEV))).to(dtype)
    x[0, :32] = 0.0                                       # an all-zero block
    rc, rs, cc, cs = torch.ops.bitlattice.mx_quantize2(x, True, True, e5m2)
    fmt = F.get("mxfp8_e5m2" if e5m2 else "mxfp8")
    f8 = torch.float8_e5m2 if e5m2 else torch.float8_e4m3fn
    for codes, scales, src in ((rc, rs, x.float()), (cc, cs, x.float().t().contiguous())):
        q = R.quantize(src, fmt, method="rtn")              # scales: block maximum / max element, rounded up
        nz = src.view(src.shape[0], -1, 32).abs().amax(-1) > 0     # (the scale of an all-zero block is arbitrary)
        assert torch.equal(q.scales.view(torch.uint8)[nz], scales.view(torch.uint8)[nz])
        s = torch.exp2(scales.view(torch.uint8).float() - 127).repeat_interleave(32, 1)
        ref = (src / s).to(f8).float() * s                   # elements: round to nearest even (OCP MX)
        got = _dequant(codes, scales)
        assert torch.equal(got, ref)
        assert ((got - src).norm() / src.norm()).item() < (0.06 if e5m2 else 0.03)


def test_mx_blocked_scales_and_gemm():
    from bitlattice.backend import to_blocked
    torch.manual_seed(1)
    a = torch.randn(256, 512, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(384, 512, device=DEV, dtype=torch.bfloat16)
    ac, asc, _, _ = torch.ops.bitlattice.mx_quantize2(a, True, False, False, False)
    ab, asb, _, _ = torch.ops.bitlattice.mx_quantize2(a, True, False, False, True)
    assert torch.equal(to_blocked(asc).view(torch.uint8), asb.view(torch.uint8))
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 GEMM needs compute capability >= 10.0")
    bc, bsb, _, _ = torch.ops.bitlattice.mx_quantize2(b, True, False, False, True)
    y = torch._scaled_mm(ac, bc.t(), scale_a=asb, scale_b=bsb, out_dtype=torch.bfloat16)
    ref = a.float() @ b.float().t()
    assert ((y.float() - ref).norm() / ref.norm()).item() < 0.05
