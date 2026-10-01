"""NVFP4 quantization kernel against the reference: layouts, round to nearest / stochastic / MS-EDEN, four over
six, rotation, NVFP4 input."""
import pytest
import torch

import bitlattice as bl
from bitlattice import fp4ref as Q

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and bl.HAVE_EXT), reason="needs the CUDA extension")
DEV = "cuda"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _decode(packed, scales, g, rows, cols):
    p = packed.to(torch.int32)
    codes = torch.stack([p & 15, p >> 4], -1).reshape(rows, cols)
    lv = E2M1.to(packed.device)[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return lv, scales.float()


def _data(rows=256, cols=512, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, cols, device=DEV, generator=g) * torch.exp(torch.randn(rows, 1, device=DEV, generator=g))
    x[:, 3] *= 25
    return x.to(torch.bfloat16)


MODES = {"rtn": 0, "sr": 1, "eden": 2}


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mode", ["rtn", "sr", "eden"])
@pytest.mark.parametrize("fos", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_kernel_matches_reference(transpose, mode, fos, rotate):
    if mode == "eden" and not rotate:
        rotate = True
    x = _data()
    src = x.float().t().contiguous() if transpose else x.float()
    v_ref, s_ref, g_ref = Q.quantize(src, mode=mode, four_over_six=fos, seed=11, rotate=rotate)
    packed, scales, g = torch.ops.bitlattice.nvfp4_quantize(x, transpose, MODES[mode], fos, False, rotate, 11)
    v, s = _decode(packed, scales, g, *src.shape)
    assert torch.allclose(g.reshape(()), g_ref.reshape(()), rtol=1e-6)
    # identical up to rare decisions at floating-point ties (summation order of the 4/6 errors and EDEN factors)
    assert (s != s_ref).float().mean().item() < 2e-3
    assert (v != v_ref).float().mean().item() < 2e-3


def test_four_over_six_lowers_error():
    x = _data()
    errs = {}
    for fos in (False, True):
        p, s, g = torch.ops.bitlattice.nvfp4_quantize(x, False, 0, fos, False, False, 0)
        v, sc = _decode(p, s, g, *x.shape)
        d = (v.view(x.shape[0], -1, 16) * (sc * g)[..., None]).reshape(x.shape)
        errs[fos] = ((d - x.float()) ** 2).mean().item()
    assert errs[True] < errs[False]


def test_eden_is_unbiased_and_tighter_than_sr():
    x = _data(64, 256)
    rot = {}
    for mode in ("sr", "eden"):
        acc, mse = 0, 0
        for sd in range(48):
            p, s, g = torch.ops.bitlattice.nvfp4_quantize(x, False, MODES[mode], False, False, True, sd * 13 + 1)
            v, sc = _decode(p, s, g, *x.shape)
            d = (v.view(64, -1, 16) * (sc * g)[..., None]).reshape(64, 256)
            back = Q.rht128(d, sd * 13 + 1, inverse=True)
            acc = acc + back
            mse += ((back - x.float()) ** 2).mean().item()
        rot[mode] = (((acc / 48) - x.float()).norm() / x.float().norm()).item(), mse / 48
    assert rot["eden"][1] < 0.6 * rot["sr"][1]          # about 2x lower error (data dependent)
    assert rot["eden"][0] < rot["sr"][0]


def test_nvfp4_input_transposed():
    w = bl.LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    p, s, g = torch.ops.bitlattice.nvfp4_quantize(w.packed, True, 0, False, False, False, 0, None, w.scales,
                                                  w.global_scale)
    v, sc = _decode(p, s, g, 256, 128)
    d = (v.view(256, -1, 16) * (sc * g)[..., None]).reshape(256, 128)
    ref = w.dequantize(torch.float32).t()
    assert ((d - ref).norm() / ref.norm()).item() < 0.12
