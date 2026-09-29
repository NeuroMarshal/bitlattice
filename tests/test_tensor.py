"""LatticeTensor behaviour inside ordinary PyTorch code."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _lt(fmt="nvfp4", shape=(64, 128), keep=0.0, dtype=torch.bfloat16):
    torch.manual_seed(0)
    w = torch.randn(*shape, device=DEV) * 0.05
    return w, LatticeTensor.from_float(w, fmt, keep=keep, dtype=dtype)


@pytest.mark.parametrize("fmt", ["nvfp4", "int4", "mxfp4", "fp8", "ternary"])
def test_linear_forward_backward_matches_dense(fmt):
    _, lt = _lt(fmt, dtype=torch.float32)
    W = lt.dequantize(torch.float32).detach().requires_grad_(True)
    p = torch.nn.Parameter(lt)
    x = torch.randn(8, 5, 128, device=DEV, requires_grad=True)
    b = torch.randn(64, device=DEV, requires_grad=True)
    y = torch.nn.functional.linear(x, p, b)
    x2 = x.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    y2 = torch.nn.functional.linear(x2, W, b2)
    assert torch.allclose(y, y2, atol=1e-5)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g)
    assert type(p.grad) is torch.Tensor and p.grad.shape == p.shape
    assert torch.allclose(p.grad, W.grad, atol=1e-4)
    assert torch.allclose(x.grad, x2.grad, atol=1e-5)
    assert torch.allclose(b.grad, b2.grad, atol=1e-5)


def test_keep_columns_are_exact():
    w, lt = _lt("int4", keep=0.25, dtype=torch.float32)
    assert lt.hi_idx.numel() == 32
    d = lt.dequantize(torch.float32)
    assert torch.equal(d[:, lt.hi_idx], w[:, lt.hi_idx].to(torch.bfloat16).float())
    err_keep = (d - w).norm() / w.norm()
    err_none = (LatticeTensor.from_float(w, "int4", dtype=torch.float32).dequantize(torch.float32) - w).norm() / w.norm()
    assert err_keep < err_none


def test_embedding_rows_and_grad():
    _, lt = _lt("nvfp4", shape=(100, 32), dtype=torch.float32)
    p = torch.nn.Parameter(lt)
    idx = torch.tensor([[1, 5, 5], [99, 0, 7]], device=DEV)
    out = torch.nn.functional.embedding(idx, p)
    ref = lt.dequantize(torch.float32)[idx]
    assert torch.allclose(out, ref)
    out.sum().backward()
    expect = torch.zeros(100, 32, device=DEV)
    expect.index_add_(0, idx.reshape(-1), torch.ones(6, 32, device=DEV))
    assert torch.allclose(p.grad, expect)


def test_conv_falls_back_with_autograd():
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(8, 16, 3, padding=1, device=DEV)
    bl.quantize_(conv, bl.QuantConfig(format="int8", targets=(torch.nn.Conv2d,), min_features=8))
    assert isinstance(conv.weight, LatticeTensor)
    x = torch.randn(2, 8, 10, 10, device=DEV)
    y = conv(x)
    ref = torch.nn.functional.conv2d(x, conv.weight.dequantize(torch.float32), conv.bias, padding=1)
    assert torch.allclose(y, ref, atol=1e-5)
    y.sum().backward()
    assert conv.weight.grad is not None and conv.weight.grad.shape == conv.weight.shape


def test_module_to_and_state_dict_roundtrip():
    lin = torch.nn.Linear(128, 64, device=DEV)
    bl.quantize_(lin, format="nvfp4")
    sd = lin.state_dict()
    assert isinstance(sd["weight"], LatticeTensor)
    lin2 = torch.nn.Linear(128, 64, device=DEV)
    bl.quantize_(lin2, format="nvfp4")
    lin2.load_state_dict(sd)
    assert torch.equal(lin2.weight.dequantize(), lin.weight.dequantize())
    cpu = lin.to("cpu")
    assert isinstance(cpu.weight, LatticeTensor) and cpu.weight.packed.device.type == "cpu"
    back = cpu.to(DEV)
    assert torch.equal(back.weight.dequantize(), lin2.weight.dequantize())


def test_load_dense_into_lattice_requantizes():
    lin = torch.nn.Linear(128, 64, device=DEV)
    dense = {k: v.clone() for k, v in lin.state_dict().items()}
    bl.quantize_(lin, format="int8")
    lin.load_state_dict(dense)
    err = (lin.weight.dequantize(torch.float32) - dense["weight"]).norm() / dense["weight"].norm()
    assert err < 0.02


def test_dequantize_model_restores_dense():
    lin = torch.nn.Linear(128, 64, device=DEV)
    bl.quantize_(lin, format="int4")
    ref = lin.weight.dequantize()
    bl.dequantize_(lin)
    assert type(lin.weight.data) is torch.Tensor and torch.equal(lin.weight.data, ref)
