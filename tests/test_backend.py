"""FP4 compute path (block-scaled NVFP4 GEMM); skipped on devices without FP4 tensor cores."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor, reference as R

DEV = "cuda"
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and bl.backend.fp4_supported(0)),
                                reason="needs FP4 tensor cores")


def test_activation_quantizer_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    g = (x.abs().amax().float() / (6 * 448)).reshape(1)
    packed, scales = torch.ops.bitlattice.quantize_nvfp4(x, g)
    q = R.quantize(x.float(), "nvfp4", method="rtn")
    assert torch.equal(scales.view(torch.uint8), q.scales.view(torch.uint8))
    assert torch.equal(packed, q.packed)


@pytest.mark.parametrize("M", [1, 7, 256])
def test_fp4_linear_matches_quantized_reference(M):
    torch.manual_seed(1)
    x = torch.randn(M, 512, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(384, 512, device=DEV) * 0.05, "nvfp4", method="rtn",
                                 dtype=torch.bfloat16)
    ref = R.dequantize(R.quantize(x.float(), "nvfp4", method="rtn"), "nvfp4") @ w.dequantize(torch.float32).t()
    with bl.compute("fp4"):
        y = torch.nn.functional.linear(x, torch.nn.Parameter(w))
    assert ((y.float() - ref).norm() / ref.norm()) < 1e-2


def test_fp4_training_step_runs_and_learns():
    torch.manual_seed(2)
    net = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)).to(DEV).bfloat16()
    bl.quantize_(net, format="nvfp4")
    opt = bl.optim.build(net, "muon", lr=5e-3)
    x = torch.randn(512, 128, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(512, 64, device=DEV, dtype=torch.bfloat16) * 0.1
    losses = []
    with bl.compute("fp4"):
        for _ in range(60):
            loss = torch.nn.functional.mse_loss(net(x).float(), y.float())
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]
