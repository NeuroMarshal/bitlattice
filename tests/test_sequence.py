"""Linear-attention recurrences over sequence chunks: same outputs, state and gradients."""
import pytest
import torch

import bitlattice as bl

fla = pytest.importorskip("fla.ops.gated_delta_rule")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


@pytest.mark.parametrize("T", [300, 1024])
def test_chunked_gated_delta_rule_matches(T):
    torch.manual_seed(0)
    B, H, HV, K, V = 1, 2, 4, 64, 64
    fn = fla.chunk_gated_delta_rule
    base = dict(q=torch.randn(B, T, H, K, device=DEV, dtype=torch.bfloat16),
                k=torch.randn(B, T, H, K, device=DEV, dtype=torch.bfloat16),
                v=torch.randn(B, T, HV, V, device=DEV, dtype=torch.bfloat16),
                g=-torch.rand(B, T, HV, device=DEV).clamp_min(0.01),
                beta=torch.rand(B, T, HV, device=DEV).to(torch.bfloat16))
    h0 = torch.randn(B, HV, K, V, device=DEV) * 0.1
    res = []
    for f in (fn, bl.nn.sequence_chunked(fn, 128)):
        t = {n: x.clone().requires_grad_(True) for n, x in base.items()}
        s0 = h0.clone().requires_grad_(True)
        o, s = f(t["q"], t["k"], t["v"], g=t["g"], beta=t["beta"], initial_state=s0, output_final_state=True,
                 use_qk_l2norm_in_kernel=True)
        (o.float().square().mean() + s.square().mean()).backward()
        res.append([o.float(), s.float()] + [t[n].grad.float() for n in base] + [s0.grad.float()])
    for a, b in zip(*res):
        assert ((a - b).norm() / b.norm().clamp_min(1e-9)).item() < 2e-2
