"""fused_block conversion of Qwen3.5 decoder layers: agreement with the original layers, CUDA graph replays, SwiGLU
kernels (needs transformers with Qwen3.5 and fla)."""
import copy

import pytest
import torch

import bitlattice as bl
from bitlattice import backend

transformers = pytest.importorskip("transformers")
pytest.importorskip("fla")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PROJ = r"re:layers\.\d+\..*(in_proj_qkv|in_proj_z|out_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def _model():
    cfg = transformers.Qwen3_5TextConfig(hidden_size=256, intermediate_size=768, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=64,
                                         linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64,
                                         linear_value_head_dim=64, layer_types=["linear_attention", "full_attention"],
                                         vocab_size=256)
    torch.manual_seed(0)
    m = M.Qwen3_5TextModel(cfg).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 1:
                p.add_(torch.randn_like(p) * 0.1)        # norms, decay, gates away from their initial values
    return m


def _inputs(m, T=200):
    ids = torch.randint(0, 256, (2, T), device="cuda")
    return bl.parallel._block_inputs(m, m.layers, ids, {})


def _run(layer, kw, x, gy, defer=False):
    xi = x.detach().requires_grad_(True)
    y = layer(xi, **kw)
    y = y[0] if isinstance(y, tuple) else y
    with backend.defer_wgrad() if defer else torch.enable_grad():
        y.backward(gy)
    if defer:
        backend.flush_wgrad()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    for p in layer.parameters():
        p.grad = None
    return y.detach(), xi.grad, grads


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def test_fused_block_matches_the_layers():
    base = _model()
    x0, kws = _inputs(base)
    ref = copy.deepcopy(base)
    bl.surgery.convert_(ref, "*.linear_attn", "gated_delta_net")       # the same delta-rule kernels
    fused = copy.deepcopy(base)
    done = bl.surgery.convert_(fused, r"re:layers\.\d+$", "fused_block")
    assert [d[2] for d in done] == ["FusedBlock", "FusedBlock"]
    for i in range(2):
        torch.manual_seed(i)
        x = x0 + torch.randn_like(x0)
        gy = torch.randn_like(x0) * 0.1
        ya, xa, ga = _run(ref.layers[i], kws[i], x, gy)
        yb, xb, gb = _run(fused.layers[i], kws[i], x, gy)
        assert set(ga) == set(gb)
        assert _rel(yb, ya) < 1e-2 and _rel(xb, xa) < 2e-2
        for n in ga:
            # the decay parameters' gradients are sums with heavy cancellation over all tokens
            assert _rel(gb[n], ga[n]) < (0.15 if n.endswith(("A_log", "dt_bias")) else 3e-2), n


@pytest.mark.parametrize("compute", ["exact", "quartet2_fast"])
def test_graph_replays_match_eager(compute):
    if compute != "exact" and not backend.fp4_supported(0):
        pytest.skip("needs FP4 tensor cores")
    base = _model()
    bl.surgery.quantize_(base, PROJ, "nvfp4")
    x0, kws = _inputs(base, T=256)
    runs = []
    for graphs in (False, True):
        m = copy.deepcopy(base)
        bl.surgery.convert_(m, r"re:layers\.\d+$", "fused_block", graphs=graphs)
        bl.surgery.set_compute_(m, "*", compute, skip_unfit=True)
        out = []
        for i in range(2):
            for it in range(3):                               # eager warm-up, capture, replay
                torch.manual_seed(it)
                x = x0 + torch.randn_like(x0)
                gy = torch.randn_like(x0) * 0.1
                out.append(_run(m.layers[i], kws[i], x, gy, defer=compute != "exact"))
        runs.append(out)
    for (ya, xa, ga), (yb, xb, gb) in zip(*runs):
        assert set(ga) == set(gb)
        assert _rel(yb, ya) == 0.0                            # forward: the same kernels, deterministic rounding
        if compute == "exact":
            # the convolution's weight gradient is summed over time tiles with atomics (order not fixed)
            bad = {n: _rel(gb[n], ga[n]) for n in ga if _rel(gb[n], ga[n]) > (1e-4 if "conv1d" in n else 0.0)}
            assert _rel(xb, xa) == 0.0 and not bad, bad
        else:                                                 # independent stochastic draws, same error level
            assert _rel(xb, xa) < 0.15 and all(_rel(gb[n], ga[n]) < 0.6 for n in ga if not n.endswith(
                ("A_log", "dt_bias")))


def test_swiglu_kernels_match_autograd():
    g = torch.randn(64, 1000, device="cuda", dtype=torch.bfloat16) * 3
    u = torch.randn_like(g)
    da = torch.randn_like(g)
    gr, ur = g.clone().requires_grad_(True), u.clone().requires_grad_(True)
    a = torch.nn.functional.silu(gr) * ur
    a.backward(da)
    assert torch.equal(torch.ops.bitlattice.swiglu_fwd(g, u), a.detach())
    dg, du = torch.ops.bitlattice.swiglu_bwd(g, u, da)
    assert torch.equal(du, ur.grad)
    assert (dg.float() - gr.grad.float()).abs().max().item() <= 1e-2 * gr.grad.float().abs().max().item()
