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


def _dequant_fp4(op, rows, cols):
    from bitlattice import formats as F
    q = R.Quantized(op.packed, op.scales, op.global_scale, None, rows, cols)
    return R.dequantize(q, F.get("nvfp4"), torch.float32)


def test_stochastic_fp4_quantizer_is_unbiased():
    torch.manual_seed(3)
    x = torch.randn(32, 256, device=DEV)
    acc = torch.zeros_like(x)
    n = 400
    for _ in range(n):
        acc += _dequant_fp4(bl.backend.quantize_fp4(x, stochastic=True), 32, 256)
    rtn = _dequant_fp4(bl.backend.quantize_fp4(x), 32, 256)
    err_sr = ((acc / n - x).norm() / x.norm()).item()
    err_rtn = ((rtn - x).norm() / x.norm()).item()
    assert err_sr < 0.25 * err_rtn, (err_sr, err_rtn)


def test_hadamard_rotation_preserves_inner_products():
    a, b = torch.randn(8, 64, device=DEV), torch.randn(5, 64, device=DEV)
    assert torch.allclose(bl.backend.rht_last(a) @ bl.backend.rht_last(b).t(), a @ b.t(), atol=1e-4)


@pytest.mark.parametrize("M", [5, 96, 512])
def test_fp4_all_gradients_close_to_exact(M):
    torch.manual_seed(4)
    x = torch.randn(M, 256, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    gy = torch.randn(M, 128, device=DEV, dtype=torch.bfloat16)
    out = {}
    for pol in ("exact", "fp4_all"):
        p = torch.nn.Parameter(w.detach())
        xi = x.clone().requires_grad_(True)
        with bl.compute(pol):
            y = torch.nn.functional.linear(xi, p)
        y.backward(gy)
        out[pol] = (y.detach().float(), xi.grad.float(), p.grad.float())
    for i, name in enumerate(("y", "dx", "dw")):
        ref, got = out["exact"][i], out["fp4_all"][i]
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 0.25, (name, rel)


def test_fp4_all_training_learns_and_saves_fp4_inputs():
    torch.manual_seed(5)
    net = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)).to(DEV).bfloat16()
    bl.quantize_(net, format="nvfp4")
    opt = bl.optim.build(net, "muon", lr=5e-3, fused_backward=True)
    x = torch.randn(512, 128, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(512, 64, device=DEV, dtype=torch.bfloat16) * 0.1
    losses = []
    with bl.compute("fp4_all"):
        for _ in range(60):
            loss = torch.nn.functional.mse_loss(net(x).float(), y.float())
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_cached_weight_operands_follow_updates():
    torch.manual_seed(6)
    lin = torch.nn.Linear(256, 128, bias=False).to(DEV).bfloat16()
    bl.quantize_(lin, format="nvfp4")
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    opt = bl.optim.build(lin, "muon", lr=5e-2)
    with bl.compute("fp4_all"):
        y1 = lin(x)
        y1.float().square().mean().backward()
        opt.step()
        opt.zero_grad()
        y2 = lin(x)
        bl.backend._OPERANDS.clear()
        bl.backend._OPERAND_BYTES[0] = 0
        y3 = lin(x)
    assert not torch.equal(y1, y2)
    assert torch.equal(y2, y3)


def test_transposed_requantization_matches_dense_transpose():
    torch.manual_seed(7)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    p, s, g = torch.ops.bitlattice.requantize_nvfp4_t(w.packed, w.scales, w.global_scale)
    from bitlattice import formats as F
    wt = R.dequantize(R.Quantized(p, s, g, None, 256, 128), F.get("nvfp4"), torch.float32)
    ref = w.dequantize(torch.float32).t()
    assert ((wt - ref).norm() / ref.norm()).item() < 0.12


RECIPES = ("nvidia", "4over6", "quartet2", "quartet2_4o6", "quartet2_fast", "quartet2_delayed")


def _grads(x, w, gy, policy, recipe):
    p = torch.nn.Parameter(w.detach())
    xi = x.clone().requires_grad_(True)
    with bl.compute(policy), bl.fp4_recipe(recipe):
        y = torch.nn.functional.linear(xi, p)
    y.backward(gy)
    return y.detach().float(), xi.grad.float(), p.grad.float()


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("M", [5, 96, 512])
def test_recipes_gradients_close_to_exact(recipe, M):
    torch.manual_seed(4)
    x = torch.randn(M, 256, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    gy = torch.randn(M, 128, device=DEV, dtype=torch.bfloat16)
    ref = _grads(x, w, gy, "exact", "nvidia")
    got = _grads(x, w, gy, "fp4_all", recipe)
    for name, r, g in zip(("y", "dx", "dw"), ref, got):
        rel = ((g - r).norm() / r.norm()).item()
        assert rel < 0.25, (recipe, name, rel)


@pytest.mark.parametrize("recipe", ["quartet2", "quartet2_4o6"])
def test_eden_recipes_are_unbiased(recipe):
    torch.manual_seed(8)
    M = 256
    x = torch.randn(M, 256, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    gy = torch.randn(M, 128, device=DEV, dtype=torch.bfloat16)
    _, rdx, rdw = _grads(x, w, gy, "exact", "nvidia")
    n, sdx, sdw, one = 48, 0, 0, None
    for i in range(n):
        _, dx, dw = _grads(x, w, gy, "fp4_all", recipe)
        sdx, sdw = sdx + dx, sdw + dw
        if one is None:
            one = (((dx - rdx).norm() / rdx.norm()).item(), ((dw - rdw).norm() / rdw.norm()).item())
    mean_dx = ((sdx / n - rdx).norm() / rdx.norm()).item()
    mean_dw = ((sdw / n - rdw).norm() / rdw.norm()).item()
    assert mean_dx < 0.35 * one[0], (mean_dx, one)
    assert mean_dw < 0.35 * one[1], (mean_dw, one)


def test_recipe_selection_and_structural_nvfp4():
    from bitlattice import formats as F
    assert bl.backend.is_nvfp4(F.get("nvfp4")) and bl.backend.is_nvfp4(F.get("nvfp4_4o6"))
    assert not bl.backend.is_nvfp4(F.get("mxfp4")) and not bl.backend.is_nvfp4(F.get("int4"))
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4_4o6")
    assert bl.backend.get_fp4_recipe(w).legacy
    w.fp4_recipe = "quartet2"
    assert bl.backend.get_fp4_recipe(w) == bl.backend.FP4_RECIPES["quartet2"]
    with bl.fp4_recipe("4over6"):
        assert bl.backend.get_fp4_recipe(w) == bl.backend.FP4_RECIPES["4over6"]
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    assert bl.backend.fp4_policy(x, w) == "exact" or bl.backend.fp4_policy(x, w) in ("fp4", "fp4_all")
    with bl.compute("fp4_all"):
        assert bl.backend.fp4_policy(x, w) == "fp4_all"
    with pytest.raises(ValueError):
        bl.backend.Fp4Recipe(gradients="eden", rotation=16)


def test_fp4_recipes_train():
    for recipe in RECIPES:
        torch.manual_seed(5)
        net = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 128)).to(DEV)
        net = net.bfloat16()
        bl.quantize_(net, format="nvfp4")
        opt = bl.optim.build(net, "muon", lr=5e-3, fused_backward=True)
        x = torch.randn(512, 128, device=DEV, dtype=torch.bfloat16)
        y = torch.randn(512, 128, device=DEV, dtype=torch.bfloat16) * 0.1
        losses = []
        with bl.compute("fp4_all"), bl.fp4_recipe(recipe):
            for _ in range(60):
                loss = torch.nn.functional.mse_loss(net(x).float(), y.float())
                loss.backward()
                opt.step()
                losses.append(loss.item())
        assert losses[-1] < 0.8 * losses[0], (recipe, losses[0], losses[-1])


@pytest.mark.parametrize("recipe", ["nvidia", "quartet2_fast"])
def test_consumers_share_one_input_quantization(recipe):
    torch.manual_seed(9)
    x = torch.randn(256, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    ws = [torch.nn.Parameter(LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4"))
          for _ in range(3)]
    calls = {"n": 0}
    orig = bl.backend._input_operands

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    bl.backend._input_operands = counting
    try:
        with bl.compute("fp4_all"), bl.fp4_recipe(recipe):
            ys = [torch.nn.functional.linear(x, w) for w in ws]
            sum(y.float().square().mean() for y in ys).backward()
    finally:
        bl.backend._input_operands = orig
    assert calls["n"] == 1
    assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in ws)
    del ys
    import gc
    gc.collect()
    x2 = x.detach().clone()
    with bl.compute("fp4_all"), bl.fp4_recipe(recipe):
        torch.nn.functional.linear(x2, ws[0])
    n = len(bl.backend._SHARED)
    del x2
    gc.collect()
    assert len(bl.backend._SHARED) == n - 1          # the entry died with its input


@pytest.mark.parametrize("recipe", RECIPES)
def test_recipes_with_checkpoint_recomputation(recipe):
    """Non-reentrant checkpointing recomputes the saved operands in backward (new rotations and stochastic draws):
    the gradients must stay consistent with them."""
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(10)
    M = 256
    x = torch.randn(M, 256, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4", method="rtn")
    gy = torch.randn(M, 128, device=DEV, dtype=torch.bfloat16)
    _, rdx, rdw = _grads(x, w, gy, "exact", "nvidia")
    p = torch.nn.Parameter(w.detach())
    xi = x.clone().requires_grad_(True)
    with bl.compute("fp4_all"), bl.fp4_recipe(recipe):            # recomputation runs in backward: keep it inside
        # an op before the layer: its input is a new tensor in the recomputation (as a norm output in a block)
        y = checkpoint(lambda t: torch.nn.functional.linear(t * 1.0, p), xi, use_reentrant=False)
        y.backward(gy)
    for name, r, g in (("dx", rdx, xi.grad.float()), ("dw", rdw, p.grad.float())):
        rel = ((g - r).norm() / r.norm()).item()
        assert rel < 0.25, (recipe, name, rel)


def test_delayed_scales_follow_the_previous_call():
    torch.manual_seed(11)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4")
    p = torch.nn.Parameter(w.detach())                 # one parameter: its sites carry the amax between calls
    rel = []
    for scale in (1.0, 1.0, 1.0, 1.5):                 # the last call exceeds the previous amax, within the margin
        x = (torch.randn(256, 256, device=DEV) * scale).to(torch.bfloat16)
        gy = torch.randn(256, 128, device=DEV, dtype=torch.bfloat16)
        _, rdx, rdw = _grads(x, w, gy, "exact", "nvidia")
        p.grad = None
        xi = x.clone().requires_grad_(True)
        with bl.compute("fp4_all"), bl.fp4_recipe("quartet2_delayed"):
            torch.nn.functional.linear(xi, p).backward(gy)
        dx, dw = xi.grad.float(), p.grad.float()
        rel.append(max(((dx - rdx).norm() / rdx.norm()).item(), ((dw - rdw).norm() / rdw.norm()).item()))
    assert p._bl_sites["x"]["known"] == (True, True) and p._bl_sites["g"]["known"] == (True, True)
    assert max(rel) < 0.25, rel


@pytest.mark.parametrize("recipe", ["quartet2", "quartet2_fast"])
def test_deferred_weight_gradients_match_accumulated(recipe):
    torch.manual_seed(12)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4")
    xs = [(torch.randn(128, 256, device=DEV) * s).to(torch.bfloat16) for s in (1.0, 3.0, 0.5)]
    gys = [torch.randn(128, 128, device=DEV, dtype=torch.bfloat16) for _ in xs]
    ref = sum(_grads(x, w, gy, "exact", "nvidia")[2] for x, gy in zip(xs, gys))
    p = torch.nn.Parameter(w.detach())
    with bl.compute("fp4_all"), bl.fp4_recipe(recipe):
        with bl.backend.defer_wgrad():
            for x, gy in zip(xs, gys):
                torch.nn.functional.linear(x.clone().requires_grad_(True), p).backward(gy)
        assert p.grad is None and bl.backend.pending_wgrad() == [p]
        assert bl.backend.flush_wgrad() == 1
    rel = ((p.grad.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.25, rel
    assert not bl.backend.pending_wgrad()



@pytest.mark.skipif(not (torch.cuda.is_available() and bl.backend.mxfp8_supported(0)), reason="needs MXFP8 GEMMs")
@pytest.mark.parametrize("fmt", ["int4", "nvfp4", "mxfp4", "int8", "fp6_e3m2_g32_se4m3_gs"])
def test_mxfp8_all_any_format(fmt):
    torch.manual_seed(13)
    M = 200
    x = torch.randn(M, 256, device=DEV, dtype=torch.bfloat16)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, fmt, dtype=torch.bfloat16)
    gy = torch.randn(M, 128, device=DEV, dtype=torch.bfloat16)
    ref = _grads(x, w, gy, "exact", "nvidia")
    got = _grads(x, w, gy, "mxfp8_all", "nvidia")
    for name, r, g in zip(("y", "dx", "dw"), ref, got):
        rel = ((g - r).norm() / r.norm()).item()
        assert rel < 0.06, (fmt, name, rel)


@pytest.mark.skipif(not (torch.cuda.is_available() and bl.backend.mxfp8_supported(0)), reason="needs MXFP8 GEMMs")
def test_auto_policy_routes_by_format():
    lin = torch.nn.Sequential(torch.nn.Linear(256, 128, bias=False), torch.nn.Linear(128, 128, bias=False)).cuda()
    bl.quantize_(lin, format="nvfp4", rules=[bl.Rule("1", {"format": "int4"})])
    with bl.compute("auto"):
        routes = {n: r for n, f, r in bl.backend.explain(lin)}
    assert routes["0.weight"] == "fp4_all" and routes["1.weight"] == "mxfp8_all", routes



def test_deferred_weight_gradients_with_delayed_scales():
    torch.manual_seed(14)
    w = LatticeTensor.from_float(torch.randn(128, 256, device=DEV) * 0.05, "nvfp4")
    p = torch.nn.Parameter(w.detach())
    for step in range(3):                              # the first step is exact; later ones use delayed scales
        xs = [(torch.randn(128, 256, device=DEV) * s).to(torch.bfloat16) for s in (1.0, 1.3)]
        gys = [torch.randn(128, 128, device=DEV, dtype=torch.bfloat16) for _ in xs]
        ref = sum(_grads(x, w, gy, "exact", "nvidia")[2] for x, gy in zip(xs, gys))
        p.grad = None
        with bl.compute("fp4_all"), bl.fp4_recipe("quartet2_delayed"):
            with bl.backend.defer_wgrad():
                for x, gy in zip(xs, gys):
                    torch.nn.functional.linear(x.clone().requires_grad_(True), p).backward(gy)
            for _, items in bl.backend._PENDING.values():   # the operands to concatenate: power-of-two scales
                for gt, xt in items:
                    for g in (gt.global_scale, xt.global_scale):
                        assert torch.equal(g, torch.exp2(torch.round(torch.log2(g))))
            bl.backend.flush_wgrad()
        rel = ((p.grad.float() - ref).norm() / ref.norm()).item()
        assert rel < 0.25, (step, rel)
