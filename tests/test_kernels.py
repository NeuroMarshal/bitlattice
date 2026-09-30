"""CUDA kernels against the reference implementation."""
import math

import pytest
import torch

from bitlattice import _ops, formats as F, reference as R
from bitlattice.packing import pack, sign_extend, unpack

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and _ops.HAVE_EXT), reason="needs CUDA extension")
DEV = "cuda"
QFORMATS = ["nvfp4", "mxfp4", "fp4", "nf4", "int4", "int3", "int8", "ternary", "binary", "fp8", "fp6_e3m2", "uint4"]


def _weights(rows=64, cols=256, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(rows, cols, device=DEV, generator=g) * 0.02


@pytest.mark.parametrize("name", QFORMATS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dequantize_matches_reference(name, dtype):
    fmt = F.get(name)
    q = R.quantize(_weights(), fmt, method="mse")
    ref = R.dequantize(q, fmt, dtype)
    out = _ops.dequantize(q.packed, q.scales, q.global_scale, q.zeros, fmt, q.rows, q.cols, dtype)
    assert torch.equal(out, ref)


def _phase_packed(phase_i8, bits):
    cb = _ops.phase_container(bits)
    return pack(phase_i8.to(torch.int32) & ((1 << bits) - 1), cb), cb


def _phase_unpacked(packed, bits, cols):
    return sign_extend(unpack(packed, _ops.phase_container(bits), cols), bits).to(torch.int8)


@pytest.mark.parametrize("name", QFORMATS)
@pytest.mark.parametrize("bits", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("rule", ["fixed", "fit", "fit_latent"])
def test_step_quantized_matches_reference(name, bits, rule):
    fmt = F.get(name)
    w = _weights(seed=1)
    q = R.quantize(w, fmt)
    rows, cols = q.rows, q.cols
    g = torch.Generator(device=DEV).manual_seed(2)
    phase = torch.randint(-(1 << max(bits - 1, 0)), 1 << max(bits - 1, 0), (rows, cols), device=DEV,
                          dtype=torch.int8, generator=g) if bits else None
    D = torch.randn(rows, cols, device=DEV, generator=g)
    lr = 0.3 * R.effective_scales(q.scales, fmt.scale, q.global_scale).float().mean().item()
    kw = dict(lr=lr, weight_decay=0.01, phase_bits=bits, rounding="sr", seed=7, step=3, scale_rule=rule,
              trust=0.01, trust_lr=1.0)
    ref_packed, ref_scales, ref_phase = R.lattice_step_quantized(q, fmt, phase, D, **kw)
    packed = q.packed.clone()
    ph_packed, cb = _phase_packed(phase, bits) if bits else (None, 0)
    scales = _ops.step_quantized(packed, ph_packed, q.scales, q.global_scale, q.zeros, D, fmt, rows, cols,
                                 phase_cb=cb, strata=16, **kw)
    codes = unpack(packed, fmt.container_bits, cols)
    ref_codes = unpack(ref_packed, fmt.container_bits, cols)
    same_scale = R.expand_groups(torch.isclose(scales.float(), ref_scales.float(), rtol=1e-6, atol=0.0).float(),
                                 rows, cols, fmt.scale).bool()      # compare codes where the scale decision agrees
    tol = 0.0 if rule == "fixed" else 5e-3
    assert (codes != ref_codes)[same_scale].float().mean().item() <= tol
    # the accept/reject test of a scale step can flip at a numerical tie (different summation order); a flip moves
    # the scale by at most the trust region, so fp32 scales are compared with rtol = 2 * kappa
    rtol = 2 * _ops.kappa(0.01, 1.0, lr)
    differ = ~torch.isclose(scales.float(), ref_scales.float(), rtol=rtol, atol=0.0)
    assert differ.float().mean().item() <= (0.0 if rule == "fixed" else 0.02)
    if bits:
        assert (_phase_unpacked(ph_packed, bits, cols) != ref_phase)[same_scale].float().mean().item() <= tol


@pytest.mark.parametrize("rounding", ["sr", "strat", "rtn"])
def test_rounding_modes_match_reference(rounding):
    fmt = F.get("int4")
    q = R.quantize(_weights(seed=3), fmt)
    D = torch.randn(q.rows, q.cols, device=DEV)
    kw = dict(lr=1e-3, weight_decay=0.0, phase_bits=2, rounding=rounding, seed=11, step=5, scale_rule="fixed",
              trust=0.01, trust_lr=1.0)
    ref_packed, _, ref_phase = R.lattice_step_quantized(q, fmt, torch.zeros(q.rows, q.cols, dtype=torch.int8,
                                                                                  device=DEV), D, **kw)
    packed = q.packed.clone()
    ph, cb = _phase_packed(torch.zeros(q.rows, q.cols, dtype=torch.int8, device=DEV), 2)
    _ops.step_quantized(packed, ph, q.scales, None, None, D, fmt, q.rows, q.cols, phase_cb=cb, strata=16, **kw)
    assert torch.equal(packed, ref_packed)
    assert torch.equal(_phase_unpacked(ph, 2, q.cols), ref_phase)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("bits", [0, 2, 8])
def test_step_native_matches_reference(dtype, bits):
    g = torch.Generator(device=DEV).manual_seed(4)
    n = 4096 + 5
    w = (torch.randn(n, device=DEV, generator=g) * 0.05).to(dtype)
    D = torch.randn(n, device=DEV, generator=g)
    phase = torch.randint(-(1 << max(bits - 1, 0)), 1 << max(bits - 1, 0), (n,), device=DEV, dtype=torch.int8,
                          generator=g) if bits else None
    lr = 1e-3 if dtype != torch.float32 else 1e-7
    kw = dict(lr=lr, weight_decay=0.01, phase_bits=bits, rounding="sr", seed=5, step=9)
    ref_w, ref_phase = R.lattice_step_native(w, phase, D, **kw)
    out = w.clone()
    cb = _ops.phase_container(bits)
    ph = pack(torch.nn.functional.pad(phase.to(torch.int32) & ((1 << bits) - 1), (0, _ops.phase_numel(n) - n)),
              cb) if bits else None
    _ops.step_native(out, ph, D, phase_cb=cb, strata=16, **kw)
    assert (out != ref_w).float().mean().item() <= 1e-3
    if bits:
        got = sign_extend(unpack(ph, cb, _ops.phase_numel(n)), bits)[:n].to(torch.int8)
        assert (got != ref_phase).float().mean().item() <= 1e-3


@pytest.mark.parametrize("name", ["nvfp4", "int4", "ternary"])
def test_step_is_unbiased(name):
    """E[latent after the step] equals the target (fixed scales, 2-bit phase)."""
    fmt = F.get(name)
    q = R.quantize(_weights(rows=32, cols=128, seed=6), fmt)
    se = R.expand_groups(R.effective_scales(q.scales, fmt.scale, q.global_scale), q.rows, q.cols, fmt.scale)
    D = torch.randn(q.rows, q.cols, device=DEV)
    lr = 0.37 * se.mean().item() / D.abs().mean().item()
    start = R.dequantize(q, fmt)
    L = R.num_levels(fmt)
    acc = torch.zeros_like(start)
    n = 2000
    for s in range(n):
        packed = q.packed.clone()
        ph, cb = _phase_packed(torch.zeros(q.rows, q.cols, dtype=torch.int8, device=DEV), 2)
        _ops.step_quantized(packed, ph, q.scales, q.global_scale, q.zeros, D, fmt, q.rows, q.cols, lr=lr,
                            weight_decay=0.0, phase_bits=2, phase_cb=cb, rounding="sr", seed=0, step=s, strata=16,
                            scale_rule="fixed", trust=0.01, trust_lr=1.0)
        rank = R.raw_to_rank(fmt, unpack(packed, fmt.container_bits, q.cols))
        a = R.rank_level(fmt, rank)
        lo, hi = R.rank_level(fmt, (rank - 1).clamp_min(0)), R.rank_level(fmt, (rank + 1).clamp_max(L - 1))
        acc += R.latent(a, lo, hi, _phase_unpacked(ph, 2, q.cols), 2) * se
    target = start - lr * D
    lo_all = R.rank_level(fmt, torch.zeros_like(rank)) * se
    hi_all = R.rank_level(fmt, torch.full_like(rank, L - 1)) * se
    inside = (target > lo_all) & (target < hi_all)                  # targets outside the range are clamped
    err = ((acc / n - target).abs() / se)[inside]
    assert err.mean().item() < 0.02


@pytest.mark.parametrize("kind", ["muon", "muon_block", "factored", "sinkhorn", "colnorm", "sign", "raw"])
def test_directions_match_reference(kind):
    G = torch.randn(96, 200, device=DEV)
    if kind == "factored":
        r1, c1 = torch.zeros(96, device=DEV), torch.zeros(200, device=DEV)
        st = {}
        for k in range(1, 4):
            D = torch.ops.bitlattice.direction(G * k, 4, r1, c1, k, 0.95, 0, 5, 0, 2, 1.0)
            U = R.clip_rms(R.factored_direction(G * k, st, 0.95), 1.0)
        assert torch.allclose(D, U, rtol=1e-4, atol=1e-5)
        return
    D = _ops.direction(G, kind, block=32, ns_dtype="fp32")
    if kind in ("muon", "muon_block"):
        U = R.muon_direction(G, 32 if kind == "muon_block" else 0, 5, torch.float32)
    elif kind == "sinkhorn":
        U = R.sinkhorn_direction(G)
    elif kind == "colnorm":
        U = G / (G.square().mean(0, keepdim=True).sqrt() + 1e-30)
    elif kind == "sign":
        U = G.sign()
    else:
        U = G
    assert torch.allclose(D, R.clip_rms(U, 1.0), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("shape", [(96, 200), (1, 512), (300, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_factored_matches_reference(shape, dtype):
    torch.manual_seed(0)
    r1, c1 = torch.zeros(shape[0], device=DEV), torch.zeros(shape[1], device=DEV)
    st = {}
    for k in range(1, 4):
        G = (torch.randn(*shape, device=DEV) * torch.rand(shape[0], 1, device=DEV) * k).to(dtype)
        D = _ops.direction(G, "factored", r1, c1, k)
        U = R.clip_rms(R.factored_direction(G, st, 0.95), 1.0)
        assert torch.allclose(r1, st["r"], rtol=1e-5) and torch.allclose(c1, st["c"], rtol=1e-5)
        assert torch.allclose(D, U, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("coeffs", ["muon", "polar_express"])
@pytest.mark.parametrize("algo", ["standard", "gram"])
@pytest.mark.parametrize("block", [0, 32, (16, 64), (16, None)])
def test_newton_schulz_variants_match_reference(coeffs, algo, block):
    torch.manual_seed(0)
    G = torch.randn(96, 200, device=DEV)
    kind = "muon_block" if block else "muon"
    D = _ops.direction(G, kind, block=block, ns_dtype="fp32", ns_coeffs=coeffs, ns_algo=algo)
    U = R.clip_rms(R.muon_direction(G, block, 5, torch.float32, coeffs, algo), 1.0)
    assert ((D - U).norm() / U.norm()).item() < 1e-4


@pytest.mark.parametrize("coeffs", ["muon", "polar_express"])
def test_gram_newton_schulz_equals_standard(coeffs):
    torch.manual_seed(0)
    G = torch.randn(128, 1024, device=DEV)
    a = R.newton_schulz(G.double(), 5, coeffs=coeffs, algo="standard")
    for dtype, tol in ((torch.float32, 1e-4), (torch.float16, 3e-2)):
        b = torch.ops.bitlattice.direction(G.to(dtype), 5, None, None, 1, 0.95, 0, 5,
                                           0 if dtype == torch.float32 else 2, 2, 1e9, 0, 1 if coeffs != "muon" else 0, 2)
        ref = (a * math.sqrt(1024)).float()
        assert ((b - ref).norm() / ref.norm()).item() < tol, dtype


def test_polar_express_is_closer_to_orthogonal():
    torch.manual_seed(0)
    G = torch.randn(256, 256, device=DEV) @ torch.diag(torch.logspace(-3, 0, 256, device=DEV))
    errs = {}
    for coeffs in ("muon", "polar_express"):
        X = R.newton_schulz(G, 5, coeffs=coeffs)
        sv = torch.linalg.svdvals(X)
        errs[coeffs] = (sv - 1).abs().mean().item()
    assert errs["polar_express"] < errs["muon"]
