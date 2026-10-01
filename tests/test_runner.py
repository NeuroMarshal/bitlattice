"""Runner: one training step = micro-batches, backward, update; Memory placement."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _model(seed=0):
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(seed)
    m = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    bl.surgery.quantize_(m, "*", "int8", min_features=16)
    m.config.use_cache = False
    return m


@pytest.mark.parametrize("activations", ["gpu", "ram"])
def test_micro_batches_equal_one_batch(activations):
    x = torch.randint(0, 128, (4, 31), device=DEV, generator=torch.Generator(DEV).manual_seed(1))  # odd: padded masks
    a, b = _model(), _model()
    w0 = {n: p.dequantize(torch.float32) for n, p in a.named_parameters() if isinstance(p, bl.LatticeTensor)}
    ra = bl.Runner(a, bl.optim.Lattice(a, "muon", lr=1e-2, fused_backward=True), micro=1)
    rb = bl.Runner(b, bl.optim.Lattice(b, "muon", lr=1e-2, fused_backward=True), micro=2,
                   memory=bl.Memory(activations=activations))
    for _ in range(3):
        la, lb = ra.step(x), rb.step(x)
        assert abs(la.item() - lb.item()) <= 2e-2 * la.item()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if isinstance(pa, bl.LatticeTensor):       # stochastic rounding differs; the updates agree in direction
            wa, wb = pa.dequantize(torch.float32), pb.dequantize(torch.float32)
            assert ((wa - wb).norm() / (wa - w0[n]).norm()).item() < 0.35, n


def test_runner_learns_and_evaluates():
    m = _model()
    r = bl.Runner(m, bl.optim.Lattice(m, "muon", lr=3e-3, fused_backward=True), micro=2)
    x = torch.randint(0, 128, (4, 33), device=DEV, generator=torch.Generator(DEV).manual_seed(2))
    first = r.loss(x).item()
    for _ in range(20):
        r.step(x)
    assert r.loss(x).item() < 0.8 * first
    with pytest.raises(ValueError):
        bl.Memory(activations="disk")


def test_offloaded_slices_are_stored_compactly():
    base = torch.randn(2, 4096, 64, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    sizes = []
    orig = torch.empty_strided

    def spy(size, stride, **kw):
        t = orig(size, stride, **kw)
        if kw.get("device") == "cpu":
            sizes.append(t.untyped_storage().nbytes())
        return t
    torch.empty_strided = spy
    try:
        with bl.storage.activations_in_ram():
            y = (base[:, :256] * 2).sin()                           # saves a 256-token slice of the input
    finally:
        torch.empty_strided = orig
    y.sum().backward()
    assert sizes and max(sizes) <= 2 * 256 * 64 * 2 * 2              # not the 4096-token extent of the parent


def test_state_estimate_counts_momentum_formats():
    from bitlattice.storage import _state_estimate
    p = torch.empty(256, 512)
    base = dict(phase_bits=2, momentum="none", direction="muon_block", vector_direction="factored")
    sizes = {m: _state_estimate(p, {**base, "momentum": m}) for m in ("none", "nvfp4", "int8", "bf16", "fp32")}
    assert sizes["none"] < sizes["nvfp4"] < sizes["int8"] < sizes["bf16"] < sizes["fp32"]
    assert _state_estimate(p, {**base, "normalize": "row"}) == sizes["none"] + 4 * 256


def test_nvfp4_momentum_needs_cuda_parameters():
    with pytest.raises(ValueError, match="nvfp4"):
        bl.optim.Lattice(torch.nn.Linear(32, 32), "muon", momentum="nvfp4")


def test_reference_kernels_compute_the_same_training():
    x = torch.randint(0, 128, (4, 33), device=DEV, generator=torch.Generator(DEV).manual_seed(5))
    out = []
    for k in (None, "reference"):
        m = _model()
        bl.surgery.convert_(m, "*norm", "rms_norm", skip_unfit=True)
        r = bl.Runner(m, bl.optim.Lattice(m, "muon", lr=1e-2, fused_backward=True), micro=2, kernels=k)
        out.append([r.step(x).item() for _ in range(3)])
    assert all(abs(a - b) <= 2e-3 * abs(a) for a, b in zip(*out)), out
    with pytest.raises(TypeError, match="unknown kernel choices"):
        with bl.kernels(fast=True):
            pass
