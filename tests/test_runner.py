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
