"""Weights streamed from host memory (checkpoint conversion with weights='ram')."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


@pytest.fixture(autouse=True)
def _small_threshold(monkeypatch):
    import bitlattice.nn.offload as O
    monkeypatch.setattr(O, "STREAM_MIN_NUMEL", 1024)                # stream the tiny test model's matrices too


def _model(fmt):
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=64)
    torch.manual_seed(0)
    m = transformers.Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16)
    if fmt != "dense":
        bl.surgery.quantize_(m, "re:layers\\.\\d+\\.(self_attn|mlp)\\.\\w+_proj$", fmt)
    return m.train()


def _train(m, steps=4, micro=2):
    opt = bl.optim.Lattice(m, "muon", lr=1e-2, fused_backward=True)
    r = bl.Runner(m, opt, micro=micro)
    x = torch.randint(0, 256, (4, 65), generator=torch.Generator().manual_seed(3))
    return [r.step(x).item() for _ in range(steps)], r.loss(x).item()


@pytest.mark.parametrize("fmt", ["nvfp4", "int8", "dense"])
def test_streamed_weights_train_identically(fmt):
    a, b = _model(fmt), _model(fmt)
    bl.surgery.convert_(a, "re:layers\\.\\d+$", "checkpoint")
    bl.surgery.convert_(b, "re:layers\\.\\d+$", "checkpoint", weights="ram")
    items = [getattr(p, k) for layer in b.model.layers for p, k in layer.residency.items]
    assert items and not any(t.is_cuda for t in items)
    la, ea = _train(a)
    lb, eb = _train(b)
    assert la == lb and ea == eb
    bl.surgery.revert_(b, "re:layers\\.\\d+$")                    # weights back on the GPU, equal to a's
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        da = pa.dequantize() if isinstance(pa, bl.LatticeTensor) else pa
        db = pb.dequantize() if isinstance(pb, bl.LatticeTensor) else pb
        assert db.is_cuda and torch.equal(da, db), n


def test_streamed_weights_lower_the_gpu_memory():
    m = _model("int8")
    weights = torch.cuda.memory_allocated()
    bl.surgery.convert_(m, "re:layers\\.\\d+$", "checkpoint", weights="ram")
    freed = weights - torch.cuda.memory_allocated()
    per_layer = sum(p.storage_bytes() for p in m.model.layers[0].parameters() if isinstance(p, bl.LatticeTensor))
    assert freed >= 3.5 * per_layer
    _train(m, steps=1)                                              # at most two layers' weights resident


def test_streamed_weights_refuse_deferred_weight_gradients():
    if not bl.backend.fp4_supported(0):
        pytest.skip("needs FP4 tensor cores")
    m = _model("nvfp4")
    bl.surgery.set_compute_(m, "*", "fp4_all")
    bl.surgery.convert_(m, "re:layers\\.\\d+$", "checkpoint", weights="ram")
    r = bl.Runner(m, bl.optim.Lattice(m, "muon", lr=1e-2, fused_backward=True), micro=2, defer_wgrad=True)
    with pytest.raises(RuntimeError, match="deferred"):
        r.step(torch.randint(0, 256, (4, 65)))
