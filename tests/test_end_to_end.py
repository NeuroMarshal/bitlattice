"""A small language model trained from scratch in a low-bit format learns a task that needs its context, keeps its
weights in that format the whole time, and survives a save and load."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"
VOCAB = 64
LENGTH = 24


def _model():
    cfg = transformers.Qwen2Config(vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)


def _progressions(n, generator):
    """Sequences start + step * i mod VOCAB with a random start and a step of 1 to 4: the next token follows from
    the previous two, so the model has to read its context."""
    start = torch.randint(0, VOCAB, (n, 1), generator=generator)
    step = torch.randint(1, 5, (n, 1), generator=generator)
    return ((start + step * torch.arange(LENGTH)) % VOCAB).to(DEV)


@torch.no_grad()
def _accuracy(model, x):
    """Next-token accuracy from the third token on, where the step is known."""
    predicted = model(x[:, :-1]).logits.argmax(-1)
    return (predicted[:, 2:] == x[:, 3:]).float().mean().item()


def _lattice_weights(model):
    return {n: p for n, p in model.named_parameters() if isinstance(p, bl.LatticeTensor)}


@pytest.mark.parametrize("fmt", ["nvfp4", "int4"])
def test_low_bit_language_model_learns_a_task_that_needs_context(fmt, tmp_path):
    model = _model()
    bl.surgery.quantize_(model, "re:layers\\.", fmt, min_features=16, targets=(torch.nn.Linear,))
    weights = _lattice_weights(model)
    assert len(weights) == 2 * 7                     # q, k, v, o, gate, up, down per layer
    opt = bl.optim.Lattice(model, "muon", lr=3e-3, fused_backward=True, seed=0)
    runner = bl.Runner(model, opt, micro=2)
    train, held_out = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    test = _progressions(256, held_out)
    before = _accuracy(model, test)
    model.train()
    losses = [runner.step(_progressions(64, train)).item() for _ in range(300)]
    model.eval()
    after = _accuracy(model, test)
    assert before < 0.2 and after > 0.9, (before, after)
    assert sum(losses[-20:]) < 0.2 * sum(losses[:20])

    # The weights never left the format: same objects, same format, no dense copy anywhere.
    for name, p in _lattice_weights(model).items():
        assert p is weights[name] and p.fmt.name == bl.formats.get(fmt).name, name
    dense_bytes = sum(p.numel() * 2 for p in weights.values())
    stored = sum(p.storage_bytes() for p in weights.values())
    assert stored < 0.4 * dense_bytes, (stored, dense_bytes)

    # A saved model loads back bit for bit.
    bl.io.save(model, str(tmp_path))
    loaded = bl.io.load(str(tmp_path), device=DEV)
    with torch.no_grad():
        assert torch.equal(model(test[:8]).logits, loaded(test[:8]).logits)
    assert _accuracy(loaded, test) == after
