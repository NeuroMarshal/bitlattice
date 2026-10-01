"""The operating table: explicit, checked, reversible layer conversions that keep the parameter objects."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _model():
    cfg = transformers.Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)


def test_convert_keeps_parameters_and_reverts():
    m = _model()
    keys = list(m.state_dict())
    params = {n: p for n, p in m.named_parameters()}
    x = torch.randint(0, 100, (1, 16), device=DEV)
    ref = m(x).logits.float()
    rep = bl.surgery.convert_(m, "*.mlp", "token_chunks", chunk=8)
    rep += bl.surgery.convert_(m, "*norm", "rms_norm")
    assert len(rep) == 2 + 5
    assert list(m.state_dict()) == keys                                  # same names
    assert all(p is params[n] for n, p in m.named_parameters())          # same objects
    assert torch.allclose(m(x).logits.float(), ref, atol=5e-2, rtol=5e-2)
    assert "TokenChunks" in str(m) and "converted" in str(bl.explain(m))
    with torch.no_grad():                                                # edits made after converting survive revert
        m.model.norm.weight.add_(1.0)
    back = bl.surgery.revert_(m, "*")
    assert len(back) == 7 and type(m.model.norm).__name__ == "Qwen2RMSNorm"
    assert m.model.norm.weight is params["model.norm.weight"] and float(m.model.norm.weight.float().mean()) > 1.5
    assert list(m.state_dict()) == keys


def test_stacked_conversions_revert_one_level():
    m = _model()
    bl.surgery.convert_(m, "model.layers.0.mlp", "token_chunks", chunk=8)
    bl.surgery.convert_(m, "model.layers.0.mlp", "compressed_saves")
    assert type(m.model.layers[0].mlp).__name__ == "CompressedSaves"
    bl.surgery.revert_(m, "model.layers.0.mlp")
    assert type(m.model.layers[0].mlp).__name__ == "TokenChunks"
    bl.surgery.revert_(m, "model.layers.0.mlp")
    assert type(m.model.layers[0].mlp).__name__ == "Qwen2MLP"


def test_refusals_change_nothing():
    m = _model()
    with pytest.raises(ValueError, match="cannot be converted"):
        bl.surgery.convert_(m, "*", "rms_norm")                         # the model itself is not a norm
    with pytest.raises(ValueError, match="no module matches"):
        bl.surgery.convert_(m, "*.nothing", "rms_norm")
    with pytest.raises(ValueError, match="unknown conversion"):
        bl.surgery.convert_(m, "*norm", "magic")
    assert not any(isinstance(x, bl.nn.Converted) for x in m.modules())
    assert set(bl.surgery.conversions()) >= {"rms_norm", "gated_rms_norm", "gated_delta_net", "token_chunks",
                                             "compressed_saves"}


def test_selectors():
    m = _model()
    sel = bl.select.select
    assert [n for n, _ in sel(m, "*.mlp")] == ["model.layers.0.mlp", "model.layers.1.mlp"]
    assert [n for n, _ in sel(m, "re:layers\\.1\\.mlp$")] == ["model.layers.1.mlp"]
    assert len(sel(m, transformers.models.qwen2.modeling_qwen2.Qwen2MLP)) == 2
    assert sel(m, m.model.norm) == [("model.norm", m.model.norm)]
    assert len(sel(m, ["*.mlp", "model.norm"])) == 3
    assert len(sel(m, lambda n, mod: n.endswith("down_proj"))) == 2
    assert sel(m, "*", outermost=True) == [("", m)]


def test_selectors_that_select_nothing_are_errors():
    m = _model()
    for op in (lambda: bl.surgery.freeze_(m, "lm_head.*"), lambda: bl.surgery.quantize_(m, "*.nothing", "int8"),
               lambda: bl.surgery.set_compute_(m, "*.nothing", "exact"), lambda: bl.surgery.drop_(m, "*.nothing")):
        with pytest.raises(ValueError, match="no module matches"):
            op()
    assert bl.surgery.freeze_(m, "lm_head") == ["lm_head.weight"] and not m.lm_head.weight.requires_grad
    assert bl.surgery.unfreeze_(m, "lm_head") == ["lm_head.weight"] and m.lm_head.weight.requires_grad


def test_checkpointed_decoder_layers_keep_their_attributes_and_gradients():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    a = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    b = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        m.config.use_cache = False
        m.train()
    bl.surgery.convert_(b, "re:layers\\.\\d+$", "checkpoint")
    layer, original = b.model.layers[0], b.model.layers[0].original
    for k, v in vars(original).items():               # configuration the model's forward may read (attention_type)
        if not k.startswith("_"):
            assert getattr(layer, k) is v or k == "training", k
    x = torch.randint(0, 128, (2, 17), device=DEV)
    la, lb = bl.nn.causal_lm_loss(a, x), bl.nn.causal_lm_loss(b, x)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-3, rtol=1e-2), n


def test_unfit_modules_are_skipped_with_a_reason_when_asked(caplog):
    class OddRMSNorm(torch.nn.Module):            # an RMSNorm by name, but not a convention the library implements
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(16))
            self.eps = 1e-6

        def forward(self, x):
            return x * self.weight + 1

    m = torch.nn.Sequential(OddRMSNorm(), torch.nn.Linear(16, 16)).to(DEV)
    with torch.device("meta"):                    # a module another pipeline process holds
        m.append(OddRMSNorm())
    with pytest.raises(ValueError, match="not an RMSNorm convention"):
        bl.surgery.convert_(m, OddRMSNorm, "rms_norm")
    with caplog.at_level("INFO", logger="bitlattice"):
        assert bl.surgery.convert_(m, OddRMSNorm, "rms_norm", skip_unfit=True) == []
    assert "not an RMSNorm convention" in caplog.text and "meta" in caplog.text
    with pytest.raises(TypeError):                # invalid options are never skipped
        bl.surgery.convert_(m, "1", "token_chunks", chunk=4, colour="red", skip_unfit=True)


@pytest.mark.parametrize("chunk", [16, 40])
def test_attention_chunks_reproduce_the_layer_and_its_gradients(chunk):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=32)
    torch.manual_seed(0)
    a = transformers.Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16)
    b = transformers.Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16)
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        m.config.use_cache = False
        m.train()
    done = bl.surgery.convert_(b, "*.self_attn", "attention_chunks", chunk=chunk)
    assert len(done) == 2 and done[0][2] == "AttentionChunks"
    x = torch.randint(0, 128, (2, 97), device=DEV)
    la, lb = bl.nn.causal_lm_loss(a, x), bl.nn.causal_lm_loss(b, x)
    la.backward()
    lb.backward()
    assert abs(la.item() - lb.item()) < 1e-2
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad.float(), pb.grad.float(), atol=2e-3, rtol=5e-2), n
    with torch.no_grad():                                # generation with a cache goes to the original layer
        out = b.generate(x[:1, :8], max_new_tokens=4, do_sample=False, use_cache=True)
        ref = a.generate(x[:1, :8], max_new_tokens=4, do_sample=False, use_cache=True)
    assert torch.equal(out, ref)
    with pytest.raises(TypeError, match="chunk"):
        bl.surgery.convert_(b, "*.self_attn", "attention_chunks")


def test_host_embedding_frees_the_device_table_and_reverts():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    m.config.use_cache = False
    x = torch.randint(0, 512, (2, 17), device=DEV)
    ref = bl.nn.causal_lm_loss(m, x)
    with pytest.raises(ValueError, match="trainable"):
        bl.surgery.convert_(m, "*embed_tokens", "host_embedding")
    assert m.model.embed_tokens.weight.is_cuda                     # a refused conversion changes nothing
    bl.surgery.freeze_(m, "*embed_tokens")
    w = m.model.embed_tokens.weight
    before = torch.cuda.memory_allocated()
    bl.surgery.convert_(m, "*embed_tokens", "host_embedding")
    assert m.model.embed_tokens.weight is w and not w.is_cuda and w.is_pinned()
    assert torch.cuda.memory_allocated() <= before - w.numel() * w.element_size()
    assert torch.equal(bl.nn.causal_lm_loss(m, x), ref)
    loss = bl.nn.causal_lm_loss(m, x)
    loss.backward()                                                # trains the rest; no gradient for the table
    assert w.grad is None and m.lm_head.weight.grad is not None
    bl.surgery.revert_(m, "*embed_tokens")
    assert m.model.embed_tokens.weight is w and w.is_cuda


def test_host_embedding_refuses_tied_tables():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    m = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    bl.surgery.freeze_(m, "*embed_tokens")
    with pytest.raises(ValueError, match="tied"):
        bl.surgery.convert_(m, "*embed_tokens", "host_embedding")


def test_host_linear_head_streams_and_matches():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    m.config.use_cache = False
    bl.surgery.freeze_(m, "lm_head")
    x = torch.randint(0, 1000, (2, 50), device=DEV)
    labels = x.clone()
    labels[:, :10] = -100
    ref = bl.nn.causal_lm_loss(m, x, labels, chunk=16)
    ref.backward()
    ref_g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        ref_eval = bl.nn.causal_lm_loss(m, x, labels, chunk=16)
        logits = m(x).logits
    w = m.lm_head.weight
    bl.surgery.convert_(m, "lm_head", "host_linear")
    assert not w.is_cuda and w.is_pinned()
    import bitlattice.nn.loss as L
    old = (L._HOST_TILE_BYTES, L._HOST_GROUP_BYTES)
    L._HOST_TILE_BYTES, L._HOST_GROUP_BYTES = 128 * 64 * 2 * 3, 64 * 4 * 32       # several tiles and token groups
    try:
        out = bl.nn.causal_lm_loss(m, x, labels, chunk=16)
        out.backward()
        with torch.no_grad():
            out_eval = bl.nn.causal_lm_loss(m, x, labels, chunk=16)
    finally:
        L._HOST_TILE_BYTES, L._HOST_GROUP_BYTES = old
    assert abs(out.item() - ref.item()) < 1e-3 and abs(out_eval.item() - ref_eval.item()) < 1e-3
    for n, g in ref_g.items():
        p = dict(m.named_parameters())[n]
        assert torch.allclose(p.grad.float(), g.float(), atol=1e-3, rtol=2e-2), n
    with torch.no_grad():
        assert torch.allclose(m(x).logits.float(), logits.float(), atol=2e-2, rtol=2e-2)
    bl.surgery.revert_(m, "lm_head")
    assert w.is_cuda


def _qwen3(use_cache=False, layers=2):
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=32, use_cache=use_cache)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16).train()


def _grads(m, x):
    m.zero_grad(set_to_none=True)
    loss = bl.nn.causal_lm_loss(m, x)
    loss.backward()
    return loss.item(), {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}


def test_checkpoint_passes_gradients_to_keyword_tensors():
    a, b = _qwen3(), _qwen3()
    bl.surgery.convert_(b, "*.self_attn", "checkpoint")             # called as self_attn(hidden_states=...)
    x = torch.randint(0, 128, (2, 33), device=DEV)
    (la, ga), (lb, gb) = _grads(a, x), _grads(b, x)
    assert abs(la - lb) < 1e-3 and set(ga) == set(gb)
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-3, rtol=5e-2), n


def test_conversions_inside_converted_modules_take_effect():
    a, b = _qwen3(use_cache=True), _qwen3(use_cache=True)            # default config of HF models: a cache
    bl.surgery.convert_(b, "re:layers\\.\\d+$", "checkpoint")
    bl.surgery.convert_(b, "*.self_attn", "attention_chunks", chunk=8)
    assert all(isinstance(layer.original.self_attn, bl.nn.AttentionChunks) for layer in b.model.layers)
    calls = []
    b.model.layers[0].self_attn.register_forward_hook(lambda *_: calls.append(1))
    x = torch.randint(0, 128, (2, 33), device=DEV)
    (la, ga), (lb, gb) = _grads(a, x), _grads(b, x)
    assert calls and abs(la - lb) < 1e-2
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=2e-3, rtol=5e-2), n
    loss = bl.nn.causal_lm_loss(b, x, use_cache=True)              # a training call the conversion cannot run:
    with pytest.raises((ValueError, RuntimeError), match="cache"):   # refused when its graph is built (recompute)
        loss.backward()


def test_runner_with_host_embedding_and_a_plain_torch_optimizer():
    m = _qwen3()
    bl.surgery.freeze_(m, "*embed_tokens")
    bl.surgery.convert_(m, "*embed_tokens", "host_embedding")
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=1e-1)
    r = bl.Runner(m, opt, micro=2)
    x = torch.randint(0, 128, (4, 33))                              # batch on the host
    first = r.loss(x).item()
    for _ in range(5):
        r.step(x)
    assert r.loss(x).item() < first
    assert all(p.grad is None for p in m.parameters())             # zero_grad after each step
