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
    assert b.model.layers[0].attention_type == a.model.layers[0].attention_type    # read by the model's forward
    x = torch.randint(0, 128, (2, 17), device=DEV)
    la, lb = bl.nn.causal_lm_loss(a, x), bl.nn.causal_lm_loss(b, x)
    la.backward()
    lb.backward()
    assert torch.equal(la, lb)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-3, rtol=1e-2), n
