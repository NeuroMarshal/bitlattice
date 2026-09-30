"""Model editing: vocabulary extension (dense and lattice weights), removing and replacing submodules."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


@pytest.fixture(scope="module")
def tokenizer():
    """A small byte-level BPE trained on the spot (no network, no cache)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    text = ["int main(void) { return 0; } mov eax, 1 push rbp 55 48 89 e5 c3"] * 50
    tk.train_from_iterator(text, trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>"],
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>")


def _model(vocab):
    cfg = transformers.Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                   tie_word_embeddings=False)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).to(DEV)


@pytest.mark.parametrize("quantized_head", [False, True])
def test_extend_vocab_with_hex_tokens(tokenizer, quantized_head):
    tok = __import__("copy").deepcopy(tokenizer)
    m = _model(len(tok))
    if quantized_head:
        bl.quantize_(m, format="int8", rules=(bl.Rule("*layers*", "skip"),), min_features=16)
        assert isinstance(m.lm_head.weight, LatticeTensor)
    old_emb = m.get_input_embeddings().weight.detach().float().clone()
    hexes = [f"<h{b:02x}>" for b in range(256)]
    pieces = tok("<h4f>", add_special_tokens=False).input_ids
    v0 = len(tok)
    ids = bl.surgery.extend_vocab(m, tok, hexes)
    assert ids == list(range(v0, v0 + 256)) and len(tok) == v0 + 256
    emb = m.get_input_embeddings().weight
    rows = -(-(v0 + 256) // 64) * 64
    assert emb.shape[0] == rows and m.lm_head.weight.shape[0] == rows and m.config.vocab_size == rows
    assert torch.allclose(emb[ids[0x4F]].float(), old_emb[pieces].mean(0), atol=1e-3)
    assert torch.equal(emb[:v0].float(), old_emb[:v0])
    assert tok.tokenize("<h55><h48>") == ["<h55>", "<h48>"]
    x = torch.tensor([tok("mov <h55><h48><h89><he5> end").input_ids], device=DEV)
    assert x.max().item() >= v0
    opt = bl.optim.build(m, "muon", lr=3e-3)
    losses = []
    for _ in range(20):
        loss = m(x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_drop_and_replace_modules():
    m = _model(1000)
    removed = bl.surgery.drop_modules(m, "model.layers.1")
    assert removed == ["model.layers.1"] and isinstance(m.model.layers[1], torch.nn.Identity)
    old = bl.surgery.replace_module(m, "model.norm", torch.nn.Identity())
    assert old is not None and isinstance(m.model.norm, torch.nn.Identity)


def test_primitives_on_dense_and_lattice():
    m = _model(1000)
    bl.quantize_(m, format="int8", min_features=16, rules=(bl.Rule("*lm_head"),))
    name = "model.layers.0.mlp.up_proj.weight"
    before = bl.surgery.values(m.get_parameter(name))
    bl.surgery.resize_(m, name, 160, dim=0, init="zeros")                 # grow output channels
    after = m.get_parameter(name)
    assert isinstance(after, LatticeTensor) and after.shape == (160, 64)
    assert torch.allclose(bl.surgery.values(after)[:128], before, atol=2e-2 * before.abs().max())
    bl.surgery.select_(m, name, torch.arange(128), dim=0)                 # prune back
    bl.surgery.resize_(m, "model.layers.0.mlp.down_proj.weight", 128, dim=1)
    bl.surgery.update_(m, "lm_head.weight", lambda v: v.mul_(0.5))
    bl.surgery.sync_shapes_(m)
    assert m.model.layers[0].mlp.up_proj.out_features == 128
    x = torch.randint(0, 1000, (1, 8), device=DEV)
    assert torch.isfinite(m(x).logits).all()
    bl.surgery.insert_module(m, "model.layers", 1, torch.nn.Identity())
    assert len(m.model.layers) == 3
