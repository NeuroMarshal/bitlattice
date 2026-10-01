"""Quantized KV cache: exact for dense formats, close for quantized ones, compatible with generation."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


@pytest.fixture(scope="module")
def model():
    cfg = transformers.Qwen2Config(vocab_size=200, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16).eval()


def _chunked_logits(model, ids, cache, step=37):
    outs = []
    with torch.no_grad():
        for s in range(0, ids.shape[1], step):
            outs.append(model(ids[:, s:s + step], past_key_values=cache, use_cache=True).logits)
    return torch.cat(outs, 1).float()


def test_dense_format_equals_plain_cache(model):
    ids = torch.randint(0, 200, (2, 300), device=DEV)
    ref = _chunked_logits(model, ids, transformers.DynamicCache(config=model.config))
    got = _chunked_logits(model, ids, bl.kvcache.quantized_cache(model, k="bf16", residual=32, chunk=32))
    assert torch.allclose(got, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("fmt,tol", [("fp8", 0.03), ("mxfp8", 0.03), ("int8", 0.03), ("nvfp4", 0.15), ("nvfp4_4o6", 0.15),
                                     ("mxfp4", 0.2), ("int4", 0.2), ("mxfp6", 0.06)])
@pytest.mark.parametrize("rotate", [False, True])
def test_quantized_formats_stay_close(model, fmt, tol, rotate):
    ids = torch.randint(0, 200, (2, 300), device=DEV)
    ref = _chunked_logits(model, ids, transformers.DynamicCache(config=model.config))
    cache = bl.kvcache.quantized_cache(model, k=fmt, residual=32, chunk=32, sink=4, rotate=rotate)
    got = _chunked_logits(model, ids, cache)
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < tol, rel
    b = bl.kvcache.cache_bytes(cache)
    assert b["quantized"] > 0


def test_separate_formats_and_axes(model):
    ids = torch.randint(0, 200, (1, 200), device=DEV)
    cache = bl.kvcache.quantized_cache(model, k=bl.kvcache.KVSpec("int4", "channel"),
                                       v=bl.kvcache.KVSpec("fp8", "token"), residual=16, chunk=32)
    _chunked_logits(model, ids, cache)
    layer = cache.layers[0]
    assert layer.k_store.fmt.name.startswith("int4") and layer.v_store.fmt.name == "fp8"
    assert layer.get_seq_length() == 200


def test_generate_and_batch_reorder(model):
    ids = torch.randint(0, 200, (2, 100), device=DEV)
    cache = bl.kvcache.quantized_cache(model, k="nvfp4", residual=16, chunk=16)
    out = model.generate(ids, past_key_values=cache, max_new_tokens=20, do_sample=False)
    assert out.shape == (2, 120)
    cache.layers[0].batch_select_indices(torch.tensor([1], device=DEV))
    assert cache.layers[0].keys.shape[0] == 1
