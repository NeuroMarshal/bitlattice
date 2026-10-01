"""Checkpoint loading: dense, pre-quantized layouts, bitlattice round trip, training after load."""
import json
import os

import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor, formats as F
from bitlattice.packing import sign_extend, unpack

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _tiny_config():
    return transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                    tie_word_embeddings=False, torch_dtype="bfloat16")


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(_tiny_config()).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("dense")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def _ids():
    return torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1)).to(DEV)


def test_dense_load_matches_transformers(dense_dir):
    ref = transformers.Qwen2ForCausalLM.from_pretrained(dense_dir, torch_dtype=torch.bfloat16).to(DEV).eval()
    m = bl.io.load(dense_dir, device=DEV)
    with torch.no_grad():
        assert torch.equal(m(_ids()).logits, ref(_ids()).logits)


def test_quantized_load_matches_quantize_after_load(dense_dir):
    cfg = bl.quant.presets.default("nvfp4")
    m = bl.io.load(dense_dir, cfg, device=DEV)
    assert isinstance(m.model.layers[0].mlp.down_proj.weight, LatticeTensor)
    assert not isinstance(m.lm_head.weight, LatticeTensor)
    ref = transformers.Qwen2ForCausalLM.from_pretrained(dense_dir, torch_dtype=torch.bfloat16).to(DEV).eval()
    bl.surgery.quantize_(ref, "*", cfg)
    with torch.no_grad():
        assert torch.equal(m(_ids()).logits, ref(_ids()).logits)


# --------------------------------------------------------------------------------------------------------------
# writers of external layouts (from the published specifications), used to build synthetic checkpoints

def _to_i32(x64):
    x64 = x64 & 0xFFFFFFFF
    return torch.where(x64 >= 2 ** 31, x64 - 2 ** 32, x64).to(torch.int32)


def _pack_i32(u, dim):
    """unsigned nibbles -> int32 words of 8 (low first) along dim."""
    u = u.to(torch.int64).movedim(dim, -1)
    u = u.reshape(*u.shape[:-1], u.shape[-1] // 8, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=u.device)
    return _to_i32((u << shifts).sum(-1)).movedim(-1, dim)


def _codes(lt):
    return unpack(lt.packed, lt.fmt.container_bits, lt.cols)


def _write(path, tensors, config_update, extra_files=None):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in tensors.items()}, os.path.join(path, "model.safetensors"))
    cfg = _tiny_config().to_dict()
    cfg.update(config_update)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    for name, obj in (extra_files or {}).items():
        with open(os.path.join(path, name), "w") as f:
            json.dump(obj, f)


def _source(dense_dir, fmt):
    m = transformers.Qwen2ForCausalLM.from_pretrained(dense_dir, torch_dtype=torch.bfloat16).to(DEV)
    bl.surgery.quantize_(m, "*", bl.quant.presets.default(fmt), method="rtn")
    return m


def _layout_tensors(model, writer):
    out = {}
    for name, p in model.named_parameters():
        if isinstance(p, LatticeTensor):
            prefix = name.rpartition(".")[0]
            out.update({f"{prefix}.{k}": v for k, v in writer(p).items()})
        else:
            out[name] = p.detach()
    return out


def _check_same_weights(src, loaded):
    for (n1, p1), (n2, p2) in zip(src.named_parameters(), loaded.named_parameters()):
        assert n1 == n2
        a = p1.dequantize(torch.float32) if isinstance(p1, LatticeTensor) else p1.float()
        b = p2.dequantize(torch.float32) if isinstance(p2, LatticeTensor) else p2.float()
        assert torch.equal(a, b), n1


def test_modelopt_nvfp4(dense_dir, tmp_path):
    src = _source(dense_dir, "nvfp4")
    t = _layout_tensors(src, lambda p: {"weight": p.packed, "weight_scale": p.scales,
                                        "weight_scale_2": p.global_scale.reshape(()),
                                        "input_scale": torch.ones((), device=DEV)})
    _write(tmp_path, t, {}, {"hf_quant_config.json": {"quantization": {"quant_algo": "NVFP4", "group_size": 16,
                                                                       "exclude_modules": ["lm_head"]}}})
    _check_same_weights(src, bl.io.load(str(tmp_path), device=DEV))


def test_compressed_tensors_nvfp4(dense_dir, tmp_path):
    src = _source(dense_dir, "nvfp4")
    t = _layout_tensors(src, lambda p: {"weight_packed": p.packed, "weight_scale": p.scales,
                                        "weight_global_scale": (1.0 / p.global_scale).reshape(1),
                                        "input_global_scale": torch.ones(1, device=DEV)})
    _write(tmp_path, t, {"quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                                 "ignore": ["lm_head"]}})
    loaded = bl.io.load(str(tmp_path), device=DEV)
    for (n, a), (_, b) in zip(src.named_parameters(), loaded.named_parameters()):
        if isinstance(a, LatticeTensor):       # global scale passes through a reciprocal: compare values closely
            assert torch.allclose(a.dequantize(torch.float32), b.dequantize(torch.float32), rtol=1e-6, atol=0), n
            assert torch.equal(a.packed, b.packed) and torch.equal(a.scales, b.scales)


def test_fp8_channel(dense_dir, tmp_path):
    src = _source(dense_dir, "fp8")
    t = _layout_tensors(src, lambda p: {"weight": p.packed.view(torch.float8_e4m3fn), "weight_scale": p.scales})
    _write(tmp_path, t, {"quantization_config": {"quant_method": "compressed-tensors", "format": "float-quantized",
                                                 "ignore": ["lm_head"]}})
    _check_same_weights(src, bl.io.load(str(tmp_path), device=DEV))


def test_compressed_tensors_int4(dense_dir, tmp_path):
    src = _source(dense_dir, F.uniform(4, group=32, scale="fp16"))

    def writer(p):
        k = sign_extend(_codes(p), 4)
        return {"weight_packed": _pack_i32((k + 8), 1), "weight_scale": p.scales,
                "weight_shape": torch.tensor(list(p.shape), device=DEV)}
    t = _layout_tensors(src, writer)
    _write(tmp_path, t, {"quantization_config": {
        "quant_method": "compressed-tensors", "format": "pack-quantized", "ignore": ["lm_head"],
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 32, "symmetric": True}}}}})
    _check_same_weights(src, bl.io.load(str(tmp_path), device=DEV))


def test_gptq_v1(dense_dir, tmp_path):
    src = _source(dense_dir, F.uniform(4, group=32, scale="fp16", zero_point=True))

    def writer(p):
        u = _codes(p)                                     # [out, in]
        z = p.zeros.float().round().long()                # [out, G]
        return {"qweight": _pack_i32(u.t(), 0), "qzeros": _pack_i32(((z - 1) & 0xF).t(), 1),
                "scales": p.scales.t().to(torch.float16), "g_idx": torch.arange(p.shape[1], device=DEV) // 32}
    t = _layout_tensors(src, writer)
    _write(tmp_path, t, {"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32,
                                                 "desc_act": False, "sym": False}})
    _check_same_weights(src, bl.io.load(str(tmp_path), device=DEV))


def test_awq_gemm(dense_dir, tmp_path):
    from bitlattice.io.convert import AWQ_ORDER
    src = _source(dense_dir, F.uniform(4, group=32, scale="fp16", zero_point=True))

    def awq_pack(v):                                      # [rows, cols] -> int32 [rows, cols/8], AWQ nibble order
        v = v.reshape(v.shape[0], -1, 8)[..., list(AWQ_ORDER)]
        return _pack_i32(v.reshape(v.shape[0], -1), 1)

    def writer(p):
        u = _codes(p)
        z = p.zeros.float().round().long()
        return {"qweight": awq_pack(u.t()), "qzeros": awq_pack(z.t()), "scales": p.scales.t().to(torch.float16)}
    t = _layout_tensors(src, writer)
    _write(tmp_path, t, {"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 32,
                                                 "zero_point": True, "version": "gemm"}})
    _check_same_weights(src, bl.io.load(str(tmp_path), device=DEV))


def test_bitlattice_roundtrip_and_training(dense_dir, tmp_path):
    cfg = bl.quant.presets.default("int4", keep=0.1)
    m = bl.io.load(dense_dir, cfg, device=DEV)
    bl.io.save(m, str(tmp_path))
    m2 = bl.io.load(str(tmp_path), device=DEV)
    _check_same_weights(m, m2)
    m2.train()
    opt = bl.optim.Lattice(m2, "muon", lr=3e-3)
    ids = _ids()
    losses = []
    for _ in range(30):
        loss = m2(ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert isinstance(m2.model.layers[1].self_attn.q_proj.weight, LatticeTensor)


def test_export_compressed_tensors_roundtrip(dense_dir, tmp_path):
    m = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV)
    bl.io.export_compressed_tensors(m, str(tmp_path))
    back = bl.io.load(str(tmp_path), device=DEV)
    for (n, a), (_, b) in zip(m.named_parameters(), back.named_parameters()):
        if isinstance(a, LatticeTensor):
            assert torch.equal(a.packed, b.packed) and torch.equal(a.scales, b.scales), n


def test_generate_and_gradient_checkpointing(dense_dir):
    m = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV)
    ids = _ids()
    out = m.generate(ids[:1, :4], max_new_tokens=5, do_sample=False)
    assert out.shape == (1, 9)
    m.gradient_checkpointing_enable()
    m.config.use_cache = False
    m.train()
    opt = bl.optim.Lattice(m, "muon", lr=3e-3, fused_backward=True)
    with bl.storage.activations_in_ram():
        loss = m(ids, labels=ids).loss
    loss.backward()
    opt.step()
    assert all(p.grad is None for p in m.parameters())
    assert torch.isfinite(m(ids, labels=ids).loss)


def test_causal_lm_loss_matches_hf(dense_dir):
    m = bl.io.load(dense_dir, bl.quant.presets.default("int8"), device=DEV, dtype=torch.float32)
    ids = _ids()
    ref = m(ids, labels=ids).loss
    loss = bl.nn.causal_lm_loss(m, ids, chunk=7)
    assert torch.allclose(loss, ref, rtol=1e-4)
