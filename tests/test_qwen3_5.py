"""Qwen3.5 integration on a tiny random model (needs transformers with Qwen3.5 and fla)."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytest.importorskip("fla")
M = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _layer():
    cfg = transformers.Qwen3_5TextConfig(hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=2,
                                         linear_num_value_heads=6, linear_key_head_dim=64, linear_value_head_dim=64,
                                         layer_types=["linear_attention", "full_attention"], vocab_size=256)
    torch.manual_seed(0)
    layer = M.Qwen3_5GatedDeltaNet(cfg, 0).cuda().to(torch.bfloat16)
    return layer


def test_gated_delta_net_conversion_matches_and_trains():
    import copy
    layer = _layer()
    orig = copy.deepcopy(layer)
    holder = torch.nn.ModuleList([layer])
    assert bl.surgery.convert_(holder, "0", "gated_delta_net") == [("0", "Qwen3_5GatedDeltaNet", "GatedDeltaNet")]
    layer = holder[0]
    x = torch.randn(2, 300, 256, device="cuda", dtype=torch.bfloat16)
    outs = []
    for m in (orig, layer):
        xi = x.clone().requires_grad_(True)
        y = m(xi)
        y.float().square().mean().backward()
        outs.append((y.float(), xi.grad.float(), m.in_proj_qkv.weight.grad.float()))
    for a, b in zip(outs[1], outs[0]):
        assert ((a - b).norm() / b.norm()).item() < 3e-2
