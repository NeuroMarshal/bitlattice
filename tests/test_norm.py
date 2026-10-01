"""Lean RMSNorm: same outputs and gradients as the reference modules, less saved memory."""
import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


class OffsetRMSNorm(torch.nn.Module):          # Gemma / Qwen3.5 convention
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, x):
        o = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (o * (1.0 + self.weight.float())).type_as(x)


def _compare(ref_mod, new_mod, d):
    x = torch.randn(2, 300, d, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(x)
    y0 = ref_mod(x)
    gx0, gw0 = torch.autograd.grad(y0, (x, ref_mod.weight), g)
    bl.nn.fuse_rmsnorm_(torch.nn.ModuleList([new_mod]))
    y1 = new_mod(x)
    gx1, gw1 = torch.autograd.grad(y1, (x, new_mod.weight), g)
    assert torch.allclose(y0.float(), y1.float(), atol=2e-2, rtol=2e-2)
    assert torch.allclose(gx0.float(), gx1.float(), atol=3e-2, rtol=3e-2)
    # the reference accumulates the weight gradient from the bf16-rounded normalized input
    assert ((gw0.float() - gw1.float()).norm() / gw0.float().norm()).item() < 1e-2


def test_llama_convention():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    torch.manual_seed(0)
    a = Qwen3RMSNorm(256).to(DEV, torch.bfloat16)
    a.weight.data = torch.randn(256, device=DEV, dtype=torch.bfloat16) * 0.5 + 1
    b = Qwen3RMSNorm(256).to(DEV, torch.bfloat16)
    b.weight.data = a.weight.data.clone()
    _compare(a, b, 256)
    assert "forward" in b.__dict__


def test_offset_convention_and_memory():
    torch.manual_seed(1)
    a = OffsetRMSNorm(512).to(DEV)
    b = OffsetRMSNorm(512).to(DEV)
    b.weight.data = a.weight.data.clone()
    _compare(a, b, 512)
    x = torch.randn(4096, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)

    def saved(mod):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        y = mod(x)
        used = torch.cuda.memory_allocated() - base - y.numel() * y.element_size()
        del y
        return used

    assert saved(b) < 0.3 * saved(a)


def test_fuse_on_a_model():
    cfg = transformers.Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2)
    m = transformers.Qwen2ForCausalLM(cfg).to(DEV, torch.bfloat16)
    x = torch.randint(0, 100, (1, 16), device=DEV)
    ref = m(x).logits
    changed = bl.nn.fuse_rmsnorm_(m)
    assert changed == {"Qwen2RMSNorm": 5}
    assert torch.allclose(m(x).logits.float(), ref.float(), atol=5e-2, rtol=5e-2)
