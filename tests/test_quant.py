"""quantize_: rules, budget slider, presets, calibration."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor
from bitlattice.quant import QuantConfig, Rule, presets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = torch.nn.Linear(d, d)
        self.o_proj = torch.nn.Linear(d, d)
        self.up = torch.nn.Linear(d, 4 * d)
        self.down = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.down(torch.relu(self.up(self.o_proj(self.q_proj(x)))))


class Tiny(torch.nn.Module):
    def __init__(self, d=64, n=4, vocab=100):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([Block(d) for _ in range(n)])
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.embed_tokens(idx)
        for b in self.layers:
            x = b(x)
        return self.lm_head(x)


def test_rules_first_match_wins():
    m = Tiny().to(DEV)
    cfg = QuantConfig(format="nvfp4", rules=(Rule("*lm_head"), Rule("layers.0.*"),
                                             Rule("*o_proj", {"format": "int8"})))
    rep = bl.quantize_(m, cfg, report=True)
    assert not isinstance(m.lm_head.weight, LatticeTensor)
    assert not isinstance(m.layers[0].up.weight, LatticeTensor)
    assert m.layers[1].o_proj.weight.fmt.name == "int8"
    assert m.layers[1].up.weight.fmt.name == "nvfp4"
    assert "lm_head" in rep.skipped and len(rep.layers) == 12


@pytest.mark.parametrize("keep", [0.0, 0.1, 0.5])
def test_keep_slider_trades_error_for_bits(keep):
    m = Tiny().to(DEV)
    rep = bl.quantize_(m, presets.default("int3"), keep=keep, report=True)
    assert all(l.keep == round(keep * l.shape[1]) for l in rep.layers)
    test_keep_slider_trades_error_for_bits.results[keep] = (rep.bits_per_weight,
                                                           sum(l.rel_error for l in rep.layers) / len(rep.layers))
    r = test_keep_slider_trades_error_for_bits.results
    if len(r) == 3:
        assert r[0.0][0] < r[0.1][0] < r[0.5][0]
        assert r[0.0][1] > r[0.1][1] > r[0.5][1]


test_keep_slider_trades_error_for_bits.results = {}


def test_accuracy_critical_preset_keeps_edges_dense():
    m = Tiny().to(DEV)
    bl.quantize_(m, presets.accuracy_critical(m))
    assert not isinstance(m.layers[0].q_proj.weight, LatticeTensor)
    assert not isinstance(m.layers[3].up.weight, LatticeTensor)
    assert m.layers[1].o_proj.weight.fmt.name == "int8"
    assert m.layers[1].up.weight.fmt.name == "nvfp4" and m.layers[1].up.weight.hi_idx is not None


def test_activation_saliency_uses_calibration():
    m = Tiny().to(DEV)
    batches = [torch.randint(0, 100, (4, 16), device=DEV) for _ in range(3)]
    stats = bl.quant.calibrate(m, batches)
    assert "layers.1.up" in stats and stats["layers.1.up"].shape == (64,)
    bl.quantize_(m, presets.default("int4"), keep=0.1, saliency="activation", calibration=stats)
    assert m.layers[1].up.weight.hi_idx.numel() == 6
    with pytest.raises(ValueError):
        bl.quantize_(Tiny().to(DEV), presets.default("int4"), keep=0.1, saliency="activation")


def test_quantized_model_output_close():
    torch.manual_seed(0)
    m = Tiny().to(DEV)
    idx = torch.randint(0, 100, (2, 16), device=DEV)
    ref = m(idx)
    bl.quantize_(m, presets.default("int8"))
    out = m(idx)
    assert ((out - ref).norm() / ref.norm()).item() < 0.02
