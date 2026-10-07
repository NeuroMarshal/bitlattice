"""Layer split over devices, gradient accumulation with fused updates, token-chunked modules, state placement."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"
two_gpus = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")


def _config(layers=4, tie=False):
    return transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                    tie_word_embeddings=tie, torch_dtype="bfloat16")


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(_config()).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("dense4")
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


def _ids(b=2, t=12):
    return torch.randint(0, 128, (b, t), generator=torch.Generator().manual_seed(1)).to(DEV)


def _unit_devices(dm, layers):
    return [dm[f"model.layers.{i}"] for i in range(layers)]


def test_infer_device_map_contiguous_and_balanced():
    m = transformers.Qwen2ForCausalLM(_config(layers=8)).to(torch.bfloat16)
    dm = bl.infer_device_map(m, {"cuda:0": 10 ** 9, "cuda:1": 10 ** 9})
    devs = [d.index for d in _unit_devices(dm, 8)]
    assert devs == sorted(devs) and set(devs) == {0, 1}
    assert dm["model.embed_tokens"].index == 0 and dm["lm_head"].index == 1
    assert abs(devs.count(0) - devs.count(1)) <= 2
    # a device with a three times larger budget receives more blocks
    dm = bl.infer_device_map(m, {"cuda:0": 10 ** 9, "cuda:1": 3 * 10 ** 9})
    devs = [d.index for d in _unit_devices(dm, 8)]
    assert devs.count(1) > devs.count(0)
    with pytest.raises(RuntimeError):
        bl.infer_device_map(m, {"cuda:0": 1000, "cuda:1": 1000})


def test_tied_head_follows_embedding():
    m = transformers.Qwen2ForCausalLM(_config(layers=8, tie=True)).to(torch.bfloat16)
    dm = bl.infer_device_map(m, {"cuda:0": 10 ** 9, "cuda:1": 10 ** 9})
    assert dm["lm_head"] == dm["model.embed_tokens"]


def test_split_with_host_device_matches_and_trains(dense_dir):
    """Split across cuda:0 and the CPU (exercises the hooks and per-device updates on one GPU)."""
    ref = bl.io.load(dense_dir, bl.quant.presets.default("int8"), device=DEV, dtype=torch.float32)
    dm = {"model.embed_tokens": "cuda:0", "model.rotary_emb": "cuda:0", "model.layers.0": "cuda:0",
          "model.layers.1": "cuda:0", "model.layers.2": "cpu", "model.layers.3": "cpu", "model.norm": "cpu",
          "lm_head": "cpu"}
    m = bl.io.load(dense_dir, bl.quant.presets.default("int8"), device_map=dm, dtype=torch.float32)
    assert m.model.layers[3].mlp.up_proj.weight.device.type == "cpu"
    assert isinstance(m.model.layers[3].mlp.up_proj.weight, LatticeTensor)
    ids = _ids()
    with torch.no_grad():
        a, b = ref(ids).logits, m(ids).logits
    assert b.device.type == "cpu"
    assert torch.allclose(a.cpu(), b, atol=1e-3, rtol=1e-3)
    m.train()
    opt = bl.optim.Lattice(m, "muon", lr=3e-3, fused_backward=True)
    losses = []
    for _ in range(15):
        loss = bl.nn.causal_lm_loss(m, ids)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert m.model.layers[0].mlp.up_proj.weight.device.type == "cuda"
    assert m.model.layers[3].mlp.up_proj.weight.device.type == "cpu"


def test_bitlattice_checkpoint_loads_with_device_map(dense_dir, tmp_path):
    m = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV)
    bl.io.save(m, str(tmp_path))
    back = bl.io.load(str(tmp_path), device_map="auto")
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(m(ids).logits, back(ids).logits.to(DEV))


@two_gpus
def test_two_gpu_split_matches_single_device(dense_dir):
    single = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV)
    split = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device_map="auto",
                       max_memory={0: "1MiB", 1: "1MiB"})
    devs = {p.device.index for p in split.parameters()}
    assert devs == {0, 1}
    ids = _ids()
    with torch.no_grad():
        assert torch.allclose(single(ids).logits, split(ids).logits.to(DEV), atol=1e-2, rtol=1e-2)
    out = split.generate(ids[:1, :4], max_new_tokens=5, do_sample=False)
    assert out.shape == (1, 9)


@two_gpus
@pytest.mark.parametrize("state", ["gpu", "ram"])
def test_two_gpu_training_equals_single_device(dense_dir, state):
    def run(model):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        opt = bl.optim.Lattice(model, "muon", lr=3e-3, fused_backward=True, state=state)
        ids = _ids()
        out = []
        for _ in range(10):
            loss = bl.nn.causal_lm_loss(model, ids)
            loss.backward()
            opt.step()
            out.append(loss.item())
        opt.synchronize()
        return out

    a = run(bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV))
    b = run(bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device_map="auto",
                       max_memory={0: "1MiB", 1: "1MiB"}))
    assert b[-1] < b[0]
    assert all(abs(x - y) <= 1e-2 * abs(x) for x, y in zip(a, b)), (a, b)


def test_accumulate_equals_summed_gradient():
    def model():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Tanh(), torch.nn.Linear(128, 32)).to(DEV)
        bl.surgery.quantize_(m, "*", "nvfp4", min_features=16)
        return m

    xs = [torch.randn(16, 64, device=DEV, generator=torch.Generator(DEV).manual_seed(i)) for i in range(3)]
    a, b = model(), model()
    oa = bl.optim.Lattice(a, "muon", lr=1e-2, fused_backward=True)
    ob = bl.optim.Lattice(b, "muon", lr=1e-2)
    with oa.accumulate():
        for x in xs[:-1]:
            a(x).square().mean().backward()
    assert a[0].weight.grad is not None
    a(xs[-1]).square().mean().backward()
    oa.step()
    for x in xs:
        b(x).square().mean().backward()
    ob.step()
    assert a[0].weight.grad is None
    for pa, pb in zip(a.parameters(), b.parameters()):
        if isinstance(pa, LatticeTensor):
            assert torch.equal(pa.packed, pb.packed) and torch.equal(pa.scales, pb.scales)
        else:
            assert torch.equal(pa, pb)


def test_token_chunks_match_and_save_memory():
    torch.manual_seed(0)
    mlp = transformers.models.qwen2.modeling_qwen2.Qwen2MLP(
        transformers.Qwen2Config(hidden_size=512, intermediate_size=2048)).to(DEV, torch.bfloat16)
    x = torch.randn(1, 4096, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)

    def run():
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        y = mlp(x)
        saved = torch.cuda.memory_allocated() - base
        y.float().square().mean().backward()
        g = x.grad.clone()
        x.grad = None
        gw = mlp.up_proj.weight.grad.clone()
        mlp.zero_grad()
        return y.detach(), g, gw, saved

    y0, g0, w0, s0 = run()
    holder = torch.nn.ModuleDict({"mlp": mlp})
    assert bl.surgery.convert_(holder, "mlp", "token_chunks", chunk=1000) == [("mlp", "Qwen2MLP", "TokenChunks")]
    mlp = holder["mlp"]
    y1, g1, w1, s1 = run()
    assert torch.allclose(y0, y1, atol=1e-2, rtol=1e-2)
    assert torch.allclose(g0, g1, atol=1e-3, rtol=1e-2)
    assert torch.allclose(w0, w1, atol=1e-2, rtol=2e-2)
    assert s1 < 0.4 * s0, (s0, s1)


def test_auto_state_placement(monkeypatch):
    m = torch.nn.Sequential(torch.nn.Linear(64, 64)).to(DEV)
    assert bl.optim.Lattice(m, "muon", state="auto").state_placement == "gpu"
    monkeypatch.setattr(bl.storage, "AUTO_STATE_FRACTION", 0.0)
    opt = bl.optim.Lattice(m, "muon", state="auto")
    assert opt.state_placement == "ram" and opt.state_device == "cpu"
    assert len(bl.plan(m, opt).weights_by_device) == 1


def test_interleaved_map_alternates_and_balances():
    m = transformers.Qwen2ForCausalLM(_config(layers=8)).to(torch.bfloat16)
    dm = bl.parallel.interleaved_device_map(m, ["cuda:0", "cuda:1"])
    devs = [d.index for d in _unit_devices(dm, 8)]
    assert devs[:4] == [0, 1, 0, 1]
    assert dm["lm_head"].index == 1 and dm["model.embed_tokens"].index == 0
    assert devs.count(0) >= devs.count(1)                   # the head device gets fewer blocks
    dm2 = bl.parallel.interleaved_device_map(m, ["cuda:0", "cuda:1"], chunk=2)
    assert [d.index for d in _unit_devices(dm2, 8)][:4] == [0, 0, 1, 1]


@pytest.mark.parametrize("micro", [1, 2, 3])
def test_pipelined_loss_matches_plain_loss(dense_dir, micro):
    m = bl.io.load(dense_dir, bl.quant.presets.default("int8"), device=DEV, dtype=torch.float32)
    ids = _ids(b=6, t=12)
    ref = bl.nn.causal_lm_loss(m, ids)
    got = bl.parallel.pipelined_causal_lm_loss(m, ids, micro=micro)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    g_ref = torch.autograd.grad(bl.nn.causal_lm_loss(m, ids), m.model.layers[1].mlp.up_proj.weight)[0]
    g_got = torch.autograd.grad(bl.parallel.pipelined_causal_lm_loss(m, ids, micro=micro),
                                m.model.layers[1].mlp.up_proj.weight)[0]
    assert torch.allclose(g_got, g_ref, rtol=1e-4, atol=1e-6)


def test_pipelined_training_with_checkpointing_and_host_split(dense_dir):
    dm = bl.parallel.interleaved_device_map(
        transformers.Qwen2ForCausalLM(_config()).to(torch.bfloat16), ["cuda:0", "cpu"])
    m = bl.io.load(dense_dir, bl.quant.presets.default("int8"), device_map=dm, dtype=torch.float32)
    assert {p.device.type for p in m.model.layers[0].parameters()} != {p.device.type for p in
                                                                         m.model.layers[1].parameters()}
    m.gradient_checkpointing_enable()
    m.config.use_cache = False
    m.train()
    opt = bl.optim.Lattice(m, "muon", lr=3e-3, fused_backward=True)
    ids = _ids(b=4)
    losses = []
    for _ in range(12):
        loss = bl.parallel.pipelined_causal_lm_loss(m, ids, micro=2)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(p.grad is None for p in m.parameters())


@two_gpus
def test_two_gpu_pipelined_equals_single_device(dense_dir):
    def run(model, pipelined):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        opt = bl.optim.Lattice(model, "muon", lr=3e-3, fused_backward=True)
        ids = _ids(b=4)
        out = []
        for _ in range(8):
            loss = (bl.parallel.pipelined_causal_lm_loss(model, ids, micro=4) if pipelined
                    else bl.nn.causal_lm_loss(model, ids))
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    a = run(bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device=DEV), False)
    b = run(bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device_map="interleaved"), True)
    # identical first step; later steps differ by the bf16 accumulation order of the micro-batch gradients
    assert abs(a[0] - b[0]) <= 1e-3 * a[0] and b[-1] < 0.8 * b[0]
    assert all(abs(x - y) <= 0.05 * abs(x) for x, y in zip(a, b)), (a, b)


def test_interleaved_map_tail_cost_moves_blocks_off_the_head_device():
    m = transformers.Qwen2ForCausalLM(_config(layers=16)).to(torch.bfloat16)
    count = lambda dm: sum(1 for k, v in dm.items() if ".layers." in k and v.index == 1)   # noqa: E731
    a = bl.parallel.interleaved_device_map(m, [0, 1])
    b = bl.parallel.interleaved_device_map(m, [0, 1], tail_cost=50.0)       # a tiny head made expensive
    assert count(b) < count(a)
    assert all(v.index == 1 for k, v in b.items() if k.startswith("lm_head"))
