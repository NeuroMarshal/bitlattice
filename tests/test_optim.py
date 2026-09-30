"""Lattice optimizer: training in quantized and native formats."""
import pytest
import torch

import bitlattice as bl
from bitlattice import LatticeTensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def _teacher_student(d_in=64, d_hidden=128, d_out=32, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    teacher = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(d_hidden, d_out)).to(DEV)
    for p in teacher.parameters():
        p.data = torch.randn(p.shape, device=DEV, generator=g) * (1.0 / p.shape[-1] ** 0.5)
    torch.manual_seed(seed + 1)
    student = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.Tanh(),
                                  torch.nn.Linear(d_hidden, d_out)).to(DEV)
    return teacher, student


def _train(student, teacher, opt, steps=400, bs=256, d_in=64, seed=2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    losses = []
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=opt.param_groups[0]["lr"] * 0.1)
    dtype = next(student.parameters()).dtype
    for _ in range(steps):
        x = torch.randn(bs, d_in, device=DEV, generator=g).to(dtype)
        with torch.no_grad():
            y = teacher(x)
        loss = torch.nn.functional.mse_loss(student(x).float(), y.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("fmt", ["bf16", "int8", "nvfp4", "int4", "mxfp4"])
@pytest.mark.parametrize("preset", ["muon", "balanced", "light"])
def test_trains_in_format(fmt, preset):
    teacher, student = _teacher_student()
    if fmt != "bf16":
        bl.quantize_(student, format=fmt, min_features=16, targets=(torch.nn.Linear,))
        assert isinstance(student[0].weight, LatticeTensor)
    else:
        student.to(torch.bfloat16)
        teacher.to(torch.bfloat16)
    opt = bl.optim.build(student, preset, lr=1e-2)
    losses = _train(student, teacher, opt)
    start, end = sum(losses[:10]) / 10, sum(losses[-20:]) / 20
    assert end < 0.35 * start, (start, end)
    if fmt != "bf16":
        assert isinstance(student[0].weight, LatticeTensor)


def test_fused_backward_matches_step():
    teacher, s1 = _teacher_student(seed=3)
    _, s2 = _teacher_student(seed=3)
    for s in (s1, s2):
        bl.quantize_(s, format="int4", min_features=16)
    o1 = bl.optim.build(s1, "muon", lr=5e-3)
    o2 = bl.optim.build(s2, "muon", lr=5e-3, fused_backward=True)
    l1 = _train(s1, teacher, o1, steps=30)
    l2 = _train(s2, teacher, o2, steps=30)
    assert l1 == l2
    assert all(p.grad is None for p in s2.parameters())
    assert torch.equal(s1[0].weight.packed, s2[0].weight.packed)


def test_state_accounting_and_state_dict():
    teacher, student = _teacher_student()
    bl.quantize_(student, format="nvfp4", min_features=16)
    opt = bl.optim.build(student, "muon", lr=1e-2)
    _train(student, teacher, opt, steps=3)
    bits = opt.state_bits()
    n = sum(p.numel() for p in student.parameters())
    assert bits["phase"] == 2 * n and bits["momentum"] == 0
    sd = opt.state_dict()
    opt2 = bl.optim.build(student, "muon", lr=1e-2)
    opt2.load_state_dict(sd)
    k = next(iter(opt.state))
    assert torch.equal(opt2.state[k]["phase"], opt.state[k]["phase"])


def test_options_validated():
    lin = torch.nn.Linear(16, 16, device=DEV)
    with pytest.raises(ValueError):
        bl.Lattice(lin.parameters(), direction="adam")
    with pytest.raises(TypeError):
        bl.Lattice(lin.parameters(), betas=(0.9, 0.99))


def test_keep_columns_train():
    teacher, student = _teacher_student()
    bl.quantize_(student, format="int3", keep=0.1, min_features=16)
    before = student[0].weight.hi_val.clone()
    opt = bl.optim.build(student, "muon", lr=1e-2)
    losses = _train(student, teacher, opt, steps=200)
    assert not torch.equal(student[0].weight.hi_val, before)
    assert sum(losses[-20:]) / 20 < 0.5 * sum(losses[:10]) / 10


def test_discrete_activation_trains():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(32, 64), bl.nn.DiscreteAct("sign", "expect"),
                              torch.nn.Linear(64, 1)).to(DEV)
    bl.quantize_(net, format="int4", min_features=16)
    opt = bl.optim.build(net, "muon", lr=1e-2)
    g = torch.Generator(device=DEV).manual_seed(0)
    w = torch.randn(32, device=DEV, generator=g)
    losses = []
    for _ in range(300):
        x = torch.randn(256, 32, device=DEV, generator=g)
        y = (x @ w > 0).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(-1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-20:]) / 20 < 0.45


@pytest.mark.parametrize("fused", [False, True])
def test_state_in_ram_matches_gpu_state(fused):
    teacher, s1 = _teacher_student(seed=5)
    _, s2 = _teacher_student(seed=5)
    for s in (s1, s2):
        bl.quantize_(s, format="nvfp4", min_features=16)
    o1 = bl.optim.build(s1, "muon", lr=5e-3, fused_backward=fused)
    o2 = bl.optim.build(s2, "muon", lr=5e-3, fused_backward=fused, storage="state_in_ram")
    l1 = _train(s1, teacher, o1, steps=25)
    l2 = _train(s2, teacher, o2, steps=25)
    assert l1 == l2
    o2.synchronize()
    for p1, p2 in zip(s1.parameters(), s2.parameters()):
        assert not o2.state[p2]["phase"].is_cuda
        assert torch.equal(o1.state[p1]["phase"].cpu(), o2.state[p2]["phase"])
    plan = bl.plan(s2, o2)
    assert plan.state_gpu_bytes < plan.state_host_bytes


def test_activations_in_ram_same_gradients():
    teacher, s1 = _teacher_student(seed=6)
    bl.quantize_(s1, format="int4", min_features=16)
    x = torch.randn(64, 64, device=DEV)
    y = teacher(x).detach()
    torch.nn.functional.mse_loss(s1(x), y).backward()
    ref = [p.grad.clone() for p in s1.parameters()]
    for p in s1.parameters():
        p.grad = None
    with bl.storage.activations_in_ram():
        loss = torch.nn.functional.mse_loss(s1(x), y)
    loss.backward()
    for r, p in zip(ref, s1.parameters()):
        assert torch.equal(r, p.grad)


def test_plan_reports_compression():
    _, student = _teacher_student()
    bl.quantize_(student, format="nvfp4", min_features=16)
    opt = bl.optim.build(student, "muon", fused_backward=True)
    p = bl.plan(student, opt)
    assert p.bits_per_weight < 8 and p.state_gpu_bytes > 0 and "weights" in str(p)
