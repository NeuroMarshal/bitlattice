"""One process per GPU: the interleaved pipeline reproduces single-GPU training. With two GPUs the blocks alternate
between two processes; with one GPU a single process runs the same distributed path."""
import json
import os
import subprocess
import sys

import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
GPUS = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 0
pytestmark = pytest.mark.skipif(GPUS == 0, reason="needs CUDA")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def dense_dir(tmp_path_factory):
    cfg = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   tie_word_embeddings=False, torch_dtype="bfloat16")
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("pipe")
    transformers.Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.mark.parametrize("activations", ["gpu", "ram"])
def test_two_process_pipeline_matches_single_gpu(dense_dir, tmp_path, activations):
    out = tmp_path / "pipe.jsonl"
    args = ["--model", dense_dir, "--steps", "8", "--batch", "4", "--micro", "2", "--seq", "12", "--preset", "muon",
            "--lr", "3e-3", "--data", "fixed", "--json", str(out), "--activations", activations]
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(GPUS),
                    os.path.join(ROOT, "tests", "pipeline_worker.py"), *args], check=True, env=env, cwd=ROOT,
                   timeout=600)
    piped = json.loads(out.read_text().splitlines()[-1])["losses"]

    m = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device="cuda")
    m.config.use_cache = False
    m.gradient_checkpointing_enable()
    m.train()
    opt = bl.optim.Lattice(m, "muon", lr=3e-3, fused_backward=True, seed=0)
    x = torch.randint(0, 128, (4, 13), generator=torch.Generator().manual_seed(1000)).cuda()
    ref = []
    for _ in range(8):
        loss = bl.nn.causal_lm_loss(m, x)
        loss.backward()
        opt.step()
        ref.append(loss.item())
    assert abs(piped[0] - ref[0]) <= 1e-3 * ref[0]
    assert piped[-1] < 0.8 * piped[0]
    assert all(abs(a - b) <= 0.05 * a for a, b in zip(ref, piped)), (ref, piped)


def _pipe(dense_dir, tmp_path, tag, extra):
    out = tmp_path / f"{tag}.jsonl"
    args = ["--model", dense_dir, "--steps", "8", "--batch", "4", "--micro", "2", "--seq", "127", "--preset", "muon",
            "--lr", "3e-3", "--data", "fixed", "--json", str(out)] + extra
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", str(GPUS),
                    os.path.join(ROOT, "tests", "pipeline_worker.py"), *args], check=True, env=env, cwd=ROOT,
                   timeout=600)
    return json.loads(out.read_text().splitlines()[-1])["losses"]


@pytest.mark.skipif(not torch.cuda.is_available() or not bl.backend.fp4_supported(0), reason="needs FP4 tensor cores")
def test_pipeline_deferred_weight_gradients(dense_dir, tmp_path):
    fp4 = ["--compute", "quartet2_fast"]
    plain = _pipe(dense_dir, tmp_path, "plain", fp4)
    deferred = _pipe(dense_dir, tmp_path, "deferred", fp4 + ["--defer-wgrad"])
    assert abs(plain[0] - deferred[0]) <= 1e-3 * plain[0]
    assert deferred[-1] < 0.9 * deferred[0]
    assert all(abs(a - b) <= 0.05 * a for a, b in zip(plain, deferred)), (plain, deferred)


def test_shared_head_matches_the_head_process(dense_dir, tmp_path):
    base = _pipe(dense_dir, tmp_path, "frozen", ["--freeze-head", "--micro", "4"])
    shared = _pipe(dense_dir, tmp_path, "shared", ["--share-head", "--micro", "4"])
    assert abs(base[0] - shared[0]) <= 1e-5 * base[0]
    assert all(abs(a - b) <= 1e-3 * a for a, b in zip(base, shared)), (base, shared)
    norms = [torch.tensor(json.loads((tmp_path / f"{tag}.jsonl").read_text().splitlines()[-1])["norm"])
             for tag in ("frozen", "shared")]
    assert torch.allclose(*norms, rtol=1e-3, atol=1e-3), norms


def test_overlapped_updates_match(dense_dir, tmp_path):
    base = _pipe(dense_dir, tmp_path, "inline", [])
    overlapped = _pipe(dense_dir, tmp_path, "overlap", ["--overlap"])
    assert all(abs(a - b) <= 1e-3 * a for a, b in zip(base, overlapped)), (base, overlapped)


@pytest.mark.skipif(GPUS < 2, reason="needs two GPUs")
@pytest.mark.parametrize("shared", [False, True])
def test_host_transport_matches_nccl(dense_dir, tmp_path, shared):
    extra = ["--micro", "4"] + (["--share-head"] if shared else [])
    nccl = _pipe(dense_dir, tmp_path, "nccl", extra)
    host = _pipe(dense_dir, tmp_path, "host", extra + ["--transport", "host"])
    assert nccl == host, (nccl, host)
