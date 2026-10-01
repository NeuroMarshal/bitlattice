"""One process per GPU: the interleaved pipeline reproduces single-GPU training (needs two GPUs)."""
import json
import os
import subprocess
import sys

import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                                reason="needs two GPUs")
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


@pytest.mark.parametrize("offload", [False, True])
def test_two_process_pipeline_matches_single_gpu(dense_dir, tmp_path, offload):
    out = tmp_path / "pipe.jsonl"
    args = ["--model", dense_dir, "--steps", "8", "--batch", "4", "--micro", "2", "--seq", "12", "--preset", "muon",
            "--lr", "3e-3", "--data", "fixed", "--json", str(out)] + (["--offload"] if offload else [])
    env = dict(os.environ, PYTHONPATH=ROOT)
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "2",
                    os.path.join(ROOT, "bench", "pipeline_train.py"), *args], check=True, env=env, cwd=ROOT,
                   timeout=600)
    piped = json.loads(out.read_text().splitlines()[-1])["losses"]

    m = bl.io.load(dense_dir, bl.quant.presets.default("nvfp4"), device="cuda")
    m.config.use_cache = False
    m.gradient_checkpointing_enable()
    m.train()
    opt = bl.optim.build(m, "muon", lr=3e-3, fused_backward=True, seed=0)
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
