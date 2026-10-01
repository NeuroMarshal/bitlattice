"""Optimizer option ablation on a small byte-level language model trained from scratch in a lattice format.

python bench/optimizer_ablation.py --format nvfp4 --steps 300 --seeds 0 1 2 \
    --variant default --variant "pe_rect:block=(256,1024),ns_dtype='fp16',ns_coeffs='polar_express'"
Each variant is ``name:key=value,...`` (Python literals) applied on top of the preset. Reports held-out loss (mean
and spread over seeds) and optimizer time per step.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sysconfig
import time

import torch

import bitlattice as bl


def corpus(max_bytes: int) -> torch.Tensor:
    """Bytes of the Python standard library sources (available everywhere, deterministic order)."""
    root = sysconfig.get_paths()["stdlib"]
    buf, n = [], 0
    for f in sorted(glob.glob(os.path.join(root, "*.py"))):
        with open(f, "rb") as fh:
            b = fh.read()
        buf.append(b)
        n += len(b)
        if n >= max_bytes:
            break
    return torch.frombuffer(bytearray(b"".join(buf)[:max_bytes]), dtype=torch.uint8).long()


def parse_variant(s: str) -> tuple[str, dict]:
    name, _, rest = s.partition(":")
    kw = {}
    if rest:
        tree = ast.parse(f"f({rest})", mode="eval").body
        kw = {k.arg: ast.literal_eval(k.value) for k in tree.keywords}
    return name, kw


def model_for(a, seed: int):
    import transformers
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=a.hidden, intermediate_size=a.intermediate,
                                   num_hidden_layers=a.layers, num_attention_heads=a.hidden // 128,
                                   num_key_value_heads=max(1, a.hidden // 256), head_dim=128,
                                   max_position_embeddings=a.seq, tie_word_embeddings=False)
    torch.manual_seed(seed)
    with torch.device("cuda"):
        m = transformers.Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    if a.format != "bf16":
        bl.surgery.quantize_(m, "*", bl.quant.presets.default(a.format))
        bl.surgery.set_compute_(m, "*", a.compute, skip_unfit=True)
    m.config.use_cache = False
    return m.train()


def run(a, name: str, kw: dict, seed: int, train: torch.Tensor, held: torch.Tensor) -> dict:
    m = model_for(a, seed)
    opt = bl.optim.Lattice(m, a.preset, lr=a.lr, fused_backward=True, seed=seed, **kw)
    g = torch.Generator().manual_seed(1000 + seed)
    warm = max(1, a.steps // 20)
    t0 = time.perf_counter()
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for grp in opt.param_groups:
            grp["lr"] = lr
        starts = torch.randint(0, train.numel() - a.seq - 1, (a.batch,), generator=g)
        x = torch.stack([train[s:s + a.seq + 1] for s in starts]).cuda()
        loss = bl.nn.causal_lm_loss(m, x)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    m.eval()
    tot = n = 0
    with torch.no_grad():
        for s in range(0, held.numel() - a.seq - 1, a.seq):
            x = held[s:s + a.seq + 1].unsqueeze(0).cuda()
            tot += bl.nn.causal_lm_loss(m, x).item()
            n += 1
    return dict(variant=name, seed=seed, held_loss=tot / n, train_loss=loss.item(), wall_s=wall, options=kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="nvfp4")
    ap.add_argument("--preset", default="muon")
    ap.add_argument("--compute", default="exact", choices=sorted(bl.COMPUTE_PRESETS))
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--intermediate", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--bytes", type=int, default=8 << 20)
    ap.add_argument("--out", default="bench/results/optimizer_ablation.jsonl")
    a = ap.parse_args()
    data = corpus(a.bytes)
    n_held = data.numel() // 20
    train, held = data[:-n_held], data[-n_held:][: 64 * a.seq + 1]
    variants = [parse_variant(v) for v in (a.variant or ["default"])]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    results = []
    for seed in a.seeds:
        for name, kw in variants:
            r = run(a, name, kw, seed, train, held)
            r.update(format=a.format, preset=a.preset, steps=a.steps, compute=a.compute)
            results.append(r)
            print(json.dumps(r), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            torch.cuda.empty_cache()
    print("\nvariant                 held loss (mean ± sd)   wall s")
    for name, _ in variants:
        rs = [r for r in results if r["variant"] == name]
        hl = torch.tensor([r["held_loss"] for r in rs])
        sd = hl.std().item() if len(rs) > 1 else 0.0
        print(f"{name:22s}  {hl.mean().item():.4f} ± {sd:.4f}        {sum(r['wall_s'] for r in rs) / len(rs):.1f}")


if __name__ == "__main__":
    main()
