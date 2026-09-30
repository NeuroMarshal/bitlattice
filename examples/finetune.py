"""Fine-tune a Hugging Face causal LM directly in a low-bit format.

python examples/finetune.py --model Qwen/Qwen3-0.6B --format nvfp4 --keep 0.05 --data train.txt \
    --preset muon --lr 3e-5 --steps 500 --seq 1024 --batch 4 --storage gpu --out out/
"""
from __future__ import annotations

import argparse
import math
import time

import torch

import bitlattice as bl


def batches(ids: torch.Tensor, seq: int, batch: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    while True:
        starts = torch.randint(0, ids.numel() - seq - 1, (batch,), generator=g)
        yield torch.stack([ids[s:s + seq + 1] for s in starts])


@torch.no_grad()
def evaluate(model, ids: torch.Tensor, seq: int) -> float:
    model.eval()
    tot = n = 0
    for s in range(0, ids.numel() - 1, seq):
        x = ids[s:s + seq + 1].unsqueeze(0).cuda()
        if x.shape[1] < 2:
            break
        tot += model(x[:, :-1], labels=x[:, 1:].contiguous()).loss.item() * (x.shape[1] - 1)
        n += x.shape[1] - 1
    model.train()
    return math.exp(tot / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="nvfp4", help="target format for dense checkpoints ('none' keeps dense)")
    ap.add_argument("--keep", type=float, default=0.0, help="fraction of input channels kept in BF16")
    ap.add_argument("--data", required=True, help="plain text file")
    ap.add_argument("--preset", default="muon")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--storage", default="gpu", choices=bl.storage.POLICIES)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = torch.tensor(tok(open(a.data, encoding="utf-8").read()).input_ids)
    n_eval = int(ids.numel() * a.eval_frac)
    train, held = ids[:-n_eval], ids[-n_eval:]

    quant = None if a.format == "none" else bl.quant.presets.default(a.format, keep=a.keep)
    model = bl.io.load(a.model, quant, device="cuda")
    if a.checkpointing or a.storage == "state_and_acts_in_ram":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = bl.optim.build(model, a.preset, lr=a.lr, fused_backward=True, storage=a.storage)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.steps // 20)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    print(bl.plan(model, opt))
    print(f"held-out perplexity before: {evaluate(model, held, a.seq):.3f}")
    it = batches(train, a.seq, a.batch, 0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x = next(it).cuda()
        with bl.storage.maybe_activations_in_ram(a.storage):
            loss = model(x[:, :-1], labels=x[:, 1:].contiguous()).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        sched.step()
        if step % 10 == 0 or step == a.steps:
            tok_s = step * a.seq * a.batch / (time.time() - t0)
            print(f"step {step:5d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {tok_s:,.0f} tok/s "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
    print(f"held-out perplexity after: {evaluate(model, held, a.seq):.3f}")
    if a.out:
        bl.io.save(model, a.out)
        tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
