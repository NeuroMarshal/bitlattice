r"""Fine-tune a Hugging Face causal LM directly in a low-bit format.

python examples/finetune.py --model Qwen/Qwen3-0.6B --format nvfp4 --keep 0.05 --data train.txt \
    --preset muon --lr 3e-5 --steps 500 --seq 1024 --batch 4 --out out/

Every change to the model is an explicit line below; nothing else happens to it. Several GPUs: --device-map auto
(contiguous layer split). Long contexts: --checkpoint 're:layers\.\d+$' --mlp-chunk 4096 --activations ram.
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


def perplexity(runner: bl.Runner, ids: torch.Tensor, seq: int, device) -> float:
    tot = n = 0
    for s in range(0, ids.numel() - 1, seq):
        x = ids[s:s + seq + 1].unsqueeze(0).to(device)
        if x.shape[1] < 2:
            break
        tot += runner.loss(x).item() * (x.shape[1] - 1)
        n += x.shape[1] - 1
    return math.exp(tot / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="nvfp4", help="target format for dense checkpoints ('none' keeps dense)")
    ap.add_argument("--keep", type=float, default=0.0, help="fraction of input channels kept in BF16")
    ap.add_argument("--data", required=True, help="plain text file")
    ap.add_argument("--preset", default="muon", help="optimizer preset: " + ", ".join(sorted(bl.optim.PRESETS)))
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4, help="sequences per update")
    ap.add_argument("--micro", type=int, default=1, help="micro-batches the batch is split into")
    ap.add_argument("--compute", default="exact", choices=sorted(bl.COMPUTE_PRESETS),
                    help="how the quantized layers compute (set on the layers that support it)")
    ap.add_argument("--state", default="auto", choices=["gpu", "ram", "auto"], help="optimizer state placement")
    ap.add_argument("--activations", default="gpu", choices=["gpu", "ram"], help="saved activation placement")
    ap.add_argument("--device-map", default=None, help="'auto' to split the layers over all GPUs")
    ap.add_argument("--checkpoint", default="", help="selector of the modules recomputed in backward, "
                                                     "e.g. 're:layers\\.\\d+$'")
    ap.add_argument("--mlp-chunk", type=int, default=0, help="run MLPs on slices of this many tokens")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = torch.tensor(tok(open(a.data, encoding="utf-8").read()).input_ids)
    n_eval = int(ids.numel() * a.eval_frac)
    train, held = ids[:-n_eval], ids[-n_eval:]

    quant = None if a.format == "none" else bl.quant.presets.default(a.format, keep=a.keep)
    model = bl.io.load(a.model, quant, device="cuda", device_map=a.device_map)
    model.config.use_cache = False
    if a.mlp_chunk:
        bl.surgery.convert_(model, "*.mlp", "token_chunks", chunk=a.mlp_chunk)
    if a.checkpoint:
        bl.surgery.convert_(model, a.checkpoint, "checkpoint")
    bl.surgery.set_compute_(model, "*", a.compute, skip_unfit=True)
    model.train()
    print(bl.explain(model))

    opt = bl.optim.Lattice(model, a.preset, lr=a.lr, state=a.state, fused_backward=True)
    warmup = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps))))
    runner = bl.Runner(model, opt, micro=a.micro, memory=bl.Memory(activations=a.activations))
    print(bl.plan(model, opt))
    dev = model.get_input_embeddings().weight.device
    print(f"held-out perplexity before: {perplexity(runner, held, a.seq, dev):.3f}")
    it = batches(train, a.seq, a.batch, 0)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        loss = runner.step(next(it).to(dev))
        sched.step()
        if step % 10 == 0 or step == a.steps:
            tok_s = step * a.seq * a.batch / (time.time() - t0)
            peak = max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
            print(f"step {step:5d} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {tok_s:,.0f} tok/s "
                  f"peak {peak / 2 ** 30:.2f} GiB", flush=True)
    print(f"held-out perplexity after: {perplexity(runner, held, a.seq, dev):.3f}")
    if a.out:
        bl.io.save(model, a.out)
        tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
