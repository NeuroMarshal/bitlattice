"""One process of the two-process pipeline test, started by torch.distributed.run from test_pipeline.py.

Trains a local checkpoint on random token batches (or the first batch repeated with --data fixed); every process
draws the same batches. Rank 0 appends the losses to --json.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist

import bitlattice as bl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="muon")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--seq", type=int, default=12)
    ap.add_argument("--compute", default="exact", choices=sorted(bl.COMPUTE_PRESETS))
    ap.add_argument("--data", default="random", choices=["random", "fixed"])
    ap.add_argument("--activations", default="gpu", choices=["gpu", "ram"])
    ap.add_argument("--defer-wgrad", action="store_true")
    ap.add_argument("--freeze-head", action="store_true")
    ap.add_argument("--share-head", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--transport", default="nccl", choices=["nccl", "host"])
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    bl.parallel.init(transport=a.transport)
    model = bl.io.load(a.model, bl.quant.presets.default("nvfp4"), device_map="interleaved")
    bl.surgery.set_compute_(model, "*", a.compute, skip_unfit=True)
    model.config.use_cache = False
    bl.surgery.convert_(model, "re:layers\\.\\d+$", "checkpoint")
    model.train()
    if a.freeze_head or a.share_head:
        bl.surgery.freeze_(model, "lm_head")
    if a.share_head:
        bl.parallel.share_head_(model)
    opt = bl.optim.Lattice(model, a.preset, lr=a.lr, fused_backward=True, seed=0, overlap=a.overlap)
    runner = bl.Runner(model, opt, micro=a.micro, memory=bl.Memory(activations=a.activations),
                       defer_wgrad=a.defer_wgrad)
    g = torch.Generator().manual_seed(1000)
    losses = []
    for step in range(a.steps):
        if a.data == "random" or step == 0:
            x = torch.randint(0, model.config.vocab_size, (a.batch, a.seq + 1), generator=g)
        losses.append(runner.step(x).item())
    if bl.parallel.is_main():
        with open(a.json, "a") as f:
            f.write(json.dumps({"losses": losses}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
