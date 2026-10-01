"""Forward + backward time and gradient error of one NVFP4 linear layer per compute policy and FP4 recipe.

python bench/bench_fp4_layer.py --tokens 8192 --shapes 5120x17408 17408x5120 5120x6144
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import bitlattice as bl


def run(x, p, gy, policy, recipe):
    """One forward and backward with a persistent parameter (cached weight operands are reused, as in training
    between optimizer steps)."""
    p.grad = None
    xi = x.detach().requires_grad_(True)
    with bl.compute(policy), bl.fp4_recipe(recipe):
        y = torch.nn.functional.linear(xi, p)
        y.backward(gy)
    return y, xi.grad, p.grad


def timed(fn, n):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--shapes", nargs="+", default=["5120x17408", "17408x5120", "5120x6144"])
    ap.add_argument("--recipes", nargs="+", default=["nvidia", "4over6", "quartet2", "quartet2_4o6", "quartet2_fast"])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    torch.manual_seed(0)
    for shape in a.shapes:
        K, N = map(int, shape.split("x"))
        x = torch.randn(a.tokens, K, device="cuda", dtype=torch.bfloat16)
        gy = torch.randn(a.tokens, N, device="cuda", dtype=torch.bfloat16) * 0.01
        w = torch.nn.Parameter(bl.LatticeTensor.from_float(torch.randn(N, K, device="cuda") * K ** -0.5, "nvfp4",
                                                           method="rtn", dtype=torch.bfloat16))
        ref = run(x, w, gy, "exact", "nvidia")
        rows = [("exact", "-", timed(lambda: run(x, w, gy, "exact", "nvidia"), a.iters), None)]
        for r in a.recipes:
            got = run(x, w, gy, "fp4_all", r)
            err = {k: ((g.float() - e.float()).norm() / e.float().norm()).item()
                   for k, g, e in zip(("y", "dx", "dw"), got, ref)}
            rows.append(("fp4_all", r, timed(lambda: run(x, w, gy, "fp4_all", r), a.iters), err))
        for pol, r, ms, err in rows:
            rec = {"K": K, "N": N, "tokens": a.tokens, "policy": pol, "recipe": r, "ms": round(ms, 3)}
            if err:
                rec.update({k: round(v, 4) for k, v in err.items()})
            print(json.dumps(rec), flush=True)
        del x, gy, w, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
