"""Kernel benchmarks: optimizer step and linear layer.  python bench/bench_kernels.py [--rows R --cols C]"""
from __future__ import annotations

import argparse
import json
import time

import torch

import bitlattice as bl
from bitlattice import _ops


def timeit(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def bench_step(rows, cols):
    out = {}
    g = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 1e-3
    # AdamW (fused) on bf16 weights with fp32 master
    p = torch.nn.Parameter(torch.randn(rows, cols, device="cuda") * 0.02)
    p.grad = g.float()
    adam = torch.optim.AdamW([p], lr=1e-3, fused=True)
    out["adamw_fused_fp32"] = timeit(adam.step)
    for fmt in ["nvfp4", "int4", "int8"]:
        w = bl.LatticeTensor.from_float(torch.randn(rows, cols, device="cuda") * 0.02, fmt, method="rtn", dtype=torch.bfloat16)
        for direction in ["factored", "muon_block"]:
            q = torch.nn.Parameter(w.clone())
            opt = bl.Lattice([q], lr=1e-3, direction=direction, phase_bits=2, block=512)
            q.grad = g

            def step():
                q.grad = g
                opt.step()
            out[f"lattice_{fmt}_{direction}"] = timeit(step)
        st = {}
        D = _ops.direction(g, "factored", torch.zeros(rows, device="cuda"), torch.zeros(cols, device="cuda"), 1)
        phase = torch.zeros(rows, cols // 4, dtype=torch.uint8, device="cuda")
        packed = w.packed.clone()
        kernel = lambda: _ops.step_quantized(packed, phase, w.scales, w.global_scale, w.zeros, D, w.fmt, rows, cols,
                                             lr=1e-3, weight_decay=0.0, phase_bits=2, phase_cb=2, rounding="sr",
                                             seed=0, step=1, strata=16, scale_rule="fit", trust=0.01, trust_lr=1.0)
        t = timeit(kernel)
        moved = rows * cols * (4 + 2 * w.fmt.container_bits / 8 + 2 * 2 / 8) + 2 * w.scales.numel() * \
            w.scales.element_size()
        out[f"kernel_{fmt}"] = t
        out[f"kernel_{fmt}_GBps"] = moved / t / 1e9
    for direction in ["factored", "muon_block", "muon"]:
        r, c = torch.zeros(rows, device="cuda"), torch.zeros(cols, device="cuda")
        out[f"direction_{direction}"] = timeit(lambda: _ops.direction(g, direction, r, c, 1, block=512), reps=5)
    wb = torch.nn.Parameter(torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.02)
    ob = bl.Lattice([wb], lr=1e-3, direction="factored", phase_bits=2)

    def stepb():
        wb.grad = g
        ob.step()
    out["lattice_bf16_native_factored"] = timeit(stepb)
    return out


def bench_linear(tokens, rows, cols):
    out = {}
    x = torch.randn(tokens, cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    wd = torch.nn.Parameter(torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.02)
    for name, w in [("bf16", wd)] + [(f, torch.nn.Parameter(bl.LatticeTensor.from_float(wd.detach().float(), f,
                                                                                         dtype=torch.bfloat16)))
                                     for f in ["nvfp4", "int4", "int8"]]:
        out[f"linear_fwd_{name}"] = timeit(lambda: torch.nn.functional.linear(x, w))

        def fb():
            y = torch.nn.functional.linear(x, w)
            y.backward(torch.ones_like(y))
            x.grad = None
            w.grad = None
        out[f"linear_fwdbwd_{name}"] = timeit(fb)
    return out


def bench_fp4(tokens=4096, rows=4096, cols=4096):
    if not bl.backend.fp4_supported(0):
        return {}
    x = torch.randn(tokens, cols, device="cuda", dtype=torch.bfloat16)
    wd = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.02
    w = torch.nn.Parameter(bl.LatticeTensor.from_float(wd.float(), "nvfp4", dtype=torch.bfloat16))
    out = {"linear_fwd_bf16_dense": timeit(lambda: torch.nn.functional.linear(x, wd))}
    with torch.no_grad():
        out["linear_fwd_nvfp4_exact"] = timeit(lambda: torch.nn.functional.linear(x, w))
        with bl.compute("fp4"):
            out["linear_fwd_nvfp4_fp4"] = timeit(lambda: torch.nn.functional.linear(x, w))
    xg = x.clone().requires_grad_(True)
    wdg = wd.clone().requires_grad_(True)
    gy = torch.randn(tokens, rows, device="cuda", dtype=torch.bfloat16)

    def fwd_bwd(weight):
        def run():
            torch.nn.functional.linear(xg, weight).backward(gy)
            xg.grad = None
            weight.grad = None
        return run

    out["linear_fwdbwd_bf16_dense"] = timeit(fwd_bwd(wdg))
    for pol in ("exact", "fp4", "fp4_all"):
        with bl.compute(pol):
            out[f"linear_fwdbwd_nvfp4_{pol}"] = timeit(fwd_bwd(w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=4096)
    a = ap.parse_args()
    res = {"gpu": torch.cuda.get_device_name(0), "rows": a.rows, "cols": a.cols, "tokens": a.tokens}
    res.update(bench_step(a.rows, a.cols))
    res.update(bench_linear(a.tokens, a.rows, a.cols))
    res.update(bench_fp4(a.tokens, a.rows, a.cols))
    for k, v in res.items():
        print(f"{k:40s} {v * 1e3:10.3f} ms" if isinstance(v, float) and not k.endswith("GBps") else f"{k:40s} {v}")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
