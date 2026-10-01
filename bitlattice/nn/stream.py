"""Row tiles of a matrix kept in pinned host memory, streamed to a GPU.

    for v0, v1, tile in stream_rows(W_host, rows=4096, device=x.device):
        y[:, v0:v1] = x @ tile.t()

Two device buffers alternate: while the current stream computes with one tile, the next is copied into the other on
a side stream (its copy engine works alongside the compute). A buffer is refilled only after the compute stream has
finished with it. The ``host_linear`` conversion keeps a frozen layer's weight this way; :func:`linear_cross_entropy`
streams such a weight once per group of tokens.
"""
from __future__ import annotations

import torch

__all__ = ["stream_rows"]


def stream_rows(W: torch.Tensor, rows: int, device: torch.device):
    """Yield (v0, v1, tile) for consecutive row ranges [v0, v1) of the host matrix ``W``, ``tile`` = W[v0:v1] on
    ``device`` (valid until the next tile is requested)."""
    device = torch.device(device)
    V = W.shape[0]
    rows = max(1, min(rows, V))
    starts = list(range(0, V, rows))
    side = torch.cuda.Stream(device=device)
    bufs = [torch.empty(rows, *W.shape[1:], dtype=W.dtype, device=device) for _ in range(min(2, len(starts)))]
    free = [None] * len(bufs)                         # event: the compute stream is done with the buffer

    def load(k):
        b, v0 = k % len(bufs), starts[k]
        v1 = min(V, v0 + rows)
        with torch.cuda.stream(side):
            if free[b] is not None:
                side.wait_event(free[b])
            bufs[b][:v1 - v0].copy_(W[v0:v1], non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(side)
        return ready

    cur = torch.cuda.current_stream(device)
    # the buffers were allocated on the current stream, whose queued work may still read that memory (freed by
    # earlier tensors): the side stream writes only after it
    side.wait_stream(cur)
    try:
        nxt = load(0)
        for k, v0 in enumerate(starts):
            ready = nxt
            cur.wait_event(ready)
            if k + 1 < len(starts):
                nxt = load(k + 1) if len(bufs) > 1 else None
            v1 = min(V, v0 + rows)
            yield v0, v1, bufs[k % len(bufs)][:v1 - v0]
            done = torch.cuda.Event()
            done.record(cur)
            free[k % len(bufs)] = done
            if len(bufs) == 1 and k + 1 < len(starts):
                nxt = load(k + 1)
    finally:
        cur.wait_stream(side)                         # no copy may still write into the buffers once they are freed
