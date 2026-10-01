"""Exact-size page-locked host memory."""
import gc

import pytest
import torch

from bitlattice.hostmem import HostPool, pinned_empty, to_pinned

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_pinned_empty_is_exact_and_pinned():
    t = pinned_empty((3, 1000), torch.bfloat16)
    assert t.is_pinned() and t.untyped_storage().nbytes() == 6000
    g = torch.randn(3, 1000, device="cuda").to(torch.bfloat16)
    t.copy_(g, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(t, g.cpu())
    assert torch.equal(to_pinned(g), g.cpu())
    view = t[1:]                                   # the storage outlives the tensor that created it
    del t
    gc.collect()
    assert view.is_pinned()


def test_pool_reuses_completed_buffers_only():
    pool = HostPool()
    a = pool.take((4, 4), torch.float32)
    ev = torch.cuda.Event()
    ev.record()
    ev.synchronize()
    pool.give(a, ev)
    assert pool.take((4, 4), torch.float32) is a
    assert pool.take((4, 4), torch.float32) is not a


def test_streamed_rows_are_exact_while_the_compute_stream_is_busy():
    from bitlattice.nn.stream import stream_rows
    torch.manual_seed(0)
    W = torch.randn(4096, 512, dtype=torch.bfloat16)
    Wh, Wg = to_pinned(W), W.cuda()
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    ref = torch.cat([x @ Wg[v:v + 512].t() for v in range(0, 4096, 512)], 1)
    for _ in range(10):
        junk = torch.randn(2048, 2048, device="cuda")
        for _ in range(3):
            junk = junk @ junk / 32                    # queued work reading memory freed just below
        del junk
        out = torch.empty_like(ref)
        for v0, v1, t in stream_rows(Wh, 512, x.device):
            out[:, v0:v1] = x @ t.t()
        assert torch.equal(out, ref)
