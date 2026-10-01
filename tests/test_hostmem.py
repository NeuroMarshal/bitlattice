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
