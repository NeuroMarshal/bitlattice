"""Host links obey CUDA stream order, preserve messages across ring wraps and close with pending transfers."""
import time
import uuid

import pytest
import torch

import bitlattice  # registers the native operations
from bitlattice.transport import HostLink

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def links():
    tag = uuid.uuid4().hex
    device = torch.device("cuda", torch.cuda.current_device())
    for a, b in ((0, 1), (1, 0)):
        torch.ops.bitlattice.ring_create(f"/bitlattice-{tag}-{a}-{b}", 4096, 2)
    a = HostLink(0, 1, device, tag=tag, slot_bytes=4096, slots=2)
    b = HostLink(1, 0, device, tag=tag, slot_bytes=4096, slots=2)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def test_receive_waits_for_previous_stream_users(links):
    a, b = links
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        x = torch.full((1024,), 7, device="cuda", dtype=torch.int32)
        a.send(x)
    out = torch.empty_like(x)
    # A reused allocation can still have a writer queued on its allocating stream.
    torch.cuda._sleep(100_000_000)
    out.fill_(-1)
    ticket = torch.ops.bitlattice.link_recv(b._h, out)
    torch.ops.bitlattice.link_take(b._h, ticket)
    assert torch.equal(out, x)


def test_messages_cross_slots_and_wrap(links):
    a, b = links
    for i, n in enumerate((0, 1, 1023, 4097, 17, 8193, 2048)):
        x = torch.arange(n, device="cuda", dtype=torch.int32) + i
        y = b.recv(x.shape, x.dtype)
        a.send(x)
        assert torch.equal(y.take(), x)


def test_close_with_a_full_ring(links):
    a, _ = links
    a.send(torch.zeros(16384, device="cuda"))
    time.sleep(0.05)
    a.close()
    a.close()
