"""Copy-engine transfers through shared pinned-memory rings for processes on one host.

C++ threads submit copies when the peer's sequence flags allow slot reuse. ``take`` waits on the host until a copy
has been issued, then orders the caller's CUDA stream after it.

    link = HostLink(me, peer, device)          # every pair of processes: both call it, in the same order
    link.send(x)                               # after the work that produced x on the current stream
    y = link.recv(shape, dtype).take()         # the current stream waits for the copy on the GPU
"""
from __future__ import annotations

import os

import torch

__all__ = ["HostLink"]


class HostLink:
    """Ordered messages to and from ``peer``, using ``slot_bytes * slots`` shared bytes per direction.
    Both peers must construct their links in the same order; messages larger than a slot span consecutive slots."""

    def __init__(self, me: int, peer: int, device: torch.device, *, tag: str | None = None,
                 slot_bytes: int = 4 << 20, slots: int = 64):
        import torch.distributed as dist
        tag = tag or os.environ.get("MASTER_PORT", "0")
        name = lambda a, b: f"/bitlattice-{tag}-{a}-{b}"  # noqa: E731
        self.device, self.peer = device, peer
        torch.ops.bitlattice.ring_create(name(peer, me), slot_bytes, slots)     # each side creates what it receives
        if dist.is_initialized():
            dist.barrier()
        self._h = torch.ops.bitlattice.link_open(name(peer, me), name(me, peer), slot_bytes, slots,
                                                 device.index if device.index is not None else 0)
        self._open = True
        import atexit
        atexit.register(self.close)                  # before the CUDA runtime goes away

    def send(self, x: torch.Tensor) -> None:
        """Send ``x`` (contiguous) once the work issued so far on the current stream is done; ``x`` may be released
        right after (the link keeps it until it has been copied)."""
        torch.ops.bitlattice.link_send(self._h, x)

    def recv(self, shape, dtype: torch.dtype) -> _Incoming:
        """Post the receipt of the next message (a tensor of ``shape`` and ``dtype``)."""
        out = torch.empty(shape, dtype=dtype, device=self.device)
        return _Incoming(self._h, out, torch.ops.bitlattice.link_recv(self._h, out))

    def close(self) -> None:
        if self._open:
            self._open = False
            torch.ops.bitlattice.link_close(self._h, True)


class _Incoming:
    __slots__ = ("h", "tensor", "ticket")

    def __init__(self, h: int, tensor: torch.Tensor, ticket: int):
        self.h, self.tensor, self.ticket = h, tensor, ticket

    def take(self) -> torch.Tensor:
        """The tensor, once the current stream has waited (on the GPU) for its copy."""
        torch.ops.bitlattice.link_take(self.h, self.ticket)
        return self.tensor
