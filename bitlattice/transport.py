"""Transfers between the processes of one host through shared page-locked memory.

GPUs without peer access (consumer cards) exchange data through host memory. NCCL does this with copy kernels that
occupy SMs and with a proxy thread per process that must keep running, so the host can only run a few units of work
ahead of its GPU. A :class:`HostLink` uses a ring of slots in shared memory per direction instead: the copy engines
move the data, and a host thread per direction (C++, no interpreter lock) watches the sequence flags of the ring and
issues each copy when it can run at once (``csrc/transport.cu``). No GPU stream ever waits for the other process,
and the calling thread waits only for data it asks to use before it has arrived.

    link = HostLink(me, peer, device)          # every pair of processes: both call it, in the same order
    link.send(x)                               # after the work that produced x on the current stream
    y = link.recv(shape, dtype).take()         # the current stream waits for the copy on the GPU
"""
from __future__ import annotations

import os

import torch

__all__ = ["HostLink"]


class HostLink:
    """Both directions between this process and ``peer``. Messages of a direction are received in the order they
    were sent. ``slot_bytes`` x ``slots`` of shared memory per direction; a message takes ceil(bytes / slot_bytes)
    slots, and a sender's copy waits only when all slots hold messages the receiver has not taken yet."""

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
