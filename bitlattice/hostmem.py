"""Page-locked host memory of exact size.

PyTorch's pinned-memory allocator rounds every request up to a power of two (a 665 MiB buffer takes 1 GiB of host
RAM), which adds up for large host-resident tensors: frozen tables, offloaded activations, optimizer state. Here a
plain host allocation of exactly the requested size is page-locked with ``cudaHostRegister`` -- asynchronous copies
from and to it are as fast -- and unregistered when its storage is freed.

    t = pinned_empty((rows, cols), torch.bfloat16)
    h = to_pinned(gpu_tensor)                  # an exact-size pinned copy
    pool = HostPool()                          # reuse of buffers whose last copy has completed
    buf = pool.take(shape, dtype); ...; pool.give(buf, done_event)
"""
from __future__ import annotations

import weakref

import torch

__all__ = ["pinned_empty", "to_pinned", "HostPool"]


def _unregister(ptr: int) -> None:
    try:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:                                  # interpreter shutdown: CUDA may already be gone
        pass


def pinned_empty(shape, dtype: torch.dtype) -> torch.Tensor:
    """An uninitialized host tensor of exactly ``shape`` x ``dtype`` bytes, page-locked (plain host memory without
    CUDA)."""
    t = torch.empty(shape, dtype=dtype, device="cpu")
    if not torch.cuda.is_available() or t.numel() == 0:
        return t
    st = t.untyped_storage()
    err = torch.cuda.cudart().cudaHostRegister(st.data_ptr(), st.nbytes(), 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister of {st.nbytes()} bytes failed ({err}); the host may be out of "
                           "lockable memory (ulimit -l)")
    weakref.finalize(st, _unregister, st.data_ptr())
    return t


def to_pinned(t: torch.Tensor) -> torch.Tensor:
    """An exact-size page-locked host copy of ``t`` (complete on return)."""
    h = pinned_empty(t.shape, t.dtype)
    h.copy_(t.detach())
    return h


class HostPool:
    """Exact-size pinned buffers, reused once the copy that last read or wrote them has completed."""

    def __init__(self):
        self._free: dict = {}          # (shape, dtype) -> [(buffer, event or None)]

    def take(self, shape, dtype: torch.dtype) -> torch.Tensor:
        key = (tuple(shape), dtype)
        free = self._free.get(key, [])
        for i, (buf, ev) in enumerate(free):
            if ev is None or ev.query():
                del free[i]
                return buf
        return pinned_empty(shape, dtype)

    def give(self, buf: torch.Tensor, event: torch.cuda.Event | None = None) -> None:
        """Return ``buf``; ``event`` marks the completion of the last copy that uses it."""
        self._free.setdefault((tuple(buf.shape), buf.dtype), []).append((buf, event))

    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for bufs in self._free.values() for b, _ in bufs)
