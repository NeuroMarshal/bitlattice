"""Multi-process pipeline: one process per GPU, the blocks of a decoder-only model interleaved over the GPUs.

Each process holds only its own blocks (``bl.io.load(..., device_map="interleaved")`` after :func:`init` leaves the
rest on the meta device) and drives only its own GPU, so no process waits on another's command queue or on a shared
interpreter lock. The model is cut into segments: maximal runs of consecutive units (embedding, blocks, output
head + loss) on one GPU. Activations and gradients cross between segments with asynchronous NCCL send/receive; one
communicator per direction keeps the in-order NCCL streams from deadlocking.

Schedule of :func:`train_step` for ``micro`` micro-batches: every process runs its segments in model order, each
segment for all micro-batches (forward), then in reverse order (backward). A segment's micro-batch m is sent on as
soon as it is computed, so GPU b works on segment s+1 of micro-batch m while GPU a works on segment s of m+1. In the
backward pass the gradients of a segment's parameters are accumulated over the micro-batches and the fused
optimizer applies them after the last one: one segment of gradients in memory, the update of the full batch.

Consumer GPUs have no peer-to-peer access; NCCL then stages transfers through host memory with copy kernels
(~5 GB/s per direction here). ``NCCL_SHM_USE_CUDA_MEMCPY=1`` moves them with the copy engines (~18 GB/s) but
serves all transfers of a process from one proxy thread in order: a send posted before its receive blocks the
others, and with many transfers in flight the two processes deadlock. It is therefore not used.

    torchrun --nproc-per-node 2 train.py      # every process runs the same script on the same batches
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from . import parallel as P

__all__ = ["init", "train_step", "is_main"]

_GROUPS: dict = {}
_DEBUG = os.environ.get("BL_PIPELINE_DEBUG", "") == "1"


def _log(*a):
    if _DEBUG:
        print(f"[rank {dist.get_rank()}]", *a, flush=True)


def init(backend: str = "nccl") -> tuple[int, int]:
    """Start the process group for a one-process-per-GPU run (environment from torchrun). Returns (rank, world)."""
    os.environ.setdefault("TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false")   # one group per direction
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend, device_id=torch.device("cuda", local))
    return dist.get_rank(), dist.get_world_size()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _group(src: int, dst: int):
    """Communicator for transfers from ``src`` to ``dst`` (created collectively for all ordered pairs at once)."""
    if not _GROUPS:
        w = dist.get_world_size()
        for i in range(w):
            for j in range(w):
                if i != j:
                    _GROUPS[(i, j)] = dist.new_group([i, j])
    return _GROUPS[(src, dst)]


@dataclass
class _Segment:
    owner: int
    items: list = field(default_factory=list)       # ("embed",) | ("block", i) | ("head",)
    prev: int | None = None
    next: int | None = None


def _module_name(model, module) -> str:
    return next(n for n, m in model.named_modules() if m is module)


def _plan(model) -> list[_Segment]:
    dm = model.bl_device_map
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if emb.weight is head.weight:
        raise NotImplementedError("tied input and output embeddings are not supported by the pipeline")
    owner = lambda name: P.device_of(name, dm).index  # noqa: E731
    blocks_name = _module_name(model, backbone.layers)
    units = [(("embed",), owner(_module_name(model, emb)))]
    units += [(("block", i), owner(f"{blocks_name}.{i}")) for i in range(len(backbone.layers))]
    units += [(("head",), owner(_module_name(model, head)))]
    segs: list[_Segment] = []
    for item, o in units:
        if not segs or segs[-1].owner != o:
            segs.append(_Segment(o))
        segs[-1].items.append(item)
    for a, b in zip(segs, segs[1:]):
        a.next, b.prev = b.owner, a.owner
    return segs


def _block_kwargs(backbone, blocks, ids, embed: bool, width: int, dtype, device):
    """(hidden or None, [kwargs per block]) prepared by the model on this process's device."""
    if embed:
        return P._block_inputs(backbone, blocks, ids, {})
    dummy = torch.empty(*ids.shape, width, dtype=dtype, device=device)
    with torch.no_grad():
        _, kws = P._block_inputs(backbone, blocks, None, {}, inputs_embeds=dummy)
    return None, kws


class _RunAhead:
    """Bounds how far the host runs ahead of its GPU: at most ``window`` units (segment x micro-batch) are queued.
    Without the bound the command queue fills up, the host blocks inside a kernel launch, and NCCL's proxy thread,
    which drives the host-staged copies with CUDA calls, can no longer make progress: the two processes deadlock."""

    def __init__(self, window: int):
        self.window, self.events = window, []

    def tick(self):
        ev = torch.cuda.Event()
        ev.record()
        self.events.append(ev)
        if len(self.events) > self.window:
            self.events.pop(0).synchronize()


def train_step(model, input_ids: torch.Tensor, labels: torch.Tensor | None = None, *, micro: int = 4,
               optimizer=None, backward: bool = True, chunk: int = 1024, run_ahead: int = 3) -> torch.Tensor:
    """Forward (and backward) of a causal LM batch over all processes; returns the mean next-token loss (the same
    value on every process). Every process must call it with the same ``input_ids`` / ``labels``.

    optimizer -- the fused-backward optimizer of this process (``bl.optim.build(..., fused_backward=True)``): its
                 updates are deferred to the last micro-batch of each segment. With a non-fused optimizer pass None
                 and call ``optimizer.step()`` afterwards.
    backward  -- False: evaluation only (no graph, no update).
    run_ahead -- units of work (segment x micro-batch) the host may queue ahead of the GPU."""
    me = dist.get_rank()
    dev = model.bl_local_device
    segs = getattr(model, "_bl_pipeline_plan", None)
    if segs is None:
        segs = model._bl_pipeline_plan = _plan(model)
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    blocks = backbone.layers
    norm = next((getattr(backbone, n) for n in P._NORM_NAMES if hasattr(backbone, n)), None)
    head = model.get_output_embeddings()
    emb = model.get_input_embeddings()
    width, dtype = model.config.hidden_size, emb.weight.dtype
    labels = input_ids if labels is None else labels
    ids_mb = [t.to(dev) for t in input_ids.chunk(micro, 0)]
    lab_mb = [t.to(dev) for t in labels.chunk(micro, 0)]
    M = len(ids_mb)
    n_tok = [lb[:, 1:].numel() for lb in lab_mb]
    embed_here = segs[0].owner == me

    grad = torch.enable_grad() if backward else torch.no_grad()
    ins, outs, losses, keep = {}, {}, {}, []
    ahead = _RunAhead(max(1, run_ahead))
    with grad:
        prep = [_block_kwargs(backbone, blocks, ids, embed_here, width, dtype, dev) for ids in ids_mb]
        for s, seg in enumerate(segs):
            if seg.owner != me:
                continue
            for m in range(M):
                shape = (*ids_mb[m].shape, width)
                y = None
                if seg.prev is not None:
                    y = torch.empty(shape, dtype=dtype, device=dev)
                    _log("fwd recv", s, m, "from", seg.prev)
                    dist.irecv(y, seg.prev, group=_group(seg.prev, me)).wait()
                    if backward:
                        y.requires_grad_(True)
                    ins[(s, m)] = y
                for item in seg.items:
                    if item[0] == "embed":
                        y = prep[m][0]
                    elif item[0] == "block":
                        out = blocks[item[1]](y, **prep[m][1][item[1]])
                        y = out[0] if isinstance(out, tuple) else out
                    else:
                        from .nn.loss import linear_cross_entropy
                        h = norm(y) if norm is not None else y
                        losses[m] = linear_cross_entropy(h[:, :-1], head.weight, lab_mb[m][:, 1:],
                                                         getattr(head, "bias", None), chunk) * (n_tok[m] / sum(n_tok))
                        y = None
                if y is not None:
                    outs[(s, m)] = y
                    if seg.next is not None:
                        send = y.detach()
                        _log("fwd send", s, m, "to", seg.next)
                        keep.append((dist.isend(send, seg.next, group=_group(me, seg.next)), send))
                ahead.tick()
        if backward:
            for s in reversed(range(len(segs))):
                seg = segs[s]
                if seg.owner != me:
                    continue
                for m in range(M):
                    defer = optimizer.accumulate() if optimizer is not None and m < M - 1 else contextlib.nullcontext()
                    with defer:
                        if m in losses and ("head",) in seg.items:
                            torch.autograd.backward(losses[m])
                        else:
                            g = torch.empty_like(outs[(s, m)])
                            _log("bwd recv", s, m, "from", seg.next)
                            dist.irecv(g, seg.next, group=_group(seg.next, me)).wait()
                            torch.autograd.backward(outs[(s, m)], g)
                    outs.pop((s, m), None)
                    if seg.prev is not None:
                        gi = ins.pop((s, m)).grad
                        _log("bwd send", s, m, "to", seg.prev)
                        keep.append((dist.isend(gi, seg.prev, group=_group(me, seg.prev)), gi))
                    ahead.tick()
    _log("step done, waiting for", len(keep), "sends")
    for work, _ in keep:
        work.wait()
    total = (sum(losses.values()).detach().float() if losses else torch.zeros((), device=dev)).reshape(1)
    dist.all_reduce(total)
    return total[0]
