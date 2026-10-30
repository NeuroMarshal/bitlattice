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
import functools
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from . import backend
from . import parallel as P
from .kernel_choice import active as active_kernels

__all__ = ["init", "is_main", "share_head_", "train_step"]

_GROUPS: dict = {}
_DEBUG = os.environ.get("BL_PIPELINE_DEBUG", "") == "1"
# BL_NVTX=1: NVTX ranges per unit and micro-batch, for timelines from Nsight Systems
_NVTX = os.environ.get("BL_NVTX", "") == "1"


def _range(name: str):
    if not _NVTX:
        return contextlib.nullcontext()
    return torch.cuda.nvtx.range(name)


def _log(*a):
    if _DEBUG:
        print(f"[rank {dist.get_rank()}]", *a, flush=True)


_LINKS: dict = {}          # peer rank -> bitlattice.transport.HostLink (transport='host')


def init(backend: str = "nccl", transport: str = "nccl") -> tuple[int, int]:
    """Start the process group for a one-process-per-GPU run (environment from torchrun). Returns (rank, world).

    transport -- how activations and gradients cross between the processes: 'nccl' (point-to-point operations of
                 the process group) or 'host' (all processes on one host: rings in shared page-locked memory moved
                 by the copy engines and synchronized by the GPUs, :mod:`bitlattice.transport`; no SM time, no proxy
                 thread, the host may run far ahead of its GPU)"""
    if transport not in ("nccl", "host"):
        raise ValueError("transport must be 'nccl' or 'host'")
    os.environ.setdefault("TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false")   # one group per direction
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group(backend, device_id=torch.device("cuda", local))
    if transport == "host" and not _LINKS:
        if int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size())) != dist.get_world_size():
            raise ValueError("transport='host' needs every process on one host")
        from .transport import HostLink
        me = dist.get_rank()
        dev = torch.device("cuda", local)
        for a in range(dist.get_world_size()):              # every pair, in the same order on all processes
            for b in range(a + 1, dist.get_world_size()):
                if me in (a, b):
                    _LINKS[b if me == a else a] = HostLink(me, b if me == a else a, dev)
                else:
                    dist.barrier()                          # the barrier inside the pair's link construction
    return dist.get_rank(), dist.get_world_size()


def share_head_(model) -> None:
    """Replicate the frozen output head; every process must call this collective.
    Micro-batch m's loss runs on rank m % world. A trainable final norm is synchronized before use, and its gradients
    return to the owner in micro-batch order for one update."""
    from .nn.norm import _convention
    me, world = dist.get_rank(), dist.get_world_size()
    segs = getattr(model, "_bl_pipeline_plan", None) or _plan(model)
    model._bl_pipeline_plan = segs
    owner = segs[-1].owner
    head = model.get_output_embeddings()
    backbone = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    norm = next((getattr(backbone, n) for n in P._NORM_NAMES if hasattr(backbone, n)), None)
    dev = model.bl_local_device
    meta = [None]
    if me == owner:
        w = head.weight
        if w.requires_grad or getattr(head, "bias", None) is not None:
            raise ValueError("share_head_: the output layer must be frozen and without bias")
        if not w.is_cuda:
            raise ValueError("share_head_: the output layer must be on the GPU (not host_linear)")
        conv = None
        if norm is not None:
            from .nn.convert import _eps
            eps = _eps(norm)
            c = _convention(norm, eps) if eps is not None else None
            if c is None:
                raise ValueError(f"share_head_: the final norm {type(norm).__name__} is not an RMSNorm convention "
                                 "this library implements")
            conv = (eps, c[1], c[2], tuple(norm.weight.shape), norm.weight.dtype, norm.weight.requires_grad)
        meta = [(tuple(w.shape), w.dtype, conv)]
    dist.broadcast_object_list(meta, src=owner)
    shape, dtype, conv = meta[0]
    w = head.weight.detach() if me == owner else torch.empty(shape, dtype=dtype, device=dev)
    dist.broadcast(w, src=owner)
    norm_fn, nw, train_norm = None, None, False
    if conv is not None:
        eps, offset, cast_first, nshape, ndtype, train_norm = conv
        nw = norm.weight if me == owner else torch.empty(nshape, dtype=ndtype, device=dev)
        with torch.no_grad():
            dist.broadcast(nw, src=owner)
        if me != owner:
            nw.requires_grad_(train_norm)
        if me == owner:
            norm_fn = norm
        else:
            import copy
            norm_fn = copy.deepcopy(norm)
            norm_fn.weight = torch.nn.Parameter(nw, requires_grad=train_norm)
            nw = norm_fn.weight
    model._bl_head_share = _HeadShare(w, norm_fn, nw, train_norm, owner)


@dataclass
class _HeadShare:
    weight: torch.Tensor
    norm: object
    norm_weight: torch.Tensor | None
    train_norm: bool
    owner: int

    def sync(self):
        if self.train_norm:
            with torch.no_grad():
                dist.broadcast(self.norm_weight, src=self.owner)
            if dist.get_rank() != self.owner:
                self.norm_weight.grad = None


class _Weight:
    """A stand-in output layer: a weight and no bias."""

    def __init__(self, w):
        self.weight = w


class _Received:
    """The gradient of a unit input computed on another process (stands in for the input in the schedule)."""

    __slots__ = ("grad", "data", "dtype", "device")

    def __init__(self, grad):
        self.grad, self.data, self.dtype, self.device = grad, None, grad.dtype, grad.device


class _Posted:
    """A receive posted on a NCCL communicator."""

    __slots__ = ("tensor", "work")

    def __init__(self, tensor, work):
        self.tensor, self.work = tensor, work

    def take(self):
        self.work.wait()
        return self.tensor


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


_CHANNELS = ("main", "aux")


def _group(src: int, dst: int, channel: str = "main"):
    """Communicator for transfers from ``src`` to ``dst`` on ``channel`` (created collectively for all ordered pairs
    and channels at once). NCCL matches point-to-point operations by order per communicator, so transfers issued in
    a different order (auxiliary inputs of the head) use their own channel."""
    if not _GROUPS:
        w = dist.get_world_size()
        for ch in _CHANNELS:
            for i in range(w):
                for j in range(w):
                    if i != j:
                        _GROUPS[(i, j, ch)] = dist.new_group([i, j])
        # NCCL creates a point-to-point communicator on its first transfer, blocking until the peer does the same;
        # a first transfer issued at different points of the schedule would deadlock. Exchange one element on every
        # communicator now, in the same order on all processes.
        me = dist.get_rank()
        probe = torch.zeros(1, device=torch.device("cuda", torch.cuda.current_device()))
        for (i, j, ch), g in sorted(_GROUPS.items()):
            if me == i:
                dist.send(probe, j, group=g)
            elif me == j:
                dist.recv(probe, i, group=g)
        torch.cuda.synchronize()
    return _GROUPS[(src, dst, channel)]


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
    owner = lambda name: P.device_of(name, dm).index
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


def _prune(keep: list) -> None:
    """Drop finished sends (their communication works hold the sent tensors until then)."""
    alive = []
    for work, t in keep:
        if work.is_completed():
            work.wait()
        else:
            alive.append((work, t))
    keep[:] = alive


def _release(t: torch.Tensor) -> None:
    """Drop the storage of a graph output that is no longer read (its shape is kept separately)."""
    t.data = torch.empty(1, dtype=t.dtype, device=t.device)


def _backward_released(out: torch.Tensor, grad: torch.Tensor) -> None:
    """backward(out, grad) for an output whose storage was released: the engine is called directly, because
    torch.autograd.backward checks that the gradient has the output's (now placeholder) shape."""
    torch.autograd.Variable._execution_engine.run_backward(
        (out,), (grad,), False, False, (), True, True)


def _head_loss(model, norm, head, labels, input_ids, chunk, extra_loss, y, e):
    from .nn.loss import linear_cross_entropy
    h = norm(y) if norm is not None else y
    loss = linear_cross_entropy(h, head.weight, labels[:, 1:], getattr(head, "bias", None), chunk)
    if extra_loss is not None:
        loss = loss + extra_loss(model, y, input_ids, labels, e)
    return loss


class _ActivationOffload:
    """Unit inputs kept in pinned host memory between the forward and the backward of a step.

    ``put`` copies a tensor's data to host memory on a side stream and leaves the tensor object (still referenced
    by the autograd graph and by checkpoint recomputation) with an empty GPU storage; ``prefetch`` starts the copy
    back, ``restore`` makes the compute stream wait for it and reattaches the data. The backward visits units in a
    known order, so the next unit's inputs are prefetched while the current one computes."""

    _POOLS: dict = {}          # device -> HostPool: exact-size buffers kept across steps

    def __init__(self, device: torch.device):
        from .hostmem import HostPool
        self.stream = torch.cuda.Stream(device=device)
        self.pool = self._POOLS.setdefault(device, HostPool())
        self.host: dict = {}
        self.ready: dict = {}

    def save_input(self, t: torch.Tensor):
        """Resolve saved input views through their restored owner; replacing t.data alone leaves SavedVariable's
        detached storage alive. Other saved tensors keep their usual lifetime."""
        storage = t.untyped_storage().data_ptr()
        offset = t.storage_offset()

        def pack(saved):
            if saved.device == t.device and saved.dtype == t.dtype and saved.untyped_storage().data_ptr() == storage:
                return (t, saved.shape, saved.stride(), saved.storage_offset() - offset)
            return saved

        def unpack(saved):
            if isinstance(saved, tuple):
                owner, shape, stride, delta = saved
                return owner.as_strided(shape, stride, owner.storage_offset() + delta)
            return saved

        return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

    def put(self, key, t: torch.Tensor) -> None:
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(t.device))
        self.stream.wait_event(ev)
        h = self.pool.take(t.shape, t.dtype)
        data = t.data
        with torch.cuda.stream(self.stream):
            h.copy_(data, non_blocking=True)
        data.record_stream(self.stream)
        self.host[key] = (t, h)
        t.data = torch.empty(0, dtype=t.dtype, device=t.device)

    def prefetch(self, key) -> None:
        if key not in self.host:
            return
        t, h = self.host.pop(key)
        with torch.cuda.stream(self.stream):
            g = h.to(t.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.pool.give(h, ev)                          # reusable once this copy has completed
        self.ready[key] = (t, g, ev)

    def restore(self, key) -> None:
        self.prefetch(key)
        if key not in self.ready:
            return
        t, g, ev = self.ready.pop(key)
        cur = torch.cuda.current_stream(t.device)
        cur.wait_event(ev)
        g.record_stream(cur)
        t.data = g


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
               optimizer=None, backward: bool = True, chunk: int = 1024, run_ahead: int = 3,
               extra_loss=None, offload_activations: bool = False, checkpoint_head="auto",
               defer_wgrad: bool = False) -> torch.Tensor:
    """Forward (and backward) of a causal LM batch over all processes; returns the mean next-token loss (the same
    value on every process). Every process must call it with the same ``input_ids`` / ``labels``.

    optimizer -- the fused-backward optimizer of this process (``bl.optim.build(..., fused_backward=True)``): its
                 updates are deferred to the last micro-batch of each segment. With a non-fused optimizer pass None
                 and call ``optimizer.step()`` afterwards.
    backward  -- False: evaluation only (no graph, no update).
    run_ahead -- units of work (segment x micro-batch) the host may queue ahead of the GPU.
    extra_loss -- fn(model, hidden, input_ids, labels, next_embeds) -> loss added on the head's process, with
                 ``hidden`` the last block output (before the final norm) at positions 0..T-2 and ``next_embeds``
                 the input embeddings of tokens 1..T-1 (detached; sent from the embedding's process), aligned with
                 ``hidden``, e.g. a multi-token-prediction head.
    offload_activations -- keep the unit inputs (the only activations that live from forward to backward with
                 checkpointed blocks) in pinned host memory: GPU memory then no longer grows with the number of
                 tokens per step beyond one unit's working set.
    checkpoint_head -- recompute the head (final norm, loss over the vocabulary, extra heads) in backward instead of
                 keeping its activations; 'auto': only when the output layer is trained (a frozen one gets its input
                 gradient in forward, so extra heads should checkpoint their own blocks).
    defer_wgrad -- FP4 layers keep their weight-gradient operands over a unit's micro-batches and compute the
                 gradient once after the last one (one GEMM over all micro-batches' tokens, no accumulation into
                 .grad per micro-batch; ``bitlattice.backend.defer_wgrad``)."""
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
    # the model runs on positions 0..T-2: the last token is only a target, and a window of T = seq + 1 tokens then
    # gives GEMMs of exactly seq rows (no padding to the next multiple of 128)
    run_mb = [t[:, :-1] for t in ids_mb]
    # micro-batch losses are means over their own labelled tokens; weighted by those counts (taken where the labels
    # are, no device synchronization for host labels) they add up to the mean over the batch
    n_tok = [int((t[:, 1:] != -100).sum()) for t in labels.chunk(micro, 0)]
    lab_mb = [t.to(dev) for t in labels.chunk(micro, 0)]
    M = len(ids_mb)
    embed_here = segs[0].owner == me

    grad = torch.enable_grad() if backward else torch.no_grad()
    ins, outs, losses, keep = {}, {}, {}, []

    def send(t, dst, channel="main"):
        if _LINKS:
            _LINKS[dst].send(t.contiguous())
        else:
            keep.append((dist.isend(t.contiguous(), dst, group=_group(me, dst, channel)), None))

    def recv(shape, src, channel="main", dt=dtype):
        if _LINKS:
            return _LINKS[src].recv(shape, dt)
        t = torch.empty(shape, dtype=dt, device=dev)
        return _Posted(t, dist.irecv(t, src, group=_group(src, me, channel)))

    # with a shared head (share_head_), micro-batch m's loss is computed on process m % world
    share = getattr(model, "_bl_head_share", None)
    if share is not None and extra_loss is not None:
        raise ValueError("a shared head (share_head_) does not support extra_loss")
    if share is not None:
        share.sync()
    world = dist.get_world_size()
    head_of = [(m % world) if share is not None else segs[-1].owner for m in range(M)]
    remote = {}                                       # micro-batch -> its loss (and gradient) from another process
    # NCCL's proxy thread needs the host to keep up with the GPU; host links need nothing from the host
    ahead = _RunAhead(max(1, run_ahead) if not _LINKS else max(64, run_ahead))
    offload = _ActivationOffload(dev) if offload_activations and backward else None
    head_owner = segs[-1].owner
    # the extra loss's input embeddings of tokens 1.. are sent by the embedding's process right after its last
    # segment's activation of the same micro-batch, and received at the head: a send is never pending while its
    # sender still owes the receiver activations (a device-wide synchronization on the sender -- the caching
    # allocator releasing memory under pressure calls cudaFree -- would wait for it and deadlock both), and neither
    # side holds the embeddings for longer than the head needs them
    aux_from = segs[0].owner
    send_aux_after = (max(s for s, seg in enumerate(segs) if seg.owner == me)
                      if extra_loss is not None and embed_here and head_owner != me else None)

    def aux_embeddings(m):
        if embed_here:
            with torch.no_grad():
                return emb(ids_mb[m][:, 1:]).detach()
        e = torch.empty(ids_mb[m].shape[0], ids_mb[m].shape[1] - 1, width, dtype=dtype, device=dev)
        dist.irecv(e, aux_from, group=_group(aux_from, me, "aux")).wait()
        return e
    with grad:
        prep = [_block_kwargs(backbone, blocks, ids, embed_here, width, dtype, dev) for ids in run_mb]
        # forward: each segment (run of units on this GPU) micro-batch by micro-batch, so its output for micro-batch
        # m leaves for the next GPU as early as possible. The graph is cut between every two units, also locally,
        # so that the backward can go unit by unit.
        for s, seg in enumerate(segs):
            if seg.owner != me:
                continue
            last = len(seg.items) - 1
            # with host links all of the segment's inputs are posted at once: they land while this GPU computes
            incoming = ([_LINKS[seg.prev].recv((*run_mb[m].shape, width), dtype) for m in range(M)]
                        if _LINKS and seg.prev is not None else None)
            for m in range(M):
                shape = (*run_mb[m].shape, width)
                y = None
                for j, item in enumerate(seg.items):
                    if j == 0 and seg.prev is not None:
                        _log("fwd recv", s, m, "from", seg.prev)
                        if incoming is not None:
                            y = incoming[m].take()
                        else:
                            y = torch.empty(shape, dtype=dtype, device=dev)
                            dist.irecv(y, seg.prev, group=_group(seg.prev, me)).wait()
                    elif j > 0 and y is not None:
                        prev = y
                        y = y.detach()
                        if offload is not None:
                            _release(prev)              # the graph node keeps no data; the input owns it now
                    if y is not None and backward:          # evaluation keeps no unit inputs or outputs
                        y.requires_grad_(True)
                        ins[(s, j, m)] = y
                    if item[0] == "embed":
                        y = prep[m][0]
                    elif item[0] == "block":
                        with _range(f"F{item[1]}.{m}"), (offload.save_input(y) if offload is not None
                                                        else contextlib.nullcontext()):
                            out = blocks[item[1]](y, **prep[m][1][item[1]])
                        y = out[0] if isinstance(out, tuple) else out
                    elif head_of[m] != me:
                        # another process computes this micro-batch's loss: it gets the last hidden states
                        send(y.detach(), head_of[m], "aux")
                        remote[m] = None
                        if backward:
                            _release(y)                     # only its gradient is needed from here on
                        y = None
                    else:
                        e = aux_embeddings(m) if extra_loss is not None else None
                        fn = functools.partial(_head_loss, model, norm, head, lab_mb[m], ids_mb[m], chunk,
                                               extra_loss)
                        # A frozen head saves its input gradient in forward; checkpointing would repeat that work.
                        ckpt = (checkpoint_head if checkpoint_head != "auto" else
                                getattr(getattr(head, "weight", None), "requires_grad", True)
                                or not active_kernels().grad_in_forward)
                        loss = (checkpoint(fn, y, e, use_reentrant=False) if backward and ckpt else fn(y, e))
                        losses[m] = loss * (n_tok[m] / max(1, sum(n_tok)))
                        y = None
                    if offload is not None and (s, j, m) in ins and item[0] == "block":
                        offload.put((s, j, m), ins[(s, j, m)])
                    if y is not None:
                        if backward:
                            outs[(s, j, m)] = (y, y.shape)
                        if j == last and seg.next is not None:
                            _log("fwd send", s, m, "to", seg.next)
                            if _LINKS:
                                _LINKS[seg.next].send(y.detach())
                            else:
                                keep.append((dist.isend(y.detach(), seg.next, group=_group(me, seg.next)), None))
                            # the activation now lives on the next process; backward needs only the graph node
                            if backward:
                                _release(y)
                if s == send_aux_after:
                    keep.append((dist.isend(aux_embeddings(m), head_owner, group=_group(me, head_owner, "aux")),
                                 None))
                ahead.tick()
                _prune(keep)
        head_owner_ = segs[-1].owner
        if share is not None and me != head_owner_:
            # the loss of this process's micro-batches: hidden states in, loss and input gradient out
            w_head, norm_fn = share.weight, share.norm
            mine = [m for m in range(M) if head_of[m] == me]
            posted = [recv((*run_mb[m].shape, width), head_owner_, "aux") for m in mine]
            for m, inc in zip(mine, posted):
                yh = inc.take()
                weight = n_tok[m] / max(1, sum(n_tok))
                if backward:
                    yh.requires_grad_(True)
                    with torch.enable_grad():
                        loss = _head_loss(model, norm_fn, _Weight(w_head), lab_mb[m], ids_mb[m], chunk, None, yh,
                                          None) * weight
                    loss.backward()
                    send(yh.grad, head_owner_, "aux")
                    if share.train_norm:
                        send(share.norm_weight.grad, head_owner_, "aux")
                        share.norm_weight.grad = None
                else:
                    loss = _head_loss(model, norm_fn, _Weight(w_head), lab_mb[m], ids_mb[m], chunk, None, yh,
                                      None) * weight
                send(loss.detach().float().reshape(1), head_owner_, "aux")
        elif share is not None and remote:
            for m in sorted(remote):
                remote[m] = (recv((*run_mb[m].shape, width), head_of[m], "aux") if backward else None,
                             recv(share.norm_weight.shape, head_of[m], "aux", share.norm_weight.dtype)
                             if backward and share.train_norm else None,
                             recv((1,), head_of[m], "aux", torch.float32))
        # backward: unit by unit, each over all micro-batches; a unit's gradients are complete after its last
        # micro-batch and the fused update applies and frees them, so one unit of gradients exists at a time
        if backward:
            for s in reversed(range(len(segs))):
                seg = segs[s]
                if seg.owner != me:
                    continue
                last = len(seg.items) - 1
                for j in reversed(range(len(seg.items))):
                    if offload is not None:
                        for m in range(M):
                            offload.restore((s, j, m))
                        nxt = (s, j - 1) if j > 0 else next(((t, len(segs[t].items) - 1) for t in
                                                             reversed(range(s)) if segs[t].owner == me), None)
                        if nxt is not None:
                            for m in range(M):
                                offload.prefetch((*nxt, m))
                    incoming = ([_LINKS[seg.next].recv(outs[(s, j, m)][1], dtype) for m in range(M)]
                                if _LINKS and j == last and seg.items[j][0] != "head" else None)
                    for m in range(M):
                        shared_norm = (share is not None and share.train_norm and seg.items[j][0] == "head")
                        defer = (optimizer.accumulate() if optimizer is not None and (m < M - 1 or shared_norm)
                                 else contextlib.nullcontext())
                        with defer, (backend.defer_wgrad() if defer_wgrad else contextlib.nullcontext()):
                            if seg.items[j][0] == "head" and m in remote:
                                g_in, norm_in, loss_in = remote[m]
                                ins[(s, j, m)] = _Received(g_in.take())    # read by the previous unit as .grad
                                if norm_in is not None:
                                    # Accumulate in micro-batch order, as when every loss lives on the head process.
                                    w, gn = share.norm_weight, norm_in.take()
                                    w.grad = gn if w.grad is None else w.grad.add_(gn)
                                losses[m] = loss_in.take()[0]
                            elif seg.items[j][0] == "head":
                                with _range(f"Bhead.{m}"):
                                    torch.autograd.backward(losses[m])
                            else:
                                y, shape = outs.pop((s, j, m))
                                if j == last:
                                    _log("bwd recv", s, m, "from", seg.next)
                                    if incoming is not None:
                                        g = incoming[m].take()
                                    else:
                                        g = torch.empty(shape, dtype=dtype, device=dev)
                                        dist.irecv(g, seg.next, group=_group(seg.next, me)).wait()
                                else:
                                    g = ins.pop((s, j + 1, m)).grad
                                if y.requires_grad and g is not None:
                                    with _range(f"B{seg.items[j][1] if seg.items[j][0] == 'block' else 'x'}.{m}"):
                                        _backward_released(y, g)
                            x = ins.get((s, j, m))
                            if x is not None and x.grad is not None:
                                x.data = torch.empty(0, dtype=x.dtype, device=x.device)   # only its .grad is read from now on
                        if j == 0 and seg.prev is not None:
                            gi = ins.pop((s, 0, m)).grad
                            _log("bwd send", s, m, "to", seg.prev)
                            if _LINKS:
                                _LINKS[seg.prev].send(gi.contiguous())
                            else:
                                keep.append((dist.isend(gi, seg.prev, group=_group(me, seg.prev)), None))
                        ahead.tick()
                        _prune(keep)
                    if share is not None and seg.items[j][0] == "head":
                        if share.train_norm and optimizer is not None:
                            optimizer._on_grad(share.norm_weight)
                    if defer_wgrad:
                        with _range(f"flush{s}.{j}"):
                            backend.flush_wgrad(optimizer=optimizer)    # this unit's weight gradients, then updates
                    if _DEBUG:
                        live = sum(1 for p in model.parameters() if p.grad is not None)
                        _log("bwd unit", s, j, "done:", f"{torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB,",
                             live, "params hold a gradient")
    if not backward:
        for m, (_, _, loss_in) in remote.items():
            losses[m] = loss_in.take()[0]
    _log("step done, waiting for", len(keep), "sends")
    for work, _ in keep:
        work.wait()                      # orders this process's stream after its sends (the works hold the tensors)
    total = (sum(losses.values()).detach().float() if losses else torch.zeros((), device=dev)).reshape(1)
    dist.all_reduce(total)
    return total[0]
