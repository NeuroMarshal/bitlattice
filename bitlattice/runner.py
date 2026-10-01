"""Training steps of causal language models.

    runner = bl.Runner(model, optimizer, micro=4, memory=bl.Memory(activations="ram"))
    loss = runner.step(input_ids)          # forward and backward over micro-batches, then the update
    held_out = runner.loss(input_ids)      # evaluation

One process drives the model on its devices (a layer-split model on several GPUs works through its device map). When
torch.distributed is initialized with one process per GPU (``bitlattice.pipeline.init``), the step runs the
interleaved pipeline of :mod:`bitlattice.pipeline` instead: every process calls ``step`` with the same batch.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

__all__ = ["Memory", "Runner"]


@dataclass(frozen=True)
class Memory:
    """Where activations live during a training step.

    activations  'gpu', or 'ram': tensors kept from forward to backward go to pinned host memory and come back for
                 backward (one process: every saved tensor -- with checkpointed layers only their inputs; pipeline:
                 the inputs of its units)
    """

    activations: str = "gpu"

    def __post_init__(self):
        if self.activations not in ("gpu", "ram"):
            raise ValueError("activations must be 'gpu' or 'ram'")


class Runner:
    """Executes training steps.

    ``optimizer``  a :class:`bitlattice.optim.Lattice` (with ``fused_backward`` updates are applied during backward)
                   or any torch optimizer; None: gradients only
    ``micro``      micro-batches per step (the batch is split along its first dimension)
    ``memory``     :class:`Memory`
    ``defer_wgrad``  layers on the FP4 routes keep their weight-gradient operands over the micro-batches and compute
                   the gradient once (``bitlattice.backend.defer_wgrad``)
    ``loss_chunk`` tokens per chunk of the vocabulary-wide loss
    ``extra_loss`` pipeline only: fn(model, hidden, input_ids, labels, next_embeds) added on the head's process
    """

    def __init__(self, model: torch.nn.Module, optimizer=None, *, micro: int = 1, memory: Memory = Memory(),
                 defer_wgrad: bool = False, loss_chunk: int = 1024, extra_loss=None):
        if micro < 1:
            raise ValueError("micro must be >= 1")
        self.model, self.optimizer, self.micro, self.memory = model, optimizer, micro, memory
        self.defer_wgrad, self.loss_chunk, self.extra_loss = defer_wgrad, loss_chunk, extra_loss
        import torch.distributed as dist
        self.distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if extra_loss is not None and not self.distributed:
            raise ValueError("extra_loss is supported by the pipeline (one process per GPU)")

    @property
    def _fused(self) -> bool:
        return getattr(self.optimizer, "fused_backward", False)

    def _on_input_device(self, input_ids, labels):
        dev = self.model.get_input_embeddings().weight.device
        return input_ids.to(dev), None if labels is None else labels.to(dev)

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Forward and backward of the batch over ``micro`` micro-batches, then the optimizer update. Returns the
        mean next-token loss over the batch's tokens (positions with label -100 are ignored). The batch may be on
        any device; it is moved to the device of the input embeddings."""
        if self.distributed:
            from .pipeline import train_step
            return train_step(self.model, input_ids, labels, micro=self.micro, optimizer=self.optimizer,
                              chunk=self.loss_chunk, extra_loss=self.extra_loss,
                              offload_activations=self.memory.activations == "ram", defer_wgrad=self.defer_wgrad)
        from . import backend
        from .nn.loss import causal_lm_loss
        from .storage import activations_in_ram
        input_ids, labels = self._on_input_device(input_ids, labels)
        labels = input_ids if labels is None else labels
        ids_mb, lab_mb = input_ids.chunk(self.micro, 0), labels.chunk(self.micro, 0)
        counts = [int((lab[:, 1:] != -100).sum()) for lab in lab_mb]
        total = max(1, sum(counts))
        loss_sum = None
        for i, (ids, lab) in enumerate(zip(ids_mb, lab_mb)):
            with activations_in_ram() if self.memory.activations == "ram" else contextlib.nullcontext():
                loss = causal_lm_loss(self.model, ids, lab, chunk=self.loss_chunk) * (counts[i] / total)
            accumulate = self.optimizer.accumulate() if self._fused and i < len(ids_mb) - 1 else \
                contextlib.nullcontext()
            with accumulate, backend.defer_wgrad() if self.defer_wgrad else contextlib.nullcontext():
                loss.backward()
            loss_sum = loss.detach() if loss_sum is None else loss_sum + loss.detach()
        if self.defer_wgrad:
            backend.flush_wgrad(optimizer=self.optimizer if self._fused else None)
        if self.optimizer is not None:
            self.optimizer.step()
            if not self._fused:
                self.optimizer.zero_grad(set_to_none=True)
        return loss_sum

    @torch.no_grad()
    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Mean next-token loss of the batch in eval mode, without gradients."""
        was = self.model.training
        self.model.eval()
        try:
            if self.distributed:
                from .pipeline import train_step
                return train_step(self.model, input_ids, labels, micro=min(self.micro, input_ids.shape[0]),
                                  backward=False, chunk=self.loss_chunk)
            from .nn.loss import causal_lm_loss
            return causal_lm_loss(self.model, *self._on_input_device(input_ids, labels), chunk=self.loss_chunk)
        finally:
            self.model.train(was)
