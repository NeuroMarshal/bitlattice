"""Pipeline input offload releases storage retained by autograd and restores saved views for backward."""
import pytest
import torch

from bitlattice.pipeline import _ActivationOffload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _SavedView(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x[:, 1:])
        return x.square().sum()

    @staticmethod
    def backward(ctx, g):
        view, = ctx.saved_tensors
        dx = torch.zeros(view.shape[0], view.shape[1] + 1, device=view.device)
        dx[:, 1:] = 2 * view * g
        return dx


def test_offloaded_saved_input_releases_storage_and_restores_views():
    device = torch.device("cuda", torch.cuda.current_device())
    offload = _ActivationOffload(device)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    x = torch.full((1024, 4096), 3.0, device=device, requires_grad=True)
    size = x.numel() * x.element_size()
    with offload.save_input(x):
        loss = _SavedView.apply(x)
    offload.put("x", x)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < size // 4
    offload.restore("x")
    loss.backward()
    assert x.shape == (1024, 4096)
    assert torch.all(x.grad[:, 1:] == 6) and torch.all(x.grad[:, 0] == 0)
