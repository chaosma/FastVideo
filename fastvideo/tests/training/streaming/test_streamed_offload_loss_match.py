"""1-step loss-match integration test for STREAMED_OFFLOAD.

Asserts that running one forward+backward through a transformer-block stack
wrapped with ``CheckpointType.STREAMED_OFFLOAD`` produces bit-identical loss
and parameter gradients to the same stack wrapped with
``CheckpointType.FULL_OFFLOAD`` (PyTorch's existing offload_wrapper).

If this test diverges, the async-stream pack/unpack is doing something
non-transparent -- the streamed version must give the same numerics as the
naive baseline, just with different runtime behaviour.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="streaming hook requires CUDA")

from fastvideo.training.activation_checkpoint import (  # noqa: E402
    CheckpointType, apply_activation_checkpointing)


class _Block(nn.Module):
    """Standin transformer block: MLP + residual, two saved tensors."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.lin2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.lin2(torch.nn.functional.gelu(self.lin1(h)))
        return x + h


class _Stack(nn.Module):
    """Exposes a ``blocks`` ModuleList so apply_activation_checkpointing
    finds it via TRANSFORMER_BLOCK_NAMES."""

    def __init__(self, n: int = 4, dim: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def _run_step(model: nn.Module, seed: int = 7
              ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(seed)
    x = torch.randn(4, 16, 64, device="cuda", requires_grad=True)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad.detach().clone() for p in model.parameters()
             if p.grad is not None]
    return loss.detach().clone(), grads


def _build(ckpt: CheckpointType) -> nn.Module:
    torch.manual_seed(42)
    base = _Stack().to("cuda").to(torch.float32)
    base.train()
    return apply_activation_checkpointing(base, checkpointing_type=ckpt)


def test_streamed_offload_matches_full_offload_bitexact():
    """STREAMED_OFFLOAD vs FULL_OFFLOAD: bit-exact loss and grads.

    Both modes do H2D / D2H memcpys of the same saved tensors, so the only
    runtime difference is which stream the copies execute on. Numerics must
    be identical.
    """
    base = _Stack().to("cuda").to(torch.float32)
    base.train()
    torch.manual_seed(42)
    for p in base.parameters():
        p.data.normal_()

    m_full = apply_activation_checkpointing(
        copy.deepcopy(base), checkpointing_type=CheckpointType.FULL_OFFLOAD)
    m_streamed = apply_activation_checkpointing(
        copy.deepcopy(base), checkpointing_type=CheckpointType.STREAMED_OFFLOAD)

    loss_full, grads_full = _run_step(m_full)
    loss_streamed, grads_streamed = _run_step(m_streamed)

    assert torch.equal(loss_full, loss_streamed), (
        f"loss mismatch: full_offload={loss_full.item()} "
        f"streamed={loss_streamed.item()}")

    assert len(grads_full) == len(grads_streamed)
    for i, (gf, gs) in enumerate(zip(grads_full, grads_streamed)):
        assert torch.equal(gf, gs), f"grad mismatch at param index {i}"


def test_streamed_offload_matches_full_recompute_bitexact():
    """STREAMED_OFFLOAD vs FULL (recompute, no offload): bit-exact.

    save_on_cpu and the streamed hook both round-trip through CPU but at
    fp32 the values land back identical -- this guards against silent
    precision regressions in pack/unpack."""
    base = _Stack().to("cuda").to(torch.float32)
    base.train()
    torch.manual_seed(42)
    for p in base.parameters():
        p.data.normal_()

    m_full = apply_activation_checkpointing(
        copy.deepcopy(base), checkpointing_type=CheckpointType.FULL)
    m_streamed = apply_activation_checkpointing(
        copy.deepcopy(base), checkpointing_type=CheckpointType.STREAMED_OFFLOAD)

    loss_full, grads_full = _run_step(m_full)
    loss_streamed, grads_streamed = _run_step(m_streamed)

    assert torch.equal(loss_full, loss_streamed), (
        f"loss mismatch: recompute={loss_full.item()} "
        f"streamed={loss_streamed.item()}")
    for i, (gf, gs) in enumerate(zip(grads_full, grads_streamed)):
        assert torch.equal(gf, gs), f"grad mismatch at param index {i}"
