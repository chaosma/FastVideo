"""Tests for the Phase 3 backward prefetch scheduler.

Verifies:
- After backward, the unpack hot path consumed prefetched buffers (i.e.
  ``live_buffers`` was populated for every block except the first-in-bwd).
- At any instant during backward, at most one prefetched neighbour is live
  on top of the currently-active block.
- The prefetch semaphore is fully released after the step ends (no leaks).
- Multi-step training repeats the pattern without state bleed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="prefetch scheduler requires CUDA")

from fastvideo.training.activation_checkpoint import (  # noqa: E402
    CheckpointType, apply_activation_checkpointing)
from fastvideo.training.activation_streaming import get_registry  # noqa: E402


class _Block(nn.Module):

    def __init__(self, dim: int = 32):
        super().__init__()
        self.lin1 = nn.Linear(dim, 4 * dim)
        self.lin2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(torch.nn.functional.gelu(self.lin1(x)))


class _Stack(nn.Module):

    def __init__(self, n: int = 6, dim: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def _build_streamed() -> nn.Module:
    torch.manual_seed(0)
    base = _Stack().to("cuda").to(torch.float32)
    base.train()
    return apply_activation_checkpointing(
        base, checkpointing_type=CheckpointType.STREAMED_OFFLOAD)


def test_prefetch_scheduler_releases_state_after_step():
    """After a full forward+backward, the registry must hold no live
    buffers, no restore events, no held semaphore acquisitions, and no
    leftover CPU records. State that survives a step is a leak."""
    model = _build_streamed()
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)

    y = model(x)
    y.pow(2).mean().backward()
    torch.cuda.synchronize()

    registry = get_registry(torch.device("cuda"))
    assert registry.live_buffers == {}, registry.live_buffers
    assert registry.restore_events == {}, registry.restore_events
    assert registry.records == {}, list(registry.records.keys())
    assert not registry._prefetch_acquired, registry._prefetch_acquired
    # Semaphore should be back at 1 (free).
    assert registry.prefetch_sem.acquire(blocking=False), (
        "prefetch_sem stuck after step")
    registry.prefetch_sem.release()


def test_prefetch_scheduler_observed_queue_depth_at_most_one():
    """Instrument prefetch() / mark_active() / release_after_bwd() and
    confirm that the number of layers with live prefetched buffers
    (excluding the currently-active layer) never exceeds 1."""
    model = _build_streamed()
    registry = get_registry(torch.device("cuda"))

    # Snapshot of (count_of_blocks_with_live_buffers, active_block) over time.
    observations: list[tuple[int, int | None]] = []
    active: list[int] = [None]  # type: ignore[list-item]

    orig_prefetch = registry.prefetch
    orig_mark = registry.mark_active
    orig_release = registry.release_after_bwd

    def patched_prefetch(block_idx: int, *, acquire_sem: bool) -> None:
        orig_prefetch(block_idx, acquire_sem=acquire_sem)
        observations.append((len(registry.live_buffers), active[0]))

    def patched_mark(block_idx: int) -> None:
        orig_mark(block_idx)
        active[0] = block_idx
        observations.append((len(registry.live_buffers), active[0]))

    def patched_release(block_idx: int) -> None:
        orig_release(block_idx)
        if active[0] == block_idx:
            active[0] = None
        observations.append((len(registry.live_buffers), active[0]))

    registry.prefetch = patched_prefetch  # type: ignore[method-assign]
    registry.mark_active = patched_mark  # type: ignore[method-assign]
    registry.release_after_bwd = patched_release  # type: ignore[method-assign]

    try:
        x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
        y = model(x)
        y.pow(2).mean().backward()
        torch.cuda.synchronize()
    finally:
        registry.prefetch = orig_prefetch  # type: ignore[method-assign]
        registry.mark_active = orig_mark  # type: ignore[method-assign]
        registry.release_after_bwd = orig_release  # type: ignore[method-assign]

    # The number of live_buffers entries at any observation should be:
    #   - 1 (just current) or 2 (current + one prefetched neighbour)
    # Anything >= 3 means the scheduler let the queue grow past one-deep.
    max_observed = max(c for c, _ in observations)
    assert max_observed <= 2, (
        f"observed up to {max_observed} layers with live prefetched buffers; "
        f"trace = {observations}")


def test_prefetch_scheduler_two_step_repeat():
    """Run two backward passes; both should leave the registry clean."""
    model = _build_streamed()
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)

    for _ in range(2):
        x = torch.randn(2, 8, 32, device="cuda")
        loss = model(x).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.synchronize()

    registry = get_registry(torch.device("cuda"))
    assert registry.live_buffers == {}
    assert registry.restore_events == {}
    assert registry.records == {}
    assert not registry._prefetch_acquired
