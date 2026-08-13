# SPDX-License-Identifier: Apache-2.0
"""Wiring tests: block_index flows activation_checkpoint -> sac_cpu_offload cache.

CPU-only and structural. The actual GPU->host D2H-into-staging and H2D reload run
only under ``x.is_cuda`` and are validated on a GPU box; here we verify that the
self_attn_cpu path threads a per-block index into the offload cache (which the
SSD manager schedules on) and that block wrapping builds without error.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import CheckpointPolicy

from fastvideo.training.activation_checkpoint import (
    _self_attn_cpu_offload_context_fn, apply_activation_checkpointing)
from fastvideo.training.block_context import (
    current_block_index, wrap_forward_with_block_context)
from fastvideo.training.sac_cpu_offload import (
    create_cpu_offload_selective_checkpoint_contexts)


def _recompute_policy(ctx, func, *args, **kwargs):
    return CheckpointPolicy.PREFER_RECOMPUTE


def test_block_index_threads_into_shared_cache() -> None:
    caching, replay = create_cpu_offload_selective_checkpoint_contexts(
        _recompute_policy, block_index=5)
    assert caching.cache.block_index == 5
    assert replay.cache is caching.cache  # forward + recompute share one cache
    assert caching.cache.manager is None  # no SSD manager until an SSD pack


def test_cpu_offload_context_fn_binds_block_index() -> None:
    caching, _ = _self_attn_cpu_offload_context_fn(3)
    assert caching.cache.block_index == 3


def test_none_block_index_is_inert() -> None:
    caching, _ = create_cpu_offload_selective_checkpoint_contexts(
        _recompute_policy)
    assert caching.cache.block_index is None


def test_apply_self_attn_cpu_wraps_blocks_with_global_indices() -> None:
    m = nn.Module()
    m.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    before = [type(b) for b in m.blocks]
    apply_activation_checkpointing(m, "self_attn_cpu")  # must not raise
    after = [type(b) for b in m.blocks]
    # Every covered block was replaced by a checkpoint wrapper.
    assert all(a is not b for a, b in zip(after, before))
    assert len(m.blocks) == 3


def test_block_context_tags_forward_execution() -> None:
    seen = []

    class Recorder(nn.Module):

        def forward(self, x):
            seen.append(current_block_index())
            return x

    block = wrap_forward_with_block_context(Recorder(), 7)
    block("x")
    assert seen == [7]
    assert current_block_index() is None


def test_checkpoint_boundary_saves_are_block_tagged(monkeypatch) -> None:
    # Block tagging is gated on the generic saved-tensor SSD adapter (its only
    # consumer), so enable that gate; with it off the wrap is skipped (covered
    # by test_saved_tensors_ssd).
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")

    class Block(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x).relu()

    for checkpointing_type in ("full", "self_attn_cpu"):
        m = nn.Module()
        m.blocks = nn.ModuleList([Block(), Block()])
        apply_activation_checkpointing(m, checkpointing_type)

        seen: list[int | None] = []

        def pack(tensor):
            seen.append(current_block_index())
            return tensor

        def unpack(tensor):
            return tensor

        x = torch.randn(2, 4, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            m.blocks[1](m.blocks[0](x)).sum().backward()

        assert seen, checkpointing_type
        assert set(seen) == {0, 1}
    assert current_block_index() is None
