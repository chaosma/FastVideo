# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SSD spill manager (fastvideo.training.ssd_manager).

CPU-only: the manager's residency logic, deterministic schedule, budget
interaction, and prefetch/underrun accounting are exercised with host tensors
(no CUDA). GPU<->host copies and CUDA-event fences are validated separately on a
GPU box (phase 4).
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.training.ssd_manager import Residency, SsdSpillManager
from fastvideo.training.ssd_spill import align_up, tensor_byte_view

# One bf16 element-heavy tensor pads to one 4096 block; size the cap in blocks.
BLOCK = align_up(1)


def _admit(mgr, block_index, tensor):
    """Stage ``tensor`` into a manager buffer and admit it for ``block_index``."""
    nbytes = tensor.numel() * tensor.element_size()
    buf = mgr.acquire_staging(nbytes)
    tensor_byte_view(buf)[:nbytes] = tensor_byte_view(tensor.contiguous())
    return mgr.admit(buf, block_index=block_index, shape=tuple(tensor.shape),
                     dtype=tensor.dtype, nbytes=nbytes)


def _mk(vals) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.bfloat16)


def _admit_generic(mgr, policy_key, tensor, block_index=-1):
    nbytes = tensor.numel() * tensor.element_size()
    buf = mgr.acquire_staging(nbytes)
    tensor_byte_view(buf)[:nbytes] = tensor_byte_view(tensor.contiguous())
    return mgr.admit(buf, shape=tuple(tensor.shape), dtype=tensor.dtype,
                     nbytes=nbytes, category="saved_tensor",
                     policy_key=policy_key, block_index=block_index)


def test_spill_and_reload_byte_exact(tmp_path) -> None:
    # cap = 1 block -> the second admit must spill the first.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=BLOCK, pin=False)
    try:
        t0 = torch.randn(600).bfloat16()
        t1 = torch.randn(600).bfloat16()
        h0 = _admit(mgr, 0, t0)
        h1 = _admit(mgr, 1, t1)
        # h0 (lower index) spilled to make room for h1.
        assert h0.state is Residency.SPILLED
        assert h1.state is Residency.RESIDENT
        assert mgr.n_spilled == 1
        got0 = mgr.ensure_resident(h0).clone()
        mgr.release(h0)
        got1 = mgr.ensure_resident(h1).clone()
        mgr.release(h1)
        # Byte-exact reload is the point. (With cap=1 the two tensors cannot be
        # co-resident, so each on-demand read is an underrun; that the cap is
        # actually honored is covered by test_allocated_rss_never_exceeds_cap.)
        assert torch.equal(got0, t0)
        assert torch.equal(got1, t1)
        assert mgr.prefetch_underruns >= 1
    finally:
        mgr.close()


def test_schedule_spills_lowest_indices(tmp_path) -> None:
    # cap = 3 blocks, admit 8 blocks -> newest 3 resident, lowest 5 spilled.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=3 * BLOCK, pin=False)
    try:
        handles = [_admit(mgr, i, _mk([float(i)] * 4)) for i in range(8)]
        resident = {h.block_index for h in handles
                    if h.state is Residency.RESIDENT}
        spilled = {h.block_index for h in handles
                   if h.state is Residency.SPILLED}
        assert resident == {5, 6, 7}
        assert spilled == {0, 1, 2, 3, 4}
    finally:
        mgr.close()


def test_prefetch_makes_reload_a_hit(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK,
                          prefetch_window=8, pin=False)
    try:
        handles = [_admit(mgr, i, _mk([float(i)] * 4)) for i in range(6)]
        # Consume in reverse (backward order). Before each block, prefetch the
        # lower spilled blocks; every reload should then be a prefetch hit.
        for i in reversed(range(6)):
            mgr.on_block_recompute_enter(i)
            got = mgr.ensure_resident(handles[i]).clone()
            assert torch.equal(got, _mk([float(i)] * 4))
            mgr.release(handles[i])
        assert mgr.prefetch_underruns == 0
        # Spilled blocks were prefetched -- clean hit (read landed first) or late
        # (still in flight at reload); either way not a full underrun.
        assert mgr.prefetch_hits + mgr.prefetch_late >= 1
    finally:
        mgr.close()


def test_no_prefetch_counts_underruns(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK, pin=False)
    try:
        handles = [_admit(mgr, i, _mk([float(i)] * 4)) for i in range(6)]
        n_spilled = mgr.n_spilled
        for i in reversed(range(6)):
            mgr.ensure_resident(handles[i])  # no prefetch call
            mgr.release(handles[i])
        # Every spilled block was read on demand -> one underrun each.
        assert mgr.prefetch_underruns == n_spilled
    finally:
        mgr.close()


def _live_buffer_bytes(mgr) -> int:
    in_use = sum(h.padded for h in mgr._handles if h.host is not None)
    free = sum(size * len(bufs) for size, bufs in mgr._free_host.items())
    return in_use + free


def test_allocated_rss_never_exceeds_cap(tmp_path) -> None:
    # The cap must bound *physical* pinned bytes (in-use + free-list), not just
    # the resident working set. Admit far more blocks than fit.
    cap = 3 * BLOCK
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=cap, pin=False)
    try:
        for i in range(20):
            _admit(mgr, i, _mk([float(i)] * 4))
            # accounting matches physical buffers, and neither exceeds the cap
            assert _live_buffer_bytes(mgr) == mgr.budget.allocated
            assert mgr.budget.allocated <= cap
    finally:
        mgr.close()


def test_handles_do_not_grow_across_steps(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=3 * BLOCK,
                          prefetch_window=8, pin=False)
    try:
        for _step in range(3):
            hs = [_admit(mgr, i, _mk([float(i)] * 4)) for i in range(6)]
            for i in reversed(range(6)):
                mgr.on_block_recompute_enter(i)
                mgr.ensure_resident(hs[i])
                mgr.release(hs[i])
            assert len(mgr._handles) == 0  # fully drained each step
    finally:
        mgr.close()


def test_eviction_uses_production_order_not_block_index(tmp_path) -> None:
    # block_index descending, production ascending: the first-admitted handle
    # must spill first (consumed last in backward), regardless of block_index.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=BLOCK, pin=False)
    try:
        h0 = _admit(mgr, 99, _mk([0.0] * 4))  # produced first
        h1 = _admit(mgr, 0, _mk([1.0] * 4))   # produced second -> spills h0
        assert h0.state is Residency.SPILLED
        assert h1.state is Residency.RESIDENT
    finally:
        mgr.close()


def test_sac_prefetch_ignores_untagged_generic_handles(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK,
                          prefetch_window=64, pin=False)
    try:
        gens = [_admit_generic(mgr, i, _mk([float(i)] * 4)) for i in range(5)]
        spilled = [h for h in gens if h.state is Residency.SPILLED]
        assert spilled
        # Block-boundary prefetch must not guess for generic handles whose
        # adapter could not attach block provenance.
        mgr.on_block_recompute_enter(1000)
        assert all(h.state is Residency.SPILLED for h in spilled)
    finally:
        mgr.close()


def test_block_prefetch_reads_tagged_generic_handles(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK,
                          prefetch_window=8, pin=False)
    try:
        gens = [
            _admit_generic(mgr, i, _mk([float(i)] * 4), block_index=i)
            for i in range(6)
        ]
        # Free current residents so the prefetch has room to read lower blocks.
        for h in list(gens):
            if h.state is Residency.RESIDENT:
                mgr.release(h)
        spilled = [h for h in gens if h.state is Residency.SPILLED]
        assert spilled
        mgr.on_block_recompute_enter(6)
        prefetched = [h for h in spilled if h.state is Residency.INFLIGHT_IN]
        assert prefetched
        for h in prefetched:
            got = mgr.ensure_resident(h).clone()
            assert torch.equal(got, _mk([float(h.policy_key)] * 4))
            mgr.release(h)
        assert mgr.prefetch_hits >= 1
    finally:
        mgr.close()


def test_discard_staging_returns_buffer_to_pool(tmp_path) -> None:
    # A buffer acquired but never admitted (adapter faulted mid-pack) must go
    # back to the pool, not leak budget bytes.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK, pin=False)
    try:
        buf = mgr.acquire_staging(4)
        alloc = mgr.budget.allocated
        assert alloc > 0
        mgr.discard_staging(buf)
        # Reused on the next same-size acquire -> no new allocation, no leak.
        mgr.acquire_staging(4)
        assert mgr.budget.allocated == alloc
    finally:
        mgr.close()


def test_acquire_larger_than_cap_raises(tmp_path) -> None:
    # Nothing to spill or drop and the request exceeds the cap -> raise rather
    # than silently over-allocate.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK, pin=False)
    try:
        big = torch.randn(5000, dtype=torch.bfloat16)  # ~10 KB -> 3 blocks
        with pytest.raises(RuntimeError, match="budget exceeded"):
            _admit(mgr, 0, big)
    finally:
        mgr.close()


def test_untagged_generic_not_block_prefetched_at_low_cursor(tmp_path) -> None:
    # Untagged generic handles (block_index=-1) must stay on the reactive path.
    # Before the fix, a low backward cursor (lo = cursor - window <= -1) pulled
    # them into block prefetch; probing block 1000 could not catch that.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK,
                          prefetch_window=8, pin=False)
    try:
        gens = [_admit_generic(mgr, i, _mk([float(i)] * 4)) for i in range(5)]
        spilled = [h for h in gens if h.state is Residency.SPILLED]
        assert spilled and all(h.block_index == -1 for h in gens)
        mgr.on_block_recompute_enter(2)  # lo = 2 - 8 = -6, so lo <= -1 < 2
        assert all(h.state is Residency.SPILLED for h in spilled)
    finally:
        mgr.close()


def test_reset_step_reclaims_orphans(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=3 * BLOCK, pin=False)
    try:
        [_admit(mgr, i, _mk([float(i)] * 4)) for i in range(5)]  # never consumed
        assert len(mgr._handles) == 5
        alloc = mgr.budget.allocated
        assert mgr.reset_step() == 5           # orphan count
        assert len(mgr._handles) == 0
        assert mgr.budget.allocated == alloc   # buffers pooled, not leaked/freed
        _admit(mgr, 0, _mk([1.0] * 4))         # reuses within cap
        assert mgr.budget.allocated <= 3 * BLOCK
        assert mgr.reset_step() == 1
    finally:
        mgr.close()


def test_reload_after_release_raises_retain_graph(tmp_path) -> None:
    # A second reload of an already-consumed handle (double backward /
    # retain_graph) must fail with a message that names the cause.
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK, pin=False)
    try:
        h = _admit(mgr, 0, _mk([1.0] * 4))
        mgr.ensure_resident(h)
        mgr.release(h)
        with pytest.raises(RuntimeError, match="retain_graph"):
            mgr.ensure_resident(h)
    finally:
        mgr.close()


def test_release_on_inflight_handle_raises(tmp_path) -> None:
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=2 * BLOCK, pin=False)
    try:
        h = _admit(mgr, 0, _mk([1.0] * 4))
        h.state = Residency.INFLIGHT_IN  # simulate an in-flight SSD read
        with pytest.raises(RuntimeError, match="INFLIGHT_IN"):
            mgr.release(h)
    finally:
        h.state = Residency.RESIDENT  # let close() shut down cleanly
        mgr.close()


def test_make_room_drops_free_before_spilling_wrong_size(tmp_path) -> None:
    # Cap full of one-block residents; freeing two, then admitting a two-block
    # tensor should DROP the free buffers rather than spill the wrong-size
    # resident (avoids write amplification with diverse generic shapes).
    mgr = SsdSpillManager(str(tmp_path), cap_bytes=3 * BLOCK, pin=False)
    try:
        h1 = _admit(mgr, 0, _mk([1.0] * 4))          # 1 block
        h2 = _admit(mgr, 1, _mk([2.0] * 4))          # 1 block
        h3 = _admit(mgr, 2, _mk([3.0] * 4))          # 1 block
        mgr.release(h1)
        mgr.release(h2)                               # two 1-block free buffers
        assert h3.state is Residency.RESIDENT
        spilled_before = mgr.n_spilled
        big = torch.randn(3000, dtype=torch.bfloat16)  # 6000 B -> 2 blocks
        _admit(mgr, 3, big)
        assert h3.state is Residency.RESIDENT         # wrong-size resident kept
        assert mgr.n_spilled == spilled_before        # no spill traffic
        # NOTE: budget.allocated stays <= cap here, but this only proves the
        # LOGICAL accounting -- for cross-size drops PyTorch's caching host
        # allocator may retain the dropped block, so physical pinned RSS can
        # exceed the cap under mixed generic sizes (see HostBudget caveat / the
        # _make_room step-2 comment). The cap is exact only for uniform (SAC).
        assert mgr.budget.allocated <= 3 * BLOCK
    finally:
        mgr.close()


