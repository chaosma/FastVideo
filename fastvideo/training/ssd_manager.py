# SPDX-License-Identifier: Apache-2.0
"""SsdSpillManager: residency + deterministic schedule for the SSD spill tier.

Sits between a host-resident (pinned) staging buffer and the SSD backing file
(see ``.agents/exploration/ssd-offload-design.md``). It owns:

* a **byte budget** (``HostBudget``) that bounds pinned RAM -- allocation is
  capped, never freed-to-OS, so RSS is bounded by construction;
* a **residency state machine** per handle
  (``RESIDENT -> INFLIGHT_OUT -> SPILLED -> INFLIGHT_IN -> RESIDENT``);
* a **deterministic schedule**: activations are produced in increasing block
  order and consumed in reverse, so the lowest-index (last-consumed) resident
  buffers are spilled first and prefetched back late.

The GPU<->host copy and its CUDA event live in the adapter (phase 4); this
manager takes an already-host-resident buffer plus an optional CUDA event that
signals the D2H completed, and waits on it before writing to SSD. All SSD I/O
goes through the aligned, direct-I/O transport in ``ssd_spill``.

v1: spill-out is synchronous and inline on the forward pack path -- it is
reached from ``acquire_staging`` -> ``_acquire_host`` -> ``_make_room`` and does
a blocking ``pwrite`` while holding the manager lock, so the caller (the pack
hook) eats the write; it does NOT overlap forward. At ~1.3 GB SAC buffers that
is ~200-400 ms/spill on NVMe -- tolerable at ~20-min steps but the dominant
avoidable cost under a tight budget, and making spill-out async is the main
planned refinement. Prefetch reads during backward ARE asynchronous (the read
latency that actually needs hiding).
"""
from __future__ import annotations

import contextlib
import enum
import os
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.training.ssd_budget import HostBudget
from fastvideo.training.ssd_spill import (SpillFile, align_up,
                                          alloc_aligned_host)

logger = init_logger(__name__)


class Residency(enum.Enum):
    RESIDENT = "resident"
    INFLIGHT_OUT = "inflight_out"
    SPILLED = "spilled"
    INFLIGHT_IN = "inflight_in"
    DONE = "done"  # consumed in backward; buffer returned to the pool


@dataclass(slots=True)
class SpillHandle:
    nbytes: int  # logical size of the tensor payload
    padded: int  # block-padded size actually allocated / written
    shape: tuple[int, ...]
    dtype: torch.dtype
    state: Residency
    host: torch.Tensor | None  # aligned, padded uint8 staging buffer
    # Adapter that produced this handle: "sac" (block-indexed prefetch) or
    # "saved_tensor" (generic). Determines which prefetch path may touch it.
    category: str = "sac"
    # Global forward-order rank across ALL categories, assigned at admit. The
    # eviction key: the earliest-produced handle is consumed last in backward,
    # so it is the safest to spill regardless of category.
    production_index: int = -1
    # Global transformer block index used by block-boundary prefetch; -1 when
    # an adapter cannot attach block provenance.
    block_index: int = -1
    # Adapter-specific ordering metadata (e.g. generic pack index) for prefetch.
    policy_key: int | None = None
    spill_offset: int = -1
    read_future: Future | None = None
    # CUDA event signalling the D2H fill of ``host`` completed; the writer waits
    # on it before pwrite. None on CPU / when the fill was synchronous.
    d2h_event: Any = None
    # True once this handle has been spilled and a read (prefetch or on-demand)
    # issued for it -- lets a completed prefetch be counted as a prefetch hit,
    # not confused with a block that was never spilled.
    prefetched: bool = False
    # True while the adapter holds the ensure_resident() alias (until release()).
    # Reserved handles are excluded from eviction so a concurrent spill cannot
    # recycle a buffer whose H2D copy is still in flight.
    reserved: bool = False


class SsdSpillManager:
    """Per-rank manager owning one SSD spill file + a capped host pool."""

    def __init__(self,
                 spill_dir: str,
                 cap_bytes: int,
                 *,
                 io_threads: int = 2,
                 prefetch_window: int = 4,
                 pin: bool = False) -> None:
        os.makedirs(spill_dir, exist_ok=True)
        path = os.path.join(spill_dir, f"sac_spill.{os.getpid()}.bin")
        self.file = SpillFile(path)
        self.budget = HostBudget(cap_bytes)
        self.prefetch_window = prefetch_window
        self.pin = pin
        self._exec = ThreadPoolExecutor(max_workers=io_threads)
        self._lock = threading.Lock()
        self._free_host: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._free_offsets: dict[int, list[int]] = defaultdict(list)
        self._next_offset = 0
        self._next_production_index = 0
        self._handles: list[SpillHandle] = []
        # Counters (design metric: prefetch_underruns == 0).
        self.resident_hits = 0
        self.prefetch_hits = 0       # spilled, read finished before reload
        self.prefetch_late = 0       # spilled + prefetched, but reload still waited
        self.prefetch_underruns = 0  # spilled, no read issued -> full stall
        self.n_spilled = 0
        self.n_reloaded = 0
        self.spilled_bytes = 0
        # Actual disk spills split by adapter, so combined SAC + generic runs can
        # attribute disk traffic (admissions != spills).
        self.n_spilled_by_category: dict[str, int] = defaultdict(int)
        self.spilled_bytes_by_category: dict[str, int] = defaultdict(int)

    # -- host buffer pool: cap ALLOCATED bytes (in-use + free-list) -----------
    # RSS is bounded only if total live pinned bytes stay <= cap. A buffer on
    # the free-list still holds RSS, so we reuse before allocating and, when a
    # new allocation would exceed the cap, first spill a resident block to free
    # a reusable buffer (or drop a free buffer to reclaim bytes). Callers hold
    # the lock.

    def _acquire_host(self, nbytes: int) -> torch.Tensor:
        padded = align_up(nbytes)
        free = self._free_host.get(padded)
        if free:
            return free.pop()  # reuse -> no new RSS
        # alloc_aligned_host allocates exactly ``padded`` bytes when the base
        # allocation is already block-aligned (always true for pinned host
        # memory), so ``padded`` is the true RSS and the cap does not drift.
        while not self.budget.can_allocate(padded):
            if not self._make_room(padded):
                raise RuntimeError(
                    "SSD spill: host RAM budget exceeded with nothing left to "
                    f"spill or drop (cap={self.budget.cap_bytes} B, "
                    f"allocated={self.budget.allocated} B, "
                    f"live_handles={len(self._handles)}); lower coverage or "
                    "raise FASTVIDEO_HOST_RAM_BUDGET_GB")
            free = self._free_host.get(padded)
            if free:  # a spill may have freed a reusable same-size buffer
                return free.pop()
        buf = alloc_aligned_host(padded, pin=self.pin)
        self.budget.add_allocation(padded)
        return buf

    def _make_room(self, padded: int) -> bool:
        # A spill only helps the caller if it frees a *reusable* (same-size)
        # buffer; spilling a different-size resident is pure write traffic that
        # yields no usable slot. So, in order:
        # 1. spill a same-size resident (productive: frees a reusable buffer);
        victim = self._lowest_resident(padded=padded)
        if victim is not None:
            self._spill_out(victim)
            return True
        # 2. drop a free buffer of any size to reclaim allocation for a fresh
        #    same-size alloc -- avoids spilling residents whose buffers wouldn't
        #    match (write amplification with diverse generic saved-tensor sizes).
        #    CAVEAT: dropping the last reference returns the block to PyTorch's
        #    caching host allocator (bucketed by size), which does NOT return it
        #    to the OS, so the physical pinned RSS is not actually reclaimed and
        #    a later different-size alloc adds new RSS on top. The cap is thus
        #    EXACT only for uniform-size (SAC) workloads, where step 1 always
        #    fires; under mixed generic sizes it is a close but soft bound
        #    (real RSS may exceed cap by the cross-size fragmentation). torch
        #    2.12 exposes no host empty-cache to force the reclaim.
        for size, lst in self._free_host.items():
            if lst:
                lst.pop()
                self.budget.drop_allocation(size)
                return True
        # 3. nothing free to drop: spill any resident so its buffer can be
        #    dropped on the next pass.
        victim = self._lowest_resident()
        if victim is not None:
            self._spill_out(victim)
            return True
        return False

    def _has_room(self, padded: int) -> bool:
        """Room to make a buffer of ``padded`` bytes without evicting live data."""
        return bool(self._free_host.get(padded)) or self.budget.can_allocate(
            padded)

    def _release_host(self, buf: torch.Tensor) -> None:
        # Back to the free-list: still allocated (RSS unchanged), reusable.
        self._free_host[buf.numel()].append(buf)

    # -- SSD offset allocation (recycled per size across steps) ---------------

    def _alloc_offset(self, padded: int) -> int:
        free = self._free_offsets.get(padded)
        if free:
            return free.pop()
        off = self._next_offset
        self._next_offset += padded
        return off

    def _free_offset(self, off: int, padded: int) -> None:
        self._free_offsets[padded].append(off)

    # -- tensor reconstruction (contiguous only) ------------------------------

    @staticmethod
    def _as_tensor(h: SpillHandle) -> torch.Tensor:
        assert h.host is not None
        return h.host[:h.nbytes].view(h.dtype).view(h.shape)

    # -- forward: acquire (spills to make room) + admit -----------------------

    def acquire_staging(self, nbytes: int) -> torch.Tensor:
        """An aligned, padded uint8 buffer for the adapter to D2H-fill.

        Acquisition is where the cap is enforced: making room (spilling the
        lowest-index resident block) happens *before* any new allocation, so
        allocated RSS never exceeds the cap.
        """
        with self._lock:
            return self._acquire_host(nbytes)

    def discard_staging(self, buf: torch.Tensor) -> None:
        """Return a staging buffer acquired but never ``admit``ted.

        If an adapter faults between ``acquire_staging`` and ``admit`` (e.g. a
        CUDA error in the D2H fill), the buffer is not in ``_handles`` and
        ``reset_step`` cannot reclaim it, so it would leak budget bytes forever.
        The caller must first quiesce any pending fill into ``buf`` (its copy
        stream) so a later reuse cannot race the DMA.
        """
        with self._lock:
            self._release_host(buf)

    def admit(self,
              host: torch.Tensor,
              *,
              shape: tuple[int, ...],
              dtype: torch.dtype,
              nbytes: int,
              category: str = "sac",
              block_index: int = -1,
              policy_key: int | None = None,
              d2h_event: Any = None) -> SpillHandle:
        """Register an acquired, D2H-filled buffer as a resident handle."""
        with self._lock:
            h = SpillHandle(nbytes=nbytes,
                            padded=host.numel(),
                            shape=tuple(shape),
                            dtype=dtype,
                            state=Residency.RESIDENT,
                            host=host,
                            category=category,
                            production_index=self._next_production_index,
                            block_index=block_index,
                            policy_key=policy_key,
                            d2h_event=d2h_event)
            self._next_production_index += 1
            self._handles.append(h)
            return h

    def _lowest_resident(self,
                         padded: int | None = None) -> SpillHandle | None:
        # Evict by global production order (earliest produced = consumed last in
        # backward), across every category, so a mixed SAC + generic workload
        # spills the safest handle regardless of adapter. Skip reserved handles
        # (an adapter still holds their alias for an in-flight H2D). With
        # ``padded`` set, only consider residents whose buffer matches that size.
        best: SpillHandle | None = None
        for h in self._handles:
            if (h.state is Residency.RESIDENT and h.host is not None
                    and not h.reserved):
                if padded is not None and h.padded != padded:
                    continue
                if best is None or h.production_index < best.production_index:
                    best = h
        return best

    def _spill_out(self, h: SpillHandle) -> None:
        # Synchronous write under lock. Wait for the D2H fill to land first, or
        # the file would capture stale/partial bytes. The buffer returns to the
        # free-list (still allocated, reusable) -- RSS is unchanged by a spill.
        if h.d2h_event is not None:
            h.d2h_event.synchronize()
            h.d2h_event = None
        off = self._alloc_offset(h.padded)
        self.file.write(off, h.host)
        self._release_host(h.host)
        h.spill_offset = off
        h.host = None
        h.state = Residency.SPILLED
        self.n_spilled += 1
        self.spilled_bytes += h.nbytes
        self.n_spilled_by_category[h.category] += 1
        self.spilled_bytes_by_category[h.category] += h.nbytes

    # -- backward: prefetch + ensure_resident ---------------------------------

    def on_block_recompute_enter(self, block_index: int) -> None:
        """Prefetch spilled handles just below ``block_index`` (lookahead).

        Backward descends, so the blocks consumed *soonest* are the highest
        indices below the cursor. Prefetch nearest-first and stop once the
        budget is full, so a tight budget spends its headroom on the reads that
        are needed next rather than the farthest-future ones. Tagged generic
        saved tensors share this schedule; untagged generic tensors stay on the
        reactive path.
        """
        lo = block_index - self.prefetch_window
        with self._lock:
            candidates = [
                h for h in self._handles if h.state is Residency.SPILLED
                and h.block_index >= 0
                and lo <= h.block_index < block_index
                # Forward guard: a no-op today (both live categories match), but
                # keeps a future non-block category out of the block schedule.
                and h.category in {"sac", "saved_tensor"}
            ]
            candidates.sort(
                key=lambda h: (h.block_index, h.policy_key
                               if h.policy_key is not None else
                               h.production_index),
                reverse=True)
            for h in candidates:
                # Prefetch only into genuinely free headroom -- never evict a
                # still-needed resident block to prefetch a farther one.
                if not self._has_room(h.padded):
                    break
                self._start_read(h, prefetch=True)

    def _start_read(self, h: SpillHandle, *, prefetch: bool) -> None:
        # Called under lock. Acquire a buffer (reuse/alloc within cap) and kick
        # off the async read; the worker flips the handle to RESIDENT under lock
        # when the bytes land. ``prefetch`` records that a read was issued so a
        # completed read is later scored as a prefetch hit, not a resident hit.
        buf = self._acquire_host(h.nbytes)
        h.host = buf
        h.state = Residency.INFLIGHT_IN
        h.prefetched = prefetch
        off = h.spill_offset

        def _task() -> None:
            self.file.read_into(off, buf)
            with self._lock:
                self._free_offset(off, h.padded)
                h.spill_offset = -1
                h.state = Residency.RESIDENT
                self.n_reloaded += 1

        h.read_future = self._exec.submit(_task)

    def ensure_resident(self, h: SpillHandle) -> torch.Tensor:
        """Return the tensor, waiting on / issuing an SSD read if needed.

        The returned tensor aliases the pooled host buffer; the caller must
        consume it (e.g. copy H2D) and then call ``release`` before the buffer
        can be recycled.
        """
        with self._lock:
            if h.state is Residency.RESIDENT:
                # A spilled-then-prefetched block whose read already finished is
                # a prefetch hit, not a "never spilled" resident hit.
                if h.prefetched:
                    self.prefetch_hits += 1
                else:
                    self.resident_hits += 1
                h.reserved = True
                return self._as_tensor(h)
            if h.state is Residency.INFLIGHT_IN:
                # Prefetch was issued but the read hasn't landed, so reload will
                # block on it below: a late prefetch, not a clean hit. Counted
                # separately so prefetch_underruns==0 can't hide reload stalls.
                self.prefetch_late += 1
                fut = h.read_future
            elif h.state is Residency.SPILLED:
                # Not prefetched in time: read now and stall (a real underrun).
                self.prefetch_underruns += 1
                self._start_read(h, prefetch=False)
                fut = h.read_future
            elif h.state is Residency.DONE:
                # Already consumed and released once: its buffer was recycled, so
                # a second reload is impossible. Name the likely cause instead of
                # surfacing an opaque state error.
                raise RuntimeError(
                    "SSD spill: this saved tensor was already reloaded and its "
                    "buffer recycled; double backward / retain_graph=True is "
                    "not supported by SSD spill. Run a single backward, or "
                    "disable FASTVIDEO_SSD_SPILL / "
                    "FASTVIDEO_SAVED_TENSORS_SSD_SPILL for double-backward.")
            else:
                raise RuntimeError(f"ensure_resident on {h.state} handle")
        assert fut is not None
        fut.result()  # wait outside the lock
        with self._lock:
            h.reserved = True
            return self._as_tensor(h)

    def release(self, h: SpillHandle) -> None:
        """Return a consumed handle's buffer to the pool and drop the handle.

        The buffer stays allocated (reusable via the free-list); only the handle
        is removed so ``_handles`` does not grow across steps.
        """
        with self._lock:
            if h.state is Residency.INFLIGHT_IN:
                # An I/O worker is still reading into h.host; recycling it now
                # would corrupt whatever reuses the buffer. Callers must
                # ensure_resident() (which waits for the read) before release().
                raise RuntimeError(
                    "release() on an INFLIGHT_IN handle: the SSD read has not "
                    "completed; call ensure_resident() first")
            if h.host is not None:
                self._release_host(h.host)
                h.host = None
            if h.spill_offset >= 0:
                # A SPILLED handle released without a reload (public-API use)
                # must free its file offset, matching reset_step; otherwise the
                # offset leaks.
                self._free_offset(h.spill_offset, h.padded)
                h.spill_offset = -1
            h.reserved = False
            h.state = Residency.DONE
            with contextlib.suppress(ValueError):
                self._handles.remove(h)

    def reset_step(self) -> int:
        """Reclaim handles backward never consumed (orphans); return the count.

        Handles only leave ``_handles`` via ``release`` on the unpack/reload
        path. A saved tensor whose backward is never reached (partial
        ``autograd.grad``, an unused branch, a step aborted by a caught
        exception) would otherwise leak forever -- RESIDENT ones pin budget
        bytes, SPILLED ones leak file offsets -- ratcheting toward the
        "budget exceeded" error across steps. Call at each step boundary. A
        nonzero return means orphans were reclaimed and is worth logging.
        """
        with self._lock:
            futures = [
                h.read_future for h in self._handles
                if h.state is Residency.INFLIGHT_IN and h.read_future is not None
            ]
        for fut in futures:  # let in-flight reads finish before reclaiming
            fut.result()
        with self._lock:
            n = len(self._handles)
            for h in self._handles:
                if h.host is not None:
                    # An orphan may still have a pending D2H into h.host (its
                    # d2h_event never got synced by a spill/reload). Wait for it
                    # before returning the buffer to the pool -- a later
                    # prefetch preadv from an I/O thread would otherwise race the
                    # DMA. The two adapters use different copy streams sharing
                    # this pool, so pool-reuse safety rests on this sync.
                    if h.d2h_event is not None:
                        h.d2h_event.synchronize()
                        h.d2h_event = None
                    self._release_host(h.host)
                    h.host = None
                if h.spill_offset >= 0:
                    self._free_offset(h.spill_offset, h.padded)
                    h.spill_offset = -1
                h.reserved = False
                h.state = Residency.DONE
            self._handles.clear()
            return n

    def close(self) -> None:
        self._exec.shutdown(wait=True)
        self.file.close()
