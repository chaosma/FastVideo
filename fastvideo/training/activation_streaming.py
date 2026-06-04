"""Async activation streaming on a dedicated CUDA copy stream.

Phase 2 of the fine-grained activation streaming plan (see
``memory_prefetch_plan.md``). Replaces ``torch.autograd.graph.save_on_cpu`` --
which runs H2D / D2H on the compute stream and therefore can't overlap with
kernels -- with a saved-tensors-hook pair that uses a separate CUDA stream.

The Phase 2 deliverable on its own (no prefetch scheduler) is expected to hit
the same GPU peak as ``offload_wrapper``; the wins come in Phase 3 when the
``BlockRegistry`` scaffolding here gets driven by a one-layer-ahead
prefetcher.

Pieces
------
- ``get_copy_stream(device)`` -- module-level singleton ``torch.cuda.Stream``
  per device.
- ``BlockRegistry`` -- per-stream record of staged pack metadata. Phase 2
  reads it during on-demand unpack; Phase 3 reuses it for prefetch.
- ``AsyncCpuSaveHook`` -- ``saved_tensors_hooks`` subclass. Pack stages an
  H2D copy on the copy stream; unpack stages the D2H restore on the same
  stream and forces the compute stream to wait before backward consumes the
  restored tensor.
- ``StreamedOffloadCheckpointWrapper`` -- nn.Module wrapper that installs the
  hook around the (already-checkpoint-wrapped) inner module's forward.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch.autograd.graph import saved_tensors_hooks

__all__ = [
    "get_copy_stream",
    "get_registry",
    "BlockRegistry",
    "AsyncCpuSaveHook",
    "StreamedOffloadCheckpointWrapper",
]


_copy_streams: dict[int, torch.cuda.Stream] = {}
_registries: dict[int, "BlockRegistry"] = {}
_streams_lock = threading.Lock()


def _resolve_device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.index is None:
            return torch.cuda.current_device()
        return device.index
    return int(device)


def get_copy_stream(
        device: torch.device | int | None = None) -> torch.cuda.Stream:
    """Return the shared copy stream for ``device`` (one per CUDA device)."""
    dev = _resolve_device_index(device)
    with _streams_lock:
        s = _copy_streams.get(dev)
        if s is None:
            with torch.cuda.device(dev):
                # priority=-1 (higher than the default 0) so FSDP all-gather
                # kernels on the compute stream don't starve the copy lane.
                s = torch.cuda.Stream(device=dev, priority=-1)
            _copy_streams[dev] = s
        return s


class _Record:
    """Per-packed-tensor staging entry held by the BlockRegistry."""

    __slots__ = ("cpu", "pack_event", "shape", "dtype", "device")

    def __init__(self, cpu: torch.Tensor | None, pack_event: torch.cuda.Event,
                 shape: torch.Size, dtype: torch.dtype,
                 device: torch.device):
        self.cpu = cpu
        self.pack_event = pack_event
        self.shape = shape
        self.dtype = dtype
        self.device = device


class BlockRegistry:
    """Per-copy-stream registry of staged pack records and the Phase 3
    one-layer-ahead prefetch scheduler.

    Forward path
        ``new_call(block_idx)`` is called at the entry of each forward of a
        wrapped block, then ``register_pack`` is called once per saved
        tensor by ``AsyncCpuSaveHook.pack``.

    Backward path (Phase 3)
        ``prefetch(block_idx)`` stages D2H for the recorded packs onto GPU
        on the copy stream; the unpack hook then consumes from
        ``live_buffers``. The ``prefetch_sem`` semaphore bounds the queue
        to one in-flight neighbour at a time.

    Cross-step lifecycle
        ``begin_step`` resets per-step state without forgetting fwd_order;
        Block ordering is stable across steps in a normal training loop.
    """

    def __init__(self, copy_stream: torch.cuda.Stream):
        self.copy_stream = copy_stream
        # block_idx -> list[_Record], ordered by pack call within the block.
        self.records: dict[int, list[_Record]] = {}
        # Restored GPU buffers, populated by prefetch(); consumed by unpack.
        self.live_buffers: dict[int, list[torch.Tensor | None]] = {}
        self.restore_events: dict[int, torch.cuda.Event] = {}
        # Forward execution order (first-seen). Reversed for backward driving.
        self.fwd_order: list[int] = []
        self._fwd_order_seen: set[int] = set()
        # Semaphore bounding one in-flight prefetched neighbour at a time.
        # Released by the post-bwd hook of the just-finished layer.
        self.prefetch_sem = threading.Semaphore(1)
        # Tracks which prefetches have acquired the semaphore so post-bwd
        # can release exactly once per acquire.
        self._prefetch_acquired: set[int] = set()
        # True while an end-of-backward sweep callback is registered with
        # the autograd engine for the in-flight backward pass.
        self._finalizer_armed = False

    def new_call(self, block_idx: int) -> None:
        """Reset per-call pack state for ``block_idx``.

        Must be called at the entry of every forward of the wrapped block,
        because pack records from a prior step would otherwise accumulate.
        """
        self.records[block_idx] = []
        if block_idx not in self._fwd_order_seen:
            self.fwd_order.append(block_idx)
            self._fwd_order_seen.add(block_idx)

    def register_pack(self, block_idx: int, record: _Record) -> int:
        rec_list = self.records.setdefault(block_idx, [])
        slot = len(rec_list)
        rec_list.append(record)
        return slot

    def consume(self, block_idx: int, slot: int) -> _Record:
        return self.records[block_idx][slot]

    # ------------------------------------------------------------------
    # Phase 3 — backward prefetch scheduler
    # ------------------------------------------------------------------

    def next_bwd_block(self, block_idx: int) -> int | None:
        """Block that will run backward immediately after ``block_idx``.

        Derived from the insertion order of ``records``: ``new_call``
        re-inserts every block at forward entry and cleanup pops it after
        its backward, so the keys are exactly the blocks of the *current*
        step in forward order. ``fwd_order`` is unsuitable here — it is
        first-seen-ever and accumulates block indices across wrapper sets
        (e.g. several models built in one process), which would make the
        scheduler prefetch blocks whose model is long gone.
        """
        pending = list(self.records.keys())
        try:
            pos = pending.index(block_idx)
        except ValueError:
            return None
        if pos == 0:
            return None
        return pending[pos - 1]

    def prefetch(self, block_idx: int, *, acquire_sem: bool) -> None:
        """Stage D2H for every packed record of ``block_idx`` onto GPU.

        Idempotent: a second call while live_buffers is still populated is
        a no-op. If ``acquire_sem`` is True the call blocks until the
        prefetch semaphore is free (post-bwd of the previously-finished
        layer releases it).
        """
        if block_idx in self.live_buffers:
            # Already prefetched (or no records).
            return
        records = self.records.get(block_idx)
        if not records:
            # Block packed no CUDA tensors (e.g. trivial passthrough).
            self.live_buffers[block_idx] = []
            return

        if acquire_sem:
            self.prefetch_sem.acquire()
            self._prefetch_acquired.add(block_idx)

        copy_stream = self.copy_stream
        bufs: list[torch.Tensor | None] = [None] * len(records)
        with torch.cuda.stream(copy_stream):
            for slot, rec in enumerate(records):
                if rec.cpu is None:
                    # Slot already consumed by an out-of-band restore
                    # (shouldn't happen in normal flow, but be defensive).
                    continue
                copy_stream.wait_event(rec.pack_event)
                gpu = torch.empty(rec.shape,
                                  dtype=rec.dtype,
                                  device=rec.device)
                gpu.copy_(rec.cpu, non_blocking=True)
                bufs[slot] = gpu
            restore_event = torch.cuda.Event()
            restore_event.record(copy_stream)

        self.live_buffers[block_idx] = bufs
        self.restore_events[block_idx] = restore_event

    def mark_active(self, block_idx: int) -> None:
        """Called by pre-bwd: ``block_idx`` graduates from "prefetched
        neighbour" to "active block being backpropagated".

        Releases the prefetch semaphore so the next neighbour's prefetch
        (kicked off later in the same pre-bwd hook) can proceed. Idempotent
        for blocks that never held the semaphore (the very first block in
        backward, which prefetched itself with ``acquire_sem=False``).
        """
        if block_idx in self._prefetch_acquired:
            self._prefetch_acquired.discard(block_idx)
            self.prefetch_sem.release()

    def release_after_bwd(self, block_idx: int) -> None:
        """Drop live_buffers, restore_event, and packed records for a block
        whose backward just finished.

        Compute-stream lifetimes are managed via ``record_stream`` in the
        unpack hook, so dropping the Python references here is safe even
        if the kernels are still running.

        Caveat: for a module none of whose *inputs* require grad (the
        first block of a model fed a leaf without requires_grad), the
        ``full_backward_hook`` fires degenerately early — at output-grad
        time, BEFORE the block's own backward (and unpack) has run.
        Releasing then would destroy state the imminent unpack still
        needs. Detect that case via unconsumed live slots and leave the
        cleanup to ``finalize_backward`` instead.
        """
        live = self.live_buffers.get(block_idx)
        if live is not None and any(b is not None for b in live):
            # Prefetched buffers not yet consumed by unpack: this is the
            # early-firing-hook case described above. Do nothing; the
            # end-of-backward finalizer sweeps this block's state.
            return
        self.live_buffers.pop(block_idx, None)
        self.restore_events.pop(block_idx, None)
        # Defensive: if for some reason mark_active was never called,
        # release here so a stuck semaphore doesn't poison the next step.
        if block_idx in self._prefetch_acquired:
            self._prefetch_acquired.discard(block_idx)
            self.prefetch_sem.release()
        # Clear the now-consumed CPU records to release pinned-host pages;
        # future steps will re-pack and re-allocate.
        self.records.pop(block_idx, None)

    # ------------------------------------------------------------------
    # End-of-backward finalizer
    # ------------------------------------------------------------------

    def arm_finalizer(self) -> None:
        """Register a once-per-backward callback that sweeps all remaining
        registry state when the backward pass completes.

        This is the backstop for hook-ordering quirks (see
        ``release_after_bwd``): whatever the per-block hooks failed to
        clean is guaranteed gone before ``loss.backward()`` returns, so
        no stale entries can leak into the next step.
        """
        if self._finalizer_armed:
            return
        self._finalizer_armed = True
        torch.autograd.Variable._execution_engine.queue_callback(
            self.finalize_backward)

    def finalize_backward(self) -> None:
        self._finalizer_armed = False
        self.live_buffers.clear()
        self.restore_events.clear()
        self.records.clear()
        for _ in range(len(self._prefetch_acquired)):
            self.prefetch_sem.release()
        self._prefetch_acquired.clear()


def get_registry(device: torch.device | int | None = None) -> BlockRegistry:
    """Return the BlockRegistry bound to ``device``'s copy stream."""
    stream = get_copy_stream(device)
    dev = stream.device.index
    assert dev is not None
    with _streams_lock:
        reg = _registries.get(dev)
        if reg is None:
            reg = BlockRegistry(stream)
            _registries[dev] = reg
        return reg


_PACK_TAG = "fv_streamed_offload_v1"


class AsyncCpuSaveHook(saved_tensors_hooks):
    """Per-block ``saved_tensors_hooks`` running H2D / D2H on a copy stream.

    A fresh instance is created for each forward call of a
    ``StreamedOffloadCheckpointWrapper`` so the pack/unpack closures bind to
    the correct registry + block_idx.
    """

    def __init__(self, registry: BlockRegistry, block_idx: int):
        self._registry = registry
        self._block_idx = block_idx
        copy_stream = registry.copy_stream

        def pack_hook(t: torch.Tensor) -> Any:
            # Pass through non-CUDA tensors (small scalars, RNG state if it
            # ever sneaks in) and zero-size tensors. The bookkeeping cost
            # isn't worth it on either.
            if t.device.type != "cuda" or t.numel() == 0:
                return t
            # Order H2D after whatever compute stream produced t.
            copy_stream.wait_stream(torch.cuda.current_stream())
            cpu = torch.empty(
                t.size(),
                dtype=t.dtype,
                layout=t.layout,
                pin_memory=True,
            )
            with torch.cuda.stream(copy_stream):
                cpu.copy_(t, non_blocking=True)
                pack_event = torch.cuda.Event()
                pack_event.record(copy_stream)
            # Keep the source GPU buffer alive on the copy stream until the
            # H2D actually drains -- the autograd engine drops its reference
            # to t once pack returns.
            t.record_stream(copy_stream)
            rec = _Record(
                cpu=cpu,
                pack_event=pack_event,
                shape=t.size(),
                dtype=t.dtype,
                device=t.device,
            )
            slot = registry.register_pack(self._block_idx, rec)
            return (_PACK_TAG, self._block_idx, slot)

        def unpack_hook(packed: Any) -> torch.Tensor:
            if not (isinstance(packed, tuple) and len(packed) == 3
                    and packed[0] is _PACK_TAG):
                # Passthrough for non-CUDA packs returned verbatim above.
                return packed
            _, block_idx, slot = packed

            # Phase 3 fast path: if a prefetcher has already restored
            # this slot, take it without touching ``records`` -- the
            # post-bwd hook may have already cleared records for the
            # previous block.
            live = registry.live_buffers.get(block_idx)
            if live is not None and slot < len(live) and live[slot] is not None:
                gpu = live[slot]
                compute = torch.cuda.current_stream(gpu.device)
                restore_event = registry.restore_events.get(block_idx)
                if restore_event is not None:
                    compute.wait_event(restore_event)
                gpu.record_stream(compute)
                # Drop the live entry so the buffer can be released as
                # soon as backward kernels are done consuming it.
                live[slot] = None  # type: ignore[index]
                return gpu

            # Phase 2 path: on-demand restore on the copy stream.
            rec = registry.consume(block_idx, slot)
            compute = torch.cuda.current_stream(rec.device)
            copy_stream.wait_event(rec.pack_event)
            with torch.cuda.stream(copy_stream):
                gpu = torch.empty(rec.shape,
                                  dtype=rec.dtype,
                                  device=rec.device)
                gpu.copy_(rec.cpu, non_blocking=True)
                restore_event = torch.cuda.Event()
                restore_event.record(copy_stream)
            compute.wait_event(restore_event)
            # Tell the caching allocator the compute stream now uses gpu so
            # the buffer isn't recycled while backward kernels are mid-flight.
            gpu.record_stream(compute)
            # Drop the CPU reference -- this slot has only one consumer.
            rec.cpu = None
            return gpu

        super().__init__(pack_hook, unpack_hook)


_block_idx_counter = 0
_block_idx_lock = threading.Lock()


def _next_block_idx() -> int:
    global _block_idx_counter
    with _block_idx_lock:
        idx = _block_idx_counter
        _block_idx_counter += 1
    return idx


class StreamedOffloadCheckpointWrapper(torch.nn.Module):
    """Install the async saved-tensors hook around an inner module's forward,
    plus the Phase 3 backward-driven prefetch scheduler.

    Compose as ``StreamedOffloadCheckpointWrapper(checkpoint_wrapper(block))``
    so the inner block's intermediates are recomputed (saving GPU memory)
    and only the block's external inputs flow through the offload hook --
    matching the design in §3 of ``memory_prefetch_plan.md``.

    Backward scheduler
        - ``full_backward_pre_hook``: fires before the wrapper's backward
          kernels. If this is the first block in backward (last in
          forward), prefetch its own saves so the about-to-run unpack
          finds ``live_buffers`` populated. Then always kick off the
          prefetch for the next block in the backward order.
        - ``full_backward_hook``: fires after the wrapper's backward
          finishes. Frees ``live_buffers`` for this block and releases
          the prefetch semaphore.
    """

    def __init__(self, mod: torch.nn.Module):
        super().__init__()
        self._streamed_offload_inner = mod
        self._block_idx = _next_block_idx()
        self._cached_registry: BlockRegistry | None = None
        self.register_full_backward_pre_hook(self._streamed_pre_backward)
        self.register_full_backward_hook(self._streamed_post_backward)

    def forward(self, *args, **kwargs):
        device: int | None = None
        for a in args:
            if torch.is_tensor(a) and a.is_cuda:
                device = a.device.index
                break
        if device is None:
            for v in kwargs.values():
                if torch.is_tensor(v) and v.is_cuda:
                    device = v.device.index
                    break
        registry = get_registry(device)
        self._cached_registry = registry
        registry.new_call(self._block_idx)
        hook = AsyncCpuSaveHook(registry, self._block_idx)
        with hook:
            return self._streamed_offload_inner(*args, **kwargs)

    # ------------------------------------------------------------------
    # Phase 3 backward hooks
    # ------------------------------------------------------------------

    def _streamed_pre_backward(self, _module, _grad_output) -> None:
        registry = self._cached_registry
        if registry is None:
            return
        # Guarantee a state sweep when this backward pass ends, whatever
        # the per-block hook ordering turns out to be.
        registry.arm_finalizer()
        # First block in backward (= last in forward) has no preceding
        # prefetch from a neighbour. Restore self inline so unpack finds
        # the live buffer ready when recompute fires.
        if self._block_idx not in registry.live_buffers:
            registry.prefetch(self._block_idx, acquire_sem=False)
        # Transition this block from "prefetched neighbour" (semaphore
        # held) to "currently active": release the semaphore so the
        # next-neighbour prefetch below can acquire.
        registry.mark_active(self._block_idx)
        # Kick off prefetch for the next block in backward order. The
        # semaphore now bounds the queue depth to one in-flight neighbour.
        nxt = registry.next_bwd_block(self._block_idx)
        if nxt is not None:
            registry.prefetch(nxt, acquire_sem=True)

    def _streamed_post_backward(self, _module, _grad_input,
                                _grad_output) -> None:
        registry = self._cached_registry
        if registry is None:
            return
        registry.release_after_bwd(self._block_idx)
