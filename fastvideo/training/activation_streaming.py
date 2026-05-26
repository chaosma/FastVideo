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
    """Per-copy-stream registry of staged pack records.

    Phase 2 uses ``records`` (the CPU staging buffers + pack events) during
    on-demand unpack. ``live_buffers`` / ``restore_events`` / ``fwd_order``
    are scaffolded here for the Phase 3 prefetch scheduler that follows.
    """

    def __init__(self, copy_stream: torch.cuda.Stream):
        self.copy_stream = copy_stream
        # block_idx -> list[_Record], ordered by pack call within the block.
        self.records: dict[int, list[_Record]] = {}
        # Phase 3 fields (unused in Phase 2, kept so callers can opt in
        # without changing the registry shape later).
        self.live_buffers: dict[int, list[torch.Tensor]] = {}
        self.restore_events: dict[int, torch.cuda.Event] = {}
        self.fwd_order: list[int] = []
        self._fwd_order_seen: set[int] = set()

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
            rec = registry.consume(block_idx, slot)
            compute = torch.cuda.current_stream(rec.device)
            # If a Phase 3 prefetcher has already restored this slot, use it
            # straight away; otherwise stage the D2H now.
            live = registry.live_buffers.get(block_idx)
            if live is not None and slot < len(live) and live[slot] is not None:
                gpu = live[slot]
                restore_event = registry.restore_events.get(block_idx)
                if restore_event is not None:
                    compute.wait_event(restore_event)
                gpu.record_stream(compute)
                # Drop the live entry so the buffer can be released after
                # backward consumes it.
                live[slot] = None  # type: ignore[index]
                return gpu

            # Phase 2 path: on-demand restore on the copy stream.
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
    """Install the async saved-tensors hook around an inner module's forward.

    Compose as ``StreamedOffloadCheckpointWrapper(checkpoint_wrapper(block))``
    so the inner block's intermediates are recomputed (saving GPU memory)
    and only the block's external inputs flow through the offload hook --
    matching the design in §3 of ``memory_prefetch_plan.md``.
    """

    def __init__(self, mod: torch.nn.Module):
        super().__init__()
        self._streamed_offload_inner = mod
        self._block_idx = _next_block_idx()

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
        registry.new_call(self._block_idx)
        hook = AsyncCpuSaveHook(registry, self._block_idx)
        with hook:
            return self._streamed_offload_inner(*args, **kwargs)
