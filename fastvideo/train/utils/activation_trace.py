# SPDX-License-Identifier: Apache-2.0
"""Pack/unpack hook skeleton for activation tracing and CPU offload.

``ActivationTrace`` installs ``saved_tensors_hooks`` for the forward pass. CPU
offload (``FASTVIDEO_SAVE_ON_CPU=1``) applies to tensors saved through those
hooks. Selective activation checkpointing keeps its own cache, so SAC-saved
operator outputs are not offloaded by this module.
"""
from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.training.block_context import current_block_index

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

_WARNED_SSD_BUDGET_EXHAUSTED = False


def _warn_ssd_budget_exhausted_once(detail: str) -> None:
    # SSD budget could not fit a saved tensor. Degrade to the pinned/GPU path
    # rather than crash; warn once so the config problem is visible.
    global _WARNED_SSD_BUDGET_EXHAUSTED
    if _WARNED_SSD_BUDGET_EXHAUSTED:
        return
    _WARNED_SSD_BUDGET_EXHAUSTED = True
    logger.warning(
        "Saved-tensor SSD spill budget exhausted (%s); falling back to the "
        "pinned host path for this and further oversized saves. Raise "
        "FASTVIDEO_HOST_RAM_BUDGET_GB to actually spill them.", detail)

# Dedicated per-device stream for host<->device copies so offload transfers
# overlap compute on the default stream instead of serializing against it.
# Keyed by device (not a single global stream) so that, sharing the manager's
# per-device pinned pool, a buffer's fill/read stay on that device's stream.
_COPY_STREAMS: "dict[torch.device, torch.cuda.Stream]" = {}


def _get_copy_stream(
        device: "torch.device | None" = None) -> "torch.cuda.Stream | None":
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    stream = _COPY_STREAMS.get(device)
    if stream is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
        _COPY_STREAMS[device] = stream
    return stream


@dataclass(slots=True)
class _SavedTensorHandle:
    tensor: torch.Tensor
    # Non-None when the tensor was offloaded: holds the original CUDA
    # device so unpack can move it back.
    orig_device: torch.device | None = None


@dataclass(slots=True)
class _SavedTensorSsdHandle:
    """A saved tensor whose bytes are managed by the SSD spill tier.

    Correctness never depends on order: PyTorch hands the exact handle object
    back to unpack, so we reload via ``spill_handle`` directly. Only the shape /
    dtype needed to rebuild the device tensor are carried (generic offload is
    contiguous-only, so a contiguous rebuild matches the saved tensor).
    """
    manager: Any
    spill_handle: Any
    orig_device: torch.device
    shape: tuple[int, ...]
    dtype: torch.dtype
    pack_index: int
    block_index: int | None


class _NoopActivationTrace(AbstractContextManager["_NoopActivationTrace"]):
    enabled = False

    def __enter__(self) -> "_NoopActivationTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_phase(self, phase: str) -> None:
        pass

    def log_marker(self,
                   event: str,
                   payload: dict[str, Any] | None = None) -> None:
        pass


class ActivationTrace(AbstractContextManager["ActivationTrace"]):
    """saved_tensors_hooks context: CPU offload + tracing skeleton."""
    enabled = True

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        step: int,
        offload_to_cpu: bool = False,
        min_offload_bytes: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.step = step
        self.offload_to_cpu = offload_to_cpu
        # Only tensors at least this large are offloaded; smaller saved tensors
        # stay on GPU (the transfer isn't worth the pinned alloc + sync).
        self.min_offload_bytes = min_offload_bytes
        self.metadata = metadata or {}
        self._ctx: Any = None
        # Diagnostics: what actually got offloaded this forward. Lets us confirm
        # whether the large self-attn SDPA outputs reach these hooks (they may
        # be held in SAC's own cache and bypass saved_tensors_hooks entirely).
        self._offloaded_bytes = 0
        self._offloaded_count = 0
        self._offloaded_shapes: set[tuple[int, ...]] = set()
        # Generic saved-tensor SSD spill (opt-in, separate gate). When on,
        # eligible offloaded tensors are routed to the shared per-device
        # SsdSpillManager instead of the uncapped pinned path.
        from fastvideo.training import ssd_config
        self._ssd_spill = ssd_config.saved_tensors_ssd_enabled()
        self._pack_index = 0
        # Tensors routed to the SSD manager this forward. NOTE: "admitted" is
        # not "spilled" -- an admitted handle may stay RAM-resident and never be
        # written to disk. Actual disk spills are the manager's per-category
        # n_spilled counters (see ssd_config stats).
        self._ssd_admitted_bytes = 0
        self._ssd_admitted_count = 0
        self._ssd_block_tagged_bytes = 0
        self._ssd_block_tagged_count = 0

    def __enter__(self) -> "ActivationTrace":
        self._ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook,
            self._unpack_hook,
        )
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc, tb)
            self._ctx = None
        # Log once on the first couple of steps so the offload coverage can be
        # verified cheaply before committing to a long run.
        if self.offload_to_cpu and self.step < 2:
            logger.info(
                "saved_tensors_hooks offload step=%d: %d tensors, %.2f GB "
                "(pinned); ssd-managed %d tensors, %.2f GB (admitted to the "
                "spill manager, not necessarily written to disk); "
                "block-tagged %d tensors, %.2f GB; shapes=%s",
                self.step, self._offloaded_count,
                self._offloaded_bytes / 1e9, self._ssd_admitted_count,
                self._ssd_admitted_bytes / 1e9,
                self._ssd_block_tagged_count,
                self._ssd_block_tagged_bytes / 1e9,
                sorted(self._offloaded_shapes))
        return False

    def set_phase(self, phase: str) -> None:
        # TODO: record phase transition
        pass

    def log_marker(self,
                   event: str,
                   payload: dict[str, Any] | None = None) -> None:
        # TODO: record marker event
        pass

    def _pack_hook(self, tensor: torch.Tensor):
        # TODO: record saved_tensor_pack event
        # Only contiguous tensors are offloaded: the host buffer below is
        # contiguous, so offloading a strided tensor would silently change its
        # layout on reload. Non-contiguous saves stay on GPU.
        eligible = (self.offload_to_cpu and tensor.is_cuda
                    and tensor.is_contiguous()
                    and tensor.numel() * tensor.element_size()
                    >= self.min_offload_bytes)
        copy_stream = _get_copy_stream(tensor.device) if eligible else None
        if copy_stream is None:
            return _SavedTensorHandle(tensor=tensor.detach(), orig_device=None)
        # SSD spill tier (opt-in): route through the capped manager instead of
        # the uncapped pinned pool. Falls back below if the manager gate fails.
        if self._ssd_spill:
            ssd = self._pack_ssd(tensor, copy_stream)
            if ssd is not None:
                return ssd
        # Pinned host memory enables non-blocking, overlappable copies.
        stored = torch.empty(tensor.size(),
                             dtype=tensor.dtype,
                             layout=tensor.layout,
                             pin_memory=True)
        # D2H on the side stream so forward compute (default stream) is not
        # blocked; the saved output isn't needed again until backward.
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            stored.copy_(tensor, non_blocking=True)
        # Keep the GPU buffer alive until the side-stream D2H consumes it.
        tensor.record_stream(copy_stream)
        self._offloaded_bytes += tensor.numel() * tensor.element_size()
        self._offloaded_count += 1
        self._offloaded_shapes.add(tuple(tensor.shape))
        return _SavedTensorHandle(tensor=stored, orig_device=tensor.device)

    def _pack_ssd(self, tensor: torch.Tensor,
                  copy_stream) -> "_SavedTensorSsdHandle | None":
        from fastvideo.training import ssd_config
        mgr = ssd_config.ssd_manager(tensor.device)
        if mgr is None:
            return None
        nbytes = tensor.numel() * tensor.element_size()
        try:
            staging = mgr.acquire_staging(nbytes)  # aligned, padded, pinned
        except RuntimeError as e:
            # Budget exhausted: degrade to the pinned/GPU-resident path (return
            # None) instead of crashing the run mid-forward.
            _warn_ssd_budget_exhausted_once(str(e))
            return None
        try:
            dst = staging[:nbytes].view(tensor.dtype).view(tensor.shape)
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                dst.copy_(tensor, non_blocking=True)
            tensor.record_stream(copy_stream)
            event = torch.cuda.Event()
            event.record(copy_stream)  # manager waits on this before SSD write
            self._pack_index += 1
            block_index = current_block_index()
            spill_handle = mgr.admit(staging,
                                     shape=tuple(tensor.shape),
                                     dtype=tensor.dtype,
                                     nbytes=nbytes,
                                     category="saved_tensor",
                                     block_index=(block_index
                                                  if block_index is not None
                                                  else -1),
                                     policy_key=self._pack_index,
                                     d2h_event=event)
        except Exception:
            # Return the still-unadmitted buffer to the pool so a fault here does
            # not leak budget bytes; quiesce any enqueued D2H first.
            try:
                copy_stream.synchronize()
            except Exception:
                pass
            mgr.discard_staging(staging)
            raise
        self._ssd_admitted_bytes += nbytes
        self._ssd_admitted_count += 1
        if block_index is not None:
            self._ssd_block_tagged_bytes += nbytes
            self._ssd_block_tagged_count += 1
        return _SavedTensorSsdHandle(manager=mgr,
                                     spill_handle=spill_handle,
                                     orig_device=tensor.device,
                                     shape=tuple(tensor.shape),
                                     dtype=tensor.dtype,
                                     pack_index=self._pack_index,
                                     block_index=block_index)

    def _unpack_hook(self, handle) -> torch.Tensor:
        # TODO: record saved_tensor_unpack event
        if isinstance(handle, _SavedTensorSsdHandle):
            return self._unpack_ssd(handle)
        if handle.orig_device is None:
            return handle.tensor
        copy_stream = _get_copy_stream(handle.orig_device)
        if copy_stream is None:
            return handle.tensor.to(handle.orig_device)
        current = torch.cuda.current_stream(handle.orig_device)
        with torch.cuda.stream(copy_stream):
            device_tensor = handle.tensor.to(handle.orig_device,
                                             non_blocking=True)
        # The consumer on the default stream must wait for the H2D to finish.
        current.wait_stream(copy_stream)
        device_tensor.record_stream(current)
        return device_tensor

    def _unpack_ssd(self, handle: _SavedTensorSsdHandle) -> torch.Tensor:
        # Exact-handle reload (reactive v1: no order prediction). Generic
        # offload is contiguous-only, so a contiguous rebuild matches the save.
        host_tensor = handle.manager.ensure_resident(handle.spill_handle)
        copy_stream = _get_copy_stream(handle.orig_device)
        current = torch.cuda.current_stream(handle.orig_device)
        with torch.cuda.stream(copy_stream):
            device_tensor = torch.empty(handle.shape,
                                        dtype=handle.dtype,
                                        device=handle.orig_device)
            device_tensor.copy_(host_tensor, non_blocking=True)
        current.wait_stream(copy_stream)
        device_tensor.record_stream(current)
        # Conservative: sync before release so the manager cannot recycle the
        # staging buffer (via a prefetch read) while this H2D still reads it.
        copy_stream.synchronize()
        handle.manager.release(handle.spill_handle)
        return device_tensor


def maybe_activation_trace(
    model: torch.nn.Module,
    *,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> ActivationTrace | _NoopActivationTrace:
    """Return trace hooks if ``FASTVIDEO_SAVE_ON_CPU=1``, else a no-op.

    ``FASTVIDEO_OFFLOAD_MIN_MB`` (default 1) sets the minimum tensor size for
    hook-based offload; smaller tensors stay on GPU. SAC's internal save cache
    bypasses these hooks, so SAC-saved attention outputs remain on GPU.
    """
    offload = (os.environ.get("FASTVIDEO_SAVE_ON_CPU",
                              "").strip().lower() in _TRUE_VALUES)
    if not offload:
        return _NoopActivationTrace()
    min_mb = float(os.environ.get("FASTVIDEO_OFFLOAD_MIN_MB", "1") or "1")
    return ActivationTrace(model,
                           step=step,
                           offload_to_cpu=True,
                           min_offload_bytes=int(min_mb * 1024 * 1024),
                           metadata=metadata)
