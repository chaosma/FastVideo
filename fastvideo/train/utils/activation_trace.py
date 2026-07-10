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

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Dedicated stream for host<->device copies so offload transfers overlap
# compute on the default stream instead of serializing against it. Created
# lazily and shared across steps.
_COPY_STREAM: "torch.cuda.Stream | None" = None


def _get_copy_stream() -> "torch.cuda.Stream | None":
    global _COPY_STREAM
    if not torch.cuda.is_available():
        return None
    if _COPY_STREAM is None:
        _COPY_STREAM = torch.cuda.Stream()
    return _COPY_STREAM


@dataclass(slots=True)
class _SavedTensorHandle:
    tensor: torch.Tensor
    # Non-None when the tensor was offloaded: holds the original CUDA
    # device so unpack can move it back.
    orig_device: torch.device | None = None


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
                "saved_tensors_hooks offload step=%d: %d tensors, %.2f GB; "
                "shapes=%s",
                self.step, self._offloaded_count,
                self._offloaded_bytes / 1e9,
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

    def _pack_hook(self, tensor: torch.Tensor) -> _SavedTensorHandle:
        # TODO: record saved_tensor_pack event
        # Only contiguous tensors are offloaded: the host buffer below is
        # contiguous, so offloading a strided tensor would silently change its
        # layout on reload. Non-contiguous saves stay on GPU.
        eligible = (self.offload_to_cpu and tensor.is_cuda
                    and tensor.is_contiguous()
                    and tensor.numel() * tensor.element_size()
                    >= self.min_offload_bytes)
        copy_stream = _get_copy_stream() if eligible else None
        if copy_stream is None:
            return _SavedTensorHandle(tensor=tensor.detach(), orig_device=None)
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

    def _unpack_hook(self, handle: _SavedTensorHandle) -> torch.Tensor:
        # TODO: record saved_tensor_unpack event
        if handle.orig_device is None:
            return handle.tensor
        copy_stream = _get_copy_stream()
        if copy_stream is None:
            return handle.tensor.to(handle.orig_device)
        current = torch.cuda.current_stream()
        with torch.cuda.stream(copy_stream):
            device_tensor = handle.tensor.to(handle.orig_device,
                                             non_blocking=True)
        # The consumer on the default stream must wait for the H2D to finish.
        current.wait_stream(copy_stream)
        device_tensor.record_stream(current)
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
