# SPDX-License-Identifier: Apache-2.0
"""Pack/unpack hook skeleton for activation tracing and CPU offload.

``ActivationTrace`` installs ``saved_tensors_hooks`` for the forward+backward
pass. CPU offload (``FASTVIDEO_SAVE_ON_CPU=1``) is implemented; tracing bodies
are stubs to be filled in.
"""
from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}


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
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.step = step
        self.offload_to_cpu = offload_to_cpu
        self.metadata = metadata or {}
        self._ctx: Any = None

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
        orig_device: torch.device | None = None
        if self.offload_to_cpu and tensor.is_cuda:
            orig_device = tensor.device
            # Always pin: pinned host memory enables non-blocking H2D copies
            # in the unpack hook, which overlaps transfer with backward compute.
            stored = torch.empty(tensor.size(),
                                 dtype=tensor.dtype,
                                 layout=tensor.layout,
                                 pin_memory=True)
            stored.copy_(tensor)
        else:
            stored = tensor.detach()
        return _SavedTensorHandle(tensor=stored, orig_device=orig_device)

    def _unpack_hook(self, handle: _SavedTensorHandle) -> torch.Tensor:
        # TODO: record saved_tensor_unpack event
        if handle.orig_device is not None:
            return handle.tensor.to(handle.orig_device, non_blocking=True)
        return handle.tensor


def maybe_activation_trace(
    model: torch.nn.Module,
    *,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> ActivationTrace | _NoopActivationTrace:
    """Return an ``ActivationTrace`` if ``FASTVIDEO_SAVE_ON_CPU=1``, else a no-op."""
    offload = (os.environ.get("FASTVIDEO_SAVE_ON_CPU",
                              "").strip().lower() in _TRUE_VALUES)
    if not offload:
        return _NoopActivationTrace()
    return ActivationTrace(model,
                           step=step,
                           offload_to_cpu=True,
                           metadata=metadata)
