# SPDX-License-Identifier: Apache-2.0
"""Opt-in activation lifecycle tracing for training.

The tracer is intentionally observational: it records autograd saved-tensor
pack/unpack events, selected module forward outputs, output-gradient arrivals,
and CUDA allocator samples. It does not offload or prefetch tensors.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


_TRUE_VALUES = {"1", "true", "yes", "on"}
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")
# Strip wrapper-introduced name segments so downstream regexes can match the
# logical module path. ``_orig_mod`` is inserted by ``torch.compile``;
# ``_checkpoint_wrapped_module`` by ``torch.distributed.algorithms._checkpoint``
# ``checkpoint_wrapper`` used for full activation checkpointing.
_WRAPPER_SEGMENT_RE = re.compile(
    r"(?:_orig_mod|_checkpoint_wrapped_module)\.")


def _normalize_module_path(name: str) -> str:
    return _WRAPPER_SEGMENT_RE.sub("", name)


@dataclass(slots=True)
class _SavedTensorHandle:
    tensor: torch.Tensor
    save_event_id: int
    module_path: str | None
    layer: int | None
    # When the tensor was offloaded to CPU at pack time, this is the
    # original CUDA device it must be copied back to at unpack time.
    # None means no offload happened --- ``tensor`` is the original
    # detached tensor and is returned to autograd as-is.
    orig_device: torch.device | None = None


class _NoopActivationTrace(AbstractContextManager["_NoopActivationTrace"]):
    enabled = False

    def __enter__(self) -> "_NoopActivationTrace":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def set_phase(self, phase: str) -> None:
        del phase

    def log_marker(self, event: str, payload: dict[str, Any] | None = None) -> None:
        del event, payload


class ActivationTrace(AbstractContextManager["ActivationTrace"]):
    enabled = True

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        trace_dir: str,
        step: int,
        rank: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.trace_dir = Path(trace_dir)
        self.step = int(step)
        self.rank = int(rank)
        self.metadata = metadata or {}
        self.phase = "init"
        self.module_detail = os.environ.get(
            "FASTVIDEO_ACTIVATION_TRACE_MODULE_DETAIL",
            "block",
        ).strip().lower()
        self.with_memory = os.environ.get(
            "FASTVIDEO_ACTIVATION_TRACE_MEMORY",
            "1",
        ).strip().lower() in _TRUE_VALUES
        self.flush_every = int(os.environ.get(
            "FASTVIDEO_ACTIVATION_TRACE_FLUSH_EVERY",
            "1000",
        ))
        # Compose ``torch.autograd.graph.save_on_cpu`` behavior into the
        # tracer's own pack/unpack. ``saved_tensors_hooks`` is
        # winner-takes-all (innermost context only); since the tracer's
        # hooks are entered inside ``trainer.run()``'s save_on_cpu
        # wrapper, the tracer is innermost and would otherwise preempt
        # offload entirely. Doing the D2H/H2D copy inside our pack/unpack
        # keeps both the lifecycle JSONL and the offload working.
        self.offload_to_cpu = os.environ.get(
            "FASTVIDEO_SAVE_ON_CPU",
            "",
        ).strip().lower() in _TRUE_VALUES
        self.offload_pin = os.environ.get(
            "FASTVIDEO_SAVE_ON_CPU_PIN",
            "1",
        ).strip().lower() in _TRUE_VALUES
        self._module_stack: list[str] = []
        self._handles: list[Any] = []
        self._event_id = 0
        self._events_since_flush = 0
        self._file = None
        self._saved_tensor_ctx = None

    def __enter__(self) -> "ActivationTrace":
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / (
            f"activation_trace_rank{self.rank:05d}_step{self.step:06d}.jsonl"
        )
        self._file = path.open("w", encoding="utf-8")
        self._register_module_hooks()
        self._saved_tensor_ctx = torch.autograd.graph.saved_tensors_hooks(
            self._pack_hook,
            self._unpack_hook,
        )
        self._saved_tensor_ctx.__enter__()
        self.log_marker("trace_start", {
            "trace_path": str(path),
            "module_detail": self.module_detail,
            "offload_to_cpu": self.offload_to_cpu,
            "offload_pin": self.offload_pin if self.offload_to_cpu else None,
            "metadata": self.metadata,
        })
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.log_marker("trace_exception", {
                "exc_type": getattr(exc_type, "__name__", str(exc_type)),
                "exc": str(exc),
            })
        self.log_marker("trace_end", {})
        if self._saved_tensor_ctx is not None:
            self._saved_tensor_ctx.__exit__(exc_type, exc, tb)
            self._saved_tensor_ctx = None
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        return False

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase)
        self.log_marker("phase", {"phase": self.phase})

    def log_marker(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self._write_event({
            "event": event,
            "payload": self._jsonable(payload or {}),
        })

    def _pack_hook(self, tensor: torch.Tensor) -> _SavedTensorHandle:
        module_path = self._current_module()
        layer = self._extract_layer(module_path)
        event_id = self._next_event_id()
        # Log the ORIGINAL tensor's metadata (device=cuda for offloadable
        # tensors), so the JSONL still reflects what autograd saved.
        self._write_tensor_event(
            "saved_tensor_pack",
            tensor,
            module_path=module_path,
            layer=layer,
            event_id=event_id,
        )
        orig_device: torch.device | None = None
        if self.offload_to_cpu and tensor.is_cuda:
            orig_device = tensor.device
            if self.offload_pin:
                # Same pattern as torch.autograd.graph.save_on_cpu with
                # pin_memory=True: allocate a pinned-host buffer of the
                # exact shape/dtype, then enqueue a D2H copy on the
                # current stream. Stream-ordering keeps the source GPU
                # storage valid until the copy completes; the caching
                # allocator can then reclaim it.
                stored = torch.empty(
                    tensor.size(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    pin_memory=True,
                )
                stored.copy_(tensor)
            else:
                stored = tensor.detach().cpu()
        else:
            stored = tensor.detach()
        return _SavedTensorHandle(
            tensor=stored,
            save_event_id=event_id,
            module_path=module_path,
            layer=layer,
            orig_device=orig_device,
        )

    def _unpack_hook(self, handle: _SavedTensorHandle) -> torch.Tensor:
        self._write_tensor_event(
            "saved_tensor_unpack",
            handle.tensor,
            module_path=handle.module_path,
            layer=handle.layer,
            save_event_id=handle.save_event_id,
        )
        if handle.orig_device is not None:
            # H2D copy back to the original CUDA device. ``non_blocking``
            # is only honored when the source is pinned, which matches
            # ``offload_pin``.
            return handle.tensor.to(
                handle.orig_device,
                non_blocking=self.offload_pin,
            )
        return handle.tensor

    def _register_module_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not name or not self._should_trace_module(name):
                continue
            self._handles.append(module.register_forward_pre_hook(
                self._make_pre_hook(name),
            ))
            # ``always_call=True`` so the post hook still runs when the inner
            # forward raises — notably the ``_StopRecomputationError`` that
            # NO_REENTRANT activation checkpointing uses to abort the recompute
            # once it has captured all tensors needed for backward. Without
            # this, the last submodule of every checkpointed block (e.g.
            # ``mlp_residual``) gets a pre hook with no matching post hook.
            self._handles.append(module.register_forward_hook(
                self._make_forward_hook(name),
                always_call=True,
            ))

    def _should_trace_module(self, name: str) -> bool:
        normalized = _normalize_module_path(name)
        if self.module_detail == "all":
            return True
        if normalized in {
            "patch_embedding",
            "condition_embedder",
            "norm_out",
            "proj_out",
        }:
            return True
        if re.fullmatch(r"blocks\.\d+", normalized):
            return True
        if self.module_detail == "subblock":
            return re.fullmatch(
                r"blocks\.\d+\.(attn1|attn2|ffn|norm1|norm_q|norm_k|"
                r"to_q|to_k|to_v|to_out|self_attn_residual_norm|"
                r"cross_attn_residual_norm|mlp_residual)",
                normalized,
            ) is not None
        return False

    def _make_pre_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            del module, inputs
            self._module_stack.append(name)
            self._write_event({
                "event": "module_forward_pre",
                "module_path": name,
                "layer": self._extract_layer(name),
            })

        return hook

    def _make_forward_hook(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            for output_path, tensor in self._iter_tensors(output):
                self._write_tensor_event(
                    "module_forward_output",
                    tensor,
                    module_path=name,
                    layer=self._extract_layer(name),
                    output_path=output_path,
                )
                if tensor.requires_grad:
                    tensor.register_hook(
                        self._make_grad_hook(name, output_path),
                    )
            if self._module_stack and self._module_stack[-1] == name:
                self._module_stack.pop()
            else:
                self._module_stack.clear()
            self._write_event({
                "event": "module_forward_post",
                "module_path": name,
                "layer": self._extract_layer(name),
            })
            return output

        return hook

    def _make_grad_hook(self, name: str, output_path: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            self._write_tensor_event(
                "module_output_grad",
                grad,
                module_path=name,
                layer=self._extract_layer(name),
                output_path=output_path,
            )
            return grad

        return hook

    def _current_module(self) -> str | None:
        return self._module_stack[-1] if self._module_stack else None

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def _write_tensor_event(
        self,
        event: str,
        tensor: torch.Tensor,
        *,
        module_path: str | None,
        layer: int | None,
        event_id: int | None = None,
        save_event_id: int | None = None,
        output_path: str | None = None,
    ) -> None:
        payload = self._tensor_metadata(tensor)
        if event_id is not None:
            payload["event_id"] = event_id
        if save_event_id is not None:
            payload["save_event_id"] = save_event_id
        if output_path is not None:
            payload["output_path"] = output_path
        self._write_event({
            "event": event,
            "module_path": module_path,
            "layer": layer,
            "tensor": payload,
        })

    def _write_event(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        base = {
            "time_s": time.perf_counter(),
            "step": self.step,
            "rank": self.rank,
            "phase": self.phase,
        }
        base.update(record)
        if self.with_memory:
            memory = self._memory_sample()
            if memory:
                base["cuda_memory"] = memory
        self._file.write(json.dumps(self._jsonable(base), sort_keys=True) + "\n")
        self._events_since_flush += 1
        if self.flush_every > 0 and self._events_since_flush >= self.flush_every:
            self._file.flush()
            self._events_since_flush = 0

    def _tensor_metadata(self, tensor: torch.Tensor) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key, getter in {
            "shape": lambda: list(tensor.shape),
            "dtype": lambda: str(tensor.dtype),
            "device": lambda: str(tensor.device),
            "requires_grad": lambda: bool(tensor.requires_grad),
            "is_leaf": lambda: bool(tensor.is_leaf),
            "numel": lambda: int(tensor.numel()),
            "element_size": lambda: int(tensor.element_size()),
            "bytes": lambda: int(tensor.numel() * tensor.element_size()),
            "data_ptr": lambda: int(tensor.data_ptr()),
            "storage_ptr": lambda: int(tensor.untyped_storage().data_ptr()),
            "stride": lambda: list(tensor.stride()),
            "grad_fn": lambda: (
                type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None
            ),
        }.items():
            try:
                metadata[key] = getter()
            except Exception as exc:  # pragma: no cover - defensive metadata
                metadata[key] = f"<unavailable: {type(exc).__name__}>"
        return metadata

    def _memory_sample(self) -> dict[str, int] | None:
        if not torch.cuda.is_available():
            return None
        try:
            device = torch.cuda.current_device()
            return {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "max_allocated": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
        except Exception:
            return None

    def _extract_layer(self, module_path: str | None) -> int | None:
        if module_path is None:
            return None
        match = _BLOCK_RE.search(_normalize_module_path(module_path))
        return int(match.group(1)) if match else None

    def _iter_tensors(self, value: Any, prefix: str = "output"):
        if isinstance(value, torch.Tensor):
            yield prefix, value
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                yield from self._iter_tensors(item, f"{prefix}.{i}")
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from self._iter_tensors(item, f"{prefix}.{key}")

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, torch.Size):
            return list(value)
        if isinstance(value, torch.Tensor):
            return self._tensor_metadata(value)
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        return str(value)


def maybe_activation_trace(
    model: torch.nn.Module,
    *,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> ActivationTrace | _NoopActivationTrace:
    if not _trace_enabled():
        return _NoopActivationTrace()
    rank = _rank()
    if not _rank_enabled(rank):
        return _NoopActivationTrace()
    if not _step_enabled(step):
        return _NoopActivationTrace()
    trace_dir = os.environ.get(
        "FASTVIDEO_ACTIVATION_TRACE_DIR",
        "/tmp/fastvideo_activation_traces",
    )
    return ActivationTrace(
        model,
        trace_dir=trace_dir,
        step=step,
        rank=rank,
        metadata=metadata,
    )


def _trace_enabled() -> bool:
    return (
        os.environ.get("FASTVIDEO_ACTIVATION_TRACE", "").strip().lower()
        in _TRUE_VALUES
        or bool(os.environ.get("FASTVIDEO_ACTIVATION_TRACE_DIR"))
    )


def _step_enabled(step: int) -> bool:
    max_steps = int(os.environ.get("FASTVIDEO_ACTIVATION_TRACE_MAX_STEPS", "1"))
    return int(step) <= max_steps


def _rank_enabled(rank: int) -> bool:
    raw = os.environ.get("FASTVIDEO_ACTIVATION_TRACE_RANKS", "0").strip().lower()
    if raw == "all":
        return True
    allowed = {part.strip() for part in raw.split(",") if part.strip()}
    return str(rank) in allowed


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", "0"))
