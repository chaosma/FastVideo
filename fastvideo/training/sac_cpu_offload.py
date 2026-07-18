# SPDX-License-Identifier: Apache-2.0
"""Selective activation checkpointing (SAC) with CPU offload of saved outputs.

PyTorch's eager SAC (``torch.utils.checkpoint.create_selective_checkpoint_contexts``)
keeps every ``MUST_SAVE`` operator output resident on GPU in its own cache; the
public ``CheckpointPolicy.MUST_CPU_OFFLOAD`` enum is not wired into the eager
dispatch modes, so returning it just falls through to recomputation. Those
save-cached tensors also bypass ``saved_tensors_hooks``, so the generic
``FASTVIDEO_SAVE_ON_CPU`` offload path cannot see them either.

This module mirrors the two small ``TorchDispatchMode``s that implement eager
SAC (caching mode on forward, replay mode on recompute), but for offload
policies it streams the selected op output to pinned host memory during forward
and streams it back to GPU during recompute -- while still skipping the
expensive attention forward. The transport layer (dedicated copy stream, stream
waits, ``record_stream``, bounded pinned pool) follows the same mechanics used
by ``fastvideo/train/utils/activation_trace.py`` and the layerwise weight
offload.

Scope (v1): eager execution, reactive reload (no prefetch). See
``.agents/exploration/sac-cpu-offload-design.md`` for the full design and the
memory/PCIe trade-off analysis.
"""
from __future__ import annotations

import itertools
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode
# These are private helpers of torch.utils.checkpoint. They are imported rather
# than reimplemented so the offload cache stays byte-for-byte compatible with
# the version-check / view-detach / ignored-op semantics of eager SAC. If a
# future torch renames them, this import fails loudly instead of silently
# diverging from upstream behavior.
from torch.utils.checkpoint import (  # type: ignore[attr-defined]
    SAC_IGNORED_OPS, CheckpointPolicy, SelectiveCheckpointContext,
    _is_compiling, _maybe_detach, _policy_from_bool, _VersionWrapper)

from fastvideo.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}

# Policies whose output is materialized during forward and reused during
# recompute (as opposed to recomputed). PREFER_* still means "do XXX" -- it is a
# soft hint the AC framework may override, whereas MUST_* is a hard requirement;
# neither is the opposite of XXX. Our policy_fn only ever returns MUST_* (or
# PREFER_RECOMPUTE), but the PREFER_* variants are matched too for parity with
# eager SAC's enum semantics.
_SAVE_POLICIES = (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)
_OFFLOAD_POLICIES = (CheckpointPolicy.MUST_CPU_OFFLOAD,
                     CheckpointPolicy.PREFER_CPU_OFFLOAD)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _min_offload_bytes() -> int:
    """Tensors smaller than this stay on GPU (e.g. attention LSE/metadata).

    Offloading a few-MB tensor costs a pinned alloc + two PCIe hops that are not
    worth it; only the large attention output ``O`` should cross the bus.
    """
    mb = float(os.environ.get("FASTVIDEO_SAC_OFFLOAD_MIN_MB", "1") or "1")
    return int(mb * 1024 * 1024)


def _pinned_cap_bytes() -> int | None:
    gb = os.environ.get("FASTVIDEO_SAC_OFFLOAD_PINNED_GB")
    if not gb:
        return None
    return int(float(gb) * 1024**3)


def _strict_version() -> bool:
    return _env_flag("FASTVIDEO_SAC_OFFLOAD_STRICT_VERSION")


def _log_steps() -> int:
    """How many block contexts to emit a detailed offload/reload log for.

    Named after the design's ``FASTVIDEO_SAC_OFFLOAD_LOG_STEPS``. The checkpoint
    ``context_fn`` API gives no training-step handle, so this counts the first N
    *block contexts* whose recompute completes -- enough to confirm the per-block
    signal (one self-attn output cached and one reloaded) cheaply before a long
    run, without per-step plumbing.
    """
    return int(os.environ.get("FASTVIDEO_SAC_OFFLOAD_LOG_STEPS", "2") or "2")


# Budget of detailed per-context logs already emitted (module-global so it is
# shared across every block context in the run).
_LOG_BUDGET_USED = 0

# Emitted at most once: a covered block whose forward cached zero attention
# outputs, i.e. the active attention backend's op is not intercepted by the
# offload policy and self_attn_cpu is silently degrading to full recompute.
_WARNED_NO_OFFLOAD = False


def _warn_no_offload_once() -> None:
    global _WARNED_NO_OFFLOAD
    if _WARNED_NO_OFFLOAD:
        return
    _WARNED_NO_OFFLOAD = True
    logger.warning(
        "SAC CPU offload: a checkpointed block cached 0 attention outputs -- "
        "the attention op used by the active backend is not intercepted by the "
        "offload policy, so self_attn_cpu is silently degrading to full "
        "recompute (no memory offloaded, no compute saved). Only aten SDPA "
        "(TORCH_SDPA) and FlashAttention-3 are intercepted; FlashAttention-4 / "
        "cute and the standalone flash_attn (FA2) op are not. Pick an "
        "intercepted backend or extend the policy before relying on this.")


# ---------------------------------------------------------------------------
# Transport layer: per-device copy stream + bounded pinned host pool.
# These are long-lived process singletons shared across every checkpointed
# block, so pinned buffers are allocated once (during warmup) and reused every
# step instead of re-registering ~51 GB/rank of host memory each forward.
# ---------------------------------------------------------------------------

_COPY_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _COPY_STREAMS.get(device)
    if stream is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream()
        _COPY_STREAMS[device] = stream
    return stream


class _PinnedPool:
    """Free-list of pinned host buffers keyed by (dtype, shape, stride).

    Self-attn outputs share one shape across all covered blocks and all steps,
    so a handful of distinct keys covers the whole run. Buffers are acquired on
    the forward D2H and released back on the backward H2D reload; reuse across
    steps is safe because every copy for a device goes through that device's
    single copy stream, whose FIFO ordering serializes each buffer's H2D read
    ahead of any later D2H write.
    """

    def __init__(self, device: torch.device, cap_bytes: int | None) -> None:
        self.device = device
        self.cap_bytes = cap_bytes
        self.allocated_bytes = 0
        self._free: dict[Any, list[torch.Tensor]] = defaultdict(list)

    @staticmethod
    def _key(t: torch.Tensor) -> Any:
        return (t.dtype, tuple(t.shape), tuple(t.stride()))

    def acquire(self, template: torch.Tensor) -> torch.Tensor | None:
        """Return a pinned host buffer matching ``template``'s layout, or None.

        None means the caller should fall back to keeping the tensor on GPU
        (pool over its byte cap, or the layout cannot be pinned).
        """
        key = self._key(template)
        free = self._free.get(key)
        if free:
            return free.pop()
        nbytes = template.numel() * template.element_size()
        if (self.cap_bytes is not None
                and self.allocated_bytes + nbytes > self.cap_bytes):
            return None
        try:
            host = torch.empty_strided(template.size(),
                                       template.stride(),
                                       dtype=template.dtype,
                                       device="cpu",
                                       pin_memory=True)
        except (RuntimeError, NotImplementedError):
            return None
        self.allocated_bytes += nbytes
        return host

    def release(self, host: torch.Tensor) -> None:
        self._free[self._key(host)].append(host)


_PINNED_POOLS: dict[torch.device, _PinnedPool] = {}


def _pinned_pool(device: torch.device) -> _PinnedPool:
    pool = _PINNED_POOLS.get(device)
    if pool is None:
        cap = _pinned_cap_bytes()
        pool = _PinnedPool(device, cap)
        _PINNED_POOLS[device] = pool
        logger.info(
            "SAC CPU offload: created pinned pool on %s (cap=%s)", device,
            f"{cap / 1024**3:.1f} GB" if cap is not None else "uncapped")
    return pool


# ---------------------------------------------------------------------------
# Diagnostics. Each distinct offloaded / fallback shape is logged once so the
# large self-attn output can be confirmed cheaply without per-step plumbing.
# ---------------------------------------------------------------------------


class _OffloadStats:

    def __init__(self) -> None:
        self._offloaded_shapes: set[tuple] = set()
        self._fallback_shapes: set[tuple] = set()

    def record_offload(self, t: torch.Tensor, device: torch.device) -> None:
        key = (tuple(t.shape), str(t.dtype))
        if key not in self._offloaded_shapes:
            self._offloaded_shapes.add(key)
            pool = _pinned_pool(device)
            logger.info(
                "SAC CPU offload: offloading shape=%s dtype=%s (%.3f GB each); "
                "pinned pool now %.2f GB on %s", tuple(t.shape), t.dtype,
                t.numel() * t.element_size() / 1e9,
                pool.allocated_bytes / 1e9, device)

    def record_fallback(self, t: torch.Tensor, reason: str) -> None:
        key = (tuple(t.shape), str(t.dtype), reason)
        if key not in self._fallback_shapes:
            self._fallback_shapes.add(key)
            logger.warning(
                "SAC CPU offload: keeping shape=%s dtype=%s on GPU (%s)",
                tuple(t.shape), t.dtype, reason)


_STATS = _OffloadStats()


# ---------------------------------------------------------------------------
# Offloaded-tensor handle.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OffloadedTensor:
    """Handle to a self-attn output snapshotted into pinned host memory."""
    host: torch.Tensor | None
    pool: _PinnedPool
    device: torch.device
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    # Populated only under FASTVIDEO_SAC_OFFLOAD_STRICT_VERSION for a debug
    # mutation check; keeps a GPU reference alive, so it is not for long runs.
    strict_ref: torch.Tensor | None = None
    version: int | None = None

    def reload(self) -> torch.Tensor:
        assert self.host is not None, "OffloadedTensor already reloaded"
        if (self.strict_ref is not None and self.version is not None
                and self.strict_ref._version != self.version):
            raise RuntimeError(
                "Tensor offloaded during SAC CPU offload was mutated "
                "between forward and recompute")
        stream = _copy_stream(self.device)
        with torch.cuda.stream(stream):
            device_tensor = torch.empty_strided(self.shape,
                                                 self.stride,
                                                 dtype=self.dtype,
                                                 device=self.device)
            device_tensor.copy_(self.host, non_blocking=True)
        current = torch.cuda.current_stream(self.device)
        # The recompute graph (default stream) must wait for the H2D to land.
        current.wait_stream(stream)
        device_tensor.record_stream(current)
        # Safe to hand the host buffer back now: the H2D read above is already
        # enqueued on ``stream``, so any future D2H write into this same buffer
        # (next step's forward) is ordered strictly after it on that FIFO stream.
        self.pool.release(self.host)
        self.host = None
        self.strict_ref = None
        return device_tensor


class _OffloadCache:
    """Per-block-context store plus diagnostics counters.

    Replaces PyTorch SAC's raw ``defaultdict(list)`` storage so this module owns
    the cache end-to-end (PyTorch's own SAC cache cannot be offloaded). Counters
    let a caller/test confirm the expected signal: one self-attn op cached on
    forward and reloaded on recompute per covered block.
    """
    __slots__ = ("entries", "n_offload_ops", "n_reloaded_ops",
                 "n_offloaded_leaves", "n_reloaded_leaves", "offloaded_bytes",
                 "fallback_leaves", "fallback_bytes", "op_names")

    def __init__(self) -> None:
        self.entries: dict[Any, list] = defaultdict(list)
        self.n_offload_ops = 0  # ops stored under an offload policy (forward)
        self.n_reloaded_ops = 0  # ops replayed under an offload policy (bwd)
        self.n_offloaded_leaves = 0  # tensor leaves physically moved to host
        self.n_reloaded_leaves = 0  # tensor leaves streamed back to GPU
        self.offloaded_bytes = 0
        # Selected leaves that stayed GPU-resident because the pinned pool was
        # capped or the layout was unpinnable. A high fallback count means the
        # GPU-resident SAC floor is still present despite the feature being on.
        self.fallback_leaves = 0
        self.fallback_bytes = 0
        self.op_names: set[str] = set()


def _pack_leaf(x: Any, *, offload: bool, alias_info: bool,
               cache: _OffloadCache) -> Any:
    """Forward transform for one op-output leaf.

    Non-tensors pass through. Tensors either get offloaded to pinned host
    (returning an ``OffloadedTensor``) or kept on GPU wrapped in torch's
    ``_VersionWrapper`` -- identical to eager SAC's save path, so small
    metadata tensors and fallbacks retain the mutation check.
    """
    if not isinstance(x, torch.Tensor):
        return x
    detached = _maybe_detach(x, alias_info)
    if not offload or not x.is_cuda:
        return _VersionWrapper(detached)
    nbytes = x.numel() * x.element_size()
    if nbytes < _min_offload_bytes():
        return _VersionWrapper(detached)
    device = x.device
    host = _pinned_pool(device).acquire(x)
    if host is None:
        _STATS.record_fallback(x, "pinned pool cap or unpinnable layout")
        cache.fallback_leaves += 1
        cache.fallback_bytes += nbytes
        return _VersionWrapper(detached)
    stream = _copy_stream(device)
    # D2H on the side stream so forward compute is not blocked; wait for the
    # producer first, and keep the source GPU buffer alive until the copy runs.
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        host.copy_(x, non_blocking=True)
    x.record_stream(stream)
    _STATS.record_offload(x, device)
    cache.n_offloaded_leaves += 1
    cache.offloaded_bytes += nbytes
    handle = OffloadedTensor(host=host,
                             pool=_pinned_pool(device),
                             device=device,
                             shape=x.shape,
                             stride=x.stride(),
                             dtype=x.dtype)
    if _strict_version():
        handle.strict_ref = x
        handle.version = x._version
    return handle


def _unpack_leaf(x: Any, *, allow_mutation: bool, cache: _OffloadCache) -> Any:
    if isinstance(x, OffloadedTensor):
        cache.n_reloaded_leaves += 1
        return x.reload()
    if isinstance(x, _VersionWrapper):
        return x.get_val(allow_mutation)
    return x


# ---------------------------------------------------------------------------
# Dispatch modes (mirror torch.utils.checkpoint's SAC modes, plus offload).
# ---------------------------------------------------------------------------


def _any_ret_has_alias_info(func) -> bool:
    if isinstance(func, torch._ops.HigherOrderOperator):
        return False
    return any(ret.alias_info is not None for ret in func._schema.returns)


class _CpuOffloadCachingMode(TorchDispatchMode):
    """Forward pass: run each op, then cache selected outputs (GPU or CPU)."""

    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __init__(self, policy_fn, cache: _OffloadCache, ac_graph_id=None):
        self.policy_fn = policy_fn
        self.cache = cache
        self.ac_graph_id = ac_graph_id

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        is_compiling = _is_compiling(func, args, kwargs)
        if is_compiling:
            fx_traceback.current_meta["ac_graph_id"] = self.ac_graph_id
            fx_traceback.current_meta[
                "recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        proxy_mode = None
        graph_len_before = 0
        if is_compiling:
            from torch.fx.experimental.proxy_tensor import get_proxy_mode
            proxy_mode = get_proxy_mode()
            if proxy_mode is not None:
                graph_len_before = len(proxy_mode.tracer.graph.nodes)

        out = func(*args, **kwargs)
        alias_info = _any_ret_has_alias_info(func)

        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=False, op_output=out),
            func, *args, **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)

        if is_compiling and proxy_mode is not None:
            graph = proxy_mode.tracer.graph
            num_new = len(graph.nodes) - graph_len_before
            for node in itertools.islice(reversed(graph.nodes), num_new):
                node.meta["recompute"] = policy

        # Under torch.compile we do not offload (the proxy trace must stay on
        # the GPU-resident save path); fall back to GPU-resident caching.
        offload = policy in _OFFLOAD_POLICIES and not is_compiling
        if policy in _SAVE_POLICIES or policy in _OFFLOAD_POLICIES or is_compiling:
            if offload:
                self.cache.n_offload_ops += 1
                self.cache.op_names.add(getattr(func, "_name", str(func)))
            self.cache.entries[func].append(
                tree_map(
                    lambda x: _pack_leaf(x,
                                         offload=offload,
                                         alias_info=alias_info,
                                         cache=self.cache), out))
        return out

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Forward finished for this block. If nothing was cached under an
        # offload policy, the selected attention op was never seen -- warn once
        # so a silent no-op cannot masquerade as an enabled feature.
        if exc_type is None and self.cache.n_offload_ops == 0:
            _warn_no_offload_once()
        return super().__exit__(exc_type, exc_val, exc_tb)


class _CpuOffloadReplayMode(TorchDispatchMode):
    """Recompute pass: replay cached outputs instead of re-running the op."""

    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __init__(self, policy_fn, cache: _OffloadCache,
                 allow_cache_entry_mutation: bool):
        self.policy_fn = policy_fn
        self.cache = cache
        self.allow_cache_entry_mutation = allow_cache_entry_mutation

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in SAC_IGNORED_OPS:
            return func(*args, **kwargs)

        kwargs = {} if kwargs is None else kwargs
        policy = self.policy_fn(
            SelectiveCheckpointContext(is_recompute=True), func, *args,
            **kwargs)
        if isinstance(policy, bool):
            policy = _policy_from_bool(policy)
        is_compiling = _is_compiling(func, args, kwargs)

        if (policy in _SAVE_POLICIES or policy in _OFFLOAD_POLICIES
                or is_compiling):
            entries = self.cache.entries.get(func)
            if entries is None:
                raise RuntimeError(
                    f"{func} encountered during backward, but not found in "
                    "storage")
            if len(entries) == 0:
                raise RuntimeError(
                    "Trying to backward an extra time. You are only allowed "
                    "to backward once on any region computed under selective "
                    "activation checkpoint.")
            if policy in _OFFLOAD_POLICIES and not is_compiling:
                self.cache.n_reloaded_ops += 1
            return tree_map(
                lambda x: _unpack_leaf(
                    x,
                    allow_mutation=self.allow_cache_entry_mutation,
                    cache=self.cache), entries.pop(0))
        return func(*args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Recompute finished for this block: emit a one-line offload/reload
        # summary for the first few block contexts of the run (see _log_steps).
        # Note recompute normally exits by raising checkpoint's internal
        # _StopRecomputationError (early stop), so we gate on a completed reload
        # (n_reloaded_ops), not on exc_type being None.
        global _LOG_BUDGET_USED
        if (self.cache.n_reloaded_ops > 0
                and _log_steps() > _LOG_BUDGET_USED):
            _LOG_BUDGET_USED += 1
            logger.info(
                "SAC CPU offload block: cached=%d op(s) reloaded=%d op(s); "
                "leaves offloaded=%d (%.3f GB to host) reloaded=%d; "
                "GPU-resident fallbacks=%d (%.3f GB); ops=%s",
                self.cache.n_offload_ops, self.cache.n_reloaded_ops,
                self.cache.n_offloaded_leaves, self.cache.offloaded_bytes / 1e9,
                self.cache.n_reloaded_leaves, self.cache.fallback_leaves,
                self.cache.fallback_bytes / 1e9, sorted(self.cache.op_names))
        return super().__exit__(exc_type, exc_val, exc_tb)


def create_cpu_offload_selective_checkpoint_contexts(
        policy_fn, allow_cache_entry_mutation: bool = False):
    """CPU-offload analogue of ``create_selective_checkpoint_contexts``.

    ``policy_fn`` returns a ``CheckpointPolicy`` per op. Return
    ``MUST_CPU_OFFLOAD`` / ``PREFER_CPU_OFFLOAD`` for ops whose output should be
    streamed to pinned host memory (and streamed back on recompute);
    ``MUST_SAVE`` / ``PREFER_SAVE`` keep the output GPU-resident;
    ``PREFER_RECOMPUTE`` recomputes as usual.
    """
    cache = _OffloadCache()
    return (
        _CpuOffloadCachingMode(policy_fn, cache),
        _CpuOffloadReplayMode(policy_fn, cache, allow_cache_entry_mutation),
    )
