# SPDX-License-Identifier: Apache-2.0
"""Per-allocation CUDA memory event tracking.

This module installs PyTorch's CUDA caching-allocator history callback so that
**every** allocation event is recorded as it happens, on a single rank ("one
card"). Two event classes are captured and later distinguished by the analysis
tools:

* **True driver allocations** — ``segment_alloc`` / ``segment_free`` events,
  i.e. each ``cudaMalloc`` / ``cudaFree``. (With
  ``PYTORCH_ALLOC_CONF=expandable_segments:True`` these become ``segment_map`` /
  ``segment_unmap`` — physical pages mapped into a reserved virtual range.)
* **Reserved-space pseudo-allocations** — ``alloc`` / ``free_requested`` /
  ``free_completed`` events, i.e. each tensor block carved out of (or returned
  to) an already-reserved segment.

Two recording modes (selected by ``FASTVIDEO_MEM_SNAPSHOT_WARMUP``):

* **Bracketed (default):** recording is enabled at the start of each selected
  step and dumped at its end — one pickle per step. The absolute memory level
  before the step is *not* in the event stream, so the analysis tools must
  reconstruct a baseline (imprecise across allocator backends).
* **Warmup/continuous:** recording is enabled at *process entry* (before model
  load) and runs continuously through the last selected step, then dumps once.
  Because it starts from zero, every byte of reserved/allocated memory is in the
  event stream and the absolute curve is **exact** — no baseline reconstruction.

Each event carries a *size* and a captured *stack trace*. :func:`mem_region`
adds a coarse phase label (prepare_batch / dit_forward / loss / backward /
optimizer_step / zero_grad) via ``record_function`` (captured into the
snapshot's ``external_annotations``) and records the ``memory_stats`` delta for
the region to a sidecar JSON. The recorder also brackets the captured window
with ``warmup`` / ``step{N}`` annotations.

Configuration is via ``FASTVIDEO_MEM_SNAPSHOT_*`` environment variables; see
``fastvideo/envs.py`` and ``scripts/memory/README.md``.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import torch

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Phase labels are pushed as ``record_function`` ranges with this prefix so the
# post-processor can recognise them among the captured stack frames.
REGION_PREFIX = "fvmem::"

_GLOBAL_RECORDER: "MemoryHistoryRecorder | None" = None


def _parse_steps(spec: str) -> tuple[set[int], bool]:
    """Parse ``FASTVIDEO_MEM_SNAPSHOT_STEPS`` into (steps, all_steps)."""
    spec = (spec or "").strip().lower()
    if not spec:
        return set(), False
    if spec == "all":
        return set(), True
    steps: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        steps.add(int(tok))
    return steps, False


def _mem_now(device: int | None) -> dict[str, int]:
    """Snapshot the allocator's current/peak counters (authoritative absolute)."""
    if device is None or not torch.cuda.is_available():
        return {}
    s = torch.cuda.memory_stats(device)
    return {
        "allocated_current": int(s.get("allocated_bytes.all.current", 0)),
        "reserved_current": int(s.get("reserved_bytes.all.current", 0)),
        "allocated_peak": int(s.get("allocated_bytes.all.peak", 0)),
        "reserved_peak": int(s.get("reserved_bytes.all.peak", 0)),
    }


class MemoryHistoryRecorder:
    """Controls allocator history recording for one rank.

    Off-rank instances are inert: every method is a no-op, so the other GPUs
    are never touched.
    """

    def __init__(
        self,
        *,
        out_dir: str,
        steps: set[int],
        all_steps: bool,
        record_rank: int,
        global_rank: int,
        max_entries: int,
        stacks: str,
        warmup: bool,
    ) -> None:
        self.out_dir = out_dir
        self.steps = steps
        self.all_steps = all_steps
        self.record_rank = record_rank
        self.global_rank = global_rank
        self.max_entries = max_entries
        self.stacks = stacks if stacks in ("all", "python") else "all"
        self.warmup = warmup
        # In warmup/continuous mode we dump once, after the last selected step.
        self.last_step = max(steps) if steps else None

        self.enabled = global_rank == record_rank
        self.active = False
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._region_deltas: list[dict[str, Any]] = []
        self._open_regions: dict[str, Any] = {}
        self._start_stats: dict[str, int] = {}
        # Label of the captured window, used for the output filename.
        self._window_label: str | None = None
        self._oom_observer_installed = False

        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
            logger.info(
                "MEM_SNAPSHOT: enabled on rank %s (device=%s) mode=%s steps=%s "
                "max_entries=%s stacks=%s out_dir=%s",
                self.global_rank, self.device,
                "warmup+continuous" if self.warmup else "bracketed",
                "all" if self.all_steps else sorted(self.steps),
                self.max_entries, self.stacks, self.out_dir,
            )

    # -- low-level recording --------------------------------------------

    def _enable_recording(self) -> None:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks=self.stacks,
            max_entries=self.max_entries)
        self.active = True
        self._start_stats = _mem_now(self.device)
        self._maybe_install_oom_observer()

    def _maybe_install_oom_observer(self) -> None:
        """Dump the in-flight recording if the allocator hits CUDA OOM.

        Without this, a run that OOMs before the scheduled dump (e.g. a baseline
        that doesn't fit) loses its entire recorded history. The observer fires
        the moment an allocation can't be satisfied — just before the
        ``OutOfMemoryError`` propagates — so we still capture a timeline up to
        (and including) the fatal allocation. Dumping is host-side serialization
        and needs no GPU memory, so it is safe to do here.
        """
        if self._oom_observer_installed or not self.enabled:
            return

        def _oom_observer(device: int, alloc: int, device_alloc: int,
                          device_free: int) -> None:
            if not self.active:
                return
            self._window_label = (self._window_label or "trace") + "_oom"
            logger.warning(
                "MEM_SNAPSHOT: CUDA OOM on device %s (tried %.2f GiB, "
                "%.2f GiB free); dumping in-flight snapshot to %s.",
                device, alloc / 2**30, device_free / 2**30, self._window_label)
            self.dump()

        try:
            torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
            self._oom_observer_installed = True
        except Exception as exc:  # noqa: BLE001 - best-effort instrumentation
            logger.warning("MEM_SNAPSHOT: could not attach OOM observer: %s",
                           exc)

    def _enter_region(self, name: str) -> None:
        if not self.active:
            return
        ctx = torch.profiler.record_function(f"{REGION_PREFIX}{name}")
        ctx.__enter__()
        self._open_regions[name] = ctx

    def _exit_region(self, name: str) -> None:
        ctx = self._open_regions.pop(name, None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # -- gating ---------------------------------------------------------

    def should_record(self, step: int) -> bool:
        if not self.enabled:
            return False
        return self.all_steps or (step in self.steps)

    # -- lifecycle: warmup/continuous mode ------------------------------

    def begin_warmup(self) -> None:
        """Start one continuous recording window at process entry."""
        if not self.enabled or not self.warmup or self.active:
            return
        self._window_label = (
            f"warmup_step{min(self.steps)}-{max(self.steps)}"
            if self.steps else "warmup")
        self._enable_recording()
        self._enter_region("warmup")
        logger.info("MEM_SNAPSHOT: continuous recording started at warmup "
                    "(start reserved=%.2f GiB)",
                    self._start_stats.get("reserved_current", 0) / 2**30)

    # -- lifecycle: per-step hooks (used by the trainer) ----------------

    def on_step_begin(self, step: int) -> None:
        if not self.enabled:
            return
        if self.warmup:
            if not self.active:
                return
            # First recorded step: close the warmup region.
            self._exit_region("warmup")
            if self.last_step is None or step <= self.last_step:
                self._enter_region(f"step{step}")
        else:
            # Bracketed mode: a fresh ring buffer per selected step.
            if self.should_record(step):
                self._region_deltas = []
                self._window_label = f"step{step}"
                self._enable_recording()
                logger.info("MEM_SNAPSHOT: recording started for step %s", step)

    def on_step_end(self, step: int) -> None:
        if not self.enabled or not self.active:
            return
        if self.warmup:
            self._exit_region(f"step{step}")
            if self.last_step is not None and step >= self.last_step:
                self.dump()
        else:
            if self.should_record(step):
                self.dump()

    def finalize(self) -> None:
        """Dump if recording is still active when training ends.

        Covers warmup/continuous mode where training stopped before the last
        selected step (e.g. ``max_train_steps`` below the largest requested
        step) or used ``all`` (no explicit last step): the per-step hook never
        fired its dump, yet the recording — and its allocator overhead — is
        still live. No-op in bracketed mode (each selected step already dumped
        and disabled recording) and on the OOM path (the observer already
        dumped, leaving ``active`` False)."""
        if not self.enabled or not self.active:
            return
        logger.info("MEM_SNAPSHOT: finalizing — dumping active recording at "
                    "training end.")
        self.dump()

    # -- dump -----------------------------------------------------------

    def dump(self) -> None:
        """Serialize the recorded history and stop recording."""
        if not self.enabled or not self.active:
            return
        # Close any still-open regions so annotations are balanced.
        for name in list(self._open_regions):
            self._exit_region(name)
        label = self._window_label or "trace"
        base = os.path.join(self.out_dir, f"{label}_rank{self.global_rank}")
        pickle_path = f"{base}.pickle"
        end_stats = _mem_now(self.device)
        try:
            torch.cuda.memory._dump_snapshot(pickle_path)
            logger.info("MEM_SNAPSHOT: dumped %s", pickle_path)
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            self.active = False

        sidecar = {
            "label": label,
            "step_window": sorted(self.steps),
            "rank": self.global_rank,
            "device": self.device,
            "mode": "warmup+continuous" if self.warmup else "bracketed",
            "snapshot": os.path.basename(pickle_path),
            "region_prefix": REGION_PREFIX,
            # Authoritative absolute counters at record start / dump. In warmup
            # mode start_* ~= 0, so the event-stream replay is exact; either way
            # the analysis tools can anchor the absolute curve precisely.
            "memory_stats_start": self._start_stats,
            "memory_stats_end": end_stats,
            "regions": self._region_deltas,
        }
        regions_path = f"{base}.regions.json"
        try:
            with open(regions_path, "w") as fh:
                json.dump(sidecar, fh, indent=2)
            logger.info("MEM_SNAPSHOT: wrote phase summary %s", regions_path)
        except Exception as exc:  # noqa: BLE001 - best-effort sidecar
            logger.warning("MEM_SNAPSHOT: failed to write %s: %s", regions_path, exc)

    # -- region accounting (for mem_region) -----------------------------

    def record_region_delta(self, name: str, before: dict[str, Any],
                            after: dict[str, Any]) -> None:
        def g(stats: dict[str, Any], key: str) -> int:
            return int(stats.get(key, 0))

        self._region_deltas.append({
            "region": name,
            "reserved_bytes_delta": g(after, "reserved_bytes.all.current") - g(before, "reserved_bytes.all.current"),
            "allocated_bytes_delta": g(after, "allocated_bytes.all.current") - g(before, "allocated_bytes.all.current"),
            "segment_alloc_count": g(after, "segment.all.allocated") - g(before, "segment.all.allocated"),
            "segment_free_count": g(after, "segment.all.freed") - g(before, "segment.all.freed"),
            "block_alloc_count": g(after, "allocation.all.allocated") - g(before, "allocation.all.allocated"),
            "block_free_count": g(after, "allocation.all.freed") - g(before, "allocation.all.freed"),
            "num_alloc_retries": g(after, "num_alloc_retries") - g(before, "num_alloc_retries"),
            "allocated_bytes_peak_after": g(after, "allocated_bytes.all.peak"),
            "reserved_bytes_peak_after": g(after, "reserved_bytes.all.peak"),
        })


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


def configure_from_env(global_rank: int) -> MemoryHistoryRecorder | None:
    """Build and register the global recorder from env, or return ``None``.

    Idempotent: if a recorder is already registered for this rank it is reused
    (so the early warmup call in the entrypoint and the later call in the
    trainer share one recorder). Returns ``None`` when ``FASTVIDEO_MEM_SNAPSHOT_DIR``
    is unset or no steps are requested.
    """
    global _GLOBAL_RECORDER
    if _GLOBAL_RECORDER is not None and _GLOBAL_RECORDER.global_rank == global_rank:
        return _GLOBAL_RECORDER

    out_dir = envs.FASTVIDEO_MEM_SNAPSHOT_DIR
    if not out_dir:
        return None

    steps, all_steps = _parse_steps(envs.FASTVIDEO_MEM_SNAPSHOT_STEPS)
    if not steps and not all_steps:
        logger.warning("MEM_SNAPSHOT: FASTVIDEO_MEM_SNAPSHOT_DIR is set but "
                       "FASTVIDEO_MEM_SNAPSHOT_STEPS is empty; nothing recorded.")
        return None
    # warmup/continuous mode dumps either at the last explicitly-selected step
    # or, for `all` (which has no explicit last step), at training end via
    # finalize(); both are valid, so warmup works with `steps` or `all`. (The
    # both-empty case is already rejected above.)
    warmup = bool(envs.FASTVIDEO_MEM_SNAPSHOT_WARMUP)

    recorder = MemoryHistoryRecorder(
        out_dir=out_dir, steps=steps, all_steps=all_steps,
        record_rank=int(envs.FASTVIDEO_MEM_SNAPSHOT_RANK),
        global_rank=int(global_rank),
        max_entries=int(envs.FASTVIDEO_MEM_SNAPSHOT_MAX_ENTRIES),
        stacks=str(envs.FASTVIDEO_MEM_SNAPSHOT_STACKS),
        warmup=warmup,
    )
    _GLOBAL_RECORDER = recorder
    return recorder


def begin_warmup_recording(global_rank: int) -> MemoryHistoryRecorder | None:
    """Configure from env and, in warmup mode, start recording immediately.

    Call this as early as possible in the entrypoint (after distributed init,
    before model load) so warmup allocations are captured.
    """
    recorder = configure_from_env(global_rank)
    if recorder is not None:
        recorder.begin_warmup()
    return recorder


def get_recorder() -> MemoryHistoryRecorder | None:
    return _GLOBAL_RECORDER


@contextlib.contextmanager
def mem_region(name: str):
    """Label a phase of the step for the active recording window.

    Pushes a ``record_function`` range (captured into the snapshot's event
    stack frames / ``external_annotations`` as ``fvmem::<name>``) and records
    the ``memory_stats`` delta for the region. A no-op unless the global
    recorder is actively recording.
    """
    recorder = _GLOBAL_RECORDER
    if recorder is None or not recorder.active:
        yield
        return

    before = torch.cuda.memory_stats(recorder.device)
    with torch.profiler.record_function(f"{REGION_PREFIX}{name}"):
        try:
            yield
        finally:
            after = torch.cuda.memory_stats(recorder.device)
            recorder.record_region_delta(name, before, after)
