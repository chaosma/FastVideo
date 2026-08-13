# SPDX-License-Identifier: Apache-2.0
"""Enable gate + per-device manager factory for the SSD spill tier.

SSD spill is an explicit capacity escape hatch, not an auto-enabled behavior. It
activates only when BOTH hold (see ``.agents/exploration/ssd-offload-design.md``):

* ``FASTVIDEO_SSD_SPILL=1``, and
* a valid local-NVMe ``FASTVIDEO_SSD_DIR`` that passes the direct-I/O probe
  (rented instances can present overlay/network mounts as "NVMe"), and a
  determinable host-RAM budget.

If any precondition fails the factory logs why and returns ``None`` -- callers
then behave exactly as if SSD spill were off. The result (including ``None``) is
cached per device so the probe runs once.
"""
from __future__ import annotations

import atexit
import os
import threading

import torch

from fastvideo.logger import init_logger
from fastvideo.training.ssd_budget import (local_world_size,
                                           resolve_host_ram_budget_bytes)
from fastvideo.training.ssd_manager import SsdSpillManager
from fastvideo.training.ssd_spill import direct_io_supported

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def ssd_spill_enabled() -> bool:
    return os.environ.get("FASTVIDEO_SSD_SPILL",
                          "").strip().lower() in _TRUE_VALUES


def ssd_dir() -> str | None:
    d = os.environ.get("FASTVIDEO_SSD_DIR")
    return d or None


def saved_tensors_ssd_enabled() -> bool:
    """Generic ``saved_tensors_hooks`` SSD spill gate.

    Opt-in and separate from SAC so the two can be A/B'd. Requires the core SSD
    gate (``FASTVIDEO_SSD_SPILL``); the dir/probe/budget are still validated
    lazily when the per-device manager is built. Off by default: the generic
    offload path keeps its existing CPU-pinned behavior unless this is set.
    """
    return (os.environ.get("FASTVIDEO_SAVED_TENSORS_SSD_SPILL",
                           "").strip().lower() in _TRUE_VALUES
            and ssd_spill_enabled())


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, warning and using ``default`` on a bad value.

    Matches ``resolve_budget_gb``'s tolerance so a malformed knob never crashes
    training.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("SSD spill: ignoring non-integer %s=%r; using %d", name,
                       raw, default)
        return default


def _prefetch_window() -> int:
    return _env_int("FASTVIDEO_SSD_SPILL_PREFETCH_WINDOW", 4)


def _io_threads() -> int:
    return _env_int("FASTVIDEO_SSD_SPILL_IO_THREADS", 2)


_MANAGERS: dict[torch.device, SsdSpillManager | None] = {}
_LOCK = threading.Lock()


def _proc_self_rss_bytes() -> int | None:
    """This process's resident set size in bytes (Linux), else None."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB -> B
    except (OSError, ValueError):
        return None
    return None


def _measured_baseline_bytes() -> int:
    """Node non-SAC RAM baseline: this rank's current RSS x local ranks.

    Built lazily on the first SAC pack, before the SAC staging pool has grown,
    so this rank's RSS approximates the per-rank non-SAC footprint (weights,
    optimizer, framework). Scaling by the local world size estimates the node
    baseline the derived budget subtracts. Ignored when an explicit budget is
    set.
    """
    rss = _proc_self_rss_bytes()
    return rss * local_world_size() if rss is not None else 0


def _build(device: torch.device) -> SsdSpillManager | None:
    if not ssd_spill_enabled():
        return None
    directory = ssd_dir()
    if not directory:
        logger.warning("FASTVIDEO_SSD_SPILL is set but FASTVIDEO_SSD_DIR is "
                       "missing; SSD spill disabled.")
        return None
    if not direct_io_supported(directory):
        logger.warning("SSD spill disabled: %s failed the direct-I/O probe "
                       "(overlay/network mount?). Point FASTVIDEO_SSD_DIR at "
                       "local NVMe.", directory)
        return None
    budget = resolve_host_ram_budget_bytes(
        baseline_bytes=_measured_baseline_bytes())
    if budget <= 0:
        logger.warning("SSD spill disabled: could not determine a host-RAM "
                       "budget; set FASTVIDEO_HOST_RAM_BUDGET_GB.")
        return None
    mgr = SsdSpillManager(directory,
                          budget,
                          io_threads=_io_threads(),
                          prefetch_window=_prefetch_window(),
                          pin=(device.type == "cuda"))
    logger.info("SSD spill enabled on %s: dir=%s budget=%.1f GB window=%d", device,
                directory, budget / 1024**3, _prefetch_window())
    return mgr


def _stats_line(mgr: SsdSpillManager, device: torch.device) -> str:
    return (
        f"SSD spill stats on {device}: n_spilled={mgr.n_spilled} "
        f"spilled_bytes={mgr.spilled_bytes} "
        f"({mgr.spilled_bytes / 1024**3:.3f} GiB) "
        f"n_reloaded={mgr.n_reloaded} resident_hits={mgr.resident_hits} "
        f"prefetch_hits={mgr.prefetch_hits} "
        f"prefetch_late={mgr.prefetch_late} "
        f"prefetch_underruns={mgr.prefetch_underruns} "
        f"budget_allocated={mgr.budget.allocated} "
        f"budget_cap={mgr.budget.cap_bytes} "
        f"live_handles={len(mgr._handles)} "
        f"n_spilled_by_category={dict(mgr.n_spilled_by_category)} "
        f"spilled_bytes_by_category={dict(mgr.spilled_bytes_by_category)}")


def log_all_manager_stats() -> None:
    """Log residency counters for every built manager (observability only).

    Called from the training finalize path so validation runs can record
    n_spilled / prefetch_underruns etc. Safe to call when SSD spill is off
    (no managers -> no-op).
    """
    with _LOCK:
        managers = [(d, m) for d, m in _MANAGERS.items() if m is not None]
    for device, mgr in managers:
        logger.info("%s", _stats_line(mgr, device))


_ORPHAN_TOTAL = 0
_WARNED_ORPHANS = False


def reset_step_all() -> int:
    """Reclaim orphaned handles for every built manager; return the total.

    Handles leave a manager only via ``release`` on the unpack/reload path, so a
    saved tensor whose backward node never executes would otherwise leak forever
    and ratchet toward "budget exceeded". Call after each backward / grad-accum
    micro-batch. No-op when SSD spill is off.

    Unconsumed saves are common and benign under generic saved-tensor offload
    (``saved_tensors_hooks`` captures every eligible save, including tensors on
    frozen submodules, detached branches, or discarded outputs whose grad node
    never runs). So warn only once -- with a running total so the volume is
    still visible -- and log subsequent reclaims at debug level, rather than
    emitting a per-micro-batch warning for the whole run.
    """
    global _ORPHAN_TOTAL, _WARNED_ORPHANS
    total = 0
    with _LOCK:
        managers = [(d, m) for d, m in _MANAGERS.items() if m is not None]
    for device, mgr in managers:
        n = mgr.reset_step()
        total += n
        if n:
            _ORPHAN_TOTAL += n
            if not _WARNED_ORPHANS:
                _WARNED_ORPHANS = True
                logger.warning(
                    "SSD spill: reclaimed %d orphaned handle(s) on %s (saved "
                    "tensors whose backward never ran -- normal for frozen or "
                    "detached branches). Further reclaims log at debug level; "
                    "running total is reported in the finalize stats.", n,
                    device)
            else:
                logger.debug(
                    "SSD spill: reclaimed %d orphaned handle(s) on %s "
                    "(running total %d)", n, device, _ORPHAN_TOTAL)
    return total


def _atexit_dump_stats() -> None:
    """File-based backup dump (survives OOM/error before finalize).

    Writes to ``$FASTVIDEO_SSD_STATS_DIR/ssd_stats_rank<rank>.txt`` using a
    freshly opened file so it works even if logging streams are torn down.
    """
    out_dir = os.environ.get("FASTVIDEO_SSD_STATS_DIR")
    if not out_dir:
        return
    with _LOCK:
        managers = [(d, m) for d, m in _MANAGERS.items() if m is not None]
    if not managers:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        path = os.path.join(out_dir, f"ssd_stats_rank{rank}.txt")
        with open(path, "w") as fh:
            for device, mgr in managers:
                fh.write(_stats_line(mgr, device) + "\n")
    except Exception:  # never let observability break shutdown
        pass


atexit.register(_atexit_dump_stats)


def ssd_manager(device: torch.device) -> SsdSpillManager | None:
    """Per-device spill manager if the gate passes, else ``None`` (cached)."""
    with _LOCK:
        if device not in _MANAGERS:
            _MANAGERS[device] = _build(device)
        return _MANAGERS[device]


def reset_managers() -> None:
    """Tear down cached managers (tests / reconfiguration)."""
    with _LOCK:
        for mgr in _MANAGERS.values():
            if mgr is not None:
                mgr.close()
        _MANAGERS.clear()
