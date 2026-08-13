# SPDX-License-Identifier: Apache-2.0
"""Host-RAM byte budget for the SSD spill tier.

The SSD tier bounds pinned-host RAM by a per-rank byte budget: pinned
activations beyond it spill to SSD. Two failure modes this module exists to
avoid (see ``.agents/exploration/ssd-offload-design.md``):

* **Container blindness.** Inside a cgroup, ``/proc/meminfo`` reports the *host*
  memory, not the container limit, so a budget derived from it overcommits. We
  read the cgroup v2 ``memory.max`` and take the binding (smaller) of it and
  physical RAM.
* **Per-rank overcommit.** Eight local ranks each deriving "available RAM"
  independently would claim 8x the node's memory. We divide by the local world
  size.

The pure ``compute_budget_bytes`` takes all system values as arguments so it is
fully testable without a real cgroup; ``resolve_host_ram_budget_bytes`` wires it
to the environment and filesystem.
"""
from __future__ import annotations

import os

from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Env knobs. The neutral name is preferred; the SAC-scoped name is kept as an
# alias because the SAC adapter is the first consumer.
_BUDGET_ENV = ("FASTVIDEO_HOST_RAM_BUDGET_GB", "FASTVIDEO_SAC_CPU_RAM_BUDGET_GB")

# Fraction of derived headroom held back for allocator slack, dataloader
# buffers, and measurement error.
_DEFAULT_MARGIN = 0.1


def local_world_size() -> int:
    """Ranks sharing this node's RAM (torchrun sets ``LOCAL_WORLD_SIZE``)."""
    v = os.environ.get("LOCAL_WORLD_SIZE")
    if v:
        try:
            return max(1, int(v))
        except ValueError:
            pass
    return 1


def resolve_budget_gb() -> float | None:
    """Explicit per-rank budget in GB from the environment, if set."""
    for name in _BUDGET_ENV:
        v = os.environ.get(name)
        if v:
            try:
                return float(v)
            except ValueError:
                logger.warning("SSD spill: ignoring non-numeric %s=%r", name, v)
    return None


def _read_first_int(path: str) -> int | None:
    try:
        with open(path) as fh:
            return int(fh.read().split()[0])
    except (OSError, ValueError):
        return None


def cgroup_v2_memory_max() -> int | None:
    """cgroup v2 memory limit in bytes, or None if unlimited/unavailable."""
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def proc_mem_total() -> int | None:
    """Physical RAM in bytes from /proc/meminfo (MemTotal), or None."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # kB -> B
    except (OSError, ValueError):
        return None
    return None


def compute_budget_bytes(*, explicit_gb: float | None, cgroup_max: int | None,
                         total_ram: int | None, baseline_bytes: int,
                         local_ranks: int, margin: float) -> int:
    """Per-rank host-RAM budget for spillable activations (pure).

    An explicit budget is taken as-is (already per-rank). Otherwise the node's
    binding memory limit is ``min(cgroup_max, total_ram)``; from it we subtract
    the measured non-SAC baseline, hold back ``margin``, and divide across local
    ranks. Returns 0 when no limit can be determined (caller should then require
    an explicit budget rather than guess).
    """
    if explicit_gb is not None:
        return max(0, int(explicit_gb * 1024**3))
    limits = [x for x in (cgroup_max, total_ram) if x is not None]
    if not limits:
        return 0
    total = min(limits)
    usable = (total - baseline_bytes) * (1.0 - margin)
    per_rank = usable / max(1, local_ranks)
    return max(0, int(per_rank))


def resolve_host_ram_budget_bytes(baseline_bytes: int = 0,
                                  margin: float = _DEFAULT_MARGIN) -> int:
    """Per-rank budget wired to env + cgroup + /proc/meminfo."""
    budget = compute_budget_bytes(explicit_gb=resolve_budget_gb(),
                                  cgroup_max=cgroup_v2_memory_max(),
                                  total_ram=proc_mem_total(),
                                  baseline_bytes=baseline_bytes,
                                  local_ranks=local_world_size(),
                                  margin=margin)
    logger.info(
        "SSD spill: host-RAM budget = %.1f GB/rank (baseline=%.1f GB, "
        "local_ranks=%d, margin=%.2f)", budget / 1024**3,
        baseline_bytes / 1024**3, local_world_size(), margin)
    return budget


class HostBudget:
    """Accounts *allocated* pinned-host bytes against the per-rank cap.

    The cap bounds physical RSS, so it must count every pinned buffer that is
    physically alive -- both in use and sitting in a reuse free-list -- not just
    the currently-resident working set (a freed-to-pool buffer still holds RSS,
    and PyTorch's caching host allocator would not return it to the OS anyway).
    ``allocated`` is therefore charged when a buffer is created and released only
    when it is actually dropped. ``nonspillable`` is the subset of resident bytes
    that cannot be spilled; ``overcommitted`` flags the unrecoverable case where
    those alone exceed the cap.

    Caveat: ``drop_allocation`` decrements ``allocated`` on the assumption that
    dropping a buffer frees its RSS, but PyTorch's caching host allocator retains
    the block (bucketed by size). So for a *uniform*-size workload (SAC, one
    size class, reused in place) the cap is exact, while for *mixed*-size
    workloads (generic saved tensors) a cross-size drop-then-realloc can leave
    physical RSS above the accounted ``allocated`` -- the cap is a close but soft
    bound there. See ``SsdSpillManager._make_room`` step 2.
    """

    __slots__ = ("cap_bytes", "allocated", "nonspillable")

    def __init__(self, cap_bytes: int) -> None:
        self.cap_bytes = cap_bytes
        self.allocated = 0
        self.nonspillable = 0

    @property
    def used(self) -> int:
        return self.allocated + self.nonspillable

    def free(self) -> int:
        return max(0, self.cap_bytes - self.used)

    def can_allocate(self, nbytes: int) -> bool:
        return self.used + nbytes <= self.cap_bytes

    def add_allocation(self, nbytes: int) -> None:
        self.allocated += nbytes

    def drop_allocation(self, nbytes: int) -> None:
        # Raise, don't clamp: an over-drop is exactly the accounting bug that
        # later corrupts capacity decisions, so surface it immediately.
        if nbytes > self.allocated:
            raise ValueError(
                f"dropping {nbytes} allocated bytes but only {self.allocated} "
                "are charged (double-drop or accounting bug)")
        self.allocated -= nbytes

    def add_nonspillable(self, nbytes: int) -> None:
        self.nonspillable += nbytes

    def release_nonspillable(self, nbytes: int) -> None:
        if nbytes > self.nonspillable:
            raise ValueError(
                f"releasing {nbytes} nonspillable bytes but only "
                f"{self.nonspillable} are charged (double-release?)")
        self.nonspillable -= nbytes

    def overcommitted(self) -> bool:
        return self.nonspillable > self.cap_bytes
