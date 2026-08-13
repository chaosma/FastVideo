# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SSD spill host-RAM budget (fastvideo.training.ssd_budget).

Pure logic: the cgroup/proc values are injected, so these run anywhere.
"""
from __future__ import annotations

import pytest

from fastvideo.training.ssd_budget import (HostBudget, compute_budget_bytes,
                                           local_world_size, resolve_budget_gb)

GB = 1024**3


def test_explicit_budget_is_taken_as_is() -> None:
    # Explicit is per-rank and bypasses derivation (ranks/margin ignored).
    b = compute_budget_bytes(explicit_gb=100, cgroup_max=None, total_ram=None,
                             baseline_bytes=0, local_ranks=8, margin=0.1)
    assert b == 100 * GB


def test_prefers_binding_cgroup_limit_over_host_ram() -> None:
    # cgroup limit (1.9 TiB) is smaller than host RAM (4 TiB) -> binding.
    b = compute_budget_bytes(explicit_gb=None,
                             cgroup_max=1900 * GB, total_ram=4000 * GB,
                             baseline_bytes=0, local_ranks=1, margin=0.0)
    assert b == 1900 * GB


def test_falls_back_to_proc_when_no_cgroup() -> None:
    b = compute_budget_bytes(explicit_gb=None, cgroup_max=None,
                             total_ram=200 * GB, baseline_bytes=0,
                             local_ranks=1, margin=0.0)
    assert b == 200 * GB


def test_subtracts_baseline_and_divides_by_ranks_and_margin() -> None:
    # (1000 - 400) GB * (1 - 0.1) / 8 ranks = 600 * 0.9 / 8 = 67.5 GB.
    b = compute_budget_bytes(explicit_gb=None, cgroup_max=1000 * GB,
                             total_ram=None, baseline_bytes=400 * GB,
                             local_ranks=8, margin=0.1)
    assert b == int(600 * GB * 0.9 / 8)


def test_returns_zero_when_no_limit_known() -> None:
    # No explicit, no cgroup, no proc -> cannot guess; caller must require one.
    assert compute_budget_bytes(explicit_gb=None, cgroup_max=None,
                                total_ram=None, baseline_bytes=0,
                                local_ranks=8, margin=0.1) == 0


def test_never_negative_when_baseline_exceeds_total() -> None:
    b = compute_budget_bytes(explicit_gb=None, cgroup_max=100 * GB,
                             total_ram=None, baseline_bytes=200 * GB,
                             local_ranks=4, margin=0.1)
    assert b == 0


def test_resolve_budget_gb_reads_env_and_alias(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_HOST_RAM_BUDGET_GB", raising=False)
    monkeypatch.delenv("FASTVIDEO_SAC_CPU_RAM_BUDGET_GB", raising=False)
    assert resolve_budget_gb() is None
    monkeypatch.setenv("FASTVIDEO_SAC_CPU_RAM_BUDGET_GB", "64")
    assert resolve_budget_gb() == 64.0
    # Neutral name wins over the alias.
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "128")
    assert resolve_budget_gb() == 128.0


def test_local_world_size_env(monkeypatch) -> None:
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    assert local_world_size() == 1
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert local_world_size() == 8


def test_host_budget_caps_allocated_plus_nonspillable() -> None:
    # The cap bounds physical RSS = allocated (in-use + free-list) plus any
    # nonspillable resident bytes.
    b = HostBudget(cap_bytes=10)
    b.add_allocation(4)
    b.add_nonspillable(3)
    assert b.used == 7 and b.free() == 3
    assert b.can_allocate(3) and not b.can_allocate(4)
    assert not b.overcommitted()


def test_host_budget_allocation_add_and_drop() -> None:
    b = HostBudget(cap_bytes=10)
    b.add_allocation(6)
    assert not b.can_allocate(5) and b.can_allocate(4)
    b.drop_allocation(6)  # a free-list buffer was actually dropped
    assert b.allocated == 0 and b.can_allocate(10)


def test_host_budget_overcommit_when_nonspillable_exceeds_cap() -> None:
    b = HostBudget(cap_bytes=5)
    b.add_nonspillable(6)
    assert b.overcommitted()


def test_host_budget_over_release_raises_not_clamps() -> None:
    # Over-drop/over-release is an accounting bug that would corrupt later
    # capacity decisions; it must raise, not silently clamp to zero.
    b = HostBudget(cap_bytes=10)
    b.add_allocation(4)
    with pytest.raises(ValueError):
        b.drop_allocation(5)
    b.add_nonspillable(2)
    with pytest.raises(ValueError):
        b.release_nonspillable(3)
