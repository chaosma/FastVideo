# SPDX-License-Identifier: Apache-2.0
"""Tests for the SSD spill enable gate + manager factory (ssd_config).

CPU-only: exercises the gate preconditions (env, dir, probe, budget) on the CPU
device. The GPU<->host transfer path is validated separately on a GPU box.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.training import ssd_config
from fastvideo.training.ssd_manager import SsdSpillManager

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("FASTVIDEO_SSD_SPILL", "FASTVIDEO_SSD_DIR",
              "FASTVIDEO_HOST_RAM_BUDGET_GB", "FASTVIDEO_SAC_CPU_RAM_BUDGET_GB"):
        monkeypatch.delenv(k, raising=False)
    ssd_config.reset_managers()
    yield
    ssd_config.reset_managers()


def test_disabled_by_default() -> None:
    assert ssd_config.ssd_manager(CPU) is None


def test_enabled_but_no_dir_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    assert ssd_config.ssd_manager(CPU) is None


def test_enabled_but_bad_dir_fails_probe(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", "/nonexistent/path/that/cannot/open")
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "8")
    assert ssd_config.ssd_manager(CPU) is None


def test_enabled_but_no_budget_is_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", str(tmp_path))
    # No explicit budget; force the derivation to yield 0 by stubbing sources.
    monkeypatch.setattr(ssd_config, "resolve_host_ram_budget_bytes",
                        lambda *a, **k: 0)
    assert ssd_config.ssd_manager(CPU) is None


def test_full_gate_builds_manager_and_caches(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", str(tmp_path))
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "8")
    mgr = ssd_config.ssd_manager(CPU)
    assert isinstance(mgr, SsdSpillManager)
    assert mgr.budget.cap_bytes == 8 * 1024**3
    # Cached: same instance, probe not re-run.
    assert ssd_config.ssd_manager(CPU) is mgr


def test_reset_managers_closes_and_clears(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", str(tmp_path))
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "8")
    mgr = ssd_config.ssd_manager(CPU)
    ssd_config.reset_managers()
    mgr2 = ssd_config.ssd_manager(CPU)
    assert mgr2 is not mgr  # rebuilt after reset


def test_reset_step_all_warns_once_for_benign_orphans(monkeypatch, tmp_path):
    # Unconsumed saves are normal under generic offload (frozen/detached
    # branches), so reset_step_all must warn once, not per micro-batch.
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", str(tmp_path))
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "8")
    monkeypatch.setattr(ssd_config, "_WARNED_ORPHANS", False)
    monkeypatch.setattr(ssd_config, "_ORPHAN_TOTAL", 0)
    warns: list = []
    monkeypatch.setattr(ssd_config.logger, "warning",
                        lambda *a, **k: warns.append(a))

    mgr = ssd_config.ssd_manager(CPU)

    def _admit_orphan(i: int) -> None:
        t = torch.tensor([float(i)], dtype=torch.bfloat16)
        nbytes = t.numel() * t.element_size()
        buf = mgr.acquire_staging(nbytes)
        mgr.admit(buf, shape=tuple(t.shape), dtype=t.dtype, nbytes=nbytes)

    _admit_orphan(0)
    assert ssd_config.reset_step_all() == 1
    _admit_orphan(1)
    assert ssd_config.reset_step_all() == 1
    assert len(warns) == 1  # warned once across both reclaims
    assert ssd_config._ORPHAN_TOTAL == 2  # running total tracked
