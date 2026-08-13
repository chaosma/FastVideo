# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for generic saved-tensors SSD spill (activation_trace + gate).

CPU-only: the SSD pack/unpack path is CUDA-gated (eligibility requires a CUDA
tensor), so here we verify the opt-in gate and that the path is inert -- the
existing behavior is preserved -- for ineligible tensors and when the gate is
off. The CUDA round-trip and gradient parity are validated on a GPU box.
"""
from __future__ import annotations

import pytest
import torch

from fastvideo.train.utils.activation_trace import (ActivationTrace,
                                                    _SavedTensorHandle,
                                                    _SavedTensorSsdHandle)
from fastvideo.training import ssd_config


def test_gate_requires_both_flags(monkeypatch) -> None:
    for k in ("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "FASTVIDEO_SSD_SPILL"):
        monkeypatch.delenv(k, raising=False)
    assert not ssd_config.saved_tensors_ssd_enabled()
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")
    assert not ssd_config.saved_tensors_ssd_enabled()  # core gate still off
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    assert ssd_config.saved_tensors_ssd_enabled()


def test_gate_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", raising=False)
    monkeypatch.delenv("FASTVIDEO_SSD_SPILL", raising=False)
    at = ActivationTrace(torch.nn.Linear(2, 2), step=0, offload_to_cpu=True)
    assert at._ssd_spill is False


def test_activation_trace_reads_gate(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    at = ActivationTrace(torch.nn.Linear(2, 2), step=0, offload_to_cpu=True)
    assert at._ssd_spill is True


def test_ineligible_cpu_tensor_never_uses_ssd(monkeypatch) -> None:
    # Gate on, but a CPU tensor is ineligible (not CUDA) -> existing detached
    # path, no SSD handle, round-trips unchanged.
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    at = ActivationTrace(torch.nn.Linear(2, 2), step=0, offload_to_cpu=True,
                         min_offload_bytes=0)
    t = torch.randn(128)
    handle = at._pack_hook(t)
    assert isinstance(handle, _SavedTensorHandle)
    assert not isinstance(handle, _SavedTensorSsdHandle)
    assert handle.orig_device is None
    assert torch.equal(at._unpack_hook(handle), t)
    assert at._ssd_admitted_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_generic_ssd_roundtrip_grad_parity_cuda(tmp_path, monkeypatch) -> None:
    # Force real generic saved tensors through
    # acquire_staging -> admit -> spill -> ensure_resident -> H2D -> release,
    # and confirm gradients match the no-offload baseline (byte-exact reload).
    from fastvideo.training import ssd_config

    def _run(use_trace: bool):
        torch.manual_seed(0)
        x = torch.randn(2048, 2048, device="cuda", requires_grad=True)
        w = torch.randn(2048, 2048, device="cuda", requires_grad=True)
        if use_trace:
            trace = ActivationTrace(torch.nn.Identity(), step=0,
                                    offload_to_cpu=True,
                                    min_offload_bytes=1 << 20)
            assert trace._ssd_spill  # gate must be on for this run
            with trace:
                loss = torch.relu(torch.relu(x @ w) @ w).pow(2).sum()
        else:
            loss = torch.relu(torch.relu(x @ w) @ w).pow(2).sum()
        loss.backward()
        return x.grad.detach().clone(), w.grad.detach().clone()

    # Baseline: no offload.
    for k in ("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "FASTVIDEO_SSD_SPILL"):
        monkeypatch.delenv(k, raising=False)
    ssd_config.reset_managers()
    gx0, gw0 = _run(use_trace=False)

    # SSD spill with a tiny budget (~32 MB) so the 16 MB saved tensors spill.
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SSD_DIR", str(tmp_path))
    monkeypatch.setenv("FASTVIDEO_HOST_RAM_BUDGET_GB", "0.03")
    ssd_config.reset_managers()
    try:
        gx1, gw1 = _run(use_trace=True)
        mgr = ssd_config.ssd_manager(torch.device("cuda:0"))
        assert mgr is not None
        # A generic saved tensor actually hit disk and was read back.
        assert mgr.n_spilled_by_category.get("saved_tensor", 0) >= 1
        assert mgr.n_reloaded >= 1
        torch.testing.assert_close(gx1, gx0, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(gw1, gw0, rtol=1e-4, atol=1e-4)
    finally:
        ssd_config.reset_managers()


def test_full_checkpoint_skips_block_context_when_ssd_off(monkeypatch) -> None:
    # Blast-radius guard: with the generic SSD gate off, non-SSD checkpoint
    # types must NOT wrap block forwards (no contextvar overhead, no closure
    # that breaks deepcopy/pickle).
    from fastvideo.training.activation_checkpoint import (
        apply_activation_checkpointing)
    monkeypatch.delenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", raising=False)
    monkeypatch.delenv("FASTVIDEO_SSD_SPILL", raising=False)
    m = torch.nn.Module()
    m.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    apply_activation_checkpointing(m, "full")
    assert all(not hasattr(b, "_fastvideo_block_context_index")
               for b in m.blocks)


def test_full_checkpoint_wraps_block_context_when_generic_ssd_on(
        monkeypatch) -> None:
    from fastvideo.training.activation_checkpoint import (
        apply_activation_checkpointing)
    monkeypatch.setenv("FASTVIDEO_SSD_SPILL", "1")
    monkeypatch.setenv("FASTVIDEO_SAVED_TENSORS_SSD_SPILL", "1")
    m = torch.nn.Module()
    m.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
    apply_activation_checkpointing(m, "full")
    assert all(hasattr(b, "_fastvideo_block_context_index") for b in m.blocks)
