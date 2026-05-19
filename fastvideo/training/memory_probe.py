# SPDX-License-Identifier: Apache-2.0
"""Memory probe for the FastVideo training step.

Records two kinds of events per training step:

1. Phase-boundary snapshots (9 named phases per the wan-calculator spec).
   Calls torch.cuda.synchronize() and reads memory_allocated /
   max_memory_allocated / memory_reserved. max is reset between phases.
2. Per-layer fwd/bwd events. Forward + backward hooks on every transformer
   block fire 4 times per block per step (fwd_pre, fwd_post, bwd_pre,
   bwd_post). These reads do NOT synchronize.

Enabled by env vars (rank-0 only by default):
  FASTVIDEO_MEM_PROBE_STEPS=2     how many steps to trace (>=1)
  FASTVIDEO_MEM_PROBE_DIR=<path>  where to write outputs (default: cwd)
  FASTVIDEO_MEM_PROBE_TOPK=0      >0: also capture top-K live allocations
                                   per phase snap via cuda.memory._snapshot()
  FASTVIDEO_MEM_PROBE_ALL_RANKS=0 1: probe on every rank, not just rank 0

Outputs (written when the probe auto-disables after STEPS steps):
  <dir>/layer_trace.csv       per-block fwd/bwd events, all probed steps
  <dir>/phase_memory.json     phase-boundary snapshots, all probed steps
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)


# Phase names follow wan-calculator/instructions.md.
PHASES = (
    "P0_idle",
    "P1_inputs_done",
    "P2_fwd_start",
    "P3_fwd_half",
    "P4_fwd_end",
    "P5_bwd_peak",
    "P6_post_bwd",
    "P7_opt_done",
    "P0_next",
)

# Per-block per-step event labels.
LAYER_PHASES = ("fwd_pre", "fwd_post", "bwd_pre", "bwd_post")

# Module-level singleton (rank-local). None when probe is disabled.
_PROBE: "MemoryProbe | None" = None
_INIT_ATTEMPTED: bool = False


def get_probe() -> "MemoryProbe | None":
    """Return the active probe singleton, or None if disabled."""
    return _PROBE


def maybe_init_from_env(rank: int) -> "MemoryProbe | None":
    """Initialize the probe singleton from env vars, if requested.

    Idempotent; subsequent calls return the same instance (or None)."""
    global _PROBE, _INIT_ATTEMPTED
    if _INIT_ATTEMPTED:
        return _PROBE
    _INIT_ATTEMPTED = True

    n_steps_str = os.environ.get("FASTVIDEO_MEM_PROBE_STEPS", "0")
    try:
        n_steps = int(n_steps_str)
    except ValueError:
        logger.warning("FASTVIDEO_MEM_PROBE_STEPS=%r is not an int; "
                       "probe disabled.", n_steps_str)
        return None
    if n_steps <= 0:
        return None

    all_ranks = os.environ.get("FASTVIDEO_MEM_PROBE_ALL_RANKS", "0") == "1"
    if rank != 0 and not all_ranks:
        return None

    out_dir = os.environ.get("FASTVIDEO_MEM_PROBE_DIR", ".")
    topk = int(os.environ.get("FASTVIDEO_MEM_PROBE_TOPK", "0"))
    _PROBE = MemoryProbe(num_steps=n_steps, out_dir=out_dir, topk=topk,
                         rank=rank)
    logger.info("MemoryProbe initialized: steps=%d dir=%s topk=%d rank=%d",
                n_steps, out_dir, topk, rank)
    return _PROBE


def snap(name: str) -> None:
    """Module-level shortcut: snap() if probe is active, else no-op."""
    if _PROBE is not None:
        _PROBE.snap(name)


def step_begin() -> None:
    if _PROBE is not None:
        _PROBE.step_begin()


def step_end() -> None:
    if _PROBE is not None:
        _PROBE.step_end()


class MemoryProbe:
    """Per-step memory tracing. See module docstring."""

    def __init__(self, num_steps: int, out_dir: str, topk: int,
                 rank: int) -> None:
        self.num_steps = num_steps
        self.out_dir = out_dir
        self.topk = topk
        self.rank = rank

        self.step_idx: int = 0
        self.installed: bool = False
        self.finalized: bool = False

        # Hooks we registered, for later cleanup.
        self._handles: list[Any] = []
        # Wrapper modules we found, in order.
        self._block_modules: list[nn.Module] = []
        self._num_layers_total: int = 0
        self._mid_layer_idx: int = -1

        # Buffers.
        self._layer_rows: list[dict[str, Any]] = []
        self._phase_snapshots: "OrderedDict[int, OrderedDict[str, Any]]" = (
            OrderedDict())

        # For categorize(), held weakly via direct refs (probe lifetime ends
        # at uninstall, well before training ends).
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._extras: dict[str, nn.Module] = {}

        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def install(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        extras: dict[str, nn.Module] | None = None,
    ) -> None:
        """Discover transformer blocks and install fwd/bwd hooks.

        Looks for the first attribute named in TRANSFORMER_BLOCK_NAMES (as
        defined in activation_checkpoint.py). Each entry in that ModuleList
        is expected to be a checkpoint_wrapper (post `apply_activation_
        checkpointing`), and we install hooks on the wrapper itself."""
        if self.installed:
            return

        self._model = model
        self._optimizer = optimizer
        self._extras = extras or {}

        from fastvideo.training.activation_checkpoint import (
            TRANSFORMER_BLOCK_NAMES, )

        blocks: nn.Module | None = None
        for name in TRANSFORMER_BLOCK_NAMES:
            candidate = getattr(model, name, None)
            if candidate is not None:
                blocks = candidate
                break

        if blocks is None:
            logger.warning(
                "MemoryProbe.install: no transformer block container found "
                "on model (looked for %s); per-layer hooks not installed.",
                TRANSFORMER_BLOCK_NAMES)
            self.installed = True  # still allow phase snaps
            return

        for idx, block in enumerate(blocks):
            self._block_modules.append(block)
            # forward_pre and forward hooks fire on the wrapper. With
            # checkpoint_wrapper, the wrapper's own .forward() runs once per
            # training step (the real forward). The recompute happens inside
            # an autograd.Function during backward and calls the *inner*
            # block's forward — so to flag "bwd start" we use a backward
            # pre-hook on the wrapper.
            h1 = block.register_forward_pre_hook(
                self._make_fwd_pre_hook(idx))
            h2 = block.register_forward_hook(self._make_fwd_post_hook(idx))
            h3 = block.register_full_backward_pre_hook(
                self._make_bwd_pre_hook(idx))
            h4 = block.register_full_backward_hook(
                self._make_bwd_post_hook(idx))
            self._handles.extend((h1, h2, h3, h4))

        self._num_layers_total = len(self._block_modules)
        self._mid_layer_idx = self._num_layers_total // 2
        self.installed = True
        logger.info(
            "MemoryProbe.install: hooked %d transformer blocks; P3_fwd_half "
            "will fire at layer idx %d.",
            self._num_layers_total,
            self._mid_layer_idx,
        )

    def uninstall(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        self.installed = False

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------

    def step_begin(self) -> None:
        if self.finalized:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()

    def step_end(self) -> None:
        if self.finalized:
            return
        self.step_idx += 1
        if self.step_idx >= self.num_steps:
            self.finalize()

    def finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        self.uninstall()
        try:
            self._flush()
        except Exception as e:  # noqa: BLE001
            logger.exception("MemoryProbe: failed to flush outputs: %s", e)
        logger.info("MemoryProbe: finalized after %d step(s); outputs in %s",
                    self.step_idx, self.out_dir)

    # ------------------------------------------------------------------
    # Phase snap
    # ------------------------------------------------------------------

    def snap(self, name: str) -> None:
        """Record a named phase boundary. Synchronizes, reads memory, resets
        the peak counter for the next phase."""
        if self.finalized:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        rec: dict[str, Any] = {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
        if self._model is not None and self._optimizer is not None:
            try:
                rec["categories_gb"] = _categorize(self._model,
                                                   self._optimizer,
                                                   self._extras)
            except Exception as e:  # noqa: BLE001
                logger.debug("categorize failed at %s: %s", name, e)
        if self.topk > 0:
            try:
                rec["top_allocations"] = _top_allocations(self.topk)
            except Exception as e:  # noqa: BLE001
                logger.debug("top_allocations failed at %s: %s", name, e)
        self._phase_snapshots.setdefault(self.step_idx,
                                         OrderedDict())[name] = rec
        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------------------
    # Per-layer hooks (closures bound to block idx)
    # ------------------------------------------------------------------

    def _make_fwd_pre_hook(self, idx: int):

        def hook(module, args):  # noqa: ARG001
            self._record_layer(idx, "fwd_pre")

        return hook

    def _make_fwd_post_hook(self, idx: int):

        def hook(module, args, output):  # noqa: ARG001
            self._record_layer(idx, "fwd_post")
            if idx == self._mid_layer_idx:
                # Cheap to inline P3 here; avoids needing an external snap()
                # site for it.
                self.snap("P3_fwd_half")

        return hook

    def _make_bwd_pre_hook(self, idx: int):

        def hook(module, grad_output):  # noqa: ARG001
            self._record_layer(idx, "bwd_pre")

        return hook

    def _make_bwd_post_hook(self, idx: int):

        def hook(module, grad_input, grad_output):  # noqa: ARG001
            self._record_layer(idx, "bwd_post")

        return hook

    def _record_layer(self, block_idx: int, which: str) -> None:
        if self.finalized:
            return
        if not torch.cuda.is_available():
            return
        # No synchronize: we want non-perturbing snapshots. memory_allocated
        # is allocator-side bookkeeping, not kernel-side, so this is cheap
        # and correct without a barrier.
        self._layer_rows.append({
            "step": self.step_idx,
            "layer_idx": block_idx,
            "phase": which,
            "t_ns": time.monotonic_ns(),
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        suffix = "" if self.rank == 0 else f".rank{self.rank}"
        csv_path = os.path.join(self.out_dir, f"layer_trace{suffix}.csv")
        json_path = os.path.join(self.out_dir,
                                 f"phase_memory{suffix}.json")

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=("step", "layer_idx", "phase", "t_ns",
                            "allocated_gb", "reserved_gb",
                            "max_allocated_gb"),
            )
            w.writeheader()
            for row in self._layer_rows:
                w.writerow(row)
        logger.info("MemoryProbe wrote %d rows to %s",
                    len(self._layer_rows), csv_path)

        with open(json_path, "w") as f:
            json.dump(self._phase_snapshots, f, indent=2, default=str)
        logger.info("MemoryProbe wrote phase snapshots to %s", json_path)


# ----------------------------------------------------------------------
# Categorization
# ----------------------------------------------------------------------


def _tensor_local_bytes(t: torch.Tensor) -> int:
    """Bytes resident on this rank's device for `t`, handling DTensor."""
    try:
        local = t.to_local() if hasattr(t, "to_local") else t
    except Exception:  # noqa: BLE001
        local = t
    if local.device.type != "cuda":
        return 0
    return local.numel() * local.element_size()


def _categorize(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    extras: dict[str, nn.Module],
) -> dict[str, float]:
    """Approximate per-card memory by category, in GB.

    Categories follow wan-calculator/instructions.md. Numbers reflect *what
    is currently allocated on this rank's device*, not global model state.
    Categories that don't have a single canonical home (FSDP all-gather
    workspaces, NCCL buffers, activation stashes, etc.) are not attributed
    here — the residual against allocated_gb is the framework / activation
    band."""
    cats = {
        "weights_bf16": 0,
        "grads_bf16": 0,
        "master_fp32": 0,
        "adam_m": 0,
        "adam_v": 0,
        "vae": 0,
        "t5": 0,
        "other_named": 0,
    }
    seen_param_ids: set[int] = set()
    for p in model.parameters():
        seen_param_ids.add(id(p))
        cats["weights_bf16"] += _tensor_local_bytes(p.data)
        if p.grad is not None:
            cats["grads_bf16"] += _tensor_local_bytes(p.grad)

    for group_state in optimizer.state.values():
        if not isinstance(group_state, dict):
            continue
        for k, v in group_state.items():
            if not torch.is_tensor(v):
                continue
            b = _tensor_local_bytes(v)
            key = str(k).lower()
            if key in ("exp_avg", ):
                cats["adam_m"] += b
            elif key in ("exp_avg_sq", ):
                cats["adam_v"] += b
            elif "fp32" in key or key in ("master_param", "fp32_param"):
                cats["master_fp32"] += b
            elif key == "step":
                continue
            else:
                cats["other_named"] += b

    for label in ("vae", "t5"):
        m = extras.get(label)
        if m is None:
            continue
        for p in m.parameters():
            if id(p) in seen_param_ids:
                continue
            cats[label] += _tensor_local_bytes(p.data)

    return {k: v / 1e9 for k, v in cats.items()}


def _top_allocations(k: int) -> list[dict[str, Any]]:
    """Return the top-K largest live allocations from cuda.memory._snapshot.

    Aggregated by stack-trace-tip frame; sizes in GB. Best-effort: snapshot
    schema varies between PyTorch versions."""
    snap_dict = torch.cuda.memory._snapshot()
    segments = snap_dict.get("segments", [])
    allocs: list[dict[str, Any]] = []
    for seg in segments:
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            allocs.append({
                "size_gb": blk.get("size", 0) / 1e9,
                "frames": [
                    f.get("name", "?")
                    for f in (blk.get("frames") or [])[:3]
                ],
            })
    allocs.sort(key=lambda a: a["size_gb"], reverse=True)
    return allocs[:k]
