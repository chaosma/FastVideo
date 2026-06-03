#!/usr/bin/env python3
"""Phase 5 verifier for the activation-streaming work.

Reads probe outputs produced by ``phase5_validation.sh`` and grades the
results against the §3 / §10 acceptance criteria in
``memory_prefetch_plan.md``. Emits both a Markdown report and a JSON
report. Exit code 0 = PASS, non-zero = FAIL.

Inputs (under ``--probe-root``)::

    <root>/pytest.log
    <root>/pytest.exit
    <root>/full/phase_memory.json
    <root>/full/layer_trace.csv
    <root>/full/train.log
    <root>/full_offload/phase_memory.json
    <root>/full_offload/layer_trace.csv
    <root>/full_offload/train.log
    <root>/streamed_offload/phase_memory.json
    <root>/streamed_offload/layer_trace.csv
    <root>/streamed_offload/train.log

Acceptance summary (mirrors §3 Phase 5 + §10 Phase 2 GPU validation steps):

1. Streaming pytest suite exits 0.
2. All three probe runs complete (exit 0, finite final loss).
3. STREAMED_OFFLOAD ``P5_bwd_peak.max_allocated_gb`` ≤ FULL_OFFLOAD's
   (within ``--peak-slack-gb`` slack).
4. STREAMED_OFFLOAD vs FULL_OFFLOAD step-2 loss match within ``--loss-tol``.
5. STREAMED_OFFLOAD step time ≤ ``--step-time-ratio`` × FULL_OFFLOAD step
   time at step 2.
6. STREAMED_OFFLOAD layer_trace shows strict FILO backward order
   (29 → 0, or whatever block count the model has).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

MODES = ("full", "full_offload", "streamed_offload")

LOSS_RE = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([\-0-9.eE+nanif]+)\s+step_time_sec=([\-0-9.eE+nanif]+)")


def _parse_loss_log(path: Path) -> list[dict[str, float]]:
    """Return per-step {step, loss, step_time_sec} parsed from train.log.

    Only matches the trainer's `[step N] loss=... step_time_sec=...`
    console echo (trainer.py:212), which is rank-0 only and always
    produced regardless of W&B mode.
    """
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "step_time_sec": float(m.group(3)),
                })
    return rows


def _load_phase_memory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _layer_trace_bwd_order(csv_path: Path) -> list[int] | None:
    """Return the sequence of layer_idx values for `bwd_pre` rows on the
    last probed step, or None if the file is unreadable."""
    if not csv_path.exists():
        return None
    last_step_rows: list[tuple[int, int]] = []
    last_step: int | None = None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["step"])
                phase = row["phase"]
                layer_idx = int(row["layer_idx"])
            except (KeyError, ValueError):
                continue
            if phase != "bwd_pre":
                continue
            if last_step is None or step > last_step:
                last_step = step
                last_step_rows = []
            if step == last_step:
                last_step_rows.append((step, layer_idx))
    return [li for _, li in last_step_rows]


def _is_strict_decreasing(seq: list[int]) -> bool:
    return len(seq) >= 2 and all(b < a for a, b in zip(seq, seq[1:]))


def _phase_max_alloc(phase_mem: dict[str, Any], phase: str,
                     prefer_step: int | None = None) -> float | None:
    """Return ``max_allocated_gb`` for the given phase on the highest
    available step (or ``prefer_step`` if specified)."""
    if not phase_mem:
        return None
    steps = sorted(phase_mem.keys(), key=lambda s: int(s))
    if prefer_step is not None:
        steps = [str(prefer_step)] if str(prefer_step) in phase_mem else steps
    for s in reversed(steps):
        snap = phase_mem.get(s, {}).get(phase)
        if snap and "max_allocated_gb" in snap:
            return float(snap["max_allocated_gb"])
    return None


def _check(name: str, condition: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "pass": bool(condition), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-root", required=True)
    ap.add_argument("--report-md", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--peak-slack-gb", type=float, default=1.0,
                    help="STREAMED_OFFLOAD peak may exceed FULL_OFFLOAD peak "
                    "by at most this many GB and still pass.")
    ap.add_argument("--loss-tol", type=float, default=1e-5,
                    help="Maximum |loss_streamed - loss_full_offload| at "
                    "step 2 (or the last probed step).")
    ap.add_argument("--step-time-ratio", type=float, default=1.20,
                    help="STREAMED_OFFLOAD step time / FULL_OFFLOAD step "
                    "time at the last probed step.")
    args = ap.parse_args()

    root = Path(args.probe_root)
    results: list[dict[str, Any]] = []

    # --- 1. pytest ----------------------------------------------------
    pytest_exit_path = root / "pytest.exit"
    pytest_exit = int(pytest_exit_path.read_text().strip()) \
        if pytest_exit_path.exists() else None
    results.append(_check(
        "streaming_pytest_passes",
        pytest_exit == 0,
        f"pytest exit code = {pytest_exit} (expected 0; pytest.log at "
        f"{root/'pytest.log'})"))

    # --- Load probe outputs ------------------------------------------
    probes: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        d = root / mode
        probes[mode] = {
            "exit": int((d / "exit").read_text().strip())
            if (d / "exit").exists() else None,
            "phase_memory": _load_phase_memory(d / "phase_memory.json"),
            "loss_rows": _parse_loss_log(d / "train.log"),
            "bwd_order": _layer_trace_bwd_order(d / "layer_trace.csv"),
        }

    # --- 2. all probe runs completed ---------------------------------
    for mode in MODES:
        p = probes[mode]
        finite_loss = bool(p["loss_rows"]) and math.isfinite(
            p["loss_rows"][-1]["loss"])
        results.append(_check(
            f"probe_{mode}_completed",
            p["exit"] == 0 and bool(p["phase_memory"]) and finite_loss,
            f"exit={p['exit']}, phase_memory entries="
            f"{len(p['phase_memory'])}, last loss="
            f"{p['loss_rows'][-1]['loss'] if p['loss_rows'] else 'n/a'}"))

    # --- 3. STREAMED_OFFLOAD peak ≤ FULL_OFFLOAD peak ----------------
    full_off_peak = _phase_max_alloc(
        probes["full_offload"]["phase_memory"], "P5_bwd_peak")
    streamed_peak = _phase_max_alloc(
        probes["streamed_offload"]["phase_memory"], "P5_bwd_peak")
    if full_off_peak is not None and streamed_peak is not None:
        results.append(_check(
            "streamed_peak_at_or_below_full_offload_peak",
            streamed_peak <= full_off_peak + args.peak_slack_gb,
            f"FULL_OFFLOAD P5_bwd_peak.max_alloc = {full_off_peak:.2f} GB; "
            f"STREAMED_OFFLOAD = {streamed_peak:.2f} GB; "
            f"slack = {args.peak_slack_gb:.2f} GB"))
    else:
        results.append(_check(
            "streamed_peak_at_or_below_full_offload_peak", False,
            f"missing P5_bwd_peak max_allocated_gb (full_offload="
            f"{full_off_peak}, streamed={streamed_peak})"))

    # --- 3b. STREAMED_OFFLOAD step peak should beat FULL (recompute) -
    #         on host metrics this is a softer comparison; informational
    full_peak = _phase_max_alloc(
        probes["full"]["phase_memory"], "P7_opt_done")
    streamed_step_peak = _phase_max_alloc(
        probes["streamed_offload"]["phase_memory"], "P7_opt_done")
    note = ""
    if full_peak is not None and streamed_step_peak is not None:
        note = (f"FULL P7_opt_done = {full_peak:.2f} GB, "
                f"STREAMED_OFFLOAD = {streamed_step_peak:.2f} GB")
    results.append({
        "name": "step_peak_streamed_vs_full_recompute_info",
        "pass": True,  # informational only
        "detail": note or "step-peak comparison unavailable",
        "informational": True,
    })

    # --- 4. loss match ------------------------------------------------
    fo_rows = probes["full_offload"]["loss_rows"]
    so_rows = probes["streamed_offload"]["loss_rows"]
    if fo_rows and so_rows:
        # Compare on the highest common step.
        common = sorted(set(r["step"] for r in fo_rows)
                        & set(r["step"] for r in so_rows))
        if common:
            target_step = common[-1]
            fo_loss = next(r["loss"] for r in fo_rows
                           if r["step"] == target_step)
            so_loss = next(r["loss"] for r in so_rows
                           if r["step"] == target_step)
            diff = abs(fo_loss - so_loss)
            results.append(_check(
                "streamed_vs_full_offload_loss_match",
                diff <= args.loss_tol,
                f"step {target_step}: full_offload={fo_loss:.7f}, "
                f"streamed={so_loss:.7f}, diff={diff:.2e} "
                f"(tol={args.loss_tol:.0e})"))
        else:
            results.append(_check(
                "streamed_vs_full_offload_loss_match", False,
                "no common step between full_offload and streamed_offload"))
    else:
        results.append(_check(
            "streamed_vs_full_offload_loss_match", False,
            f"loss rows missing (full_offload={len(fo_rows)}, "
            f"streamed={len(so_rows)})"))

    # --- 5. step-time ratio ------------------------------------------
    if fo_rows and so_rows:
        # Use the last probed step (skip step 0 which is dominated by
        # warmup / FSDP first-allgather).
        fo_step = max(r["step"] for r in fo_rows)
        so_step = max(r["step"] for r in so_rows)
        fo_time = next(r["step_time_sec"] for r in fo_rows
                       if r["step"] == fo_step)
        so_time = next(r["step_time_sec"] for r in so_rows
                       if r["step"] == so_step)
        ratio = so_time / fo_time if fo_time > 0 else float("inf")
        results.append(_check(
            "streamed_step_time_within_budget",
            ratio <= args.step_time_ratio,
            f"FULL_OFFLOAD step {fo_step} = {fo_time:.2f}s; "
            f"STREAMED_OFFLOAD step {so_step} = {so_time:.2f}s; "
            f"ratio = {ratio:.2f} (limit {args.step_time_ratio:.2f})"))
    else:
        results.append(_check(
            "streamed_step_time_within_budget", False,
            "step_time_sec rows missing in train.log"))

    # --- 6. FILO backward order in STREAMED_OFFLOAD ------------------
    bwd_order = probes["streamed_offload"]["bwd_order"]
    if bwd_order:
        results.append(_check(
            "streamed_filo_backward_order",
            _is_strict_decreasing(bwd_order),
            f"observed {len(bwd_order)} bwd_pre events: "
            f"first 3 = {bwd_order[:3]}, last 3 = {bwd_order[-3:]}"))
    else:
        results.append(_check(
            "streamed_filo_backward_order", False,
            "no bwd_pre rows in streamed_offload/layer_trace.csv"))

    # --- emit reports -------------------------------------------------
    hard_results = [r for r in results if not r.get("informational")]
    all_pass = all(r["pass"] for r in hard_results)

    payload = {
        "pass": all_pass,
        "checks": results,
        "metrics": {
            "full_offload_P5_bwd_peak_max_alloc_gb": full_off_peak,
            "streamed_offload_P5_bwd_peak_max_alloc_gb": streamed_peak,
            "full_P7_opt_done_max_alloc_gb": full_peak,
            "streamed_offload_P7_opt_done_max_alloc_gb": streamed_step_peak,
            "thresholds": {
                "peak_slack_gb": args.peak_slack_gb,
                "loss_tol": args.loss_tol,
                "step_time_ratio": args.step_time_ratio,
            },
        },
    }
    Path(args.report_json).write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Phase 5 validation report",
        "",
        f"**Verdict:** {'PASS' if all_pass else 'FAIL'}",
        "",
        "## Checks",
        "",
        "| # | Check | Result | Detail |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(results, start=1):
        flag = "✅ PASS" if r["pass"] else "❌ FAIL"
        if r.get("informational"):
            flag = "ℹ info"
        md_lines.append(
            f"| {i} | `{r['name']}` | {flag} | {r['detail']} |")
    md_lines += [
        "",
        "## Key metrics",
        "",
        f"- FULL_OFFLOAD `P5_bwd_peak.max_alloc_gb`: "
        f"{full_off_peak if full_off_peak is not None else 'n/a'}",
        f"- STREAMED_OFFLOAD `P5_bwd_peak.max_alloc_gb`: "
        f"{streamed_peak if streamed_peak is not None else 'n/a'}",
        f"- FULL (recompute) `P7_opt_done.max_alloc_gb`: "
        f"{full_peak if full_peak is not None else 'n/a'}",
        f"- STREAMED_OFFLOAD `P7_opt_done.max_alloc_gb`: "
        f"{streamed_step_peak if streamed_step_peak is not None else 'n/a'}",
        "",
        "## Thresholds in effect",
        "",
        f"- `--peak-slack-gb`: {args.peak_slack_gb}",
        f"- `--loss-tol`: {args.loss_tol}",
        f"- `--step-time-ratio`: {args.step_time_ratio}",
        "",
        "Re-run with different thresholds via "
        "`python scripts/4k_milestone/phase5_verify.py --probe-root <dir> "
        "--report-md … --report-json … --peak-slack-gb X --loss-tol Y "
        "--step-time-ratio Z`.",
    ]
    Path(args.report_md).write_text("\n".join(md_lines))

    print(f"Phase 5 verifier: {'PASS' if all_pass else 'FAIL'}")
    for r in results:
        flag = "PASS" if r["pass"] else ("INFO" if r.get("informational") else "FAIL")
        print(f"  [{flag}] {r['name']}: {r['detail']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
