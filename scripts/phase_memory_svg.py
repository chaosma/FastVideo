#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a stacked per-card GPU memory timeline SVG from phase_memory.json.

Reads the JSON written by ``fastvideo/training/memory_probe.py`` and produces
a phase-aligned area chart: static layers at the bottom (Adam m+v, master
fp32, bf16 weights, grads), a band for the activation stash, and the transient
burst (= max_allocated - allocated).

The transient is an *in-window* peak --- it happens between two phase snaps,
not at a snap. So each phase window is drawn with two x-samples: the boundary
(allocated; transient = 0) and a mid-point (the window's max_allocated). At the
mid-point the bands are stacked at their peak-time composition, so the red
transient reads as a localized bump rather than a band smeared across the whole
timeline. For the backward window this matters: the peak is hit at the *start*
of backward while the stash is still resident, so the mid-point stash must be
the resident-at-peak value (taken from layer_trace.csv via --layer-trace), not
the post-drain boundary value of ~0. Without a layer trace it falls back to the
forward-end live set.

Usage:
    python3 scripts/phase_memory_svg.py \\
        --phase-memory logs/mem_probe_8gpu_121f/phase_memory.json \\
        --layer-trace  logs/mem_probe_8gpu_121f/layer_trace.csv \\
        --title "Wan 2.1 14B / 121f / 8x H200 (baseline)" \\
        --out reports/figures/121f_baseline.svg
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

# Plot canvas
W, H = 1300, 760
MARGIN_L, MARGIN_R = 70, 320
MARGIN_T, MARGIN_B = 90, 90
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

H200_CAP_GB = 141.0

# Colors --- match the reference plot palette.
COLORS = {
    "adam":       "#1e293b",   # dark navy
    "master":     "#334155",   # slate
    "weights":    "#475569",   # slate light
    "grads":      "#94a3b8",   # gray-blue
    "vae":        "#a3a3a3",   # gray
    "activations":"#fb923c",   # orange (the offloadable stash)
    "transient":  "#dc2626",   # red (max - allocated peak)
    "adam_step":  "#10b981",   # teal-green (the P7 transient)
    "axis":       "#1f2937",
    "grid":       "#e5e7eb",
    "cap":        "#dc2626",
}

LAYER_ORDER = [
    ("adam",        "Adam m+v (fp32)"),
    ("weights",     "params (fp32 master)"),
    ("master",      "fp32 master copy"),
    ("grads",       "grads (fp32)"),
    ("vae",         "VAE"),
    ("activations", "activation stash"),
]

# Bottom-to-top stack order, transient last (it caps each window's peak).
BANDS = [k for k, _ in LAYER_ORDER] + ["transient"]


def _esc(s: str) -> str:
    return html.escape(s)


def _text(x: float, y: float, content: str, *, size: int = 12,
          anchor: str = "start", weight: str = "400",
          fill: str = "#111827") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{fill}" font-family="-apple-system, BlinkMacSystemFont, '
            f'Helvetica, Arial, sans-serif">{_esc(content)}</text>')


def _components(rec: dict[str, Any]) -> dict[str, float]:
    """Pull the named category bytes out of one phase snapshot.

    Note on labels: the probe dumps every ``model.parameter()`` into its
    ``weights_bf16`` bucket regardless of dtype. In this training config
    the params are fp32 and serve as the master copy (bf16 is cast on the
    fly inside forward, not resident), so we surface that bucket as
    ``weights`` / "params (fp32 master)". The separate ``master_fp32``
    bucket is only non-zero when the optimizer state carries an explicit
    fp32 master tensor --- usually 0 here. Grads are read directly from
    ``grads_bf16`` at every phase (populated once backward runs)."""
    cats = rec.get("categories_gb") or {}
    return {
        "adam":    float(cats.get("adam_m", 0.0)) + float(cats.get("adam_v", 0.0)),
        "master":  float(cats.get("master_fp32", 0.0)),
        "weights": float(cats.get("weights_bf16", 0.0)),
        "grads":   float(cats.get("grads_bf16", 0.0)),
        "vae":     float(cats.get("vae", 0.0)),
    }


def _resident_at_peak(layer_trace: Path | None,
                      step: str | None = None
                      ) -> tuple[float, float] | None:
    """``(allocated_GB, backward_progress)`` at the backward-peak instant.

    The 9-phase JSON only stores the boundary ``allocated`` and the window
    ``max_allocated``; it cannot say how much was *resident* when the peak
    occurred. During the backward window the peak lands at the very start
    (the saved stash is all still on-card) while the boundary snapshot is
    taken after the stash has drained to ~0 --- so the resident-at-peak is
    exactly what separates genuine recompute transient from stash. Read it
    from the per-block trace: the ``allocated`` at the backward event with
    the highest ``max_allocated``.

    Also returns how far into backward that peak is (0 = first block .. 1 =
    last), so the caller can scale the grads that have accumulated by then ---
    grads grow from 0 to their full value across backward, so the baseline
    (peaks early) has ~0 grads at its peak while offload (peaks late) has the
    full amount. Returns None if no trace is supplied."""
    if layer_trace is None or not layer_trace.exists():
        return None
    rows = list(csv.DictReader(layer_trace.open()))
    if not rows:
        return None
    if step is None:
        step = rows[0]["step"]
    bwd = [r for r in rows
           if r["step"] == step and r["phase"].startswith("bwd")]
    bwd.sort(key=lambda r: int(r["t_ns"]))
    if not bwd:
        return None
    pk_i = max(range(len(bwd)),
               key=lambda j: float(bwd[j]["max_allocated_gb"]))
    progress = pk_i / (len(bwd) - 1) if len(bwd) > 1 else 0.0
    return float(bwd[pk_i]["allocated_gb"]), progress


def render(phase_memory_path: Path, *, title: str, subtitle: str,
           highlight_phase: str = "P5_bwd_peak",
           resident_at_peak: float | None = None,
           bwd_progress: float | None = None) -> str:
    data = json.loads(phase_memory_path.read_text())
    snap = data[next(iter(data))]   # first (only) step
    phases = list(snap.keys())
    n = len(phases)

    # Per-phase boundary numbers.
    rows: list[dict[str, Any]] = []
    for ph in phases:
        rec = snap[ph]
        comp = _components(rec)
        allocated = float(rec["allocated_gb"])
        max_alloc = float(rec["max_allocated_gb"])
        named_sum = sum(comp.values())
        rows.append({
            "phase":     ph,
            "components": comp,
            "allocated": allocated,
            "max_allocated": max_alloc,
            "reserved":  float(rec["reserved_gb"]),
            "named_sum": named_sum,
            "activations": max(0.0, allocated - named_sum),
        })

    hl_idx = phases.index(highlight_phase) if highlight_phase in phases else -1

    # Y-axis scale.
    y_max = max(H200_CAP_GB, max(r["max_allocated"] for r in rows) * 1.05)
    y_max = float(int(y_max / 20 + 1) * 20)

    def x_of(i: int) -> float:
        if n <= 1:
            return MARGIN_L + PLOT_W / 2
        return MARGIN_L + (PLOT_W * i / (n - 1))

    def y_of(v: float) -> float:
        return MARGIN_T + PLOT_H - (PLOT_H * v / y_max)

    xb = [x_of(i) for i in range(n)]

    # ------------------------------------------------------------------
    # Build the draw samples: each phase contributes a boundary sample
    # (allocated; transient = 0) and, when the window peaked above its
    # boundary, a mid-point sample carrying that window's peak composition.
    # ------------------------------------------------------------------
    def boundary_comp(i: int) -> dict[str, float]:
        c = dict(rows[i]["components"])
        c["activations"] = rows[i]["activations"]
        c["transient"] = 0.0
        return c

    def midpoint_comp(i: int) -> dict[str, float]:
        """Peak-time composition for the window ending at phase i.

        For forward / optimizer windows the live set grows monotonically, so
        the resident set at the peak equals the boundary's --- the transient
        is just per-block compute working set on top. For the backward window
        the stash both peaks and fully drains inside the window, so its
        boundary stash (~0) is NOT its peak stash: use the resident-at-peak
        and the *previous* boundary's named bands (grads have not formed yet
        at the start of backward)."""
        if i == hl_idx:
            resident = (resident_at_peak if resident_at_peak is not None
                        else rows[i - 1]["allocated"])
            # Pre-backward bands (P4: grads not yet formed), but add back the
            # grads that have accumulated by the peak instant (0 if it peaks
            # at the start of backward, ~full if it peaks at the end).
            named = dict(rows[i - 1]["components"])
            named["grads"] = rows[i]["components"].get("grads", 0.0) * (
                bwd_progress if bwd_progress is not None else 0.0)
        else:
            resident = rows[i]["allocated"]
            named = dict(rows[i]["components"])
        c = dict(named)
        c["activations"] = max(0.0, resident - sum(named.values()))
        c["transient"] = max(0.0, rows[i]["max_allocated"] - resident)
        return c

    samples: list[dict[str, Any]] = []
    for i in range(n):
        if i > 0 and rows[i]["max_allocated"] - rows[i]["allocated"] > 0.5:
            samples.append({"x": (xb[i - 1] + xb[i]) / 2,
                            "comp": midpoint_comp(i), "phase": i, "mid": True})
        samples.append({"x": xb[i], "comp": boundary_comp(i),
                        "phase": i, "mid": False})
    m = len(samples)
    sx = [s["x"] for s in samples]

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}">')
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # Title.
    parts.append(_text(W / 2, 38, title, size=22, weight="700",
                       anchor="middle"))
    parts.append(_text(W / 2, 62, subtitle, size=13, anchor="middle",
                       fill="#4b5563"))

    # Y-axis grid + ticks.
    tick_step = 20 if y_max <= 160 else 40
    v = 0
    while v <= y_max + 0.5:
        y = y_of(v)
        parts.append(f'<line x1="{MARGIN_L}" y1="{y:.1f}" '
                     f'x2="{MARGIN_L + PLOT_W}" y2="{y:.1f}" '
                     f'stroke="{COLORS["grid"]}" stroke-width="1"/>')
        parts.append(_text(MARGIN_L - 10, y + 4, f"{v:g}",
                           size=11, anchor="end", fill="#6b7280"))
        v += tick_step
    parts.append(_text(MARGIN_L - 50, MARGIN_T - 12,
                       "GB per card (per H200)", size=12, weight="600",
                       anchor="start", fill="#374151"))

    # Stacked bands over the interleaved boundary/mid-point samples.
    cum = [0.0] * m
    for key in BANDS:
        tops = [cum[j] + samples[j]["comp"].get(key, 0.0) for j in range(m)]
        pts: list[str] = []
        for j in range(m):
            pts.append(f"{sx[j]:.1f},{y_of(tops[j]):.1f}")
        for j in range(m - 1, -1, -1):
            pts.append(f"{sx[j]:.1f},{y_of(cum[j]):.1f}")
        op = "0.85" if key == "transient" else "0.95"
        parts.append(f'<polygon points="{" ".join(pts)}" '
                     f'fill="{COLORS[key]}" opacity="{op}"/>')
        cum = tops

    # H200 cap line.
    cap_y = y_of(H200_CAP_GB)
    parts.append(f'<line x1="{MARGIN_L}" y1="{cap_y:.1f}" '
                 f'x2="{MARGIN_L + PLOT_W}" y2="{cap_y:.1f}" '
                 f'stroke="{COLORS["cap"]}" stroke-width="1.4" '
                 f'stroke-dasharray="6 4"/>')
    parts.append(_text(MARGIN_L + PLOT_W - 8, cap_y - 6,
                       f"H200 cap {H200_CAP_GB:.0f} GB",
                       size=11, anchor="end", fill=COLORS["cap"],
                       weight="600"))

    # X-axis phase labels (at boundary positions).
    pretty = {
        "P0_idle":         ("P0",  "Idle"),
        "P1_inputs_done":  ("P1",  "Inputs"),
        "P2_fwd_start":    ("P2",  "Fwd start"),
        "P3_fwd_half":     ("P3",  "Fwd 1/2"),
        "P4_fwd_end":      ("P4",  "Fwd end"),
        "P5_bwd_peak":     ("P5",  "Bwd PEAK"),
        "P6_post_bwd":     ("P6",  "Post-bwd"),
        "P7_opt_done":     ("P7",  "Optimizer"),
        "P0_next":         ("P0'", "Next idle"),
    }
    axis_y = MARGIN_T + PLOT_H + 8
    for i, ph in enumerate(phases):
        short, label = pretty.get(ph, (ph, ""))
        is_peak = (ph == highlight_phase)
        fill = COLORS["transient"] if is_peak else "#374151"
        weight = "700" if is_peak else "500"
        parts.append(_text(xb[i], axis_y + 14, short,
                           size=12, anchor="middle", fill=fill,
                           weight=weight))
        parts.append(_text(xb[i], axis_y + 32, label,
                           size=11, anchor="middle", fill="#6b7280"))

    # Peak callout (at the highlight window's mid-point sample).
    peak_x, peak_v = xb[hl_idx] if hl_idx >= 0 else xb[-1], 0.0
    for s in samples:
        if s["mid"] and s["phase"] == hl_idx:
            peak_x = s["x"]
            peak_v = sum(s["comp"].values())
            break
    if peak_v <= 0 and hl_idx >= 0:
        peak_v = rows[hl_idx]["max_allocated"]
    peak_y = y_of(peak_v)
    parts.append(f'<line x1="{peak_x:.1f}" y1="{peak_y:.1f}" '
                 f'x2="{peak_x:.1f}" y2="{MARGIN_T + 20}" '
                 f'stroke="{COLORS["transient"]}" stroke-width="1" '
                 f'stroke-dasharray="3 3"/>')
    parts.append(_text(peak_x + 8, MARGIN_T + 30,
                       f"STEP PEAK · {peak_v:.1f} GB",
                       size=13, weight="700", fill=COLORS["transient"]))
    headroom = H200_CAP_GB - peak_v
    sign = "headroom" if headroom >= 0 else "OVERFLOW"
    parts.append(_text(peak_x + 8, MARGIN_T + 48,
                       f"{abs(headroom):.1f} GB {sign}",
                       size=11, fill=COLORS["transient"]))

    # Stash callouts: at fwd-end (boundary) and at the backward peak.
    def _stash_label(i: int, sample_mid: bool) -> None:
        for s in samples:
            if s["phase"] == i and s["mid"] == sample_mid:
                stash = s["comp"]["activations"]
                if stash <= 1.0:
                    return
                base = sum(s["comp"][k] for k in BANDS
                           if k not in ("activations", "transient"))
                parts.append(_text(s["x"] - 6, y_of(base + stash) - 4,
                                   f"stash {stash:.1f} GB", size=11,
                                   weight="600", anchor="end", fill="#9a3412"))
                return

    if "P4_fwd_end" in phases:
        _stash_label(phases.index("P4_fwd_end"), False)
    if hl_idx >= 0:
        _stash_label(hl_idx, True)   # resident stash at the backward peak

    # Legend.
    leg_x = MARGIN_L + PLOT_W + 22
    leg_y = MARGIN_T + 8
    parts.append(f'<rect x="{leg_x - 8}" y="{leg_y - 22}" '
                 f'width="290" height="320" rx="6" fill="#f9fafb" '
                 f'stroke="#e5e7eb" stroke-width="1"/>')
    parts.append(_text(leg_x, leg_y - 6, "Memory components (per-phase)",
                       size=12, weight="700", fill="#111827"))

    def _legend_row(y: float, color: str, name: str, value: str) -> str:
        s = []
        s.append(f'<rect x="{leg_x}" y="{y - 9}" width="16" height="12" '
                 f'fill="{color}" opacity="0.95"/>')
        s.append(_text(leg_x + 22, y, name, size=11, weight="600",
                       fill="#374151"))
        s.append(_text(leg_x + 280, y, value, size=11, anchor="end",
                       fill="#374151"))
        return "".join(s)

    # Per-band peak across all samples (boundary + mid-point) and its phase.
    def _band_max(key: str) -> tuple[float, str]:
        best, bi = 0.0, 0
        for s in samples:
            val = s["comp"].get(key, 0.0)
            if val > best:
                best, bi = val, s["phase"]
        return best, pretty.get(phases[bi], (phases[bi],))[0]

    row_y = leg_y + 16
    for key, name in LAYER_ORDER:
        mx, ph = _band_max(key)
        if mx <= 0.05:
            continue
        parts.append(_legend_row(row_y, COLORS[key], name,
                                 f"{mx:.1f} GB @ {ph}"))
        row_y += 22
    t_mx, t_ph = _band_max("transient")
    parts.append(_legend_row(row_y, COLORS["transient"],
                             "transient burst",
                             f"{t_mx:.1f} GB @ {t_ph}"))
    row_y += 22
    parts.append(_text(leg_x, row_y + 12,
                       "transient = in-window peak, drawn as a bump at each",
                       size=10, fill="#6b7280"))
    parts.append(_text(leg_x, row_y + 26,
                       "window mid-point (0 at the phase boundaries).",
                       size=10, fill="#6b7280"))

    # Bottom strip: phase memory table.
    table_y = MARGIN_T + PLOT_H + 60
    parts.append(_text(MARGIN_L, table_y,
                       "phase boundary · allocated / max_allocated (GB)",
                       size=11, weight="600", fill="#374151"))
    for i, ph in enumerate(phases):
        r = rows[i]
        parts.append(_text(xb[i], table_y + 18,
                           f"{r['allocated']:.0f}/{r['max_allocated']:.0f}",
                           size=10, anchor="middle", fill="#374151"))

    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-memory", required=True, type=Path)
    ap.add_argument("--layer-trace", type=Path, default=None,
                    help="layer_trace.csv; supplies the resident-at-peak for "
                    "the backward window so its mid-point bump splits stash "
                    "from recompute correctly")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--highlight-phase", default="P5_bwd_peak",
                    help="phase whose window holds the step peak")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rap = _resident_at_peak(args.layer_trace)
    svg = render(args.phase_memory, title=args.title,
                 subtitle=args.subtitle,
                 highlight_phase=args.highlight_phase,
                 resident_at_peak=rap[0] if rap else None,
                 bwd_progress=rap[1] if rap else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
