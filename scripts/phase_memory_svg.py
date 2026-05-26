#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a stacked per-card GPU memory timeline SVG from phase_memory.json.

Reads the JSON written by ``fastvideo/training/memory_probe.py`` and produces
a phase-aligned area chart in the style of the H200_measured reference plot:
static layers at the bottom (Adam m+v, master fp32, bf16 weights, grads),
a separate band for the activation stash, and a red peak region for the
transient burst inside each phase (= max_allocated - allocated). Annotates
the step peak and the H200 capacity line.

Usage:
    python3 scripts/phase_memory_svg.py \\
        --phase-memory logs/mem_probe_8gpu_121f/phase_memory.json \\
        --title "Wan 2.1 14B / 121f / 8x H200 (baseline)" \\
        --out reports/figures/121f_baseline.svg
"""
from __future__ import annotations

import argparse
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


def render(phase_memory_path: Path, *, title: str, subtitle: str,
           highlight_phase: str = "P5_bwd_peak") -> str:
    data = json.loads(phase_memory_path.read_text())
    snap = data[next(iter(data))]   # first (only) step
    phases = list(snap.keys())
    n = len(phases)

    # Per-phase numbers.
    rows: list[dict[str, Any]] = []
    for ph in phases:
        rec = snap[ph]
        comp = _components(rec)
        allocated = float(rec["allocated_gb"])
        max_alloc = float(rec["max_allocated_gb"])
        reserved = float(rec["reserved_gb"])
        # Activations = whatever's allocated beyond the named buckets
        # (the saved-for-backward stash + framework residual). Clamp >= 0.
        # Grads are now read directly from grads_bf16 per phase, so they
        # no longer leak into this residual.
        named_sum = sum(comp.values())
        rows.append({
            "phase":     ph,
            "components": comp,
            "allocated": allocated,
            "max_allocated": max_alloc,
            "reserved":  reserved,
            "named_sum": named_sum,
            "activations": max(0.0, allocated - named_sum),
            "transient": max(0.0, max_alloc - allocated),
        })

    # Y-axis scale.
    y_max = max(H200_CAP_GB, max(r["max_allocated"] for r in rows) * 1.05)
    y_max = float(int(y_max / 20 + 1) * 20)

    def x_of(i: int) -> float:
        if n <= 1:
            return MARGIN_L + PLOT_W / 2
        return MARGIN_L + (PLOT_W * i / (n - 1))

    def y_of(v: float) -> float:
        return MARGIN_T + PLOT_H - (PLOT_H * v / y_max)

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

    # Stacked layers: walk phases, accumulate component areas.
    xs = [x_of(i) for i in range(n)]
    cum_bottom = [0.0] * n   # rolling top of stack so far
    for key, _ in LAYER_ORDER:
        if key == "activations":
            tops = [cum_bottom[i] + rows[i]["activations"] for i in range(n)]
        else:
            tops = [cum_bottom[i] + rows[i]["components"].get(key, 0.0)
                    for i in range(n)]
        # Build polygon (top edge then bottom edge reversed).
        pts: list[str] = []
        for i in range(n):
            pts.append(f"{xs[i]:.1f},{y_of(tops[i]):.1f}")
        for i in range(n - 1, -1, -1):
            pts.append(f"{xs[i]:.1f},{y_of(cum_bottom[i]):.1f}")
        parts.append(f'<polygon points="{" ".join(pts)}" '
                     f'fill="{COLORS[key]}" opacity="0.95"/>')
        cum_bottom = tops

    # Transient peak region: between cum_bottom (= allocated) and max_alloc.
    # Drawn as a separate filled polygon per non-zero phase span.
    transient_tops = [rows[i]["max_allocated"] for i in range(n)]
    transient_bottoms = cum_bottom[:]
    pts = []
    for i in range(n):
        pts.append(f"{xs[i]:.1f},{y_of(transient_tops[i]):.1f}")
    for i in range(n - 1, -1, -1):
        pts.append(f"{xs[i]:.1f},{y_of(transient_bottoms[i]):.1f}")
    parts.append(f'<polygon points="{" ".join(pts)}" '
                 f'fill="{COLORS["transient"]}" opacity="0.85"/>')

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

    # X-axis phase labels.
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
        parts.append(_text(xs[i], axis_y + 14, short,
                           size=12, anchor="middle", fill=fill,
                           weight=weight))
        parts.append(_text(xs[i], axis_y + 32, label,
                           size=11, anchor="middle", fill="#6b7280"))

    # Peak callout.
    peak_idx = phases.index(highlight_phase) if highlight_phase in phases else 5
    peak_x = xs[peak_idx]
    peak_v = rows[peak_idx]["max_allocated"]
    peak_y = y_of(peak_v)
    parts.append(f'<line x1="{peak_x:.1f}" y1="{peak_y:.1f}" '
                 f'x2="{peak_x:.1f}" y2="{MARGIN_T + 20}" '
                 f'stroke="{COLORS["transient"]}" stroke-width="1" '
                 f'stroke-dasharray="3 3"/>')
    parts.append(_text(peak_x + 8, MARGIN_T + 30,
                       f"STEP PEAK · {peak_v:.1f} GB",
                       size=13, weight="700",
                       fill=COLORS["transient"]))
    headroom = H200_CAP_GB - peak_v
    sign = "headroom" if headroom >= 0 else "OVERFLOW"
    parts.append(_text(peak_x + 8, MARGIN_T + 48,
                       f"{abs(headroom):.1f} GB {sign}",
                       size=11, fill=COLORS["transient"]))

    # Stash callout at fwd_end.
    p4_idx = phases.index("P4_fwd_end") if "P4_fwd_end" in phases else -1
    if p4_idx >= 0:
        stash = rows[p4_idx]["activations"]
        stash_y = y_of(rows[p4_idx]["allocated"])
        if stash > 1.0:
            parts.append(_text(xs[p4_idx] - 6, stash_y - 4,
                               f"stash {stash:.1f} GB",
                               size=11, weight="600", anchor="end",
                               fill="#9a3412"))

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
        s.append(_text(leg_x + 22, y, name, size=11,
                       weight="600", fill="#374151"))
        s.append(_text(leg_x + 280, y, value, size=11, anchor="end",
                       fill="#374151"))
        return "".join(s)

    # Show each band's PEAK value across phases and the phase it occurs.
    # A single-phase snapshot is misleading: e.g. grads are 0 at
    # P4_fwd_end (before backward) but 7.1 GB at P5-P7, and the
    # activation stash is maximal at P4. Per-band max + phase tag keeps
    # every band honest (the bands don't all peak simultaneously).
    def _band_vals(key: str) -> list[float]:
        if key == "activations":
            return [r["activations"] for r in rows]
        return [r["components"].get(key, 0.0) for r in rows]

    def _max_and_phase(vals: list[float]) -> tuple[float, str]:
        mx = max(vals)
        arg = vals.index(mx)
        return mx, pretty.get(phases[arg], (phases[arg],))[0]

    row_y = leg_y + 16
    for key, name in LAYER_ORDER:
        mx, ph = _max_and_phase(_band_vals(key))
        # Skip bands that are zero in every phase (e.g. the separate
        # fp32-master bucket when params already are the master, or a
        # skipped VAE) --- they only clutter the legend.
        if mx <= 0.05:
            continue
        parts.append(_legend_row(row_y, COLORS[key], name,
                                 f"{mx:.1f} GB @ {ph}"))
        row_y += 22
    t_mx, t_ph = _max_and_phase([r["transient"] for r in rows])
    parts.append(_legend_row(row_y, COLORS["transient"],
                             "transient burst",
                             f"{t_mx:.1f} GB @ {t_ph}"))
    row_y += 22

    parts.append(_text(leg_x, row_y + 12,
                       "(peak per band + phase; bands don't all peak together)",
                       size=10, fill="#6b7280"))

    # Bottom strip: phase memory table.
    table_y = MARGIN_T + PLOT_H + 60
    parts.append(_text(MARGIN_L, table_y,
                       "phase boundary · allocated / max_allocated / reserved (GB)",
                       size=11, weight="600", fill="#374151"))
    for i, ph in enumerate(phases):
        r = rows[i]
        parts.append(_text(xs[i], table_y + 18,
                           f"{r['allocated']:.0f}/{r['max_allocated']:.0f}",
                           size=10, anchor="middle", fill="#374151"))

    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-memory", required=True, type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--highlight-phase", default="P5_bwd_peak",
                    help="phase used for the peak line + x-axis bold")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    svg = render(args.phase_memory, title=args.title,
                 subtitle=args.subtitle,
                 highlight_phase=args.highlight_phase)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
