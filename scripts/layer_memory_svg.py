#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a per-block GPU-memory curve from the memory probe's layer_trace.csv.

Where phase_memory_svg.py gives the coarse 9-phase view, this plots the
fine-grained ramp: 40 transformer blocks × {fwd_pre, fwd_post, bwd_pre,
bwd_post} = 160 samples per step, in chronological order (forward
blocks 0→39, then backward blocks 39→0). Shows ``allocated`` as a filled
area and ``max_allocated`` as the peak envelope on top, so you can see
exactly where in the step the high-water mark lands.

Usage:
    python3 scripts/layer_memory_svg.py \\
        --layer-trace logs/mem_probe_8gpu_121f/layer_trace.csv \\
        --title "Wan 2.1 14B / 121f / 8x H200 (baseline)" \\
        --out reports/figures/121f_layer_baseline.svg
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

W, H = 1500, 700
MARGIN_L, MARGIN_R = 70, 40
MARGIN_T, MARGIN_B = 80, 70
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B
H200_CAP_GB = 141.0

C = {
    "allocated": "#1d4ed8",   # blue line (resident outline)
    "persistent":"#475569",   # slate fill (params + optimizer floor)
    "stash":     "#fb923c",   # orange fill (activation stash = alloc - floor)
    "maxalloc":  "#dc2626",   # red line (peak envelope)
    "fwd_bg":    "#eff6ff",   # pale blue forward region
    "bwd_bg":    "#fef2f2",   # pale red backward region
    "grid":      "#e5e7eb",
    "cap":       "#dc2626",
    "axis":      "#374151",
}


def _floor_gb(phase_memory: Path | None) -> float:
    """Constant params+optimizer GB, read from phase_memory.json categories.

    This is the resident floor below the activation stash (it does not vary
    across the step). Returns 0.0 when no phase_memory is supplied, in which
    case the allocated area is drawn as a single band (legacy behaviour)."""
    if phase_memory is None or not phase_memory.exists():
        return 0.0
    data = json.loads(phase_memory.read_text())
    snap = data[next(iter(data))]
    rec = snap.get("P2_fwd_start") or snap[next(iter(snap))]
    c = rec.get("categories_gb") or {}
    return (float(c.get("adam_m", 0)) + float(c.get("adam_v", 0)) +
            float(c.get("master_fp32", 0)) + float(c.get("weights_bf16", 0)))


def _text(x, y, s, *, size=12, anchor="start", weight="400", fill="#111827"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" '
            f'font-family="-apple-system,Helvetica,Arial,sans-serif">'
            f'{html.escape(s)}</text>')


def _load(path: Path, step: str | None):
    rows = [r for r in csv.DictReader(path.open())]
    if step is None:
        step = rows[0]["step"]
    rows = [r for r in rows if r["step"] == step]
    rows.sort(key=lambda r: int(r["t_ns"]))   # chronological
    return rows, step


def render(path: Path, *, title: str, subtitle: str, step: str | None,
           floor: float = 0.0) -> str:
    rows, step = _load(path, step)
    n = len(rows)
    if n == 0:
        raise SystemExit("no rows for that step")

    alloc = [float(r["allocated_gb"]) for r in rows]
    maxa = [float(r["max_allocated_gb"]) for r in rows]
    is_fwd = [r["phase"].startswith("fwd") for r in rows]

    # First backward index (the fwd→bwd transition).
    trans_i = next((i for i, f in enumerate(is_fwd) if not f), n)

    y_max = max(H200_CAP_GB, max(maxa) * 1.05)
    y_max = float(int(y_max / 20 + 1) * 20)

    def x_of(i):
        return MARGIN_L + (PLOT_W * i / max(1, n - 1))

    def y_of(v):
        return MARGIN_T + PLOT_H - PLOT_H * v / y_max

    p: list[str] = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}">')
    p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(_text(W / 2, 34, title, size=20, weight="700", anchor="middle"))
    p.append(_text(W / 2, 56, subtitle, size=12, anchor="middle",
                   fill="#4b5563"))

    # Forward / backward background bands.
    trans_x = x_of(trans_i)
    p.append(f'<rect x="{MARGIN_L}" y="{MARGIN_T}" '
             f'width="{trans_x - MARGIN_L:.1f}" height="{PLOT_H}" '
             f'fill="{C["fwd_bg"]}"/>')
    p.append(f'<rect x="{trans_x:.1f}" y="{MARGIN_T}" '
             f'width="{MARGIN_L + PLOT_W - trans_x:.1f}" height="{PLOT_H}" '
             f'fill="{C["bwd_bg"]}"/>')

    # Y grid + ticks.
    v = 0
    while v <= y_max + 0.5:
        y = y_of(v)
        p.append(f'<line x1="{MARGIN_L}" y1="{y:.1f}" '
                 f'x2="{MARGIN_L + PLOT_W}" y2="{y:.1f}" '
                 f'stroke="{C["grid"]}" stroke-width="1"/>')
        p.append(_text(MARGIN_L - 8, y + 4, f"{v:g}", size=11, anchor="end",
                       fill="#6b7280"))
        v += 20
    p.append(_text(MARGIN_L - 50, MARGIN_T - 12, "GB per card", size=12,
                   weight="600", fill="#374151"))

    # allocated filled area, split at the persistent floor so the activation
    # stash (everything saved-for-backward above params+optimizer) reads as
    # its own band. The stash top traces allocated(t) exactly per block, so
    # its true shape shows: a triangle for the baseline (ramps up over the 40
    # forward blocks, drains over the 40 backward blocks) vs a flat sliver for
    # offload (each block's stash is shipped to CPU as it is produced).
    fl = min(floor, min(alloc)) if floor > 0 else 0.0
    if fl > 0:
        p.append(f'<polygon points="{MARGIN_L:.1f},{y_of(0):.1f} '
                 f'{MARGIN_L + PLOT_W:.1f},{y_of(0):.1f} '
                 f'{MARGIN_L + PLOT_W:.1f},{y_of(fl):.1f} '
                 f'{MARGIN_L:.1f},{y_of(fl):.1f}" '
                 f'fill="{C["persistent"]}" opacity="0.95"/>')
    stash_top = [f"{x_of(i):.1f},{y_of(alloc[i]):.1f}" for i in range(n)]
    p.append(f'<polygon points="{" ".join(stash_top)} '
             f'{x_of(n - 1):.1f},{y_of(fl):.1f} {x_of(0):.1f},{y_of(fl):.1f}" '
             f'fill="{C["stash"]}" opacity="0.95"/>')
    # allocated line.
    line = " ".join(f"{x_of(i):.1f},{y_of(alloc[i]):.1f}" for i in range(n))
    p.append(f'<polyline points="{line}" fill="none" '
             f'stroke="{C["allocated"]}" stroke-width="1.6"/>')
    # max_allocated peak envelope.
    line = " ".join(f"{x_of(i):.1f},{y_of(maxa[i]):.1f}" for i in range(n))
    p.append(f'<polyline points="{line}" fill="none" '
             f'stroke="{C["maxalloc"]}" stroke-width="1.6" '
             f'stroke-dasharray="4 2"/>')

    # H200 cap.
    cy = y_of(H200_CAP_GB)
    p.append(f'<line x1="{MARGIN_L}" y1="{cy:.1f}" x2="{MARGIN_L + PLOT_W}" '
             f'y2="{cy:.1f}" stroke="{C["cap"]}" stroke-width="1.4" '
             f'stroke-dasharray="6 4"/>')
    p.append(_text(MARGIN_L + PLOT_W - 6, cy - 6, f"H200 cap {H200_CAP_GB:.0f} GB",
                   size=11, anchor="end", fill=C["cap"], weight="600"))

    # fwd/bwd divider + labels.
    p.append(f'<line x1="{trans_x:.1f}" y1="{MARGIN_T}" x2="{trans_x:.1f}" '
             f'y2="{MARGIN_T + PLOT_H}" stroke="#9ca3af" stroke-width="1.2" '
             f'stroke-dasharray="3 3"/>')
    p.append(_text((MARGIN_L + trans_x) / 2, MARGIN_T + 18, "FORWARD (blk 0→39)",
                   size=12, weight="700", anchor="middle", fill="#1d4ed8"))
    p.append(_text((trans_x + MARGIN_L + PLOT_W) / 2, MARGIN_T + 18,
                   "BACKWARD (blk 39→0)", size=12, weight="700",
                   anchor="middle", fill="#b91c1c"))

    # Peak marker.
    peak_i = max(range(n), key=lambda i: maxa[i])
    px, py = x_of(peak_i), y_of(maxa[peak_i])
    p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
             f'fill="{C["maxalloc"]}"/>')
    p.append(_text(px + 8, py + 4,
                   f"peak {maxa[peak_i]:.1f} GB  ({rows[peak_i]['phase']} "
                   f"blk {rows[peak_i]['layer_idx']})",
                   size=12, weight="700", fill=C["maxalloc"]))

    # X ticks: every 5th block boundary, labelled by block idx.
    for i in range(0, n, 10):
        x = x_of(i)
        p.append(f'<line x1="{x:.1f}" y1="{MARGIN_T + PLOT_H}" x2="{x:.1f}" '
                 f'y2="{MARGIN_T + PLOT_H + 5}" stroke="{C["axis"]}" '
                 f'stroke-width="1"/>')
        p.append(_text(x, MARGIN_T + PLOT_H + 18,
                       f"blk{rows[i]['layer_idx']}", size=10, anchor="middle",
                       fill="#6b7280"))
    p.append(_text(MARGIN_L + PLOT_W / 2, H - 14,
                   "chronological per-block events  (fwd_pre/fwd_post then "
                   "bwd_pre/bwd_post)", size=11, anchor="middle",
                   fill="#6b7280"))

    # Legend.
    lx, ly = MARGIN_L + 12, MARGIN_T + 40
    if fl > 0:
        p.append(f'<rect x="{lx}" y="{ly - 9}" width="24" height="11" '
                 f'fill="{C["stash"]}" opacity="0.95"/>')
        p.append(_text(lx + 30, ly + 1, "activation stash (live)", size=11,
                       fill="#374151"))
        p.append(f'<rect x="{lx}" y="{ly + 9}" width="24" height="11" '
                 f'fill="{C["persistent"]}" opacity="0.95"/>')
        p.append(_text(lx + 30, ly + 19,
                       f"params + optimizer ({fl:.1f} GB)", size=11,
                       fill="#374151"))
        ly += 36
    else:
        p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" '
                 f'stroke="{C["allocated"]}" stroke-width="2"/>')
        p.append(_text(lx + 30, ly + 4, "allocated (resident)", size=11,
                       fill="#374151"))
        ly += 18
    p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" '
             f'stroke="{C["maxalloc"]}" stroke-width="2" '
             f'stroke-dasharray="4 2"/>')
    p.append(_text(lx + 30, ly + 4, "max_allocated (high-water; per-block "
                   "bwd transient not resolved)", size=11, fill="#374151"))

    p.append('</svg>')
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-trace", required=True, type=Path)
    ap.add_argument("--phase-memory", type=Path, default=None,
                    help="phase_memory.json; supplies the params+optimizer "
                    "floor so the stash is drawn as its own band")
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--step", default=None,
                    help="step index to plot (default: first in file)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    svg = render(args.layer_trace, title=args.title, subtitle=args.subtitle,
                 step=args.step, floor=_floor_gb(args.phase_memory))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
