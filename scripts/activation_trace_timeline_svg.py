#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render an activation trace memory timeline as a standalone SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


GB = 1e9
H200_CAP_GB = 141.0


def _tensor_bytes(event: dict[str, Any]) -> int:
    tensor = event.get("tensor") or {}
    value = tensor.get("bytes", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_timeline(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    packs: dict[int, int] = {}
    unpacked: set[int] = set()
    live_saved = 0
    first_time = None
    phase_marks: list[tuple[float, str]] = []
    peak_saved = None
    peak_alloc = None
    peak_reserved = None

    for event in _iter_events(path):
        time_s = event.get("time_s")
        if not isinstance(time_s, (int, float)):
            continue
        if first_time is None:
            first_time = float(time_s)
        t = float(time_s) - first_time
        event_name = str(event.get("event"))
        phase = str(event.get("phase") or "")

        if event_name == "phase":
            payload = event.get("payload") or {}
            mark = str(payload.get("phase") or phase)
            phase_marks.append((t, mark))

        if event_name == "saved_tensor_pack":
            tensor = event.get("tensor") or {}
            event_id = tensor.get("event_id")
            if event_id is not None:
                event_id = int(event_id)
                size = _tensor_bytes(event)
                packs[event_id] = size
                live_saved += size
        elif event_name == "saved_tensor_unpack":
            tensor = event.get("tensor") or {}
            event_id = tensor.get("save_event_id")
            if event_id is not None:
                event_id = int(event_id)
                if event_id not in unpacked:
                    live_saved -= packs.get(event_id, 0)
                    unpacked.add(event_id)

        memory = event.get("cuda_memory") or {}
        allocated = int(memory.get("allocated", 0)) / GB
        reserved = int(memory.get("reserved", 0)) / GB
        row = {
            "t": t,
            "event": event_name,
            "phase": phase,
            "module_path": event.get("module_path"),
            "layer": event.get("layer"),
            "allocated": allocated,
            "reserved": reserved,
            "live_saved": live_saved / GB,
        }
        rows.append(row)

        if peak_saved is None or row["live_saved"] > peak_saved["live_saved"]:
            peak_saved = row
        if peak_alloc is None or row["allocated"] > peak_alloc["allocated"]:
            peak_alloc = row
        if peak_reserved is None or row["reserved"] > peak_reserved["reserved"]:
            peak_reserved = row

    if not rows:
        raise ValueError(f"{path}: no trace rows with cuda memory samples")

    return {
        "rows": rows,
        "phase_marks": phase_marks,
        "peak_saved": peak_saved,
        "peak_alloc": peak_alloc,
        "peak_reserved": peak_reserved,
        "initial_allocated": rows[0]["allocated"],
        "duration": rows[-1]["t"],
        "saved_pack_count": len(packs),
        "saved_unpack_count": len(unpacked),
    }


def _points(
    rows: list[dict[str, Any]],
    x,
    y,
    key: str,
) -> str:
    return " ".join(f"{x(r['t']):.1f},{y(r[key]):.1f}" for r in rows)


def _area_path(
    rows: list[dict[str, Any]],
    x,
    y,
    lower_key: str,
    upper_key: str,
) -> str:
    upper = " ".join(f"{x(r['t']):.1f},{y(r[upper_key]):.1f}" for r in rows)
    lower = " ".join(
        f"{x(r['t']):.1f},{y(r[lower_key]):.1f}" for r in reversed(rows)
    )
    return f"M {upper} L {lower} Z"


def _text(x: float, y: float, content: str, *, size: int = 14, anchor: str = "start", weight: str = "400", fill: str = "#111827") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(content)}</text>"
    )


def _callout(x: float, y: float, lines: list[str], *, width: float = 285) -> str:
    height = 28 + 18 * len(lines)
    parts = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        'rx="6" fill="#fff8dc" stroke="#c7aa5a" stroke-width="1.2"/>'
    ]
    for i, line in enumerate(lines):
        parts.append(_text(x + 14, y + 25 + 18 * i, line, size=13, fill="#342b09"))
    return "\n".join(parts)


def render_svg(trace: dict[str, Any], *, title: str, subtitle: str) -> str:
    rows = trace["rows"]
    duration = max(trace["duration"], 1e-6)
    y_max = max(
        H200_CAP_GB,
        max(r["reserved"] for r in rows) * 1.08,
        max(r["allocated"] for r in rows) * 1.18,
    )
    y_max = math.ceil(y_max / 20.0) * 20.0
    initial = trace["initial_allocated"]
    for r in rows:
        r["static"] = min(initial, r["allocated"])
        r["saved_stack"] = r["static"] + min(
            r["live_saved"],
            max(r["allocated"] - r["static"], 0.0),
        )
        r["allocated_stack"] = r["allocated"]

    width = 1700
    height = 1000
    left = 95
    right = 430
    top = 130
    bottom = 125
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x(t: float) -> float:
        return left + (t / duration) * plot_w

    def y(v: float) -> float:
        return top + plot_h - (v / y_max) * plot_h

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif}.axis{stroke:#1f2937;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.minor{stroke:#f3f4f6;stroke-width:1}.dash{stroke-dasharray:7 6}</style>',
        _text(width / 2, 52, title, size=24, anchor="middle", weight="700"),
        _text(width / 2, 78, subtitle, size=14, anchor="middle", fill="#4b5563"),
    ]

    for tick in range(0, int(y_max) + 1, 20):
        yy = y(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}"/>')
        parts.append(_text(left - 12, yy + 5, str(tick), size=12, anchor="end", fill="#4b5563"))

    for frac in [i / 10 for i in range(11)]:
        xx = left + frac * plot_w
        parts.append(f'<line class="minor" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}"/>')
        parts.append(_text(xx, top + plot_h + 28, f"{duration * frac:.0f}s", size=12, anchor="middle", fill="#4b5563"))

    cap_y = y(H200_CAP_GB)
    parts.append(f'<line class="dash" x1="{left}" y1="{cap_y:.1f}" x2="{left + plot_w}" y2="{cap_y:.1f}" stroke="#9b1c1c" stroke-width="1.4"/>')
    parts.append(_text(left + plot_w + 8, cap_y + 4, "H200 cap 141 GB", size=12, fill="#9b1c1c", weight="700"))

    for t, phase in trace["phase_marks"]:
        xx = x(t)
        parts.append(f'<line class="dash" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + plot_h}" stroke="#9ca3af" stroke-width="1"/>')
        parts.append(_text(xx + 4, top + plot_h + 52, phase, size=12, fill="#374151"))

    parts.extend([
        f'<path d="{_area_path(rows, x, y, "static", "allocated_stack")}" fill="#6b7f7e" opacity="0.48"/>',
        f'<path d="{_area_path(rows, x, y, "static", "saved_stack")}" fill="#ef4444" opacity="0.78"/>',
        f'<path d="{_area_path(rows, x, y, "allocated_stack", "reserved")}" fill="#94a3b8" opacity="0.18"/>',
        f'<polygon points="{left},{y(0):.1f} {_points(rows, x, y, "static")} {left + plot_w},{y(0):.1f}" fill="#24384a" opacity="0.95"/>',
        f'<polyline points="{_points(rows, x, y, "allocated")}" fill="none" stroke="#0f172a" stroke-width="2.2"/>',
        f'<polyline points="{_points(rows, x, y, "reserved")}" fill="none" stroke="#64748b" stroke-width="1.7" stroke-dasharray="8 6"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<text x="0" y="0" transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" '
        'font-size="14" font-weight="400" text-anchor="middle" fill="#111827">GB per rank</text>',
        _text(left + plot_w / 2, height - 32, "elapsed trace time", size=14, anchor="middle"),
    ])

    peak_saved = trace["peak_saved"]
    peak_alloc = trace["peak_alloc"]
    peak_reserved = trace["peak_reserved"]
    saved_x = x(peak_saved["t"])
    saved_y = y(peak_saved["static"] + peak_saved["live_saved"])
    alloc_x = x(peak_alloc["t"])
    alloc_y = y(peak_alloc["allocated"])
    parts.append(f'<circle cx="{saved_x:.1f}" cy="{saved_y:.1f}" r="5" fill="#b91c1c"/>')
    parts.append(f'<line x1="{saved_x:.1f}" y1="{saved_y:.1f}" x2="{saved_x - 180:.1f}" y2="{saved_y - 150:.1f}" stroke="#6b7280"/>')
    parts.append(_callout(saved_x - 420, saved_y - 205, [
        f"Saved-tensor peak: {peak_saved['live_saved']:.1f} GB",
        f"CUDA allocated here: {peak_saved['allocated']:.1f} GB",
        "all saved tensors live at loss boundary",
    ], width=320))
    parts.append(f'<circle cx="{alloc_x:.1f}" cy="{alloc_y:.1f}" r="5" fill="#0f766e"/>')
    parts.append(f'<line x1="{alloc_x:.1f}" y1="{alloc_y:.1f}" x2="{alloc_x + 120:.1f}" y2="{alloc_y - 95:.1f}" stroke="#6b7280"/>')
    parts.append(_callout(min(alloc_x + 135, left + plot_w - 330), alloc_y - 140, [
        f"CUDA allocated peak: {peak_alloc['allocated']:.1f} GB",
        f"phase={peak_alloc['phase']} layer={peak_alloc['layer']}",
        "checkpoint recompute + gradients + buffers",
    ], width=330))

    legend_x = left + plot_w + 45
    legend_y = top + 60
    parts.append(f'<rect x="{legend_x - 20}" y="{legend_y - 35}" width="350" height="335" rx="8" fill="#ffffff" stroke="#d1d5db"/>')
    parts.append(_text(legend_x, legend_y, "Memory components", size=16, weight="700"))
    legend = [
        ("#24384a", "trace-start allocated baseline", f"{initial:.1f} GB"),
        ("#ef4444", "live saved tensors", f"peaks {peak_saved['live_saved']:.1f} GB"),
        ("#6b7f7e", "other allocated/transient", "actual CUDA allocated area"),
        ("#94a3b8", "reserved but not allocated", f"peaks {peak_reserved['reserved']:.1f} GB"),
    ]
    for i, (color, label, value) in enumerate(legend):
        yy = legend_y + 38 + i * 42
        parts.append(f'<rect x="{legend_x}" y="{yy - 13}" width="18" height="18" fill="{color}" opacity="0.85"/>')
        parts.append(_text(legend_x + 30, yy, label, size=13))
        parts.append(_text(legend_x + 300, yy, value, size=13, anchor="end", fill="#4b5563"))

    parts.append(_text(legend_x, legend_y + 230, "Trace summary", size=16, weight="700"))
    summary = [
        f"packs/unpacks: {trace['saved_pack_count']} / {trace['saved_unpack_count']}",
        f"duration: {trace['duration']:.1f} s",
        f"max allocated: {peak_alloc['allocated']:.1f} GB",
        f"max reserved: {peak_reserved['reserved']:.1f} GB",
    ]
    for i, line in enumerate(summary):
        parts.append(_text(legend_x, legend_y + 258 + i * 22, line, size=13, fill="#374151"))

    parts.append(_text(left, height - 18, "Generated from FastVideo activation_trace JSONL: CUDA memory samples plus saved-tensor pack/unpack live-byte accounting.", size=12, fill="#6b7280"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="Activation trace JSONL path.")
    parser.add_argument("-o", "--output", required=True, help="Output SVG path.")
    parser.add_argument("--title", default="Wan2.1 14B Training Step - Per-Rank GPU Memory Timeline")
    parser.add_argument(
        "--subtitle",
        default="4 x H200 | 2160x3840x17 frames | N=162,000 tokens | N_local=40,500 | SP=4 | FSDP=4 | grad_ckpt=ON",
    )
    args = parser.parse_args()

    trace = _load_timeline(Path(args.trace))
    svg = render_svg(trace, title=args.title, subtitle=args.subtitle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
