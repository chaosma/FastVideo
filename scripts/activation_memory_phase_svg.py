#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a phase-aligned activation memory schematic from a trace JSONL file."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


GB = 1e9
H200_CAP_GB = 141.0


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mem(event: dict[str, Any], key: str) -> float:
    return float((event.get("cuda_memory") or {}).get(key, 0)) / GB


def _tensor_bytes(event: dict[str, Any]) -> int:
    value = (event.get("tensor") or {}).get("bytes", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: str = "400",
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(content)}</text>"
    )


def _poly(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _area(
    xs: list[float],
    lower: list[float],
    upper: list[float],
    y,
    *,
    fill: str,
    opacity: float = 1.0,
) -> str:
    upper_points = [(x, y(v)) for x, v in zip(xs, upper)]
    lower_points = [(x, y(v)) for x, v in reversed(list(zip(xs, lower)))]
    return (
        f'<polygon points="{_poly(upper_points + lower_points)}" '
        f'fill="{fill}" opacity="{opacity:.2f}"/>'
    )


def _load_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    packs: dict[int, int] = {}
    unpacked: set[int] = set()
    live_saved = 0
    first_time: float | None = None

    for index, event in enumerate(_iter_events(path)):
        time_s = event.get("time_s")
        if not isinstance(time_s, (int, float)):
            continue
        if first_time is None:
            first_time = float(time_s)
        tensor = event.get("tensor") or {}
        if event.get("event") == "saved_tensor_pack":
            event_id = tensor.get("event_id")
            if event_id is not None:
                size = _tensor_bytes(event)
                packs[int(event_id)] = size
                live_saved += size
        elif event.get("event") == "saved_tensor_unpack":
            event_id = tensor.get("save_event_id")
            if event_id is not None and int(event_id) not in unpacked:
                live_saved -= packs.get(int(event_id), 0)
                unpacked.add(int(event_id))

        row = {
            "index": index,
            "event": str(event.get("event")),
            "phase": str(event.get("phase") or ""),
            "module": str(event.get("module_path") or ""),
            "layer": event.get("layer"),
            "time": float(time_s) - first_time,
            "allocated": _mem(event, "allocated"),
            "reserved": _mem(event, "reserved"),
            "live_saved": live_saved / GB,
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"{path}: no events with memory samples")

    summary = {
        "duration": rows[-1]["time"],
        "peak_alloc": max(rows, key=lambda r: r["allocated"]),
        "peak_reserved": max(rows, key=lambda r: r["reserved"]),
        "peak_saved": max(rows, key=lambda r: r["live_saved"]),
        "packs": len(packs),
        "unpacks": len(unpacked),
    }
    return rows, summary


def _first(rows: list[dict[str, Any]], pred) -> dict[str, Any] | None:
    for row in rows:
        if pred(row):
            return row
    return None


def _phase_samples(rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("P0", "Init", rows[0]),
        ("P1", "Forward start", _first(rows, lambda r: r["event"] == "phase" and r["phase"] == "forward")),
        ("P2", "Fwd block 0", _first(rows, lambda r: r["event"] == "module_forward_output" and r["phase"] == "forward" and r["module"] == "blocks.0")),
        ("P3", "Fwd block 20", _first(rows, lambda r: r["event"] == "module_forward_output" and r["phase"] == "forward" and r["module"] == "blocks.20")),
        ("P4", "Fwd block 39", _first(rows, lambda r: r["event"] == "module_forward_output" and r["phase"] == "forward" and r["module"] == "blocks.39")),
        ("P5", "Saved peak", summary["peak_saved"]),
        ("P6", "CUDA peak", summary["peak_alloc"]),
        ("P7", "Reserved peak", summary["peak_reserved"]),
        ("P8", "Trace end", rows[-1]),
    ]
    samples = []
    for phase, label, row in specs:
        if row is None:
            continue
        item = dict(row)
        item["phase_id"] = phase
        item["label"] = label
        samples.append(item)
    return samples


def render_svg(rows: list[dict[str, Any]], summary: dict[str, Any], *, title: str, subtitle: str) -> str:
    samples = _phase_samples(rows, summary)
    baseline = rows[0]["allocated"]
    for sample in samples:
        sample["baseline"] = min(baseline, sample["allocated"])
        sample["saved_top"] = sample["baseline"] + min(
            sample["live_saved"],
            max(sample["allocated"] - sample["baseline"], 0.0),
        )
        sample["allocated_top"] = sample["allocated"]
        sample["reserved_gap"] = max(sample["reserved"] - sample["allocated"], 0.0)

    y_max = max(H200_CAP_GB, max(s["reserved"] for s in samples), summary["peak_reserved"]["reserved"])
    y_max = math.ceil(y_max * 1.08 / 20.0) * 20.0

    width = 1900
    height = 980
    left = 95
    right = 590
    top = 120
    bottom = 155
    plot_w = width - left - right
    plot_h = height - top - bottom
    step = plot_w / (len(samples) - 1)
    xs = [left + i * step for i in range(len(samples))]

    def y(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    baseline = [s["baseline"] for s in samples]
    saved_top = [s["saved_top"] for s in samples]
    allocated_top = [s["allocated_top"] for s in samples]
    zero = [0.0 for _ in samples]
    reserved_points = _poly([(x, y(s["reserved"])) for x, s in zip(xs, samples)])
    allocated_points = _poly([(x, y(s["allocated"])) for x, s in zip(xs, samples)])

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif}.grid{stroke:#e5e7eb}.phase{stroke:#d1d5db;stroke-dasharray:3 5}.axis{stroke:#111827;stroke-width:1.2}</style>',
        _text(width / 2, 44, title, size=23, anchor="middle", weight="700"),
        _text(width / 2, 70, subtitle, size=13, anchor="middle", fill="#4b5563"),
    ]

    for tick in range(0, int(y_max) + 1, 20):
        yy = y(tick)
        parts.append(f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}"/>')
        parts.append(_text(left - 12, yy + 4, str(tick), size=12, anchor="end", fill="#4b5563"))

    for x, sample in zip(xs, samples):
        parts.append(f'<line class="phase" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(_text(x, top + plot_h + 28, sample["phase_id"], size=13, anchor="middle", weight="700"))
        parts.append(_text(x, top + plot_h + 48, sample["label"], size=11, anchor="middle", fill="#4b5563"))
        parts.append(_text(x, y(sample["allocated"]) - 8, f'{sample["allocated"]:.1f}', size=11, anchor="middle", weight="700"))

    cap_y = y(H200_CAP_GB)
    parts.append(f'<line x1="{left}" y1="{cap_y:.1f}" x2="{left + plot_w}" y2="{cap_y:.1f}" stroke="#b91c1c" stroke-width="1.3" stroke-dasharray="7 6"/>')
    parts.append(_text(left + plot_w + 10, cap_y + 4, "H200 cap 141 GB", size=12, fill="#b91c1c", weight="700"))

    parts.extend([
        _area(xs, zero, baseline, y, fill="#244763", opacity=0.96),
        _area(xs, baseline, saved_top, y, fill="#f59e0b", opacity=0.88),
        _area(xs, saved_top, allocated_top, y, fill="#dc2626", opacity=0.82),
        f'<polyline points="{reserved_points}" fill="none" stroke="#6b7280" stroke-width="2" stroke-dasharray="8 6"/>',
        f'<polyline points="{allocated_points}" fill="none" stroke="#111827" stroke-width="2.2"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
        f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
        f'<text x="0" y="0" transform="translate(31 {top + plot_h / 2:.1f}) rotate(-90)" font-size="14" text-anchor="middle" fill="#111827">GB per rank</text>',
        _text(left + plot_w / 2, height - 38, "phase-aligned training step schematic", size=14, anchor="middle"),
    ])

    peak_x = xs[samples.index(next(s for s in samples if s["phase_id"] == "P6"))]
    peak = summary["peak_alloc"]
    parts.append(f'<rect x="{peak_x + 22:.1f}" y="{y(peak["allocated"]) - 78:.1f}" width="315" height="96" rx="6" fill="#fff7ed" stroke="#c2410c"/>')
    parts.append(_text(peak_x + 42, y(peak["allocated"]) - 52, f'CUDA peak {peak["allocated"]:.1f} GB', size=15, weight="700", fill="#9a3412"))
    parts.append(_text(peak_x + 42, y(peak["allocated"]) - 30, f'live saved {peak["live_saved"]:.1f} GB', size=12, fill="#7c2d12"))
    parts.append(_text(peak_x + 42, y(peak["allocated"]) - 10, 'recompute + gradients + buffers', size=12, fill="#7c2d12"))

    legend_x = left + plot_w + 48
    legend_y = top + 42
    peak_other = summary["peak_alloc"]["allocated"] - rows[0]["allocated"] - summary["peak_alloc"]["live_saved"]
    peak_allocator = summary["peak_alloc"]["reserved"] - summary["peak_alloc"]["allocated"]
    max_allocator = summary["peak_reserved"]["reserved"] - summary["peak_reserved"]["allocated"]
    parts.append(f'<rect x="{legend_x - 20}" y="{legend_y - 40}" width="530" height="430" rx="7" fill="#ffffff" stroke="#d1d5db"/>')
    parts.append(_text(legend_x, legend_y, "Stacked components", size=16, weight="700"))
    legend = [
        ("#244763", "baseline: param/state + framework", f'{rows[0]["allocated"]:.1f} GB'),
        ("#f59e0b", "M_activation,saved", f'peak {summary["peak_saved"]["live_saved"]:.1f} GB'),
        ("#dc2626", "unattributed allocated transient", f'CUDA peak {peak_other:.1f} GB'),
        ("#6b7280", "reserved line / allocator slack", f'reserved peak {summary["peak_reserved"]["reserved"]:.1f} GB'),
    ]
    for i, (color, label, value) in enumerate(legend):
        yy = legend_y + 40 + i * 42
        parts.append(f'<rect x="{legend_x}" y="{yy - 15}" width="18" height="18" fill="{color}"/>')
        parts.append(_text(legend_x + 30, yy, label, size=13))
        parts.append(_text(legend_x + 490, yy, value, size=13, anchor="end", fill="#4b5563"))

    parts.append(_text(legend_x, legend_y + 210, "At CUDA allocated peak", size=16, weight="700"))
    peak_lines = [
        f'baseline measured: {rows[0]["allocated"]:.1f} GB',
        f'saved activations: {summary["peak_alloc"]["live_saved"]:.1f} GB',
        f'recompute/grad/FSDP/SP/workspace: {peak_other:.1f} GB',
        f'allocator slack at peak: {peak_allocator:.1f} GB',
        f'max allocator slack observed: {max_allocator:.1f} GB',
    ]
    for i, line in enumerate(peak_lines):
        parts.append(_text(legend_x, legend_y + 238 + i * 20, line, size=12, fill="#374151"))

    parts.append(_text(legend_x, legend_y + 356, "Trace facts", size=16, weight="700"))
    parts.append(_text(legend_x, legend_y + 382, f'packs/unpacks {summary["packs"]}/{summary["unpacks"]}', size=12, fill="#374151"))
    parts.append(_text(legend_x + 160, legend_y + 382, f'duration {summary["duration"]:.1f} s', size=12, fill="#374151"))

    parts.append(_text(left, height - 14, "Generated from FastVideo activation trace JSONL. Equal phase spacing is schematic; the red region is measured as a lump and needs torch memory profiler/NCCL detail to split.", size=12, fill="#6b7280"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="Activation trace JSONL path.")
    parser.add_argument("-o", "--output", required=True, help="Output SVG path.")
    parser.add_argument("--title", default="Wan2.1 14B Training Step - Per-Rank GPU Memory Schematic")
    parser.add_argument(
        "--subtitle",
        default="8 x H200 | 2160x3840x121 frames | N=1,004,400 | N_local=125,550 | SP=8 | FSDP=8 | grad_ckpt=ON",
    )
    args = parser.parse_args()

    rows, summary = _load_samples(Path(args.trace))
    svg = render_svg(rows, summary, title=args.title, subtitle=args.subtitle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
