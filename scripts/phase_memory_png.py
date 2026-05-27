#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a stacked per-card GPU memory timeline from phase_memory.json.

Supports SVG (vector, no dependencies) and PNG (2x supersampled via Pillow).
Format is auto-detected from the --out extension.

Named bands and the activation stash use boundary-only vertices, so they draw
as straight lines between phase boundaries. The transient burst is a separate
polygon with one mid-point bump per window; the bump base = linear interpolation
of allocated between the two flanking boundaries.

Transient burst in the legend = max_allocated - left_boundary_allocated, i.e.
the true extra allocation above the window start state.

Usage (SVG):
    python3 scripts/phase_memory_png.py \\
        --phase-memory logs/mem_probe_8gpu_121f/phase_memory.json \\
        --title "..." --out reports/figures/121f_baseline.svg

Usage (PNG, requires pillow >= 10.1):
    uv run --with pillow scripts/phase_memory_png.py \\
        --phase-memory logs/mem_probe_8gpu_121f/phase_memory.json \\
        --title "..." --out reports/figures/121f_baseline.png
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
W, H = 1300, 760
MARGIN_L, MARGIN_R = 70, 320
MARGIN_T, MARGIN_B = 90, 90
PLOT_W = W - MARGIN_L - MARGIN_R
PLOT_H = H - MARGIN_T - MARGIN_B

H200_CAP_GB = 141.0

# Colors as (R, G, B) tuples.
C = {
    "adam":        (0x1e, 0x29, 0x3b),
    "master":      (0x33, 0x41, 0x55),
    "weights":     (0x47, 0x55, 0x69),
    "grads":       (0x94, 0xa3, 0xb8),
    "vae":         (0xa3, 0xa3, 0xa3),
    "activations": (0xfb, 0x92, 0x3c),
    "transient":   (0xdc, 0x26, 0x26),
    "grid":        (0xe5, 0xe7, 0xeb),
    "cap":         (0xdc, 0x26, 0x26),
    "txt_dark":    (0x11, 0x18, 0x27),
    "txt_mid":     (0x37, 0x41, 0x51),
    "txt_gray":    (0x6b, 0x72, 0x80),
    "txt_blue":    (0x4b, 0x55, 0x63),
    "txt_red":     (0xdc, 0x26, 0x26),
    "txt_stash":   (0x9a, 0x34, 0x12),
    "leg_bg":      (0xf9, 0xfa, 0xfb),
    "leg_border":  (0xe5, 0xe7, 0xeb),
}

LAYER_ORDER = [
    ("adam",        "Adam m+v (fp32)"),
    ("weights",     "params (fp32 master)"),
    ("master",      "fp32 master copy"),
    ("grads",       "grads (fp32)"),
    ("vae",         "VAE"),
    ("activations", "activation stash"),
]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _chex(key: str) -> str:
    r, g, b = C[key]
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgba(key: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    r, g, b = C[key]
    return (r, g, b, int(opacity * 255))


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _components(rec: dict[str, Any]) -> dict[str, float]:
    cats = rec.get("categories_gb") or {}
    return {
        "adam":    float(cats.get("adam_m", 0.0)) + float(cats.get("adam_v", 0.0)),
        "master":  float(cats.get("master_fp32", 0.0)),
        "weights": float(cats.get("weights_bf16", 0.0)),
        "grads":   float(cats.get("grads_bf16", 0.0)),
        "vae":     float(cats.get("vae", 0.0)),
    }


def _prepare(phase_memory_path: Path, *,
             highlight_phase: str) -> dict[str, Any]:
    """Load JSON and compute all derived geometry."""
    data = json.loads(phase_memory_path.read_text())
    snap = data[next(iter(data))]
    phases = list(snap.keys())
    n = len(phases)

    rows: list[dict[str, Any]] = []
    for ph in phases:
        rec = snap[ph]
        comp = _components(rec)
        allocated = float(rec["allocated_gb"])
        max_alloc = float(rec["max_allocated_gb"])
        named_sum = sum(comp.values())
        rows.append({
            "phase":        ph,
            "components":   comp,
            "allocated":    allocated,
            "max_allocated": max_alloc,
            "named_sum":    named_sum,
            "activations":  max(0.0, allocated - named_sum),
        })

    hl_idx = phases.index(highlight_phase) if highlight_phase in phases else -1

    y_max = max(H200_CAP_GB, max(r["max_allocated"] for r in rows) * 1.05)
    y_max = float(int(y_max / 20 + 1) * 20)

    def x_of(i: int) -> float:
        return (MARGIN_L + PLOT_W * i / (n - 1)) if n > 1 else MARGIN_L + PLOT_W / 2

    def y_of(v: float) -> float:
        return MARGIN_T + PLOT_H - PLOT_H * v / y_max

    bx = [x_of(i) for i in range(n)]   # boundary x-coords

    # Transient samples: boundary points (transient=0) + midpoints (bumps).
    t_samp: list[dict[str, Any]] = []
    for i in range(n):
        if i > 0 and rows[i]["max_allocated"] - rows[i]["allocated"] > 0.5:
            mid_x = (bx[i - 1] + bx[i]) / 2
            interp_alloc = (rows[i - 1]["allocated"] + rows[i]["allocated"]) / 2
            t_samp.append({
                "x":     mid_x,
                "base":  interp_alloc,            # bottom of the bump
                "top":   rows[i]["max_allocated"], # top of the bump
                "phase": i,
                "mid":   True,
            })
        t_samp.append({
            "x":     bx[i],
            "base":  rows[i]["allocated"],
            "top":   rows[i]["allocated"],
            "phase": i,
            "mid":   False,
        })

    # Transient burst per window = max_allocated - left_boundary_allocated.
    def transient_max() -> tuple[float, str]:
        pretty_short = {
            "P0_idle": "P0", "P1_inputs_done": "P1", "P2_fwd_start": "P2",
            "P3_fwd_half": "P3", "P4_fwd_end": "P4", "P5_bwd_peak": "P5",
            "P6_post_bwd": "P6", "P7_opt_done": "P7", "P0_next": "P0'",
        }
        best, bp = 0.0, ""
        for s in t_samp:
            if s["mid"]:
                val = s["top"] - rows[s["phase"] - 1]["allocated"]
                if val > best:
                    best = val
                    bp = pretty_short.get(phases[s["phase"]], phases[s["phase"]])
        return best, bp

    return dict(rows=rows, t_samp=t_samp, bx=bx, n=n, phases=phases,
                hl_idx=hl_idx, y_max=y_max, x_of=x_of, y_of=y_of,
                transient_max=transient_max)


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------

def _to_svg(ctx: dict[str, Any], *, title: str, subtitle: str) -> str:
    rows      = ctx["rows"]
    t_samp    = ctx["t_samp"]
    bx        = ctx["bx"]
    n         = ctx["n"]
    phases    = ctx["phases"]
    hl_idx    = ctx["hl_idx"]
    y_max     = ctx["y_max"]
    y_of      = ctx["y_of"]
    tmx       = ctx["transient_max"]

    def _e(s: str) -> str:
        return html.escape(s)

    def _t(x: float, y: float, content: str, *, size: int = 12,
           anchor: str = "start", weight: str = "400",
           fill: str = "#374151") -> str:
        return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                f'font-weight="{weight}" text-anchor="{anchor}" '
                f'fill="{fill}" font-family="ui-sans-serif,system-ui,'
                f'Helvetica,Arial,sans-serif">{_e(content)}</text>')

    def _poly(pts_top: list[tuple], pts_bot: list[tuple],
              color_key: str, opacity: str) -> str:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts_top)
        pts += " " + " ".join(f"{x:.1f},{y:.1f}" for x, y in reversed(pts_bot))
        return (f'<polygon points="{pts}" fill="{_chex(color_key)}" '
                f'opacity="{opacity}"/>')

    pretty = {
        "P0_idle": ("P0", "Idle"), "P1_inputs_done": ("P1", "Inputs"),
        "P2_fwd_start": ("P2", "Fwd start"), "P3_fwd_half": ("P3", "Fwd 1/2"),
        "P4_fwd_end": ("P4", "Fwd end"), "P5_bwd_peak": ("P5", "Bwd PEAK"),
        "P6_post_bwd": ("P6", "Post-bwd"), "P7_opt_done": ("P7", "Optimizer"),
        "P0_next": ("P0'", "Next idle"),
    }

    p: list[str] = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    p.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # Title + subtitle
    p.append(_t(W / 2, 38, title, size=22, anchor="middle", weight="700",
                fill=_chex("txt_dark")))
    p.append(_t(W / 2, 62, subtitle, size=13, anchor="middle",
                fill=_chex("txt_blue")))

    # Grid + Y-axis
    tick_step = 20 if y_max <= 160 else 40
    v = 0
    while v <= y_max + 0.5:
        gy = y_of(v)
        p.append(f'<line x1="{MARGIN_L}" y1="{gy:.1f}" '
                 f'x2="{MARGIN_L + PLOT_W}" y2="{gy:.1f}" '
                 f'stroke="{_chex("grid")}" stroke-width="1"/>')
        p.append(_t(MARGIN_L - 10, gy + 4, f"{v:g}", size=11,
                    anchor="end", fill=_chex("txt_gray")))
        v += tick_step
    p.append(_t(MARGIN_L - 50, MARGIN_T - 12, "GB per card (per H200)",
                size=12, weight="600", fill=_chex("txt_mid")))

    # Named bands (boundary-only vertices)
    named_keys = ["adam", "master", "weights", "grads", "vae"]
    cum = [0.0] * n
    for key in named_keys:
        tops = [cum[j] + rows[j]["components"].get(key, 0.0) for j in range(n)]
        p.append(_poly([(bx[j], y_of(tops[j])) for j in range(n)],
                       [(bx[j], y_of(cum[j])) for j in range(n)],
                       key, "0.95"))
        cum = tops

    # Stash band (boundary-only, top = allocated, bottom = named_sum)
    p.append(_poly([(bx[j], y_of(rows[j]["allocated"])) for j in range(n)],
                   [(bx[j], y_of(cum[j])) for j in range(n)],
                   "activations", "0.95"))

    # Transient bumps (boundary + midpoints)
    p.append(_poly([(s["x"], y_of(s["top"])) for s in t_samp],
                   [(s["x"], y_of(s["base"])) for s in t_samp],
                   "transient", "0.85"))

    # H200 cap dashed line
    cap_y = y_of(H200_CAP_GB)
    p.append(f'<line x1="{MARGIN_L}" y1="{cap_y:.1f}" '
             f'x2="{MARGIN_L + PLOT_W}" y2="{cap_y:.1f}" '
             f'stroke="{_chex("cap")}" stroke-width="1.4" '
             f'stroke-dasharray="6 4"/>')
    p.append(_t(MARGIN_L + PLOT_W - 8, cap_y - 6,
                f"H200 cap {H200_CAP_GB:.0f} GB",
                size=11, anchor="end", weight="600", fill=_chex("cap")))

    # X-axis phase labels
    axis_y = MARGIN_T + PLOT_H + 8
    for i, ph in enumerate(phases):
        short, label = pretty.get(ph, (ph, ""))
        is_peak = (i == hl_idx)
        fc = _chex("txt_red") if is_peak else _chex("txt_mid")
        fw = "700" if is_peak else "500"
        p.append(_t(bx[i], axis_y + 14, short, size=12,
                    anchor="middle", fill=fc, weight=fw))
        p.append(_t(bx[i], axis_y + 32, label, size=11,
                    anchor="middle", fill=_chex("txt_gray")))

    # Peak callout
    peak_x = bx[hl_idx] if hl_idx >= 0 else bx[-1]
    peak_v = rows[hl_idx]["max_allocated"] if hl_idx >= 0 else 0.0
    for s in t_samp:
        if s["mid"] and s["phase"] == hl_idx:
            peak_x, peak_v = s["x"], s["top"]
            break
    peak_y = y_of(peak_v)
    p.append(f'<line x1="{peak_x:.1f}" y1="{peak_y:.1f}" '
             f'x2="{peak_x:.1f}" y2="{MARGIN_T + 20}" '
             f'stroke="{_chex("transient")}" stroke-width="1" '
             f'stroke-dasharray="3 3"/>')
    p.append(_t(peak_x + 8, MARGIN_T + 30,
                f"STEP PEAK · {peak_v:.1f} GB",
                size=13, weight="700", fill=_chex("txt_red")))
    headroom = H200_CAP_GB - peak_v
    sign = "headroom" if headroom >= 0 else "OVERFLOW"
    p.append(_t(peak_x + 8, MARGIN_T + 48,
                f"{abs(headroom):.1f} GB {sign}",
                size=11, fill=_chex("txt_red")))

    # Stash callout at P4 boundary
    if "P4_fwd_end" in phases:
        p4 = phases.index("P4_fwd_end")
        stash = rows[p4]["activations"]
        if stash > 1.0:
            p.append(_t(bx[p4] - 6, y_of(rows[p4]["allocated"]) - 4,
                        f"stash {stash:.1f} GB", size=11, weight="600",
                        anchor="end", fill=_chex("txt_stash")))

    # Legend
    leg_x = MARGIN_L + PLOT_W + 22
    leg_y = MARGIN_T + 8
    p.append(f'<rect x="{leg_x - 8}" y="{leg_y - 22}" '
             f'width="290" height="310" rx="6" fill="#f9fafb" '
             f'stroke="#e5e7eb" stroke-width="1"/>')
    p.append(_t(leg_x, leg_y - 6, "Memory components (per-phase)",
                size=12, weight="700", fill=_chex("txt_dark")))

    def _leg_row(y: float, color: str, name: str, value: str) -> str:
        out = [f'<rect x="{leg_x}" y="{y - 9}" width="16" height="12" '
               f'fill="{color}" opacity="0.95"/>']
        out.append(_t(leg_x + 22, y, name, size=11, weight="600",
                      fill=_chex("txt_mid")))
        out.append(_t(leg_x + 280, y, value, size=11, anchor="end",
                      fill=_chex("txt_mid")))
        return "".join(out)

    def _named_max(key: str) -> tuple[float, str]:
        best, bi = 0.0, 0
        for j in range(n):
            val = rows[j]["components"].get(key, 0.0)
            if val > best:
                best, bi = val, j
        return best, pretty.get(phases[bi], (phases[bi],))[0]

    row_y = leg_y + 16
    for key, name in LAYER_ORDER:
        if key == "activations":
            mx = max(rows[j]["activations"] for j in range(n))
            bi = max(range(n), key=lambda j: rows[j]["activations"])
            ph_label = pretty.get(phases[bi], (phases[bi],))[0]
        else:
            mx, ph_label = _named_max(key)
        if mx <= 0.05:
            continue
        p.append(_leg_row(row_y, _chex(key), name,
                          f"{mx:.1f} GB @ {ph_label}"))
        row_y += 22

    t_mx, t_ph = tmx()
    p.append(_leg_row(row_y, _chex("transient"), "transient burst",
                      f"{t_mx:.1f} GB @ {t_ph}"))
    row_y += 22
    p.append(_t(leg_x, row_y + 12,
                "transient = max_allocated - left_boundary_allocated,",
                size=10, fill=_chex("txt_gray")))
    p.append(_t(leg_x, row_y + 26,
                "drawn as a bump at each window mid-point.",
                size=10, fill=_chex("txt_gray")))

    # Bottom table
    table_y = MARGIN_T + PLOT_H + 60
    p.append(_t(MARGIN_L, table_y,
                "phase boundary · allocated / max_allocated (GB)",
                size=11, weight="600", fill=_chex("txt_mid")))
    for i in range(n):
        r = rows[i]
        p.append(_t(bx[i], table_y + 18,
                    f"{r['allocated']:.0f}/{r['max_allocated']:.0f}",
                    size=10, anchor="middle", fill=_chex("txt_mid")))

    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------------------
# PNG renderer (2x supersampled)
# ---------------------------------------------------------------------------

def _to_png(ctx: dict[str, Any], *, title: str, subtitle: str,
            scale: int = 2):
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    rows      = ctx["rows"]
    t_samp    = ctx["t_samp"]
    bx        = ctx["bx"]
    n         = ctx["n"]
    phases    = ctx["phases"]
    hl_idx    = ctx["hl_idx"]
    y_max     = ctx["y_max"]
    y_of_base = ctx["y_of"]
    tmx       = ctx["transient_max"]

    S = scale
    # Scaled layout helpers
    def _x(v: float) -> float: return v * S
    def _y(v: float) -> float: return y_of_base(v) * S
    def _f(size: int) -> Any: return ImageFont.load_default(size=size * S)

    def _dpoly(img: Image.Image, pts: list[tuple],
               color_key: str, opacity: float) -> Image.Image:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).polygon(pts, fill=_rgba(color_key, opacity))
        return Image.alpha_composite(img, ov)

    def _dash(draw: ImageDraw.ImageDraw, x1: float, y1: float,
              x2: float, y2: float, fill: tuple,
              width: int = 1, dash: tuple = (6, 4)) -> None:
        on, off = dash[0] * S, dash[1] * S
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length:
            end = min(pos + on, length)
            draw.line([(x1 + dx * pos, y1 + dy * pos),
                       (x1 + dx * end, y1 + dy * end)],
                      fill=fill, width=width)
            pos += on + off

    pretty = {
        "P0_idle": ("P0", "Idle"), "P1_inputs_done": ("P1", "Inputs"),
        "P2_fwd_start": ("P2", "Fwd start"), "P3_fwd_half": ("P3", "Fwd 1/2"),
        "P4_fwd_end": ("P4", "Fwd end"), "P5_bwd_peak": ("P5", "Bwd PEAK"),
        "P6_post_bwd": ("P6", "Post-bwd"), "P7_opt_done": ("P7", "Optimizer"),
        "P0_next": ("P0'", "Next idle"),
    }

    img = Image.new("RGBA", (W * S, H * S), (255, 255, 255, 255))

    # Grid
    gl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(gl)
    tick_step = 20 if y_max <= 160 else 40
    v = 0
    while v <= y_max + 0.5:
        gy = _y(v)
        gd.line([(_x(MARGIN_L), gy), (_x(MARGIN_L + PLOT_W), gy)],
                fill=(*C["grid"], 255), width=S)
        v += tick_step
    img = Image.alpha_composite(img, gl)

    # Named bands
    named_keys = ["adam", "master", "weights", "grads", "vae"]
    cum = [0.0] * n
    for key in named_keys:
        tops = [cum[j] + rows[j]["components"].get(key, 0.0) for j in range(n)]
        pts = ([(_x(bx[j]), _y(tops[j])) for j in range(n)] +
               [(_x(bx[j]), _y(cum[j])) for j in range(n - 1, -1, -1)])
        img = _dpoly(img, pts, key, 0.95)
        cum = tops

    # Stash band
    pts = ([(_x(bx[j]), _y(rows[j]["allocated"])) for j in range(n)] +
           [(_x(bx[j]), _y(cum[j])) for j in range(n - 1, -1, -1)])
    img = _dpoly(img, pts, "activations", 0.95)

    # Transient bumps
    pts = ([(_x(s["x"]), _y(s["top"])) for s in t_samp] +
           [(_x(s["x"]), _y(s["base"])) for s in reversed(t_samp)])
    img = _dpoly(img, pts, "transient", 0.85)

    draw = ImageDraw.Draw(img)

    # H200 cap
    cap_y = _y(H200_CAP_GB)
    cap_c = (*C["cap"], 255)
    _dash(draw, _x(MARGIN_L), cap_y, _x(MARGIN_L + PLOT_W), cap_y,
          fill=cap_c, width=2 * S)
    draw.text((_x(MARGIN_L + PLOT_W - 8), cap_y - 8 * S),
              f"H200 cap {H200_CAP_GB:.0f} GB",
              fill=cap_c, font=_f(10), anchor="rs")

    # Title + subtitle (margin space, use pixel coords * S)
    draw.text((_x(W / 2), 36 * S), title,
              fill=(*C["txt_dark"], 255), font=_f(19), anchor="ms")
    draw.text((_x(W / 2), 60 * S), subtitle,
              fill=(*C["txt_blue"], 255), font=_f(12), anchor="ms")

    draw.text((_x(MARGIN_L - 50), _x(MARGIN_T - 18)),
              "GB per card (per H200)",
              fill=(*C["txt_mid"], 255), font=_f(11), anchor="ls")

    v = 0
    while v <= y_max + 0.5:
        draw.text((_x(MARGIN_L - 8), _y(v)),
                  f"{v:g}", fill=(*C["txt_gray"], 255),
                  font=_f(10), anchor="rs")
        v += tick_step

    # X-axis labels
    axis_y = _y(0) + 8 * S
    for i, ph in enumerate(phases):
        short, label = pretty.get(ph, (ph, ""))
        is_peak = (i == hl_idx)
        fc = (*C["txt_red"], 255) if is_peak else (*C["txt_mid"], 255)
        draw.text((_x(bx[i]), axis_y + 14 * S), short, fill=fc,
                  font=_f(12), anchor="ms")
        draw.text((_x(bx[i]), axis_y + 30 * S), label,
                  fill=(*C["txt_gray"], 255), font=_f(10), anchor="ms")

    # Peak callout
    peak_x = _x(bx[hl_idx]) if hl_idx >= 0 else _x(bx[-1])
    peak_v = rows[hl_idx]["max_allocated"] if hl_idx >= 0 else 0.0
    for s in t_samp:
        if s["mid"] and s["phase"] == hl_idx:
            peak_x, peak_v = _x(s["x"]), s["top"]
            break
    peak_y = _y(peak_v)
    red_a = (*C["transient"], 200)
    _dash(draw, peak_x, peak_y, peak_x, MARGIN_T * S + 20 * S,
          fill=red_a, width=S)
    draw.text((peak_x + 8 * S, MARGIN_T * S + 28 * S),
              f"STEP PEAK · {peak_v:.1f} GB",
              fill=(*C["txt_red"], 255), font=_f(13), anchor="ls")
    headroom = H200_CAP_GB - peak_v
    sign = "headroom" if headroom >= 0 else "OVERFLOW"
    draw.text((peak_x + 8 * S, MARGIN_T * S + 46 * S),
              f"{abs(headroom):.1f} GB {sign}",
              fill=(*C["txt_red"], 255), font=_f(11), anchor="ls")

    # Stash callout
    if "P4_fwd_end" in phases:
        p4 = phases.index("P4_fwd_end")
        stash = rows[p4]["activations"]
        if stash > 1.0:
            draw.text((_x(bx[p4]) - 8 * S, _y(rows[p4]["allocated"]) - 4 * S),
                      f"stash {stash:.1f} GB",
                      fill=(*C["txt_stash"], 255), font=_f(10), anchor="rs")

    # Legend
    leg_x, leg_y = _x(MARGIN_L + PLOT_W + 22), _x(MARGIN_T + 8)
    leg_w = _x(290)
    draw.rounded_rectangle(
        [leg_x - 8 * S, leg_y - 22 * S,
         leg_x - 8 * S + leg_w, leg_y - 22 * S + _x(310)],
        radius=6 * S,
        fill=(*C["leg_bg"], 255), outline=(*C["leg_border"], 255))
    draw.text((leg_x, leg_y - 6 * S), "Memory components (per-phase)",
              fill=(*C["txt_dark"], 255), font=_f(12), anchor="ls")

    def _named_max(key: str) -> tuple[float, str]:
        best, bi = 0.0, 0
        for j in range(n):
            val = rows[j]["components"].get(key, 0.0)
            if val > best:
                best, bi = val, j
        return best, pretty.get(phases[bi], (phases[bi],))[0]

    row_y = leg_y + 16 * S
    for key, name in LAYER_ORDER:
        if key == "activations":
            mx = max(rows[j]["activations"] for j in range(n))
            bi = max(range(n), key=lambda j: rows[j]["activations"])
            ph_label = pretty.get(phases[bi], (phases[bi],))[0]
        else:
            mx, ph_label = _named_max(key)
        if mx <= 0.05:
            continue
        draw.rectangle([leg_x, row_y - 9 * S, leg_x + 16 * S, row_y + 3 * S],
                       fill=(*C[key], 242))
        draw.text((leg_x + 22 * S, row_y), name,
                  fill=(*C["txt_mid"], 255), font=_f(11), anchor="ls")
        draw.text((leg_x + leg_w - 12 * S, row_y),
                  f"{mx:.1f} GB @ {ph_label}",
                  fill=(*C["txt_mid"], 255), font=_f(11), anchor="rs")
        row_y += 22 * S

    t_mx, t_ph = tmx()
    draw.rectangle([leg_x, row_y - 9 * S, leg_x + 16 * S, row_y + 3 * S],
                   fill=(*C["transient"], 220))
    draw.text((leg_x + 22 * S, row_y), "transient burst",
              fill=(*C["txt_mid"], 255), font=_f(11), anchor="ls")
    draw.text((leg_x + leg_w - 12 * S, row_y), f"{t_mx:.1f} GB @ {t_ph}",
              fill=(*C["txt_mid"], 255), font=_f(11), anchor="rs")
    row_y += 22 * S
    draw.text((leg_x, row_y + 12 * S),
              "transient = max_alloc - left_boundary_alloc,",
              fill=(*C["txt_gray"], 255), font=_f(10), anchor="ls")
    draw.text((leg_x, row_y + 26 * S),
              "bump base = lerp(left, right) allocated.",
              fill=(*C["txt_gray"], 255), font=_f(10), anchor="ls")

    # Bottom table
    table_y = _y(0) + 60 * S
    draw.text((_x(MARGIN_L), table_y),
              "phase boundary · allocated / max_allocated (GB)",
              fill=(*C["txt_mid"], 255), font=_f(11), anchor="ls")
    for i in range(n):
        r = rows[i]
        draw.text((_x(bx[i]), table_y + 18 * S),
                  f"{r['allocated']:.0f}/{r['max_allocated']:.0f}",
                  fill=(*C["txt_mid"], 255), font=_f(10), anchor="ms")

    return img.convert("RGB").resize((W, H), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(phase_memory_path: Path, *, title: str, subtitle: str,
           highlight_phase: str = "P5_bwd_peak",
           fmt: str = "svg") -> Any:
    ctx = _prepare(phase_memory_path, highlight_phase=highlight_phase)
    if fmt == "png":
        return _to_png(ctx, title=title, subtitle=subtitle)
    return _to_svg(ctx, title=title, subtitle=subtitle)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-memory", required=True, type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--highlight-phase", default="P5_bwd_peak")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    fmt = "png" if str(args.out).lower().endswith(".png") else "svg"
    result = render(args.phase_memory, title=args.title,
                    subtitle=args.subtitle,
                    highlight_phase=args.highlight_phase,
                    fmt=fmt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "svg":
        args.out.write_text(result)
    else:
        result.save(args.out, format="PNG", optimize=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
