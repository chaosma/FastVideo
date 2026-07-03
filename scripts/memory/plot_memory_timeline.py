#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Render an interactive GPU-memory-over-time timeline from a snapshot pickle.

Unlike a flamegraph (which has no time axis), this answers the question the
flamegraph cannot: *how much GPU memory is in use at each moment of the training
step, and what is responsible for each allocation.*

Input is the ``stepN_rankM.pickle`` produced by ``fastvideo/memory_tracking.py``
(``torch.cuda.memory._dump_snapshot``). Each event carries ``time_us`` and a
captured stack; the snapshot's ``external_annotations`` carry the ``fvmem::``
phase markers (prepare_batch / dit_forward / loss / backward / optimizer_step /
zero_grad) with their own timestamps.

Output: a single self-contained ``*.timeline.html`` (Plotly JS embedded, opens
offline) with

* two filled step-lines — **Reserved** (real cudaMalloc pool = GPU footprint)
  and **Allocated** (live tensor bytes) — on a labelled GiB-vs-ms axis;
* shaded, labelled **phase bands** across the time axis;
* hoverable markers for every real ``cudaMalloc`` and every sizable block
  allocation, each tooltip giving the size, phase, and the originating op
  (top non-allocator stack frames).

Usage:
    python scripts/memory/plot_memory_timeline.py <snapshot.pickle> [-o out.html]
                                                  [--min-marker-mib 1.0]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from bisect import bisect_right

# Python files that are framework plumbing, not "what the allocation is for".
_PY_SKIP_FILES = (
    "eval_frame.py", "_compile.py", "_dynamo", "_inductor",
    "contextlib.py", "_contextlib.py", "_jit_internal.py", "_ops.py",
    "_meta_registrations.py", "_prims_common",
)
# Generic interpreter/Module-dispatch frame names worth de-prioritising (kept
# only if nothing better is found).
_PY_LOWVALUE_NAMES = (
    "_call_impl", "_wrapped_call_impl", "_fn", "decorate_context",
    "inner", "wrapper", "<module>", "_run_ddp_forward",
)
# C++ frames that name the actual op (used as a secondary context tag).
_CPP_OP_HINTS = ("at::native::", "VariableType", "THPVariable_")
# C++ op names that are pure allocation primitives (uninformative as the op).
_CPP_OP_SKIP = ("empty", "empty_cuda", "empty_generic", "empty_strided",
                "_to_copy", "resize_", "copy_")
# Path substrings that mark application / model code (highest-value frames).
_APP_HINTS = ("fastvideo", "diffusers", "wan")

# JS injected after plot creation: when the user zooms/pans the X axis (box
# zoom, scroll, or pan), auto-fit the Y axis to just the data in the visible
# window so a zoomed section rescales nicely; double-click restores autorange.
# Markers use fixed pixel sizing, so they stay the same size at any zoom level.
_ZOOM_FIT_JS = """
var gd = document.getElementById('{plot_id}');
function fvFitY() {
  var xa = (gd.layout.xaxis && gd.layout.xaxis.range) ? gd.layout.xaxis.range : null;
  if (!xa) { return; }
  var lo = xa[0], hi = xa[1], ymin = Infinity, ymax = -Infinity;
  gd.data.forEach(function(tr) {
    if (!tr.y) { return; }
    var xs = tr.x, ys = tr.y;
    for (var i = 0; i < ys.length; i++) {
      var x = xs ? xs[i] : i;
      if (x >= lo && x <= hi) {
        var y = ys[i];
        if (y != null) { if (y < ymin) { ymin = y; } if (y > ymax) { ymax = y; } }
      }
    }
  });
  if (ymin === Infinity) { return; }
  var pad = (ymax - ymin) * 0.06 || 1;
  Plotly.relayout(gd, {'yaxis.range': [Math.max(0, ymin - pad), ymax + pad]});
}
gd.on('plotly_relayout', function(ev) {
  if (ev['xaxis.autorange']) { Plotly.relayout(gd, {'yaxis.autorange': true}); return; }
  if (('xaxis.range[0]' in ev) || ('xaxis.range' in ev)) { fvFitY(); }
});
"""

# Stable colours per phase so the bands read at a glance.
_PHASE_COLORS = {
    "prepare_batch": "#8ecae6",
    "dit_forward": "#ffb703",
    "loss": "#fb8500",
    "backward": "#e63946",
    "optimizer_step": "#2a9d8f",
    "zero_grad": "#6a4c93",
}
_PHASE_FALLBACK = "#adb5bd"


def _short_op(name: str) -> str | None:
    """Extract a terse op token from a C++ frame name, or None if uninformative."""
    base = name.split("(")[0]
    base = base.replace("torch::autograd::THPVariable_", "")
    tok = base.split("::")[-1].strip().strip("_") or base
    return None if tok in _CPP_OP_SKIP else tok


def _what_for(event, max_frames: int = 7) -> tuple[str, str]:
    """Return (short label, multi-line detail) describing an allocation's origin.

    Prefers the **Python** call stack (the readable "what for"), innermost
    first, dropping interpreter/compile plumbing; tags on a clean C++ op name
    for context. Application/model frames (fastvideo / diffusers / wan) are
    promoted as the headline label.
    """
    frames = event.get("frames") or []
    py = []  # (name, file, line, is_app)
    for fr in frames:
        fn = fr.get("filename") or ""
        if not fn.endswith(".py") or any(s in fn for s in _PY_SKIP_FILES):
            continue
        nm = (fr.get("name") or "").strip()
        if not nm:
            continue
        py.append((nm, os.path.basename(fn), fr.get("line"),
                   any(h in fn for h in _APP_HINTS)))

    # A clean op name from the C++ stack (e.g. mse_loss, linear, gelu).
    op = None
    for fr in frames:
        nm = fr.get("name") or ""
        if any(h in nm for h in _CPP_OP_HINTS):
            op = _short_op(nm)
            if op:
                break

    if not py and op is None:
        return "<allocator-internal>", "<no application stack frames captured>"

    # Detail: innermost-first python frames, but make sure at least one app
    # frame survives the truncation if one exists deeper in the stack.
    detail_frames = py[:max_frames]
    app_frames = [t for t in py if t[3]]
    if app_frames and app_frames[-1] not in detail_frames:
        detail_frames = detail_frames[:max_frames - 1] + [app_frames[-1]]
    lines = [f"{nm}  ({fl}:{ln})" for (nm, fl, ln, _) in detail_frames]
    if op:
        lines.append(f"[op: {op}]")

    # Headline: first app frame, else a meaningful python frame, else the op.
    pick = next((t for t in py if t[3]), None)
    if pick is None:
        pick = next((t for t in py if t[0] not in _PY_LOWVALUE_NAMES), None)
    if pick is None:
        pick = py[0] if py else None
    short = f"{pick[0]} ({pick[1]}:{pick[2]})" if pick else f"op: {op}"
    return short, "<br>".join(lines)


def _human(nbytes: float) -> str:
    n = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def _phase_windows(snapshot, device, t0_us):
    """Pair fvmem:: START/END annotations into (phase, start_ms, end_ms)."""
    stack: dict[str, int] = {}
    windows = []
    for ann in snapshot.get("external_annotations") or []:
        if ann.get("device") not in (None, device):
            continue
        name = ann.get("name", "")
        if not name.startswith("fvmem::"):
            continue
        phase = name[len("fvmem::"):]
        if ann.get("stage") == "START":
            stack[phase] = ann["time_us"]
        elif ann.get("stage") == "END" and phase in stack:
            start = stack.pop(phase)
            windows.append((phase, (start - t0_us) / 1e3,
                            (ann["time_us"] - t0_us) / 1e3))
    return windows


def _fsdp_block_starts(events, t0, gap_ms: float = 2.0):
    """Times (ms since t0) of FSDP all-gather clusters = block-forward starts.

    Each block's forward begins by all-gathering its params
    (``_fsdp_param.py:alloc_storage`` under ``wait_for_unshard``/``pre_forward``);
    consecutive such allocs within ``gap_ms`` are one cluster (one block).
    """
    starts, prev = [], None
    for e in events:
        if e.get("action") != "alloc":
            continue
        frs = e.get("frames") or []
        if not any((f.get("filename", "").endswith("_fsdp_param.py")
                    and f.get("name") == "alloc_storage") for f in frs):
            continue
        tms = (e["time_us"] - t0) / 1e3
        if prev is None or tms - prev > gap_ms:
            starts.append(tms)
        prev = tms
    return starts


def build_figure(snapshot, min_marker_bytes: float, sidecar: dict | None = None,
                 config_label: str = "baseline", is_oom: bool = False,
                 mark_blocks: bool = False):
    import plotly.graph_objects as go

    # H200 HBM3e capacity, for an at-a-glance "how close to OOM" reference line.
    _H200_CAPACITY_GIB = 139.81

    # Pick the device that was actually recorded (the only non-empty trace).
    traces = snapshot.get("device_traces") or []
    dev, events = next(
        ((i, t) for i, t in enumerate(traces) if t), (0, []))
    if not events:
        raise SystemExit("snapshot has no recorded allocation events")

    events = sorted(events, key=lambda e: e["time_us"])
    t0 = events[0]["time_us"]

    # Driver-level reserve events. With the default allocator these are
    # segment_alloc/segment_free (each = one cudaMalloc/cudaFree); with
    # PYTORCH_ALLOC_CONF=expandable_segments:True they are segment_map/
    # segment_unmap (cuMemMap of physical pages into a reserved range).
    seg_up = ("segment_alloc", "segment_map")
    seg_down = ("segment_free", "segment_unmap")

    # Absolute baseline = memory already resident when recording started.
    #   * EXACT path: the sidecar's memory_stats_start counters (authoritative;
    #     ~0 in warmup mode since recording begins before any allocation).
    #   * Fallback: reconstruct from the final segment state minus the net change
    #     observed in the window (imprecise across allocator backends — only used
    #     for older bracketed snapshots without a sidecar).
    start = (sidecar or {}).get("memory_stats_start") or {}
    if "reserved_current" in start and "allocated_current" in start:
        baseline_reserved = int(start["reserved_current"])
        baseline_allocated = int(start["allocated_current"])
        baseline_exact = True
    else:
        seg_state = snapshot.get("segments") or []
        end_reserved = sum(s.get("total_size", 0) for s in seg_state)
        end_active = sum(b.get("size", 0) for s in seg_state
                         for b in s.get("blocks", [])
                         if b.get("state") == "active_allocated")
        net_reserved = (
            sum(e.get("size", 0) for e in events if e["action"] in seg_up)
            - sum(e.get("size", 0) for e in events if e["action"] in seg_down))
        net_alloc = (
            sum(e.get("size", 0) for e in events if e["action"] == "alloc")
            - sum(e.get("size", 0) for e in events if e["action"] == "free_completed"))
        baseline_reserved = max(0, end_reserved - net_reserved)
        baseline_allocated = max(0, end_active - net_alloc)
        baseline_exact = False

    # Replay the event stream into reserved / allocated step-curves, starting
    # from the baseline and seeding a point at t=0 so the filled area reflects
    # the absolute footprint from the very start of the recording.
    reserved = baseline_reserved
    allocated = baseline_allocated
    res_t, res_y = [0.0], [reserved / 2**30]
    alloc_t, alloc_y = [0.0], [allocated / 2**30]
    marker = {"t": [], "y": [], "size": [], "color": [], "hover": []}
    phase_windows = _phase_windows(snapshot, dev, t0)
    # Sorted phase starts for fast "which phase is this event in" lookup.
    pw_sorted = sorted(phase_windows, key=lambda w: w[1])
    pw_starts = [w[1] for w in pw_sorted]

    def phase_at(ms: float) -> str:
        i = bisect_right(pw_starts, ms) - 1
        if 0 <= i < len(pw_sorted):
            ph, s, e = pw_sorted[i]
            if s <= ms <= e:
                return ph
        return "-"

    for e in events:
        ms = (e["time_us"] - t0) / 1e3
        act = e["action"]
        sz = e.get("size", 0)
        if act in seg_up:
            reserved += sz
            res_t.append(ms)
            res_y.append(reserved / 2**30)
        elif act in seg_down:
            reserved -= sz
            res_t.append(ms)
            res_y.append(reserved / 2**30)
        elif act == "alloc":
            allocated += sz
            alloc_t.append(ms)
            alloc_y.append(allocated / 2**30)
        elif act == "free_completed":
            allocated -= sz
            alloc_t.append(ms)
            alloc_y.append(allocated / 2**30)

        # Inspectable markers: every real driver reserve, plus big block allocs.
        is_seg = act in seg_up
        if (is_seg or act == "alloc") and (is_seg or sz >= min_marker_bytes):
            short, detail = _what_for(e)
            ph = phase_at(ms)
            kind = ("expandable segment map (cuMemMap)"
                    if act == "segment_map" else "cudaMalloc (segment)"
                    ) if is_seg else "block alloc"
            marker["t"].append(ms)
            marker["y"].append((reserved if is_seg else allocated) / 2**30)
            marker["size"].append(sz)
            marker["color"].append(_PHASE_COLORS.get(ph, _PHASE_FALLBACK))
            marker["hover"].append(
                f"<b>{kind}</b><br>size: {_human(sz)}<br>"
                f"phase: <b>{ph}</b><br>t = {ms:.1f} ms<br>"
                f"reserved now: {_human(reserved)}<br>"
                f"<br><b>origin:</b><br>{detail}")

    peak_res = max(res_y) if res_y else 0
    peak_alloc = max(alloc_y) if alloc_y else 0
    dur_ms = (events[-1]["time_us"] - t0) / 1e3

    # Authoritative peaks (true allocator counters) beat the event-replay max,
    # which can under-read the final pre-OOM spike. Use them for the headline.
    end = (sidecar or {}).get("memory_stats_end") or {}
    peak_res_auth = int(end.get("reserved_peak", 0)) / 2**30 or peak_res
    peak_alloc_auth = int(end.get("allocated_peak", 0)) / 2**30 or peak_alloc

    fig = go.Figure()

    # Structural windows (warmup / step1 / step2 ...) delineate the lifecycle;
    # phase windows (prepare_batch / dit_forward / ...) are the colored bands
    # nested inside each step. Render them differently so they don't overlap.
    import re
    def _structural(name: str) -> bool:
        return name == "warmup" or bool(re.fullmatch(r"step\d+", name))

    # Phase bands (drawn first so curves sit on top).
    for ph, s, e in phase_windows:
        if _structural(ph):
            continue
        color = _PHASE_COLORS.get(ph, _PHASE_FALLBACK)
        fig.add_vrect(
            x0=s, x1=e, fillcolor=color, opacity=0.18, line_width=0,
            layer="below",
            annotation_text=ph, annotation_position="top left",
            annotation=dict(font_size=11, font_color=color, textangle=-90),
        )

    # Structural boundaries: dashed vertical dividers + a top label.
    for name, s, e in sorted((w for w in phase_windows if _structural(w[0])),
                             key=lambda w: w[1]):
        fig.add_vline(
            x=s, line_dash="dash", line_color="#212529", opacity=0.55,
            annotation_text=f"▏{name}", annotation_position="top right",
            annotation=dict(font_size=13, font_color="#212529"))

    fig.add_trace(go.Scatter(
        x=res_t, y=res_y, name="Reserved (cudaMalloc pool)",
        mode="lines", line=dict(color="#1d3557", width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(29,53,87,0.12)",
        hovertemplate="Reserved: %{y:.2f} GiB<br>t = %{x:.1f} ms<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=alloc_t, y=alloc_y, name="Allocated (live tensors)",
        mode="lines", line=dict(color="#e63946", width=1.5, shape="hv"),
        hovertemplate="Allocated: %{y:.2f} GiB<br>t = %{x:.1f} ms<extra></extra>"))

    # Marker sizes scaled by sqrt(bytes) for visual range (~1/3 of original);
    # sizemode="diameter" keeps them a fixed pixel size so they don't grow when
    # the user zooms into a region.
    if marker["t"]:
        import math
        sizes = [max(2, min(9, (4 + math.sqrt(s / 2**20)) / 3))
                 for s in marker["size"]]
        fig.add_trace(go.Scatter(
            x=marker["t"], y=marker["y"], name="allocations (hover for detail)",
            mode="markers",
            marker=dict(size=sizes, sizemode="diameter", color=marker["color"],
                        line=dict(width=0.3, color="#222"), opacity=0.75),
            text=marker["hover"], hovertemplate="%{text}<extra></extra>"))

    # Baseline (memory already resident when recording started) as a reference.
    base_note = ("exact" if baseline_exact else "reconstructed")
    fig.add_hline(
        y=baseline_reserved / 2**30, line_dash="dot", line_color="#1d3557",
        opacity=0.6,
        annotation_text=(f"recording-start baseline reserved "
                         f"{baseline_reserved / 2**30:.2f} GiB ({base_note})"),
        annotation_position="bottom right",
        annotation_font_size=11)

    # Final reserved level. In warmup/continuous captures the reserved
    # step-curve's last point can sit well before the end of the time axis (no
    # cudaMalloc / segment_map fires during the plotted step), so the curve
    # alone visually understates the resident pool. Draw the true end-of-window
    # reserved level (authoritative counter, else the last curve point) as a
    # reference so it stays visible regardless.
    reserved_now = (int(end.get("reserved_current", 0)) / 2**30
                    or (res_y[-1] if res_y else 0.0))
    if reserved_now > 0:
        fig.add_hline(
            y=reserved_now, line_dash="dot", line_color="#1d3557", opacity=0.4,
            annotation_text=f"reserved at recording end {reserved_now:.2f} GiB",
            annotation_position="top right", annotation_font_size=11)

    # H200 capacity reference line — shows headroom to OOM at a glance.
    fig.add_hline(
        y=_H200_CAPACITY_GIB, line_dash="dashdot", line_color="#6c757d",
        opacity=0.7,
        annotation_text=f"H200 capacity {_H200_CAPACITY_GIB:.1f} GiB",
        annotation_position="top left", annotation_font_size=11)

    # If this snapshot was dumped from the OOM observer, mark the death point.
    if is_oom:
        oom_ms = (events[-1]["time_us"] - t0) / 1e3
        fig.add_vline(
            x=oom_ms, line_dash="dot", line_color="#c1121f", line_width=2,
            annotation_text="CUDA OOM", annotation_position="top left",
            annotation=dict(font_size=12, font_color="#c1121f"))

    n_steps = len([w for w in phase_windows if re.fullmatch(r"step\d+", w[0])])
    status = ("OOM in step-1 backward" if is_oom
              else f"completed {n_steps} steps")
    fig.update_layout(
        title=(f"Wan2.1-T2V-14B 4K/121f · {config_label} (8×H200, SP=8) "
               f"&nbsp;|&nbsp; peak reserved {peak_res_auth:.1f} GiB, peak "
               f"allocated {peak_alloc_auth:.1f} GiB &nbsp;|&nbsp; start "
               f"baseline {baseline_reserved / 2**30:.2f} GiB ({base_note}) "
               f"&nbsp;|&nbsp; {len(events):,} events over {dur_ms / 1e3:.1f} s "
               f"&nbsp;|&nbsp; {status}"),
        xaxis_title="Time since recording start (ms)",
        yaxis_title="GPU memory (GiB)",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=720,
    )
    # Seed the Y range to include the capacity line so it's visible on load
    # (zoom-fit JS rescales on interaction; double-click autoranges to data).
    y_top = max(peak_res, peak_res_auth, _H200_CAPACITY_GIB) * 1.06
    fig.update_yaxes(rangemode="tozero", gridcolor="#e9ecef", range=[0, y_top])
    fig.update_xaxes(gridcolor="#f1f3f5")

    if mark_blocks:
        starts = _fsdp_block_starts(events, t0)
        for i, x in enumerate(starts):
            fig.add_vline(x=x, line_width=1, line_dash="dot",
                          line_color="rgba(120,40,160,0.45)")
        if starts:
            fig.add_annotation(x=starts[0], yref="paper", y=1.0,
                               text=f"↧ FSDP block-forward starts "
                                    f"({len(starts)} blocks, dotted purple)",
                               showarrow=False, xanchor="left",
                               font=dict(color="rgb(120,40,160)", size=11))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pickle", help="stepN_rankM.pickle from memory_tracking")
    ap.add_argument("-o", "--output", default=None,
                    help="output HTML (default: <pickle>.timeline.html)")
    ap.add_argument("--min-marker-mib", type=float, default=1.0,
                    help="min block-alloc size to mark (cudaMallocs always marked)")
    ap.add_argument("--config-label", default="baseline",
                    help="experiment config description shown in the title")
    ap.add_argument("--mark-fsdp-blocks", action="store_true",
                    help="draw a dotted vertical line at each FSDP "
                         "all-gather (transformer-block forward start)")
    args = ap.parse_args()

    with open(args.pickle, "rb") as fh:
        snapshot = pickle.load(fh)

    # Load the sidecar (same basename, .regions.json) for the exact baseline.
    sidecar = None
    sidecar_path = args.pickle.rsplit(".pickle", 1)[0] + ".regions.json"
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as fh:
            sidecar = json.load(fh)

    # OOM if the recorder tagged the window (dump fired from the OOM observer)
    # or the filename says so.
    label = (sidecar or {}).get("label", "")
    is_oom = ("oom" in label.lower()
              or "oom" in os.path.basename(args.pickle).lower())

    fig = build_figure(snapshot, args.min_marker_mib * 2**20, sidecar=sidecar,
                       config_label=args.config_label, is_oom=is_oom,
                       mark_blocks=args.mark_fsdp_blocks)
    out = args.output or (args.pickle.rsplit(".pickle", 1)[0] + ".timeline.html")
    fig.write_html(
        out, include_plotlyjs="inline", full_html=True,
        config={"scrollZoom": True, "responsive": True, "displaylogo": False},
        post_script=_ZOOM_FIT_JS)
    print(f"wrote interactive timeline -> {out}")


if __name__ == "__main__":
    main()
