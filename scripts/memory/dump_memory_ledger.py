#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Print a per-event GPU allocation ledger from a memory-history snapshot.

Reads a pickle produced by ``torch.cuda.memory._dump_snapshot`` (see
``fastvideo/memory_tracking.py``) and prints every recorded allocator event in
order, distinguishing:

* TRUE driver allocations -- ``segment_alloc`` / ``segment_free``
  (a real ``cudaMalloc`` / ``cudaFree``), or ``segment_map`` / ``segment_unmap``
  under ``expandable_segments``; and
* RESERVED-space pseudo-allocations -- ``alloc`` / ``free_requested`` /
  ``free_completed`` (a tensor block carved out of / returned to a segment).

Each line shows the event's size, address, the phase label captured via
``mem_region`` (``fvmem::<phase>``), and the top user stack frame ("what for"),
alongside running reserved / live-allocated totals.

Usage::

    python scripts/memory/dump_memory_ledger.py STEP.pickle
    python scripts/memory/dump_memory_ledger.py STEP.pickle --kind real
    python scripts/memory/dump_memory_ledger.py STEP.pickle --min-size 1MiB --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

REGION_PREFIX = "fvmem::"

REAL_ACTIONS = {"segment_alloc", "segment_free", "segment_map", "segment_unmap"}
PSEUDO_ACTIONS = {"alloc", "free_requested", "free_completed"}
# record_function ranges surface in the trace as standalone annotation events
# (record_function does not change the Python call stack, so the label is NOT a
# normal frame). We track these to attribute a running "current phase".
ANNOTATION_ACTIONS = {"user_defined", "annotation"}

# Stack frames inside these paths are not the "what for" -- skip past them to
# the first frame that belongs to model / training code.
_SKIP_FRAME_SUBSTRINGS = (
    "torch/cuda/memory",
    "fastvideo/memory_tracking",
    "torch/autograd/profiler",
    "torch/profiler",
)


def _human(n: int) -> str:
    neg = n < 0
    n = abs(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            s = f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
            return ("-" + s) if neg else s
        n /= 1024.0
    return str(n)


def _parse_size(s: str) -> int:
    s = s.strip()
    units = {"b": 1, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "kb": 1000, "mb": 1000**2, "gb": 1000**3}
    low = s.lower()
    for u, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if low.endswith(u):
            return int(float(low[:-len(u)]) * mult)
    return int(float(s))


def _frames(event: dict) -> list[dict]:
    fr = event.get("frames")
    return fr if isinstance(fr, list) else []


def _annotation_label(event: dict) -> str | None:
    """Return the fvmem phase label if ``event`` is one of our annotations.

    Handles both shapes a region marker can take: a dedicated
    ``user_defined``/``annotation`` event carrying the label in ``name`` (the
    common case), or the label embedded as a captured stack frame.
    """
    if event.get("action") in ANNOTATION_ACTIONS:
        name = event.get("name", "") or ""
        if name.startswith(REGION_PREFIX):
            return name[len(REGION_PREFIX):]
        # Some builds put the annotation string in the (single) frame instead.
        for f in _frames(event):
            fn = f.get("name", "")
            if fn.startswith(REGION_PREFIX):
                return fn[len(REGION_PREFIX):]
    return None


def _frame_phase(event: dict) -> str | None:
    """fvmem:: label embedded directly in a normal event's frames, if any."""
    for f in _frames(event):
        name = f.get("name", "")
        if name.startswith(REGION_PREFIX):
            return name[len(REGION_PREFIX):]
    return None


def _what_for(event: dict) -> str:
    """First user stack frame that isn't allocator / profiler plumbing."""
    for f in _frames(event):
        name = f.get("name", "")
        fname = f.get("filename", "")
        if name.startswith(REGION_PREFIX):
            continue
        if any(sub in fname for sub in _SKIP_FRAME_SUBSTRINGS):
            continue
        short = fname.split("fastvideo/", 1)[-1] if "fastvideo/" in fname else os.path.basename(fname)
        return f"{name} ({short}:{f.get('line', '?')})"
    fr = _frames(event)
    if fr:
        f = fr[0]
        return f"{f.get('name', '?')} ({os.path.basename(f.get('filename', '?'))}:{f.get('line', '?')})"
    return "<no stack captured>"


def _pick_device(snapshot: dict, requested: int | None) -> int:
    traces = snapshot.get("device_traces", [])
    if requested is not None:
        return requested
    # Default to the device with the most events.
    best, best_len = 0, -1
    for dev, evs in enumerate(traces):
        if len(evs) > best_len:
            best, best_len = dev, len(evs)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", help="Path to the _dump_snapshot pickle")
    ap.add_argument("--device", type=int, default=None, help="CUDA device index in the snapshot (default: busiest)")
    ap.add_argument("--kind", choices=["all", "real", "pseudo"], default="all",
                    help="real = cudaMalloc/free only; pseudo = reserved-space block events only")
    ap.add_argument("--min-size", type=_parse_size, default=0, help="Only show events >= this size (e.g. 1MiB)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of event lines to print")
    args = ap.parse_args()

    with open(args.snapshot, "rb") as fh:
        snapshot = pickle.load(fh)

    traces = snapshot.get("device_traces", [])
    if not traces:
        print("No device_traces in snapshot -- was recording enabled with "
              "context='all'?", file=sys.stderr)
        return 1

    dev = _pick_device(snapshot, args.device)
    events = traces[dev]
    print(f"# snapshot: {args.snapshot}")
    print(f"# device:   {dev}   events: {len(events)}")

    # Optional phase summary sidecar.
    regions_json = args.snapshot.replace(".pickle", ".regions.json")
    if os.path.exists(regions_json):
        with open(regions_json) as fh:
            summary = json.load(fh)
        print("#\n# per-phase memory_stats deltas (from sidecar):")
        hdr = (f"#   {'phase':<16}{'reservedΔ':>12}{'allocΔ':>12}"
               f"{'cudaMalloc':>11}{'cudaFree':>9}{'blkAlloc':>9}{'blkFree':>9}")
        print(hdr)
        for r in summary.get("regions", []):
            print(f"#   {r['region']:<16}"
                  f"{_human(r['reserved_bytes_delta']):>12}"
                  f"{_human(r['allocated_bytes_delta']):>12}"
                  f"{r['segment_alloc_count']:>11}{r['segment_free_count']:>9}"
                  f"{r['block_alloc_count']:>9}{r['block_free_count']:>9}")

    print("#\n# legend: [REAL]=cudaMalloc/cudaFree  [PSD]=reserved-space block")
    print(f"# {'idx':>7} {'kind':<5} {'action':<15} {'size':>11} "
          f"{'reserved':>11} {'live':>11}  phase | what-for")

    reserved = 0
    live = 0
    shown = 0
    current_phase = "-"
    for idx, ev in enumerate(events):
        action = ev.get("action", "?")
        size = int(ev.get("size", 0))

        # Region markers update the running phase; they carry no allocation.
        label = _annotation_label(ev)
        if label is not None:
            current_phase = label
            if args.kind == "all":
                print(f"  {idx:>7} {'MARK':<5} {action:<15} {'':>11} "
                      f"{_human(reserved):>11} {_human(live):>11}  "
                      f">>> phase: {label}")
            continue

        # Maintain running totals across ALL events (not just filtered ones).
        if action in ("segment_alloc", "segment_map"):
            reserved += size
        elif action in ("segment_free", "segment_unmap"):
            reserved -= size
        elif action == "alloc":
            live += size
        elif action == "free_completed":
            live -= size

        is_real = action in REAL_ACTIONS
        if args.kind == "real" and not is_real:
            continue
        if args.kind == "pseudo" and action not in PSEUDO_ACTIONS:
            continue
        if size < args.min_size and action not in REAL_ACTIONS:
            continue
        if args.limit is not None and shown >= args.limit:
            print(f"# ... truncated at --limit {args.limit}")
            break

        phase = _frame_phase(ev) or current_phase
        kind = "REAL" if is_real else ("PSD" if action in PSEUDO_ACTIONS else "-")
        print(f"  {idx:>7} {kind:<5} {action:<15} {_human(size):>11} "
              f"{_human(reserved):>11} {_human(live):>11}  "
              f"{phase} | {_what_for(ev)}")
        shown += 1

    print(f"#\n# end totals -> reserved(net cudaMalloc): {_human(reserved)}   "
          f"live(net block alloc): {_human(live)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
