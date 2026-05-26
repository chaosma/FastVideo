#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a human-readable view of an activation-trace JSONL file.

Shows phase markers, the FILO pack/unpack pattern, and per-event GPU
memory so you can eyeball the tensor lifecycle without writing
one-off Python on the command line.

Usage:
    python3 scripts/activation_lifecycle_view.py <path-to-jsonl>
    python3 scripts/activation_lifecycle_view.py <path-to-jsonl> --section memory
    python3 scripts/activation_lifecycle_view.py <path-to-jsonl> --section all

Sections:
    phases   --- offsets of the phase markers (prepare_batch / forward / loss / backward)
    counts   --- event-type counts + saved-tensor pack/unpack count per layer
    filo     --- first and last few packs, then the matching unpacks in reverse
    memory   --- allocated GB at each pack/unpack event, top-N by allocated
    all      --- run every section (default)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  (line {lineno}: failed to parse: {e})",
                      file=sys.stderr)
    return events


def _short_module(path: str | None) -> str:
    if not path:
        return ""
    p = re.sub(r"(_orig_mod|_checkpoint_wrapped_module)\.", "", path)
    return p


def _hdr(title: str) -> None:
    print()
    print(f"=== {title} ===")


def section_phases(events: list[dict[str, Any]]) -> None:
    _hdr("phases")
    print(f"  total events: {len(events)}")
    for i, e in enumerate(events):
        if e.get("event") == "phase":
            phase = (e.get("payload") or {}).get("phase", "?")
            print(f"  index {i:6d}  phase={phase}")
        elif e.get("event") in ("trace_start", "trace_exception", "trace_end"):
            payload = e.get("payload") or {}
            extras = []
            if "offload_to_cpu" in payload:
                extras.append(f"offload={payload['offload_to_cpu']}")
            if "exc_type" in payload:
                extras.append(f"exc={payload['exc_type']}")
            tag = f" ({'; '.join(extras)})" if extras else ""
            print(f"  index {i:6d}  {e['event']}{tag}")


def section_counts(events: list[dict[str, Any]]) -> None:
    _hdr("event counts")
    counts = Counter(e.get("event", "?") for e in events)
    for k in sorted(counts):
        print(f"  {k:25s} {counts[k]:6d}")

    _hdr("saved_tensor_pack by layer")
    per_layer = Counter()
    for e in events:
        if e.get("event") == "saved_tensor_pack":
            per_layer[e.get("layer")] += 1
    if per_layer:
        total = sum(per_layer.values())
        print(f"  total: {total} packs")
        for layer in sorted(per_layer, key=lambda x: (x is None, x)):
            print(f"  layer={str(layer):>5s}  {per_layer[layer]:4d} packs")


def section_filo(events: list[dict[str, Any]], n: int = 10) -> None:
    _hdr(f"FILO pattern (first/last {n} packs and unpacks)")
    packs = [e for e in events if e.get("event") == "saved_tensor_pack"]
    unpacks = [e for e in events if e.get("event") == "saved_tensor_unpack"]
    if not packs:
        print("  no packs in this trace")
        return

    def _row(e: dict[str, Any]) -> str:
        # event_id / save_event_id live under "tensor" in the JSONL.
        tmeta = e.get("tensor") or {}
        ev_id = (tmeta.get("event_id") if e.get("event") == "saved_tensor_pack"
                 else tmeta.get("save_event_id"))
        layer = e.get("layer")
        mod = _short_module(e.get("module_path"))
        tdev = tmeta.get("device") or "?"
        return (f"  id={str(ev_id):>5s}  layer={str(layer):>5s}  "
                f"dev={tdev:<8s} {mod}")

    print(f"  --- forward packs ({len(packs)} total) ---")
    print("  FIRST:")
    for e in packs[:n]:
        print(_row(e))
    if len(packs) > 2 * n:
        print("  ...")
    print("  LAST:")
    for e in packs[-n:]:
        print(_row(e))

    print()
    print(f"  --- backward unpacks ({len(unpacks)} total) ---")
    print("  FIRST  (should mirror LAST packs above if FILO):")
    for e in unpacks[:n]:
        print(_row(e))
    if len(unpacks) > 2 * n:
        print("  ...")
    print("  LAST  (should mirror FIRST packs above if FILO):")
    for e in unpacks[-n:]:
        print(_row(e))

    # FILO verification with tolerance: autograd often saves adjacent
    # tensors (e.g., both inputs of an op) in slightly different orders
    # than backward consumes them, so strict reverse rarely holds, but
    # a small-window match captures the macro-FILO pattern.
    pack_ids = [(e.get("tensor") or {}).get("event_id") for e in packs]
    unpack_ids = [(e.get("tensor") or {}).get("save_event_id") for e in unpacks]
    if pack_ids and unpack_ids and all(i is not None for i in pack_ids + unpack_ids):
        reversed_packs = list(reversed(pack_ids))
        n_compare = min(len(pack_ids), len(unpack_ids))
        # Position offset: where each unpack lands vs strict FILO.
        # pack_position[id] = index in reversed_packs.
        pos_map = {pid: i for i, pid in enumerate(reversed_packs)}
        offsets = [abs(pos_map[uid] - i)
                   for i, uid in enumerate(unpack_ids) if uid in pos_map]
        strict = sum(1 for o in offsets if o == 0)
        within_2 = sum(1 for o in offsets if o <= 2)
        within_5 = sum(1 for o in offsets if o <= 5)
        max_off = max(offsets) if offsets else 0
        mean_off = sum(offsets) / len(offsets) if offsets else 0
        print()
        print(f"  reverse-order match (strict):      "
              f"{strict}/{n_compare} ({strict / n_compare:.1%})")
        print(f"  reverse-order match (within ±2):   "
              f"{within_2}/{n_compare} ({within_2 / n_compare:.1%})")
        print(f"  reverse-order match (within ±5):   "
              f"{within_5}/{n_compare} ({within_5 / n_compare:.1%})")
        print(f"  position offset: mean={mean_off:.1f}  max={max_off}")
        if mean_off < 3:
            print("  → near-FILO: any prefetch hook can predict the next "
                  "unpacked tensor from the previous one with small lookahead.")


def section_memory(events: list[dict[str, Any]], n: int = 20) -> None:
    _hdr(f"memory at pack/unpack events (top {n} by allocated GB)")
    rows: list[tuple[str, int | None, int | None, float, str, str]] = []
    for e in events:
        ev = e.get("event")
        if ev not in ("saved_tensor_pack", "saved_tensor_unpack"):
            continue
        mem = e.get("cuda_memory") or {}
        alloc_gb = float(mem.get("allocated", 0)) / 1e9
        layer = e.get("layer")
        tmeta = e.get("tensor") or {}
        ev_id = (tmeta.get("event_id") if ev == "saved_tensor_pack"
                 else tmeta.get("save_event_id"))
        dev = tmeta.get("device") or "?"
        rows.append((ev, ev_id, layer, alloc_gb, dev,
                     _short_module(e.get("module_path") or "")))

    if not rows:
        print("  no pack/unpack events")
        return

    print(f"  {'event':<22s} {'id':>5s}  {'layer':>5s}  "
          f"{'alloc GB':>9s}  {'dev':<8s}  module")
    print("  " + "-" * 90)
    sorted_rows = sorted(rows, key=lambda r: -r[3])
    for ev, ev_id, layer, alloc_gb, dev, mod in sorted_rows[:n]:
        print(f"  {ev:<22s} {str(ev_id):>5s}  {str(layer):>5s}  "
              f"{alloc_gb:>9.2f}  {dev:<8s}  {mod}")

    # Time-series: print every 1/Nth event so the chronological pattern
    # is visible.
    stride = max(1, len(rows) // 40)
    print()
    print(f"  --- chronological sample (every {stride}th event) ---")
    print(f"  {'idx':>5s}  {'event':<22s}  {'layer':>5s}  "
          f"{'alloc GB':>9s}")
    for i, (ev, _id, layer, alloc_gb, _dev, _mod) in enumerate(rows):
        if i % stride == 0:
            print(f"  {i:>5d}  {ev:<22s}  {str(layer):>5s}  "
                  f"{alloc_gb:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("jsonl", type=Path,
                    help="activation_trace_rank*_step*.jsonl path")
    ap.add_argument("--section",
                    default="all",
                    choices=["phases", "counts", "filo", "memory", "all"],
                    help="which view(s) to print (default: all)")
    ap.add_argument("-n", "--top",
                    type=int, default=10,
                    help="how many rows to print in head/tail/top tables")
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"error: {args.jsonl} does not exist", file=sys.stderr)
        sys.exit(2)

    print(f"file: {args.jsonl}")
    print(f"size: {os.path.getsize(args.jsonl) / 1e6:.2f} MB")
    events = _load(args.jsonl)

    if args.section in ("phases", "all"):
        section_phases(events)
    if args.section in ("counts", "all"):
        section_counts(events)
    if args.section in ("filo", "all"):
        section_filo(events, n=args.top)
    if args.section in ("memory", "all"):
        section_memory(events, n=max(args.top, 20))


if __name__ == "__main__":
    main()
