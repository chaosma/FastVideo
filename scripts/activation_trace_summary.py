#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize FastVideo activation trace JSONL files."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def _tensor_bytes(event: dict[str, Any]) -> int:
    tensor = event.get("tensor") or {}
    value = tensor.get("bytes", 0)
    return int(value) if isinstance(value, (int, float, str)) else 0


def summarize(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    packs: dict[int, dict[str, Any]] = {}
    unpack_ids: list[int] = []
    unpacked: set[int] = set()
    live_bytes = 0
    peak_live_bytes = 0
    peak_event_index = 0
    top_packs: list[dict[str, Any]] = []
    bytes_by_module: defaultdict[str, int] = defaultdict(int)
    event_index = 0

    for event in _iter_events(path):
        event_index += 1
        event_name = str(event.get("event"))
        counts[event_name] += 1
        if event_name == "saved_tensor_pack":
            tensor = event.get("tensor") or {}
            save_event_id = tensor.get("event_id")
            if save_event_id is None:
                continue
            save_event_id = int(save_event_id)
            size = _tensor_bytes(event)
            packs[save_event_id] = event
            live_bytes += size
            module_path = str(event.get("module_path") or "<unknown>")
            bytes_by_module[module_path] += size
            top_packs.append(event)
            if live_bytes > peak_live_bytes:
                peak_live_bytes = live_bytes
                peak_event_index = event_index
        elif event_name == "saved_tensor_unpack":
            tensor = event.get("tensor") or {}
            save_event_id = tensor.get("save_event_id")
            if save_event_id is None:
                continue
            save_event_id = int(save_event_id)
            unpack_ids.append(save_event_id)
            if save_event_id not in unpacked:
                pack = packs.get(save_event_id)
                if pack is not None:
                    live_bytes -= _tensor_bytes(pack)
                unpacked.add(save_event_id)

    top_packs.sort(key=_tensor_bytes, reverse=True)
    adjacent_filo_violations = sum(
        1
        for prev, cur in zip(unpack_ids, unpack_ids[1:])
        if cur > prev
    )
    return {
        "path": str(path),
        "counts": dict(counts),
        "saved_tensor_pack_count": len(packs),
        "saved_tensor_unpack_count": len(unpack_ids),
        "unique_unpacked_count": len(unpacked),
        "unmatched_pack_count": len(set(packs) - unpacked),
        "unpack_without_pack_count": len(unpacked - set(packs)),
        "adjacent_filo_violations": adjacent_filo_violations,
        "peak_live_saved_tensor_bytes": peak_live_bytes,
        "peak_live_saved_tensor_gb": peak_live_bytes / 1e9,
        "peak_event_index": peak_event_index,
        "top_saved_tensors": [
            {
                "bytes": _tensor_bytes(event),
                "gb": _tensor_bytes(event) / 1e9,
                "module_path": event.get("module_path"),
                "layer": event.get("layer"),
                "shape": (event.get("tensor") or {}).get("shape"),
                "dtype": (event.get("tensor") or {}).get("dtype"),
                "event_id": (event.get("tensor") or {}).get("event_id"),
            }
            for event in top_packs[:20]
        ],
        "top_modules_by_saved_bytes": [
            {
                "module_path": module,
                "bytes": size,
                "gb": size / 1e9,
            }
            for module, size in sorted(
                bytes_by_module.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace",
        nargs="+",
        help="Trace JSONL path(s), or glob patterns such as '/tmp/traces/*.jsonl'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a short text report.",
    )
    args = parser.parse_args()

    summaries = [summarize(path) for path in _iter_paths(args.trace)]
    if args.json:
        print(json.dumps(summaries, indent=2, sort_keys=True))
        return

    for summary in summaries:
        print(f"\n{summary['path']}")
        print(f"  saved packs/unpacks: {summary['saved_tensor_pack_count']} / {summary['saved_tensor_unpack_count']}")
        print(f"  unmatched packs: {summary['unmatched_pack_count']}")
        print(f"  adjacent FILO violations: {summary['adjacent_filo_violations']}")
        print(f"  peak live saved tensors: {summary['peak_live_saved_tensor_gb']:.3f} GB")
        print("  top saved tensors:")
        for item in summary["top_saved_tensors"][:10]:
            print(
                "    "
                f"{item['gb']:.3f} GB "
                f"layer={item['layer']} "
                f"module={item['module_path']} "
                f"shape={item['shape']} "
                f"dtype={item['dtype']}"
            )


if __name__ == "__main__":
    main()
