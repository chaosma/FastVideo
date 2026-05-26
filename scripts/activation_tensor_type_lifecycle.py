#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize saved activation lifetimes by semantic tensor type."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_WRAPPER_SEGMENT_RE = re.compile(
    r"(?:_orig_mod|_checkpoint_wrapped_module)\.")
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)")


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


def _normalize_module_path(module_path: str | None) -> str:
    if not module_path:
        return "<unknown>"
    return _WRAPPER_SEGMENT_RE.sub("", str(module_path))


def _shape(tensor: dict[str, Any]) -> list[int]:
    raw = tensor.get("shape")
    if isinstance(raw, list):
        return [int(x) for x in raw if isinstance(x, int | float)]
    return []


def _shape_key(shape: list[int]) -> str:
    if not shape:
        return "[]"
    return "[" + ",".join(str(x) for x in shape) + "]"


def _layer(module_path: str, event: dict[str, Any]) -> int | None:
    raw = event.get("layer")
    if isinstance(raw, int):
        return raw
    match = _BLOCK_RE.search(module_path)
    return int(match.group(1)) if match else None


def _module_family(module_path: str) -> str:
    return re.sub(r"blocks\.\d+", "blocks.*", module_path)


def _classify(event: dict[str, Any]) -> tuple[str, str]:
    module_path = _normalize_module_path(event.get("module_path"))
    family = _module_family(module_path)
    tensor = event.get("tensor") or {}
    shape = _shape(tensor)
    dtype = str(tensor.get("dtype") or "")
    layer = _layer(module_path, event)

    if module_path == "patch_embedding":
        if len(shape) == 5 and shape[0] == 1:
            return "patch_embedding.latent_input_saved", family
        return "patch_embedding.weight_or_aux_saved", family

    if module_path == "condition_embedder":
        if len(shape) >= 2 and shape[-1] == 5120:
            return "condition_embedder.hidden_or_weight_saved", family
        return "condition_embedder.aux_saved", family

    if module_path == "norm_out":
        if len(shape) == 3 and shape[-1] == 5120 and "float32" in dtype:
            return "output_norm.fp32_residual_saved", family
        if len(shape) == 3 and shape[-1] == 1:
            return "output_norm.scalar_stats_saved", family
        return "output_norm.aux_saved", family

    if module_path == "proj_out":
        if len(shape) == 2 and shape[-1] == 5120:
            return "output_projection.input_tokens_saved", family
        return "output_projection.weight_or_aux_saved", family

    if layer is not None:
        if len(shape) == 3 and shape[-1] == 5120:
            if len(shape) >= 2 and shape[1] > 4096:
                return "block.residual_stream_checkpoint", family
            if len(shape) >= 2 and shape[1] == 512:
                return "block.text_conditioning_saved", family
            return "block.modulation_or_small_hidden_saved", family
        if shape == [0]:
            return "block.checkpoint_placeholder", family
        return "block.other_saved", family

    if not module_path or module_path == "<unknown>":
        if len(shape) == 5:
            return "loss_or_output.latent_tensor_saved", family
        return "unknown.saved_tensor", family

    return "other.saved_tensor", family


def summarize(paths: list[Path]) -> list[dict[str, Any]]:
    pack_records: dict[tuple[str, int, int], dict[str, Any]] = {}
    rows_by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in paths:
        rank = None
        event_index = 0
        for event in _iter_events(path):
            event_index += 1
            if rank is None and isinstance(event.get("rank"), int):
                rank = int(event["rank"])
            event_name = str(event.get("event"))
            tensor = event.get("tensor") or {}
            if event_name == "saved_tensor_pack":
                event_id = tensor.get("event_id")
                if event_id is None:
                    continue
                event_id = int(event_id)
                module_path = _normalize_module_path(event.get("module_path"))
                tensor_type, family = _classify(event)
                key = (str(path), int(rank or 0), event_id)
                pack_records[key] = {
                    "path": str(path),
                    "rank": int(rank or 0),
                    "event_id": event_id,
                    "tensor_type": tensor_type,
                    "module_family": family,
                    "module_path": module_path,
                    "layer": _layer(module_path, event),
                    "shape": _shape(tensor),
                    "dtype": str(tensor.get("dtype")),
                    "bytes": int(tensor.get("bytes", 0) or 0),
                    "pack_index": event_index,
                    "pack_time_s": event.get("time_s"),
                    "unpack_index": None,
                    "unpack_time_s": None,
                }
            elif event_name == "saved_tensor_unpack":
                event_id = tensor.get("save_event_id")
                if event_id is None:
                    continue
                key = (str(path), int(rank or 0), int(event_id))
                record = pack_records.get(key)
                if record is not None:
                    record["unpack_index"] = event_index
                    record["unpack_time_s"] = event.get("time_s")

        for record in pack_records.values():
            if record["path"] == str(path):
                rows_by_type[record["tensor_type"]].append(record)

    summaries: list[dict[str, Any]] = []
    for tensor_type, records in sorted(rows_by_type.items()):
        bytes_values = [int(r["bytes"]) for r in records]
        lifetimes_s = [
            float(r["unpack_time_s"]) - float(r["pack_time_s"])
            for r in records
            if isinstance(r["pack_time_s"], int | float)
            and isinstance(r["unpack_time_s"], int | float)
        ]
        lifetimes_events = [
            int(r["unpack_index"]) - int(r["pack_index"])
            for r in records
            if isinstance(r["pack_index"], int)
            and isinstance(r["unpack_index"], int)
        ]
        layers = sorted({
            int(r["layer"])
            for r in records
            if isinstance(r["layer"], int)
        })
        shape_counts: defaultdict[str, int] = defaultdict(int)
        dtype_counts: defaultdict[str, int] = defaultdict(int)
        families: defaultdict[str, int] = defaultdict(int)
        for record in records:
            shape_counts[_shape_key(record["shape"])] += 1
            dtype_counts[str(record["dtype"])] += 1
            families[str(record["module_family"])] += 1
        summaries.append({
            "tensor_type": tensor_type,
            "module_families": "; ".join(
                f"{k} ({v})"
                for k, v in sorted(families.items(), key=lambda item: item[0])
            ),
            "layers": (
                f"{layers[0]}-{layers[-1]}"
                if layers and len(layers) > 1
                else str(layers[0]) if layers else ""
            ),
            "count": len(records),
            "unpacked": sum(1 for r in records if r["unpack_index"] is not None),
            "bytes_each_min_gb": min(bytes_values) / 1e9,
            "bytes_each_max_gb": max(bytes_values) / 1e9,
            "total_bytes_gb": sum(bytes_values) / 1e9,
            "lifetime_s_median": (
                statistics.median(lifetimes_s) if lifetimes_s else None
            ),
            "lifetime_s_max": max(lifetimes_s) if lifetimes_s else None,
            "lifetime_events_median": (
                statistics.median(lifetimes_events) if lifetimes_events else None
            ),
            "shapes": "; ".join(
                f"{k} ({v})"
                for k, v in sorted(
                    shape_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:6]
            ),
            "dtypes": "; ".join(
                f"{k} ({v})"
                for k, v in sorted(dtype_counts.items())
            ),
        })
    return summaries


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "tensor_type",
        "module_families",
        "layers",
        "count",
        "unpacked",
        "bytes_each_min_gb",
        "bytes_each_max_gb",
        "total_bytes_gb",
        "lifetime_s_median",
        "lifetime_s_max",
        "shapes",
        "dtypes",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(_fmt(row.get(col)) for col in columns) + " |")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+", help="Trace JSONL path(s) or globs.")
    parser.add_argument("--csv", type=str, help="Optional CSV output path.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    rows = summarize(_iter_paths(args.trace))
    if args.csv:
        write_csv(rows, Path(args.csv))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
