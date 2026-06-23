# Per-allocation GPU memory tracking

Records **every** CUDA caching-allocator event during one training step on one
GPU, each with its size and a captured stack ("what for"), then renders an
interactive memory-over-time timeline and a plain-text ledger.

Two event classes are tracked and kept distinct:

| class | snapshot actions | meaning |
| --- | --- | --- |
| **true driver alloc** | `segment_alloc` / `segment_free` | a real `cudaMalloc` / `cudaFree` (allocator reserves/returns a segment) |
| **reserved-space pseudo** | `alloc` / `free_requested` / `free_completed` | a tensor block carved out of / returned to an already-reserved segment |

Recording is continuous and per-event (an allocator callback fires on every
event); we only bracket the *dumped window* to one step so the ring buffer holds
a complete step.

## 1. Run training with recording on

Prefix your normal command with the `FASTVIDEO_MEM_SNAPSHOT_*` env vars:

```bash
FASTVIDEO_MEM_SNAPSHOT_DIR=outputs/mem_snapshots \
FASTVIDEO_MEM_SNAPSHOT_STEPS=5 \
FASTVIDEO_MEM_SNAPSHOT_RANK=0 \
examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --models.student.init_from Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --training.distributed.num_gpus 8 \
    --training.distributed.sp_size 8 \
    --training.distributed.hsdp_shard_dim 8 \
    --training.data.data_path data/synthetic_4k_14b_rung0 \
    --training.data.num_height 2160 --training.data.num_width 3840 \
    --training.data.num_frames 121 --training.data.num_latent_t 20 \
    --pipeline.flow_shift 5.0 \
    --training.loop.max_train_steps 10
```

This writes, for each captured step:

* `outputs/mem_snapshots/step5_rank0.pickle` — the full per-event history.
* `outputs/mem_snapshots/step5_rank0.regions.json` — per-phase
  `memory_stats` deltas (reserved/allocated growth and cudaMalloc/free counts).

### Env vars

| var | default | meaning |
| --- | --- | --- |
| `FASTVIDEO_MEM_SNAPSHOT_DIR` | unset (off) | output dir; setting it enables the feature |
| `FASTVIDEO_MEM_SNAPSHOT_STEPS` | `""` | 1-based steps to capture: `5`, `3,7`, or `all` |
| `FASTVIDEO_MEM_SNAPSHOT_RANK` | `0` | global rank that records ("one card") |
| `FASTVIDEO_MEM_SNAPSHOT_MAX_ENTRIES` | `1000000` | ring-buffer capacity (events) |
| `FASTVIDEO_MEM_SNAPSHOT_STACKS` | `all` | `all` (Python + C++) or `python` |

Phase labels recorded: `prepare_batch`, `dit_forward`, `loss`, `backward`,
`optimizer_step`, `zero_grad`.

## 2. Inspect the per-event ledger

```bash
python scripts/memory/dump_memory_ledger.py outputs/mem_snapshots/step5_rank0.pickle
# only real cudaMalloc/cudaFree:
python scripts/memory/dump_memory_ledger.py outputs/mem_snapshots/step5_rank0.pickle --kind real
# large reserved-space allocations only:
python scripts/memory/dump_memory_ledger.py outputs/mem_snapshots/step5_rank0.pickle --min-size 16MiB
```

Each line: event index, `REAL`/`PSD`, action, size, running reserved & live
totals, phase, and the top user stack frame.

## 3. Interactive memory-over-time timeline

```bash
python scripts/memory/plot_memory_timeline.py outputs/mem_snapshots/step5_rank0.pickle
# optional: label the run in the title, raise the block-alloc marker threshold
python scripts/memory/plot_memory_timeline.py outputs/mem_snapshots/step5_rank0.pickle \
    --config-label "expandable_segments" --min-marker-mib 4
```

…produces a single self-contained `.timeline.html` (Plotly embedded, opens
offline): filled **Reserved** (cudaMalloc pool) and **Allocated** (live tensor)
step-curves on a GiB-vs-ms axis, shaded per-phase bands, dashed step/warmup
dividers, an H200 capacity reference line, and a hoverable marker on every real
`cudaMalloc` and every sizable block alloc giving its size, phase, and
originating op. Zoom/pan auto-fits the Y axis; double-click restores autorange.
Snapshots dumped from the OOM observer (label or filename contains `oom`) get a
"CUDA OOM" marker at the death point. Peaks in the title come from the
authoritative allocator counters when the sidecar is present.

For an alternative interactive flamegraph view, drop the `.pickle` onto
<https://pytorch.org/memory_viz> (runs client-side).

## Notes / caveats

* **One card.** Recording is gated to `FASTVIDEO_MEM_SNAPSHOT_RANK`; with
  `sp_size=8` the model is sharded, so the snapshot reflects that one GPU's
  shard — which is what "on one card" means here.
* **Pick a steady-state step.** Step 1 includes one-time allocations (lazy
  optimizer state, cuDNN/attention workspaces). A mid-run step (e.g. 5) is more
  representative; `all` captures every step (larger output).
* **Activation checkpointing.** This run wraps the transformer in activation
  checkpointing, so `backward` re-runs the forward and re-allocates the
  activations — expect a second wave of forward-shaped allocations under the
  `backward` phase.
* **expandable_segments.** With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (see `reports/4k_14B_report.md`), real allocations use `cuMemMap` and appear as
  `segment_map` / `segment_unmap` rather than `segment_alloc` / `segment_free`;
  the ledger labels both as `REAL`.
* The existing torch profiler (`fastvideo/profiler.py`, `FASTVIDEO_TORCH_PROFILE*`)
  is separate and unaffected.
