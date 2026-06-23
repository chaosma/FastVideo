---
description: Capture and analyze per-allocation GPU memory traces of a training step
---

# GPU Memory Tracing SOP

How to record **every** CUDA allocation event on one rank during a training
step — each with a size and a stack trace ("what for") — and turn it into an
interactive memory-over-time timeline and a plain-text ledger. Use this to find
peaks, locate the op responsible for an allocation, or diagnose an OOM.

Implemented in `fastvideo/memory_tracking.py`; tooling in `scripts/memory/`.

## Steps

### 1. Choose a mode

- **Warmup/continuous** (`FASTVIDEO_MEM_SNAPSHOT_WARMUP=1`): recording starts at
  process entry and dumps once after the last selected step. The absolute curve
  is **exact** (no baseline reconstruction). Requires explicit steps. Prefer
  this for OOM diagnosis and absolute-footprint questions.
- **Bracketed** (default): records each selected step in isolation, one pickle
  per step. Lighter, but the pre-step baseline is reconstructed (imprecise
  across allocator backends). Fine for "what allocates inside this step".

### 2. Run training with recording on

Prefix the normal launch command with the env vars (one card only):

```bash
FASTVIDEO_MEM_SNAPSHOT_DIR=outputs/mem_snapshots \
FASTVIDEO_MEM_SNAPSHOT_STEPS=1,2 \
FASTVIDEO_MEM_SNAPSHOT_WARMUP=1 \
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

Env vars (see `fastvideo/envs.py`):

| var | default | meaning |
| --- | --- | --- |
| `FASTVIDEO_MEM_SNAPSHOT_DIR` | unset (off) | output dir; setting it enables the feature |
| `FASTVIDEO_MEM_SNAPSHOT_STEPS` | `""` | 1-based steps: `5`, `3,7`, or `all` |
| `FASTVIDEO_MEM_SNAPSHOT_WARMUP` | `0` | `1` = continuous/exact mode (needs explicit steps) |
| `FASTVIDEO_MEM_SNAPSHOT_RANK` | `0` | global rank that records ("one card") |
| `FASTVIDEO_MEM_SNAPSHOT_MAX_ENTRIES` | `1000000` | ring-buffer capacity (events) |
| `FASTVIDEO_MEM_SNAPSHOT_STACKS` | `all` | `all` (Python + C++) or `python` |

This writes `<label>_rank0.pickle` (full per-event history) and
`<label>_rank0.regions.json` (sidecar: per-phase `memory_stats` deltas plus
`memory_stats_start`/`memory_stats_end` counters). `<label>` reflects the window
(e.g. `warmup_step1-2`, `step5`, or a `*_oom` suffix if the OOM observer fired).

### 3. Analyze

```bash
# Ordered per-event ledger (REAL cudaMalloc/free vs reserved-space PSD events):
python scripts/memory/dump_memory_ledger.py outputs/mem_snapshots/<label>_rank0.pickle
python scripts/memory/dump_memory_ledger.py ... --kind real        # cudaMalloc/free only
python scripts/memory/dump_memory_ledger.py ... --min-size 16MiB   # large blocks only

# Interactive timeline (self-contained .timeline.html):
python scripts/memory/plot_memory_timeline.py outputs/mem_snapshots/<label>_rank0.pickle
python scripts/memory/plot_memory_timeline.py ... --config-label "expandable_segments"
```

The timeline shows filled Reserved/Allocated step-curves on a GiB-vs-ms axis,
shaded phase bands, step/warmup dividers, an H200 capacity line, an OOM marker
(when the snapshot is `_oom`), and a hoverable marker per cudaMalloc / sizable
block alloc giving size, phase, and originating op. Title peaks come from the
sidecar's authoritative counters. For an alternative flamegraph view, drop the
`.pickle` onto <https://pytorch.org/memory_viz> (client-side).

## How it works (reference)

- `torch.cuda.memory._record_memory_history(enabled="all", context="all",
  stacks=<env>)` installs the allocator callback; `_dump_snapshot` serializes the
  history. Each event records `{action, size, addr, stream, frames}`.
- Two event classes are kept distinct: **true driver allocs**
  (`segment_alloc`/`segment_free`, each a real `cudaMalloc`/`cudaFree`; or
  `segment_map`/`segment_unmap` under `expandable_segments:True`) vs.
  **reserved-space pseudo** (`alloc`/`free_requested`/`free_completed`).
- `mem_region(name)` pushes a `torch.profiler.record_function` range
  `fvmem::<name>` (captured into the snapshot's `external_annotations`) for the
  step phases: prepare_batch / dit_forward / loss / backward / optimizer_step /
  zero_grad. The recorder also brackets the window with `warmup` / `step{N}`.
- An OOM observer (`_cuda_attach_out_of_memory_observer`) dumps the in-flight
  recording just before `OutOfMemoryError` propagates, so an OOM run still yields
  a timeline up to the fatal allocation.
- Module API: `configure_from_env(rank)` (trainer), `begin_warmup_recording(rank)`
  (entrypoint, early), `get_recorder()`, `mem_region(name)`; recorder lifecycle
  `begin_warmup` / `on_step_begin` / `on_step_end` / `dump`. Wired in
  `train/entrypoint/train.py`, `train/trainer.py`, and
  `train/methods/fine_tuning/finetune.py`.

## Caveats

- **One card.** Recording is gated to `FASTVIDEO_MEM_SNAPSHOT_RANK`; with
  `sp_size=8` the model is sharded, so the trace is that one GPU's shard.
- **Encoders are not in-step.** `prepare_batch` reads precomputed
  `text_embedding` / `vae_latent`; the heavy phases are `dit_forward` and
  `backward`.
- **Activation checkpointing.** `backward` re-runs the forward, so expect a
  second wave of forward-shaped allocations under the `backward` phase.
- **Pick a steady-state step.** Step 1 includes one-time allocations (lazy
  optimizer state, cuDNN/attention workspaces); a mid-run step is more
  representative.
- The torch profiler (`fastvideo/profiler.py`, `FASTVIDEO_TORCH_PROFILE*`) is
  separate and unaffected.

## Related

A memory knob came out of this work: `FASTVIDEO_ADAM_FUSED=1` selects the fused
AdamW kernel (in-place update, no large foreach intermediate buffers;
`fastvideo/train/utils/optimizer.py`), which requires the per-param `step` tensor
on-device (`fastvideo/train/methods/base.py`).
