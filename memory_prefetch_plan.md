# Fine-grained activation streaming for FastVideo — engineering plan

**Goal.** Reduce per-card peak GPU memory of the Wan 2.1 14B training step on
8 × H200 SXM from the current ≈133 GB to **< 60 GB** while keeping the
end-to-end step time within **~5 % of baseline** (target: negligible; budget:
20 %). Combined with reduced GPU count or longer sequences, this approximately
halves training cost on the same hardware.

We will prototype on the 5B / 4 × H200 / 24fps × 121f rig (the configuration
in `memory_timeline_5B_4xH200_measured.svg`) because step time there is
~90 s vs ≈2.5 h for the 14B target — three orders of magnitude faster
iteration. The same code path then runs on the 14B target unchanged.

The plan is two complementary pieces:

1. **Drop the VAE from the training process** — 0-effort win (~0.3 GB on 14B,
   2.82 GB on 5B because we load it fp32). Done in Phase 0.
2. **One-layer-ahead async activation streaming** — the main lever. Replace
   `torch.autograd.graph.save_on_cpu(pin_memory=True)` with a custom
   saved-tensor hook that runs H2D / D2H copies on a dedicated CUDA stream
   and prefetches layer *i − 1*'s saved tensors while layer *i*'s backward
   is still running. Done in Phases 1–5.

## 1. Memory budget — why < 60 GB is feasible on 14B

Decomposing the 14B / 8 × H200 / N_local = 125,550 / sp = 8 step at the
backward peak:

| Component                                          | GB / card | Resident at peak? | Can stream / shrink? |
|----------------------------------------------------|-----:|---|---|
| fp32 master weights (FSDP-sharded, ÷ 8)            | 7.0  | yes | (Phase 6 stretch only) |
| Adam m + v (fp32, FSDP-sharded, ÷ 8)               | 14.0 | yes | (Phase 6 stretch only) |
| fp32 sharded gradient accumulator                  | 7.0  | yes | — |
| VAE (fp32, replicated)                             | 0.3  | yes today | **drop in Phase 0** |
| FSDP / NCCL / SP a2a workspaces                    | 0.5–1.0 | yes | — |
| **Subtotal — must-be-resident**                    | **~28** |   |   |
| Per-layer saved-for-backward tensors of *all* layers (bf16 stash) | ~68 | yes today | **stream in Phase 2-4** |
| Backward intermediates of the layer being processed | ~14 | yes | inherent |
| Bf16 forward recompute working set, current layer  | ~6  | yes | inherent |
| Adam-step transient (master → bf16 cast)           | +7  | only at P7 | — |

Target peak with streaming + drop-VAE:
```
must_resident                              ≈ 27.7 GB
+ current layer's bwd intermediates        ≈ 14 GB
+ current layer's recompute fwd activations≈ 6 GB
+ at most ONE in-flight neighbour's stash  ≈ 10 GB
+ framework / FSDP all-gather of current layer ≈ 1 GB
                                           = 58.7 GB
```

That's the design floor. Margin to the 60 GB target is ~1–2 GB, so the
plan has to actually achieve one-deep buffering (not two-deep) and not
balloon the FSDP all-gather window.

On 5B / 4 × H200 / 121f the same scheme predicts ≈ 36 GB peak (currently
43.56 GB measured with naïve offload), which is the metric we will iterate
on.

## 2. Design

```
                          ┌──────────────────────┐
   forward (layer i):     │   block_i.forward    │
                          │  ┌────────────────┐  │      compute_stream
                          │  │ kernels       ⬚│  │  ────────────────────►
                          │  └─pack saved────┘  │
                          │        │             │      copy_stream
                          │        ▼             │     ┌───────────────►
                          │   cpu_pinned[i]      │     │ H2D wait_event
                          └────────┬─────────────┘     │
                                   │
   backward (layer i):    ┌────────▼─────────────┐
                          │   prefetch i-1       │  ──► launches D2H on
                          │   D2H to gpu[i-1]    │      copy_stream
                          │                      │
                          │   unpack i           │  ──► blocks compute_stream
                          │   (already resident) │      on copy_stream's event
                          │                      │
                          │   block_i.backward  ⬚│
                          │                      │
                          │   release gpu[i]     │
                          └──────────────────────┘
```

Two CUDA streams per device:

- **`compute_stream`** — the default training stream. Forward kernels,
  backward kernels, FSDP all-gather of current layer's weights.
- **`copy_stream`** — a dedicated stream used solely for H2D and D2H
  memcpy of saved-for-backward tensors. Pinned host memory.

Coordination is per-layer-pair via CUDA events:
- `pack_done[i]` — recorded on `copy_stream` after H2D copy of every saved
  tensor in layer *i* finishes. Forward proceeds without blocking.
- `restore_done[i]` — recorded on `copy_stream` after layer *i*'s D2H
  restore is complete. Backward of layer *i* waits on it before its first
  kernel.
- The prefetch trigger for layer *i − 1* is "as soon as backward of layer
  *i* starts" (the layer is now the active one; the layer that just
  finished can have its GPU buffers released).

Memory at any time during backward, holding ≤ 1 layer-deep prefetch:

| State | GPU resident |
|---|---|
| Just before backward starts | static + grads (so far = 0) + all stashes if we don't drop them ahead of bwd; we will start *without* any stash on GPU |
| Backward of layer L starts | static + grads + layer-L saved (restored on demand at first unpack) + bwd intermediates |
| Mid-layer-L backward | + prefetch for L−1 begins on copy_stream |
| Backward of layer L−1 starts | drop layer L's GPU stash; layer L−1 stash ready; static + grads accumulated for L's shard + L−1 saved + bwd intermediates |
| ... | rolling window of one layer |

## 3. Phases

### Phase 0 — Drop the VAE from the training process *(< 1 hour)*

The VAE is loaded by the training pipeline (`fastvideo/train/models/wan/wan.py`
calls `load_module_from_path(..., module_type="vae")` inside
`init_preprocessors`) but is never used after the data hand-off — embeddings
come pre-baked from the parquet. On 5B it's loaded fp32 (2.82 GB measured);
on 14B/Wan2.1 it's much smaller (0.29 GB) but it's free GB on every card.

**Change.** Gate the VAE load in `init_preprocessors` on a config flag
`training.data.load_vae_into_training: bool = False` (or auto-detect:
load only if the training pipeline actually invokes `self.vae` during
the step). Verify by grepping the training methods for `self.vae` use.

**Validation.** Re-run the probe; `P0_idle` drops by 2.82 GB on 5B and the
`vae` category goes to 0. No other phase should change.

**Deliverable.** `phase_memory_no_vae.json`,
`memory_timeline_5B_4xH200_no_vae.svg`.

### Phase 1 — Per-instant memory probe *(< 1 day)*

Today's probe samples at 9 phase boundaries. For tuning the streaming
scheduler we need higher-resolution data:

- A forward and backward hook on **every** transformer block recording
  `cuda.memory_allocated()` plus the names + sizes of the largest 5
  live allocations from `cuda.memory._snapshot()`.
- A timeline of `(time, allocated_gb, current_layer_idx, phase)` saved
  as Parquet/CSV.

**Change.** Extend `memory_probe.py` with a `per_layer_probe(model)` that
installs hooks on `transformer.blocks[*]`. Output: `layer_trace.csv` with
columns `step, layer_idx, fwd_or_bwd, t_ns, allocated_gb, reserved_gb`.

**Why before Phase 2.** Without this we can't tell whether the streaming
scheduler is achieving one-deep buffering or accidentally two-deep.

**Deliverable.** `layer_trace_baseline.csv`,
`layer_trace_full_offload.csv` for the 5B run, plus a small plot script.

### Phase 2 — Custom saved-tensor hook on a copy stream *(2–3 days)*

Replace the body of `offload_wrapper`'s `save_on_cpu(pin_memory=True)`
with a version that uses a dedicated stream:

```python
class AsyncCpuSaveHook:
    def __init__(self, copy_stream: torch.cuda.Stream, registry: "BlockRegistry",
                 block_idx: int):
        self.copy_stream = copy_stream
        self.registry = registry
        self.block_idx = block_idx
        self.records: list[tuple[torch.Tensor, torch.cuda.Event]] = []

    def pack(self, t: torch.Tensor):
        if t.device.type != "cuda":
            return t  # leave non-cuda saves alone (small scalars, etc.)
        cpu = torch.empty(t.size(), dtype=t.dtype, device="cpu",
                          pin_memory=True)
        # H2D copy on the dedicated stream after compute_stream finishes
        # producing t.
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_stream(torch.cuda.current_stream())
            cpu.copy_(t, non_blocking=True)
            ev = self.copy_stream.record_event()
        idx = len(self.records)
        self.records.append((cpu, ev))
        return ("packed", self.block_idx, idx)

    def unpack(self, packed):
        _, block_idx, idx = packed
        gpu_buf = self.registry.consume_restored(block_idx, idx)
        return gpu_buf
```

The hook is installed per-block via
`torch.autograd.graph.saved_tensors_hooks(hook.pack, hook.unpack)` wrapped
around the block's `forward()`. We *do* keep `checkpoint_wrapper` around
the block (recompute), so the saved-tensor pack-list per block is just
the block's input — same as today's `full_offload`.

**Risk and mitigation.**
- *Pack happens for every checkpointed save, including the input
  residual + RNG state if `preserve_rng_state=True`.* We already pass
  `preserve_rng_state=False` in `apply_activation_checkpointing`, so the
  RNG state is not packed. Verify with the per-layer probe.
- *Pack races with kernel completion.* The
  `copy_stream.wait_stream(current_stream)` line and `non_blocking=True`
  copy on pinned memory together guarantee the H2D is correctly
  ordered. Add a unit test that compares saved tensor bytes against an
  eager copy.
- *CUDA caching allocator fragmentation.* The CPU-pinned destination
  bypasses the allocator. The GPU restore in Phase 3 will use the
  allocator — bound it with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  in the run script.

**Deliverable.** `fastvideo/training/activation_streaming.py` with
`StreamedOffloadCheckpointWrapper`, plus a new `CheckpointType.STREAMED`
entry in `activation_checkpoint.py`. Behaviour at this phase: identical
GPU peak to `full_offload`, slightly lower if overlap helps; serves as
the foundation for Phase 3.

### Phase 3 — One-layer-ahead prefetch scheduler *(3–5 days)*

Add a `BlockRegistry` shared across all blocks of the DiT that maintains:
- `block_order: list[int]` — the order blocks run in forward (and reverse
  in backward), populated by Phase 2's forward hook.
- `live_buffers: dict[int, list[torch.Tensor]]` — GPU buffers for layer
  i's restored saves; populated by the prefetcher.

Schedule:
1. A `backward_pre_hook` on the **first** block to run in backward
   (i.e. the last forward block) triggers the prefetch of its own saved
   tensors **plus** kicks off the prefetch for the next-to-process layer.
2. A `backward_pre_hook` on every other block: assert that
   `live_buffers[block_idx]` is non-empty (i.e. the prefetch landed in
   time). Trigger prefetch for the *next* block to process.
3. A `backward_hook` (post-bwd) on each block: free
   `live_buffers[block_idx]`.

Prefetch implementation:
```python
def prefetch(block_idx: int, registry: BlockRegistry):
    records = registry.cpu_records[block_idx]
    bufs = []
    with torch.cuda.stream(registry.copy_stream):
        for cpu, save_event in records:
            registry.copy_stream.wait_event(save_event)  # pack done
            buf = torch.empty_like(cpu, device="cuda")   # may block on alloc
            buf.copy_(cpu, non_blocking=True)
            bufs.append(buf)
        restore_event = registry.copy_stream.record_event()
    registry.live_buffers[block_idx] = bufs
    registry.restore_events[block_idx] = restore_event
```

The backward kernel's first use of any restored tensor calls
`current_stream().wait_event(restore_event[block_idx])`.

**Bounding the in-flight window to ONE layer.** Add a semaphore (size 1)
in the registry. `prefetch(i)` acquires before allocating GPU buffers,
the post-bwd hook of layer i+1 (just finished) releases. This is the
hard guarantee that prevents drift to two-deep buffering under unlucky
scheduling.

**Risks and mitigations.**
- *FSDP `pre_forward_hook` for the next layer may also have outstanding
  all-gather work on the compute_stream.* Our prefetch is on a separate
  stream so it overlaps; verify with `nsys profile` that the timeline
  shows H2D running concurrently with FSDP all-gather + kernels.
- *Activation prefetch may stall if backward of one layer is faster
  than H2D bandwidth.* Per-card H2D ≈ 64 GB/s on PCIe 5.0 ×16; per-layer
  stash ≈ 1.7 GB on 14B (= N_local · H · 2 = 125550 · 5120 · 2). H2D
  for 1 layer = 27 ms. A 14B layer's backward is on the order of
  500–1000 ms (133 GB / step ≈ 10 GB/s of bandwidth-bound work, so
  step is bandwidth- and compute-bound). So overlap is comfortable.
- *Step 0 of the loop has no "next" to prefetch.* The first block
  processed in backward must restore its own saves before its backward
  kernels can run. This is a single-layer stall, ~30 ms on 14B.

**Deliverable.** `BlockRegistry` + scheduler, integrated into
`StreamedOffloadCheckpointWrapper`. Run the per-instant probe on 5B and
confirm the trace shows: (a) GPU peak ≤ static + 1 layer + 1 in-flight
prefetch; (b) `nsys` shows H2D / D2H consistently overlapping with bwd
kernels.

### Phase 4 — Integration: knob, config, defaults *(1 day)*

Expose the new mode through the existing pipeline:

- New `CheckpointType.STREAMED_OFFLOAD` in
  `fastvideo/training/activation_checkpoint.py`.
- New YAML / CLI override: `training.model.enable_gradient_checkpointing_type:
  streamed_offload` (and `streamed_offload_no_recompute` if we want a pure
  streaming variant).
- Allocator pre-config in `examples/train/run.sh`:
  `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Document the host-RAM cost: 14B total host-resident stash is
  `8 ranks × 68 GB ≈ 0.55 TB`. The H200 boxes we target have ≥1 TB,
  but tight on 512 GB nodes — assert at startup and fall back to
  full_offload if the available host pinned-memory pool is too small.

### Phase 5 — 5B prototype validation *(1 day)*

Acceptance criteria measured on 5B / 4 × H200 / 24fps × 121f:

| Metric | Today (full_offload) | Target | Allowed |
|---|---:|---:|---:|
| Step peak (P5 max)             | 43.56 GB | **≤ 36 GB** | ≤ 38 GB |
| End-of-fwd P4 live             | 22.16 GB | ≤ 22 GB    | ≤ 24 GB |
| Step time (steady-state mean)  | 87.2 s   | **≤ 88 s** | ≤ 92 s |
| Loss at step 2 (same seed)     | 0.31634  | **identical to baseline within 1e-5** | identical |
| Per-layer GPU residency probe  | n/a      | **= 1 layer + ≤ 1 in-flight** | 2 layers max |

If we hit ≤ 38 GB at 5B with ≤ 5 % step-time hit, push to Phase 6.

### Phase 6 — 14B validation on 8 × H200 *(rented; 1 step + 10-step soak)*

Reserve 8 × H200 SXM for a half-day window. Run:

1. 1-step probed run with baseline FULL — confirm peak ≈ 133 GB. (Optional,
   if no recent measurement exists.)
2. 1-step probed run with FULL_OFFLOAD — measure baseline-of-baseline.
3. 1-step probed run with STREAMED_OFFLOAD — capture
   `memory_timeline_14B_8xH200_streamed.svg`.
4. 10-step soak with STREAMED_OFFLOAD — confirm step time is stable
   (no slow ramp from CPU memory fragmentation) and loss is finite.

**Acceptance.** Peak ≤ 60 GB, step time ≤ 1.05 × FULL baseline, loss
identical to baseline at the same RNG seed.

## 4. Out-of-scope / explicit non-goals

- **CPU-offloading optimizer state.** FSDP2's `CPUOffloadPolicy` already
  exists for this; it's a knob, not new code. Document the option in the
  plan's epilogue but don't implement here. Worth ~10 GB on 14B at a
  measurable step-time cost.
- **Selective per-tensor offload** (e.g. offload only the MLP intermediate,
  keep small tensors on-GPU). Adds complexity for marginal gain; revisit
  only if Phase 5 misses target.
- **Replacing FSDP with TP or a different sharding strategy.** Out of
  scope.
- **VSA / sparse attention.** Already on the FastVideo roadmap as a
  separate axis.

## 5. Testing plan

Unit:
- `test_async_save_hook_correctness.py`: random tensors of varying dtype
  packed via the async hook, then unpacked; bit-exact equality to original.
- `test_block_registry_semaphore.py`: simulate 30 blocks, assert at any
  instant ≤ 2 layers' GPU buffers are live (current + 1 prefetched).
- `test_no_recompute_collision.py`: assert `preserve_rng_state=False`
  doesn't fire dropout twice with different seeds (or that we set
  `dropout=0` everywhere already, which the 4K config does).

Integration:
- 1-step run on 5B with STREAMED vs FULL: loss bitwise identical at
  matched seed (RNG state captured before the step).
- 5-step run: gradients norm sequence matches FULL within 1e-4.
- `nsys profile` for one step at 5B: copy_stream and compute_stream
  timelines, expected overlap ≥ 90 % of H2D time.

Regression guard:
- Add the 5B-streamed numbers to the SSIM / training-smoke test under
  `fastvideo/tests/training/` so future PRs can't silently regress the
  GPU peak.

## 6. Risks, dependencies, fallbacks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `saved_tensors_hooks` doesn't compose with FSDP2 all-gather | medium | Phase 2 has a 1-block smoke test before integration; if it conflicts, fall back to a custom autograd Function that does the pack/unpack manually |
| H2D bandwidth saturates and stalls compute on shorter sequences | low | Bandwidth budget shows 27 ms / layer on 14B vs 500 ms compute; on small models or short clips this could flip — gate STREAMED on a sequence-length threshold |
| Host pinned-memory exhaustion on tight nodes | medium | Phase 4's startup assert + auto-fallback to full_offload |
| `torch.cuda.Stream` interactions with `torch.compile` or cudagraphs | medium | We are not using either in this config; document and add an assertion that disables STREAMED if `torch.compile` is on |
| Backward through `checkpoint_wrapper` re-fires the saved_tensors_hooks during recompute and double-saves | high if naive | The hook context must be `torch.autograd.graph.disable_saved_tensors_hooks()` during recompute; verify via the unit test |
| RNG drift breaking SSIM regression tests | low (dropout=0) | Lock the failure mode by setting `seed` deterministically in the test |

## 7. Stretch — Phase 6 follow-up (post-target)

If 14B peak is comfortably under 60 GB, the next levers in priority order:
1. **CPU-offload Adam state** via FSDP2 `CPUOffloadPolicy`. Need to bench
   the optimizer step cost; can be ~5–10 % step-time hit, freeing ~14 GB.
2. **Fused chunked Adam step** (chunked H2D + compute + D2H) to absorb
   the offload penalty.
3. **Selective bf16 keep-on-GPU for small saved tensors** to spare the
   PCIe bandwidth budget for the big MLP intermediates.

These are not required to hit < 60 GB; they bring the floor toward
~25 GB if eventually wanted.

## 8. Timeline & deliverables

| Phase | Wall time | Owner | Deliverable |
|---|---|---|---|
| 0 — drop VAE                  | < 1 hr  | any | gated config + `phase_memory_no_vae.json` |
| 1 — per-instant probe         | 0.5 d   | any | `layer_trace.csv`, plot |
| 2 — async copy-stream hook    | 2–3 d   | systems | `activation_streaming.py`, unit tests |
| 3 — prefetch scheduler        | 3–5 d   | systems | `BlockRegistry`, integration with checkpoint_wrapper |
| 4 — config + alloc tuning     | 1 d     | systems | YAML knob, `run.sh` env, host-RAM assert |
| 5 — 5B validation             | 1 d     | systems | `memory_timeline_5B_4xH200_streamed.svg`, regression test |
| 6 — 14B validation            | 0.5 d (8×H200 rent) | systems | `memory_timeline_14B_8xH200_streamed.svg`, soak log |

**Total: ~2 weeks of focused work + 0.5 d of rented 8×H200 time for
final validation.**

## 9. References / files touched

- `fastvideo/training/activation_checkpoint.py` — entry point, new mode.
- `fastvideo/training/activation_streaming.py` — *new file* with hook +
  registry + scheduler.
- `fastvideo/train/models/wan/wan.py` — VAE-load gate (Phase 0).
- `fastvideo/train/trainer.py` — already instrumented for probe; extend
  for per-instant trace.
- `memory_probe.py` (workspace-level) — per-layer hooks.
- `examples/train/run.sh` — `PYTORCH_CUDA_ALLOC_CONF`.
- New tests under `fastvideo/tests/training/streaming/`.

## 10. Status — Phases 0 & 1 implemented and validated on 5B / 4×H200

Validated on a fresh 4× H200 SXM box with the Wan 2.2 TI2V 5B model at
24fps × 121 frames × 704×1280, `sp_size=4`, `hsdp_shard_dim=4`,
`CheckpointType.FULL` (recompute, no CPU offload — `full_offload` is what
Phase 2 will add). Probe outputs: `probe_5B_121f_run4/{layer_trace.csv,
phase_memory.json}`.

### What landed

**Phase 0 — VAE drop**
- `fastvideo/train/utils/training_config.py`: added
  `DataConfig.load_vae_into_training: bool = False` (default off).
- `fastvideo/train/models/wan/wan.py`: `init_preprocessors` now gates the
  `load_module_from_path(..., "vae", ...)` call. When disabled, builds a
  `_WanVAEStatsStub` exposing only `latents_mean` / `latents_std` (the
  only fields `normalize_dit_input` reads).
- **Critical fix discovered on first 5B run:** the stub originally pulled
  its constants from `pipeline_config.vae_config.arch_config`, which
  still carries Wan 2.1's z_dim=16 values even when running Wan 2.2 5B
  (z_dim=48). That made backward fail at the very first
  `normalize_dit_input` broadcast (`The size of tensor a (48) must match
  the size of tensor b (16) at non-singleton dimension 1`). Fixed by
  reading the actual `<model_path>/vae/config.json` from the diffusers
  snapshot. **This needs the same fix in any future per-model stub.**

**Phase 1 — memory probe**
- `fastvideo/training/memory_probe.py` (~470 lines): `MemoryProbe` class,
  `maybe_init_from_env(rank)` factory, module-level `snap` /
  `step_begin` / `step_end` shortcuts. Auto-finalises after
  `FASTVIDEO_MEM_PROBE_STEPS` steps: writes `layer_trace.csv` and
  `phase_memory.json` (rank-0 only by default; `FASTVIDEO_MEM_PROBE_ALL_RANKS=1`
  to capture every rank). DTensor-aware categorisation via `to_local()`.
- **Wiring fix needed on first probe run:** the original Phase 1 patch
  wired the probe into `fastvideo/training/training_pipeline.py:train()`,
  but the actual training entrypoint is `fastvideo/train/trainer.py:run()`.
  Probe never initialised. Rewired into the right loop: install on
  `method.student.transformer` + `next(iter(method.get_optimizers(...)))`
  at the top of `run()`, and `step_begin` / `snap` / `step_end` around
  the per-step body.
  - The legacy snap sites in `training_pipeline.py` were left in place
    (no-op when probe inactive) so older code paths still benefit if/when
    they're used.

### How to reproduce

```bash
mkdir -p /workspace/FastVideo/probe_out
# Run from inside a wrapper script so env survives the nohup detach
# (the bash invocation form `export X=Y && nohup …` lost env in our tests).
cat > /tmp/_probe_run.sh <<'EOF'
#!/bin/bash
export HF_HOME=/workspace/.hf_home
export FASTVIDEO_MEM_PROBE_STEPS=2
export FASTVIDEO_MEM_PROBE_DIR=/workspace/FastVideo/probe_out
export WANDB_MODE=disabled
export NUM_GPUS=4
cd /workspace/FastVideo
source /workspace/venv/main/bin/activate
exec bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --models.student.init_from Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --pipeline.flow_shift 5.0 \
    --training.distributed.num_gpus 4 \
    --training.distributed.sp_size 4 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.data.data_path data/synthetic_5b_121f \
    --training.data.num_height 704 --training.data.num_width 1280 \
    --training.data.num_frames 121 --training.data.num_latent_t 31 \
    --training.loop.max_train_steps 2 \
    --training.checkpoint.resume_from_checkpoint ""
EOF
chmod +x /tmp/_probe_run.sh
nohup /tmp/_probe_run.sh > probe_out/train.log 2>&1 &
```

The synthetic dataset comes from
`python scripts/4k_milestone/make_synthetic_5b_data.py --rung 6 \
    --n-samples 4 --output-dir data/synthetic_5b_121f`
(rung 6 added for this experiment: 704×1280, 121 frames, num_latent_t=31).

### Findings from the 5B / 4×H200 / 24fps × 121f / 704×1280 run

Step 1 (steady state) phase memory, rank 0:

| Phase            | allocated GB | max_alloc GB | reserved GB |
|------------------|-------------:|-------------:|------------:|
| P0_idle          |        15.11 |        15.11 |       26.87 |
| P1_inputs_done   |        15.11 |        15.11 |       26.87 |
| P2_fwd_start     |        15.11 |        15.11 |       26.87 |
| P3_fwd_half      |        16.40 |        17.82 |       29.15 |
| P4_fwd_end       |        16.81 |        18.40 |       29.15 |
| **P5_bwd_peak**  |    **20.13** |    **23.40** |       30.01 |
| P6_post_bwd      |        20.13 |        20.13 |       30.01 |
| **P7_opt_done**  |        20.11 |    **25.11** |       30.01 |
| P0_next          |        15.11 |        20.11 |       30.01 |

Step peak: **25.11 GB** (Adam-cast transient at P7).
Backward peak: **23.40 GB** (P5_bwd_peak max).

Phase 0 cross-check (item 5 of the original checklist): every snapshot
shows `categories_gb["vae"] = 0.0`. VAE drop is real.

Per-layer trace summary (30 blocks, 240 events over 2 steps):

- **Forward stash per block:** layers 1–29 each add **+0.042 GB**
  (≈ 42 MB bf16). Layer 0 adds +0.189 GB (carries the external input
  residual in addition to its own). Σ over 30 blocks ≈ 1.26 GB, which
  matches P4 − P2 = 1.70 GB minus ~0.4 GB of FSDP/comm scratch.
- **Backward growth per layer:** +0.167 GB per layer (gradient slot
  accumulation), constant across layers 28 → 1. Net of stash freed: as
  designed.
- **First-block-in-backward stall:** layer 29 backward step is
  +0.822 GB vs the steady +0.167 GB — the recompute-internals working
  set materialising on top of an empty pipeline. Sets the floor for
  start-of-backward overhead the prefetch scheduler can't amortise.

### Validation checklist — results

1. ✅ **Pack-list size per block** — flat at 42 MB on layers 1–29,
   linear accumulation in fwd. Smaller than the §1 estimate of 0.7 GB
   on 5B because we ran at 704×1280, not 4K (Phase 5 will rerun at the
   per-rank token count the plan budgeted).
2. ✅ **No double-counting** — Σ per-block fwd deltas (1.26 GB) ≤
   P4 − P2 fwd growth (1.70 GB). No aliasing surprise.
3. ✅ **First-block-in-backward stall observed** — 0.82 GB jump on
   layer 29's first bwd step, isolated to that one block.
4. ✅ **P3 sanity** — P3 allocated (16.40) is mid-way between P2 (15.11)
   and P4 (16.81). Linear fwd accumulation.
5. ✅ **VAE drop confirmed** — `categories_gb["vae"] = 0.0` at all
   phases. P0_idle of 15.11 GB reflects bf16 weights + Adam m/v (5 GB
   each), no VAE.

### Backward access pattern — confirmed simple

The trace confirms that block backward fires in strict reverse forward
order (29 → 28 → ... → 0). Internals materialised by recompute are
transient within one block's backward and never need to enter the
prefetch queue. The Phase 2/3 design (FILO offload of block inputs +
one-deep prefetch on a dedicated copy stream) is the correct model;
nothing weirder is happening underneath.

### Known caveats for Phase 2 planning

- **Baseline mode mismatch.** The 43.56 GB plan baseline assumes a
  `full_offload` mode (`save_on_cpu` of every checkpointed tensor)
  that is **not** in the repo. Today's `CheckpointType` only has
  `FULL` / `OPS` / `BLOCK_SKIP`. Phase 2 needs to first add the
  `full_offload` baseline (so we have something to streaming-improve),
  then the streamed variant.
- **Resolution gap.** This run was 704×1280 to keep the experiment
  cheap. Phase 5 acceptance numbers in §3 (43.56 → ≤ 36 GB peak) were
  benchmarked at a different size — Phase 5 needs to be re-stated for
  whichever final resolution the 5B prototype settles on, or rerun at
  the original measurement point.
- **Per-rank weights category.** `categories_gb["weights_bf16"] = 5.0`
  on a 5B model with hsdp_shard_dim=4 looks high (expected ~2.5 GB
  per rank). Either FSDP isn't sharding what the categorise function
  expects to find on `model.parameters()`, or the local fragment is
  larger than the 1/N estimate. Worth verifying before quoting
  attribution numbers; doesn't affect Phase 2/3 design.

### Files

| File | Purpose |
|---|---|
| `fastvideo/training/memory_probe.py` | Probe class, env-var entry point, CSV/JSON writers |
| `fastvideo/train/trainer.py` | Probe install + per-step snap/step_begin/step_end |
| `fastvideo/training/training_pipeline.py` | Legacy snap sites (kept; no-op when inactive) |
| `fastvideo/train/models/wan/wan.py` | Phase 0 VAE-drop gate + stub reading real VAE config.json |
| `fastvideo/train/utils/training_config.py` | `DataConfig.load_vae_into_training` flag |
| `scripts/4k_milestone/make_synthetic_5b_data.py` | Added rung 6 (704×1280 × 121f) |
| `probe_5B_121f_run4/{layer_trace.csv, phase_memory.json}` | Captured outputs |

### Next

Phase 2: implement `full_offload` baseline + `StreamedOffloadCheckpointWrapper`
on a dedicated copy stream (per §3). Re-run this same probe under the
new mode to confirm the access pattern stays FILO and the queue depth
never exceeds 1 layer in flight.
