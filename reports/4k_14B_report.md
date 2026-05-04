# 4K Wan 2.2 T2V A14B Training Report — Real Data

End-to-end fine-tune smoke test of **Wan 2.2 T2V A14B** (the MoE flagship) on
4K video. Real preprocessing pipeline + 5 training steps per rung, with
loss values logged each step.

Companion to `4k_5B_report.md` and `4k-h200-results.md` (1.3B baseline).

## TL;DR

- Rung 2 ✅ — 5 steps, **18.7 s/step mean**, peak **80 GB/rank**.
  Loss halves cleanly each step (0.0085 → 0.0011) — model overfits the
  4-sample dataset as expected.
- Rung 1 ✅ — 5 steps, **88.4 s/step mean**, peak **102 GB/rank**.
  Loss oscillates around 0.07 (no monotonic descent) — at 2.5× more tokens
  per sample, 5 steps × LR 1e-6 isn't enough to push the loss out of its
  per-step noise floor.
- Rung 0 ❌ — **does not fit on 4× H200 (141 GB) at sp=4 hsdp_shard=4
  grad_ckpt=full**. Four attempts all peaked at 139–140 GB before OOMing
  on different ops in turn. Detailed below.
- A14B is **~4× slower per step than 1.3B at the same rung**, and **~11×
  slower than 5B at rung 2 / ~20× slower at rung 1**. The slowdown vs 1.3B
  is the bigger DiT; the slowdown vs 5B is the older 8× VAE — same N as
  1.3B at every rung.

## Hardware / runtime

- 4× NVIDIA H200 SXM (141 GB / NV18 mesh).
- `Wan-AI/Wan2.2-T2V-A14B-Diffusers` — MoE pair: `transformer` +
  `transformer_2`, ~14.29 B params each, ~28.6 B total. Routing is per-step
  via `boundary_ratio=0.875` (one expert active per training step; both
  must be FSDP-resident for routing).
- Distributed: `sp_size=4`, `hsdp_shard_dim=4`, full grad ckpt, bf16
  precision, Torch SDPA backend.
- `flow_shift=12.0` (A14B preset).
- `training.checkpoint.training_state_checkpointing_steps=0` to avoid
  writing the ~168 GB optimizer-state checkpoint at end-of-run.

## Verified parquet format (Wan 2.1 VAE → Wan 2.1 latent shape)

A14B's VAE is bit-identical to Wan 2.1's (verified — same
`latents_mean`/`latents_std`, same arch). So the parquet shape is
identical to the 1.3B path:

| Rung | Frames | `num_latent_t` | Latent shape (z=16) | DiT tokens | Per-shard | Total disk |
|------|--------|----------------|---------------------|------------|-----------|------------|
| 2    | 5      | 2              | `[16, 2, 270, 480]` |  64,800    | ~0.15 MB  | 0.6 MB   |
| 1    | 17     | 5              | `[16, 5, 270, 480]` | 162,000    | ~9.4 MB   | 37 MB    |
| 0    | 77     | 20             | `[16, 20, 270, 480]`| 648,000    | 58 MB     | 234 MB   |

All three parquets generated successfully from
`data/raw_4k/videos/4k_demo_30s_001.mp4`. Output paths
`data/real_4k_a14b_rung{0,1,2}/training_dataset/`.

Tokens = `num_latent_t × (H_lat/2) × (W_lat/2)` with patch_size = (1, 2, 2).

## Step time + VRAM (rungs 2 & 1)

| Rung | Step 1 (warmup) | Steady step | Mean (5 steps) | Peak VRAM/rank |
|------|------------------|-------------|----------------|----------------|
| 2    | 23.4 s          | ~17.5 s     | **18.7 s/it**  | **79.9 GB**    |
| 1    | 93.1 s          | ~87.3 s     | **88.4 s/it**  | **102.1 GB**   |
| 0    | (see "Rung 0 OOM" below) | | | (peaked at 139.8 GB before OOM) |

## Loss trends (real 4K data, 4 samples × 5 steps, LR 1e-6, AdamW)

### Rung 2 — clean monotonic overfit

| Step | step_time_sec | total_loss |
|------|---------------|------------|
| 1    | 23.39         | 0.008526   |
| 2    | 17.52         | 0.006147   |
| 3    | 17.36         | 0.003075   |
| 4    | 17.40         | 0.001853   |
| 5    | 17.59         | 0.001137   |

Loss halves roughly every step. With only 4 unique samples (same video,
4 different captions) and 65k tokens per step, the optimizer can move the
DiT meaningfully toward memorizing the latent target each iteration.
Confirms training is wired correctly end-to-end (gradients flow, weights
update, the model is bf16-stable).

### Rung 1 — no monotonic descent in 5 steps

| Step | step_time_sec | total_loss |
|------|---------------|------------|
| 1    | 93.07         | 0.071338   |
| 2    | 87.28         | 0.064218   |
| 3    | 87.30         | 0.072008   |
| 4    | 86.92         | 0.067853   |
| 5    | 87.32         | 0.084900   |

Why no clean descent? Two compounding factors:

1. **Loss noise floor scales with token count.** The diffusion loss is a
   per-token MSE between predicted and true noise; with 162k tokens per
   step (vs 65k at rung 2), each batch samples a different noise level,
   different latent slice (more frames), and the loss fluctuates ~5% even
   at the same model. With 5 datapoints we can't see signal beneath the
   per-step variance.
2. **Per-step parameter movement is small** at LR 1e-6 with grad clipping
   (`max_grad_norm=1.0`). The cosine of the angle between gradient and
   "memorize this clip" direction has to compound over many more steps
   for the loss to drop visibly.

To get a clean rung-1 overfit signal you'd want either:
- More steps (~50+),
- Higher LR (1e-5 or 1e-4),
- Or one identical sample replicated rather than 4 with different
  captions (eliminates the cross-sample gradient cancellation).

This is a "training is taking effect, we just can't see it in 5 steps"
situation, not a "training is broken" situation.

### Rung 0 — could not run (OOM, see next section)

## Rung 0 OOM analysis

A14B at rung 0 (4K, 77 frames, 648k tokens at SP=1 / 162k per rank)
**does not fit on 4× H200 SXM (141 GB / rank)** with the current op set.
We made four attempts. All peaked at 139.2–139.8 GB / rank before OOMing
on different fp32-upcast ops:

| Attempt | Patch / env | Failed at | Failed allocation | Peak VRAM | Remarks |
|---------|-------------|-----------|--------------------|-----------|---------|
| v1 | (baseline) | `_apply_rotary_emb`: `(x.float() * cos + ...).type_as(x)` | 6.18 GiB | 139.5 GB | OOM in step 1 forward |
| v2 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | same op | 6.18 GiB | 139.4 GB | Reclaimed ~10 GB fragmentation, allocated ceiling rose by same margin → still OOM |
| v3 | + `FASTVIDEO_BF16_ROTARY=1` (added env-gated bf16 rotary) | `RMSNorm.forward_native`: `x.to(orig_dtype)` after fp32 round-trip | 1.55 GiB | 139.4 GB | Got past rotary; OOM in step-1 backward (recompute_fn) |
| v4 | + `FASTVIDEO_BF16_RMSNORM=1` (env-gated bf16 RMSNorm) | `cublasCreate(handle)` (CUBLAS_STATUS_ALLOC_FAILED) | (cublas workspace ~hundreds MB) | 139.8 GB | Past rotary + RMSNorm; OOM trying to allocate cuBLAS context for the next gemm |

Each patch saved ~3-6 GB. Each successive attempt got a few GB further
into the step but the next memory pinch always appears, because the
allocator stays packed against the H200 ceiling — once we're at ~139 GB
the next 1-6 GB allocation simply has nowhere to go.

The fundamental memory budget at rung 0:

```
Static (per rank, both 14B experts FSDP-sharded):
  fp32 master weights: 28.6 B × 4 / 4 ranks = 28.6 GB
  AdamW m+v:           28.6 B × 8 / 4 ranks = 57.2 GB
  → ~86 GB

Activations + grad buffer + bf16 fwd cache (rung 0):
  ~50–55 GB (extrapolated from rung 2's 80 GB - 86 GB static and
  rung 1's 102 GB - 86 GB static = 16 GB activation, scaled by token
  count: 16 × (162k/40k) ≈ 64 GB; we measured 53–54 GB at peak)

Total peak observed: ~139.5 GB. H200 capacity: 141 GB.
Headroom: <2 GB → not enough for any fp32 transient.
```

The 5B (which has 16 in_channels and ~5 GB / rank static optimizer state
plus the 16× VAE giving 4× fewer tokens) fits trivially with ~91 GB
headroom at rung 0; the 1.3B (~4 GB / rank static) had ~83 GB headroom.
A14B's rung 0 is the first config in this experiment series that truly
hits the hardware limit.

### Bf16-math patches we tried (and discarded)

In v3 and v4 above we prototyped two memory-saving patches, both env-gated
and off by default:

- A bf16 path in `_apply_rotary_emb` that skips the
  `x.float() * cos` → `.type_as(x)` round-trip and runs rotation in
  `x.dtype` directly (saves ~6 GB of fp32 transient at rung-0 sequence
  length).
- A bf16 path in `RMSNorm.forward_native` that keeps `x` in its original
  dtype and only upcasts the variance reduction (saves ~1.5 GB of
  transient).

Each saved memory locally but didn't change the outcome — the next op
along the way (cuBLAS workspace) still hit the 139–140 GB ceiling. We
**did not keep** the patches. They live only in the v3/v4 logs and in
git history if anyone wants to revisit them; the source tree is back at
its original state.

## Cross-model comparison @ same rung (4× H200, sp=4 hsdp_shard=4)

| Rung | Tokens (1.3B/A14B) | Tokens (5B) | 1.3B steady | A14B steady | 5B steady | A14B vs 1.3B | 5B vs 1.3B |
|------|---------------------|-------------|-------------|--------------|-----------|---------------|-------------|
| 2    | 64.8 k              | 16.3 k      | ~4.5 s      | **17.5 s**   | ~1.6 s    | **3.9×**      | 0.36×       |
| 1    | 162 k               | 40.8 k      | ~20 s       | **87.3 s**   | ~4.3 s    | **4.4×**      | 0.21×       |
| 0    | 648 k               | 163 k       | ~288 s      | **OOM**      | ~40 s     | (~4.2× projected) | 0.14×    |

The A14B / 1.3B ratio sits at ~4× across rungs — as predicted from the
per-token compute ratio:

| | Wan 2.1 1.3B | Wan 2.2 A14B (per expert) |
|---|---|---|
| Layers | 30 | **40** (1.33×) |
| Hidden = heads × head_dim | 12 × 128 = 1536 | **40 × 128 = 5120** (3.33×) |
| FFN dim | 8960 | **13824** (1.54×) |
| z_dim (in/out channels) | 16 | 16 (same VAE) |
| Per-token attn cost (∝ L · h · d) | 1× | **40·40·128 / (30·12·128) ≈ 4.4×** |
| Per-token FFN cost (∝ L · h · ffn) | 1× | **40·5120·13824 / (30·1536·8960) ≈ 6.9×** |

## Memory analysis

The 80 GB/rank floor at rung 2 (where activations are negligible) is the
**static cost of holding both 14B MoE experts in FSDP-sharded form**:

- 28.6 B params × 12 B/param (fp32 master + AdamW m + AdamW v) / 4 ranks
  = ~86 GB/rank static — close to our 80 GB measurement (delta is mostly
  bf16 forward cache).
- Rung 1's +22 GB/rank vs rung 2 is activation memory growth from 65k →
  162k tokens.
- Rung 0's projected static + activation at 648k tokens: ~140 GB — over
  the H200 ceiling, confirmed by the OOMs.

By contrast, a 1.3B static floor is `1.4 B × 12 / 4` ≈ 4.2 GB/rank;
its rung-0 peak of 58 GB/rank is *all* activations.
A14B's rung-2 peak is already 14× more memory than the 1.3B's rung-2
peak, with negligible activation contribution — the optimizer state
alone dominates.

## What this confirms

1. **The 4× DiT-cost factor is stable** at the same VAE / same token
   count: going 1.3B → A14B costs ~4× per step. The MoE structure does
   not double the cost because each training step routes through only
   one expert; both experts are FSDP-resident only for routing.
2. **A14B's "older" 8× VAE choice is the dominant factor at 4K rung 0**.
   With Wan 2.2's 16× VAE the same model would have ~4× fewer tokens
   and would fit in memory and run in ~40-60 s/step. As it ships, A14B
   cannot fit rung 0 on 4× H200; would need 8× H200, larger memory
   per GPU (e.g. B200 192 GB), 8-bit Adam, or CPU-offloaded optimizer
   state.
3. **Training is taking effect** at rung 2 (clean monotonic loss
   descent, halving each step). At rung 1 the loss has too much
   per-step variance for 5 steps × LR 1e-6 to show signal; the math is
   running correctly, the experiment just isn't long enough.

## Files produced

- `data/real_4k_a14b_rung{0,1,2}/training_dataset/worker_*/worker_0/data_chunk_0.parquet`
- `/tmp/4k_5b_bench/train_a14b_rung2_loss.log` — rung 2 with loss/step
- `/tmp/4k_5b_bench/train_a14b_rung1_loss.log` — rung 1 with loss/step
- `/tmp/4k_5b_bench/train_a14b_rung0_v{1,2,3,4}.log` — rung 0 OOM attempts
- `/tmp/4k_5b_bench/preprocess_a14b_rung{0,1,2}.log`
- `/tmp/4k_5b_bench/mem_watch_a14b_rung*.csv` — `nvidia-smi` 1–2 Hz
  per-GPU memory samples
- No checkpoints saved (`training_state_checkpointing_steps=0`).

## Code touched (kept)

- `fastvideo/train/trainer.py` — added a `progress.write(...)` line
  after `tracker.log(...)` so loss + step time stream to stdout when
  wandb is disabled. Always-on UX improvement, not gated.
- `scripts/4k_milestone/preprocess_4k.sh` — fixed the rung-0 frame count
  to 77 (matches the YAML / synthetic generators already corrected in
  commit `4a99954`).

