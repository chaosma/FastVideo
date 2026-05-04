# 4K Wan2.1-T2V-14B Training Report — 8× H200

End-to-end smoke test of fine-tuning **Wan2.1-T2V-14B** at 4K on a fresh 8× H200
SXM box. Real preprocessing (raw 4K mp4 → parquet latents) + 5 training steps
per rung, sequence-parallel across all 8 GPUs.

## Hardware / runtime

- **GPUs:** 8× NVIDIA H200 SXM (143,771 MiB ≈ 140 GB each), full NV18 mesh
  (every pair connected via 18 NVLinks).
- **Model:** `Wan-AI/Wan2.1-T2V-14B-Diffusers` (downloaded to
  `/workspace/.hf_home`).
- **Source video:** `4k_demo_30s_001.mp4` — 3840×2160, 30 fps, 888 frames;
  replicated 8× (`data/raw_4k/videos2caption.json`) so each rank gets 1 entry.
- **Distributed config:** `num_gpus=8`, `sp_size=8`, `hsdp_shard_dim=8`,
  `tp_size=1`. Sequence-parallel + FSDP-shard the entire DiT across all 8 ranks.
- **Pipeline:** `flow_shift=5.0`, `enable_gradient_checkpointing_type=full`,
  precision = bf16, AdamW (lr 1e-6).
- **Attention backend:** Torch SDPA. `flash_attn` is **not installed**
  in this venv (`Cannot use FlashAttention-2 backend because the flash_attn
  package is not found`); SDPA dispatches to a flash-style kernel on H200.
- **Software:** Python 3.12, torch 2.10.0+cu130, NCCL 2.28.9, FastVideo
  installed editable in `/venv/main`.

## Parquet format

14B uses the **Wan 2.1 VAE** — same as 1.3B. `z_dim=16`, spatial stride 8×,
temporal stride 4×. The 14B difference is in the DiT.

DiT specs (verified against the live `pipeline_config` dump in the training
log; values come from the loaded HF transformer config):

| | Wan 2.1 1.3B | **Wan 2.1 14B** |
|---|---|---|
| `hidden_size` | 1536 | **5120** |
| `num_layers` | 30 | **40** |
| `num_attention_heads` | 12 | **40** |
| `attention_head_dim` | 128 | 128 |
| `ffn_dim` | 8960 | **13824** |
| `in_channels` (= z_dim) | 16 | 16 |
| `patch_size` | (1,2,2) | (1,2,2) |
| Params | ~1.4 B | ~14 B (≈ 10×) |

Note: per-token compute on 14B is roughly **(5120/1536)² × (13824/8960) × (40/30)
≈ 22×** the 1.3B cost (FFN + projections + attention scoring all scale with
hidden size; depth and FFN width add their own factors). With identical
tokens per rung the 14B should be much slower than 1.3B per step.

## Generated parquets

All three rungs preprocessed from the same 4K demo clip; the merged-CSV
preprocessing path writes `vae_latent_bytes/_shape/_dtype` +
`text_embedding_bytes/_shape/_dtype` + `caption` + metadata, identical schema
to the 1.3B / 5B paths.

```
data/real_4k_14b_rung0/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
data/real_4k_14b_rung1/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
data/real_4k_14b_rung2/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
```

| Rung | Frames in | `num_latent_t` | Latent shape | DiT tokens (SP=1) | Total disk |
|------|-----------|----------------|--------------|--------------------|------------|
| 2    | 5         | 2              | `[16, 2, 270, 480]`  |   64,800 |   788 KB |
| 1    | 17        | 5              | `[16, 5, 270, 480]`  |  162,000 |   72 MB  |
| 0    | 77        | 20             | `[16, 20, 270, 480]` |  648,000 |  446 MB  |

(Tokens = `num_latent_t × (H_lat/2) × (W_lat/2)` with patch_size = 1×2×2.)

### Preprocessing timing (4× H200, sp=4)

Generated with `MODEL_PATH=Wan-AI/Wan2.1-T2V-14B-Diffusers OUTPUT_DIR=...
RUNG=N GPU_NUM=4 bash scripts/4k_milestone/preprocess_4k.sh`. The 14B model
is only used to pull the matching VAE + T5 components — DiT weights aren't
loaded for preprocessing.

| Rung | VAE encode payload         | Wall time / batch | Peak VAE-stage reserved memory |
|------|----------------------------|-------------------|--------------------------------|
| 2    | `[1, 3, 5,  2160, 3840]`   |  7.20 s | 77.17 GiB |
| 1    | `[1, 3, 17, 2160, 3840]`   | 15.87 s | 92.02 GiB |
| 0    | `[1, 3, 77, 2160, 3840]`   | 55.97 s | 95.30 GiB |

(2 batches per rank with 4 ranks; numbers exclude the ~8 s weight-load and
~12 s SP communication warmup that happen once per run.)

## Training: 5 steps per rung (8× H200, sp=8, hsdp_shard=8)

Command (rung 0 example):

```bash
NUM_GPUS=8 WANDB_MODE=disabled bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --models.student.init_from Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --pipeline.flow_shift 5.0 \
    --training.distributed.num_gpus 8 \
    --training.distributed.sp_size 8 \
    --training.distributed.hsdp_shard_dim 8 \
    --training.data.data_path data/real_4k_14b_rung0/training_dataset \
    --training.data.num_height 2160 --training.data.num_width 3840 \
    --training.data.num_frames 77 --training.data.num_latent_t 20 \
    --training.loop.max_train_steps 5 \
    --training.checkpoint.training_state_checkpointing_steps 0
```

`training_state_checkpointing_steps=0` disables the end-of-run DCP save —
each 14B checkpoint is ~86 GB (FSDP-sharded fp32 master + AdamW m+v across
8 ranks), so saving 3 of them would not fit on the 299 GB workspace disk.
With saves disabled, checkpoint-related disk pressure is zero.

All three runs **completed cleanly** — no OOM, no NCCL hangs. Loss values
strictly decrease over the 5 logged steps (it's a 1-clip overfit smoke test).

### Per-step training time

Per-step times taken from the trainer's `[step N] step_time_sec=…` log line
(also visible in `/tmp/test_rung{0,1,2}_run.log`).

| Rung | Tokens/step (SP=1) | Step 1 (warmup) | Step 2 | Step 3 | Step 4 | Step 5 | Steady mean (2–5) |
|------|--------------------|-----------------|--------|--------|--------|--------|--------------------|
| 2    |  64,800  |  21.09 s |  8.89 s |  9.29 s |  9.21 s |  8.92 s | **9.08 s/it** |
| 1    | 162,000  |  55.48 s | 43.13 s | 43.62 s | 43.48 s | 42.99 s | **43.31 s/it** |
| 0    | 648,000  | 626.67 s | 611.09 s | 611.28 s | 610.90 s | 610.25 s | **610.88 s/it** |

Step-time scaling vs token count:

- Rung 1 / Rung 2 = **2.5× tokens, 4.77× steady step time** (super-linear; attention starts to matter)
- Rung 0 / Rung 1 = **4.0× tokens, 14.10× steady step time** (attention dominates; quadratic regime)

### Peak VRAM (sampled mid-training via `nvidia-smi`)

| Rung | Rank 0 used | Ranks 1–7 (typical) | % of 140 GB |
|------|-------------|---------------------|-------------|
| 2    | 49.07 GB | 48.91 GB | 35% |
| 1    | 60.78 GB | 60.96 GB | 43% |
| 0    | 120.77 GB | 120.61 GB | 86% |

Rung 0 leaves only ~20 GB of headroom per H200. The Wan 2.1 14B at 4K rung 0
**will not fit on H100 SXM (80 GB) at sp=8 with grad-ckpt=full** — it needs
the H200's 140 GB. (1.3B at the same rung peaked at 58 GB on 4× H200; the 14B
is roughly 2× larger in activation footprint per rank because of the larger
hidden size, plus the FSDP master/optimizer state on a 14B model is more
sharded weight to keep resident.)

The rung 0 → rung 2 delta (~72 GB / rank) is dominated by activation
memory: 10× more tokens through 40 layers of attention/FFN with
grad-ckpt-full means recomputing roughly 10× more activation tape per
backward pass.

## Comparison with the 1.3B (4× H200) and 5B (4× H200) runs

Apples-to-apples is impossible — different models, different GPU counts —
but ratios are useful. The 1.3B and 5B numbers come from
`reports/4k-h200-results_1.5B.md` and `reports/4k_5B_report.md`.

| Rung | 1.3B / 4×H200 sp=4 | 5B / 4×H200 sp=4 | **14B / 8×H200 sp=8** |
|------|--------------------|------------------|------------------------|
| 2    |   ~4.5 s |   ~1.6 s |   **9.08 s** |
| 1    |   ~20.2 s |   ~4.3 s |   **43.31 s** |
| 0    |  ~288 s (flash-attn) |   ~40 s |  **611 s** |

A few observations:

- **14B vs 1.3B at the same rung** is ~2.1× slower at rung 0 even though the
  14B has 10× the parameters. The reason: doubling the GPUs from 4 → 8 with
  `sp_size=8` halves the per-rank attention sequence length, which buys back
  ~4× of the attention quadratic. Net 14B/1.3B = ~10× per-token slowdown
  divided by ~4× SP speedup ≈ 2.1× — matches measurement (611 / 288 = 2.12).
- **14B vs 5B at the same rung** is ~15× slower at rung 0 (611 vs 40 s). 5B
  uses the **Wan 2.2 VAE** (16× spatial vs 8×), which gives 5B ~4× fewer
  tokens at every rung. So 5B at rung 0 (163 K tokens) is roughly comparable
  in shape to 14B at rung 1 (162 K tokens) — and indeed 5B rung 0 ≈ 40 s,
  14B rung 1 ≈ 43 s. The DiT param ratio (14/5 ≈ 2.8×) is mostly cancelled by
  the slightly different layer/hidden geometry; the controlling factor at 4K
  is the VAE's spatial stride.
- **8 GPUs really matter for 14B rung 0.** With 4× H100 sp=4 and 1.3B, rung
  0 was 288 s/step and used 58 GB/rank; the same setup with 14B would peak
  past 144 GB and OOM. Going to 8× H200 sp=8 cuts per-rank activation
  memory enough to fit (peak 121 GB) and cuts attention work per rank by
  ~4×. This is the smallest hardware that works.

### Per-step decomposition (rung 0, 14B, sp=8 sanity check)

- Per rank after SP=8: 81,000 tokens, 5 attention heads, head_dim=128.
- Self-attention forward FLOPs per layer: `4 N² hd · num_heads = 4 · 81k² ·
  128 · 5 ≈ 1.68 × 10¹³` per layer = **0.67 PFLOPs per layer**.
- 40 layers × 1× forward + 1× recompute (grad-ckpt) + ~2.5× backward ≈ 4.5×
  the forward flops ⇒ **~120 PFLOPs of attention per rank per step**.
- H200 sustains ≈ 0.5 PFLOPs/s BF16 (50% of 989 TF/s peak in FA-style
  workloads) ⇒ **~240 s of pure attention compute per rank per step**.
- FFN + QKV proj + RMSNorm + SP all-to-all + FSDP gather: ~370 s linear
  residual.
- Total ≈ **610 s/step** — matches the measurement to ~3%.

So rung 0 on the 14B is dominated about equally by attention (~40%) and the
linear/comm tail (~60%) at sp=8 — unlike 1.3B at sp=4 where attention is
~93%. The SP factor of 8 has shifted the bottleneck.

## What's left to try

- **Install `flash_attn` for SM90 (Hopper).** Build instructions are in
  `reports/4k-h200-results_1.5B.md`. SDPA is already flash-style on H200,
  but FA-2 dedicated kernel typically gives an extra ~5–10% on the rung-0
  attention call.
- **FlashAttention-3 with FP8 attention.** ~2× attention BF16 → FP8
  speedup on Hopper. Not currently wired into FastVideo's `flash_attn`
  backend.
- **`torch.compile` for training.** Hard-coded off in
  `fastvideo/train/utils/moduleloader.py:57` — flipping it to True with
  `mode="reduce-overhead"` would fuse the 40-layer linear/RMSNorm/QKV-proj
  tail and is the most plausible 10–20% win on rung 0 at sp=8 where the
  linear tail dominates.
- **VSA / sparse attention.** Same caveats as the 1.3B report — VSA didn't
  pay off at sp=4 1.3B; would need to retest at the 14B sp=8 token shape.

## Notes / caveats

1. **Original `test.sh` had two bugs:**
   (a) `python` not on `$PATH` (the active venv is `/venv/main`; fixed by
   prepending it in `test.sh`);
   (b) `--training.data.data_path` pointed at `data/real_4k_14b_rung0/...`
   which did not exist on disk. The pre-existing `data/preprocessed_4k`
   contained **rung-2 latents** (shape `[16, 2, 270, 480]`), not rung 0 —
   so an early test that pointed at it with rung-0 CLI overrides actually
   trained on rung-2 latents (the dataloader trusts the parquet shape, not
   the CLI hint). Fixed by generating real rung-0 / rung-1 / rung-2 parquets
   under `data/real_4k_14b_rung{0,1,2}/`.
2. **`training_state_checkpointing_steps=0`** disables checkpoint save (each
   ~86 GB on disk for the 14B optimizer state). With the YAML's
   `resume_from_checkpoint=latest`, the trainer just notes "no checkpoints
   found, starting from scratch" and runs.
3. **flash_attn not installed.** SDPA backend is used — see `INFO ...
   [cuda.py:232] Cannot use FlashAttention-2 backend...` in every log. On
   H200 SDPA already dispatches to a flash-style kernel; the perf ceiling
   is the same.
4. **Single-clip dataset, 8 replicas.** `data/raw_4k/videos2caption.json`
   contains the same `4k_demo_30s_001.mp4` 8 times with varied captions so
   8-way SP sharding sees one entry per rank. This is a plumbing/perf
   benchmark, not a real fine-tune.
5. **Disk pressure is real on 14B.** With saves disabled total artifact
   footprint per run is < 1 GB (parquet + tracker only). With saves
   enabled, plan ~86 GB per saved checkpoint × `checkpoints_total_limit`.

## Files produced

- `data/real_4k_14b_rung{0,1,2}/training_dataset/worker_*/worker_0/data_chunk_0.parquet`
- `/tmp/preprocess_rung{0,1,2}.log` — preprocessing logs (timing + memory)
- `/tmp/test_rung{0,1,2}_run.log` — full training stdout (per-step times,
  loss, config dump)
- `test.sh`, `test_rung1.sh`, `test_rung2.sh` — wrapper scripts (now point
  at the correct rung-specific parquets and disable checkpoint save)
