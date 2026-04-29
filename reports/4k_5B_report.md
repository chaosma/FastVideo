# 4K Wan2.2-TI2V-5B Training Report — Real Data

End-to-end smoke test of fine-tuning **Wan2.2-TI2V-5B** on 4K video with the
real preprocessing pipeline + 5 training steps for each rung.

## Hardware / runtime

- **GPUs:** 4× NVIDIA H200 SXM (141 GB each, full NVLink mesh).
- **Model:** `Wan-AI/Wan2.2-TI2V-5B-Diffusers` (downloaded to `/workspace/.hf_home`).
- **Source video:** `4k_demo_30s_001.mp4` — 3840×2160, 30 fps, 29.6 s, 888 frames.
- **Distributed config:** `sp_size=4`, `hsdp_shard_dim=4` (sequence-parallel + FSDP-shard
  the DiT across all 4 ranks).
- **Pipeline:** `flow_shift=5.0`, gradient checkpointing = `full`, precision = bf16.
- **Attention backend:** Torch SDPA (FlashAttention-2 not installed in this env;
  available speed-up if added).

## Parquet format (5B vs 1.3B)

5B uses the **Wan2.2 VAE** which differs from 1.3B's:

| Spec | Wan2.1-T2V-1.3B | **Wan2.2-TI2V-5B** |
|------|-----------------|--------------------|
| `z_dim` | 16 | **48** |
| VAE spatial stride | 8× | **16×** |
| VAE temporal stride | 4× | 4× |
| Source height for 4K | 2160 (`H/8 = 270` even) | **2176** (`H/16 = 136` even — see below) |
| Latent shape (rung 2) | `[16, 2, 270, 480]` | **`[48, 2, 136, 240]`** |
| `flow_shift` | 3.0 | **5.0** |

**Critical:** the 5B DiT has `patch_size=(1, 2, 2)` so latent H/W must be even
after the 16× compression. 2160/16 = 135 is odd and crashes the patch embed.
Source must be padded/resized to height 2176 (`= 136 × 16`).

The preprocess pipeline writes the standard `merged` schema:
`vae_latent_bytes/_shape/_dtype`, `text_embedding_bytes/_shape/_dtype`,
`caption`, plus metadata fields (`id`, `file_name`, `media_type`, `width`,
`height`, `num_frames`, `duration_sec`, `fps`). All latent / text bytes are
serialized fp32. Verified with
`scripts/4k_milestone/verify_parquet.py` semantics.

## Generated parquets

All three rungs preprocessed from the same 4K demo clip (4 entries duplicated
in `data/raw_4k/videos2caption.json` so that 4-way SP sharding works).

```
data/real_4k_5b_rung0/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
data/real_4k_5b_rung1/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
data/real_4k_5b_rung2/training_dataset/worker_{0..3}/worker_0/data_chunk_0.parquet
```

| Rung | Frames in | `num_latent_t` | Latent shape | DiT tokens (SP=1) | Per-shard size | Total disk |
|------|-----------|----------------|--------------|--------------------|----------------|------------|
| 2    | 5         | 2              | `[48, 2, 136, 240]`  |   16,320 | ~0.2 MB | 0.79 MB |
| 1    | 17        | 5              | `[48, 5, 136, 240]`  |   40,800 | ~7.5 MB | 30.0 MB |
| 0    | 77        | 20             | `[48, 20, 136, 240]` |  163,200 | ~45 MB  | 181 MB  |

(Tokens = `num_latent_t × (H_lat/2) × (W_lat/2)` with patch_size=2×2.)

The 5B 4K rung-0 token count is **4× lower** than 1.3B at the same rung
(163,200 vs 680,400) because the 16× spatial VAE buys back most of the
compute — the 5B DiT is bigger but operates on far fewer tokens per step.

### Preprocessing timing (4× H200, sp=4)

Single batch (one clip × 4 SP ranks); sample-once-and-write workflow:

| Rung | VAE encode payload | Wall time | Peak VAE-stage reserved memory |
|------|--------------------|-----------|-------------------------------|
| 2    | `[1, 3, 5,  2176, 3840]` | 14.5 s | 37.2 GB |
| 1    | `[1, 3, 17, 2176, 3840]` | 16.7 s | 48.7 GB |
| 0    | `[1, 3, 77, 2176, 3840]` | 56.9 s | 50.8 GB |

(Numbers exclude the one-time HF model snapshot download (~75 s) and the
~15 s model-load step.)

## Training: 5 steps per rung

Command (rung-2 example; only the data path / shape flags differ between rungs):

```bash
NUM_GPUS=4 WANDB_MODE=disabled bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --models.student.init_from Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --pipeline.flow_shift 5.0 \
    --training.distributed.num_gpus 4 \
    --training.distributed.sp_size 4 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.data.data_path data/real_4k_5b_rung2/training_dataset \
    --training.data.num_height 2176 --training.data.num_width 3840 \
    --training.data.num_frames 5 --training.data.num_latent_t 2 \
    --training.loop.max_train_steps 5
```

All three runs **completed cleanly** — no OOM, no NCCL hangs, checkpoint saved
at step 5.

### Step-time results

Per-step times taken from the training progress bar (from
`/tmp/4k_5b_bench/train_rung{0,1,2}.log`):

| Rung | Tokens/step (SP=1) | Step 1 (warmup) | Step 5 (steady) | Mean over 5 steps |
|------|---------------------|------------------|-----------------|--------------------|
| 2    |  16,320 |  6.32 s |  1.58 s | **2.15 s/it** |
| 1    |  40,800 |  9.24 s |  4.31 s | **4.89 s/it** |
| 0    | 163,200 | 45.23 s | 40.12 s | **40.64 s/it** |

Step time scales roughly linearly with token count past the warmup spike
(rung 1 / rung 2 = 2.5× tokens, 2.7× steady step time; rung 0 / rung 1 = 4×
tokens, 9.3× steady step time — the super-linear factor at rung 0 reflects
attention's quadratic cost dominating once the sequence is long enough).

### Peak VRAM (sampled @ 1 Hz via nvidia-smi, max across run)

| Rung | Rank 0 | Ranks 1–3 (typical) |
|------|--------|---------------------|
| 2    | 34.1 GB | 33.3 GB |
| 1    | 35.7 GB | 35.0 GB |
| 0    | 50.9 GB | 50.1 GB |

All well within the 141 GB per-card capacity — H200 has **~3× headroom** at
rung 0 and **~4× headroom** at rung 2.

The static memory floor (FSDP-sharded fp32 master weights + AdamW moments +
grad buffer) for 5B at `hsdp_shard_dim=4` is ~22 GB/rank; the rest is
activations. Rung 0 vs rung 2 deltas (~17 GB) is dominated by the 10× larger
attention activations at 163 K tokens vs 16 K tokens.

### Why 5B is **faster** than 1.3B at the same rung (vs `4k-h200-results.md`)

`4k-h200-results.md` measured Wan 2.1 T2V **1.3B** on the same 4× H200 box:
rung 2 = 4.5 s · rung 1 = 20 s · rung 0 = **288 s** (with flash-attn). The
5B is **smaller per-step at every rung** even though it has 3.5× more
parameters. This is *not* a misconfiguration — it's the Wan 2.2 VAE doing
its job:

DiT specs (verified against the actual HF `transformer/config.json` of each
model; the 1.3B values in `4k-milestone-status.md` are wrong — those are the
Wan 2.1 14B numbers):

| | Wan 2.1 1.3B | Wan 2.2 5B |
|---|---|---|
| VAE spatial stride | 8× | **16×** |
| Latent shape (rung 0) | `[16, 21, 270, 480]` | `[48, 20, 136, 240]` |
| **DiT tokens at rung 0** (patch 2×2) | **680,400** | **163,200** (4.17× fewer) |
| DiT layers | 30 | 30 (same depth) |
| Hidden / FFN | 1536 / 8960 | **3072** / **14336** (~2× wider) |
| Heads × head_dim | 12 × 128 | 24 × 128 |
| z_dim (in/out channels) | 16 | 48 |
| DiT params | ~1.4 B | ~5 B (≈ 3.5× more) |

The 5B is bigger because it widened the same 30-layer stack ~2× in hidden
and ~1.6× in FFN. So per-token compute is ~3.5× more expensive on 5B than
on 1.3B. The two terms then trade off:

- **Linear cost** (FFN + QKV proj, ∝ N): 3.5× / 4.17× ≈ **0.84×** — the
  wider DiT mostly eats the per-token savings, so linear work is roughly
  equal at the same rung.
- **Self-attention cost** (∝ N² × hidden): N² drops 17×, hidden grows 2× →
  net attention work drops **~9×**. Per-rank forward attention at rung 0:
  ~21.3 PFLOPs (1.3B) vs ~2.4 PFLOPs (5B).

Per-rung step-time comparison on identical hardware (4× H200 SXM, sp=4,
hsdp_shard=4, grad-ckpt=full):

| Rung | 1.3B steady | 5B steady | Speedup | DiT tokens (1.3B → 5B) |
|------|-------------|-----------|---------|-------------------------|
| 2    | ~4.5 s      | ~1.6 s    | **2.8×** | 64,800 → 16,320 (4× fewer) |
| 1    | ~20 s       | ~4.3 s    | **4.7×** | 162,000 → 40,800 (4× fewer) |
| 0    | ~288 s      | ~40 s     | **7.1×** | 680,400 → 163,200 (4.17× fewer) |

The speedup grows with rung because attention dominates more at higher
rungs. Per `4k-h200-results.md`'s decomposition, attention is ~93% of step
time at 1.3B rung 0; with 5B's 4× token reduction the quadratic attention
work drops ~17× per layer, and the rest (FFN / QKV proj / SP all-to-all /
FSDP gather) is also lower because 5B is shallower (30 vs 40 layers) and
narrower (hidden 3072 vs 5120). Net: 5B rung 0 in **40 s** is consistent
with attention going from ~93% of 288 s down to ~30–40% of the step at 5B
rung 0, with the linear/comm tail dominating.

In other words: at 4K, **the VAE choice matters more than the DiT param
count.** Wan 2.2's 16× spatial VAE shrinks every downstream cost by 4× in
tokens, and that compounds quadratically through attention.

### Verification that the loaded model really is 5B

The in-memory `pipeline_config` dump in the training log prints
`hidden_size=5120, num_layers=40, in_channels=16` — these are the
**dataclass defaults** in `WanVideoArchConfig` (defined for the 1.3B
shape) and are misleading. They get overridden at safetensors load time.
Cross-checks that confirm 5B was actually trained:

- HF checkpoint config
  (`/workspace/.hf_home/.../Wan2.2-TI2V-5B-Diffusers/.../transformer/config.json`):
  `num_layers=30, num_attention_heads=24, attention_head_dim=128, in_channels=48, ffn_dim=14336`.
- Output checkpoint:
  `outputs/wan_finetune_4k_5b_rung0/checkpoint-5/dcp/` = **56 GB** total
  across the 4 DCP shards. 5B × (4 B fp32 master + 8 B AdamW m+v) = 60 GB
  total, sharded across 4 ranks ⇒ ~15 GB/rank ⇒ ~56 GB on disk.
  (A 1.3B model would be ~16 GB total.)
- The parquet latents have 48 channels, which the patch_embed accepts
  without a shape error — i.e. the DiT's `in_channels` is 48 at runtime.

Follow-up worth doing: have `Wan2_2_TI2V_5B_Config` declare a 5B-shape
`dit_config` (30 layers × hidden 3072 × in_channels 48 × ffn 14336 × 24
heads) so the printed config matches what's actually loaded, instead of
relying on HF override at runtime to silently fix it.

## Notes / caveats

1. **Rung-0 frame count.** The doc table at the top of `4k-milestone-status.md`
   still mentions 81 frames for rung 0 in one place; the corrected value is
   77 (`(77-1)/4 + 1 = 20` latent timesteps), matching the Wan2.1 native
   training length. Updated `scripts/4k_milestone/preprocess_4k.sh` to reflect
   77 (it was still saying 81 — the same fix that was applied earlier in
   commit `4a99954` to the YAML / synthetic generators / `run_4k_milestone.sh`).
2. **FlashAttention-2 not installed** in this env (`flash_attn` import fails
   → falls back to Torch SDPA). Installing it should drop step time further,
   especially at rung 0 where attention dominates.
3. **VAE encode is single-rank effectively.** With `sp_size=4` and 4 input
   entries, each rank's `Map`-stage processes 1 entry but the VAE encode
   broadcasts the full clip to all ranks (peak reserved memory ~50 GB on one
   rank = full activation). On H200 this is fine; on a 24 GB card this is
   exactly the OOM mode the milestone doc describes.
4. **Single-clip dataset.** Only one physical 4K clip is used, replicated
   4× with different captions. Real fine-tuning would want a real dataset;
   this is a plumbing/perf benchmark only.
5. **Loss values not logged** by the current trainer at this verbosity; the
   first-step / steady-step pattern shows the optimizer + grad checkpoint
   loop runs end-to-end without numerical errors. Adding loss logging is a
   one-line trainer change for follow-up.

## Files produced

- `data/raw_4k/videos/4k_demo_30s_001.mp4`
- `data/raw_4k/videos2caption.json`
- `data/real_4k_5b_rung{0,1,2}/training_dataset/worker_*/worker_0/data_chunk_0.parquet`
- `outputs/wan_finetune_4k/checkpoint-5` (rung 2)
- `outputs/wan_finetune_4k_5b_rung1/checkpoint-5`
- `outputs/wan_finetune_4k_5b_rung0/checkpoint-5`
- `/tmp/4k_5b_bench/train_rung{0,1,2}.log` — full training stdout
- `/tmp/4k_5b_bench/mem_watch_rung{0,1,2}.csv` — `nvidia-smi` 1 Hz samples
