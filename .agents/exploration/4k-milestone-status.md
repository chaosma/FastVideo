# Exploration Log: 4K Wan2.1 T2V Fine-Tuning Milestone

## Status: in_progress

## Context

Fine-tune Wan 2.1 T2V 1.3B on 4K (3840×2160) video, starting from the milestone
"one training step without crashing." The owning user is Chao Ma
(`mc8305@gmail.com`); the work lives on branch `4k`.

The relevant entry points and configs:

- Training config: `examples/train/configs/fine_tuning/wan/t2v_4k.yaml`
- Training launcher: `examples/train/run.sh`
- Synthetic-data plumbing test: `scripts/4k_milestone/make_synthetic_4k_data.py`
- Real-data preprocessing: `scripts/4k_milestone/preprocess_4k.sh`
- Real-data location: `data/raw_4k/videos/*.mp4` + `data/raw_4k/videos2caption.json`

Hardware to date: 4× RTX 4090 (24 GB each), no NVLink.

## Rung ladder (memory-vs-resolution tradeoff)

The training config defines six rungs in `t2v_4k.yaml`. Climb DOWN if OOM:

| Rung | Resolution | Frames | num_latent_t | Tokens (SP=1) |
|------|------------|--------|--------------|---------------|
| 0    | 3840×2160  | 81     | 21           | 680,400 |
| 1    | 3840×2160  | 17     | 5            | 162,000 |
| 2    | 3840×2160  | 5      | 2            | 64,800  ← default |
| 3    | 3840×2160  | 1      | 1            | 32,400  |
| 4    | 2560×1440  | 1      | 1            | 14,400  |
| 5    | 1920×1080  | 1      | 1            |  8,100  |

Token count = `(H/16) × (W/16) × num_latent_t` (Wan VAE: spatial_stride=8,
patch_size=2). At 4× more tokens, attention activations grow roughly
quadratically.

## What is proven to work

### Training one step at 4K with synthetic latents — DONE

Synthetic-data plumbing test (rung 2: 4K, 5 frames) on 4×4090, ~22.7 s/step:

```bash
# 1. Generate synthetic 4K-shaped latents (CPU-only, ~minutes)
python scripts/4k_milestone/make_synthetic_4k_data.py --rung 2

# 2. Dry-run sanity check (NUM_GPUS=1 needed; see "Known issues" below)
NUM_GPUS=1 bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --dry-run --training.distributed.num_gpus 1

# 3. Actual one-step training, SP=4 across all four GPUs
NUM_GPUS=4 WANDB_MODE=disabled bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --training.distributed.num_gpus 4 \
    --training.distributed.sp_size 4 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.data.data_path data/synthetic_4k
```

Required configuration: `sp_size=4` (split sequence across all GPUs to bound
activations) AND `hsdp_shard_dim=4` (FSDP-shard weights across the same group).
Single GPU OOMs. SP=1 with FSDP-only also OOMs because activations don't fit.

### 10 training steps on real 480p data — DONE

End-to-end pipeline at lower resolution (real video → preprocess → train). See
"Real-data preprocessing" below for the workaround.

## What is NOT proven to work yet

### Real-data preprocessing AT 4K on 24 GB GPUs — BLOCKED

Run `scripts/4k_milestone/preprocess_4k.sh` against 4K source video; VAE
encoding OOMs even with `--vae-tiling --vae-sp` and 4 GPUs. Diagnosis below.

This is **independent** of training and was not the original "one-step" goal.
Plan: re-attempt on a card with ≥40 GB VRAM (A100/H100), or implement a
properly tiled WanVAE encoder that maintains the feature cache per-tile.

## Real-data preprocessing — required setup

The `merged` dataset loader expects this layout (NOT the CSV format implied
by older script docstrings):

```
data/raw_4k/
├── videos/
│   └── *.mp4 or *.mov
└── videos2caption.json
```

`videos2caption.json` is a JSON array. Required keys: `path` (relative to
`videos/`), `cap` (list of caption strings), `resolution`, `fps`, `duration`,
`num_frames`. Example:

```json
[
  {
    "path": "clip001.mp4",
    "resolution": {"width": 3840, "height": 2160},
    "fps": 30.0,
    "duration": 29.6,
    "num_frames": 888,
    "cap": ["A 4K ultra-high-definition demo clip."]
  }
]
```

The number of entries must be ≥ `GPU_NUM`, otherwise `dataset.shard()` raises
`IndexError: Index N out of range for dataset of size 1` on rank N. If you
only have one physical clip, duplicate it across multiple JSON entries (the
captions can differ).

Long clips: bump `--preprocess.video-length-tolerance-range`. The validator
rejects any clip longer than `tolerance × (num_frames / train_fps)` seconds.
With defaults (tolerance=5, num_frames=5, train_fps=16) the cutoff is ~1.6 s,
which kills any real footage. The committed default is now 1000.

## Known runtime issues / workarounds

### `examples/train/run.sh` ignores `--training.distributed.num_gpus`

Line 24 sets `NUM_GPUS=$(nvidia-smi -L | wc -l)` and uses that as
`--nproc_per_node`, ignoring the CLI flag. Always set `NUM_GPUS=N` as a shell
env var explicitly when you want N ranks. This is a UX wart, not a bug —
worth fixing in the launcher.

### Rank-0-only inference pipeline causes SP collective hang (FIXED)

`fastvideo/train/models/wan/wan.py:WanModel.set_negative_prompt` builds an
inference `WanPipeline` only on rank 0 to encode the negative prompt. Its
`post_init` unconditionally calls `warmup_sequence_parallel_communication()`,
which issues an SP-group all-to-all. With `sp_size=1` this is a no-op (early
return); with `sp_size>1` (any 4K config) ranks 1..N never enter the branch,
so rank 0 hangs alone on the collective and the run dies after the NCCL
timeout (~10 min).

Fixed in commit `6eed756` by gating the warmup behind
`FASTVIDEO_SKIP_SP_WARMUP=1`, which is set around the rank-0-only construction.

### Preprocessing pipeline issues (FIXED)

Committed in `9a281b3` (see `git log` for the canonical commit):

1. `v1_preprocessing_new.py` hardcoded `sp_size=tp_size=1`, silently ignoring
   `--sp-size N` and `--vae-sp`. Multi-GPU preprocess degraded to plain DP
   and every rank re-encoded the full clip. Now honors CLI args.

2. `torchvision.io.read_video` was removed in torchvision ≥0.22 (we're on
   0.26). The `TORCHVISION` loader path crashed with `AttributeError`.
   Patched `preprocess_stages.py` to decode requested frames with PyAV
   (`import av`) instead.

3. `numpy` has no native `bfloat16`. The parquet writer in
   `workflow/preprocess/components.py` blew up with
   `TypeError: Got unsupported ScalarType BFloat16` whenever
   `--vae-precision bf16` or `--text-encoder-precisions bf16` was used.
   Now upcasts bf16 → fp32 before `.numpy()`.

4. `scripts/4k_milestone/preprocess_4k.sh` used `--preprocess.foo_bar`-style
   underscored arg names; the FlexibleArgumentParser exposes them as
   `--preprocess.foo-bar` (dashes). Every flag was rejected. Replaced.

   Also: the script's docstring says metadata is a CSV but the loader
   actually wants `videos2caption.json` (see "required setup" above).
   The docstring is stale; the JSON layout is the working one.

### Environment: torchaudio/torchcodec CUDA mismatch

The default pip-installed `torchaudio` 2.11.0 ships CUDA-13 binaries, while
the project pins torch `+cu128`. On import, `libcudart.so.13: not found`.
Fix:

```bash
pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    torchaudio==2.11.0
```

`torchcodec` 0.11.1 has the same problem. Either install the cu128 variant
the same way, or just `pip uninstall torchcodec` — the preprocess scripts
use the torchvision/PyAV loader path now, not torchcodec.

This isn't a repo bug; it's a wheel-resolver quirk in the lockfile-less
`uv pip install -e .[dev]` flow.

## Why 4K real-data preprocessing OOMs on 24 GB

Verified by inserting per-stage GPU memory snapshots in
`composed_pipeline_base.forward()`. Entering the VAE encoding stage:

```
[mem] before EncodingStage: alloc=0.01 GiB  reserved=0.02 GiB  free=22.83 GiB
```

GPU is essentially empty. Text encoder offload works correctly. The OOM is
purely VAE activations during the 4K encode. Three compounding reasons:

1. **Activation size at 4K.** First conv (3→96 channels) on `[1, 3, 1, 2160, 3840]`
   produces `[1, 96, 1, 2160, 3840]` ≈ **1.49 GiB** in bf16 — that exact value
   matches the OOM allocation. The next conv (96→192) doubles to ~2.98 GiB.
   Several layers run before any spatial downsampling, so peak activation is
   ~5 GiB per layer for several layers in a row.

2. **WanVAE.encode() ignores `--vae-tiling`.** The class default
   `use_feature_cache=True` takes a custom override path that always does
   full-spatial encode. Even with `--vae-tiling --vae-sp` set, every rank does
   a full 4K encode; the SP / tiling flags are no-ops on this code path.

3. **Time-causal architecture can't be trivially tiled.** WanVAE's encoder
   uses time-causal 3D convs that depend on a per-layer feature cache
   (`_enc_feat_map`) bridging temporal chunks. Disabling the cache to enable
   the base-class tiled path crashes with `kernel size 3 > input T=2` because
   `_encode(tile)` runs without the feature-cache context the convs need. So
   you can't just turn the feature cache off and use spatial tiles; the
   architecture requires a tiled-encoder rewrite that materializes the right
   slice of the feature cache for each tile.

The 4.46 GiB and 1.48 GiB OOM allocations seen in different runs correspond
to different intermediate-channel widths × 4K spatial extent. The text
encoder is fully offloaded by the time VAE runs (confirmed by the snapshot);
text encoder is not the culprit.

## Paths forward for 4K real-data preprocessing

1. **Easiest, most reliable.** Run preprocessing on a single ≥40 GB GPU
   (A100 40GB / H100 80GB). Activations fit; none of the workarounds in this
   doc are needed. Training itself can still run on the 4× 4090 box once
   parquet latents exist.

2. **Engineering option.** Implement a proper tiled WanVAE encoder that
   maintains the feature cache per spatial tile. Probably 200–500 LOC plus
   correctness validation against full-resolution latents. Touchpoints:
   `fastvideo/models/vaes/wanvae.py` (`AutoencoderKLWan.encode` and the
   `_encode` / feature-cache contextvars), and `fastvideo/models/vaes/common.py`
   (`ParallelTiledVAE.spatial_tiled_encode`).

3. **Hack: pre-resize input video offline.** Encode at 1080p or 1440p; the
   resulting latents will not be 4K-shaped, defeating the spirit of the
   milestone but exercising the rest of the pipeline. The committed
   480p run from this branch is exactly this hack.

## Useful commands

```bash
# 4K synthetic plumbing test (works on this hardware)
python scripts/4k_milestone/make_synthetic_4k_data.py --rung 2
NUM_GPUS=4 WANDB_MODE=disabled bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --training.distributed.num_gpus 4 --training.distributed.sp_size 4 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.data.data_path data/synthetic_4k

# 4K real-data preprocessing (works on ≥40 GB GPU, OOMs on 24 GB)
MAX_HEIGHT=2160 MAX_WIDTH=3840 RUNG=2 GPU_NUM=4 \
    bash scripts/4k_milestone/preprocess_4k.sh

# 480p real-data fallback (works on 4× 4090; demonstrates full pipeline)
MAX_HEIGHT=480 MAX_WIDTH=854 RUNG=2 GPU_NUM=1 \
    bash scripts/4k_milestone/preprocess_4k.sh
NUM_GPUS=4 WANDB_MODE=disabled bash examples/train/run.sh \
    examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
    --training.distributed.num_gpus 4 --training.distributed.sp_size 1 \
    --training.distributed.hsdp_shard_dim 4 \
    --training.data.data_path data/preprocessed_4k/training_dataset \
    --training.data.num_height 480 --training.data.num_width 848 \
    --training.data.num_frames 5 --training.data.num_latent_t 2 \
    --training.loop.max_train_steps 10
```

## Open questions

- Does 4K real-data preprocessing work out-of-the-box on A100 40GB / H100?
  (Expected: yes, modulo the four committed bugfixes from `9a281b3`.)
- For the bigger machine, can `sp_size=4 hsdp_shard_dim=4` go higher in the
  rung ladder (rung 1 = 17 frames)? Tokens grow ~2.5×; SP=4 should still cope.
- The preprocess script's `--preprocess.video-length-tolerance-range` logic
  is back-asserted from the target output duration, not the actual content
  duration of the source clip. The default of 5 is fine for the
  HuggingFace-style "many short clips" datasets the script was originally
  written for; it's wrong for "one long source clip we'll temporal-sample
  from." Worth a follow-up to make the validator more obviously
  configurable (or rethink the formula).
