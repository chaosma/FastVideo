#!/usr/bin/env bash
# Phase 2: Preprocess raw 4K video clips → parquet latents for FastVideo training.
#
# Usage:
#   bash scripts/4k_milestone/preprocess_4k.sh [--rung N]
#
# Rungs map to different frame counts (must satisfy: (num_frames-1) % 4 == 0):
#   Rung 0: 77 frames (num_latent_t=20) — full target (Wan2.1 native length)
#   Rung 1: 17 frames (num_latent_t=5)
#   Rung 2:  5 frames (num_latent_t=2)  ← default
#   Rung 3:  1 frame  (num_latent_t=1)  image mode
#
# Prerequisites:
#   - Raw 4K clips in data/raw_4k/ (any .mp4 or .mov files, ≥ RUNG_FRAMES long)
#   - A metadata CSV at data/raw_4k/metadata.csv with columns: file_path,caption
#     (relative to data/raw_4k/). See format note below.
#   - FastVideo installed: cd FastVideo && uv pip install -e .[dev]
#
# Metadata CSV format (one row per clip):
#   file_path,caption
#   clip001.mp4,"A waterfall cascades over mossy rocks in 4K resolution."
#   clip002.mp4,"Aerial shot of city skyline at sunset in ultra high definition."
#
# Output: data/preprocessed_4k/  (parquet files containing vae_latent + text_embedding)
#
# Notes:
#   - At 4K the WanVAE encoder is memory-intensive. preprocess_video_batch_size=1
#     keeps memory bounded. If it still OOMs, the VAE tiling is enabled automatically
#     by FastVideo when resolutions exceed the tile thresholds.
#   - GPU_NUM=1 is safe for preprocessing; increase if you have multiple GPUs
#     and want to parallelize across clips.

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
RUNG="${RUNG:-2}"
GPU_NUM="${GPU_NUM:-1}"

MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
DATASET_PATH="${DATASET_PATH:-data/raw_4k}"
OUTPUT_DIR="${OUTPUT_DIR:-data/preprocessed_4k}"
MAX_HEIGHT="${MAX_HEIGHT:-2160}"
MAX_WIDTH="${MAX_WIDTH:-3840}"

# ── Rung → frame count ─────────────────────────────────────────────────────────
case "$RUNG" in
  0) NUM_FRAMES=77 ;;
  1) NUM_FRAMES=17 ;;
  2) NUM_FRAMES=5  ;;
  3) NUM_FRAMES=1  ;;
  *) echo "Unknown rung: $RUNG (choose 0-3)"; exit 1 ;;
esac

# ── Parse --rung flag manually ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rung) RUNG="$2"; shift 2 ;;
    --gpu-num) GPU_NUM="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Re-apply rung after flag parsing
case "$RUNG" in
  0) NUM_FRAMES=77 ;;
  1) NUM_FRAMES=17 ;;
  2) NUM_FRAMES=5  ;;
  3) NUM_FRAMES=1  ;;
  *) echo "Unknown rung: $RUNG (choose 0-3)"; exit 1 ;;
esac

echo "=== 4K Preprocessing ==="
echo "Rung:        ${RUNG} (${NUM_FRAMES} frames)"
echo "Resolution:  3840×2160"
echo "Model:       ${MODEL_PATH}"
echo "Input:       ${DATASET_PATH}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPUs:        ${GPU_NUM}"
echo "========================="

# ── Verify input exists ────────────────────────────────────────────────────────
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "ERROR: Input directory '${DATASET_PATH}' does not exist."
  echo "  Place your 4K video clips there and create metadata.csv."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ── Run preprocessing ──────────────────────────────────────────────────────────
# Flags explained:
#   --preprocess.max_height / max_width: target spatial resolution for cropping/resizing
#   --preprocess.num_frames: number of video frames to sample per clip
#   --preprocess.train_fps 16: sample at 16 fps (consistent with Wan2.1 training)
#   --preprocess.preprocess_video_batch_size 1: one clip at a time (safe for 4K)
#   --preprocess.samples_per_file 4: few latents per parquet shard (4K latents are large)
#   --preprocess.flush_frequency 4: write to disk often (avoid large in-memory buffers)
#   --preprocess.video_length_tolerance_range 5: accept clips ≥ (num_frames - 5) frames
#   --preprocess.dataset_type merged: expects metadata.csv format in DATASET_PATH

torchrun \
  --nnodes 1 \
  --nproc_per_node "${GPU_NUM}" \
  --master_port 29514 \
  -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
  --model_path "${MODEL_PATH}" \
  --mode preprocess \
  --workload_type t2v \
  --sp-size "${GPU_NUM}" \
  --num-gpus "${GPU_NUM}" \
  --text-encoder-cpu-offload \
  --vae-cpu-offload \
  --disable-autocast \
  --text-encoder-precisions bf16 \
  --vae-precision bf16 \
  --vae-tiling \
  --vae-sp \
  --vae-config.use-parallel-tiling \
  --preprocess.video-loader-type torchvision \
  --preprocess.dataset-type merged \
  --preprocess.dataset-path "${DATASET_PATH}" \
  --preprocess.dataset-output-dir "${OUTPUT_DIR}" \
  --preprocess.max-height "${MAX_HEIGHT}" \
  --preprocess.max-width "${MAX_WIDTH}" \
  --preprocess.num-frames "${NUM_FRAMES}" \
  --preprocess.train-fps 16 \
  --preprocess.preprocess-video-batch-size 1 \
  --preprocess.dataloader-num-workers 0 \
  --preprocess.samples-per-file 4 \
  --preprocess.flush-frequency 4 \
  --preprocess.video-length-tolerance-range 1000

echo ""
echo "=== Preprocessing complete ==="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Verify the output with:"
echo "  python scripts/4k_milestone/verify_parquet.py ${OUTPUT_DIR}"
