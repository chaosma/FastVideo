#!/usr/bin/env bash
# Master convenience script for the 4K milestone.
#
# Runs through the milestone phases in order. Adjust RUNG, DATA_PATH, etc.
# below, then run on the remote Ubuntu server (requires CUDA + VRAM ≥ 24 GB).
#
# Usage:
#   bash scripts/4k_milestone/run_4k_milestone.sh [phase]
#
#   phase:
#     all          — run phases 1-4 in order (default)
#     synthetic    — Phase 0: create synthetic data only (CPU-safe, for plumbing check)
#     dry-run      — Phase 3 dry-run only (CPU-safe after synthetic data exists)
#     preprocess   — Phase 2: preprocess raw 4K clips (requires GPU)
#     train-1step  — Phase 4: one training step (requires GPU)
#     overfit      — Phase 5: 100-step overfit test (requires GPU)
#
# Environment variables:
#   RUNG         — starting rung (default: 2 = 4K 5 frames)
#   GPU_NUM      — number of GPUs for preprocessing and training (default: 1)
#   DATA_PATH    — override data path for preprocessed latents
#   MODEL_PATH   — Wan2.1 model weights (default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTVIDEO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${FASTVIDEO_ROOT}"

PHASE="${1:-all}"
RUNG="${RUNG:-2}"
GPU_NUM="${GPU_NUM:-1}"
MODEL_PATH="${MODEL_PATH:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
DATA_PATH="${DATA_PATH:-data/preprocessed_4k}"
SYNTHETIC_DATA_PATH="data/synthetic_4k"
CONFIG="examples/train/configs/fine_tuning/wan/t2v_4k.yaml"

# Map rung to (num_latent_t, num_height, num_width, num_frames)
declare -A RUNG_LATENT_T=([0]=21 [1]=5  [2]=2 [3]=1 [4]=1 [5]=1)
declare -A RUNG_HEIGHT=  ([0]=2160 [1]=2160 [2]=2160 [3]=2160 [4]=1440 [5]=1080)
declare -A RUNG_WIDTH=   ([0]=3840 [1]=3840 [2]=3840 [3]=3840 [4]=2560 [5]=1920)
declare -A RUNG_FRAMES=  ([0]=81   [1]=17   [2]=5    [3]=1    [4]=1    [5]=1)

NUM_LATENT_T="${RUNG_LATENT_T[$RUNG]}"
HEIGHT="${RUNG_HEIGHT[$RUNG]}"
WIDTH="${RUNG_WIDTH[$RUNG]}"
NUM_FRAMES="${RUNG_FRAMES[$RUNG]}"

echo "=== 4K Milestone Run ==="
echo "Phase:       ${PHASE}"
echo "Rung:        ${RUNG} (${WIDTH}×${HEIGHT}, ${NUM_FRAMES} frames, latent_t=${NUM_LATENT_T})"
echo "GPU_NUM:     ${GPU_NUM}"
echo "========================="

# ── Phase 0: Synthetic data (CPU-safe) ────────────────────────────────────────
run_synthetic() {
  echo ""
  echo "--- Phase 0: Creating synthetic ${WIDTH}×${HEIGHT} latent data (rung ${RUNG}) ---"
  python scripts/4k_milestone/make_synthetic_4k_data.py \
    --rung "${RUNG}" \
    --n-samples 8 \
    --output-dir "${SYNTHETIC_DATA_PATH}"
  python scripts/4k_milestone/verify_parquet.py "${SYNTHETIC_DATA_PATH}"
}

# ── Phase 3: Config dry-run (CPU-safe if CUDA init can be skipped) ────────────
run_dry_run() {
  echo ""
  echo "--- Phase 3: Config dry-run ---"
  bash examples/train/run.sh "${CONFIG}" \
    --dry-run \
    --training.distributed.num_gpus 1 \
    --training.data.data_path "${SYNTHETIC_DATA_PATH}" \
    --training.data.num_latent_t "${NUM_LATENT_T}" \
    --training.data.num_height "${HEIGHT}" \
    --training.data.num_width "${WIDTH}" \
    --training.data.num_frames "${NUM_FRAMES}"
  echo "Dry-run PASSED: config parsed and build_from_config succeeded."
}

# ── Phase 2: Real preprocessing (requires GPU) ────────────────────────────────
run_preprocess() {
  echo ""
  echo "--- Phase 2: Preprocessing raw 4K clips (rung ${RUNG}) ---"
  RUNG="${RUNG}" GPU_NUM="${GPU_NUM}" \
    bash scripts/4k_milestone/preprocess_4k.sh
  python scripts/4k_milestone/verify_parquet.py "${DATA_PATH}"
}

# ── Phase 4: One training step (requires GPU) ─────────────────────────────────
run_train_1step() {
  echo ""
  echo "--- Phase 4: First 4K training step (rung ${RUNG}) ---"
  NUM_GPUS="${GPU_NUM}" bash examples/train/run.sh "${CONFIG}" \
    --training.distributed.num_gpus "${GPU_NUM}" \
    --training.data.data_path "${DATA_PATH}" \
    --training.data.num_latent_t "${NUM_LATENT_T}" \
    --training.data.num_height "${HEIGHT}" \
    --training.data.num_width "${WIDTH}" \
    --training.data.num_frames "${NUM_FRAMES}" \
    --training.loop.max_train_steps 1 \
    --training.tracker.run_name "wan_4k_rung${RUNG}_1step"
}

# ── Phase 5: Overfit test (requires GPU) ──────────────────────────────────────
run_overfit() {
  echo ""
  echo "--- Phase 5: 100-step overfit test (rung ${RUNG}) ---"
  NUM_GPUS="${GPU_NUM}" bash examples/train/run.sh "${CONFIG}" \
    --training.distributed.num_gpus "${GPU_NUM}" \
    --training.data.data_path "${DATA_PATH}" \
    --training.data.num_latent_t "${NUM_LATENT_T}" \
    --training.data.num_height "${HEIGHT}" \
    --training.data.num_width "${WIDTH}" \
    --training.data.num_frames "${NUM_FRAMES}" \
    --training.loop.max_train_steps 100 \
    --training.checkpoint.training_state_checkpointing_steps 50 \
    --training.tracker.run_name "wan_4k_rung${RUNG}_overfit"
  echo ""
  echo "Check loss curve — should decrease monotonically over 100 steps."
  echo "Checkpoint at: outputs/wan_finetune_4k/"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${PHASE}" in
  synthetic)
    run_synthetic
    ;;
  dry-run)
    run_synthetic
    run_dry_run
    ;;
  preprocess)
    run_preprocess
    ;;
  train-1step)
    run_train_1step
    ;;
  overfit)
    run_overfit
    ;;
  all)
    run_synthetic
    run_dry_run
    run_preprocess
    run_train_1step
    echo ""
    echo "=== Phase 4 COMPLETE ==="
    echo "If loss is finite → milestone achieved at rung ${RUNG}!"
    echo "Next: run 'overfit' phase to confirm gradients flow correctly."
    ;;
  *)
    echo "Unknown phase: ${PHASE}"
    echo "Choose: all | synthetic | dry-run | preprocess | train-1step | overfit"
    exit 1
    ;;
esac
