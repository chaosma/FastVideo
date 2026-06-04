#!/usr/bin/env bash
# Phase 6 — 14B / 8×H200 validation study (memory_prefetch_plan.md §3).
#
# Stage 1: reuse the Phase 5 harness at the 14B rung-0 measurement point
#          (2160×3840 × 77f, num_latent_t=20, sp=8, hsdp=8) for the three
#          2-step probed runs: FULL, FULL_OFFLOAD, STREAMED_OFFLOAD.
# Stage 2: 10-step STREAMED_OFFLOAD soak — step-time stability + finite loss.
#
# Checkpoint saves are disabled (a single 14B training-state checkpoint is
# ~86 GB; disk has ~84 GB free).
set -uo pipefail

FV_ROOT=/workspace/FastVideo
OUT=${FV_ROOT}/probe_out/phase6
mkdir -p "${OUT}"

echo "=== Stage 1: 3-mode probe harness ($(date)) ==="
NUM_GPUS=8 SP_SIZE=8 HSDP_SHARD_DIM=8 \
NUM_H=2160 NUM_W=3840 NUM_FRAMES=77 NUM_LATENT_T=20 \
MODEL_INIT=Wan-AI/Wan2.1-T2V-14B-Diffusers \
DATA_DIR=data/synthetic_4k_14b_rung0 \
STEPS=2 SKIP_PYTEST=1 LOSS_TOL=5e-3 \
EXTRA_ARGS="--training.checkpoint.training_state_checkpointing_steps 0" \
OUT_ROOT="${OUT}" \
bash "${FV_ROOT}/scripts/4k_milestone/phase5_validation.sh"
HARNESS_EXIT=$?
echo "harness exit: ${HARNESS_EXIT} ($(date))"

# Abort the soak only if the streamed probe itself failed to produce
# artifacts (the verifier verdict may legitimately fail on thresholds we
# want to inspect manually).
if [[ ! -f "${OUT}/streamed_offload/phase_memory.json" ]]; then
  echo "FATAL: streamed_offload probe produced no phase_memory.json; skipping soak."
  exit 1
fi

echo "=== Stage 2: 10-step STREAMED_OFFLOAD soak ($(date)) ==="
SOAK_DIR="${OUT}/streamed_soak10"
mkdir -p "${SOAK_DIR}"
cat > /tmp/_phase6_soak.sh <<EOF
#!/bin/bash
set -e
export HF_HOME=\${HF_HOME:-/workspace/.hf_home}
export FASTVIDEO_MEM_PROBE_STEPS=10
export FASTVIDEO_MEM_PROBE_DIR=${SOAK_DIR}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled
export NUM_GPUS=8
cd ${FV_ROOT}
source /workspace/venv/main/bin/activate
exec bash examples/train/run.sh \\
  examples/train/configs/fine_tuning/wan/t2v_4k.yaml \\
  --models.student.init_from Wan-AI/Wan2.1-T2V-14B-Diffusers \\
  --pipeline.flow_shift 5.0 \\
  --training.distributed.num_gpus 8 \\
  --training.distributed.sp_size 8 \\
  --training.distributed.hsdp_shard_dim 8 \\
  --training.data.data_path data/synthetic_4k_14b_rung0 \\
  --training.data.num_height 2160 --training.data.num_width 3840 \\
  --training.data.num_frames 77 --training.data.num_latent_t 20 \\
  --training.model.enable_gradient_checkpointing_type streamed_offload \\
  --training.loop.max_train_steps 10 \\
  --training.checkpoint.resume_from_checkpoint "" \\
  --training.checkpoint.training_state_checkpointing_steps 0
EOF
chmod +x /tmp/_phase6_soak.sh
/tmp/_phase6_soak.sh > "${SOAK_DIR}/train.log" 2>&1
SOAK_EXIT=$?
echo "soak exit: ${SOAK_EXIT} ($(date))"

echo "=== Phase 6 study complete ($(date)) ==="
echo "harness=${HARNESS_EXIT} soak=${SOAK_EXIT}"
exit $(( HARNESS_EXIT != 0 || SOAK_EXIT != 0 ? 1 : 0 ))
