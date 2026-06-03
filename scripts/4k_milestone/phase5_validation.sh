#!/usr/bin/env bash
# Phase 5 validation harness for the activation-streaming work.
#
# Runs the streaming pytest suite plus three 2-step probe runs on the
# 5B / 4×H200 / 121f / 704×1280 rig (FULL recompute baseline, naive
# FULL_OFFLOAD, and the new STREAMED_OFFLOAD), then invokes
# `phase5_verify.py` to grade the results against the §3 / §10
# acceptance criteria in `memory_prefetch_plan.md`.
#
# Expectations of the host:
#   - 4× H200 SXM (or any 4× CUDA box with enough VRAM)
#   - /workspace/FastVideo checkout (or set FV_ROOT)
#   - /workspace/venv/main venv with FastVideo deps (or set VENV)
#   - data/synthetic_5b_121f present (rung 6 of make_synthetic_5b_data.py)
#
# Output: /workspace/FastVideo/probe_out/phase5/{pytest.log,
#         full/, full_offload/, streamed_offload/, phase5_report.json,
#         phase5_report.md}
#
# Exit code reflects the verifier's PASS / FAIL judgement.

set -euo pipefail

FV_ROOT="${FV_ROOT:-/workspace/FastVideo}"
VENV="${VENV:-/workspace/venv/main}"
OUT_ROOT="${OUT_ROOT:-${FV_ROOT}/probe_out/phase5}"
DATA_DIR="${DATA_DIR:-data/synthetic_5b_121f}"
STEPS="${STEPS:-2}"
NUM_GPUS="${NUM_GPUS:-4}"
SP_SIZE="${SP_SIZE:-4}"
HSDP_SHARD_DIM="${HSDP_SHARD_DIM:-4}"
NUM_FRAMES="${NUM_FRAMES:-121}"
NUM_LATENT_T="${NUM_LATENT_T:-31}"
NUM_H="${NUM_H:-704}"
NUM_W="${NUM_W:-1280}"
MODEL_INIT="${MODEL_INIT:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CONFIG="${CONFIG:-examples/train/configs/fine_tuning/wan/t2v_4k.yaml}"
SKIP_PYTEST="${SKIP_PYTEST:-0}"
SKIP_PROBES="${SKIP_PROBES:-0}"

mkdir -p "${OUT_ROOT}"

echo "================================================================="
echo " Phase 5 validation"
echo " FV_ROOT       : ${FV_ROOT}"
echo " OUT_ROOT      : ${OUT_ROOT}"
echo " NUM_GPUS      : ${NUM_GPUS}"
echo " RESOLUTION    : ${NUM_H}×${NUM_W} × ${NUM_FRAMES}f"
echo " STEPS         : ${STEPS}"
echo "================================================================="

# ---------------------------------------------------------------------
# Step 1 — pytest suite for activation_streaming
# ---------------------------------------------------------------------
if [[ "${SKIP_PYTEST}" != "1" ]]; then
  echo
  echo "[1/3] Running streaming pytest suite ..."
  (
    cd "${FV_ROOT}"
    source "${VENV}/bin/activate"
    set +e
    pytest fastvideo/tests/training/streaming/ -vs \
      2>&1 | tee "${OUT_ROOT}/pytest.log"
    echo $? > "${OUT_ROOT}/pytest.exit"
  )
  PYTEST_EXIT="$(cat "${OUT_ROOT}/pytest.exit")"
  echo "  pytest exit code: ${PYTEST_EXIT}"
else
  echo
  echo "[1/3] SKIP_PYTEST=1 -- skipping pytest"
  echo "0" > "${OUT_ROOT}/pytest.exit"
fi

# ---------------------------------------------------------------------
# Steps 2 — probe each checkpoint mode
# ---------------------------------------------------------------------
run_probe() {
  local mode="$1"           # full | full_offload | streamed_offload
  local probe_dir="${OUT_ROOT}/${mode}"
  mkdir -p "${probe_dir}"

  echo
  echo "[2/3] Probing ${mode} ..."
  local extra_env=""
  if [[ "${mode}" == "streamed_offload" ]]; then
    extra_env="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  fi

  local script="/tmp/_phase5_${mode}.sh"
  cat > "${script}" <<EOF
#!/bin/bash
set -e
export HF_HOME="\${HF_HOME:-/workspace/.hf_home}"
export FASTVIDEO_MEM_PROBE_STEPS=${STEPS}
export FASTVIDEO_MEM_PROBE_DIR=${probe_dir}
export WANDB_MODE=disabled
export NUM_GPUS=${NUM_GPUS}
${extra_env}
cd ${FV_ROOT}
source ${VENV}/bin/activate
exec bash examples/train/run.sh \\
  ${CONFIG} \\
  --models.student.init_from ${MODEL_INIT} \\
  --pipeline.flow_shift 5.0 \\
  --training.distributed.num_gpus ${NUM_GPUS} \\
  --training.distributed.sp_size ${SP_SIZE} \\
  --training.distributed.hsdp_shard_dim ${HSDP_SHARD_DIM} \\
  --training.data.data_path ${DATA_DIR} \\
  --training.data.num_height ${NUM_H} --training.data.num_width ${NUM_W} \\
  --training.data.num_frames ${NUM_FRAMES} \\
  --training.data.num_latent_t ${NUM_LATENT_T} \\
  --training.model.enable_gradient_checkpointing_type ${mode} \\
  --training.loop.max_train_steps ${STEPS} \\
  --training.checkpoint.resume_from_checkpoint ""
EOF
  chmod +x "${script}"

  set +e
  "${script}" > "${probe_dir}/train.log" 2>&1
  local exit_code=$?
  set -e
  echo "${exit_code}" > "${probe_dir}/exit"
  echo "  ${mode} run exit: ${exit_code}"

  if [[ ! -f "${probe_dir}/phase_memory.json" ]]; then
    echo "  WARNING: ${probe_dir}/phase_memory.json was not produced." >&2
  fi
  if [[ ! -f "${probe_dir}/layer_trace.csv" ]]; then
    echo "  WARNING: ${probe_dir}/layer_trace.csv was not produced." >&2
  fi
}

if [[ "${SKIP_PROBES}" != "1" ]]; then
  run_probe "full"
  run_probe "full_offload"
  run_probe "streamed_offload"
else
  echo
  echo "[2/3] SKIP_PROBES=1 -- using existing probe outputs"
fi

# ---------------------------------------------------------------------
# Step 3 — verifier
# ---------------------------------------------------------------------
echo
echo "[3/3] Running verifier ..."
(
  cd "${FV_ROOT}"
  source "${VENV}/bin/activate"
  set +e
  python scripts/4k_milestone/phase5_verify.py \
    --probe-root "${OUT_ROOT}" \
    --report-md "${OUT_ROOT}/phase5_report.md" \
    --report-json "${OUT_ROOT}/phase5_report.json"
  VERIFIER_EXIT=$?
  set -e
  echo "${VERIFIER_EXIT}" > "${OUT_ROOT}/verifier.exit"
)

VERIFIER_EXIT="$(cat "${OUT_ROOT}/verifier.exit")"
echo
echo "================================================================="
echo " Phase 5 validation complete"
echo " Verifier exit  : ${VERIFIER_EXIT} (0 = PASS)"
echo " Report (md)    : ${OUT_ROOT}/phase5_report.md"
echo " Report (json)  : ${OUT_ROOT}/phase5_report.json"
echo "================================================================="

exit "${VERIFIER_EXIT}"
