#!/bin/bash
# One-step trace run for Wan 2.1 14B at 4K. NUM_FRAMES and NUM_GPUS are
# the only two knobs you usually need to change; everything else is
# derived (num_latent_t, sp_size, hsdp_shard_dim, data path, log paths).
# Sanity-check small first (NUM_FRAMES=17 NUM_GPUS=1 ./run.sh), then
# scale up (NUM_FRAMES=121 NUM_GPUS=8 ./run.sh).

NUM_FRAMES=${NUM_FRAMES:-17}
NUM_GPUS=${NUM_GPUS:-4}
MAX_STEPS=${MAX_STEPS:-1}

# Wan VAE temporal compression = 4; num_frames must satisfy (n-1) % 4 == 0.
NUM_LATENT_T=$(((NUM_FRAMES - 1) / 4 + 1))

# Convention: preprocessed parquet lives at
# data/real_4k_14b_<frames>f/training_dataset. Override DATA_PATH if your
# layout differs (e.g. DATA_PATH=data/real_4k_14b_rung0/training_dataset).
DATA_PATH=${DATA_PATH:-data/real_4k_14b_${NUM_FRAMES}f/training_dataset}

# Skip the ~84 GB final-checkpoint save for trace runs (the YAML
# defaults to ``training_state_checkpointing_steps: 50`` and save_final
# fires at end of training). Set SAVE_CHECKPOINT_STEPS=50 to restore.
SAVE_CHECKPOINT_STEPS=${SAVE_CHECKPOINT_STEPS:-0}

# ── PyTorch save_on_cpu (autograd activation offload) ───────────
# When 1, wraps forward+backward in ``torch.autograd.graph.save_on_cpu``:
# every saved-for-backward tensor is D2H-copied to pinned CPU during
# forward and H2D-copied back during backward. Trades step time for peak
# GPU memory. Use this to A/B compare against the no-offload baseline.
SAVE_ON_CPU_ENABLE=${SAVE_ON_CPU_ENABLE:-0}
SAVE_ON_CPU_PIN=${SAVE_ON_CPU_PIN:-1}

# Tag used in every output path; override TAG to customize naming.
# Auto-suffix when offload is on so logs don't overwrite the baseline.
if [ "${SAVE_ON_CPU_ENABLE}" = "1" ]; then
  TAG=${TAG:-${NUM_GPUS}gpu_${NUM_FRAMES}f_cpuoffload}
else
  TAG=${TAG:-${NUM_GPUS}gpu_${NUM_FRAMES}f}
fi

# ── Activation lifecycle trace ───────────────────────────────────
TRACE_ENABLE=${TRACE_ENABLE:-1}
TRACE_DIR=${TRACE_DIR:-logs/activation_trace_${TAG}}
TRACE_RANKS=${TRACE_RANKS:-all}
TRACE_MODULE_DETAIL=${TRACE_MODULE_DETAIL:-subblock}
TRACE_MEMORY=${TRACE_MEMORY:-1}
TRACE_FLUSH_EVERY=${TRACE_FLUSH_EVERY:-500}
TRACE_DIR_ENV="${TRACE_DIR}"
if [ "${TRACE_ENABLE}" = "0" ]; then
  TRACE_DIR_ENV=""
fi

# ── Memory probe (jd's commits) ──────────────────────────────────
# Rank-0 only by default --- FSDP makes ranks symmetric, and per-phase
# syncs would perturb step time.
MEM_PROBE_ENABLE=${MEM_PROBE_ENABLE:-1}
MEM_PROBE_DIR=${MEM_PROBE_DIR:-logs/mem_probe_${TAG}}
# Default to MAX_STEPS so the probe captures every training step and
# its phase snapshots get written even on multi-step runs.
MEM_PROBE_STEPS=${MEM_PROBE_STEPS:-${MAX_STEPS}}
MEM_PROBE_TOPK=${MEM_PROBE_TOPK:-0}
MEM_PROBE_ALL_RANKS=${MEM_PROBE_ALL_RANKS:-0}
if [ "${MEM_PROBE_ENABLE}" = "0" ]; then
  MEM_PROBE_STEPS=0
fi

# ── Outer combined stdout log ────────────────────────────────────
OUTER_LOG=${OUTER_LOG:-logs/train_wan21_14b_${TAG}.log}
mkdir -p "$(dirname "${OUTER_LOG}")"

FASTVIDEO_ACTIVATION_TRACE="${TRACE_ENABLE}" \
  FASTVIDEO_ACTIVATION_TRACE_DIR="${TRACE_DIR_ENV}" \
  FASTVIDEO_ACTIVATION_TRACE_MAX_STEPS=1 \
  FASTVIDEO_ACTIVATION_TRACE_RANKS="${TRACE_RANKS}" \
  FASTVIDEO_ACTIVATION_TRACE_MODULE_DETAIL="${TRACE_MODULE_DETAIL}" \
  FASTVIDEO_ACTIVATION_TRACE_MEMORY="${TRACE_MEMORY}" \
  FASTVIDEO_ACTIVATION_TRACE_FLUSH_EVERY="${TRACE_FLUSH_EVERY}" \
  FASTVIDEO_MEM_PROBE_STEPS="${MEM_PROBE_STEPS}" \
  FASTVIDEO_MEM_PROBE_DIR="${MEM_PROBE_DIR}" \
  FASTVIDEO_MEM_PROBE_TOPK="${MEM_PROBE_TOPK}" \
  FASTVIDEO_MEM_PROBE_ALL_RANKS="${MEM_PROBE_ALL_RANKS}" \
  FASTVIDEO_SAVE_ON_CPU="${SAVE_ON_CPU_ENABLE}" \
  FASTVIDEO_SAVE_ON_CPU_PIN="${SAVE_ON_CPU_PIN}" \
  NUM_GPUS="${NUM_GPUS}" WANDB_MODE=disabled bash examples/train/run.sh \
  examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
  --models.student.init_from Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --pipeline.flow_shift 5.0 \
  --training.distributed.num_gpus "${NUM_GPUS}" \
  --training.distributed.sp_size "${NUM_GPUS}" \
  --training.distributed.hsdp_shard_dim "${NUM_GPUS}" \
  --training.data.data_path "${DATA_PATH}" \
  --training.data.num_height 2160 \
  --training.data.num_width 3840 \
  --training.data.num_frames "${NUM_FRAMES}" \
  --training.data.num_latent_t "${NUM_LATENT_T}" \
  --training.loop.max_train_steps "${MAX_STEPS}" \
  --training.checkpoint.training_state_checkpointing_steps "${SAVE_CHECKPOINT_STEPS}" \
  --training.checkpoint.output_dir "outputs/wan_finetune_4k_14b_trace_${TAG}" \
  --training.tracker.run_name "wan_4k_14b_trace_${TAG}" \
  2>&1 | tee "${OUTER_LOG}"
