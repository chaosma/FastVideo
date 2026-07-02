#!/bin/bash
# Train Wan at 4K. Set MODEL=14b (default) or MODEL=5b.
# Adjust NUM_FRAMES and NUM_GPUS; everything else is derived.
# Sanity-check: NUM_FRAMES=5 NUM_GPUS=1 MODEL=5b ./run.sh

MODEL=${MODEL:-14b}

if [ "${MODEL}" = "5b" ]; then
  MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
  NUM_GPUS=${NUM_GPUS:-1}
  NUM_FRAMES=${NUM_FRAMES:-17}
  DATA_PATH=${DATA_PATH:-data/real_4k_5b_${NUM_FRAMES}f/training_dataset}
else
  MODEL_ID="Wan-AI/Wan2.1-T2V-14B-Diffusers"
  NUM_GPUS=${NUM_GPUS:-8}
  NUM_FRAMES=${NUM_FRAMES:-121}
  DATA_PATH=${DATA_PATH:-data/real_4k_14b_${NUM_FRAMES}f/training_dataset}
fi

MAX_STEPS=${MAX_STEPS:-1}

# Wan VAE temporal compression = 4; num_frames must satisfy (n-1) % 4 == 0.
NUM_LATENT_T=$(((NUM_FRAMES - 1) / 4 + 1))

# Skip the ~84 GB checkpoint save by default; set to 50 to restore.
SAVE_CHECKPOINT_STEPS=${SAVE_CHECKPOINT_STEPS:-0}

# SAVE_ON_CPU_ENABLE=1: offload activations to CPU during forward, restore
# during backward. Reduces peak VRAM at the cost of step time.
SAVE_ON_CPU_ENABLE=${SAVE_ON_CPU_ENABLE:-0}

# Auto-tag so offload and baseline logs don't collide.
if [ "${SAVE_ON_CPU_ENABLE}" = "1" ]; then
  TAG=${TAG:-${NUM_GPUS}gpu_${NUM_FRAMES}f_cpuoffload}
else
  TAG=${TAG:-${NUM_GPUS}gpu_${NUM_FRAMES}f}
fi

OUTER_LOG=${OUTER_LOG:-logs/train_wan_${MODEL}_${TAG}.log}
mkdir -p "$(dirname "${OUTER_LOG}")"

FASTVIDEO_SAVE_ON_CPU="${SAVE_ON_CPU_ENABLE}" \
  NUM_GPUS="${NUM_GPUS}" WANDB_MODE=disabled bash examples/train/run.sh \
  examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
  --models.student.init_from "${MODEL_ID}" \
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
  --training.checkpoint.output_dir "outputs/wan_finetune_4k_${MODEL}_${TAG}" \
  --training.tracker.run_name "wan_4k_${MODEL}_${TAG}" \
  "$@" \
  2>&1 | tee "${OUTER_LOG}"
