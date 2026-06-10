#!/bin/bash

TRACE_DIR=${TRACE_DIR:-logs/activation_trace_8xh200_121f}
TRACE_ENABLE=${TRACE_ENABLE:-1}
TRACE_RANKS=${TRACE_RANKS:-all}
TRACE_MODULE_DETAIL=${TRACE_MODULE_DETAIL:-subblock}
TRACE_MEMORY=${TRACE_MEMORY:-1}
TRACE_FLUSH_EVERY=${TRACE_FLUSH_EVERY:-500}
TRACE_DIR_ENV="${TRACE_DIR}"
if [ "${TRACE_ENABLE}" = "0" ]; then
  TRACE_DIR_ENV=""
fi

FASTVIDEO_ACTIVATION_TRACE="${TRACE_ENABLE}" \
FASTVIDEO_ACTIVATION_TRACE_DIR="${TRACE_DIR_ENV}" \
FASTVIDEO_ACTIVATION_TRACE_MAX_STEPS=1 \
FASTVIDEO_ACTIVATION_TRACE_RANKS="${TRACE_RANKS}" \
FASTVIDEO_ACTIVATION_TRACE_MODULE_DETAIL="${TRACE_MODULE_DETAIL}" \
FASTVIDEO_ACTIVATION_TRACE_MEMORY="${TRACE_MEMORY}" \
FASTVIDEO_ACTIVATION_TRACE_FLUSH_EVERY="${TRACE_FLUSH_EVERY}" \
NUM_GPUS=8 WANDB_MODE=disabled bash examples/train/run.sh \
  examples/train/configs/fine_tuning/wan/t2v_4k.yaml \
  --models.student.init_from Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --pipeline.flow_shift 5.0 \
  --training.distributed.num_gpus 8 \
  --training.distributed.sp_size 8 \
  --training.distributed.hsdp_shard_dim 8 \
  --training.data.data_path data/real_4k_14b_rung0/training_dataset \
  --training.data.num_height 2160 \
  --training.data.num_width 3840 \
  --training.data.num_frames 121 \
  --training.data.num_latent_t 31 \
  --training.loop.max_train_steps 1 \
  --training.checkpoint.output_dir outputs/wan_finetune_4k_14b_trace_8xh200_121f \
  --training.tracker.run_name wan_4k_14b_trace_8xh200_121f \
  2>&1 | tee /workspace/train_wan21_14b_trace_8xh200_121f.log
