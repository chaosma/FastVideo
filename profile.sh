#!/bin/bash
# Profile the *training compute* (forward/backward kernels: attention, FFN,
# layernorm, optimizer, ...) of run.sh. Memory is already covered by the
# FASTVIDEO_MEM_SNAPSHOT_* path; this is the complementary compute view.
#
# Usage:
#   ./profile.sh [extra run.sh flags ...]                 # torch profiler (Chrome/Perfetto trace)
#   PROFILER=nsys ./profile.sh [extra run.sh flags ...]   # Nsight Systems (kernel timeline)
#
# Examples:
#   MODEL=5b ./profile.sh
#   MODEL=5b PROFILER=nsys ./profile.sh --training.distributed.fsdp_cpu_offload true
#
# Any flags after ./profile.sh are forwarded to run.sh (which forwards "$@" to
# training), exactly like calling run.sh directly.

set -euo pipefail

PROFILER="${PROFILER:-torch}"          # torch | nsys
TS="$(date +%Y%m%d_%H%M%S)"
MODEL="${MODEL:-14b}"
OUTDIR="${OUTDIR:-outputs/profiles/${MODEL}_${PROFILER}_${TS}}"
mkdir -p "${OUTDIR}"

# torch.profiler schedule: skip WAIT steps, WARMUP, then record ACTIVE steps.
# The captured steps are (WAIT+WARMUP+1 .. WAIT+WARMUP+ACTIVE), 1-based.
WAIT="${WAIT:-1}"; WARMUP="${WARMUP:-1}"; ACTIVE="${ACTIVE:-2}"
NEED=$(( WAIT + WARMUP + ACTIVE ))
# Ensure enough steps to reach the active window (override-able via MAX_STEPS).
export MAX_STEPS="${MAX_STEPS:-${NEED}}"

echo "=== profile.sh ==="
echo "Backend:   ${PROFILER}"
echo "Model:     ${MODEL}"
echo "Out dir:   ${OUTDIR}"
echo "Steps:     MAX_STEPS=${MAX_STEPS} (need >= ${NEED} to reach active window)"
echo "Extra:     $*"
echo "=================="

if [ "${PROFILER}" = "torch" ]; then
  FASTVIDEO_TRAIN_PROFILE="${OUTDIR}" \
  FASTVIDEO_TRAIN_PROFILE_WAIT="${WAIT}" \
  FASTVIDEO_TRAIN_PROFILE_WARMUP="${WARMUP}" \
  FASTVIDEO_TRAIN_PROFILE_ACTIVE="${ACTIVE}" \
    bash run.sh "$@"
  echo
  echo "torch profiler trace(s) written under: ${OUTDIR}"
  echo "  *.pt.trace.json.gz  -> open in chrome://tracing  (or https://ui.perfetto.dev)"
  echo "  or:  tensorboard --logdir ${OUTDIR}   (PyTorch Profiler tab)"

elif [ "${PROFILER}" = "nsys" ]; then
  command -v nsys >/dev/null 2>&1 || {
    echo "nsys not found. Install Nsight Systems or use the default torch backend." >&2
    exit 1; }
  # nsys traces the whole process tree (torchrun + its workers). Keep the run
  # short; a full 14B step trace is large.
  nsys profile \
    --output "${OUTDIR}/nsys_report" \
    --force-overwrite true \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=none \
    bash run.sh "$@"
  echo
  echo "nsys report: ${OUTDIR}/nsys_report.nsys-rep"
  echo "  open in the Nsight Systems GUI (File > Open), or:  nsys stats ${OUTDIR}/nsys_report.nsys-rep"
else
  echo "Unknown PROFILER='${PROFILER}' (use 'torch' or 'nsys')" >&2
  exit 1
fi
