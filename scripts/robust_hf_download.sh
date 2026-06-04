#!/bin/bash
# Supervised HF download: restarts `hf download` whenever byte progress
# stalls for >2 minutes. Exits 0 only when the download completes.
set -u
REPO="$1"
DIR="/workspace/.hf_home/hub/models--${REPO//\//--}"
source /workspace/venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export HF_HUB_DOWNLOAD_TIMEOUT=30

attempt=0
while true; do
  attempt=$((attempt+1))
  echo "[supervisor] attempt $attempt for $REPO"
  hf download "$REPO" &
  PID=$!
  prev=-1; stall=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 30
    cur=$(du -sb "$DIR" 2>/dev/null | cut -f1)
    if [ "$cur" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
    prev=$cur
    if [ "$stall" -ge 4 ]; then
      echo "[supervisor] no progress for 2 min at $((cur/1024/1024)) MB — restarting"
      kill "$PID" 2>/dev/null; sleep 5; kill -9 "$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] $REPO download COMPLETE after $attempt attempt(s)"
    exit 0
  fi
  echo "[supervisor] attempt $attempt ended rc=$rc; retrying in 10s"
  sleep 10
done
