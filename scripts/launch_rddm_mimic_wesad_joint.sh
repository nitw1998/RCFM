#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CUDA_VISIBLE_DEVICE_LIST [RUN_ID]" >&2
  echo "Example: $0 0,1,2,3 rddm_mimic_wesad_joint_s31_v1" >&2
  exit 2
fi

GPU_LIST="$1"
RUN_ID="${2:-rddm_mimic_wesad_joint_s31_v1}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RDDM_JOINT_DATA_ROOT:?Set RDDM_JOINT_DATA_ROOT to the cleaned joint artifact root}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the joint training run root}"
CONFIG="$REPO_ROOT/configs/rddm/mimic_wesad_joint_official_clean_seed31.yaml"
MANIFEST="$DATA_ROOT/dataset_manifest.json"
RUN_DIR="$RUN_ROOT/ppg2ecg/MIMIC-AFib__WESAD/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"

for path in "$CONFIG" "$MANIFEST" \
  "$DATA_ROOT/MIMIC-AFib/ecg_train_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/ppg_train_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/region_masks_train.npy" \
  "$DATA_ROOT/WESAD/ecg_train_4sec.npy" \
  "$DATA_ROOT/WESAD/ppg_train_4sec.npy" \
  "$DATA_ROOT/WESAD/region_masks_train.npy"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing to overwrite run: $RUN_DIR" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python executable unavailable: $PYTHON_BIN" >&2; exit 1;
}

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ ${#GPUS[@]} -lt 2 ]]; then
  echo "The frozen config uses DataParallel and requires at least two visible GPUs." >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 1; }
  nvidia-smi -i "$gpu" >/dev/null 2>&1 || { echo "GPU $gpu is unavailable" >&2; exit 1; }
  active="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  [[ -z "$active" ]] || { echo "GPU $gpu already has compute process(es): $active" >&2; exit 1; }
done

mkdir -p "$LOG_ROOT" "$RUN_ROOT/pids" "$RUN_ROOT/wandb"
LOG_PATH="$LOG_ROOT/${RUN_ID}.log"
PID_PATH="$RUN_ROOT/pids/${RUN_ID}.pid"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  PYTHONUNBUFFERED=1 \
  RDDM_JOINT_DATA_ROOT="$DATA_ROOT" \
  RCFM_RUNS_ROOT="$RUN_ROOT" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "$PYTHON_BIN" "$REPO_ROOT/train_rddm_joint_clean.py" \
    --config "$CONFIG" --run_id "$RUN_ID" >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started joint MIMIC-AFib+WESAD RDDM on GPUs $GPU_LIST (PID $PID)."
echo "Log: $LOG_PATH"
echo "Run: $RUN_DIR"
