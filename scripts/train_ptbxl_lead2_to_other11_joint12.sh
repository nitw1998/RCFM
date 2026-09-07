#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 GPU_INDEX" >&2
  exit 2
fi

GPU_INDEX="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${PTBXL_JOINT12_DATA_ROOT:?Set PTBXL_JOINT12_DATA_ROOT to the directory containing PTBXL/}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
CONFIG="configs/ptbxl/cfm_compare_record_joint12_minmax_neg1_1_no_ot_seed31.yaml"
MANIFEST="$DATA_ROOT/PTBXL/dataset_manifest.json"
RUN_ID="ptbxl_cfm_lead2_to_other11_joint12_minmax_s31_v1"
RUN_DIR="$RUN_ROOT/ecg2ecg/PTBXL/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

for path in \
  "$MANIFEST" \
  "$DATA_ROOT/PTBXL/X_train_resampled.npy" \
  "$DATA_ROOT/PTBXL/X_val_resampled.npy" \
  "$DATA_ROOT/PTBXL/record_joint_minima_train.npy" \
  "$DATA_ROOT/PTBXL/record_joint_ranges_train.npy" \
  "$DATA_ROOT/PTBXL/record_joint_minima_val.npy" \
  "$DATA_ROOT/PTBXL/record_joint_ranges_val.npy" \
  "$REPO_ROOT/$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing training run: $RUN_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -c '
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["dataset_version"] == "ptbxl-1.0.1-official-folds-record-joint12-minmax-neg1-1-v1"
assert m["split_hash"] == "8c14a068e98ab485fb9f519b1ff9f414b81be08cc43e41fef55cc4ef081d07cd"
n = m["normalization"]
assert n["normalization_id"] == "record_joint12_minmax_neg1_1_v1"
assert n["preserves_interlead_relative_amplitudes_and_offsets"] is True
assert n["heldout_target_statistics_used"] is True
' "$MANIFEST"

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "GPU $GPU_INDEX is not visible to nvidia-smi." >&2
  exit 1
fi
ACTIVE_PIDS="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$ACTIVE_PIDS" ]]; then
  echo "GPU $GPU_INDEX already has compute process(es): $ACTIVE_PIDS" >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/wandb" "$MPL_ROOT"
cd "$REPO_ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$RUN_ROOT/pycache}" \
  MPLCONFIGDIR="$MPL_ROOT" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "$PYTHON_BIN" train_cfm_compare.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --run_id "$RUN_ID" \
    >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started PTB-XL CFM Lead II to the other 11 leads with joint-12-lead min-max."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
