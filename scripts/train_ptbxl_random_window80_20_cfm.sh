#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 GPU_INDEX" >&2
  exit 2
fi

GPU_INDEX="$1"
MIN_FREE_MEMORY_MIB=20480
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${PTBXL_RANDOM_WINDOW_DATA_ROOT:?Set PTBXL_RANDOM_WINDOW_DATA_ROOT to the directory containing PTBXL/}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
CONFIG="configs/ptbxl/cfm_compare_random_window80_20_joint12_seed31.yaml"
MANIFEST="$DATA_ROOT/PTBXL/dataset_manifest.json"
RUN_ID="ptbxl_cfm_random_window80_20_record_joint12_s31_e200_v2"
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
assert m["status"] == "completed"
assert m["dataset_version"] == "ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2"
assert m["split_hash"] == "9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c"
assert m["splits"]["train"]["windows"] == 34939
assert m["splits"]["val"]["windows"] == 8735
assert m["overlap"]["record_disjoint"] is False
assert m["overlap"]["patient_disjoint"] is False
assert m["overlap"]["allowed_by_protocol"] is True
assert m["normalization"]["normalization_id"] == "source_record_joint12_minmax_neg1_1_v1"
assert m["normalization"]["preserves_within_record_interwindow_scale"] is True
' "$MANIFEST"

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "GPU $GPU_INDEX is not visible to nvidia-smi." >&2
  exit 1
fi
FREE_MEMORY_MIB="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ ! "$FREE_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free memory for GPU $GPU_INDEX." >&2
  exit 1
fi
if (( FREE_MEMORY_MIB <= MIN_FREE_MEMORY_MIB )); then
  echo "GPU $GPU_INDEX has ${FREE_MEMORY_MIB} MiB free; more than ${MIN_FREE_MEMORY_MIB} MiB (20 GiB) is required." >&2
  exit 1
fi
echo "GPU $GPU_INDEX has ${FREE_MEMORY_MIB} MiB free; launching despite any existing compute processes."

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
echo "Started PTB-XL random-window 80:20 CFM (Lead II to other 11 leads), 200 epochs."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
