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
DATA_ROOT="${WESAD_RANDOM_WINDOW_RECORD_MINMAX_DATA_ROOT:?Set WESAD_RANDOM_WINDOW_RECORD_MINMAX_DATA_ROOT to the directory containing WESAD/}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
CONFIG="configs/wesad/cfm_random_window80_20_record_minmax_seed31.yaml"
MANIFEST="$DATA_ROOT/WESAD/dataset_manifest.json"
RUN_ID="wesad_cfm_random_window80_20_record_minmax_s31_e200_v2"
RUN_DIR="$RUN_ROOT/ppg2ecg/WESAD/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

for path in \
  "$MANIFEST" \
  "$DATA_ROOT/WESAD/ppg_train_4sec.npy" \
  "$DATA_ROOT/WESAD/ecg_train_4sec.npy" \
  "$DATA_ROOT/WESAD/ppg_test_4sec.npy" \
  "$DATA_ROOT/WESAD/ecg_test_4sec.npy" \
  "$DATA_ROOT/WESAD/target_record_minima_train.npy" \
  "$DATA_ROOT/WESAD/target_record_ranges_train.npy" \
  "$DATA_ROOT/WESAD/condition_record_minima_train.npy" \
  "$DATA_ROOT/WESAD/condition_record_ranges_train.npy" \
  "$DATA_ROOT/WESAD/target_record_minima_test.npy" \
  "$DATA_ROOT/WESAD/target_record_ranges_test.npy" \
  "$DATA_ROOT/WESAD/condition_record_minima_test.npy" \
  "$DATA_ROOT/WESAD/condition_record_ranges_test.npy" \
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
assert m["dataset_version"] == "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2"
assert m["split_hash"] == "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c"
assert m["splits"]["train"]["windows"] == 17365
assert m["splits"]["test"]["windows"] == 4342
assert m["overlap"]["subject_disjoint"] is False
assert m["overlap"]["subjects_in_both_train_and_test"] == 15
assert m["normalization"]["normalization_id"] == "source_record_minmax_neg1_1_v1"
assert m["normalization"]["computed_before_random_window_split"] is True
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
echo "Started WESAD random-window 80:20 record-minmax CFM, 200 epochs."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
