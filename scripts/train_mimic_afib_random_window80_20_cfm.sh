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
DATA_ROOT="${MIMIC_AFIB_RANDOM_WINDOW_DATA_ROOT:?Set MIMIC_AFIB_RANDOM_WINDOW_DATA_ROOT to the directory containing MIMIC-AFib/}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
CONFIG="configs/mimic_afib/cfm_random_window80_20_seed31.yaml"
MANIFEST="$DATA_ROOT/MIMIC-AFib/dataset_manifest.json"
RUN_ID="mimic_afib_cfm_random_window80_20_s31_e200_v1"
RUN_DIR="$RUN_ROOT/ppg2ecg/MIMIC-AFib/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

for path in \
  "$MANIFEST" \
  "$DATA_ROOT/MIMIC-AFib/ppg_train_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/ecg_train_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/ppg_test_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/ecg_test_4sec.npy" \
  "$DATA_ROOT/MIMIC-AFib/subject_ids_train.npy" \
  "$DATA_ROOT/MIMIC-AFib/subject_ids_test.npy" \
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
assert m["dataset_version"] == "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1"
assert m["split_hash"] == "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232"
assert m["splits"]["train"]["windows"] == 8160
assert m["splits"]["test"]["windows"] == 2040
assert m["overlap"]["subject_disjoint"] is False
assert m["overlap"]["record_disjoint"] is False
assert m["overlap"]["subjects_in_both_train_and_test"] == 34
assert m["overlap"]["records_in_both_train_and_test"] == 34
assert m["overlap"]["raw_samples_overlap_across_splits"] is False
assert m["overlap"]["allowed_by_protocol"] is True
assert m["normalization"]["normalization_id"] == "rddm_window_minmax_neg1_1_v1"
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
echo "Started MIMIC-AFib random-window 80:20 CFM, 200 epochs."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
