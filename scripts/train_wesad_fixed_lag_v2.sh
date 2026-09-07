#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(cfm|rcfm|rcfm_ot|rddm)$ ]]; then
  echo "Usage: $0 GPU_INDEX {cfm|rcfm|rcfm_ot|rddm}" >&2
  exit 2
fi

GPU_INDEX="$1"
VARIANT="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${WESAD_ALIGNED_DATA_ROOT:?Set WESAD_ALIGNED_DATA_ROOT to the directory containing WESAD/}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
MANIFEST="$DATA_ROOT/WESAD/dataset_manifest.json"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"

case "$VARIANT" in
  cfm)
    ENTRY="train_cfm_compare.py"
    CONFIG="configs/wesad/cfm_train_fixed_lag_v2_seed31.yaml"
    ;;
  rcfm)
    ENTRY="train_rcfm.py"
    CONFIG="configs/wesad/rcfm_train_fixed_lag_v2_seed31.yaml"
    ;;
  rcfm_ot)
    ENTRY="train_rcfm.py"
    CONFIG="configs/wesad/rcfm_ot_train_fixed_lag_v2_seed31.yaml"
    ;;
  rddm)
    ENTRY="train_rddm_compare.py"
    CONFIG="configs/wesad/rddm_train_fixed_lag_v2_seed31.yaml"
    ;;
esac

RUN_ID="wesad_${VARIANT}_train_fixed_lag_v2_s31"
RUN_DIR="$RUN_ROOT/ppg2ecg/WESAD/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
LOG_PATH="$LOG_ROOT/${VARIANT}_s31_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${VARIANT}_s31_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

for path in "$MANIFEST" "$DATA_ROOT/WESAD/ppg_train_4sec.npy" "$DATA_ROOT/WESAD/ecg_train_4sec.npy" "$REPO_ROOT/$CONFIG"; do
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

"$PYTHON_BIN" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["status"]=="completed"; assert m["dataset_version"]=="wesad-subject-fold1-train-fixed-lag-aligned-v2"; assert m["alignment"]["calibration_split"]=="training_subjects_only"; assert m["alignment"]["heldout_target_used_for_calibration"] is False' "$MANIFEST"

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
  "$PYTHON_BIN" "$ENTRY" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --run_id "$RUN_ID" \
    >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started WESAD fixed-lag-v2 $VARIANT seed 31 on GPU $GPU_INDEX."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
