#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(cfm|cfm_ot|rcfm|rcfm_ot|rddm)$ || ! "$3" =~ ^(31|32|33|all)$ ]]; then
  echo "Usage: $0 GPU_INDEX {cfm|cfm_ot|rcfm|rcfm_ot|rddm} {31|32|33|all}" >&2
  exit 2
fi

GPU_INDEX="$1"
VARIANT="$2"
SEED_REQUEST="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${CPSC2018_DATA_ROOT:?Set CPSC2018_DATA_ROOT to the frozen preprocessing root}"
RUN_ROOT="${RCFM_RUNS_ROOT:-$REPO_ROOT/runs/training/cpsc2018_factorial_multiseed_v1}"
WORKER="$REPO_ROOT/scripts/run_cpsc2018_factorial_multiseed_worker.sh"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"

if [[ "$SEED_REQUEST" == "all" ]]; then
  SEEDS=(31 32 33)
  SEED_TAG="s31_s32_s33"
else
  SEEDS=("$SEED_REQUEST")
  SEED_TAG="s$SEED_REQUEST"
fi

case "$VARIANT" in
  cfm) CONFIG="configs/cpsc2018/cfm_compare_record_minmax_neg1_1_no_ot_seed31.yaml"; RUN_PREFIX="cpsc2018_cfm_no_ot" ;;
  cfm_ot) CONFIG="configs/cpsc2018/cfm_ot_record_minmax_neg1_1_seed31.yaml"; RUN_PREFIX="cpsc2018_cfm_exact_ot" ;;
  rcfm) CONFIG="configs/cpsc2018/rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"; RUN_PREFIX="cpsc2018_rcfm_pan_no_ot" ;;
  rcfm_ot) CONFIG="configs/cpsc2018/rcfm_record_minmax_neg1_1_exact_ot_seed31.yaml"; RUN_PREFIX="cpsc2018_rcfm_pan_exact_ot" ;;
  rddm) CONFIG="configs/cpsc2018/rddm_adapted_record_minmax_neg1_1_seed31.yaml"; RUN_PREFIX="cpsc2018_rddm_ecg_adapted_minmax" ;;
esac

REQUIRED_FILES=(
  "$WORKER"
  "$REPO_ROOT/$CONFIG"
  "$DATA_ROOT/CPSC2018/dataset_manifest.json"
  "$DATA_ROOT/CPSC2018/X_train_resampled.npy"
  "$DATA_ROOT/CPSC2018/X_val_resampled.npy"
)
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  RUN_DIR="$RUN_ROOT/ecg2ecg/CPSC2018/${RUN_PREFIX}_s${seed}_v1"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing training run: $RUN_DIR" >&2
    exit 1
  fi
done
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
LOG_PATH="$LOG_ROOT/${VARIANT}_${SEED_TAG}_gpu${GPU_INDEX}.log"
PID_PATH="$PID_ROOT/${VARIANT}_${SEED_TAG}_gpu${GPU_INDEX}.pid"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  MPLCONFIGDIR="$MPL_ROOT" \
  CPSC2018_DATA_ROOT="$DATA_ROOT" \
  RCFM_RUNS_ROOT="$RUN_ROOT" \
  RCFM_PYTHON="$PYTHON_BIN" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "$WORKER" "$VARIANT" "${SEEDS[@]}" >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started CPSC2018 $VARIANT seeds ${SEEDS[*]} on GPU $GPU_INDEX (PID $PID)."
echo "Log: $LOG_PATH"
