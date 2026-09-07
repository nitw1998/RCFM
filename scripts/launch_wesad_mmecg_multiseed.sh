#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(wesad|mmecg)$ || ! "$3" =~ ^(cfm|cfm_ot|rcfm|rcfm_ot|rddm|cat|direct_cnn)$ || ! "$4" =~ ^(32|33|all)$ ]]; then
  echo "Usage: $0 GPU_INDEX {wesad|mmecg} {cfm|cfm_ot|rcfm|rcfm_ot|rddm|cat|direct_cnn} {32|33|all}" >&2
  echo "Seed 31 is reused from the completed single-seed experiment." >&2
  exit 2
fi

GPU_INDEX="$1"
DATASET_KEY="$2"
VARIANT="$3"
SEED_REQUEST="$4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the multiseed training root}"
WORKER="$REPO_ROOT/scripts/run_wesad_mmecg_multiseed_worker.sh"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"

case "$DATASET_KEY" in
  wesad)
    DATA_ROOT="${WESAD_DATA_ROOT:?Set WESAD_DATA_ROOT to the frozen WESAD preprocessing root}"
    DATASET_NAME="WESAD"
    TASK="ppg2ecg"
    DATA_ENV="WESAD_DATA_ROOT"
    MANIFEST="$DATA_ROOT/WESAD/dataset_manifest.json"
    TRAIN_ARRAY="$DATA_ROOT/WESAD/ppg_train_4sec.npy"
    TARGET_ARRAY="$DATA_ROOT/WESAD/ecg_train_4sec.npy"
    ;;
  mmecg)
    DATA_ROOT="${MMECG_DATA_ROOT:?Set MMECG_DATA_ROOT to the frozen mmECG preprocessing root}"
    DATASET_NAME="mmECG"
    TASK="rcg2ecg"
    DATA_ENV="MMECG_DATA_ROOT"
    MANIFEST="$DATA_ROOT/mmECG/dataset_manifest.json"
    TRAIN_ARRAY="$DATA_ROOT/mmECG/ppg_train_4sec.npy"
    TARGET_ARRAY="$DATA_ROOT/mmECG/ecg_train_4sec.npy"
    ;;
esac

case "$VARIANT" in
  cfm) CONFIG="configs/$DATASET_KEY/cfm_window_minmax_no_ot_seed31.yaml"; RUN_PREFIX="${DATASET_KEY}_cfm" ;;
  cfm_ot) CONFIG="configs/$DATASET_KEY/cfm_ot_window_minmax_seed31.yaml"; RUN_PREFIX="${DATASET_KEY}_cfm_exact_ot" ;;
  rcfm) CONFIG="configs/$DATASET_KEY/rcfm_window_minmax_no_ot_seed31.yaml"; RUN_PREFIX="${DATASET_KEY}_rcfm" ;;
  rcfm_ot) CONFIG="configs/$DATASET_KEY/rcfm_window_minmax_exact_ot_seed31.yaml"; RUN_PREFIX="${DATASET_KEY}_rcfm_exact_ot" ;;
  direct_cnn) CONFIG="configs/$DATASET_KEY/direct_cnn_regression_window_minmax_seed31.yaml"; RUN_PREFIX="${DATASET_KEY}_direct_cnn" ;;
  rddm)
    RUN_PREFIX="${DATASET_KEY}_rddm"
    if [[ "$DATASET_KEY" == "wesad" ]]; then CONFIG="configs/wesad/rddm_matched_window_minmax_seed31.yaml"; else CONFIG="configs/mmecg/rddm_adapted_window_minmax_seed31.yaml"; fi
    ;;
  cat)
    RUN_PREFIX="${DATASET_KEY}_cat"
    if [[ "$DATASET_KEY" == "wesad" ]]; then CONFIG="configs/wesad/cat_ppg_reproduced_window_minmax_seed31.yaml"; else CONFIG="configs/mmecg/cat_rcg_adapted_window_minmax_seed31.yaml"; fi
    ;;
esac

if [[ "$SEED_REQUEST" == "all" ]]; then
  SEEDS=(32 33)
  SEED_TAG="s32_s33"
else
  SEEDS=("$SEED_REQUEST")
  SEED_TAG="s$SEED_REQUEST"
fi

REQUIRED_FILES=("$WORKER" "$REPO_ROOT/$CONFIG" "$MANIFEST" "$TRAIN_ARRAY" "$TARGET_ARRAY")
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
  RUN_DIR="$RUN_ROOT/$TASK/$DATASET_NAME/${RUN_PREFIX}_s${seed}_v1"
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

LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/wandb" "$MPL_ROOT"
LOG_PATH="$LOG_ROOT/${DATASET_KEY}_${VARIANT}_${SEED_TAG}_gpu${GPU_INDEX}.log"
PID_PATH="$PID_ROOT/${DATASET_KEY}_${VARIANT}_${SEED_TAG}_gpu${GPU_INDEX}.pid"

if [[ "$DATA_ENV" == "WESAD_DATA_ROOT" ]]; then
  DATA_ENV_ARGS=(WESAD_DATA_ROOT="$DATA_ROOT")
else
  DATA_ENV_ARGS=(MMECG_DATA_ROOT="$DATA_ROOT")
fi
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  MPLCONFIGDIR="$MPL_ROOT" \
  RCFM_RUNS_ROOT="$RUN_ROOT" \
  RCFM_PYTHON="$PYTHON_BIN" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "${DATA_ENV_ARGS[@]}" \
  "$WORKER" "$DATASET_KEY" "$VARIANT" "${SEEDS[@]}" >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started $DATASET_NAME $VARIANT additional seeds ${SEEDS[*]} on GPU $GPU_INDEX (PID $PID)."
echo "Seed 31 is reused; log: $LOG_PATH"
echo "Run root: $RUN_ROOT/$TASK/$DATASET_NAME"
