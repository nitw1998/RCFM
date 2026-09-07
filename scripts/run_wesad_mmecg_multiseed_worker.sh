#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || ! "$1" =~ ^(wesad|mmecg)$ || ! "$2" =~ ^(cfm|cfm_ot|rcfm|rcfm_ot|rddm|cat|direct_cnn)$ ]]; then
  echo "Usage: $0 {wesad|mmecg} {cfm|cfm_ot|rcfm|rcfm_ot|rddm|cat|direct_cnn} SEED [SEED ...]" >&2
  exit 2
fi

DATASET_KEY="$1"
VARIANT="$2"
shift 2
SEEDS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the multiseed training root}"
WANDB_MODE="${WANDB_MODE:-online}"
EXTRA_ARGS=(--wandb_mode "$WANDB_MODE")
SUPPORTS_WANDB_RUN_NAME=true

case "$DATASET_KEY" in
  wesad)
    DATA_ROOT="${WESAD_DATA_ROOT:?Set WESAD_DATA_ROOT to the frozen WESAD preprocessing root}"
    DATASET_NAME="WESAD"
    TASK="ppg2ecg"
    CONFIG_DIR="wesad"
    PREFIX="wesad"
    WANDB_GROUP="wesad-multiseed-v1"
    ;;
  mmecg)
    DATA_ROOT="${MMECG_DATA_ROOT:?Set MMECG_DATA_ROOT to the frozen mmECG preprocessing root}"
    DATASET_NAME="mmECG"
    TASK="rcg2ecg"
    CONFIG_DIR="mmecg"
    PREFIX="mmecg"
    WANDB_GROUP="mmecg-multiseed-v1"
    ;;
esac

case "$VARIANT" in
  cfm)
    ENTRY="train_cfm_compare.py"
    CONFIG="configs/$CONFIG_DIR/cfm_window_minmax_no_ot_seed31.yaml"
    RUN_PREFIX="${PREFIX}_cfm"
    EXTRA_ARGS+=(--region_weight 0 --no-use_minibatch_ot)
    ;;
  cfm_ot)
    ENTRY="train_cfm_ot.py"
    CONFIG="configs/$CONFIG_DIR/cfm_ot_window_minmax_seed31.yaml"
    RUN_PREFIX="${PREFIX}_cfm_exact_ot"
    EXTRA_ARGS+=(--region_weight 0 --use_minibatch_ot --ot_method exact --ot_sampling_strategy assignment)
    ;;
  rcfm)
    ENTRY="train_rcfm.py"
    CONFIG="configs/$CONFIG_DIR/rcfm_window_minmax_no_ot_seed31.yaml"
    RUN_PREFIX="${PREFIX}_rcfm"
    EXTRA_ARGS+=(--no-use_minibatch_ot)
    ;;
  rcfm_ot)
    ENTRY="train_rcfm.py"
    CONFIG="configs/$CONFIG_DIR/rcfm_window_minmax_exact_ot_seed31.yaml"
    RUN_PREFIX="${PREFIX}_rcfm_exact_ot"
    EXTRA_ARGS+=(--use_minibatch_ot --ot_method exact --ot_sampling_strategy assignment)
    ;;
  rddm)
    ENTRY="train_rddm_compare.py"
    if [[ "$DATASET_KEY" == "wesad" ]]; then
      CONFIG="configs/wesad/rddm_matched_window_minmax_seed31.yaml"
    else
      CONFIG="configs/mmecg/rddm_adapted_window_minmax_seed31.yaml"
    fi
    RUN_PREFIX="${PREFIX}_rddm"
    SUPPORTS_WANDB_RUN_NAME=false
    ;;
  cat)
    ENTRY="scripts/train_cat.py"
    if [[ "$DATASET_KEY" == "wesad" ]]; then
      CONFIG="configs/wesad/cat_ppg_reproduced_window_minmax_seed31.yaml"
    else
      CONFIG="configs/mmecg/cat_rcg_adapted_window_minmax_seed31.yaml"
    fi
    RUN_PREFIX="${PREFIX}_cat"
    SUPPORTS_WANDB_RUN_NAME=false
    ;;
  direct_cnn)
    ENTRY="train_direct_cnn.py"
    CONFIG="configs/$CONFIG_DIR/direct_cnn_regression_window_minmax_seed31.yaml"
    RUN_PREFIX="${PREFIX}_direct_cnn"
    ;;
esac

cd "$REPO_ROOT"
for seed in "${SEEDS[@]}"; do
  if [[ "$seed" != "32" && "$seed" != "33" ]]; then
    echo "Formal additional seeds are restricted to 32 or 33: $seed" >&2
    exit 2
  fi
  RUN_ID="${RUN_PREFIX}_s${seed}_v1"
  RUN_DIR="$RUN_ROOT/$TASK/$DATASET_NAME/$RUN_ID"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing training run: $RUN_DIR" >&2
    exit 1
  fi
  WANDB_ARGS=(--wandb_group "$WANDB_GROUP" --wandb_job_type multiseed-train)
  if [[ "$SUPPORTS_WANDB_RUN_NAME" == true ]]; then
    WANDB_ARGS+=(--wandb_run_name "$RUN_ID")
  fi
  echo "Starting $DATASET_NAME $VARIANT seed $seed at $(date --iso-8601=seconds)"
  "$PYTHON_BIN" "$ENTRY" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --seed "$seed" \
    --run_id "$RUN_ID" \
    "${WANDB_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
  echo "Completed $DATASET_NAME $VARIANT seed $seed at $(date --iso-8601=seconds)"
done
