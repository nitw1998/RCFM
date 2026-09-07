#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {cfm|cfm_ot|rcfm|rcfm_ot|rddm} SEED [SEED ...]" >&2
  exit 2
fi

VARIANT="$1"
shift
SEEDS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${CPSC2018_DATA_ROOT:?Set CPSC2018_DATA_ROOT to the frozen preprocessing root}"
RUN_ROOT="${RCFM_RUNS_ROOT:-$REPO_ROOT/runs/training/cpsc2018_factorial_multiseed_v1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="cpsc2018-core-factorial-multiseed-v1"
EXTRA_ARGS=(--wandb_mode "$WANDB_MODE")

case "$VARIANT" in
  cfm)
    ENTRY="train_cfm_compare.py"
    CONFIG="configs/cpsc2018/cfm_compare_record_minmax_neg1_1_no_ot_seed31.yaml"
    RUN_PREFIX="cpsc2018_cfm_no_ot"
    EXTRA_ARGS+=(--validation_interval_epochs 1 --region_weight 0 --no-use_minibatch_ot)
    ;;
  cfm_ot)
    ENTRY="train_cfm_ot.py"
    CONFIG="configs/cpsc2018/cfm_ot_record_minmax_neg1_1_seed31.yaml"
    RUN_PREFIX="cpsc2018_cfm_exact_ot"
    EXTRA_ARGS+=(--validation_interval_epochs 1 --region_weight 0 --use_minibatch_ot --ot_method exact --ot_sampling_strategy assignment)
    ;;
  rcfm)
    ENTRY="train_rcfm.py"
    CONFIG="configs/cpsc2018/rcfm_record_minmax_neg1_1_no_ot_seed31.yaml"
    RUN_PREFIX="cpsc2018_rcfm_pan_no_ot"
    EXTRA_ARGS+=(--validation_interval_epochs 1 --no-use_minibatch_ot)
    ;;
  rcfm_ot)
    ENTRY="train_rcfm.py"
    CONFIG="configs/cpsc2018/rcfm_record_minmax_neg1_1_exact_ot_seed31.yaml"
    RUN_PREFIX="cpsc2018_rcfm_pan_exact_ot"
    EXTRA_ARGS+=(--validation_interval_epochs 1 --use_minibatch_ot --ot_method exact --ot_sampling_strategy assignment)
    ;;
  rddm)
    ENTRY="train_rddm_compare.py"
    CONFIG="configs/cpsc2018/rddm_adapted_record_minmax_neg1_1_seed31.yaml"
    RUN_PREFIX="cpsc2018_rddm_ecg_adapted_minmax"
    ;;
  *)
    echo "Unknown CPSC2018 factorial variant: $VARIANT" >&2
    exit 2
    ;;
esac

cd "$REPO_ROOT"
for seed in "${SEEDS[@]}"; do
  if [[ "$seed" != "31" && "$seed" != "32" && "$seed" != "33" ]]; then
    echo "Formal seeds are restricted to 31, 32, and 33: $seed" >&2
    exit 2
  fi
  RUN_ID="${RUN_PREFIX}_s${seed}_v1"
  RUN_DIR="$RUN_ROOT/ecg2ecg/CPSC2018/$RUN_ID"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing training run: $RUN_DIR" >&2
    exit 1
  fi
  echo "Starting CPSC2018 $VARIANT seed $seed at $(date --iso-8601=seconds)"
  "$PYTHON_BIN" "$ENTRY" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --seed "$seed" \
    --run_id "$RUN_ID" \
    --wandb_group "$WANDB_GROUP" \
    --wandb_job_type multiseed-factorial-train \
    "${EXTRA_ARGS[@]}"
  echo "Completed CPSC2018 $VARIANT seed $seed at $(date --iso-8601=seconds)"
done
