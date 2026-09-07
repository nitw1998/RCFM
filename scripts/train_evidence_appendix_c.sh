#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(direct_cnn|cat)$ || ! "$2" =~ ^(mimic_afib|ptbxl|cpsc2018|wesad|mmecg)$ ]]; then
  echo "Usage: $0 {direct_cnn|cat} {mimic_afib|ptbxl|cpsc2018|wesad|mmecg} [--dry-run] [trainer arguments ...]" >&2
  exit 2
fi

MODEL="$1"
DATASET="$2"
shift 2
DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RCFM_DATA_ROOT:?Set RCFM_DATA_ROOT to the frozen preprocessing root}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to an output directory outside the repository}"

if [[ "$MODEL" == "direct_cnn" ]]; then
  ENTRY="train_direct_cnn.py"
  case "$DATASET" in
    mimic_afib) CONFIG="configs/mimic_afib/direct_cnn_regression_minmax_zero_qc_seed31.yaml" ;;
    ptbxl) CONFIG="configs/ptbxl/direct_cnn_regression_record_minmax_seed31.yaml" ;;
    cpsc2018) CONFIG="configs/cpsc2018/direct_cnn_regression_record_minmax_seed31.yaml" ;;
    wesad) CONFIG="configs/wesad/direct_cnn_regression_window_minmax_seed31.yaml" ;;
    mmecg) CONFIG="configs/mmecg/direct_cnn_regression_window_minmax_seed31.yaml" ;;
  esac
else
  ENTRY="scripts/train_cat.py"
  case "$DATASET" in
    mimic_afib) CONFIG="configs/baselines/cat_ppg_mimic_afib_seed31.yaml" ;;
    ptbxl) CONFIG="configs/ptbxl/cat_ecg_adapted_record_minmax_seed31.yaml" ;;
    cpsc2018) CONFIG="configs/cpsc2018/cat_ecg_adapted_record_minmax_seed31.yaml" ;;
    wesad) CONFIG="configs/wesad/cat_ppg_reproduced_window_minmax_seed31.yaml" ;;
    mmecg) CONFIG="configs/mmecg/cat_rcg_adapted_window_minmax_seed31.yaml" ;;
  esac
fi

COMMAND=(
  "$PYTHON_BIN" "$REPO_ROOT/$ENTRY"
  --config "$REPO_ROOT/$CONFIG"
  --data_root "$DATA_ROOT"
  --output_dir "$RUN_ROOT"
  "$@"
)

if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
