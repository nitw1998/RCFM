#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(ptbxl|cpsc2018|mimic-afib|wesad|mmecg)$ ]]; then
  echo "Usage: $0 GPU_INDEX {ptbxl|cpsc2018|mimic-afib|wesad|mmecg} [--dry-run] [trainer arguments ...]" >&2
  exit 2
fi

GPU_INDEX="$1"
DATASET_KEY="$2"
shift 2
DRY_RUN=false
if [[ ${1:-} == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RCFM_DATA_ROOT:?Set RCFM_DATA_ROOT to the preprocessing root containing the dataset directory}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"

case "$DATASET_KEY" in
  ptbxl)
    DATASET_NAME="PTBXL"; TASK="ecg2ecg"
    CONFIG="configs/ptbxl/direct_cnn_random_window80_20_joint12_seed31.yaml"
    RUN_ID="ptbxl_direct_cnn_random_window80_20_joint12_s31_e500_v1"
    ;;
  cpsc2018)
    DATASET_NAME="CPSC2018"; TASK="ecg2ecg"
    CONFIG="configs/cpsc2018/direct_cnn_random_window80_20_joint12_seed31.yaml"
    RUN_ID="cpsc2018_direct_cnn_random_window80_20_joint12_s31_e500_v1"
    ;;
  mimic-afib)
    DATASET_NAME="MIMIC-AFib"; TASK="ppg2ecg"
    CONFIG="configs/mimic_afib/direct_cnn_random_window80_20_seed31.yaml"
    RUN_ID="mimic_afib_direct_cnn_random_window80_20_s31_e500_v1"
    ;;
  wesad)
    DATASET_NAME="WESAD"; TASK="ppg2ecg"
    CONFIG="configs/wesad/direct_cnn_random_window80_20_record_minmax_seed31.yaml"
    RUN_ID="wesad_direct_cnn_random_window80_20_record_minmax_s31_e500_v1"
    ;;
  mmecg)
    DATASET_NAME="mmECG"; TASK="rcg2ecg"
    CONFIG="configs/mmecg/direct_cnn_random_window80_20_seed31.yaml"
    RUN_ID="mmecg_direct_cnn_random_window80_20_s31_e500_v1"
    ;;
esac

MANIFEST="$DATA_ROOT/$DATASET_NAME/dataset_manifest.json"
for path in "$MANIFEST" "$REPO_ROOT/$CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

"$PYTHON_BIN" -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
heldout = config["heldout_split"]
assert manifest["status"] == "completed"
assert manifest["dataset_version"] == config["dataset_version"]
assert manifest["split_hash"] == config["split_hash"]
assert manifest["normalization"]["normalization_id"] == config["normalization_id"]
assert manifest["splits"]["train"]["windows"] == config["expected_train_windows"]
assert manifest["splits"][heldout]["windows"] == config["expected_heldout_windows"]
assert config["epochs"] == 500 and config["seed"] == 31
' "$MANIFEST" "$REPO_ROOT/$CONFIG"

COMMAND=(
  env "CUDA_VISIBLE_DEVICES=$GPU_INDEX"
  "$PYTHON_BIN" "$REPO_ROOT/train_direct_cnn.py"
  --config "$REPO_ROOT/$CONFIG"
  --data_root "$DATA_ROOT"
  --output_dir "$RUN_ROOT"
  --run_id "$RUN_ID"
  "$@"
)

if [[ "$DRY_RUN" == true ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "$RUN_ROOT/$TASK/$DATASET_NAME/$RUN_ID" ]]; then
  echo "Refusing to overwrite existing run: $RUN_ROOT/$TASK/$DATASET_NAME/$RUN_ID" >&2
  exit 1
fi
exec "${COMMAND[@]}"
