#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(ptbxl|cpsc2018|mimic_afib|wesad|mmecg)$ ]]; then
  echo "Usage: $0 {ptbxl|cpsc2018|mimic_afib|wesad|mmecg} SEED [SEED ...]" >&2
  exit 2
fi

DATASET_KEY="$1"
shift
SEEDS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${RCFM_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
PYTHON_BIN="${RCFM_PYTHON:-python}"
RUN_ROOT="${RCFM_RUNS_ROOT:-$WORKSPACE_ROOT/runs/training/rcfm_onestep_facm_five_dataset_v1}"
WANDB_MODE="${WANDB_MODE:-online}"

case "$DATASET_KEY" in
  ptbxl)
    DATASET_NAME="PTBXL"; CONFIG="configs/one_step/facm_ptbxl.yaml"
    DATA_ROOT="${PTBXL_DATA_ROOT:-$WORKSPACE_ROOT/runs/preprocessing/ptbxl_official_minmax_v1}"
    TEACHER="${RCFM_FACM_PTBXL_TEACHER_CHECKPOINT:-$WORKSPACE_ROOT/runs/training/ecg2ecg/PTBXL/ptbxl_rcfm_minmax_neg1_1_no_ot_seed31/checkpoint_epoch_500.pt}"
    ;;
  cpsc2018)
    DATASET_NAME="CPSC2018"; CONFIG="configs/one_step/facm_cpsc2018.yaml"
    DATA_ROOT="${CPSC2018_DATA_ROOT:-$WORKSPACE_ROOT/runs/preprocessing/cpsc2018_multilead_qc_v3}"
    TEACHER="${RCFM_FACM_CPSC2018_TEACHER_CHECKPOINT:-$WORKSPACE_ROOT/runs/training/cpsc2018_multilead_minmax_v1/ecg2ecg/CPSC2018/20260808T054129Z_rcfm/checkpoint_epoch_500.pt}"
    ;;
  mimic_afib)
    DATASET_NAME="MIMIC-AFib"; CONFIG="configs/one_step/facm_mimic.yaml"
    DATA_ROOT="${MIMIC_AFIB_DATA_ROOT:-$WORKSPACE_ROOT/runs/preprocessing/mimic_afib_rddm_zero_ppg_qc_v1}"
    TEACHER="${RCFM_FACM_MIMIC_AFIB_TEACHER_CHECKPOINT:-$WORKSPACE_ROOT/runs/mimic_afib_rcfm_rddm_minmax_zero_qc_no_ot_full/ppg2ecg/MIMIC-AFib/mimic_afib_rcfm_rddm_minmax_zero_qc_no_ot_s31_gpu1_v1/checkpoint_epoch_500.pt}"
    ;;
  wesad)
    DATASET_NAME="WESAD"; CONFIG="configs/one_step/facm_wesad.yaml"
    DATA_ROOT="${WESAD_DATA_ROOT:-$WORKSPACE_ROOT/runs/preprocessing/wesad_subject_fold1_v1}"
    TEACHER="${RCFM_FACM_WESAD_TEACHER_CHECKPOINT:-$WORKSPACE_ROOT/runs/training/wesad_subject_fold1_v1/ppg2ecg/WESAD/20260807T125811Z/checkpoint_epoch_500.pt}"
    ;;
  mmecg)
    DATASET_NAME="mmECG"; CONFIG="configs/one_step/facm_mmecg.yaml"
    DATA_ROOT="${MMECG_DATA_ROOT:-$WORKSPACE_ROOT/runs/preprocessing/mmecg_subject_split_v1}"
    TEACHER="${RCFM_FACM_MMECG_TEACHER_CHECKPOINT:-$WORKSPACE_ROOT/runs/training/mmecg_subject_split_v1/rcg2ecg/mmECG/20260807T044835Z/checkpoint_epoch_500.pt}"
    ;;
esac

for path in "$REPO_ROOT/$CONFIG" "$DATA_ROOT/$DATASET_NAME/dataset_manifest.json" "$TEACHER"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required RCFM-OneStep input: $path" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

cd "$REPO_ROOT"
for seed in "${SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^(31|32|33)$ ]]; then
    echo "Formal RCFM-OneStep seeds are restricted to 31, 32, or 33: $seed" >&2
    exit 2
  fi
  RUN_ID="${DATASET_KEY}_rcfm_onestep_s${seed}_v1"
  if find "$RUN_ROOT/one_step" -mindepth 3 -maxdepth 3 -type d -name "$RUN_ID" -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing to overwrite existing RCFM-OneStep run: $RUN_ID" >&2
    exit 1
  fi
  echo "Starting $DATASET_NAME RCFM-OneStep seed $seed at $(date --iso-8601=seconds)"
  "$PYTHON_BIN" scripts/train_facm_acceleration.py \
    --config "$CONFIG" \
    --teacher_checkpoint "$TEACHER" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --seed "$seed" \
    --run_id "$RUN_ID" \
    --device cuda:0 \
    --wandb_mode "$WANDB_MODE" \
    --wandb_run_name "$RUN_ID"
  echo "Completed $DATASET_NAME RCFM-OneStep seed $seed at $(date --iso-8601=seconds)"
done
