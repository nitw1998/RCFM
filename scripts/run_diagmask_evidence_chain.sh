#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
if [[ -z "$MODE" ]]; then
  echo "usage: $0 {transfer|faithfulness|mismatch}" >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-python}
: "${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the workspace runs directory}"
: "${PTB_DIAG_CHECKPOINT:?Set the frozen PTB-XL XResNet1D-101 checkpoint}"
: "${PTB_BENCHMARK_CODE_ROOT:?Set the ecg_ptbxl_benchmarking code root}"
: "${PTB_ALL_STATEMENTS_MLB:?Set the PTB-XL all-statements label binarizer}"
: "${PTB_TRAIN_SCALER:?Set the frozen PTB-XL training scaler}"

COMMON_XRESNET=(
  --checkpoint "$PTB_DIAG_CHECKPOINT"
  --benchmark_code_root "$PTB_BENCHMARK_CODE_ROOT"
  --mlb "$PTB_ALL_STATEMENTS_MLB"
  --scaler "$PTB_TRAIN_SCALER"
  --target_class AFIB
  --source_rate_hz 128
  --crop_aggregation mean_logit
)

case "$MODE" in
  transfer)
    : "${PTB_TRANSFER_ROOT:?Set PTB_TRANSFER_ROOT to the frozen PTB-XL preprocessing directory}"
    : "${PTB_METADATA:?Set PTB_METADATA to ptbxl_database.csv}"
    : "${CPSC_TRANSFER_ROOT:?Set CPSC_TRANSFER_ROOT to the frozen CPSC2018 split directory}"
    : "${MIMIC_TRANSFER_ROOT:?Set MIMIC_TRANSFER_ROOT to the all-window MIMIC split directory}"
    PTB_LABEL_ROOT="$RCFM_RUNS_ROOT/preprocessing/diag_transfer_labels"
    PTB_AF_LABELS="$PTB_LABEL_ROOT/ptbxl_fold10_afib.npy"
    if [[ ! -f "$PTB_AF_LABELS" ]]; then
      "$PYTHON_BIN" scripts/prepare_ptbxl_binary_labels.py \
        --metadata "$PTB_METADATA" \
        --record_ids "$PTB_TRANSFER_ROOT/record_ids_test.npy" \
        --positive_code AFIB --output "$PTB_AF_LABELS"
    fi
    "$PYTHON_BIN" scripts/evaluate_diag_classifier_transfer.py \
      --dataset PTB-XL \
      --waveforms "$PTB_TRANSFER_ROOT/X_test_resampled.npy" \
      --labels "$PTB_AF_LABELS" \
      "${COMMON_XRESNET[@]}" \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_transfer_ptbxl_af_v1"

    "$PYTHON_BIN" scripts/evaluate_diag_classifier_transfer.py \
      --dataset CPSC2018 \
      --waveforms "$CPSC_TRANSFER_ROOT/X_test_resampled.npy" \
      --labels "$CPSC_TRANSFER_ROOT/labels_test.npy" --label_column 1 \
      "${COMMON_XRESNET[@]}" \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_transfer_cpsc_af_v1"

    "$PYTHON_BIN" scripts/evaluate_diag_classifier_transfer.py \
      --dataset MIMIC-AFib \
      --waveforms "$MIMIC_TRANSFER_ROOT/ecg_train_4sec.npy" \
      --waveforms "$MIMIC_TRANSFER_ROOT/ecg_test_4sec.npy" \
      --labels "$MIMIC_TRANSFER_ROOT/afib_labels_train.npy" \
      --labels "$MIMIC_TRANSFER_ROOT/afib_labels_test.npy" \
      --groups "$MIMIC_TRANSFER_ROOT/subject_ids_train.npy" \
      --groups "$MIMIC_TRANSFER_ROOT/subject_ids_test.npy" \
      "${COMMON_XRESNET[@]}" \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_transfer_mimic_af_subject_v1"
    ;;
  faithfulness)
    : "${CPSC_TRANSFER_ROOT:?Set CPSC_TRANSFER_ROOT to the frozen CPSC2018 split directory}"
    : "${MIMIC_TRANSFER_ROOT:?Set MIMIC_TRANSFER_ROOT to the all-window MIMIC split directory}"
    "$PYTHON_BIN" scripts/audit_diagmask_faithfulness.py \
      --dataset CPSC2018 \
      --waveforms "$CPSC_TRANSFER_ROOT/X_test_resampled.npy" \
      --labels "$CPSC_TRANSFER_ROOT/labels_test.npy" --label_column 1 \
      "${COMMON_XRESNET[@]}" \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_faithfulness_cpsc_af_v1"

    "$PYTHON_BIN" scripts/audit_diagmask_faithfulness.py \
      --dataset MIMIC-AFib \
      --waveforms "$MIMIC_TRANSFER_ROOT/ecg_test_4sec.npy" \
      --labels "$MIMIC_TRANSFER_ROOT/afib_labels_test.npy" \
      --groups "$MIMIC_TRANSFER_ROOT/subject_ids_test.npy" \
      "${COMMON_XRESNET[@]}" \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_faithfulness_mimic_af_v1"
    ;;
  mismatch)
    : "${PTB_GENERATION_REFERENCE:?Set PTB_GENERATION_REFERENCE to paired_reference.npz}"
    : "${PTB_AF_LABELS:?Set PTB_AF_LABELS to the aligned fold-10 binary AF labels}"
    : "${PTB_RECORD_MINIMA:?Set PTB_RECORD_MINIMA to aligned 12-lead minima}"
    : "${PTB_RECORD_RANGES:?Set PTB_RECORD_RANGES to aligned 12-lead ranges}"
    : "${INDEPENDENT_CLASSIFIER_TS:?Set an independent TorchScript diagnostic classifier}"
    : "${INDEPENDENT_CLASS_NAMES:?Set its newline-delimited class-name file}"
    : "${CFM_PREDICTIONS:?Set aligned CFM 11-lead predictions}"
    : "${DIAGMASK_PREDICTIONS:?Set aligned DiagMask 11-lead predictions}"
    "$PYTHON_BIN" scripts/evaluate_diagmask_classifier_mismatch.py \
      --dataset PTB-XL \
      --condition "$PTB_GENERATION_REFERENCE:conditions" \
      --real_targets "$PTB_GENERATION_REFERENCE:targets" \
      --prediction "cfm=$CFM_PREDICTIONS" \
      --prediction "diagmask=$DIAGMASK_PREDICTIONS" \
      --labels "$PTB_AF_LABELS" \
      --input_domain record_minmax_neg1_1 \
      --record_minima "$PTB_RECORD_MINIMA" --record_ranges "$PTB_RECORD_RANGES" \
      --mask_generator_checkpoint "$PTB_DIAG_CHECKPOINT" \
      --evaluator_checkpoint "$INDEPENDENT_CLASSIFIER_TS" \
      --backend torchscript --class_names "$INDEPENDENT_CLASS_NAMES" \
      --target_class AFIB --source_rate_hz 128 --normalization record_zscore \
      --output_dir "$RCFM_RUNS_ROOT/diagnostics/diagmask_independent_classifier_ptbxl_v1"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
