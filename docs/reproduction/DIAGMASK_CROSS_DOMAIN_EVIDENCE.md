# Diagnostic-mask cross-domain evidence chain

This protocol addresses cross-domain transfer of the frozen PTB-XL diagnostic
classifier and its Grad-CAM. It deliberately separates three claims:

1. the classifier retains diagnostic discrimination in a target domain;
2. the Grad-CAM is faithful to that classifier's score;
3. region-weighted generation retains information for a classifier independent
   of the mask generator.

None of these experiments treats diagnostic Grad-CAM as an anatomical
P/QRS/T segmentation. In AF, P-wave measurements can be not applicable while
AF evidence remains present in fibrillatory baseline activity, irregular timing,
and QRS-related temporal context.

## Inputs

All private data, model checkpoints, label sidecars, and result arrays remain
outside Git. Set:

```bash
export RCFM_RUNS_ROOT=/path/to/workspace/runs
export PTB_DIAG_CHECKPOINT=/path/to/frozen/ptb_xresnet101_checkpoint.pth
export PTB_BENCHMARK_CODE_ROOT=/path/to/ecg_ptbxl_benchmarking/code
export PTB_ALL_STATEMENTS_MLB=/path/to/all_statements_mlb.pkl
export PTB_TRAIN_SCALER=/path/to/all_statements_standard_scaler.pkl
export PTB_TRANSFER_ROOT="$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_minmax_v1/PTBXL"
export PTB_METADATA=/path/to/ptbxl_database.csv
export CPSC_TRANSFER_ROOT="$RCFM_RUNS_ROOT/preprocessing/cpsc2018_record_split_v1/CPSC2018"
export MIMIC_TRANSFER_ROOT="$RCFM_RUNS_ROOT/preprocessing/mimic_afib_random_window80_20_v1/MIMIC-AFib"
```

The CPSC label column is fixed to column 1 (`atrial_fibrillation`) by the
preprocessing manifest. The MIMIC transfer command concatenates the disjoint
window partitions solely to recover all 10,200 QC-retained windows. The frozen
PTB model is never fitted or calibrated on those windows, and metrics are
computed only after mean-logit aggregation within source subject/record.

## Stage 1: frozen diagnostic transfer

```bash
./scripts/run_diagmask_evidence_chain.sh transfer
```

The three outputs contain record/window logits, analysis-unit logits, AUROC,
AUPRC, sensitivity, specificity, F1, Brier score, ECE, and stratified bootstrap
confidence intervals. CPSC is record-level because the authorized source does
not expose patient IDs. MIMIC is subject/record-level; windows are never treated
as independent cases.

Run the same entry separately for other strictly mapped CPSC labels by changing
`--target_class` and `--label_column`. Verify every PTB class name from the
frozen label binarizer before declaring a mapping.

The launcher creates aligned fold-10 AFIB labels when they are absent. The
equivalent explicit command is:

```bash
python scripts/prepare_ptbxl_binary_labels.py \
  --metadata /path/to/ptbxl_database.csv \
  --record_ids "$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_minmax_v1/PTBXL/record_ids_test.npy" \
  --positive_code AFIB \
  --output "$RCFM_RUNS_ROOT/preprocessing/diag_transfer_labels/ptbxl_fold10_afib.npy"
```

PTB-XL, CPSC2018, and MIMIC use the same frozen four-second observation support,
classifier crop rule, PTB scaler, and crop aggregation. This avoids letting a
variable number of target-domain crops change the score distribution.

## Stage 2: Grad-CAM faithfulness and stability

The full audit is materially more expensive than scoring because it recomputes
Grad-CAM after controlled transformations and evaluates matched random masks.
Use a deterministic smoke cap first:

```bash
python scripts/audit_diagmask_faithfulness.py \
  --dataset CPSC2018 \
  --waveforms "$CPSC_TRANSFER_ROOT/X_test_resampled.npy" \
  --labels "$CPSC_TRANSFER_ROOT/labels_test.npy" --label_column 1 \
  --checkpoint "$PTB_DIAG_CHECKPOINT" \
  --benchmark_code_root "$PTB_BENCHMARK_CODE_ROOT" \
  --mlb "$PTB_ALL_STATEMENTS_MLB" --scaler "$PTB_TRAIN_SCALER" \
  --max_records 16 --random_replicates 2 --device cuda:0 \
  --output_dir "$RCFM_RUNS_ROOT/diagnostics/diag_faithfulness_cpsc_smoke"
```

After the smoke run, execute:

```bash
./scripts/run_diagmask_evidence_chain.sh faithfulness
```

For each top-10% and top-20% mask, the primary estimands are:

- Grad-CAM deletion logit drop minus matched-random deletion drop;
- Grad-CAM insertion logit gain minus matched-random insertion gain.

A positive paired difference with a confidence interval excluding zero supports
classifier faithfulness. Time-shifted deletion, amplitude stability, noise
stability, occupancy, and degenerate-mask rate are secondary diagnostics. They
do not establish anatomical localization or clinical validity.

## Stage 3: independent downstream-classifier mismatch

This stage reuses existing frozen generated outputs. The preferred evaluator is
a different architecture exported as TorchScript. It must accept `(B,12,T)` and
return pre-sigmoid `(B,K)` logits. Its newline-delimited class-name file defines
the output order.

```bash
export PTB_GENERATION_REFERENCE="$RCFM_RUNS_ROOT/evaluation/ptbxl_diagmask_sixway_fold10_raw_seed2025_v1/paired_reference.npz"
export PTB_AF_LABELS="$RCFM_RUNS_ROOT/preprocessing/diag_transfer_labels/ptbxl_fold10_afib.npy"
export PTB_RECORD_MINIMA="$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_minmax_v1/PTBXL/record_minima_test.npy"
export PTB_RECORD_RANGES="$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_minmax_v1/PTBXL/record_ranges_test.npy"
export CFM_PREDICTIONS="$RCFM_RUNS_ROOT/evaluation/ptbxl_diagmask_sixway_fold10_raw_seed2025_v1/cfm_predictions.npy"
export DIAGMASK_PREDICTIONS="$RCFM_RUNS_ROOT/evaluation/ptbxl_diagmask_sixway_fold10_raw_seed2025_v1/diag_predictions.npy"
export INDEPENDENT_CLASSIFIER_TS=/path/to/independent_classifier.ts
export INDEPENDENT_CLASS_NAMES=/path/to/independent_classifier_classes.txt

./scripts/run_diagmask_evidence_chain.sh mismatch
```

The evaluator reconstructs all 12 leads by inserting the real Lead II condition,
inverts record-minmax normalization with the aligned held-out record statistics,
and applies the declared independent-classifier normalization. It reports each
model and paired candidate-minus-CFM AUROC/AUPRC bootstrap intervals. Reusing the
mask-generator checkpoint is rejected. A second XResNet checkpoint is supported
by the Python entry for diagnostic purposes, but a different architecture is the
stronger response to classifier mismatch.

## Interpretation and stop rules

- Stage 1 supports cross-domain discrimination, not mask localization.
- Stage 2 supports score faithfulness, not anatomical segmentation.
- Stage 3 supports downstream classifier mismatch only when evaluator provenance
  is independent of the mask generator.
- Do not proceed to expensive CPSC/MIMIC DiagMask retraining if the frozen AF
  classifier is at chance or Grad-CAM does not outperform matched random masks.
- If MIMIC generation is retrained, construct an AF-stratified subject-disjoint
  split. The old QC test sidecar contains no AF-positive test windows and is not
  valid for downstream AF classification.

Every final run must retain `summary.json`, the saved logits/CSV, complete command,
checkpoint hashes, class metadata, split manifest, and Git state under `runs/`.

## Lead-matched MIMIC-AFib addendum

The original 12-lead XResNet stress test repeats a single monitor lead and does
not isolate lead mismatch. The lead-matched addendum instead uses a binary
ECGMamba classifier trained on PTB-XL Lead II only. It reads the MIMIC WFDB ECG
in mV, retains only records explicitly labelled Lead II in their headers,
resamples complete records from 125 Hz to 100 Hz, and applies only the PTB-XL
training-fold scalar mean and standard deviation. No MIMIC waveform or label is
used for fitting or calibration.

The local evidence run retained 31 PPG-QC-aligned Lead-II records (16 AF and 15
non-AF) and 9,245 four-second windows. Three AF records with generic `V` or Lead
III headers were excluded. Fifty-five windows containing, or within 200 ms of,
nonfinite source samples were excluded before classification. The frozen
classifier achieved record-level AUROC 0.7458 (stratified-bootstrap 95% CI
0.5625--0.9000) and AUPRC 0.7358 (95% CI 0.5876--0.9152). The prespecified
transfer gate (AUROC at least 0.70 and AUROC-CI lower bound above 0.50) passed.

The consequent Grad-CAM audit used 10 equal-cardinality random controls per
window and aggregated window estimands within each source record. In the 16 AF
records, Grad-CAM-minus-random deletion advantage was 2.4518 AFIB-logit units at
top 10% (95% CI 1.5559--3.2945) and 2.5524 at top 20% (95% CI
1.7203--3.4449). Top-10% insertion was inconclusive (-0.1188, 95% CI
-0.8234--0.5014), while top-20% insertion was positive (0.6696, 95% CI
0.2927--1.0611). AF-window amplitude/noise mask Spearman means were
0.7914/0.8115, but 16.96% of AF-window masks were degenerate. Therefore this
addendum supports record-level cross-domain discrimination and deletion
faithfulness for the matched single-lead classifier; it does not validate every
window, anatomical localization, or clinical utility.

The executable entries are:

```bash
python scripts/prepare_mimic_leadii_physical_windows.py --help
python scripts/evaluate_singlelead_af_transfer.py --help
python scripts/audit_singlelead_af_faithfulness.py --help
```

The authoritative local artifacts are
`runs/preprocessing/mimic_afib_leadii_physical_v1/`,
`runs/evaluation/mimic_afib_leadii_singlelead_transfer_v1/`, and
`runs/diagnostics/mimic_afib_leadii_singlelead_faithfulness_v1/`.
