# CPSC2018 record-zscore paired evaluation

This entry evaluates the completed preprocessing-matched CFM and RCFM
checkpoints on the untouched CPSC2018 test split. It is a normalization
ablation, not the final paper-aligned min-max experiment.

## Pairing contract

- Both checkpoints must match on dataset version, split hash, normalization,
  source/target leads, alignment, architecture, output shape, and flow path.
- CFM must have `region_weight=0`; RCFM must have a positive region weight.
- One CPU-generated noise tensor is saved and passed unchanged to both models.
- Test records are loaded once in fixed order. Targets are never passed to the
  sampler or used to delineate generated signals.
- NFE defaults to 50 and checkpoint selection remains frozen from validation.

## Clinical applicability

RR, PR, QRS duration, QT, and QTc are measured after independent real and
generated delineation. Fridericia is the default QTc correction. Records
labelled atrial fibrillation are excluded from PR and P-wave measurements.

CPSC2018 exposes no patient identifier in the authorized local source, so the
analysis unit is a record and no subject-level significance claim is allowed.
The four-second records are insufficient for HRV; SDNN and RMSSD are emitted as
blocked. The local MAT files do not establish a physical amplitude unit. P/R/T
height and J+60 ms ST deviation are therefore reported only as explicitly
exploratory record-zscore morphology values, never as mV or clinical amplitude.

Each eligible parameter receives record-level Pearson correlation and
Bland--Altman outputs. Pointwise waveform correlation and Bland--Altman values
are descriptive only because time samples are autocorrelated.

## Full test command

Run from the RCFM coordination workspace root (the directory containing the
`repo` symlink and `runs/`) with the desired physical GPU exposed as `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=3 MPLCONFIGDIR=/tmp/rcfm_cpsc_eval_matplotlib \
python repo/scripts/evaluate_cpsc_zscore_paired.py \
  --cfm_checkpoint runs/cpsc2018_cfm_compare_full/ecg2ecg/CPSC2018/cpsc2018_cfm_compare_record_zscore_no_ot_s31_gpu3_v1/checkpoint_best_rmse.pt \
  --rcfm_checkpoint runs/cpsc2018_record_zscore_full/ecg2ecg/CPSC2018/cpsc2018_record_zscore_no_ot_s31_gpu4_20260804T201820/checkpoint_best_rmse.pt \
  --data_root runs/preprocessing/cpsc2018_record_zscore_qc_v2 \
  --output_dir runs/clinical/cpsc2018_zscore_seed31_test_paired \
  --device cuda:0 \
  --batch_size 64 \
  --steps 50 \
  --noise_seed 2025 \
  --qtc_formula fridericia \
  --delineation_method dwt \
  --st_offset_ms 60
```

The output directory must be absent or empty. It contains prediction arrays,
the exact initial noise, per-record waveform metrics, clinical JSON/CSV files,
agreement figures, hashes, software versions, and the complete command.
