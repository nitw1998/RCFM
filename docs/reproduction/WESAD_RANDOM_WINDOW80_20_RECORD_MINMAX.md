# WESAD random-window 80:20 with source-record min--max

This v2 comparison keeps the exact seed-31 window membership of the earlier
WESAD random-window experiment (17,365 training and 4,342 validation windows),
but replaces independent per-window scaling. For each of the 15 subjects, one
minimum and range are computed over that subject's complete retained continuous
record after ECG/BVP resampling and before four-second window splitting. The
same coefficients are reused by every window from that subject.

Set `RCFM_WORKSPACE` to the coordination-workspace root before running the
commands below.

ECG and BVP are scaled independently to `[-1,1]`; their coefficients are never
shared because their physical sensor units are different and unverified. This
preserves amplitude and offset differences between windows within one record.
The raw window arrays remain unchanged, and reversible window-aligned scaler
sidecars are stored in the v2 preprocessing artifact.

All subjects and their continuous records cross the random train/validation
boundary. Consequently, complete-record extrema include validation windows.
This is a target-informed paired-comparison preprocessing protocol, not a
subject-independent or preprocessing-independent generalization experiment.
Using the held-out target ECG scaler to convert a generated waveform back to
the source scale is oracle-only.

Rebuild the frozen artifact from the existing raw-window source:

```bash
cd "$RCFM_WORKSPACE/repo"
python \
  scripts/prepare_wesad_random_window_record_minmax.py \
  --source_dir "$RCFM_WORKSPACE/runs/preprocessing/wesad_subject_fold1_v1/WESAD" \
  --output_dir "$RCFM_WORKSPACE/runs/preprocessing/wesad_random_window80_20_record_minmax_v2/WESAD" \
  --validation_fraction 0.2 \
  --seed 31
```

The workspace already contains the generated and verified artifact. Start the
frozen seed-31 CFM 200-epoch retraining run with one GPU index:

```bash
cd "$RCFM_WORKSPACE"
./scripts/launch_wesad_random_window80_20_record_minmax_cfm.sh GPU_INDEX
```

The launcher validates the manifest, split identity, scaler sidecars, row
counts, and Python/config paths before launch. It refuses to overwrite the
fixed run ID and requires strictly more than 20 GiB free GPU memory. Outputs
are written under
`runs/training/wesad_random_window80_20_record_minmax_cfm_v2/`.

The previous per-window artifact, config, launcher, checkpoints, and results
remain available as v1 and are not overwritten.

## Epoch-200 endpoint evaluation

The completed seed-31 epoch-200 endpoint is evaluated with the frozen
validation noise (seed 2025), 50 flow steps, and all 4,342 validation windows:

```bash
python scripts/evaluate_random_window_cfm_checkpoint.py \
  --dataset wesad_record_minmax \
  --checkpoint "$RCFM_RUNS_ROOT/training/wesad_random_window80_20_record_minmax_cfm_v2/ppg2ecg/WESAD/wesad_cfm_random_window80_20_record_minmax_s31_e200_v2/checkpoint_latest.pt" \
  --data_root "$RCFM_RUNS_ROOT/preprocessing/wesad_random_window80_20_record_minmax_v2" \
  --output_dir "$RCFM_RUNS_ROOT/evaluation/wesad_random_window_record_minmax_cfm_e200_s31_raw_v1" \
  --steps 50 --noise_seed 2025 --deterministic_seed 31
```

The raw normalized-domain RMSE, MAE, waveform FD, and median per-window
Pearson correlation are `0.288114`, `0.168133`, `0.667787`, and `-0.017285`.
The checkpoint SHA-256 is
`50a09896d536d8a0f51672fd17cb9a48aba169a9870f6fdf96ed671020233f30`.

The requested phase analysis selects a target-informed integer shift for each
window by maximizing Pearson correlation over `+/-16` samples (`+/-125 ms` at
128 Hz), then evaluates the common central 480-sample support:

```bash
python scripts/analyze_wesad_random_window_cfm_phase.py \
  --input_dir "$RCFM_RUNS_ROOT/evaluation/wesad_random_window_record_minmax_cfm_e200_s31_raw_v1" \
  --output_dir "$RCFM_RUNS_ROOT/evaluation/wesad_random_window_record_minmax_cfm_e200_s31_phase_v1" \
  --dataset_variant record_minmax --max_lag_samples 16

python scripts/evaluate_wesad_random_window_phase_clinical.py \
  --phase_dir "$RCFM_RUNS_ROOT/evaluation/wesad_random_window_record_minmax_cfm_e200_s31_phase_v1" \
  --output_dir "$RCFM_RUNS_ROOT/clinical/wesad_random_window_record_minmax_cfm_e200_s31_phase_clinical_v1"
```

Oracle-aligned RMSE, MAE, waveform FD, and median per-window Pearson are
`0.264948`, `0.150869`, `0.581233`, and `0.158450`. The median absolute chosen
shift is 10 samples (78.125 ms), and 16.35% of shifts hit the search boundary.
This alignment consumes the paired target ECG and is therefore a morphology
sensitivity analysis, not a deployable preprocessing or inference step.

ECG intervals and amplitudes are delineated independently with NeuroKit2 DWT,
then averaged within subject before the primary Bland--Altman summary. Because
the WESAD ECG sensor unit is unverified, amplitudes remain source-record-
normalized values and must not be reported as mV. Randomly selected four-second
windows are not a continuous time series, so HRV is not computed.
