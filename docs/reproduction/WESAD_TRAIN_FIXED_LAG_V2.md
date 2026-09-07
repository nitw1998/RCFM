# WESAD training-derived fixed-lag alignment v2

Set `RCFM_WORKSPACE` to the coordination-workspace root before using the
workspace launcher below.

This branch tests whether preparing synchronized BVP/ECG pairs before training
reduces the phase burden otherwise left to the generator. It is a separate data
protocol and does not overwrite or relabel the original no-delay-correction
WESAD results.

## Leakage boundary

The fold remains the frozen subject-disjoint split (12 training subjects and
three test subjects). Lag calibration uses continuous signals from the 12
training subjects only. For each training subject, the pipeline detects ECG R
peaks and wrist-BVP systolic peaks and takes the median delay from an R peak to
the first BVP peak in the requested 80--500 ms interval. The applied lag is the
median of those 12 subject estimates.

The same single lag is applied to every training and held-out subject. No
held-out ECG target, generated waveform, or per-window oracle shift is used to
choose it. The delayed BVP is advanced on the continuous stream, invalid edge
support is cropped, and four-second non-overlapping windows are extracted only
afterward. There is no zero padding or circular wraparound.

This alignment deliberately removes a dataset-average pulse-transit delay. It
therefore changes the prediction target and must be treated as a sensitivity
protocol. Models trained on v2 should be compared only with models retrained on
the same v2 arrays; v1 and v2 endpoint metrics are not interchangeable.

## Build

```bash
python scripts/preprocess_wesad_train_lag_aligned.py \
  --source_root "$WESAD_RAW_ROOT" \
  --output_dir "$WESAD_ALIGNED_DATA_ROOT/WESAD"
```

The completed local artifact is under
`runs/preprocessing/wesad_train_fixed_lag_aligned_v2/WESAD/`. Its manifest
records the training-only calibration, output hashes, split, counts, and the
applied 36-sample (281.25 ms) lag. The resulting counts are 17,494 training and
4,213 test windows, unchanged from v1 because every continuous record had
enough discarded tail support to absorb the crop.

## Train

The launcher accepts a GPU index followed by `cfm`, `rcfm`, `rcfm_ot`, or
`rddm`. It verifies that the selected data manifest is the completed v2
training-derived protocol, refuses a busy GPU or an existing destination, and
starts the run with `nohup` while recording its log and PID.

```bash
cd "$RCFM_WORKSPACE"
bash scripts/launch_wesad_fixed_lag_v2.sh 0 rddm
```

The frozen seed-31 configs are in `configs/wesad/` with
`train_fixed_lag_v2` in their names. They preserve the existing architecture,
optimization, split, normalization, and sampling settings so the data alignment
is the intended changed factor.
