# WESAD non-grouped random-window 80:20 comparison

> This document freezes the original per-window-normalized v1 run. New WESAD
> retraining uses the per-source-record v2 protocol documented in
> `WESAD_RANDOM_WINDOW80_20_RECORD_MINMAX.md`; v1 results are retained only for
> provenance and must not be mixed with v2 results.

This comparison reuses all 21,707 paired four-second windows from the frozen
native-alignment WESAD artifact. Wrist BVP and chest ECG remain linearly
resampled to 128 Hz with common native start/window boundaries and no delay
correction. Only the split changes: all windows are pooled and divided with
`sklearn.model_selection.train_test_split(test_size=0.2,
random_state=31)` without subject grouping.

The artifact contains 17,365 training and 4,342 validation windows. All 15
subjects occur in both sets, so this comparison does not measure
subject-independent generalization. The held-out rows are stored under the
historical `*_test_*` file names but are used as training validation only.

BVP and ECG retain independent per-window/per-modality min--max scaling to
`[-1,1]` in the loader. A shared scaler is inappropriate because these are
different sensors with different unverified units.

Start the frozen 200-epoch seed-31 CFM comparison:

```bash
./scripts/launch_wesad_random_window80_20_cfm.sh GPU_INDEX
```

The launcher permits existing compute processes and requires strictly more
than 20 GiB reported free memory at launch time.
