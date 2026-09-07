# mmECG non-grouped random-window 80:20 comparison

This comparison reuses all 12,467 paired four-second RCG/ECG windows from the
frozen mmECG subject-split artifact. The energy-weighted RCG condition, ECG
target, same-record/same-window pairing, 128 Hz rate, 50% window overlap, and
absence of additional phase correction are unchanged. Only the split changes:
all windows are randomly divided 80:20 with seed 31 without subject or source
record grouping.

The artifact contains 9,973 training and 2,494 validation windows. All 11
subjects and all 91 source records occur in both sets. Moreover, 3,977 pairs
of adjacent windows cross the split boundary; each such pair shares 256 raw
samples (two seconds). This is an intentionally leakage-permitting comparison
and does not measure subject-, record-, or raw-sample-independent
generalization.

RCG and ECG retain independent per-window/per-modality min--max scaling to
`[-1,1]` in the loader because their sensor units are different and
unverified. Held-out arrays retain historical `*_test_*` names but serve as
training validation only.

Start the frozen 200-epoch seed-31 CFM comparison:

```bash
./scripts/launch_mmecg_random_window80_20_cfm.sh GPU_INDEX
```

The launcher permits existing compute processes and requires strictly more
than 20 GiB reported free memory at launch time.
