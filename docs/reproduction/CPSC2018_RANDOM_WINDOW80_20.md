# CPSC2018 non-grouped random-window 80:20 comparison

Every variable-length CPSC2018 source record is resampled from 500 Hz to
128 Hz in full. One min/range is computed over the complete resampled record
and all 12 leads. All complete nonoverlapping four-second windows are then
extracted; the incomplete tail is discarded. Windows are pooled and divided
with `sklearn.model_selection.train_test_split(test_size=0.2,
random_state=31)` without record grouping.

Different windows from the same source record may occur in both training and
validation. CPSC2018 exposes no patient identifier, so patient-disjointness
cannot be checked. This protocol must not be described as record-independent
or patient-independent generalization.

The artifact contains 19,364 training and 4,842 validation windows. A total
of 3,230 source records occur in both splits. Its split hash is
`b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8`.

Start the frozen 200-epoch seed-31 CFM comparison from the coordination
workspace:

```bash
./scripts/launch_cpsc2018_random_window80_20_cfm.sh GPU_INDEX
```

Start the matched RCFM, RCFM-OT, or RDDM cell from the coordination workspace;
the wrapper selects `runs/preprocessing/cpsc2018_random_window80_20_record_joint12_v1`
by default:

```bash
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rddm
```

The entry generates I, III, aVR, aVL, aVF, and V1--V6 jointly from Lead II.
It uses batch 128, validation NFE 50, no minibatch OT, no region weighting,
and latest-only checkpoints. Existing GPU compute processes are permitted;
the point-in-time launch gate requires strictly more than 20 GiB free.
