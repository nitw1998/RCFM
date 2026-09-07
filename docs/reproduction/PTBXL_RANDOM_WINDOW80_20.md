# PTB-XL non-grouped random-window 80:20 comparison

This intentionally non-grouped comparison approximates the historical split
described by the author. Every 10-second record contributes two nonoverlapping
four-second windows: 0--4 seconds and 4--8 seconds. The 8--10 second tail is
unused. All windows from all PTB-XL records are pooled and divided with
`sklearn.model_selection.train_test_split(test_size=0.2, random_state=31)`.

No patient or source-record grouping is applied. Consequently, different
windows from the same record and patient are allowed in both training and
validation. This protocol must not be described as patient-independent or
record-independent generalization.

Each original 10-second record uses one min/range over its full duration and
all 12 leads. Both its 0--4 and 4--8 second windows reuse that coefficient
pair. The same affine transform is therefore applied across time, Lead II,
and the other 11 leads, preserving both inter-lead relative amplitudes and
offsets and the relative scale between the two windows. Held-out coefficients
use the real target leads and are a paired-benchmark normalization, not a
Lead-II-only deployable rule.

The earlier local v1 artifact computed coefficients independently for each
four-second window. It was superseded before formal training and must not be
used for this comparison.

The generated artifact has 34,939 training and 8,735 validation windows. A
total of 6,983 source records and 6,669 patients occur in both splits. Its
split hash is
`9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c`.

Start the frozen 200-epoch seed-31 CFM comparison from the coordination
workspace:

```bash
./scripts/launch_ptbxl_random_window80_20_cfm.sh GPU_INDEX
```

Start the matched RCFM, RCFM-OT, or RDDM cell from the coordination workspace;
the wrapper selects `runs/preprocessing/ptbxl_random_window80_20_record_joint12_v2`
by default:

```bash
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rddm
```

The entry generates I, III, aVR, aVL, aVF, and V1--V6 jointly from Lead II.
It uses canonical conditional flow matching, batch 128, validation NFE 50,
no minibatch OT, no region weighting, and a space-saving latest-only
checkpoint policy. The launcher permits existing compute processes on the
selected GPU and starts whenever reported free memory is strictly greater
than 20 GiB (20,480 MiB); this is only a point-in-time admission check.
