# MIMIC-AFib non-grouped random-window 80:20 comparison

This comparison pools all 10,200 four-second paired windows retained by the
frozen all-zero-PPG QC artifact, then applies
`sklearn.model_selection.train_test_split(test_size=0.2, random_state=31)`.
The resulting 8,160/2,040 files are training and validation data. They are not
a patient-independent test split: all 34 subjects and all 34 source records
occur in both partitions. Windows are nonoverlapping in the source arrays, so
no raw samples are duplicated across the two partitions.

The split changes membership only. It retains the source PPG/ECG pairs and
defers the established MIMIC/RDDM-compatible preprocessing to the loader:
independent per-window, per-modality min--max scaling to `[-1,1]`, followed by
NeuroKit PPG cleaning and Pan--Tompkins ECG cleaning. The per-window scaling
does not preserve physical amplitudes and cannot be inverted after cleaning.

Build the local artifact outside the repository:

```bash
python scripts/prepare_mimic_afib_random_window_split.py \
  --source_dir "$MIMIC_AFIB_QC_ROOT/MIMIC-AFib" \
  --identity_dir "$MIMIC_AFIB_IDENTITY_ROOT" \
  --output_dir "$MIMIC_AFIB_RANDOM_WINDOW_ROOT/MIMIC-AFib"
```

The frozen seed-31 comparator is canonical conditional flow matching with 200
epochs, batch size 128, validation every 20 epochs at NFE 50, no minibatch OT,
no region weighting, and latest-only checkpoint retention. Launch it with:

```bash
export MIMIC_AFIB_RANDOM_WINDOW_DATA_ROOT=/path/containing/MIMIC-AFib
export RCFM_RUNS_ROOT=/path/for/training/output
export RCFM_PYTHON=/path/to/python
./scripts/train_mimic_afib_random_window80_20_cfm.sh GPU_INDEX
```

The launcher validates the exact dataset version, split hash, counts, overlap,
and normalization contract before it starts. It requires strictly more than 20
GiB of free memory at the admission check but cannot reserve that memory against
a simultaneous competing launch.
