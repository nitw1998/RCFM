# PTB-XL joint-12-lead min-max: Lead II to the other 11 leads

This protocol restores the author's intended per-record 12-lead shared
min-max transform. It is separate from the earlier
`record_minmax_neg1_1_v1` artifact, which independently scaled every lead.

For record `i`, the first four seconds of all 12 physical-mV leads define
one pair of coefficients:

```text
m_i = min over time and all 12 leads
r_i = max over time and all 12 leads - m_i
x_scaled[i, lead, time] = 2 * (x[i, lead, time] - m_i) / r_i - 1
```

Lead II (index 1) is the one-channel condition. The jointly generated targets
are I, III, aVR, aVL, aVF, and V1--V6 (indices
`0,2,3,4,5,6,7,8,9,10,11`). Because every lead receives the same affine
transform, inter-lead amplitude ratios and offsets are preserved.

## Build the preprocessing artifact

The output directory must be outside the Git repository and must be empty:

```bash
python scripts/preprocess_ptbxl.py \
  --source_root "$PTBXL_WFDB_ROOT" \
  --output_dir "$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_joint12_minmax_v1/PTBXL" \
  --source_rate 500 \
  --output_rate 128 \
  --duration_seconds 10 \
  --model_window_seconds 4 \
  --minimum_lead_range 1e-6 \
  --normalization_scope per_record_joint_12lead
```

The physical-mV waveform arrays remain unchanged. The new scalar sidecars are
`record_joint_minima_{split}.npy` and `record_joint_ranges_{split}.npy`.

## Start CFM training

```bash
export PTBXL_JOINT12_DATA_ROOT="$RCFM_RUNS_ROOT/preprocessing/ptbxl_official_joint12_minmax_v1"
export RCFM_PYTHON=/path/to/python
scripts/train_ptbxl_lead2_to_other11_joint12.sh GPU_INDEX
```

The frozen entry uses canonical conditional flow matching, seed 31, 50-step
validation, no minibatch OT, and no region weighting. Training uses official
folds 1--8 and validates on fold 9. Fold 10 is not loaded by this training
entry.

## Scientific boundary

The held-out record coefficient is calculated from Lead II and all 11 target
leads. It therefore contains target-derived scale information and is valid
only as a paired-benchmark reproduction of the historical normalization. It
is not a deployable Lead-II-only preprocessing rule. A deployment experiment
must instead use coefficients derived only from Lead II or frozen training-set
statistics and must be reported as a separate protocol.
