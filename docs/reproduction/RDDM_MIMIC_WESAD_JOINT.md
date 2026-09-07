# MIMIC-AFib + WESAD joint RDDM reproduction

This experiment is a two-dataset adaptation of the released RDDM training
protocol. It does not claim to reproduce the paper's five-dataset training run,
because the processed BIDMC, CAPNO, and DALIA artifacts and the upstream custom
warm-up scheduler were not released.

## Preprocessing

The artifact builder applies the released loader order independently to every
four-second ECG/PPG pair:

1. cast to `float32` and apply `nan_to_num`;
2. independently scale each ECG and PPG window to `[-1, 1]`;
3. run `neurokit2.ppg_clean(..., sampling_rate=128)`;
4. run `neurokit2.ecg_clean(..., sampling_rate=128,
   method="pantompkins1985")`;
5. detect training-target R peaks and create 32-sample ROI masks.

No held-out target mask is generated. Filtering occurs after min-max scaling,
so the final cleaned arrays are not required to remain inside `[-1, 1]`.

```bash
python scripts/prepare_rddm_mimic_wesad_clean.py \
  --mimic_root "$MIMIC_RDDM_QC_ROOT" \
  --wesad_root "$WESAD_SUBJECT_SPLIT_ROOT" \
  --output_root "$RDDM_JOINT_DATA_ROOT" \
  --workers 8
```

The output manifest records source/output SHA-256 values, package versions,
row counts, processing order, and the pinned RDDM commit. The intended frozen
counts are 8,400/1,800 MIMIC-AFib train/test windows and 17,494/4,213 WESAD
train/test windows.

## Training

The training entry uses joint proportional sampling from the two training
arrays, RDDM `nT=10`, loss weights `100/1`, 1,000 epochs, global batch 512,
and a 20-epoch linear warm-up followed by cosine decay. The warm-up curve is
labelled `paper_inferred_linear_warmup_cosine_v1`: the official repository
imports `lr_scheduler.py`, but that file is absent from commit
`7d5348843c3985c211a23ae5105a2d9497d5156a`.

For the frozen DataParallel configuration, expose at least two idle GPUs:

```bash
export RDDM_JOINT_DATA_ROOT=/path/to/rddm_mimic_wesad_joint_official_clean_v1
export RCFM_RUNS_ROOT=/path/to/joint_training_runs
export RCFM_PYTHON=/path/to/python
scripts/launch_rddm_mimic_wesad_joint.sh 0,1,2,3 \
  rddm_mimic_wesad_joint_s31_v1
```

MIMIC-AFib and WESAD test arrays are not evaluated during training. Evaluation
must report each dataset separately. Official batch-averaged FD and full-split
waveform FD are different estimators and must retain distinct labels.
