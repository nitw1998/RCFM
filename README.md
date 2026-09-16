# Region-Aware Conditional Flow Matching

This repository contains the code for **Region-Aware Conditional Flow Matching
for Cross-Modal Physiological Signal Generation**. RCFM generates diagnostic ECG
waveforms from heterogeneous source signals, including reduced-lead ECG, PPG,
and radar cardiogram (RCG) signals.

## Public release scope

The public experiment configurations are restricted to **RCFM-OT**, defined as
region-aware training (`region_weight > 0`) with minibatch optimal-transport
coupling (`use_minibatch_ot: true`). Other baselines, unpublished variants,
checkpoint fingerprints, and private review material are intentionally outside
the public release. Dataset split hashes remain because they identify the
published cohort partition without exposing patient data or model weights.

Run the release guard before pushing:

```bash
python scripts/audit_public_release.py
```

The main paper draft is in:

```text
paper/Region-Aware Conditional Flow Matching for Cross-Modal Physiological Signal Generation.tex
```

## Method Components

- Conditional flow matching backbone for 1-D ECG waveform generation.
- Multi-scale U-Net condition encoder with cross-attention.
- Region-aware velocity loss, weighted as `1 + lambda * mask` on target ECG
  diagnostic regions.
- Optional minibatch optimal transport coupling between Gaussian source samples
  and target ECG windows.

Core implementation:

```text
rcfm.py               # RCFM loss, minibatch OT coupling, Euler sampler
model.py              # 1-D U-Net and condition encoder
data.py               # paired ECG/source-signal dataset loaders
train_rcfm.py         # main training CLI
infer_rcfm.py         # checkpoint inference CLI
metrics.py            # signal metrics used by evaluation scripts
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build that matches your machine if the default
CPU/GPU wheel is not appropriate.

## Data Layout

The provided loaders expect preprocessed NumPy windows sampled at 128 Hz.

For PPG-to-ECG and RCG-to-ECG:

```text
data/<DATASET>/ecg_train_4sec.npy
data/<DATASET>/ecg_test_4sec.npy
data/<DATASET>/ppg_train_4sec.npy
data/<DATASET>/ppg_test_4sec.npy
```

For ECG-to-ECG:

```text
data/<DATASET>/X_train_resampled.npy
data/<DATASET>/X_val_resampled.npy
```

The historical mmECG preprocessing stores RCG windows using the `ppg_*` file
names, so use `--task rcg2ecg --datasets mmECG`.

## Training

Use one of the JSON-formatted YAML files under `configs/`. For example:

```bash
python train_rcfm.py \
  --config configs/mimic_afib/rcfm_ot_random_window80_20_seed31.yaml \
  --data_root /path/to/preprocessed-data \
  --output_dir /path/to/run-output
```

The following direct CLI examples show the supported task shapes; enable both
region weighting and minibatch OT when translating them into an experiment.

PPG-to-ECG:

```bash
python train_rcfm.py \
  --task ppg2ecg \
  --datasets MIMIC-AFib \
  --epochs 20 \
  --batch_size 128 \
  --output_dir saved/reviewer
```

RCG-to-ECG:

```bash
python train_rcfm.py \
  --task rcg2ecg \
  --datasets mmECG \
  --epochs 20 \
  --batch_size 128
```

ECG-to-ECG:

```bash
python train_rcfm.py \
  --task ecg2ecg \
  --datasets PTBXL \
  --epochs 20 \
  --batch_size 128
```

Useful switches:

```text
--flow_matcher {conditional,target,sb,vp}
--region_weight 1.0
--use_minibatch_ot / --no-use_minibatch_ot
--max_batches 2
```

`--max_batches` is intended for smoke tests and debugging.

## Inference

```bash
python infer_rcfm.py \
  --task ppg2ecg \
  --datasets MIMIC-AFib \
  --checkpoint_dir saved/reviewer/ppg2ecg/MIMIC-AFib \
  --num_samples 16 \
  --steps 50 \
  --output_dir outputs/rcfm_mimic
```

This writes `rcfm_predictions.npy` and `rcfm_prediction.png`.

## Quick Validation

Compile the Python entry points:

```bash
python -m py_compile rcfm.py data.py model.py train_rcfm.py infer_rcfm.py
```

Run a short training smoke test:

```bash
python train_rcfm.py \
  --task ppg2ecg \
  --datasets MIMIC-AFib \
  --epochs 1 \
  --batch_size 2 \
  --max_batches 1 \
  --num_workers 0 \
  --device cpu \
  --region_weight 0.01 \
  --use_minibatch_ot \
  --output_dir /tmp/rcfm_smoke
```

For full experiments, use GPU training and remove `--max_batches`.
