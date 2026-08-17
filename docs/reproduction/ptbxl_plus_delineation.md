# PTB-XL+ delineation-mask backbone

This entry trains a standalone single-channel ECG delineation model. It does
not use a diagnostic classifier or Grad-CAM, and it is not part of RCFM
inference. Its eventual purpose is to produce training-only P/QRS/T region
probabilities for a separately controlled region-mask ablation.

## Source and claim boundary

Waveforms come from PTB-XL v1.0.1 `records500`. Fiducials come from PTB-XL+
v1.0.1 per-lead ECGdeli WFDB annotation files. ECGdeli annotations are
algorithm-generated, not manually adjudicated clinical ground truth. Results
against these annotations must be described as annotation agreement.

The data builder preserves the official PTB-XL patient folds: 1--8 train, 9
validation, and 10 test. It joins the already frozen 128-Hz waveform artifact
to PTB-XL+ by public ECG record ID and writes only sparse annotation sidecars.
It never copies or modifies source waveforms.

## Dataset build

Run from the repository root with local roots supplied explicitly:

```bash
python scripts/prepare_ptbxl_plus_delineation.py \
  --waveform_root /path/to/frozen/ptbxl/waveform/artifact \
  --ptbxl_root /path/to/ptb-xl/1.0.1 \
  --ptbxl_plus_root /path/to/ptb-xl-plus/1.0.1 \
  --output_dir /path/outside/repository/ptbxl_plus_delineation_limb_v1 \
  --leads I II III aVR aVL aVF \
  --allow_incomplete_download \
  --verify_checksums
```

`--allow_incomplete_download` does not synthesize annotations. Files absent
from the official release and declared files missing locally are recorded
separately and excluded at record-lead level. Every parsed selected-lead file
is checked against the official SHA-256 manifest. A record-lead is eligible
only when at least two complete QRS-onset/R-peak/QRS-offset triples exist.

Coordinates are mapped from 500 to 128 Hz by nearest physical sample time.
Each ten-second record produces fixed 512-sample crops starting at 0, 384, and
768. Regions crossing a crop boundary are ignored in the affected portion.
Missing or malformed wave types are ignored rather than labelled background.
P-wave supervision is disabled for records carrying PTB-XL AFIB or AFLT
rhythm statements.

## Model and training

`DelineationResUNet1D` is a 4.81-million-parameter residual 1D U-Net with
GroupNorm, transposed-convolution decoding, skip connections, a dilated
bottleneck, and output stride one. It receives one robustly normalized
512-sample ECG lead and emits:

- three independent P/QRS/T region logits;
- nine P/QRS/T onset, peak, and offset heatmap logits.

Training uses masked focal BCE, class-wise soft Dice, and a weighted heatmap
loss. Model selection is performed only on fold 9 using the frozen macro
region Dice, fiducial timing error, and miss-rate score. MIMIC-AFib test
windows must not be used for training, threshold selection, checkpoint
selection, or per-window phase correction.

```bash
PTBXL_PLUS_DELINEATION_ROOT=/path/to/sidecar \
PTBXL_WAVEFORM_ROOT=/path/to/frozen/ptbxl/waveform/artifact \
RCFM_RUNS_ROOT=/path/outside/repository/runs \
python scripts/train_delineation_unet.py \
  --config configs/ptbxl/delineation_limb_resunet_seed31.yaml
```

The production config uses seed 31, batch size 2048, 50 epochs, automatic
mixed precision, online W&B logging, and checkpoints every five epochs. Run
directories include resolved configuration, environment, Git state, local
metrics, checkpoint metadata, and the exact dataset eligibility hash.

## Recovery behavior

The workspace entry `scripts/resume_ptbxl_plus_unet_gpu1.sh` resumes from the
latest complete epoch into a new run directory. It restores model, AdamW,
scheduler, AMP scaler, epoch, global step, best score, and best epoch. The
historical schema does not contain RNG or DataLoader-generator states, so this
is optimization-state continuation rather than bitwise minibatch-order
continuation. Recoverable AMP overflow skips one optimizer update, lowers the
scaler, and is logged; an external interrupt is recorded as
`status=interrupted` with a nonempty exception summary.

## Frozen fold-10 evaluation and visualization

Select the checkpoint exclusively from fold 9, then evaluate it once on the
official fold-10 windows. The evaluator verifies the dataset manifest,
eligibility hash, waveform split hash, checkpoint kind, and selected epoch
before inference. It saves aggregate CSV/JSON metrics and anonymous figures;
it does not save record IDs, patient IDs, waveforms, probability arrays, or
per-window metrics.

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/evaluate_delineation_unet.py \
  --checkpoint /path/to/checkpoint_best.pt \
  --data_root /path/to/ptbxl_plus_delineation_limb_v1 \
  --waveform_root /path/to/frozen/PTBXL \
  --output_dir /path/outside/repository/fold10_evaluation \
  --training_metrics /path/to/initial/metrics.csv /path/to/resume/metrics.csv
```

The region report contains Dice, IoU, precision, recall, and
reference/predicted occupancy for P, QRS, and T masks. Fiducial timing uses
the same validation-frozen threshold, local-maximum rule, and matching
tolerance as training. Visualization examples are selected before inference
at positions equally spaced among fold-10 windows with valid P/QRS/T
supervision; they are not selected by model quality. All results remain
agreement with algorithm-generated ECGdeli labels, not accuracy against
manually adjudicated clinical ground truth.
