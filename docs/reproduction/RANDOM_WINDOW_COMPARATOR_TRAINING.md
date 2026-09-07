# Random-window RCFM, RCFM-OT, and RDDM training

Set `RCFM_WORKSPACE` to the coordination-workspace root before using the
workspace launcher below.

The unified launcher freezes the same seed-31, 200-epoch, batch-128 random-window
artifacts used by the corresponding completed CFM runs. It supports 15 cells:
PTB-XL, CPSC2018, MIMIC-AFib, WESAD, or mmECG crossed with RCFM, RCFM-OT,
or RDDM.

From the coordination workspace, launch one cell with:

```bash
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX ptbxl rddm

./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX cpsc2018 rddm

./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mimic-afib rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mimic-afib rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mimic-afib rddm

./scripts/launch_random_window80_20_comparator.sh GPU_INDEX wesad rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX wesad rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX wesad rddm

./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mmecg rcfm
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mmecg rcfm-ot
./scripts/launch_random_window80_20_comparator.sh GPU_INDEX mmecg rddm
```

The launcher refuses an existing run directory, validates the completed dataset
manifest and frozen split hash, requires more than 20 GiB free GPU memory, and
writes logs/PIDs under the root selected by `RCFM_RUNS_ROOT`.

RCFM uses the canonical conditional path and training-only ECG-derived region
mask with weight 0.01. RCFM-OT changes endpoint coupling through exact minibatch
assignment; OT is not an auxiliary loss. RDDM retains the pinned upstream
architecture and loss/schedule constants. The PTB-XL/CPSC ECG, WESAD PPG, and
mmECG RCG cells are explicitly labelled adaptations rather than original-task
reproductions.

All five splits are non-grouped random-window validation protocols. PTB-XL and
CPSC2018 allow different windows from the same source record in both splits;
the three cross-modal datasets allow subject overlap, and mmECG additionally has
overlapping raw samples across split boundaries. These runs do not measure
subject-independent generalization. PTB-XL, CPSC2018, and WESAD scaling uses
complete-source-record extrema computed before the random split, so held-out
target scaling is oracle-only.

## Continue WESAD/mmECG RCFM-OT from epoch 200 to 500

The completed WESAD and mmECG RCFM-OT checkpoints ended their original
200-epoch cosine schedule at learning rate zero. Use the dedicated continuation
entry rather than rerunning the original launcher:

```bash
cd "$RCFM_WORKSPACE"
./scripts/launch_random_window_rcfm_ot_to_e500.sh GPU_INDEX wesad
./scripts/launch_random_window_rcfm_ot_to_e500.sh GPU_INDEX mmecg
```

Each command verifies the exact source-checkpoint SHA-256, epoch/global step,
dataset manifest, split hash, OT assignment contract, and zero terminal learning
rate. It then starts a new non-overwriting run at epoch 201 and stops after epoch
500. Model weights, condition-encoder weights, AdamW moments, global step, best
validation metrics, and RNG states are restored. The exhausted 200-epoch
scheduler is intentionally not restored: learning rate restarts conservatively at `1e-5` and
uses a fresh no-warm-up cosine decay over the remaining 300 epochs. Validation
continues every 20 epochs with the same NFE-50 and fixed-noise protocol.

Expected new run IDs are:

- `wesad_rcfm_ot_random_window80_20_record_minmax_s31_e500_resume_e200_v1`
- `mmecg_rcfm_ot_random_window80_20_s31_e500_resume_e200_v1`

The continuation remains a single-seed leakage-permitting random-window
optimization audit, not a subject-independent evaluation.
