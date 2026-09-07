# DirectCNN on the updated normalized five-dataset matrix

This entry retrains the deterministic DirectCNN baseline on the same frozen
random-window 80:20 preprocessing artifacts used by the current five-dataset
comparison. It does not reuse the older official-fold/subject-split DirectCNN
checkpoints.

From the coordination workspace, launch one dataset on one physical GPU:

```bash
./scripts/launch_direct_cnn_random_window80_20.sh GPU_INDEX DATASET
```

`DATASET` is one of `ptbxl`, `cpsc2018`, `mimic-afib`, `wesad`, or `mmecg`.
For example:

```bash
./scripts/launch_direct_cnn_random_window80_20.sh 0 ptbxl
```

Append `--dry-run` to validate the manifest/config contract and print the exact
command without training. Extra arguments after `--dry-run` or the dataset are
forwarded to `train_direct_cnn.py`; for example, `--wandb_mode disabled`.

The frozen seed-31 configs train for 500 epochs with batch size 128 and write to
`runs/training/direct_cnn_random_window80_20_v1/` by default. The launcher
refuses to overwrite an existing run directory.

The five input contracts are:

- PTB-XL: full-record joint-12-lead min-max normalization, 34,939/8,735 windows.
- CPSC2018: full-record joint-12-lead min-max normalization, 19,364/4,842 windows.
- MIMIC-AFib: RDDM-compatible per-window modality-wise min-max and cleaning,
  8,160/2,040 windows.
- WESAD: source-record modality-wise min-max normalization, 17,365/4,342 windows.
- mmECG: per-window modality-wise min-max normalization, 9,973/2,494 windows.

These random-window splits permit subject and/or record overlap. They are for a
matched reviewer comparison and must not be described as subject-independent
generalization.
