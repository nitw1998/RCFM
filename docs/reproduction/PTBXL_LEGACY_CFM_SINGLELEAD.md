# PTB-XL unchanged legacy CFM: III to V5

Set `RCFM_WORKSPACE` to the coordination-workspace root before using the
workspace launcher below.

This is a controlled retraining branch for diagnosing the historical PTB-XL
result. It restores the submitted code's single-lead task and unchanged legacy
network while retaining the auditable official PTB-XL folds.

## Frozen contract

- Train: official folds 1--8 (17,440 records).
- Reserved validation: fold 9 (2,193 records); it is loaded for integrity and
  count checks but is not used for training or per-epoch selection.
- Final test: fold 10 (2,203 records); it is not loaded by the trainer.
- Condition: lead III, source index 2.
- Target: lead V5, target index 10.
- Window: first 512 samples at 128 Hz.
- Scaling: independent per-record min-max to `[-1,1]` for III and V5.
- Architecture: the exact `DiffusionUNetCrossAttention`, `ConditionNet`, and
  `MinimalFlowMatching` sources from commit
  `c366eee7781eb4c7f6079d3c34b60c73b46d4e5a`.
- Optimization: Adam, learning rate `1e-4`, batch 256, 1,000 epochs, no
  scheduler, no AMP, and gradient clipping at 1.0.

The entry verifies SHA-256 hashes of `model.py` and `train_cfm_basic.py` before
loading data. The expected parameter counts are 45,828,129 for the flow model
and 26,926,016 for the condition network. A mismatch fails rather than silently
using the revised network.

The historical code saved two complete weight files every 20 epochs. With the
current disk already 98% occupied, this branch instead atomically overwrites a
resumable latest checkpoint and latest compatibility weights, then preserves
the final `minimal_cfm_epoch_999.pth` and `condition_net_epoch_999.pth`. This
changes storage policy, not the model or optimization path.

## Launch

From the coordination workspace:

```bash
cd "$RCFM_WORKSPACE"
bash scripts/launch_ptbxl_legacy_cfm_singlelead.sh GPU_INDEX
```

The launcher checks frozen inputs, rejects an existing destination or busy GPU,
starts training with `nohup`, and writes logs/PIDs/W&B data beneath
`runs/training/ptbxl_legacy_cfm_singlelead_v1/`.

This run is a single-lead controlled reproduction and must not be compared as
though it were the revised Lead-II-to-11-lead task. Its result also cannot by
itself validate the historical value near 0.08 because the historical array
split provenance remains unresolved.
