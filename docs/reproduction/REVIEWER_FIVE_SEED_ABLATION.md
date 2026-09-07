# Reviewer five-seed ablation entry point

The reviewer suite uses the prespecified seeds `31,32,33,34,35` for each of
PTB-XL, CPSC2018, MIMIC-AFib, WESAD, and mmECG. The compared methods are CFM,
RCFM, RCFM-OT, and RDDM. CFM, RCFM, and RCFM-OT train for 200 epochs; RDDM
trains for 400 epochs. This produces 100 runs in the complete factorial matrix.

The entry point is:

```bash
export RCFM_PYTHON=/path/to/python
export RCFM_DATA_ROOT=/path/to/preprocessed-data
export RCFM_RUNS_ROOT=/path/to/reviewer-five-seed-runs

./scripts/launch_reviewer_five_seed_ablation.sh GPU_INDEX DATASET MODEL all
```

`DATASET` is one of `ptbxl`, `cpsc2018`, `mimic-afib`, `wesad`, or `mmecg`.
`MODEL` is one of `cfm`, `rcfm`, `rcfm-ot`, or `rddm`. The final argument may
instead be one seed or a comma-separated subset, for example `31` or
`31,32,33`. One invocation runs its selected seeds sequentially on one GPU and
returns after starting a background worker. Allocate different dataset/model
cells to different GPUs; do not start several `all` jobs on the same GPU.

Before consuming GPU time, inspect the resolved commands:

```bash
RCFM_DRY_RUN=1 ./scripts/launch_reviewer_five_seed_ablation.sh 0 ptbxl rcfm-ot all
```

The launcher verifies the frozen dataset version, split hash, window counts,
and seed-31 base config before training. Completed compatible runs are skipped.
An existing incomplete run is never overwritten; diagnose it and set a new
`RCFM_RUN_TAG` for a clean rerun. Logs and PID files are written below
`$RCFM_RUNS_ROOT/logs/reviewer_five_seed/` and
`$RCFM_RUNS_ROOT/pids/reviewer_five_seed/`.

## Statistical contract

All methods use the same five seeds, enabling paired comparisons by seed.
Report each metric as mean ± sample standard deviation across the five seeds
(`ddof=1`), together with the five individual values. State the paired test,
comparison direction, exact two-sided p-value, and any multiplicity correction.
Five training seeds are a small inferential sample, so the individual values
and effect sizes should accompany p-values; seed-level tests must not be
described as patient-level inference.

These entries reuse the documented random-window 80/20 artifacts. Their split
limitations (including record or subject overlap where documented) still apply
and must be disclosed in the response. PTB-XL and CPSC2018 are ECG-to-ECG tasks,
so neither training nor downstream evaluation should apply phase correction.
