# Controlled OT experiment instrumentation

The canonical entry point is `train_rcfm.py`. It accepts a JSON-formatted YAML
template with `--config`; explicit CLI arguments override template values. The
five templates in `configs/ot_experiments/` differ only in region supervision,
OT enablement, solver, and sampling strategy. Exact one-to-one experiments use
the `exact` solver and `assignment` sampling. Sinkhorn retains multinomial plan
sampling. The template provenance fields intentionally start with `REQUIRED_`;
training rejects them until corrected subject-wise data, units, and alignment
have been verified.

For a controlled multi-seed comparison, run every template with seeds 31, 47,
and 83 while retaining the same split hash, batch ordering policy, architecture,
optimizer, schedule, validation seed, inference NFE, and checkpoint rules.
Current MIMIC-AFib and mmECG window-level arrays are not valid production inputs.

## Metric schema

Training scalars are `train/total_loss`, `train/velocity_mse`, `train/roi_mse`,
`train/non_roi_mse`, `train/mask_occupancy`, `train/effective_mean_weight`,
`train/gradient_norm`, `train/learning_rate`, `train/epoch`, and
`train/global_step`.

Path scalars are `flow/source_norm`, `flow/target_norm`, `flow/x_t_norm`,
`flow/target_velocity_norm`, `flow/predicted_velocity_norm`,
`flow/velocity_error_norm`, `flow/velocity_cosine_similarity`,
`flow/time_mean`, `flow/time_std`, and `flow/sigma`. The resolved config records
`flow_matcher`; the model output additionally labels `flow/path_type`.

OT scalars are `ot/cost_random_pairing`, `ot/cost_selected_pairing`,
`ot/cost_reduction`, `ot/cost_reduction_ratio`, `ot/cost_matrix_mean`,
`ot/cost_matrix_median`, `ot/cost_matrix_max`,
`ot/cost_to_regularization_ratio`, `ot/regularization_applicable`,
`ot/plan_mass`, `ot/plan_entropy`, `ot/normalized_plan_entropy`,
`ot/plan_max_probability`, `ot/plan_nonzero_fraction`,
`ot/row_marginal_error`, `ot/column_marginal_error`,
`ot/unique_source_count`, `ot/unique_target_count`,
`ot/unique_source_fraction`, `ot/unique_target_fraction`,
`ot/source_duplicate_fraction`, `ot/target_duplicate_fraction`,
`ot/fallback_count`, `ot/nonfinite_plan_count`, and
`ot/solver_warning_count`. Interval-gated histograms cover selected source and
target multiplicity, selected transport costs, and nonzero plan probabilities.

Validation scalars are `val/total_loss`, `val/velocity_mse`, `val/rmse`,
`val/mae`, `val/waveform_fd`, `val/num_samples`, `val/num_subjects`, and
`val/subject_metadata_available`. `waveform_fd` is a Gaussian Frechet distance
computed directly in normalized waveform-vector space; it is not an ECG
feature-embedding FD. Subject counts remain zero until the loader exposes a
verified subject manifest.

## Local output schema

Each run writes below `runs/<task>/<dataset>/<run_id>/`:

- `resolved_config.yaml`: JSON-compatible YAML containing the complete resolved configuration;
- `run_metadata.json`: status, timestamps, summaries, and availability flags;
- `epoch_metrics.csv`: long-format epoch-level training/path metrics;
- `ot_diagnostics.csv`: long-format epoch-level OT diagnostics;
- `validation_metrics.csv`: long-format held-out metrics;
- `clinical_metrics.csv`: clinical hook output or an empty header when unavailable;
- `checkpoint_manifest.json`: checkpoint labels, files, epochs, and global steps;
- `environment.txt`: software and compute environment;
- `git_state.txt`: commit, dirty flag, and changed-file status.

Checkpoints are `latest`, `best_rmse`, `best_waveform_fd`, and
`best_velocity_mse`. No best-clinical checkpoint is created because a clinical
composite has not been defined. Raw predictions are not written by default.

W&B modes are `disabled`, `offline`, and `online`. Dataset roots, output roots,
and local config paths are excluded from W&B config, and raw waveforms and sample
or subject identifiers are never logged.

## Path and coupling guards

Canonical `conditional` matching uses `x_t=(1-t)z+tx`, target velocity `x-z`,
and optional outer minibatch OT only to change the endpoint coupling. `target`
and variance-preserving matchers implement different probability paths; adding
outer OT is therefore noncanonical and is rejected unless the caller explicitly
sets `allow_noncanonical_ot_path=True`. The Schrodinger-bridge matcher already
performs internal OT and is always rejected when outer OT is enabled. This is a
double-coupling guard, not an assertion that the historical target/VP variants
were scientifically equivalent to canonical RCFM.

`association_debug` creates batch-local synthetic sample IDs at the canonical
training entry point and verifies target/condition/mask equality after the exact
sampled target indexing. It also checks all condition branches and masks have the
target batch size before indexing. The check uses IDs, never waveform similarity.
