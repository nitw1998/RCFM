# Public experiment configurations

This directory contains only RCFM-OT experiments. Every `*.yaml` file must be a
JSON object with both of these properties:

- `region_weight` greater than zero;
- `use_minibatch_ot` equal to `true`.

Dataset `split_hash` values are retained to bind a run to its published split.
Checkpoint, prediction, source-file, and private-artifact SHA-256 values do not
belong in public configs. Put unpublished experiments outside this repository.

Run `python scripts/audit_public_release.py` before committing.

Historical tests for omitted experiment matrices remain in the repository for
audit but are skipped in the public release. After restoring private configs in
a private checkout, set `RCFM_INCLUDE_INTERNAL_CONFIG_TESTS=1` to run them.
