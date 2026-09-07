from scripts.evaluate_mimic_afib_random_window_rddm_checkpoint import (
    DATASET_SPECS,
    DATASET_VERSION,
    SPLIT_HASH,
    UPSTREAM_COMMIT,
    validate_checkpoint,
)


def _checkpoint():
    return {
        "kind": "independent_rddm_reproduction",
        "epoch": 200,
        "global_step": 12800,
        "rddm_state": {},
        "condition_1_state": {},
        "condition_2_state": {},
        "normalization": {},
        "provenance": {},
        "config": {
            "task": "ppg2ecg",
            "datasets": ["MIMIC-AFib"],
            "dataset_version": DATASET_VERSION,
            "split_hash": SPLIT_HASH,
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "alignment_id": "paired_source_row_no_phase_correction_random80_20_v1",
            "expected_train_windows": 8160,
            "expected_test_windows": 2040,
            "heldout_split": "test",
            "nT": 10,
            "attention_heads": 8,
            "seed": 31,
            "upstream_commit": UPSTREAM_COMMIT,
            "reproduction_label": "RDDM (reproduced)",
            "target_channels": 1,
            "beta_start": 0.0001,
            "beta_end": 0.2,
        },
    }


def test_random_window_rddm_checkpoint_contract_accepts_frozen_endpoint():
    validate_checkpoint(_checkpoint())


def test_random_window_rddm_checkpoint_contract_rejects_wrong_split():
    checkpoint = _checkpoint()
    checkpoint["config"]["split_hash"] = "wrong"
    try:
        validate_checkpoint(checkpoint)
    except ValueError as error:
        assert "split_hash" in str(error)
    else:
        raise AssertionError("wrong RDDM split hash must fail")


def test_random_window_rddm_checkpoint_contract_accepts_wesad_and_mmecg():
    for dataset_key in ("wesad_record_minmax", "mmecg"):
        spec = DATASET_SPECS[dataset_key]
        checkpoint = _checkpoint()
        checkpoint["global_step"] = spec["global_step_epoch_200"]
        checkpoint["config"].update({
            "task": spec["task"],
            "datasets": [spec["dataset"]],
            "dataset_version": spec["dataset_version"],
            "split_hash": spec["split_hash"],
            "normalization_id": spec["normalization_id"],
            "alignment_id": spec["alignment_id"],
            "expected_train_windows": spec["train_rows"],
            "expected_test_windows": spec["heldout_rows"],
            "reproduction_label": spec["reproduction_label"],
        })
        validate_checkpoint(checkpoint, dataset_key)


def test_random_window_rddm_contract_accepts_epoch400_and_training_seed():
    checkpoint = _checkpoint()
    checkpoint["epoch"] = 400
    checkpoint["global_step"] = 25600
    checkpoint["config"]["seed"] = 33
    validate_checkpoint(
        checkpoint, "mimic_afib", expected_training_seed=33, expected_epoch=400
    )


def test_random_window_rddm_contract_accepts_multilead_cpsc_endpoint():
    spec = DATASET_SPECS["cpsc2018"]
    checkpoint = _checkpoint()
    checkpoint["epoch"] = 400
    checkpoint["global_step"] = 60800
    checkpoint["config"].update({
        "task": spec["task"],
        "datasets": [spec["dataset"]],
        "dataset_version": spec["dataset_version"],
        "split_hash": spec["split_hash"],
        "normalization_id": spec["normalization_id"],
        "alignment_id": spec["alignment_id"],
        "expected_train_windows": spec["train_rows"],
        "expected_test_windows": spec["heldout_rows"],
        "heldout_split": "val",
        "reproduction_label": spec["reproduction_label"],
        "target_channels": 11,
        "condition_lead_index": 1,
        "target_lead_indices": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    })
    validate_checkpoint(checkpoint, "cpsc2018", expected_epoch=400)
