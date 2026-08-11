from copy import deepcopy

import pytest

from scripts.evaluate_mmecg_fourway import (
    EXPECTED_ALIGNMENT,
    EXPECTED_DATASET_VERSION,
    EXPECTED_SPLIT_HASH,
    _validate_flow_contracts,
    _validate_rddm_checkpoint,
)


def _flow(kind, region_weight, use_ot):
    config = {
        "task": "rcg2ecg", "datasets": ["mmECG"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "window_size": 4, "attention_heads": 8, "flow_matcher": "conditional",
        "sigma": 0.0, "seed": 31, "region_weight": region_weight,
        "use_minibatch_ot": use_ot, "ot_method": "exact",
    }
    return {
        "kind": kind, "epoch": 500, "global_step": 37500, "config": config,
        "normalization": {"normalization_id": "window_minmax_neg1_1_v1"},
        "output_spec": {"channels": 1, "length": 512},
    }


def _contracts():
    return {
        "cfm": _flow("canonical_multistep_cfm", 0.0, False),
        "rcfm": _flow("canonical_multistep_rcfm", 0.01, False),
        "rcfm_ot": _flow("canonical_multistep_rcfm", 0.01, True),
    }


def test_mmecg_flow_contracts_lock_fourway_roles():
    _validate_flow_contracts(_contracts())


def test_mmecg_flow_contracts_reject_ot_role_swap():
    contracts = _contracts()
    contracts["rcfm_ot"]["config"]["use_minibatch_ot"] = False
    with pytest.raises(ValueError, match="RCFM-OT"):
        _validate_flow_contracts(contracts)


def test_mmecg_rddm_contract_locks_adaptation_and_endpoint():
    checkpoint = {
        "schema_version": 1, "kind": "independent_rddm_reproduction",
        "epoch": 500, "global_step": 37500, "normalization": {},
        "config": {
            "task": "rcg2ecg", "datasets": ["mmECG"],
            "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
            "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
            "heldout_split": "test", "expected_train_windows": 9590,
            "expected_test_windows": 2877, "window_size": 4, "target_channels": 1,
            "nT": 10, "seed": 31, "reproduction_label": "RDDM-RCG (adapted)",
        },
        "provenance": {"upstream_commit": "7d5348843c3985c211a23ae5105a2d9497d5156a"},
    }
    _validate_rddm_checkpoint(checkpoint)
    invalid = deepcopy(checkpoint)
    invalid["config"]["reproduction_label"] = "RDDM (official)"
    with pytest.raises(ValueError, match="frozen mmECG"):
        _validate_rddm_checkpoint(invalid)
