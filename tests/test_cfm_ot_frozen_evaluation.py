import numpy as np
import pytest

from scripts.assemble_cfm_ot_pair import _paired_inference
from scripts.evaluate_cfm_ot_frozen_reference import _validate_checkpoint


def _contract(dataset="wesad"):
    details = {
        "wesad": ("ppg2ecg", "WESAD", "wesad-subject-fold1-linear-resample-window-minmax-v1", "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd", "window_minmax_neg1_1_v1", "native_common_start_same_window_no_delay_correction_subject_fold1_v1", 1),
    }[dataset]
    task, name, version, split_hash, normalization, alignment, channels = details
    return {
        "kind": "canonical_multistep_cfm_ot",
        "config": {"task": task, "datasets": [name], "dataset_version": version, "split_hash": split_hash,
                   "normalization_id": normalization, "alignment_id": alignment, "window_size": 4,
                   "attention_heads": 8, "flow_matcher": "conditional", "sigma": 0.0, "seed": 31,
                   "region_weight": 0.0, "use_minibatch_ot": True, "ot_method": "exact"},
        "normalization": {"normalization_id": normalization},
        "output_spec": {"channels": channels, "length": 512},
    }


def test_cfm_ot_checkpoint_contract_rejects_region_weight():
    contract = _contract()
    _validate_checkpoint(contract, "wesad")
    contract["config"]["region_weight"] = 0.01
    with pytest.raises(ValueError, match="region_weight"):
        _validate_checkpoint(contract, "wesad")


def test_paired_inference_aggregates_before_testing():
    identities = np.array(["a", "a", "b", "c"])
    cfm = np.array([1.0, 3.0, 2.0, 4.0])
    cfm_ot = np.array([2.0, 4.0, 1.0, 5.0])
    result = _paired_inference(cfm_ot, cfm, identities, 1, 100, "descriptive_only_extremely_underpowered_n3")
    assert result["identity_count"] == 3
    assert result["mean_difference"] == pytest.approx(1 / 3)
    assert result["raw_p_value"] is None
