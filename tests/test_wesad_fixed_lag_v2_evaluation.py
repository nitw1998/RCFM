import copy

import pytest
import numpy as np

from scripts.analyze_wesad_fixed_lag_v2_pair import subject_waveform_metrics
from scripts.evaluate_wesad_fixed_lag_v2 import (
    EXPECTED_ALIGNMENT,
    EXPECTED_SPLIT,
    EXPECTED_VERSION,
    validate_contracts,
)


def _contract(ot: bool):
    return {
        "kind": "canonical_multistep_rcfm", "epoch": 500, "global_step": 68500,
        "config": {"task": "ppg2ecg", "datasets": ["WESAD"], "dataset_version": EXPECTED_VERSION,
                   "split_hash": EXPECTED_SPLIT, "normalization_id": "window_minmax_neg1_1_v1",
                   "alignment_id": EXPECTED_ALIGNMENT, "window_size": 4, "attention_heads": 8,
                   "flow_matcher": "conditional", "sigma": 0.0, "region_weight": 0.01,
                   "seed": 31, "use_minibatch_ot": ot, "ot_method": "exact",
                   "ot_sampling_strategy": "assignment" if ot else "multinomial"},
        "normalization": {"normalization_id": "window_minmax_neg1_1_v1"},
        "output_spec": {"channels": 1, "length": 512},
    }


def test_fixed_lag_v2_contract_accepts_matched_rcfm_pair():
    validate_contracts({"rcfm": _contract(False), "rcfm_ot": _contract(True)})


def test_fixed_lag_v2_contract_rejects_v1_alignment():
    contracts = {"rcfm": _contract(False), "rcfm_ot": _contract(True)}
    bad = copy.deepcopy(contracts)
    bad["rcfm"]["config"]["alignment_id"] = "native_common_start_same_window_no_delay_correction_subject_fold1_v1"
    with pytest.raises(ValueError, match="alignment_id"):
        validate_contracts(bad)


def test_subject_waveform_metrics_detect_exact_offset_and_shape():
    reference = np.asarray([[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]]])
    generated = reference + 0.5
    result = subject_waveform_metrics(reference, generated)
    assert result["rmse"] == 0.5 and result["mae"] == 0.5
    assert result["median_record_pearson"] == pytest.approx(1.0)
