import numpy as np
import torch

from scripts.evaluate_random_window_cfm_checkpoint import (
    SPECS,
    _validate_contract,
    fixed_validation_noise,
)
from scripts.evaluate_ptbxl_fourway import _metric_summary


def test_fixed_validation_noise_restarts_generator_for_each_batch():
    actual = fixed_validation_noise(5, 2, 3, batch_size=2, seed=17)
    expected = []
    for batch_index, count in enumerate((2, 2, 1)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17 + batch_index)
        expected.append(torch.randn((count, 2, 3), generator=generator).numpy())
    np.testing.assert_array_equal(actual, np.concatenate(expected))


def test_fixed_validation_noise_rejects_invalid_shape():
    try:
        fixed_validation_noise(0, 2, 3, batch_size=2, seed=17)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid dimensions must fail")


def test_waveform_summary_accepts_single_channel_label():
    reference = np.zeros((3, 1, 512), dtype=np.float32)
    generated = np.ones_like(reference) * 0.25
    summary, per_window = _metric_summary(
        reference, generated, include_fd=False, target_leads=("single_channel_ECG",)
    )
    assert summary["rmse"] == 0.25
    assert tuple(summary["per_lead"]) == ("single_channel_ECG",)
    assert summary["waveform_fd_definition"].startswith("mean of 1 independent")
    assert per_window["rmse"].shape == (3,)


def test_single_channel_loader_shape_can_be_promoted_to_channel_axis():
    values = np.zeros((4, 512), dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    assert values.shape == (4, 1, 512)


def test_wesad_random_window_evaluation_contract_is_single_channel_128hz():
    assert SPECS["wesad"] == {
        "dataset": "WESAD",
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-window-minmax-v1",
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "records": 4342,
        "task": "ppg2ecg",
        "normalization_id": "window_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("chest_ECG",),
        "condition_lead": "wrist_BVP",
    }


def test_wesad_record_minmax_evaluation_contract_is_frozen():
    assert SPECS["wesad_record_minmax"] == {
        "dataset": "WESAD",
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "records": 4342,
        "task": "ppg2ecg",
        "normalization_id": "source_record_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("chest_ECG",),
        "condition_lead": "wrist_BVP",
    }


def test_mimic_afib_random_window_evaluation_contract_is_single_channel_128hz():
    assert SPECS["mimic_afib"] == {
        "dataset": "MIMIC-AFib",
        "dataset_version": "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1",
        "split_hash": "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232",
        "records": 2040,
        "task": "ppg2ecg",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("upstream_artifact_ecg_channel",),
        "condition_lead": "PPG",
    }


def test_flow_family_contract_distinguishes_region_and_ot_variants():
    base = {
        "kind": "canonical_multistep_rcfm",
        "epoch": 200,
        "config": {
            "task": "ppg2ecg",
            "datasets": ["MIMIC-AFib"],
            "dataset_version": SPECS["mimic_afib"]["dataset_version"],
            "split_hash": SPECS["mimic_afib"]["split_hash"],
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "flow_matcher": "conditional",
            "sigma": 0.0,
            "region_weight": 0.01,
            "use_minibatch_ot": False,
            "seed": 31,
        },
        "output_spec": {
            "channels": 1,
            "length": 512,
            "target_leads": ["upstream_artifact_ecg_channel"],
        },
    }
    _validate_contract(base, "mimic_afib", "rcfm")
    try:
        _validate_contract(base, "mimic_afib", "rcfm_ot")
    except ValueError as error:
        assert "use_minibatch_ot" in str(error)
    else:
        raise AssertionError("RCFM checkpoint must not pass the RCFM-OT contract")
