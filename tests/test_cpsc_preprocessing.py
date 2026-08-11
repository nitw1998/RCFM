from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.preprocess_cpsc2018 import (
    build_argparser,
    clinical_policy,
    make_record_splits,
    preprocess_record,
    record_zscore_statistics,
    select_records_by_lead_quality,
)


def _rows(records_per_label: int = 20):
    rows = []
    for label in range(1, 10):
        for index in range(records_per_label):
            rows.append(
                {
                    "record_id": f"L{label}-{index}",
                    "first_label": label,
                    "labels": (label,),
                }
            )
    return rows


def test_record_split_is_deterministic_disjoint_and_primary_stratified():
    rows = _rows()
    first = make_record_splits(rows, seed=31)
    second = make_record_splits(rows, seed=31)

    assert first == second
    assert len(first["train"]) == 144
    assert len(first["val"]) == 18
    assert len(first["test"]) == 18
    split_sets = [set(first[name]) for name in ("train", "val", "test")]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]
    for split, expected_per_label in (("train", 16), ("val", 2), ("test", 2)):
        counts = {
            label: sum(rows[index]["first_label"] == label for index in first[split])
            for label in range(1, 10)
        }
        assert set(counts.values()) == {expected_per_label}


def test_preprocess_record_reads_struct_and_resamples_first_window(tmp_path: Path):
    time = np.arange(3000, dtype=np.float64) / 500.0
    signal = np.stack([np.sin(2 * np.pi * (lead + 1) * time) for lead in range(12)])
    path = tmp_path / "A0001.mat"
    savemat(path, {"ECG": {"sex": "Male", "age": 50, "data": signal}})

    output = preprocess_record(path)

    assert output.shape == (512, 12)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_cpsc_clinical_policy_disables_hrv_and_physical_amplitudes():
    policy = clinical_policy()

    assert policy["hrv"]["status"] == "disabled_by_author_protocol"
    assert policy["hrv"]["window_concatenation_prohibited"]
    assert policy["physical_amplitude_metrics"]["status"] == "unavailable_unknown_source_unit"
    assert policy["interval_metrics"]["status"] == "eligible_after_independent_delineation"


def test_preprocessing_defaults_require_all_twelve_leads_for_joint_training():
    args = build_argparser().parse_args(
        ["--source_root", "/synthetic/source", "--output_dir", "/synthetic/output"]
    )

    assert args.required_lead_indices == list(range(12))


def test_record_scalers_and_required_lead_qc_are_deterministic():
    values = np.zeros((3, 8, 12), dtype=np.float32)
    ramp = np.arange(8, dtype=np.float32)
    values[:, :, 2] = ramp
    values[:, :, 10] = 2 * ramp
    values[1, :, 2] = 0.0
    rows = [
        {"record_id": f"A{index:04d}", "first_label": 1, "labels": (1,)}
        for index in range(3)
    ]

    means, scales = record_zscore_statistics(values)
    eligible_rows, eligible_values, excluded = select_records_by_lead_quality(
        rows, values, required_lead_indices=[2, 10], minimum_lead_std=1e-6
    )

    assert means.shape == scales.shape == (3, 12)
    assert [row["record_id"] for row in eligible_rows] == ["A0000", "A0002"]
    assert eligible_values.shape == (2, 8, 12)
    assert excluded[0]["record_id"] == "A0001"
    assert excluded[0]["required_lead_indices_below_threshold"] == [2]
