import csv

import pytest

from scripts.analyze_wesad_fixed_lag_v2_rddm import _pairwise_clinical


def test_pairwise_clinical_does_not_require_oracle_row_success(tmp_path):
    path = tmp_path / "rows.csv"
    fields = ["model", "phase_mode", "real_heart_rate_bpm", "generated_heart_rate_bpm"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"model": "rddm", "phase_mode": "unshifted", "real_heart_rate_bpm": 60, "generated_heart_rate_bpm": 65})
        writer.writerow({"model": "rddm", "phase_mode": "unshifted", "real_heart_rate_bpm": 80, "generated_heart_rate_bpm": 70})
    # Limit the synthetic fixture to HR while retaining the production helper's contract.
    import scripts.analyze_wesad_fixed_lag_v2_rddm as module
    original = module.CLINICAL_PARAMETERS
    module.CLINICAL_PARAMETERS = ("heart_rate_bpm",)
    try:
        result = _pairwise_clinical(path)["heart_rate_bpm"]
    finally:
        module.CLINICAL_PARAMETERS = original
    assert result["n"] == 2
    assert result["bias"] == pytest.approx(-2.5)
    assert result["mae"] == pytest.approx(7.5)
