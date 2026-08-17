import pytest

from scripts.evaluate_ptbxl_fourway import TARGET_LEADS
from scripts.export_ptbxl_per_lead_waveform_metrics import rows_from_summary


def test_per_lead_export_preserves_frozen_lead_order_and_rmse():
    per_lead = {
        lead: {
            "rmse": index + 0.1,
            "mae": index + 0.2,
            "bias": index + 0.3,
            "waveform_fd": index + 0.4,
            "per_record_pearson_mean": 0.5,
            "per_record_pearson_median": 0.6,
        }
        for index, lead in enumerate(TARGET_LEADS)
    }
    rows = rows_from_summary({"models": {"cfm": {"rmse": 0.9, "per_lead": per_lead}}})
    assert [row["lead"] for row in rows] == list(TARGET_LEADS)
    assert rows[TARGET_LEADS.index("V3")]["rmse"] == pytest.approx(TARGET_LEADS.index("V3") + 0.1)
    assert all(row["overall_rmse_all_11_leads"] == 0.9 for row in rows)


def test_per_lead_export_rejects_missing_leads():
    with pytest.raises(ValueError, match="frozen 11 target leads"):
        rows_from_summary({"models": {"cfm": {"rmse": 0.9, "per_lead": {"I": {}}}}})


def test_per_lead_export_restores_order_from_sorted_json_mapping():
    per_lead = {
        lead: {
            "rmse": 0.1, "mae": 0.2, "bias": 0.0, "waveform_fd": 1.0,
            "per_record_pearson_mean": 0.5, "per_record_pearson_median": 0.6,
        }
        for lead in sorted(TARGET_LEADS)
    }
    rows = rows_from_summary({"models": {"cfm": {"rmse": 0.1, "per_lead": per_lead}}})
    assert [row["lead"] for row in rows] == list(TARGET_LEADS)
