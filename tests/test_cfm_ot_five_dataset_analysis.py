import csv

from scripts.analyze_cfm_ot_five_dataset import _clinical_error_map


def test_clinical_error_map_uses_absolute_paired_errors(tmp_path):
    path = tmp_path / "parameters.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "patient_id", "lead", "model", "real_qrs_ms", "generated_qrs_ms"])
        writer.writeheader()
        writer.writerow({"record_id": "1", "patient_id": "p1", "lead": "V3", "model": "cfm", "real_qrs_ms": 80, "generated_qrs_ms": 90})
        writer.writerow({"record_id": "1", "patient_id": "p1", "lead": "V3", "model": "cfm_ot", "real_qrs_ms": 80, "generated_qrs_ms": 75})
    result = _clinical_error_map(path)
    assert result["cfm"][("1", "p1", "V3")]["qrs_ms"] == 10
    assert result["cfm_ot"][("1", "p1", "V3")]["qrs_ms"] == 5
