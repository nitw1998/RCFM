import numpy as np

from scripts.audit_ptbxl_cpsc_rmse import TARGET_INDICES, _loader_reference, _source_qc


def test_loader_reference_scales_each_record_and_lead_independently():
    raw = np.zeros((2, 512, 12), dtype=np.float32)
    ramp = np.linspace(-2.0, 3.0, 512, dtype=np.float32)
    for record in range(2):
        for lead in range(12):
            raw[record, :, lead] = ramp * (lead + 1) + record
    result = _loader_reference(raw)
    assert result.shape == (2, len(TARGET_INDICES), 512)
    np.testing.assert_allclose(result[..., 0], -1.0)
    np.testing.assert_allclose(result[..., -1], 1.0)


def test_source_qc_reports_constant_and_near_constant_leads():
    raw = np.tile(np.linspace(0.0, 1.0, 512, dtype=np.float32)[None, :, None], (2, 1, 12))
    raw[0, :, 3] = 0.0
    raw[1, :, 5] *= 5e-4
    summary, near = _source_qc(raw, np.asarray(["a", "b"]))
    exact = next(row for row in summary["threshold_counts"] if row["threshold"] == 0.0)
    assert exact == {"threshold": 0.0, "lead_windows": 1, "records": 1}
    assert {row["record_id"] for row in near} == {"a", "b"}
