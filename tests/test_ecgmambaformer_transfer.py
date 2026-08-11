from pathlib import Path

import numpy as np

from src.rcfm.interpretability.ecgmambaformer_compat import (
    diagnostic_class_names,
    record_global_zscore,
)


def test_record_global_zscore_matches_per_record_cross_lead_rule():
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    normalized, mean, scale = record_global_zscore(values)

    assert mean == np.mean(values)
    assert scale == np.std(values)
    assert abs(float(normalized.mean())) < 1e-6
    assert abs(float(normalized.std()) - 1.0) < 1e-6


def test_diagnostic_class_names_are_sorted_and_filter_non_diagnostic(tmp_path: Path):
    path = tmp_path / "scp.csv"
    path.write_text(
        ",description,diagnostic,form,rhythm\n"
        "NORM,normal,1.0,,\n"
        "AFIB,atrial fibrillation,,,1.0\n"
        "1AVB,first degree,1.0,,\n",
        encoding="utf-8",
    )

    assert diagnostic_class_names(path) == ["1AVB", "NORM"]
