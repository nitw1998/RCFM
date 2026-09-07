from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_crossdomain_region_mask_clinical_main.py"
SPEC = importlib.util.spec_from_file_location("plot_crossdomain_region_mask_clinical_main", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _interval_rows(*, mimic: bool) -> list[dict[str, object]]:
    rows = []
    for model_index, model in enumerate(MODULE.MODEL_ORDER):
        for parameter_index, parameter in enumerate(MODULE.INTERVAL_PARAMETERS):
            row = {"model": model, "parameter": parameter, "status": "ok", "mae": float(10 * model_index + parameter_index)}
            if mimic:
                row["phase_mode"] = "oracle_aligned"
            rows.append(row)
    return rows


def test_extract_cpsc_uses_raw_macro_lead_fields() -> None:
    waveform = {}
    for index, model in enumerate(MODULE.MODEL_ORDER):
        waveform[model] = {"per_record_pearson_median": 0.8 - index * 0.1,
                           "pointwise_bland_altman_descriptive_only": {"lower_limit": -index - 1.0, "upper_limit": index + 1.0}}
    values = MODULE.extract_cpsc({"protocol": {"alignment": "raw synchronized no phase correction", "records": 686},
                                  "waveform_agreement": waveform, "macro_lead_agreement": _interval_rows(mimic=False)})
    np.testing.assert_allclose(values["correlation"], [0.8, 0.7, 0.6, 0.5])
    np.testing.assert_allclose(values["loa_width"], [2.0, 4.0, 6.0, 8.0])
    assert values["interval_mae"].shape == (4, 5)


def test_extract_mimic_uses_only_oracle_aligned_fields() -> None:
    waveform = {}
    for index, model in enumerate(MODULE.MODEL_ORDER):
        waveform[model] = {"oracle_aligned": {"per_record_pearson_median": 0.5 + index * 0.1,
                           "pointwise_bland_altman": {"lower_limit": -0.2 - index, "upper_limit": 0.2 + index}}}
    rows = _interval_rows(mimic=True)
    rows.append({"model": "cfm", "parameter": "rr_ms", "status": "ok", "phase_mode": "unshifted", "mae": 999.0})
    values = MODULE.extract_mimic({"protocol": {"phase_correction": "target-informed per-window oracle Pearson maximization",
                                  "records": 1800, "support_samples": 480}, "waveform_agreement": waveform,
                                  "parameter_agreement": rows})
    np.testing.assert_allclose(values["correlation"], [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(values["loa_width"], [0.4, 2.4, 4.4, 6.4])
    assert values["interval_mae"][0, 0] == 0.0


def test_plot_main_writes_pdf_and_png(tmp_path: Path) -> None:
    values = {"correlation": np.asarray([0.8, 0.7, 0.6, 0.5]), "loa_width": np.asarray([0.4, 0.5, 0.6, 0.7]),
              "interval_mae": np.arange(20, dtype=float).reshape(4, 5)}
    outputs = MODULE.plot_main(values, tmp_path / "figure", ("A", "B", "C"))
    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.stat().st_size > 1000 for path in outputs)
