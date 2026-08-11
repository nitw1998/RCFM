import json

import numpy as np

from scripts.plot_phase_agreement_main_figure import (
    INTERVALS,
    MODELS,
    PHASE_MODES,
    _interval_arrays,
    _load_dataset,
)


def _summary_payload():
    waveform = {}
    rows = []
    for model_index, model in enumerate(MODELS):
        waveform[model] = {}
        for phase_index, phase_mode in enumerate(PHASE_MODES):
            waveform[model][phase_mode] = {
                "per_record_pearson_median": 0.1 + 0.2 * phase_index,
                "pointwise_bland_altman": {
                    "lower_limit": -0.4 + 0.1 * phase_index,
                    "upper_limit": 0.4 - 0.1 * phase_index,
                },
            }
            for parameter_index, parameter in enumerate(INTERVALS):
                rows.append(
                    {
                        "model": model,
                        "phase_mode": phase_mode,
                        "parameter": parameter,
                        "status": "ok",
                        "n": 100,
                        "mae": 10.0 + model_index + parameter_index - phase_index,
                    }
                )
    return {"waveform_agreement": waveform, "parameter_agreement": rows}


def test_figure_loader_requires_matched_before_after_parameter_rows(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    dataset = _load_dataset(path)
    before, after, change = _interval_arrays(dataset)

    assert before.shape == (4, 5)
    assert after.shape == (4, 5)
    assert np.all(after < before)
    assert np.all(change < 0)


def test_figure_loader_rejects_mismatched_analysis_units(tmp_path):
    payload = _summary_payload()
    payload["parameter_agreement"][0]["n"] = 99
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        _load_dataset(path)
    except ValueError as error:
        assert "matched" in str(error)
    else:
        raise AssertionError("mismatched before/after rows must be rejected")
