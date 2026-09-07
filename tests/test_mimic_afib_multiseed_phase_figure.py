import csv

import pytest

from scripts.plot_mimic_afib_multiseed_phase import _read_phase_rows


def test_phase_figure_reader_requires_three_oracle_rows_per_model(tmp_path):
    path = tmp_path / "phase.csv"
    fields = (
        "model",
        "seed",
        "phase_mode",
        "rmse",
        "mae",
        "waveform_fd",
        "pearson_window_median",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM"):
            for seed in (31, 32, 33):
                writer.writerow(
                    {
                        "model": model,
                        "seed": seed,
                        "phase_mode": "oracle_aligned_fixed_support",
                        "rmse": 0.2,
                        "mae": 0.1,
                        "waveform_fd": 2.0,
                        "pearson_window_median": 0.5,
                    }
                )
    values = _read_phase_rows(path)
    assert set(values["CFM"]) == {31, 32, 33}

    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seeds"):
        _read_phase_rows(path)
