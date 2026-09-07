from __future__ import annotations

from scripts.plot_crossmodal_delineation_window_bland_altman import pair_window_rows


def _row(source: str, value: str) -> dict[str, str]:
    row = {
        "algorithm": "ecgdeli_port_fixed", "seed": "31", "window_index": "4",
        "source": source, "success": "True", "group_id": "g", "record_id": "r", "afib": "False",
    }
    for mode in ("raw", "qc"):
        for parameter in ("heart_rate_bpm", "rr_ms", "pr_ms", "qrs_ms", "qt_ms",
                          "p_amplitude", "r_amplitude", "t_amplitude"):
            row[f"{mode}_{parameter}"] = value
    return row


def test_pair_window_rows_pivots_reference_and_generated() -> None:
    result = pair_window_rows([_row("reference", "100"), _row("generated", "101")])
    assert len(result) == 1
    assert result[0]["reference_qc_qrs_ms"] == 100.0
    assert result[0]["generated_qc_qrs_ms"] == 101.0
