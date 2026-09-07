#!/usr/bin/env python3
"""Audit WESAD with the unofficial full Python port of KIT-IBT ECGdeli.

The official ECGdeli implementation is MATLAB-only. This entry point therefore
labels every output as an unofficial-port sensitivity result. R peaks are taken
from the prior controlled audit and supplied to ECGdeli as a reference FPT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_ecgdeli_port_audit_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy


FS = 128.0
PAD = 16
EXPECTED_SHAPE = (4342, 1, 480)
METHOD = "ecgdeli_unofficial_python_port_supplied_r_peaks"
CASES = [
    ("reference QRS high", 3104, "reference"),
    ("generated QRS high", 804, "generated"),
    ("reference PR low", 1280, "reference"),
    ("generated PR low", 2409, "generated"),
    ("large P error", 1682, "reference"),
    ("control", 4, "reference"),
    ("control", 166, "reference"),
    ("control", 340, "reference"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decode_fpt_samples(samples: np.ndarray, *, pad: int, signal_length: int) -> list[dict[str, int]]:
    """Decode the port's NaN-at-missing arithmetic FPT view into local rows."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(f"expected an N x 12 FPT sample array, got {values.shape}")
    rows: list[dict[str, int]] = []
    names = {
        "p_onset": 0,
        "p_peak": 1,
        "p_offset": 2,
        "qrs_onset": 3,
        "r_peak": 5,
        "qrs_offset": 7,
        "t_peak": 10,
        "t_offset": 11,
    }
    for beat in values:
        r_peak = beat[5]
        if not np.isfinite(r_peak) or not pad <= r_peak < pad + signal_length:
            continue
        row = {
            name: int(round(beat[column])) - pad if np.isfinite(beat[column]) else -1
            for name, column in names.items()
        }
        rows.append(row)
    return rows


def _decorate(row: dict[str, object], signal: np.ndarray) -> dict[str, object]:
    p_on, p_peak, p_off = (int(row[key]) for key in ("p_onset", "p_peak", "p_offset"))
    q_on, r_peak, q_off = (int(row[key]) for key in ("qrs_onset", "r_peak", "qrs_offset"))
    pr_ms = (q_on - p_on) * 1000.0 / FS if p_on >= 0 and q_on > p_on else None
    qrs_ms = (q_off - q_on) * 1000.0 / FS if q_on >= 0 and q_off > q_on else None
    p_amplitude = None
    if 0 <= p_peak < len(signal) and q_on > 0:
        baseline_start = p_off if 0 <= p_off < q_on else max(0, q_on - round(0.04 * FS))
        if baseline_start < q_on:
            p_amplitude = float(signal[p_peak] - np.median(signal[baseline_start:q_on]))
    row.update({
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "p_amplitude": p_amplitude,
        "p_order_valid": 0 <= p_on < p_peak < p_off < q_on < r_peak,
        "qrs_order_valid": 0 <= q_on < r_peak < q_off,
        "pr_plausible_80_220": pr_ms is not None and 80 <= pr_ms <= 220,
        "qrs_plausible_60_180": qrs_ms is not None and 60 <= qrs_ms <= 180,
    })
    return row


def _reference_fpt(ecgdeli: object, r_peaks: list[int]) -> object:
    raw = np.zeros((len(r_peaks), 13), dtype=np.float64)
    raw[:, 5] = np.asarray(r_peaks, dtype=np.float64) + PAD + 1
    return ecgdeli.FPT.from_matlab(raw)


def _summarize(
    rows: list[dict[str, object]], failures: list[dict[str, object]], source: str
) -> dict[str, object]:
    take = [
        row for row in rows
        if row["source"] == source and bool(row["analysis_subset"])
    ]
    pr = np.asarray([row["pr_ms"] for row in take if row["pr_ms"] is not None], dtype=float)
    qrs = np.asarray([row["qrs_ms"] for row in take if row["qrs_ms"] is not None], dtype=float)
    return {
        "method": METHOD,
        "source": source,
        "selected_windows_with_beats": len({int(row["window_index"]) for row in take}),
        "beats": len(take),
        "failed_windows": sum(row["source"] == source and bool(row["analysis_subset"]) for row in failures),
        "p_order_valid_fraction": float(np.mean([row["p_order_valid"] for row in take])) if take else None,
        "qrs_order_valid_fraction": float(np.mean([row["qrs_order_valid"] for row in take])) if take else None,
        "pr_measurable_fraction": len(pr) / len(take) if take else None,
        "qrs_measurable_fraction": len(qrs) / len(take) if take else None,
        "pr_plausible_fraction_among_measured": float(np.mean((pr >= 80) & (pr <= 220))) if len(pr) else None,
        "qrs_plausible_fraction_among_measured": float(np.mean((qrs >= 60) & (qrs <= 180))) if len(qrs) else None,
        "pr_below_80_fraction_among_measured": float(np.mean(pr < 80)) if len(pr) else None,
        "qrs_above_180_fraction_among_measured": float(np.mean(qrs > 180)) if len(qrs) else None,
        "pr_median_ms": float(np.median(pr)) if len(pr) else None,
        "qrs_median_ms": float(np.median(qrs)) if len(qrs) else None,
    }


def _nearest_agreement(
    rows: list[dict[str, object]], comparison: list[dict[str, str]], method: str, source: str
) -> dict[str, object]:
    prior: dict[int, list[dict[str, str]]] = {}
    for row in comparison:
        if row["method"] == method and row["source"] == source:
            prior.setdefault(int(row["window_index"]), []).append(row)
    pairs: list[tuple[dict[str, object], dict[str, str]]] = []
    for row in rows:
        if row["source"] != source or not bool(row["analysis_subset"]):
            continue
        candidates = prior.get(int(row["window_index"]), [])
        if not candidates:
            continue
        match = min(candidates, key=lambda item: abs(int(item["r_peak"]) - int(row["r_peak"])))
        if abs(int(match["r_peak"]) - int(row["r_peak"])) <= 3:
            pairs.append((row, match))
    result: dict[str, object] = {
        "comparison_method": method,
        "source": source,
        "matched_beats": len(pairs),
    }
    for parameter in ("pr_ms", "qrs_ms"):
        values = [
            float(left[parameter]) - float(right[parameter])
            for left, right in pairs
            if left[parameter] is not None and right.get(parameter, "") not in (None, "")
        ]
        delta = np.asarray(values, dtype=float)
        result[f"{parameter}_matched"] = len(delta)
        result[f"{parameter}_median_port_minus_{method}"] = float(np.median(delta)) if len(delta) else None
        result[f"{parameter}_median_absolute_difference"] = float(np.median(np.abs(delta))) if len(delta) else None
    return result


def _window_parameters(
    rows: list[dict[str, object]], analysis_windows: set[int], *, quality_controlled: bool = False
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for window in sorted(analysis_windows):
        row: dict[str, object] = {"window_index": window}
        for source in ("reference", "generated"):
            take = [beat for beat in rows if int(beat["window_index"]) == window and beat["source"] == source]
            row[f"{source}_beats"] = len(take)
            for parameter in ("pr_ms", "qrs_ms", "p_amplitude"):
                usable = take
                if quality_controlled and parameter in ("pr_ms", "p_amplitude"):
                    usable = [beat for beat in take if beat["p_order_valid"] and beat["pr_plausible_80_220"]]
                elif quality_controlled and parameter == "qrs_ms":
                    usable = [beat for beat in take if beat["qrs_order_valid"] and beat["qrs_plausible_60_180"]]
                values = np.asarray([beat[parameter] for beat in usable if beat[parameter] is not None], dtype=float)
                row[f"{source}_{parameter}"] = float(np.mean(values)) if len(values) else None
        output.append(row)
    return output


def _plot_ba(
    output: Path, window_rows: list[dict[str, object]], *, stem: str
) -> dict[str, object]:
    specs = (("pr_ms", "PR (ms)"), ("qrs_ms", "QRS (ms)"), ("p_amplitude", "P amplitude (normalized)"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    summary: dict[str, object] = {}
    for axis, (parameter, label) in zip(axes, specs):
        pairs = [
            (float(row[f"reference_{parameter}"]), float(row[f"generated_{parameter}"]))
            for row in window_rows
            if row[f"reference_{parameter}"] is not None and row[f"generated_{parameter}"] is not None
        ]
        reference = np.asarray([pair[0] for pair in pairs], dtype=float)
        generated = np.asarray([pair[1] for pair in pairs], dtype=float)
        mean = (reference + generated) / 2
        difference = generated - reference
        bias = float(np.mean(difference)) if len(difference) else None
        sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else None
        if len(difference):
            axis.scatter(mean, difference, s=13, alpha=0.55, edgecolors="none")
            axis.axhline(bias, color="#d62728", linewidth=1.2)
            if sd is not None:
                axis.axhline(bias + 1.96 * sd, color="#555", linestyle="--", linewidth=1)
                axis.axhline(bias - 1.96 * sd, color="#555", linestyle="--", linewidth=1)
        axis.set_title(f"{label}; paired n={len(pairs)}")
        axis.set_xlabel(f"Mean {label}")
        axis.set_ylabel("Generated - reference")
        axis.grid(alpha=0.15)
        summary[parameter] = {"paired_windows": len(pairs), "bias": bias, "sd": sd}
    figure.savefig(output / f"{stem}.png", dpi=180)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)
    return summary


def _plot_cases(output: Path, arrays: dict[str, np.ndarray], rows: list[dict[str, object]]) -> None:
    colors = {
        "p_onset": "#9467bd", "p_peak": "#e377c2", "p_offset": "#8c564b",
        "qrs_onset": "#2ca02c", "r_peak": "#d62728", "qrs_offset": "#17becf",
    }
    figure, axes = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    for axis, (category, window, source) in zip(axes.ravel(), CASES):
        signal = arrays[source][window, 0]
        axis.plot(np.arange(len(signal)) / FS, signal, color="black", linewidth=0.8)
        take = [row for row in rows if int(row["window_index"]) == window and row["source"] == source]
        used: set[str] = set()
        for row in take:
            for key, color in colors.items():
                value = int(row[key])
                if value < 0:
                    continue
                axis.axvline(value / FS, color=color, alpha=0.65, linewidth=0.9, label=key if key not in used else None)
                used.add(key)
        axis.set_title(f"{category}: window {window}, {source}")
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.15)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, fontsize=8)
    figure.savefig(output / "selected_ecgdeli_port_overlays.png", dpi=180)
    figure.savefig(output / "selected_ecgdeli_port_overlays.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    import ecgdeli

    ecgdeli.set_fidelity(args.fidelity)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with args.previous_fiducials.open(newline="", encoding="utf-8") as handle:
        previous = list(csv.DictReader(handle))
    selected = sorted({int(row["window_index"]) for row in previous if row["method"] == "prominence"})
    cases = CASES if not args.limit else []
    if args.limit:
        selected = selected[: args.limit]
    analysis_windows = set(selected)
    execution_windows = sorted(analysis_windows | {window for _, window, _ in cases})
    with np.load(args.phase_npz, allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float64)
        generated = np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float64)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        records = np.asarray(artifact["record_ids"]).astype(str)
    if targets.shape != EXPECTED_SHAPE or generated.shape != EXPECTED_SHAPE:
        raise ValueError("phase arrays violate expected WESAD shape")
    arrays = {"reference": targets, "generated": generated}
    r_lookup: dict[tuple[int, str, str], list[int]] = {}
    for row in previous:
        if row["method"] in ("dwt", "prominence"):
            r_lookup.setdefault(
                (int(row["window_index"]), row["source"], row["method"]), []
            ).append(int(row["r_peak"]))
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for window in execution_windows:
        for source, array in arrays.items():
            try:
                signal = array[window, 0]
                padded = np.pad(signal, PAD, mode="reflect")
                filtered = ecgdeli.filter_ecg(padded, FS, notch=50)
                prominence_r = r_lookup.get((window, source, "prominence"), [])
                dwt_r = r_lookup.get((window, source, "dwt"), [])
                r_method = "prominence" if prominence_r else "dwt"
                r_peaks = sorted(set(prominence_r or dwt_r))
                if len(r_peaks) < 3:
                    raise ValueError("fewer_than_three_supplied_r_peaks")
                annotation = ecgdeli.annotate(
                    filtered,
                    FS,
                    reference=_reference_fpt(ecgdeli, r_peaks),
                )
                beats = decode_fpt_samples(annotation.leads[0].samples, pad=PAD, signal_length=len(signal))
                for beat_index, fiducials in enumerate(beats):
                    row: dict[str, object] = {
                        "window_index": window,
                        "subject_id": subjects[window],
                        "record_id": records[window],
                        "source": source,
                        "method": METHOD,
                        "fidelity": args.fidelity,
                        "supplied_r_method": r_method,
                        "analysis_subset": window in analysis_windows,
                        "beat": beat_index,
                        **fiducials,
                    }
                    rows.append(_decorate(row, signal))
                if not beats:
                    raise ValueError("no_center_beats")
            except Exception as error:
                failures.append({
                    "window_index": window,
                    "source": source,
                    "analysis_subset": window in analysis_windows,
                    "reason": f"{type(error).__name__}:{error}",
                })
    _write_csv(output / "per_beat_ecgdeli_port_fiducials.csv", rows)
    _write_csv(output / "delineation_failures.csv", failures)
    window_rows = _window_parameters(rows, analysis_windows)
    _write_csv(output / "per_window_ecgdeli_port_parameters.csv", window_rows)
    ba = _plot_ba(output, window_rows, stem="ecgdeli_port_parameter_bland_altman_raw")
    qc_window_rows = _window_parameters(rows, analysis_windows, quality_controlled=True)
    _write_csv(output / "per_window_ecgdeli_port_parameters_qc.csv", qc_window_rows)
    ba_qc = _plot_ba(
        output,
        qc_window_rows,
        stem="ecgdeli_port_parameter_bland_altman_qc",
    )
    if cases:
        _plot_cases(output, arrays, rows)

    comparison = list(previous)
    if args.ecgpuwave_fiducials:
        with args.ecgpuwave_fiducials.open(newline="", encoding="utf-8") as handle:
            comparison.extend(csv.DictReader(handle))
    methods = sorted({row["method"] for row in comparison})
    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": "WESAD",
        "sampling_rate_hz": FS,
        "selected_windows": len(selected),
        "extra_visual_windows": len(set(execution_windows) - analysis_windows),
        "implementation": {
            "identity": "unofficial community Python port of KIT-IBT ECGdeli",
            "version": ecgdeli.__version__,
            "fidelity": args.fidelity,
            "port_commit": args.port_commit,
            "upstream_matlab_commit": args.upstream_commit,
            "filter": "ECGdeli bandpass 1-40 Hz, 50-Hz notch, isoline correction",
            "r_policy": "prior NeuroKit R peaks supplied through ECGdeli reference FPT; ECGdeli re-checks QRS",
        },
        "summary": [_summarize(rows, failures, source) for source in arrays],
        "matched_method_agreement": [
            _nearest_agreement(rows, comparison, method, source)
            for method in methods for source in arrays
        ],
        "bland_altman": ba,
        "bland_altman_qc": ba_qc,
        "interpretation_boundary": (
            "Unofficial Python-port sensitivity audit, not execution of the official MATLAB toolbox "
            "and not expert fiducial ground truth."
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pywavelets_runtime": pywt.__version__,
        },
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            "phase_npz": str(args.phase_npz.resolve()),
            "phase_sha256": _sha256(args.phase_npz),
            "previous_fiducials": str(args.previous_fiducials.resolve()),
            "previous_sha256": _sha256(args.previous_fiducials),
        },
        "outputs": {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_npz", type=Path, required=True)
    parser.add_argument("--previous_fiducials", type=Path, required=True)
    parser.add_argument("--ecgpuwave_fiducials", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fidelity", choices=("matlab", "fixed"), default="matlab")
    parser.add_argument("--port_commit", default="313d6dea54d88c8a616edce820c2ab9bc2b1deec")
    parser.add_argument("--upstream_commit", default="c3738612771264e4d6c4686898ef7b2d6a700ad3")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
