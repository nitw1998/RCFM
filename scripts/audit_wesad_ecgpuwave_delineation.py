#!/usr/bin/env python3
"""Compare ecgpuwave fiducials with the existing WESAD delineation audit.

This diagnostic supplies ecgpuwave with the same NeuroKit R peaks used by the
current pipeline, isolating waveform-boundary localization from R detection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_ecgpuwave_audit_mpl")

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import wfdb


SAMPLING_RATE = 128.0
PAD_SAMPLES = 16
EXPECTED_SHAPE = (4342, 1, 480)


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


def parse_ecgpuwave_annotations(
    samples: np.ndarray,
    symbols: list[str] | np.ndarray,
    nums: np.ndarray,
    *,
    pad_samples: int,
    signal_length: int,
) -> list[dict[str, int]]:
    """Convert WFDB onset/peak/offset annotations into one row per QRS beat."""

    samples = np.asarray(samples, dtype=np.int64)
    symbols = np.asarray(symbols).astype(str)
    nums = np.asarray(nums, dtype=np.int64)
    if not (len(samples) == len(symbols) == len(nums)):
        raise ValueError("annotation arrays must have equal length")
    indices = np.arange(len(samples))
    rows: list[dict[str, int]] = []
    r_indices = np.flatnonzero(np.isin(symbols, ["N", "L", "R", "A", "V", "F", "Q"]))

    for order, r_index in enumerate(r_indices):
        r_sample = int(samples[r_index])
        if not pad_samples <= r_sample < pad_samples + signal_length:
            continue
        previous_r = int(samples[r_indices[order - 1]]) if order else -1

        def last_index(mask: np.ndarray) -> int:
            found = np.flatnonzero(mask)
            return int(found[-1]) if len(found) else -1

        def first_index(mask: np.ndarray) -> int:
            found = np.flatnonzero(mask)
            return int(found[0]) if len(found) else -1

        q_on_i = last_index((indices < r_index) & (symbols == "(") & (nums == 1))
        q_off_i = first_index((indices > r_index) & (symbols == ")") & (nums == 1))
        q_on = int(samples[q_on_i]) if q_on_i >= 0 else -1
        q_off = int(samples[q_off_i]) if q_off_i >= 0 else -1

        p_i = last_index(
            (symbols == "p")
            & (samples > previous_r)
            & (samples < (q_on if q_on >= 0 else r_sample))
        )
        p_peak = int(samples[p_i]) if p_i >= 0 else -1
        p_on_i = last_index(
            (symbols == "(") & (nums == 0) & (samples > previous_r) & (samples < p_peak)
        ) if p_peak >= 0 else -1
        p_off_i = first_index(
            (symbols == ")") & (nums == 0) & (samples > p_peak)
            & (samples < (q_on if q_on >= 0 else r_sample))
        ) if p_peak >= 0 else -1

        next_r = int(samples[r_indices[order + 1]]) if order + 1 < len(r_indices) else 10**18
        t_i = first_index(
            (symbols == "t") & (samples > q_off) & (samples < next_r)
        ) if q_off >= 0 else -1
        t_peak = int(samples[t_i]) if t_i >= 0 else -1
        t_off_i = first_index(
            (symbols == ")") & (nums == 2) & (samples > t_peak) & (samples < next_r)
        ) if t_peak >= 0 else -1

        raw = {
            "p_onset": int(samples[p_on_i]) if p_on_i >= 0 else -1,
            "p_peak": p_peak,
            "p_offset": int(samples[p_off_i]) if p_off_i >= 0 else -1,
            "qrs_onset": q_on,
            "r_peak": r_sample,
            "qrs_offset": q_off,
            "t_peak": t_peak,
            "t_offset": int(samples[t_off_i]) if t_off_i >= 0 else -1,
        }
        rows.append({key: value - pad_samples if value >= 0 else -1 for key, value in raw.items()})
    return rows


def _delineate_one(
    signal: np.ndarray,
    ecgpuwave: Path,
    temporary_root: Path,
    record_name: str,
) -> tuple[list[dict[str, int]], str]:
    padded = np.pad(np.asarray(signal, dtype=np.float64), PAD_SAMPLES, mode="reflect")
    cleaned = nk.ecg_clean(padded, sampling_rate=SAMPLING_RATE, method="neurokit")
    _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=SAMPLING_RATE, method="neurokit")
    r_peaks = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=np.int64)
    if len(r_peaks) < 2:
        raise ValueError("fewer_than_two_r_peaks")

    wfdb.wrsamp(
        record_name,
        fs=SAMPLING_RATE,
        units=["normalized"],
        sig_name=["ECG"],
        p_signal=cleaned[:, None],
        write_dir=str(temporary_root),
    )
    wfdb.wrann(
        record_name,
        "rpk",
        sample=r_peaks,
        symbol=["N"] * len(r_peaks),
        write_dir=str(temporary_root),
    )
    environment = dict(os.environ)
    environment["WFDB"] = "."
    completed = subprocess.run(
        [str(ecgpuwave), "-r", record_name, "-i", "rpk", "-a", "epu"],
        cwd=temporary_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"ecgpuwave_exit_{completed.returncode}:{completed.stderr.strip()}")
    annotation_path = temporary_root / f"{record_name}.epu"
    if not annotation_path.exists():
        raise RuntimeError("ecgpuwave_output_annotation_missing")
    annotation = wfdb.rdann(str(temporary_root / record_name), "epu")
    rows = parse_ecgpuwave_annotations(
        annotation.sample,
        annotation.symbol,
        annotation.num,
        pad_samples=PAD_SAMPLES,
        signal_length=len(signal),
    )
    return rows, (completed.stdout + completed.stderr).strip()


def _decorate(row: dict[str, object], signal: np.ndarray) -> dict[str, object]:
    p_on, p_peak, p_off = (int(row[key]) for key in ("p_onset", "p_peak", "p_offset"))
    q_on, r_peak, q_off = (int(row[key]) for key in ("qrs_onset", "r_peak", "qrs_offset"))
    p_valid = 0 <= p_on < p_peak < p_off < q_on < r_peak
    q_valid = 0 <= q_on < r_peak < q_off
    pr_ms = (q_on - p_on) * 1000.0 / SAMPLING_RATE if p_on >= 0 and q_on > p_on else None
    qrs_ms = (q_off - q_on) * 1000.0 / SAMPLING_RATE if q_on >= 0 and q_off > q_on else None
    p_amplitude = None
    if 0 <= p_peak < len(signal) and q_on > 0:
        baseline_start = p_off if 0 <= p_off < q_on else max(0, q_on - round(0.04 * SAMPLING_RATE))
        if baseline_start < q_on:
            p_amplitude = float(signal[p_peak] - np.median(signal[baseline_start:q_on]))
    row.update({
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "p_amplitude": p_amplitude,
        "p_order_valid": p_valid,
        "qrs_order_valid": q_valid,
        "pr_plausible_80_220": pr_ms is not None and 80.0 <= pr_ms <= 220.0,
        "qrs_plausible_60_180": qrs_ms is not None and 60.0 <= qrs_ms <= 180.0,
    })
    return row


def _window_parameters(
    rows: list[dict[str, object]],
    analysis_windows: set[int],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for window in sorted(analysis_windows):
        row: dict[str, object] = {"window_index": window}
        for source in ("reference", "generated"):
            take = [
                beat for beat in rows
                if int(beat["window_index"]) == window and beat["source"] == source
            ]
            row[f"{source}_beats"] = len(take)
            for parameter in ("pr_ms", "qrs_ms", "p_amplitude"):
                values = np.asarray(
                    [beat[parameter] for beat in take if beat[parameter] is not None],
                    dtype=float,
                )
                row[f"{source}_{parameter}"] = float(np.mean(values)) if len(values) else None
        output.append(row)
    return output


def _plot_bland_altman(output: Path, window_rows: list[dict[str, object]]) -> None:
    specs = (("pr_ms", "PR (ms)"), ("qrs_ms", "QRS (ms)"), ("p_amplitude", "P amplitude (normalized)"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for axis, (parameter, label) in zip(axes, specs):
        pairs = [
            (float(row[f"reference_{parameter}"]), float(row[f"generated_{parameter}"]))
            for row in window_rows
            if row[f"reference_{parameter}"] is not None and row[f"generated_{parameter}"] is not None
        ]
        reference = np.asarray([pair[0] for pair in pairs], dtype=float)
        generated = np.asarray([pair[1] for pair in pairs], dtype=float)
        mean = (reference + generated) / 2.0
        difference = generated - reference
        bias = float(np.mean(difference))
        sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
        axis.scatter(mean, difference, s=13, alpha=0.55, edgecolors="none")
        axis.axhline(bias, color="#d62728", linewidth=1.2, label="bias")
        axis.axhline(bias + 1.96 * sd, color="#555555", linestyle="--", linewidth=1.0, label="95% limits")
        axis.axhline(bias - 1.96 * sd, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_title(f"{label}; paired n={len(pairs)}")
        axis.set_xlabel(f"Mean {label}")
        axis.set_ylabel("Generated - reference")
        axis.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    figure.savefig(output / "ecgpuwave_parameter_bland_altman.png", dpi=180)
    figure.savefig(output / "ecgpuwave_parameter_bland_altman.pdf")
    plt.close(figure)


def _summarize(
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
    source: str,
) -> dict[str, object]:
    take = [
        row for row in rows
        if row["source"] == source and bool(row["analysis_subset"])
    ]
    pr = np.asarray([row["pr_ms"] for row in take if row["pr_ms"] is not None], dtype=float)
    qrs = np.asarray([row["qrs_ms"] for row in take if row["qrs_ms"] is not None], dtype=float)
    windows = {int(row["window_index"]) for row in take}
    return {
        "method": "ecgpuwave_1.3.4_supplied_neurokit_r_peaks",
        "source": source,
        "selected_windows_with_beats": len(windows),
        "beats": len(take),
        "failed_windows": sum(row["source"] == source for row in failures),
        "p_order_valid_fraction": float(np.mean([row["p_order_valid"] for row in take])) if take else None,
        "qrs_order_valid_fraction": float(np.mean([row["qrs_order_valid"] for row in take])) if take else None,
        "pr_measurable_fraction": len(pr) / len(take) if take else None,
        "qrs_measurable_fraction": len(qrs) / len(take) if take else None,
        "pr_plausible_fraction_among_measured": float(np.mean((pr >= 80) & (pr <= 220))) if len(pr) else None,
        "qrs_plausible_fraction_among_measured": float(np.mean((qrs >= 60) & (qrs <= 180))) if len(qrs) else None,
        "pr_median_ms": float(np.median(pr)) if len(pr) else None,
        "qrs_median_ms": float(np.median(qrs)) if len(qrs) else None,
    }


def _method_agreement(
    ecgpuwave_rows: list[dict[str, object]],
    previous_rows: list[dict[str, str]],
    method: str,
    source: str,
) -> dict[str, object]:
    prior = {
        (int(row["window_index"]), int(row["r_peak"])): row
        for row in previous_rows
        if row["method"] == method and row["source"] == source
    }
    pairs = [
        (row, prior.get((int(row["window_index"]), int(row["r_peak"]))))
        for row in ecgpuwave_rows
        if row["source"] == source and bool(row["analysis_subset"])
    ]
    pairs = [(left, right) for left, right in pairs if right is not None]
    result: dict[str, object] = {
        "comparison_method": method,
        "source": source,
        "matched_beats": len(pairs),
    }
    for parameter in ("pr_ms", "qrs_ms"):
        values = [
            (float(left[parameter]), float(right[parameter]))
            for left, right in pairs
            if left[parameter] is not None and right[parameter] not in (None, "")
        ]
        delta = np.asarray([left - right for left, right in values], dtype=float)
        result[f"{parameter}_matched"] = len(values)
        result[f"{parameter}_median_ecgpuwave_minus_{method}"] = float(np.median(delta)) if len(delta) else None
        result[f"{parameter}_median_absolute_difference"] = float(np.median(np.abs(delta))) if len(delta) else None
    return result


def _plot_cases(
    output: Path,
    arrays: dict[str, np.ndarray],
    rows: list[dict[str, object]],
    cases: list[tuple[str, int, str]],
) -> None:
    colors = {
        "p_onset": "#9467bd", "p_peak": "#e377c2", "p_offset": "#8c564b",
        "qrs_onset": "#2ca02c", "r_peak": "#d62728", "qrs_offset": "#17becf",
    }
    figure, axes = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    for axis, (category, window, source) in zip(axes.ravel(), cases):
        signal = arrays[source][window, 0]
        axis.plot(np.arange(len(signal)) / SAMPLING_RATE, signal, color="black", linewidth=0.8)
        take = [row for row in rows if int(row["window_index"]) == window and row["source"] == source]
        used: set[str] = set()
        for row in take:
            for key, color in colors.items():
                value = int(row[key])
                if value < 0:
                    continue
                axis.axvline(
                    value / SAMPLING_RATE,
                    color=color,
                    alpha=0.65,
                    linewidth=0.9,
                    label=key if key not in used else None,
                )
                used.add(key)
        axis.set_title(f"{category}: window {window}, {source}")
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.15)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=6, fontsize=8)
    figure.savefig(output / "selected_ecgpuwave_overlays.png", dpi=180)
    figure.savefig(output / "selected_ecgpuwave_overlays.pdf")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    phase_path = args.phase_npz.resolve()
    prior_path = args.previous_fiducials.resolve()
    ecgpuwave = args.ecgpuwave.resolve()
    output = args.output_dir.resolve()
    if not ecgpuwave.is_file() or not os.access(ecgpuwave, os.X_OK):
        raise FileNotFoundError(f"ecgpuwave is not executable: {ecgpuwave}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with prior_path.open(newline="", encoding="utf-8") as handle:
        previous_rows = list(csv.DictReader(handle))
    selected = sorted({
        int(row["window_index"])
        for row in previous_rows
        if row["method"] == "prominence"
    })
    cases = [
        ("reference QRS high", 3104, "reference"),
        ("generated QRS high", 804, "generated"),
        ("reference PR low", 1280, "reference"),
        ("generated PR low", 2409, "generated"),
        ("large P error", 1682, "reference"),
        ("control", 4, "reference"),
        ("control", 166, "reference"),
        ("control", 340, "reference"),
    ]
    if args.limit:
        selected = selected[: args.limit]
        cases = []
    analysis_windows = set(selected)
    execution_windows = sorted(analysis_windows | {window for _, window, _ in cases})
    with np.load(phase_path, allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float64)
        generated = np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float64)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        records = np.asarray(artifact["record_ids"]).astype(str)
    if targets.shape != EXPECTED_SHAPE or generated.shape != EXPECTED_SHAPE:
        raise ValueError("phase arrays violate the expected WESAD shape")

    arrays = {"reference": targets, "generated": generated}
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="rcfm_ecgpuwave_") as temporary:
        temporary_root = Path(temporary)
        for window in execution_windows:
            for source, values in arrays.items():
                name = f"w{window:04d}_{source[0]}"
                try:
                    beats, message = _delineate_one(values[window, 0], ecgpuwave, temporary_root, name)
                    for beat, values_by_field in enumerate(beats):
                        row: dict[str, object] = {
                            "window_index": window,
                            "subject_id": subjects[window],
                            "record_id": records[window],
                            "source": source,
                            "method": "ecgpuwave_1.3.4_supplied_neurokit_r_peaks",
                            "analysis_subset": window in analysis_windows,
                            "beat": beat,
                            **values_by_field,
                        }
                        rows.append(_decorate(row, values[window, 0]))
                    if not beats:
                        failures.append({
                            "window_index": window,
                            "source": source,
                            "reason": "no_center_beats",
                            "message": message,
                        })
                except Exception as error:
                    failures.append({
                        "window_index": window,
                        "source": source,
                        "reason": f"{type(error).__name__}:{error}",
                    })
                for suffix in ("dat", "hea", "rpk", "epu"):
                    path = temporary_root / f"{name}.{suffix}"
                    if path.exists():
                        path.unlink()

    _write_csv(output / "per_beat_ecgpuwave_fiducials.csv", rows)
    _write_csv(output / "delineation_failures.csv", failures)
    window_rows = _window_parameters(rows, analysis_windows)
    _write_csv(output / "per_window_ecgpuwave_parameters.csv", window_rows)
    _plot_bland_altman(output, window_rows)
    summaries = [_summarize(rows, failures, source) for source in arrays]
    agreement = [
        _method_agreement(rows, previous_rows, method, source)
        for method in ("dwt", "prominence")
        for source in arrays
    ]
    if cases:
        _plot_cases(output, arrays, rows, cases)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": "WESAD",
        "sampling_rate_hz": SAMPLING_RATE,
        "selected_windows": len(selected),
        "extra_visual_windows": len(set(execution_windows) - analysis_windows),
        "selection": "same windows as prior prominence sensitivity audit",
        "ecgpuwave": {
            "version": "1.3.4",
            "executable": str(ecgpuwave),
            "executable_sha256": _sha256(ecgpuwave),
            "qrs_policy": "supplied NeuroKit2 neurokit-method R peaks via -i rpk",
            "input_signal": "NeuroKit-cleaned, reflect-padded 16 samples per side",
            "official_check": "test_2_passed_with_boundaries_within_one_sample",
        },
        "summary": summaries,
        "matched_method_agreement": agreement,
        "interpretation_boundary": (
            "Independent traditional-algorithm sensitivity audit; ecgpuwave is not expert ground truth, "
            "and normalized 128-Hz WESAD windows remain a domain limitation."
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "neurokit2": nk.__version__,
            "wfdb_python": wfdb.__version__,
        },
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            "phase_npz": str(phase_path),
            "phase_sha256": _sha256(phase_path),
            "previous_fiducials": str(prior_path),
            "previous_sha256": _sha256(prior_path),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_npz", type=Path, required=True)
    parser.add_argument("--previous_fiducials", type=Path, required=True)
    parser.add_argument("--ecgpuwave", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="deterministic smoke-test window limit")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
