"""Compare ECG parameters before and after oracle phase correction on MIMIC-AFib."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.clinical import ECGFiducials, delineate_ecg, measure_ecg_parameters


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
PHASE_MODES = ("unshifted", "oracle_aligned")
PARAMETERS = (
    "heart_rate_bpm",
    "rr_ms",
    "pr_ms",
    "qrs_ms",
    "qt_ms",
    "qtc_ms",
    "p_amplitude",
    "r_amplitude",
    "t_amplitude",
    "st_deviation",
)
INTERVAL_PARAMETERS = {
    "heart_rate_bpm": "bpm",
    "rr_ms": "ms",
    "pr_ms": "ms",
    "qrs_ms": "ms",
    "qt_ms": "ms",
    "qtc_ms": "ms",
}
AMPLITUDE_PARAMETERS = {
    "p_amplitude",
    "r_amplitude",
    "t_amplitude",
    "st_deviation",
}
DISPLAY_NAMES = {
    "heart_rate_bpm": "Heart rate",
    "rr_ms": "RR interval",
    "pr_ms": "PR interval",
    "qrs_ms": "QRS duration",
    "qt_ms": "QT interval",
    "qtc_ms": "QTc interval",
    "p_amplitude": "P amplitude",
    "r_amplitude": "R amplitude",
    "t_amplitude": "T amplitude",
    "st_deviation": "ST deviation",
}


def _record_measurement(signal: np.ndarray, sampling_rate: float) -> dict[str, object]:
    minimum_samples = int(np.ceil(4.0 * sampling_rate))
    padding = max(0, minimum_samples - len(signal))
    left_padding = padding // 2
    right_padding = padding - left_padding
    padded = np.pad(signal, (left_padding, right_padding), mode="reflect") if padding else signal
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Too few peaks detected.*")
        delineation = delineate_ecg(padded, sampling_rate=sampling_rate, method="dwt")
    if not delineation.success:
        return {
            "success": False,
            "failure_reason": delineation.failure_reason,
            "summary": {},
            "p_wave_status": "not_measured",
            "hrv_status": "not_measured",
        }
    shifted: dict[str, np.ndarray] = {}
    centered_r_peaks = delineation.fiducials.r_peaks - left_padding
    centered_beats = (centered_r_peaks >= 0) & (centered_r_peaks < len(signal))
    for name in delineation.fiducials.__dataclass_fields__:
        values = np.asarray(getattr(delineation.fiducials, name), dtype=np.int64)
        values = values[centered_beats] - left_padding
        values[(values < 0) | (values >= len(signal))] = -1
        shifted[name] = values
    centered_fiducials = ECGFiducials(**shifted)
    measurement = measure_ecg_parameters(
        signal,
        sampling_rate=sampling_rate,
        fiducials=centered_fiducials,
        amplitude_unit="normalized",
        inverse_transformed=False,
        qtc_formula="fridericia",
        st_offset_ms=60.0,
        continuous=False,
        p_wave_applicable=True,
        allow_normalized_amplitudes=True,
    )
    summary = {
        name: float(np.mean(values))
        for name, values in measurement["parameters"].items()
        if values
    }
    if "rr_ms" in summary and summary["rr_ms"] > 0:
        summary["heart_rate_bpm"] = 60000.0 / summary["rr_ms"]
    return {
        "success": True,
        "failure_reason": None,
        "summary": summary,
        "p_wave_status": measurement["p_wave_status"],
        "hrv_status": measurement["hrv"]["status"],
    }


def _measure_task(task: tuple[np.ndarray, float]) -> dict[str, object]:
    signal, sampling_rate = task
    return _record_measurement(signal, sampling_rate)


def _measure_records(
    signals: np.ndarray,
    sampling_rate: float,
    executor: ProcessPoolExecutor | None,
) -> list[dict[str, object]]:
    tasks = ((signal, sampling_rate) for signal in signals[:, 0])
    if executor is None:
        return [_measure_task(task) for task in tasks]
    return list(executor.map(_measure_task, tasks, chunksize=16))


def _parameter_pairs(
    reference: list[dict[str, object]],
    generated: list[dict[str, object]],
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    real_values: list[float] = []
    generated_values: list[float] = []
    for index, (real, prediction) in enumerate(zip(reference, generated)):
        real_value = real["summary"].get(parameter)
        generated_value = prediction["summary"].get(parameter)
        if real_value is None or generated_value is None:
            continue
        if not np.isfinite(real_value) or not np.isfinite(generated_value):
            continue
        rows.append(index)
        real_values.append(float(real_value))
        generated_values.append(float(generated_value))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(real_values, dtype=np.float64),
        np.asarray(generated_values, dtype=np.float64),
    )


def _parameter_triplets(
    reference: list[dict[str, object]],
    unshifted: list[dict[str, object]],
    aligned: list[dict[str, object]],
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[int] = []
    real_values: list[float] = []
    unshifted_values: list[float] = []
    aligned_values: list[float] = []
    for index, (real, before, after) in enumerate(zip(reference, unshifted, aligned)):
        values = (
            real["summary"].get(parameter),
            before["summary"].get(parameter),
            after["summary"].get(parameter),
        )
        if any(value is None or not np.isfinite(value) for value in values):
            continue
        rows.append(index)
        real_values.append(float(values[0]))
        unshifted_values.append(float(values[1]))
        aligned_values.append(float(values[2]))
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(real_values, dtype=np.float64),
        np.asarray(unshifted_values, dtype=np.float64),
        np.asarray(aligned_values, dtype=np.float64),
    )


def _agreement_record(
    model: str,
    phase_mode: str,
    parameter: str,
    reference: np.ndarray,
    generated: np.ndarray,
) -> dict[str, object]:
    unit = INTERVAL_PARAMETERS.get(parameter, "normalized")
    base = {
        "model": model,
        "phase_mode": phase_mode,
        "parameter": parameter,
        "unit": unit,
        "physical_amplitude_claim_allowed": parameter not in AMPLITUDE_PARAMETERS,
        "n": int(reference.size),
        "inference_status": "descriptive_only_no_subject_ids",
    }
    if reference.size < 2:
        return {**base, "status": "insufficient_pairs"}
    correlation = paired_correlation(reference, generated)
    agreement = bland_altman(reference, generated)
    error = generated - reference
    return {
        **base,
        "status": "ok",
        "reference_mean": float(np.mean(reference)),
        "reference_sd": float(np.std(reference, ddof=1)),
        "generated_mean": float(np.mean(generated)),
        "generated_sd": float(np.std(generated, ddof=1)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": correlation["r"],
        "bland_altman_bias": agreement["bias"],
        "bland_altman_difference_sd": agreement["difference_sd"],
        "bland_altman_lower_limit": agreement["lower_limit"],
        "bland_altman_upper_limit": agreement["upper_limit"],
    }


def _waveform_summary(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    real = targets[:, 0].astype(np.float64)
    generated = predictions[:, 0].astype(np.float64)
    real_centered = real - np.mean(real, axis=1, keepdims=True)
    generated_centered = generated - np.mean(generated, axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(real_centered**2, axis=1) * np.sum(generated_centered**2, axis=1)
    )
    pearson = np.full(len(real), np.nan, dtype=np.float64)
    valid = denominator > 0
    pearson[valid] = np.sum(real_centered[valid] * generated_centered[valid], axis=1) / denominator[valid]
    agreement = bland_altman(real.reshape(-1), generated.reshape(-1))
    return {
        "per_record_pearson_mean": float(np.nanmean(pearson)),
        "per_record_pearson_median": float(np.nanmedian(pearson)),
        "per_record_pearson_usable": int(np.sum(np.isfinite(pearson))),
        "pointwise_bland_altman": {
            key: agreement[key]
            for key in (
                "n",
                "difference_definition",
                "bias",
                "difference_sd",
                "lower_limit",
                "upper_limit",
                "limits_z",
            )
        },
        "pointwise_inference_status": "descriptive_only_autocorrelated_samples",
        "pearson_per_record": pearson,
    }


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    keys = (
        "model", "phase_mode", "parameter", "unit", "status", "n",
        "reference_mean", "reference_sd", "generated_mean", "generated_sd",
        "mae", "rmse", "pearson_r", "bland_altman_bias",
        "bland_altman_difference_sd", "bland_altman_lower_limit",
        "bland_altman_upper_limit", "physical_amplitude_claim_allowed",
        "inference_status",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _write_per_record(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    keys = [
        "row", "model", "phase_mode", "real_delineation_success",
        "generated_delineation_success", "real_failure_reason", "generated_failure_reason",
        "real_p_wave_status", "generated_p_wave_status", "real_hrv_status",
        "generated_hrv_status", "waveform_pearson_r",
    ]
    for parameter in PARAMETERS:
        keys.extend((f"real_{parameter}", f"generated_{parameter}"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in keys})


def _plot_bland_altman(
    output_dir: Path,
    model: str,
    pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> list[Path]:
    colors = {"unshifted": "#6f7782", "oracle_aligned": "#c43d4b"}
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.5), constrained_layout=True)
    for axis, parameter in zip(axes.flat, PARAMETERS):
        for phase_mode in PHASE_MODES:
            reference, generated = pairs[(phase_mode, parameter)]
            if len(reference) < 2:
                continue
            means = (reference + generated) / 2.0
            differences = generated - reference
            agreement = bland_altman(reference, generated)
            color = colors[phase_mode]
            label = "Unshifted" if phase_mode == "unshifted" else "Oracle aligned"
            axis.scatter(means, differences, s=6, alpha=0.18, color=color, rasterized=True)
            axis.axhline(agreement["bias"], color=color, linewidth=1.3, label=label)
            axis.axhline(agreement["lower_limit"], color=color, linewidth=0.8, linestyle="--")
            axis.axhline(agreement["upper_limit"], color=color, linewidth=0.8, linestyle="--")
        unit = INTERVAL_PARAMETERS.get(parameter, "normalized")
        axis.set_title(DISPLAY_NAMES[parameter])
        axis.set_xlabel(f"Pair mean ({unit})")
        axis.set_ylabel(f"Generated - real ({unit})")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(handles, labels, frameon=False, fontsize=8)
    figure.suptitle(f"{model.upper()} ECG-parameter Bland-Altman: before vs oracle phase correction")
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{model}_ecg_parameter_bland_altman.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def run(args: argparse.Namespace) -> Path:
    if args.max_lag_samples != 16:
        raise ValueError("the frozen narrow phase-clinical protocol requires max_lag_samples=16")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.phase_predictions, allow_pickle=False) as artifact:
        required = {"targets"}
        for model in MODELS:
            required.update(
                {f"{model}_unshifted_predictions", f"{model}_oracle_aligned_predictions"}
            )
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"phase artifact is missing arrays: {sorted(missing)}")
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        predictions = {
            (model, phase_mode): np.asarray(
                artifact[f"{model}_{phase_mode}_predictions"], dtype=np.float32
            )
            for model in MODELS
            for phase_mode in PHASE_MODES
        }
    expected_shape = (1800, 1, 480)
    if targets.shape != expected_shape or any(value.shape != expected_shape for value in predictions.values()):
        raise ValueError("the frozen phase-clinical arrays must all have shape (1800,1,480)")
    if not np.all(np.isfinite(targets)) or any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise FloatingPointError("phase-clinical arrays must be finite")

    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        real_measurements = _measure_records(targets, args.sampling_rate, executor)
        generated_measurements = {
            key: _measure_records(values, args.sampling_rate, executor)
            for key, values in predictions.items()
        }
    finally:
        if executor is not None:
            executor.shutdown()

    summary_rows: list[dict[str, object]] = []
    per_record_rows: list[dict[str, object]] = []
    waveform_results: dict[str, dict[str, object]] = {}
    delineation_results: dict[str, dict[str, object]] = {}
    plot_paths: list[Path] = []
    for model in MODELS:
        plot_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        waveform_results[model] = {}
        delineation_results[model] = {}
        for phase_mode in PHASE_MODES:
            measured = generated_measurements[(model, phase_mode)]
            waveform = _waveform_summary(targets, predictions[(model, phase_mode)])
            pearson_values = waveform.pop("pearson_per_record")
            waveform_results[model][phase_mode] = waveform
            real_success = sum(record["success"] for record in real_measurements)
            generated_success = sum(record["success"] for record in measured)
            paired_success = sum(
                real["success"] and generated["success"]
                for real, generated in zip(real_measurements, measured)
            )
            delineation_results[model][phase_mode] = {
                "real_success": int(real_success),
                "generated_success": int(generated_success),
                "paired_success": int(paired_success),
                "total": len(targets),
            }
            for index, (real, generated) in enumerate(zip(real_measurements, measured)):
                row: dict[str, object] = {
                    "row": index,
                    "model": model,
                    "phase_mode": phase_mode,
                    "real_delineation_success": real["success"],
                    "generated_delineation_success": generated["success"],
                    "real_failure_reason": real["failure_reason"],
                    "generated_failure_reason": generated["failure_reason"],
                    "real_p_wave_status": real["p_wave_status"],
                    "generated_p_wave_status": generated["p_wave_status"],
                    "real_hrv_status": real["hrv_status"],
                    "generated_hrv_status": generated["hrv_status"],
                    "waveform_pearson_r": pearson_values[index],
                }
                for parameter in PARAMETERS:
                    row[f"real_{parameter}"] = real["summary"].get(parameter)
                    row[f"generated_{parameter}"] = generated["summary"].get(parameter)
                per_record_rows.append(row)
        for parameter in PARAMETERS:
            _, reference, unshifted_values, aligned_values = _parameter_triplets(
                real_measurements,
                generated_measurements[(model, "unshifted")],
                generated_measurements[(model, "oracle_aligned")],
                parameter,
            )
            for phase_mode, generated_values in (
                ("unshifted", unshifted_values),
                ("oracle_aligned", aligned_values),
            ):
                summary_rows.append(
                    _agreement_record(
                        model, phase_mode, parameter, reference, generated_values
                    )
                )
                plot_pairs[(phase_mode, parameter)] = (reference, generated_values)
        plot_paths.extend(_plot_bland_altman(output_dir, model, plot_pairs))

    summary_path = output_dir / "ecg_parameter_agreement_summary.csv"
    per_record_path = output_dir / "per_record_ecg_parameters.csv"
    _write_summary(summary_path, summary_rows)
    _write_per_record(per_record_path, per_record_rows)
    summary_json_path = output_dir / "ecg_phase_clinical_summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                    "schema_version": 1,
                    "protocol": {
                        "records": 1800,
                        "sampling_rate_hz": args.sampling_rate,
                        "support_samples": 480,
                        "support_seconds": 480 / args.sampling_rate,
                        "max_lag_samples": 16,
                        "max_lag_ms": 125.0,
                        "phase_correction": "target-informed per-window oracle Pearson maximization",
                        "analysis_unit": "record_level_descriptive_only",
                        "parameter_pairing": (
                            "per-model/per-parameter intersection requiring real, unshifted, "
                            "and oracle-aligned measurements"
                        ),
                        "delineation_padding": (
                            "reflect 16 samples on each side to 512 samples for NeuroKit DWT only; "
                            "fiducials mapped to 480-sample center and padding-region indices discarded"
                        ),
                        "qtc_formula": "fridericia",
                        "st_offset_ms": 60.0,
                        "amplitude_unit": "normalized",
                        "amplitude_claim_allowed": False,
                        "hrv_status": "blocked_non_continuous_and_insufficient_duration",
                        "p_wave_policy": "exploratory_algorithmic_delineation_without_certified_AF_labels",
                    },
                    "waveform_agreement": waveform_results,
                    "delineation": delineation_results,
                    "parameter_agreement": summary_rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    outputs = [summary_path, per_record_path, summary_json_path, *plot_paths]
    protocol_path = output_dir / "protocol.json"
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {
            "path": str(args.phase_predictions.resolve()),
            "sha256": _sha256(args.phase_predictions),
        },
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"),
            "workers": args.workers,
            "script_sha256": _sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Oracle target-informed phase correction is diagnostic only. Four-second normalized "
            "windows have no physical amplitude inverse, subject IDs, continuity, or certified AF labels."
        ),
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"MIMIC phase-clinical evaluation saved to {result}")
