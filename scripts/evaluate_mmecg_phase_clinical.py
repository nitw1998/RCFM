"""Evaluate mmECG waveform and ECG-parameter agreement before/after oracle shifting."""

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
from scripts.evaluate_mimic_phase_clinical import (
    AMPLITUDE_PARAMETERS,
    DISPLAY_NAMES,
    INTERVAL_PARAMETERS,
    PARAMETERS,
    PHASE_MODES,
    _agreement_record,
    _parameter_triplets,
    _waveform_summary,
)
from src.rcfm.metrics.bland_altman import bland_altman
from src.rcfm.metrics.clinical import ECGFiducials, delineate_ecg, measure_ecg_parameters


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
MODEL_LABELS = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM"}
MODEL_COLORS = {"cfm": "#2878b5", "rcfm": "#2f8f5b", "rcfm_ot": "#c43d4b", "rddm": "#d17a00"}


def _configure_ieee_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
            "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
            "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        }
    )


def _record_measurement(signal: np.ndarray, sampling_rate: float) -> dict[str, object]:
    minimum_samples = int(np.ceil(8.0 * sampling_rate))
    padding = max(0, minimum_samples - len(signal))
    left_padding = padding // 2
    padded = np.pad(signal, (left_padding, padding - left_padding), mode="reflect")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Too few peaks detected.*")
        delineation = delineate_ecg(padded, sampling_rate=sampling_rate, method="dwt")
    if not delineation.success:
        return {
            "success": False, "failure_reason": delineation.failure_reason, "summary": {},
            "p_wave_status": "not_measured", "hrv_status": "not_measured",
        }
    centered_r = delineation.fiducials.r_peaks - left_padding
    keep = (centered_r >= 0) & (centered_r < len(signal))
    if int(np.sum(keep)) < 2:
        return {
            "success": False, "failure_reason": "fewer_than_two_center_r_peaks", "summary": {},
            "p_wave_status": "not_measured", "hrv_status": "not_measured",
        }
    shifted = {}
    for name in delineation.fiducials.__dataclass_fields__:
        values = np.asarray(getattr(delineation.fiducials, name), dtype=np.int64)[keep] - left_padding
        values[(values < 0) | (values >= len(signal))] = -1
        shifted[name] = values
    measurement = measure_ecg_parameters(
        signal, sampling_rate, ECGFiducials(**shifted), amplitude_unit="normalized",
        inverse_transformed=False, qtc_formula="fridericia", st_offset_ms=60.0,
        continuous=False, p_wave_applicable=True, allow_normalized_amplitudes=True,
    )
    summary = {
        name: float(np.mean(values))
        for name, values in measurement["parameters"].items()
        if values
    }
    if summary.get("rr_ms", 0) > 0:
        summary["heart_rate_bpm"] = 60000.0 / summary["rr_ms"]
    return {
        "success": True, "failure_reason": None, "summary": summary,
        "p_wave_status": measurement["p_wave_status"], "hrv_status": measurement["hrv"]["status"],
    }


def _measure_task(task: tuple[np.ndarray, float]) -> dict[str, object]:
    return _record_measurement(*task)


def _measure_records(
    signals: np.ndarray, sampling_rate: float, executor: ProcessPoolExecutor | None
) -> list[dict[str, object]]:
    tasks = ((signal, sampling_rate) for signal in signals[:, 0])
    if executor is None:
        return [_measure_task(task) for task in tasks]
    return list(executor.map(_measure_task, tasks, chunksize=16))


def _subject_agreement(
    subjects: np.ndarray,
    paired_rows: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    model: str,
    phase_mode: str,
    parameter: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    detail = []
    for subject in sorted(set(subjects)):
        take = np.asarray([index for index, row in enumerate(paired_rows) if subjects[row] == subject])
        if len(take):
            real_mean = float(np.mean(reference[take]))
            generated_mean = float(np.mean(generated[take]))
            detail.append(
                {
                    "model": model, "phase_mode": phase_mode, "parameter": parameter,
                    "subject_id": subject, "real_mean": real_mean,
                    "generated_mean": generated_mean, "difference": generated_mean - real_mean,
                    "paired_windows": len(take),
                }
            )
    if len(detail) < 2:
        return {
            "model": model, "phase_mode": phase_mode, "parameter": parameter,
            "status": "insufficient_subjects", "n_subjects": len(detail),
        }, detail
    real = np.asarray([row["real_mean"] for row in detail])
    generated_values = np.asarray([row["generated_mean"] for row in detail])
    result = _agreement_record(model, phase_mode, parameter, real, generated_values)
    result.update(
        {
            "n_subjects": len(detail), "n_windows_contributing": int(sum(row["paired_windows"] for row in detail)),
            "inference_status": "descriptive_only_extremely_underpowered_n3_subjects",
        }
    )
    return result, detail


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_parameter_bland_altman(
    output: Path,
    model: str,
    pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> list[Path]:
    colors = {"unshifted": "#6f7782", "oracle_aligned": "#c43d4b"}
    figure, axes = plt.subplots(2, 5, figsize=(7.16, 3.7), constrained_layout=True)
    for axis, parameter in zip(axes.flat, PARAMETERS):
        for phase_mode in PHASE_MODES:
            reference, generated = pairs[(phase_mode, parameter)]
            if len(reference) < 2:
                continue
            means = (reference + generated) / 2
            differences = generated - reference
            agreement = bland_altman(reference, generated)
            color = colors[phase_mode]
            axis.scatter(means, differences, s=2, alpha=0.12, color=color, rasterized=True)
            axis.axhline(agreement["bias"], color=color, linewidth=0.8,
                         label="Raw" if phase_mode == "unshifted" else "Oracle")
            axis.axhline(agreement["lower_limit"], color=color, linewidth=0.45, linestyle="--")
            axis.axhline(agreement["upper_limit"], color=color, linewidth=0.45, linestyle="--")
        unit = INTERVAL_PARAMETERS.get(parameter, "normalized")
        axis.set_title(DISPLAY_NAMES[parameter])
        axis.set_xlabel(f"Pair mean ({unit})")
        axis.set_ylabel(f"Generated - real ({unit})")
        axis.grid(alpha=0.15, linewidth=0.4)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"{MODEL_LABELS[model]} mmECG parameter agreement", fontsize=8)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{model}_parameter_bland_altman.{suffix}"
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def _plot_waveforms(
    raw_path: Path,
    phase_path: Path,
    output: Path,
) -> list[Path]:
    with np.load(raw_path, allow_pickle=False) as raw:
        target = np.asarray(raw["targets"][:, 0])
        raw_predictions = {model: np.asarray(raw[f"{model}_predictions"][:, 0]) for model in MODELS}
    with np.load(phase_path, allow_pickle=False) as phase:
        center = np.asarray(phase["targets"][:, 0])
        aligned = {model: np.asarray(phase[f"{model}_oracle_aligned_predictions"][:, 0]) for model in MODELS}
    score = np.mean(
        np.stack([np.sqrt(np.mean((raw_predictions[model] - target) ** 2, axis=1)) for model in MODELS]),
        axis=0,
    )
    order = np.argsort(score, kind="stable")
    chosen = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    rows = [("Real ECG", center, "#111111")]
    rows.extend((f"{MODEL_LABELS[model]} raw", raw_predictions[model][:, 16:-16], MODEL_COLORS[model]) for model in MODELS)
    rows.extend((f"{MODEL_LABELS[model]} aligned", aligned[model], MODEL_COLORS[model]) for model in MODELS)
    time = np.arange(480) / 128.0
    figure, axes = plt.subplots(len(rows), 3, figsize=(7.16, 7.4), sharex=True, constrained_layout=True)
    for column, index in enumerate(chosen):
        lower = min(values[index].min() for _, values, _ in rows)
        upper = max(values[index].max() for _, values, _ in rows)
        pad = 0.06 * max(upper - lower, 1e-6)
        for row_index, (label, values, color) in enumerate(rows):
            axis = axes[row_index, column]
            axis.plot(time, values[index], linewidth=0.55, color=color)
            axis.set_ylim(lower - pad, upper + pad)
            axis.grid(alpha=0.12, linewidth=0.35)
            if column == 0:
                axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title(("Low", "Median", "High")[column] + f" shared error (window {index})")
            if row_index == len(rows) - 1:
                axis.set_xlabel("Time (s)")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"waveform_phase_examples.{suffix}"
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    plt.close(figure)
    return paths


def run(args: argparse.Namespace) -> Path:
    if args.max_lag_samples != 16:
        raise ValueError("the frozen mmECG phase protocol requires max_lag_samples=16")
    _configure_ieee_fonts()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.phase_predictions, allow_pickle=False) as artifact:
        required = {"targets", "subject_ids", "source_files"}
        for model in MODELS:
            required.update({f"{model}_unshifted_predictions", f"{model}_oracle_aligned_predictions"})
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("phase artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        sources = np.asarray(artifact["source_files"]).astype(str)
        predictions = {
            (model, phase): np.asarray(artifact[f"{model}_{phase}_predictions"], dtype=np.float32)
            for model in MODELS for phase in PHASE_MODES
        }
    expected_shape = (2877, 1, 480)
    if targets.shape != expected_shape or subjects.shape != (2877,) or sources.shape != (2877,):
        raise ValueError("mmECG phase arrays violate the frozen shape contract")
    if any(values.shape != expected_shape for values in predictions.values()):
        raise ValueError("mmECG prediction shapes differ")
    if not np.all(np.isfinite(targets)) or any(not np.all(np.isfinite(values)) for values in predictions.values()):
        raise FloatingPointError("mmECG clinical arrays contain NaN or Inf")

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

    window_agreement = []
    subject_agreement = []
    per_window = []
    per_subject = []
    waveform_agreement: dict[str, object] = {}
    delineation: dict[str, object] = {}
    plot_paths: list[Path] = []
    for model in MODELS:
        waveform_agreement[model] = {}
        delineation[model] = {}
        plot_pairs = {}
        for phase in PHASE_MODES:
            measured = generated_measurements[(model, phase)]
            waveform = _waveform_summary(targets, predictions[(model, phase)])
            pearson = waveform.pop("pearson_per_record")
            waveform_agreement[model][phase] = waveform
            delineation[model][phase] = {
                "real_success": int(sum(item["success"] for item in real_measurements)),
                "generated_success": int(sum(item["success"] for item in measured)),
                "paired_success": int(sum(real["success"] and gen["success"] for real, gen in zip(real_measurements, measured))),
                "total": len(targets),
            }
            for index, (real, generated) in enumerate(zip(real_measurements, measured)):
                row = {
                    "window": index, "subject_id": subjects[index], "source_file": sources[index],
                    "model": model, "phase_mode": phase, "waveform_pearson_r": pearson[index],
                    "real_delineation_success": real["success"],
                    "generated_delineation_success": generated["success"],
                    "real_failure_reason": real["failure_reason"],
                    "generated_failure_reason": generated["failure_reason"],
                    "real_hrv_status": real["hrv_status"], "generated_hrv_status": generated["hrv_status"],
                }
                for parameter in PARAMETERS:
                    row[f"real_{parameter}"] = real["summary"].get(parameter)
                    row[f"generated_{parameter}"] = generated["summary"].get(parameter)
                per_window.append(row)
        for parameter in PARAMETERS:
            paired_rows, reference, unshifted, aligned = _parameter_triplets(
                real_measurements, generated_measurements[(model, "unshifted")],
                generated_measurements[(model, "oracle_aligned")], parameter,
            )
            for phase, generated in (("unshifted", unshifted), ("oracle_aligned", aligned)):
                result = _agreement_record(model, phase, parameter, reference, generated)
                result["inference_status"] = "descriptive_only_overlapping_windows_clustered_within_subject"
                window_agreement.append(result)
                subject_result, details = _subject_agreement(
                    subjects, paired_rows, reference, generated, model, phase, parameter,
                )
                subject_agreement.append(subject_result)
                per_subject.extend(details)
                plot_pairs[(phase, parameter)] = (reference, generated)
        plot_paths.extend(_plot_parameter_bland_altman(output, model, plot_pairs))

    plot_paths.extend(_plot_waveforms(args.raw_predictions, args.phase_predictions, output))
    window_path = output / "window_parameter_agreement.csv"
    subject_path = output / "subject_parameter_agreement.csv"
    per_window_path = output / "per_window_ecg_parameters.csv"
    per_subject_path = output / "per_subject_parameter_means.csv"
    _write_csv(window_path, window_agreement)
    _write_csv(subject_path, subject_agreement)
    _write_csv(per_window_path, per_window)
    _write_csv(per_subject_path, per_subject)
    summary_path = output / "clinical_summary.json"
    summary = {
        "schema_version": 1,
        "protocol": {
            "windows": 2877, "subjects": sorted(set(subjects)), "support_samples": 480,
            "sampling_rate_hz": args.sampling_rate,
            "phase_correction": "target-informed per-window Pearson-maximizing shift +/-16 samples",
            "raw_full_window_results_remain_primary": True,
            "parameter_pairing": "per-model/per-parameter common real/unshifted/aligned intersection",
            "delineation_padding": "reflect to 1024 samples for DWT only; retain center 480-sample fiducials",
            "qtc_formula": "fridericia", "st_offset_ms": 60.0,
            "amplitude_unit": "normalized", "amplitude_claim_allowed": False,
            "hrv_status": "blocked_non_continuous_overlapping_short_windows",
            "window_inference": "blocked_50_percent_overlap_and_subject_clustering",
            "subject_inference": "descriptive_only_extremely_underpowered_n3",
        },
        "waveform_agreement": waveform_agreement,
        "delineation": delineation,
        "parameter_agreement": window_agreement,
        "subject_parameter_agreement": subject_agreement,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = [window_path, subject_path, per_window_path, per_subject_path, summary_path, *plot_paths]
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "inputs": {
            "raw_predictions": {"path": str(args.raw_predictions.resolve()), "sha256": _sha256(args.raw_predictions)},
            "phase_predictions": {"path": str(args.phase_predictions.resolve()), "sha256": _sha256(args.phase_predictions)},
        },
        "execution": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"), "workers": args.workers,
            "script_sha256": _sha256(Path(__file__)),
        },
        "claim_boundary": "Oracle shifting is target-informed and diagnostic only. Normalized amplitudes are not physical units. Overlapping short windows and only three held-out subjects prohibit significance, HRV-preservation, clinical, or deployment claims.",
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_predictions", type=Path, required=True)
    parser.add_argument("--phase_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"mmECG phase-clinical evaluation saved to {result}")
