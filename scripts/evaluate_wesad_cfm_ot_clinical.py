"""Evaluate WESAD CFM/CFM+OT ECG parameters, Bland--Altman, and HRV blocks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
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
    PARAMETERS,
    PHASE_MODES,
    _agreement_record,
    _measure_records,
    _parameter_triplets,
    _plot_bland_altman,
    _waveform_summary,
)
from scripts.evaluate_mmecg_phase_clinical import _subject_agreement
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation


MODELS = ("cfm", "cfm_ot")
MODEL_LABELS = {"cfm": "CFM", "cfm_ot": "CFM+OT"}
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
HRV_PARAMETERS = ("mean_rr_ms", "heart_rate_bpm", "sdnn_ms", "rmssd_ms", "pnn50_percent")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _five_minute_blocks(subjects: np.ndarray, labels: np.ndarray, windows_per_block: int = 75) -> list[np.ndarray]:
    blocks = []
    start = 0
    while start < len(subjects):
        stop = start + 1
        while stop < len(subjects) and subjects[stop] == subjects[start] and labels[stop] == labels[start]:
            stop += 1
        if int(labels[start]) in LABELS:
            for block_start in range(start, stop - windows_per_block + 1, windows_per_block):
                blocks.append(np.arange(block_start, block_start + windows_per_block))
        start = stop
    return blocks


def _hrv_summary(measurements: list[dict[str, object]], indices: np.ndarray) -> dict[str, float] | None:
    rr_by_window = [np.asarray(measurements[index].get("rr_intervals_ms", []), dtype=np.float64) for index in indices]
    rr_by_window = [values[np.isfinite(values) & (values > 0)] for values in rr_by_window]
    all_rr = np.concatenate([values for values in rr_by_window if len(values)]) if any(len(values) for values in rr_by_window) else np.empty(0)
    successive = np.concatenate([np.diff(values) for values in rr_by_window if len(values) >= 2]) if any(len(values) >= 2 for values in rr_by_window) else np.empty(0)
    if len(all_rr) < 2 or len(successive) < 1:
        return None
    mean_rr = float(np.mean(all_rr))
    return {
        "mean_rr_ms": mean_rr,
        "heart_rate_bpm": 60000.0 / mean_rr,
        "sdnn_ms": float(np.std(all_rr, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(successive ** 2))),
        "pnn50_percent": float(100.0 * np.mean(np.abs(successive) > 50.0)),
        "n_rr": int(len(all_rr)),
    }


def _hrv_agreement(rows: list[dict[str, object]], model: str, phase: str, parameter: str) -> dict[str, object]:
    selected = [row for row in rows if row["model"] == model and row["phase_mode"] == phase]
    real = np.asarray([row[f"real_{parameter}"] for row in selected], dtype=np.float64)
    generated = np.asarray([row[f"generated_{parameter}"] for row in selected], dtype=np.float64)
    base = {"model": model, "phase_mode": phase, "parameter": parameter,
            "unit": "%" if parameter == "pnn50_percent" else ("bpm" if parameter == "heart_rate_bpm" else "ms"),
            "n": int(len(real)), "inference_status": "descriptive_only_blocks_clustered_within_three_subjects"}
    if len(real) < 2:
        return {**base, "status": "insufficient_blocks"}
    error = generated - real
    agreement = bland_altman(real, generated)
    correlation = paired_correlation(real, generated)
    return {**base, "status": "ok", "reference_mean": float(np.mean(real)),
            "generated_mean": float(np.mean(generated)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "pearson_r": correlation["r"],
            "bland_altman_bias": agreement["bias"], "bland_altman_lower_limit": agreement["lower_limit"],
            "bland_altman_upper_limit": agreement["upper_limit"]}


def _plot_hrv(output: Path, rows: list[dict[str, object]], model: str) -> list[Path]:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times"],
                         "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(7.16, 4.2), constrained_layout=True)
    colors = {"unshifted": "#6f7782", "oracle_aligned": "#c43d4b"}
    for axis, parameter in zip(axes.flat, HRV_PARAMETERS):
        for phase in PHASE_MODES:
            selected = [row for row in rows if row["model"] == model and row["phase_mode"] == phase]
            real = np.asarray([row[f"real_{parameter}"] for row in selected])
            generated = np.asarray([row[f"generated_{parameter}"] for row in selected])
            agreement = bland_altman(real, generated)
            axis.scatter((real + generated) / 2, generated - real, s=10, alpha=0.4,
                         color=colors[phase], label="Raw" if phase == "unshifted" else "Oracle")
            axis.axhline(agreement["bias"], color=colors[phase], linewidth=1)
            axis.axhline(agreement["lower_limit"], color=colors[phase], linestyle="--", linewidth=0.7)
            axis.axhline(agreement["upper_limit"], color=colors[phase], linestyle="--", linewidth=0.7)
        axis.set_title(parameter); axis.set_xlabel("Pair mean"); axis.set_ylabel("Generated - real"); axis.grid(alpha=0.15)
    axes.flat[-1].set_visible(False); axes.flat[0].legend(frameon=False)
    figure.suptitle(f"{MODEL_LABELS[model]} WESAD 5-minute HRV Bland-Altman")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{model}_hrv_bland_altman.{suffix}"
        figure.savefig(path, dpi=600 if suffix == "png" else None); paths.append(path)
    plt.close(figure)
    return paths


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.phase_predictions, allow_pickle=False) as artifact:
        required = {"targets", "subject_ids", "labels"}
        for model in MODELS:
            required.update({f"{model}_unshifted_predictions", f"{model}_oracle_aligned_predictions"})
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("WESAD phase artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        labels = np.asarray(artifact["labels"], dtype=np.int16)
        predictions = {(model, phase): np.asarray(artifact[f"{model}_{phase}_predictions"], dtype=np.float32)
                       for model in MODELS for phase in PHASE_MODES}
    expected = (4213, 1, 480)
    if targets.shape != expected or any(values.shape != expected for values in predictions.values()):
        raise ValueError("WESAD phase arrays violate the 4213x1x480 contract")
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        real_measurements = _measure_records(targets, args.sampling_rate, executor)
        generated_measurements = {key: _measure_records(values, args.sampling_rate, executor)
                                  for key, values in predictions.items()}
    finally:
        if executor is not None:
            executor.shutdown()
    window_agreement, subject_agreement, per_window, per_subject = [], [], [], []
    waveform, delineation, plot_paths = {}, {}, []
    for model in MODELS:
        waveform[model], delineation[model] = {}, {}
        plot_pairs = {}
        for phase in PHASE_MODES:
            measured = generated_measurements[(model, phase)]
            summary = _waveform_summary(targets, predictions[(model, phase)])
            pearson = summary.pop("pearson_per_record")
            waveform[model][phase] = summary
            delineation[model][phase] = {"real_success": int(sum(item["success"] for item in real_measurements)),
                                         "generated_success": int(sum(item["success"] for item in measured)),
                                         "total": len(targets)}
            for index, (real, generated) in enumerate(zip(real_measurements, measured)):
                row = {"window": index, "subject_id": subjects[index], "label": int(labels[index]),
                       "model": model, "phase_mode": phase, "waveform_pearson_r": pearson[index],
                       "real_success": real["success"], "generated_success": generated["success"]}
                for parameter in PARAMETERS:
                    row[f"real_{parameter}"] = real["summary"].get(parameter)
                    row[f"generated_{parameter}"] = generated["summary"].get(parameter)
                per_window.append(row)
        for parameter in PARAMETERS:
            paired_rows, reference, unshifted, aligned = _parameter_triplets(
                real_measurements, generated_measurements[(model, "unshifted")],
                generated_measurements[(model, "oracle_aligned")], parameter)
            for phase, generated in (("unshifted", unshifted), ("oracle_aligned", aligned)):
                result = _agreement_record(model, phase, parameter, reference, generated,
                                           "descriptive_only_windows_clustered_within_three_subjects")
                window_agreement.append(result)
                subject_result, details = _subject_agreement(subjects, paired_rows, reference, generated,
                                                              model, phase, parameter)
                subject_agreement.append(subject_result); per_subject.extend(details)
                plot_pairs[(phase, parameter)] = (reference, generated)
        plot_paths.extend(_plot_bland_altman(output, model, plot_pairs))
    blocks = _five_minute_blocks(subjects, labels)
    hrv_rows = []
    for indices in blocks:
        real_hrv = _hrv_summary(real_measurements, indices)
        if real_hrv is None:
            continue
        for model in MODELS:
            for phase in PHASE_MODES:
                generated_hrv = _hrv_summary(generated_measurements[(model, phase)], indices)
                if generated_hrv is None:
                    continue
                row = {"subject_id": subjects[indices[0]], "label": int(labels[indices[0]]),
                       "label_name": LABELS[int(labels[indices[0]])], "start_window": int(indices[0]),
                       "model": model, "phase_mode": phase}
                for parameter in (*HRV_PARAMETERS, "n_rr"):
                    row[f"real_{parameter}"] = real_hrv[parameter]
                    row[f"generated_{parameter}"] = generated_hrv[parameter]
                hrv_rows.append(row)
    hrv_agreement = [_hrv_agreement(hrv_rows, model, phase, parameter)
                     for model in MODELS for phase in PHASE_MODES for parameter in HRV_PARAMETERS]
    for model in MODELS:
        plot_paths.extend(_plot_hrv(output, hrv_rows, model))
    paths = {
        "window": output / "window_parameter_agreement.csv",
        "subject": output / "subject_parameter_agreement.csv",
        "per_window": output / "per_window_ecg_parameters.csv",
        "per_subject": output / "per_subject_parameter_means.csv",
        "blocks": output / "five_minute_hrv_blocks.csv",
        "hrv": output / "hrv_agreement.csv",
    }
    for key, rows in (("window", window_agreement), ("subject", subject_agreement),
                      ("per_window", per_window), ("per_subject", per_subject),
                      ("blocks", hrv_rows), ("hrv", hrv_agreement)):
        _write_csv(paths[key], rows)
    summary_path = output / "clinical_summary.json"
    summary_path.write_text(json.dumps({"schema_version": 1, "protocol": {
        "windows": 4213, "subjects": sorted(set(subjects)), "support_samples": 480,
        "phase_correction": "target-informed per-window Pearson-maximizing +/-16 samples",
        "amplitude_unit": "normalized", "amplitude_claim_allowed": False,
        "hrv": "nonoverlapping 5-minute same-label blocks; within-window RR only; boundary RR censored",
        "hrv_candidate_blocks": len(blocks), "subject_inference": "descriptive_only_n3"},
        "waveform_agreement": waveform, "delineation": delineation,
        "window_parameter_agreement": window_agreement, "subject_parameter_agreement": subject_agreement,
        "hrv_agreement": hrv_agreement}, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = [*paths.values(), summary_path, *plot_paths]
    protocol = {"schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join(sys.argv), "input": {"path": str(args.phase_predictions.resolve()),
                "sha256": _sha256(args.phase_predictions)}, "execution": {"python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__, "neurokit2": getattr(nk, "__version__", "unknown"),
                "workers": args.workers, "script_sha256": _sha256(Path(__file__))},
                "claim_boundary": "Oracle alignment is target-informed; normalized amplitudes are not physical; n=3 subjects prohibits confirmatory significance.",
                "outputs": {path.name: _sha256(path) for path in outputs}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--workers", type=int, default=16)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"WESAD CFM/CFM+OT clinical evaluation saved to {result}")
