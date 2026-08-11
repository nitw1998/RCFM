"""Build descriptive mmECG baselines, paired model differences, and waveform QC."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256, _waveform_metrics


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")


def _negative_dominant(signal: np.ndarray) -> np.ndarray:
    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError("polarity QC requires (records,1,samples)")
    centered = signal[:, 0] - np.median(signal[:, 0], axis=1, keepdims=True)
    return np.abs(np.min(centered, axis=1)) > np.max(centered, axis=1)


def _paired_rows(per_record_rmse: dict[str, np.ndarray]) -> list[dict[str, object]]:
    if set(per_record_rmse) != set(MODELS):
        raise ValueError("paired comparison requires all four models")
    lengths = {len(values) for values in per_record_rmse.values()}
    if len(lengths) != 1:
        raise ValueError("paired model rows must have equal length")
    rows = []
    for left_index, left in enumerate(MODELS):
        for right in MODELS[left_index + 1 :]:
            difference = per_record_rmse[left] - per_record_rmse[right]
            rows.append(
                {
                    "left_model": left, "right_model": right,
                    "difference_definition": "left_minus_right_per_window_rmse",
                    "windows": len(difference), "mean_difference": float(np.mean(difference)),
                    "median_difference": float(np.median(difference)),
                    "left_lower_rmse_fraction": float(np.mean(difference < 0)),
                    "ties_fraction": float(np.mean(difference == 0)),
                    "inference_status": "descriptive_only_overlapping_windows_clustered_within_subject",
                }
            )
    return rows


def _safe_summary(reference: np.ndarray, generated: np.ndarray) -> dict[str, object]:
    summary, _ = _waveform_metrics(reference, generated)
    correlation = summary["per_record_pearson"]
    if correlation["usable_records"] == 0:
        correlation["mean"] = None
        correlation["median"] = None
    return summary


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(args.raw_predictions, allow_pickle=False) as artifact:
        required = {"targets", "conditions", "subject_ids", "source_files"}
        required.update(f"{model}_predictions" for model in MODELS)
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("mmECG artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        conditions = np.asarray(artifact["conditions"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        sources = np.asarray(artifact["source_files"]).astype(str)
        predictions = {model: np.asarray(artifact[f"{model}_predictions"], dtype=np.float32) for model in MODELS}
    expected = (2877, 1, 512)
    if targets.shape != expected or conditions.shape != expected or any(values.shape != expected for values in predictions.values()):
        raise ValueError("mmECG arrays violate the frozen shape contract")
    per_record_rmse = {
        model: np.sqrt(np.mean((predictions[model].astype(np.float64) - targets) ** 2, axis=(1, 2)))
        for model in MODELS
    }
    paired = _paired_rows(per_record_rmse)
    paired_path = output / "paired_window_rmse_differences.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    negative = _negative_dominant(targets)
    polarity_rows = [
        {
            "window": int(index), "subject_id": subjects[index], "source_file": sources[index],
            "negative_dominant_target": True,
        }
        for index in np.flatnonzero(negative)
    ]
    polarity_path = output / "negative_dominant_target_windows.csv"
    with polarity_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("window", "subject_id", "source_file", "negative_dominant_target"))
        writer.writeheader()
        writer.writerows(polarity_rows)

    summary_path = output / "analysis_summary.json"
    summary = {
        "schema_version": 1,
        "baselines": {
            "zero": _safe_summary(targets, np.zeros_like(targets)),
            "condition_copy": _safe_summary(targets, conditions),
            "condition_copy_scope": "scale-matched diagnostic only; RCG and ECG are different modalities",
        },
        "paired_window_rmse": paired,
        "target_polarity_qc": {
            "definition": "absolute median-centered negative excursion exceeds positive excursion",
            "negative_dominant_windows": int(np.sum(negative)),
            "total_windows": len(targets),
            "fraction": float(np.mean(negative)),
            "subjects": sorted(set(subjects[negative])),
            "source_files": sorted(set(sources[negative])),
        },
        "claim_boundary": "All paired window summaries are descriptive because windows overlap by 50% and cluster within only three held-out subjects. Condition copy is not a valid reconstruction model.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = (paired_path, polarity_path, summary_path)
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "input": {"path": str(args.raw_predictions.resolve()), "sha256": _sha256(args.raw_predictions)},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "script_sha256": _sha256(Path(__file__))},
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"mmECG descriptive analysis saved to {result}")
