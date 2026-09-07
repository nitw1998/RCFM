#!/usr/bin/env python3
"""Compute target-informed phase metrics for single-channel random-window models."""

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

from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


EXPECTED_SAMPLES = 512
DATASET_SPECS = {
    "wesad_window_minmax": {
        "dataset": "WESAD", "windows": 4342, "sampling_rate": 128,
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "required_sidecars": {"targets", "subject_ids", "record_ids", "labels"},
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-window-minmax-v1",
        "normalization_id": "window_minmax_neg1_1_v1",
    },
    "wesad_record_minmax": {
        "dataset": "WESAD", "windows": 4342, "sampling_rate": 128,
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "required_sidecars": {"targets", "subject_ids", "record_ids", "labels"},
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "normalization_id": "source_record_minmax_neg1_1_v1",
    },
    "mmecg": {
        "dataset": "mmECG", "windows": 2494, "sampling_rate": 200,
        "split_hash": "6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2",
        "required_sidecars": {"targets", "subject_ids", "record_ids"},
        "dataset_version": "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1",
        "normalization_id": "window_minmax_neg1_1_v1",
    },
}
VARIANTS = {
    "window_minmax": {
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-window-minmax-v1",
        "normalization_id": "window_minmax_neg1_1_v1",
    },
    "record_minmax": {
        "dataset_version": DATASET_SPECS["wesad_record_minmax"]["dataset_version"],
        "normalization_id": "source_record_minmax_neg1_1_v1",
    },
}


def _metric_view(summary: dict[str, object]) -> dict[str, object]:
    correlation = summary["per_record_pearson"]
    return {
        "rmse": summary["rmse"],
        "mae": summary["mae"],
        "waveform_fd": summary["waveform_fd"],
        "pointwise_pearson_r": summary["pointwise_correlation_descriptive_only"]["r"],
        "per_window_pearson_mean": correlation["mean"],
        "per_window_pearson_median": correlation["median"],
    }


def run(args: argparse.Namespace) -> Path:
    dataset_key = args.dataset
    if args.dataset_variant is not None:
        dataset_key = (
            "wesad_record_minmax"
            if args.dataset_variant == "record_minmax"
            else "wesad_window_minmax"
        )
    spec = DATASET_SPECS[dataset_key]
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sampling_rate = spec["sampling_rate"] if args.sampling_rate is None else args.sampling_rate
    if sampling_rate != spec["sampling_rate"] or args.max_lag_samples != 16:
        raise ValueError("phase protocol requires the dataset sampling rate and a +/-16-sample search")

    source_protocol_path = source / "protocol.json"
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    protocol = source_protocol.get("protocol", {})
    source_model = protocol.get("model")
    if source_model is None and source_protocol.get("checkpoint", {}).get("kind") == "canonical_multistep_cfm":
        source_model = "cfm"
    if (
        source_protocol.get("status") != "completed"
        or source_model != args.model
        or protocol.get("dataset") != spec["dataset"]
        or protocol.get("dataset_version") != spec["dataset_version"]
        or protocol.get("split_hash") != spec["split_hash"]
        or protocol.get("normalization_id") != spec["normalization_id"]
        or protocol.get("phase_correction_applied") is not False
        or int(protocol.get("evaluated_records", -1)) != spec["windows"]
    ):
        raise ValueError("source prediction artifact violates the random-window phase contract")

    with np.load(source / "paired_reference.npz", allow_pickle=False) as artifact:
        required = set(spec["required_sidecars"])
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("paired reference is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        record_ids = np.asarray(artifact["record_ids"]).astype(str)
        labels = np.asarray(artifact["labels"], dtype=np.int16) if "labels" in artifact.files else None
    prediction_path = source / f"{args.model}_predictions.npy"
    predictions = np.asarray(np.load(prediction_path, mmap_mode="r"), dtype=np.float32)
    expected_shape = (int(spec["windows"]), 1, EXPECTED_SAMPLES)
    if targets.shape != expected_shape or predictions.shape != expected_shape:
        raise ValueError(f"phase arrays must have shape {expected_shape}")
    sidecars = (subjects, record_ids) if labels is None else (subjects, record_ids, labels)
    if any(values.shape != (spec["windows"],) for values in sidecars):
        raise ValueError("phase sidecars must align with waveform rows")
    if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(predictions)):
        raise FloatingPointError("WESAD phase inputs contain NaN or Inf")

    lag_summary, lag_values = _lag_diagnostic(
        targets, predictions, args.max_lag_samples, sampling_rate
    )
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    center, unshifted, aligned = _fixed_support_align(
        targets, predictions, shifts, args.max_lag_samples
    )
    raw_summary, raw_window = _waveform_metrics(targets, predictions)
    before_summary, before_window = _waveform_metrics(center, unshifted)
    after_summary, after_window = _waveform_metrics(center, aligned)

    phase_path = output / "phase_predictions_maxlag16.npz"
    np.savez_compressed(
        phase_path,
        targets=center,
        unshifted_predictions=unshifted,
        oracle_aligned_predictions=aligned,
        oracle_shift_samples=shifts,
        oracle_shift_ms=shifts.astype(np.float64) * 1000.0 / sampling_rate,
        subject_ids=subjects,
        record_ids=record_ids,
        **({"labels": labels} if labels is not None else {}),
    )
    rows_path = output / "per_window_phase_metrics.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "window_index", "record_id", "subject_id", "oracle_shift_samples",
            "oracle_shift_ms", "unshifted_fixed_support_rmse", "oracle_aligned_rmse",
            "unshifted_fixed_support_mae", "oracle_aligned_mae",
            "unshifted_fixed_support_pearson_r", "oracle_aligned_pearson_r",
        ]
        if labels is not None:
            fields.insert(3, "label")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(int(spec["windows"])):
            row = {
                "window_index": index,
                "record_id": record_ids[index],
                "subject_id": subjects[index],
                "oracle_shift_samples": int(shifts[index]),
                "oracle_shift_ms": float(shifts[index]) * 1000.0 / sampling_rate,
                "unshifted_fixed_support_rmse": before_window["rmse"][index],
                "oracle_aligned_rmse": after_window["rmse"][index],
                "unshifted_fixed_support_mae": before_window["mae"][index],
                "oracle_aligned_mae": after_window["mae"][index],
                "unshifted_fixed_support_pearson_r": before_window["pearson_r"][index],
                "oracle_aligned_pearson_r": after_window["pearson_r"][index],
            }
            if labels is not None:
                row["label"] = int(labels[index])
            writer.writerow(row)

    summary_path = output / "waveform_phase_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "dataset": spec["dataset"],
        "model": args.model,
        "phase_protocol": {
            "dataset_variant": dataset_key,
            "dataset_version": spec["dataset_version"],
            "normalization_id": spec["normalization_id"],
            "target_informed": True,
            "selection": "per-window integer shift maximizing Pearson correlation",
            "search_samples": [-args.max_lag_samples, args.max_lag_samples],
            "search_ms": [
                -args.max_lag_samples * 1000.0 / sampling_rate,
                args.max_lag_samples * 1000.0 / sampling_rate,
            ],
            "fixed_support_samples": int(center.shape[-1]),
            "fixed_support_seconds": float(center.shape[-1] / sampling_rate),
            "claim_boundary": "Oracle morphology diagnostic using the paired ECG target; not deployable inference.",
        },
        "raw_full_512_samples": _metric_view(raw_summary),
        "unshifted_fixed_480_samples": _metric_view(before_summary),
        "oracle_aligned_fixed_480_samples": _metric_view(after_summary),
        "lag_distribution": {
            **lag_summary,
            "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
            "median_absolute_shift_ms": float(np.median(np.abs(shifts)) * 1000.0 / sampling_rate),
            "boundary_hit_fraction": float(np.mean(np.abs(shifts) == args.max_lag_samples)),
            "negative_shift_fraction": float(np.mean(shifts < 0)),
            "zero_shift_fraction": float(np.mean(shifts == 0)),
            "positive_shift_fraction": float(np.mean(shifts > 0)),
        },
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    outputs = (phase_path, rows_path, summary_path)
    result_protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {
            "source_protocol_sha256": _sha256(source_protocol_path),
            "paired_reference_sha256": _sha256(source / "paired_reference.npz"),
            "prediction_sha256": _sha256(prediction_path),
            "target_array_sha256": _array_sha256(targets),
            "prediction_array_sha256": _array_sha256(predictions),
        },
        "protocol": {
            "dataset": spec["dataset"], "model": args.model,
            "windows": spec["windows"], "sampling_rate_hz": sampling_rate,
            "max_lag_samples": 16, "fixed_support_samples": 480,
            "phase_correction_target_informed": True,
            "dataset_variant": dataset_key,
            "dataset_version": spec["dataset_version"],
            "normalization_id": spec["normalization_id"],
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": "Target-informed phase correction is descriptive and cannot be used at deployment.",
    }
    (output / "protocol.json").write_text(
        json.dumps(result_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), default="wesad_record_minmax")
    parser.add_argument("--model", choices=("cfm", "rcfm", "rcfm_ot", "rddm"), default="cfm")
    parser.add_argument("--sampling_rate", type=int, default=None)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument("--dataset_variant", choices=tuple(VARIANTS), default=None)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
