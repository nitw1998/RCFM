#!/usr/bin/env python3
"""Analyze phase-corrected random-window MIMIC-AFib comparator predictions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_mimic_random_phase_clinical_mpl")

import neurokit2 as nk
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_clinical import (
    AMPLITUDE_PARAMETERS,
    INTERVAL_PARAMETERS,
    PARAMETERS,
    _agreement_record,
    _plot_bland_altman,
)
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_random_window_ecg_clinical import _measure_many
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


EXPECTED_DATASET_VERSION = (
    "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-"
    "rddm-window-minmax-v1"
)
EXPECTED_SPLIT_HASH = "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232"
EXPECTED_ROWS = 2040
EXPECTED_RECORDS = 34


def grouped_parameter_triplets(
    reference: list[dict[str, object]],
    unshifted: list[dict[str, object]],
    aligned: list[dict[str, object]],
    group_ids: np.ndarray,
    parameter: str,
    selected: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average jointly measurable window triplets within each source record."""

    groups = np.asarray(group_ids).astype(str)
    if len(groups) != len(reference) or len(unshifted) != len(groups) or len(aligned) != len(groups):
        raise ValueError("measurement and source-record arrays must align")
    keep = np.ones(len(groups), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    if keep.shape != (len(groups),):
        raise ValueError("selection mask must align with source records")
    values: dict[str, tuple[list[float], list[float], list[float]]] = defaultdict(
        lambda: ([], [], [])
    )
    for index, group in enumerate(groups):
        if not keep[index]:
            continue
        triplet = (
            reference[index]["summary"].get(parameter),
            unshifted[index]["summary"].get(parameter),
            aligned[index]["summary"].get(parameter),
        )
        if any(item is None or not np.isfinite(item) for item in triplet):
            continue
        values[group][0].append(float(triplet[0]))
        values[group][1].append(float(triplet[1]))
        values[group][2].append(float(triplet[2]))
    ordered = np.asarray(sorted(values))
    return (
        ordered,
        np.asarray([np.mean(values[group][0]) for group in ordered], dtype=np.float64),
        np.asarray([np.mean(values[group][1]) for group in ordered], dtype=np.float64),
        np.asarray([np.mean(values[group][2]) for group in ordered], dtype=np.float64),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _clinical_counts(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "success": int(sum(bool(item["success"]) for item in items)),
        "total": len(items),
        "failures": dict(Counter(item["failure_reason"] for item in items if not item["success"])),
    }


def _load_input(
    input_dir: Path, model_key: str = "cfm"
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    protocol = json.loads((input_dir / "protocol.json").read_text(encoding="utf-8"))
    expected = {
        "dataset": "MIMIC-AFib",
        "dataset_version": EXPECTED_DATASET_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "evaluated_records": EXPECTED_ROWS,
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "phase_correction_applied": False,
    }
    bad = [key for key, value in expected.items() if protocol.get("protocol", {}).get(key) != value]
    if protocol.get("status") != "completed":
        bad.append("status")
    source_model = protocol.get("protocol", {}).get("model", "cfm")
    if source_model != model_key:
        bad.append("model")
    if bad:
        raise ValueError("raw prediction protocol mismatch: " + ", ".join(bad))
    with np.load(input_dir / "paired_reference.npz", allow_pickle=False) as source:
        required = {
            "targets", "conditions", "record_ids", "subject_ids", "window_start_samples",
            "afib_labels", "source_rows", "source_split_codes",
        }
        if required - set(source.files):
            raise ValueError("paired reference lacks MIMIC identity sidecars")
        arrays = {key: np.asarray(source[key]) for key in required}
    arrays["predictions"] = np.asarray(
        np.load(input_dir / f"{model_key}_predictions.npy", mmap_mode="r"),
        dtype=np.float32,
    )
    arrays["targets"] = np.asarray(arrays["targets"], dtype=np.float32)
    arrays["conditions"] = np.asarray(arrays["conditions"], dtype=np.float32)
    expected_shape = (EXPECTED_ROWS, 1, 512)
    if any(arrays[key].shape != expected_shape for key in ("targets", "conditions", "predictions")):
        raise ValueError("MIMIC waveform arrays must have shape (2040,1,512)")
    for key in ("record_ids", "subject_ids", "window_start_samples", "afib_labels", "source_rows", "source_split_codes"):
        if arrays[key].shape != (EXPECTED_ROWS,):
            raise ValueError(f"MIMIC sidecar {key} does not align")
    arrays["record_ids"] = arrays["record_ids"].astype(str)
    arrays["subject_ids"] = arrays["subject_ids"].astype(str)
    arrays["afib_labels"] = arrays["afib_labels"].astype(bool)
    if len(np.unique(arrays["record_ids"])) != EXPECTED_RECORDS:
        raise ValueError("expected 34 source records")
    identities = set(zip(arrays["record_ids"].tolist(), arrays["window_start_samples"].tolist()))
    if len(identities) != EXPECTED_ROWS:
        raise ValueError("record/start identities are not unique")
    for record in np.unique(arrays["record_ids"]):
        selected = arrays["record_ids"] == record
        if len(np.unique(arrays["afib_labels"][selected])) != 1:
            raise ValueError("AF label varies within a source record")
    if any(not np.all(np.isfinite(arrays[key])) for key in ("targets", "conditions", "predictions")):
        raise FloatingPointError("waveform arrays must be finite")
    return protocol, arrays


def run(args: argparse.Namespace) -> Path:
    if args.sampling_rate != 128 or args.max_lag_samples != 16:
        raise ValueError("frozen phase protocol requires 128 Hz and +/-16 samples")
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_key = args.model
    model_label = {"cfm": "CFM", "rcfm": "RCFM", "rcfm_ot": "RCFM-OT", "rddm": "RDDM"}[
        model_key
    ]
    source_protocol, arrays = _load_input(input_dir, model_key)
    targets = arrays["targets"]
    predictions = arrays["predictions"]

    lag_summary, lag_values = _lag_diagnostic(
        targets, predictions, args.max_lag_samples, args.sampling_rate
    )
    shifts = np.asarray(lag_values["best_lag_samples"], dtype=np.int32)
    target_fixed, unshifted_fixed, aligned_fixed = _fixed_support_align(
        targets, predictions, shifts, args.max_lag_samples
    )
    raw_summary, raw_per = _waveform_metrics(targets, predictions)
    unshifted_summary, unshifted_per = _waveform_metrics(target_fixed, unshifted_fixed)
    aligned_summary, aligned_per = _waveform_metrics(target_fixed, aligned_fixed)
    waveform = {
        "raw_full_512": raw_summary,
        "unshifted_fixed_support_480": unshifted_summary,
        "oracle_aligned_fixed_support_480": aligned_summary,
        "lag_diagnostic": lag_summary,
        "shift_distribution": {
            "mean_samples": float(np.mean(shifts)),
            "median_samples": float(np.median(shifts)),
            "median_absolute_samples": float(np.median(np.abs(shifts))),
            "negative_fraction": float(np.mean(shifts < 0)),
            "zero_fraction": float(np.mean(shifts == 0)),
            "positive_fraction": float(np.mean(shifts > 0)),
            "boundary_fraction": float(np.mean(np.abs(shifts) == args.max_lag_samples)),
            "rmse_improved_fraction": float(np.mean(aligned_per["rmse"] < unshifted_per["rmse"])),
        },
    }
    phase_path = output / "phase_corrected_predictions.npz"
    np.savez_compressed(
        phase_path,
        targets=target_fixed,
        unshifted_predictions=unshifted_fixed,
        oracle_aligned_predictions=aligned_fixed,
        oracle_shift_samples=shifts,
        record_ids=arrays["record_ids"],
        subject_ids=arrays["subject_ids"],
        window_start_samples=arrays["window_start_samples"],
        afib_labels=arrays["afib_labels"],
        source_rows=arrays["source_rows"],
        source_split_codes=arrays["source_split_codes"],
    )
    metrics_path = output / "per_window_phase_metrics.npz"
    np.savez_compressed(
        metrics_path,
        oracle_shift_samples=shifts,
        unshifted_rmse=unshifted_per["rmse"],
        aligned_rmse=aligned_per["rmse"],
        unshifted_mae=unshifted_per["mae"],
        aligned_mae=aligned_per["mae"],
        unshifted_pearson=unshifted_per["pearson_r"],
        aligned_pearson=aligned_per["pearson_r"],
    )

    p_wave_applicable = ~arrays["afib_labels"]
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        print("clinical delineation: fixed-support reference", flush=True)
        real = _measure_many(
            target_fixed[:, 0], args.sampling_rate, False, p_wave_applicable, 4.0, executor
        )
        print(f"clinical delineation: unshifted {model_label}", flush=True)
        unshifted = _measure_many(
            unshifted_fixed[:, 0], args.sampling_rate, False, p_wave_applicable, 4.0, executor
        )
        print(f"clinical delineation: oracle phase-corrected {model_label}", flush=True)
        aligned = _measure_many(
            aligned_fixed[:, 0], args.sampling_rate, False, p_wave_applicable, 4.0, executor
        )
    finally:
        if executor is not None:
            executor.shutdown()

    agreement_rows: list[dict[str, object]] = []
    plot_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    subgroup_masks = {
        "all": np.ones(EXPECTED_ROWS, dtype=bool),
        "afib": arrays["afib_labels"],
        "non_afib": ~arrays["afib_labels"],
    }
    for parameter in PARAMETERS:
        for subgroup, selected in subgroup_masks.items():
            group_ids, reference_values, before_values, after_values = grouped_parameter_triplets(
                real, unshifted, aligned, arrays["record_ids"], parameter, selected
            )
            for phase_mode, generated_values in (
                ("unshifted", before_values), ("oracle_aligned", after_values)
            ):
                row = _agreement_record(
                    model_key, phase_mode, parameter, reference_values, generated_values,
                    inference_status="descriptive_source_record_aggregate_non_grouped_validation",
                )
                row.update({"subgroup": subgroup, "group_level": "source_record", "n_groups": len(group_ids)})
                agreement_rows.append(row)
            if subgroup == "all":
                plot_pairs[("unshifted", parameter)] = (reference_values, before_values)
                plot_pairs[("oracle_aligned", parameter)] = (reference_values, after_values)

    per_window_rows: list[dict[str, object]] = []
    for index, (real_item, before_item, after_item) in enumerate(zip(real, unshifted, aligned)):
        row: dict[str, object] = {
            "window_index": index,
            "record_id": arrays["record_ids"][index],
            "subject_id": arrays["subject_ids"][index],
            "window_start_samples": int(arrays["window_start_samples"][index]),
            "afib": bool(arrays["afib_labels"][index]),
            "oracle_shift_samples": int(shifts[index]),
            "oracle_shift_ms": float(shifts[index] * 1000.0 / args.sampling_rate),
            "reference_success": bool(real_item["success"]),
            "unshifted_success": bool(before_item["success"]),
            "oracle_aligned_success": bool(after_item["success"]),
        }
        for parameter in PARAMETERS:
            row[f"reference_{parameter}"] = real_item["summary"].get(parameter)
            row[f"unshifted_{parameter}"] = before_item["summary"].get(parameter)
            row[f"oracle_aligned_{parameter}"] = after_item["summary"].get(parameter)
        per_window_rows.append(row)

    agreement_path = output / "source_record_ecg_parameter_agreement.csv"
    per_window_path = output / "per_window_ecg_parameters.csv"
    _write_csv(agreement_path, agreement_rows)
    _write_csv(per_window_path, per_window_rows)
    figure_paths = _plot_bland_altman(output, model_key, plot_pairs)
    summary_path = output / "summary.json"
    summary = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "MIMIC-AFib",
        "model": model_label,
        "protocol": {
            "validation_windows": EXPECTED_ROWS,
            "source_records": EXPECTED_RECORDS,
            "sampling_rate_hz": args.sampling_rate,
            "raw_window_samples": 512,
            "fixed_support_samples": 480,
            "max_lag_samples": args.max_lag_samples,
            "max_lag_ms": args.max_lag_samples * 1000.0 / args.sampling_rate,
            "lag_objective": "maximize per-window Pearson against paired target ECG",
            "positive_shift_definition": "delay generated waveform",
            "delineation": "independent NeuroKit2 DWT after reflect-padding 480 to 512 samples",
            "parameter_aggregation": "mean jointly measurable windows within source record before agreement",
            "p_wave_policy": "PR/P amplitude disabled for AF-labelled windows",
            "qtc_formula": "fridericia",
            "st_offset_ms": 60.0,
            "amplitude_unit": "normalized_exploratory_not_physical",
            "hrv": "blocked_noncontinuous_four_second_random_windows",
        },
        "waveform": waveform,
        "delineation": {
            "reference": _clinical_counts(real),
            "unshifted": _clinical_counts(unshifted),
            "oracle_aligned": _clinical_counts(aligned),
        },
        "source_record_parameter_agreement": agreement_rows,
        "claim_boundary": (
            "The phase correction is target-informed and oracle-only. All subjects/source records "
            "occur in both training and validation; source-record aggregation does not restore "
            "independence. Parameter and Bland-Altman results are descriptive algorithmic agreement."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    output_files = [phase_path, metrics_path, agreement_path, per_window_path, summary_path, *figure_paths]
    protocol_path = output / "protocol.json"
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {
            "directory": str(input_dir),
            "source_protocol_sha256": _sha256(input_dir / "protocol.json"),
            "paired_reference_sha256": _sha256(input_dir / "paired_reference.npz"),
            "prediction_file_sha256": _sha256(input_dir / f"{model_key}_predictions.npy"),
            "prediction_array_sha256": _array_sha256(predictions),
        },
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"),
            "workers": args.workers,
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "source_prediction_protocol": source_protocol["protocol"],
        "outputs": {path.name: _sha256(path) for path in output_files},
        "claim_boundary": summary["claim_boundary"],
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model", choices=("cfm", "rcfm", "rcfm_ot", "rddm"), default="cfm")
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
