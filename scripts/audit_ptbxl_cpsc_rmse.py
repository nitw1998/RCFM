#!/usr/bin/env python3
"""Audit PTB-XL/CPSC zero leads, labels, and independently recomputed RMSE."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
from sklearn.metrics import root_mean_squared_error


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
TARGET_INDICES = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
TARGET_LEADS = tuple(LEADS[index] for index in TARGET_INDICES)
CPSC_LABELS = (
    "normal", "atrial_fibrillation", "first_degree_av_block", "left_bundle_branch_block",
    "right_bundle_branch_block", "premature_atrial_contraction",
    "premature_ventricular_contraction", "st_depression", "st_elevation",
)


def parse_args() -> argparse.Namespace:
    # The executable repository is reached through a workspace symlink, so resolving
    # __file__ would lose the coordination-workspace location.
    workspace = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--ptbxl-source", type=Path, required=True)
    parser.add_argument("--cpsc-source", type=Path, required=True)
    parser.add_argument("--source-check-records", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "runs/evaluation/ptbxl_cpsc_rmse_source_audit_v1",
    )
    return parser.parse_args()


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _loader_reference(raw: np.ndarray) -> np.ndarray:
    selected = np.transpose(np.asarray(raw[:, :512, TARGET_INDICES], dtype=np.float32), (0, 2, 1))
    minima = selected.min(axis=-1)
    ranges = selected.max(axis=-1) - minima
    if np.any(ranges <= 0):
        raise ValueError("reference contains a constant target lead")
    return (2.0 * (selected - minima[..., None]) / ranges[..., None] - 1.0).astype(np.float32)


def _source_qc(raw: np.ndarray, record_ids: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    view = np.asarray(raw[:, :512], dtype=np.float32)
    ranges = np.ptp(view, axis=1)
    rows: list[dict[str, Any]] = []
    for threshold in (0.0, 1e-6, 1e-4, 1e-3):
        hit = ranges <= threshold
        rows.append({
            "threshold": threshold,
            "lead_windows": int(hit.sum()),
            "records": int(np.any(hit, axis=1).sum()),
        })
    per_lead = {}
    for index, lead in enumerate(LEADS):
        values = ranges[:, index]
        per_lead[lead] = {
            "minimum": float(values.min()),
            "q001": float(np.quantile(values, 0.001)),
            "q01": float(np.quantile(values, 0.01)),
            "median": float(np.median(values)),
        }
    near = []
    for record_index, lead_index in np.argwhere(ranges <= 1e-3):
        near.append({
            "record_id": str(record_ids[record_index]),
            "lead": LEADS[lead_index],
            "range": float(ranges[record_index, lead_index]),
        })
    return {
        "shape": list(view.shape),
        "nonfinite_samples": int(view.size - np.isfinite(view).sum()),
        "threshold_counts": rows,
        "range_quantiles_by_lead": per_lead,
    }, near


def _metric_rows(dataset: str, evaluation_dir: Path, reference: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    summary_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    per_record: dict[str, np.ndarray] = {}
    for path in sorted(evaluation_dir.glob("*_predictions.npy")):
        model = path.name.removesuffix("_predictions.npy")
        prediction = np.load(path, allow_pickle=False).astype(np.float64)
        if prediction.shape != reference.shape or not np.all(np.isfinite(prediction)):
            raise ValueError(f"invalid prediction artifact: {path}")
        error = prediction - reference
        lead_rmse = np.sqrt(np.mean(error**2, axis=(0, 2)))
        record_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
        prediction_ranges = np.ptp(prediction, axis=2)
        explicit = float(np.sqrt(np.mean(error**2)))
        sklearn_value = float(root_mean_squared_error(reference.reshape(-1), prediction.reshape(-1)))
        if not np.isclose(explicit, sklearn_value, rtol=0.0, atol=1e-15):
            raise AssertionError("explicit and scikit-learn RMSE disagree")
        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "global_rmse": explicit,
            "sklearn_rmse": sklearn_value,
            "macro_lead_rmse": float(lead_rmse.mean()),
            "mean_per_record_rmse": float(record_rmse.mean()),
            "median_per_record_rmse": float(np.median(record_rmse)),
            "p95_per_record_rmse": float(np.quantile(record_rmse, 0.95)),
            "p99_per_record_rmse": float(np.quantile(record_rmse, 0.99)),
            "constant_prediction_leads": int((prediction_ranges == 0).sum()),
            "prediction_leads_range_below_1e-4": int((prediction_ranges < 1e-4).sum()),
        })
        lead_rows.extend(
            {"dataset": dataset, "model": model, "lead": lead, "rmse": float(lead_rmse[index])}
            for index, lead in enumerate(TARGET_LEADS)
        )
        per_record[model] = record_rmse
    return summary_rows, lead_rows, per_record


def _top_error_rows(dataset: str, ids: np.ndarray, errors: dict[str, np.ndarray], labels: list[list[str]]) -> list[dict[str, Any]]:
    rows = []
    for model, values in errors.items():
        for rank, index in enumerate(np.argsort(values)[-10:][::-1], start=1):
            rows.append({
                "dataset": dataset, "model": model, "rank": rank,
                "record_id": str(ids[index]), "rmse": float(values[index]),
                "labels": ";".join(labels[index]),
            })
    return rows


def _label_rows(dataset: str, errors: dict[str, np.ndarray], labels: list[list[str]]) -> list[dict[str, Any]]:
    names = sorted({name for record in labels for name in record})
    rows = []
    for model, values in errors.items():
        for name in names:
            selected = np.asarray([name in record for record in labels])
            if not selected.any():
                continue
            rows.append({
                "dataset": dataset, "model": model, "label": name, "records": int(selected.sum()),
                "mean_per_record_rmse": float(values[selected].mean()),
                "median_per_record_rmse": float(np.median(values[selected])),
            })
    return rows


def _selection(count: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=int))


def _verify_ptbxl_wfdb(
    source: Path, preprocess_dir: Path, evaluation_dir: Path, maximum: int,
) -> dict[str, Any]:
    import wfdb

    database = pd.read_csv(source / "ptbxl_database.csv").set_index("ecg_id")
    ids = np.load(preprocess_dir / "record_ids_test.npy", allow_pickle=False)
    stored = np.load(preprocess_dir / "X_test_resampled.npy", mmap_mode="r", allow_pickle=False)
    paired = np.load(evaluation_dir / "paired_reference.npz", allow_pickle=False)["targets"]
    selected = _selection(len(ids), maximum)
    reconstructed = []
    maximum_source_error = 0.0
    metadata = {"sampling_rates": set(), "units": set(), "lead_orders": set()}
    for index in selected:
        row = database.loc[int(ids[index])]
        signal, fields = wfdb.rdsamp(str(source / row.filename_hr))
        waveform = resample_poly(signal, 128, 500, axis=0, padtype="line").astype(np.float32)
        maximum_source_error = max(maximum_source_error, float(np.max(np.abs(waveform - stored[index]))))
        reconstructed.append(waveform)
        metadata["sampling_rates"].add(float(fields["fs"]))
        metadata["units"].update(fields["units"])
        metadata["lead_orders"].add(tuple(fields["sig_name"]))
    wfdb_reference = _loader_reference(np.stack(reconstructed))
    paired_error = float(np.max(np.abs(wfdb_reference - paired[selected])))
    model_checks = {}
    for path in sorted(evaluation_dir.glob("*_predictions.npy")):
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)[selected].astype(np.float64)
        wfdb_rmse = float(root_mean_squared_error(wfdb_reference.reshape(-1), prediction.reshape(-1)))
        paired_rmse = float(root_mean_squared_error(paired[selected].reshape(-1), prediction.reshape(-1)))
        model_checks[path.name.removesuffix("_predictions.npy")] = {
            "wfdb_reference_subset_rmse": wfdb_rmse,
            "paired_reference_subset_rmse": paired_rmse,
            "absolute_difference": abs(wfdb_rmse - paired_rmse),
        }
    return {
        "wfdb_version": wfdb.__version__,
        "records_checked": int(len(selected)),
        "record_ids": [str(value) for value in ids[selected]],
        "source_to_preprocessed_max_abs_error": maximum_source_error,
        "wfdb_reference_to_paired_max_abs_error": paired_error,
        "sampling_rates": sorted(metadata["sampling_rates"]),
        "units": sorted(metadata["units"]),
        "lead_orders": [list(value) for value in sorted(metadata["lead_orders"])],
        "model_subset_checks": model_checks,
        "scope_note": "WFDB reads PTB-XL references; RMSE is then independently computed with scikit-learn because WFDB has no generic waveform RMSE API.",
    }


def _verify_cpsc_source(source: Path, preprocess_dir: Path, maximum: int) -> dict[str, Any]:
    records = pd.read_csv(preprocess_dir / "records_val.csv")
    stored = np.load(preprocess_dir / "X_val_resampled.npy", mmap_mode="r", allow_pickle=False)
    selected = _selection(len(records), maximum)
    maximum_error = 0.0
    for index in selected:
        row = records.iloc[index]
        payload = loadmat(source / row.source_file, squeeze_me=True, struct_as_record=False, variable_names=["ECG"])
        signal = np.asarray(payload["ECG"].data, dtype=np.float64)[:, :2000]
        waveform = resample_poly(signal, 128, 500, axis=1, padtype="line").T.astype(np.float32)
        maximum_error = max(maximum_error, float(np.max(np.abs(waveform - stored[index]))))
    return {
        "reader": "scipy.io.loadmat (CPSC source is MATLAB, not WFDB)",
        "records_checked": int(len(selected)),
        "record_ids": records.iloc[selected].record_id.astype(str).tolist(),
        "source_to_preprocessed_max_abs_error": maximum_error,
    }


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    if not (workspace / "repo").exists() or not (workspace / "runs").is_dir():
        raise ValueError("--workspace must be the RCFM coordination workspace root")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configurations = {
        "ptbxl": {
            "preprocess": workspace / "runs/preprocessing/ptbxl_official_minmax_v1/PTBXL",
            "evaluation": workspace / "runs/evaluation/ptbxl_fourway_fold10_raw_seed2025_v1",
            "split": "test",
        },
        "cpsc2018": {
            "preprocess": workspace / "runs/preprocessing/cpsc2018_multilead_qc_v3/CPSC2018",
            "evaluation": workspace / "runs/evaluation/cpsc2018_fourway_raw_seed2025_v1",
            "split": "val",
        },
    }
    summary: dict[str, Any] = {
        "rmse_definitions": {
            "global_rmse": "sqrt(mean(error^2)) over records x 11 leads x 512 samples",
            "macro_lead_rmse": "arithmetic mean of 11 independently computed lead RMSE values",
            "mean_per_record_rmse": "arithmetic mean of per-record RMSE values",
        },
        "datasets": {},
    }
    metric_rows: list[dict[str, Any]] = []
    lead_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    near_rows: list[dict[str, Any]] = []
    for dataset, config in configurations.items():
        preprocess_dir, evaluation_dir, split = config["preprocess"], config["evaluation"], config["split"]
        raw = np.load(preprocess_dir / f"X_{split}_resampled.npy", mmap_mode="r", allow_pickle=False)
        paired = np.load(evaluation_dir / "paired_reference.npz", allow_pickle=False)
        ids = paired["record_ids"].astype(str)
        reconstructed = _loader_reference(raw)
        qc, near = _source_qc(raw, ids)
        for row in near:
            near_rows.append({"dataset": dataset, **row})
        metrics, leads, errors = _metric_rows(dataset, evaluation_dir, paired["targets"].astype(np.float64))
        metric_rows.extend(metrics)
        lead_rows.extend(leads)
        if dataset == "ptbxl":
            database = pd.read_csv(args.ptbxl_source / "ptbxl_database.csv").set_index("ecg_id")
            code_map = database.scp_codes.map(ast.literal_eval)
            labels = [sorted(code_map.loc[int(value)]) for value in ids]
            vf_counts = {name: sum(name in item for item in labels) for name in ("VFIB", "VTACH", "AFIB", "AFLT")}
            source_check = _verify_ptbxl_wfdb(
                args.ptbxl_source, preprocess_dir, evaluation_dir, args.source_check_records
            )
        else:
            encoded = np.load(preprocess_dir / "labels_val.npy", allow_pickle=False).astype(bool)
            labels = [[CPSC_LABELS[index] for index in np.flatnonzero(row)] for row in encoded]
            vf_counts = {"ventricular_fibrillation": 0, "note": "not an official CPSC2018 class"}
            source_check = _verify_cpsc_source(args.cpsc_source, preprocess_dir, args.source_check_records)
        label_rows.extend(_label_rows(dataset, errors, labels))
        top_rows.extend(_top_error_rows(dataset, ids, errors, labels))
        summary["datasets"][dataset] = {
            "evaluated_split": split,
            "records": int(len(ids)),
            "paired_reference_to_loader_reconstruction_max_abs_error": float(
                np.max(np.abs(paired["targets"] - reconstructed))
            ),
            "source_qc": qc,
            "ventricular_and_atrial_rhythm_counts": vf_counts,
            "source_reconstruction_check": source_check,
        }
    _write_csv(output / "rmse_aggregation.csv", metric_rows)
    _write_csv(output / "per_lead_rmse.csv", lead_rows)
    _write_csv(output / "label_group_rmse.csv", label_rows)
    _write_csv(output / "top_error_records.csv", top_rows)
    _write_csv(output / "near_constant_source_leads.csv", near_rows)
    _json(output / "audit_summary.json", summary)
    print(json.dumps({"output_dir": str(output), "metrics": metric_rows}, indent=2))


if __name__ == "__main__":
    main()
