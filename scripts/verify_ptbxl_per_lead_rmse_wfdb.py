"""Rebuild every PTB-XL fold-10 target with WFDB and verify per-lead RMSE."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
TARGET_INDICES = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
TARGET_LEADS = tuple(LEADS[index] for index in TARGET_INDICES)
MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
SAMPLES = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_first_window(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.asarray(waveform[:SAMPLES], dtype=np.float32)
    minima = window.min(axis=0)
    ranges = window.max(axis=0) - minima
    if np.any(ranges <= 0) or not np.all(np.isfinite(ranges)):
        raise ValueError("WFDB reconstruction contains a constant/non-finite lead")
    normalized = 2.0 * (window - minima) / ranges - 1.0
    return normalized.astype(np.float32), minima, ranges


def _rmse_from_sse(sse: np.ndarray, observations: int) -> np.ndarray:
    if observations <= 0 or np.any(np.asarray(sse) < 0):
        raise ValueError("invalid SSE accumulator")
    return np.sqrt(np.asarray(sse, dtype=np.float64) / observations)


def run(args: argparse.Namespace) -> Path:
    source = args.source_root.resolve()
    preprocessed = args.preprocessed_dir.resolve()
    evaluation = args.evaluation_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(source / "ptbxl_database.csv").set_index("ecg_id")
    record_ids = np.load(preprocessed / "record_ids_test.npy", allow_pickle=False)
    stored = np.load(preprocessed / "X_test_resampled.npy", mmap_mode="r", allow_pickle=False)
    stored_minima = np.load(preprocessed / "record_minima_test.npy", mmap_mode="r", allow_pickle=False)
    stored_ranges = np.load(preprocessed / "record_ranges_test.npy", mmap_mode="r", allow_pickle=False)
    paired_file = np.load(evaluation / "paired_reference.npz", allow_pickle=False)
    paired_targets = paired_file["targets"]
    paired_ids = paired_file["record_ids"]
    if not np.array_equal(record_ids.astype(str), paired_ids.astype(str)):
        raise ValueError("evaluation rows do not match preprocessed fold-10 record IDs")
    if paired_targets.shape != (len(record_ids), len(TARGET_LEADS), SAMPLES):
        raise ValueError("paired target array has the wrong shape")
    predictions = {
        model: np.load(evaluation / f"{model}_predictions.npy", mmap_mode="r", allow_pickle=False)
        for model in MODELS
    }
    if any(values.shape != paired_targets.shape for values in predictions.values()):
        raise ValueError("prediction arrays do not match the paired targets")

    normalized_sse = {model: np.zeros(len(TARGET_LEADS), dtype=np.float64) for model in MODELS}
    physical_sse = {model: np.zeros(len(TARGET_LEADS), dtype=np.float64) for model in MODELS}
    record_normalized_rmse = {model: [] for model in MODELS}
    maximum_errors = {
        "wfdb_resampled_vs_preprocessed_mV": 0.0,
        "wfdb_minima_vs_sidecar_mV": 0.0,
        "wfdb_ranges_vs_sidecar_mV": 0.0,
        "wfdb_normalized_vs_paired_target": 0.0,
    }
    sampling_rates: set[float] = set()
    units: set[str] = set()
    lead_orders: set[tuple[str, ...]] = set()

    for index, record_id in enumerate(record_ids):
        row = metadata.loc[int(record_id)]
        signal, fields = wfdb.rdsamp(str(source / str(row.filename_hr)))
        sampling_rates.add(float(fields["fs"]))
        units.update(str(value) for value in fields["units"])
        lead_orders.add(tuple(str(value) for value in fields["sig_name"]))
        if float(fields["fs"]) != 500.0:
            raise ValueError(f"record {record_id} is not 500 Hz")
        if tuple(str(value).upper() for value in fields["sig_name"]) != tuple(
            lead.upper() for lead in LEADS
        ):
            raise ValueError(f"record {record_id} has an unexpected lead order")
        if set(str(value) for value in fields["units"]) != {"mV"}:
            raise ValueError(f"record {record_id} does not use mV")
        resampled = resample_poly(signal, 128, 500, axis=0, padtype="line").astype(np.float32)
        normalized, minima, ranges = _normalize_first_window(resampled)
        targets = normalized[:, TARGET_INDICES].T
        target_ranges = ranges[np.asarray(TARGET_INDICES)]
        maximum_errors["wfdb_resampled_vs_preprocessed_mV"] = max(
            maximum_errors["wfdb_resampled_vs_preprocessed_mV"],
            float(np.max(np.abs(resampled - stored[index]))),
        )
        maximum_errors["wfdb_minima_vs_sidecar_mV"] = max(
            maximum_errors["wfdb_minima_vs_sidecar_mV"],
            float(np.max(np.abs(minima - stored_minima[index]))),
        )
        maximum_errors["wfdb_ranges_vs_sidecar_mV"] = max(
            maximum_errors["wfdb_ranges_vs_sidecar_mV"],
            float(np.max(np.abs(ranges - stored_ranges[index]))),
        )
        maximum_errors["wfdb_normalized_vs_paired_target"] = max(
            maximum_errors["wfdb_normalized_vs_paired_target"],
            float(np.max(np.abs(targets - paired_targets[index]))),
        )
        for model, values in predictions.items():
            error = values[index].astype(np.float64) - targets.astype(np.float64)
            normalized_sse[model] += np.sum(error**2, axis=1)
            physical_error = error * target_ranges[:, None].astype(np.float64) / 2.0
            physical_sse[model] += np.sum(physical_error**2, axis=1)
            record_normalized_rmse[model].append(float(np.sqrt(np.mean(error**2))))
        if (index + 1) % 250 == 0 or index + 1 == len(record_ids):
            print(f"WFDB verified {index + 1}/{len(record_ids)} records", flush=True)

    frozen_summary = json.loads((evaluation / "waveform_summary.json").read_text(encoding="utf-8"))
    per_lead_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    observations_per_lead = len(record_ids) * SAMPLES
    for model in MODELS:
        normalized_rmse = _rmse_from_sse(normalized_sse[model], observations_per_lead)
        physical_rmse = _rmse_from_sse(physical_sse[model], observations_per_lead)
        frozen = frozen_summary["models"][model]
        for lead_index, lead in enumerate(TARGET_LEADS):
            frozen_lead = float(frozen["per_lead"][lead]["rmse"])
            per_lead_rows.append(
                {
                    "model": model,
                    "lead": lead,
                    "wfdb_normalized_rmse": float(normalized_rmse[lead_index]),
                    "frozen_evaluator_normalized_rmse": frozen_lead,
                    "absolute_difference": abs(float(normalized_rmse[lead_index]) - frozen_lead),
                    "oracle_inverse_physical_rmse_mV": float(physical_rmse[lead_index]),
                    "records": int(len(record_ids)),
                    "samples_per_record": SAMPLES,
                }
            )
        global_normalized = float(np.sqrt(np.sum(normalized_sse[model]) / (observations_per_lead * len(TARGET_LEADS))))
        global_physical = float(np.sqrt(np.sum(physical_sse[model]) / (observations_per_lead * len(TARGET_LEADS))))
        aggregate_rows.append(
            {
                "model": model,
                "wfdb_global_normalized_rmse": global_normalized,
                "frozen_global_normalized_rmse": float(frozen["rmse"]),
                "absolute_difference": abs(global_normalized - float(frozen["rmse"])),
                "macro_lead_normalized_rmse": float(np.mean(normalized_rmse)),
                "mean_per_record_normalized_rmse": float(np.mean(record_normalized_rmse[model])),
                "wfdb_global_oracle_inverse_rmse_mV": global_physical,
                "macro_lead_oracle_inverse_rmse_mV": float(np.mean(physical_rmse)),
            }
        )

    with (output / "per_lead_rmse_wfdb.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_lead_rows[0]))
        writer.writeheader()
        writer.writerows(per_lead_rows)
    with (output / "rmse_aggregation_wfdb.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    result = {
        "schema_version": 1,
        "status": "completed",
        "dataset": "PTB-XL 1.0.1",
        "split": "official_fold_10",
        "records": int(len(record_ids)),
        "target_leads": list(TARGET_LEADS),
        "wfdb_version": wfdb.__version__,
        "wfdb_observed_sampling_rates_hz": sorted(sampling_rates),
        "wfdb_observed_units": sorted(units),
        "wfdb_observed_lead_orders": [list(value) for value in sorted(lead_orders)],
        "reference_reconstruction_max_abs_errors": maximum_errors,
        "definitions": {
            "primary_normalized_rmse": "sqrt(mean((prediction-target)^2)) over all selected records and 512 samples in one lead after independent per-record/per-lead min-max scaling to [-1,1]",
            "global_normalized_rmse": "sqrt(mean squared error) over records x 11 target leads x 512 samples; not the arithmetic mean of lead RMSEs",
            "oracle_inverse_physical_rmse_mV": "prediction and target mapped to mV with each held-out target record/lead min and range; scale-assisted audit only",
            "wfdb_scope": "WFDB reads physical PTB-XL waveforms; RMSE is computed explicitly because WFDB does not provide a generic waveform RMSE metric",
        },
        "inputs": {
            "metadata_sha256": _sha256(source / "ptbxl_database.csv"),
            "paired_reference_sha256": _sha256(evaluation / "paired_reference.npz"),
            **{
                f"{model}_predictions_sha256": _sha256(evaluation / f"{model}_predictions.npy")
                for model in MODELS
            },
        },
        "aggregate_results": aggregate_rows,
        "maximum_per_lead_rmse_difference_vs_frozen_evaluator": max(
            float(row["absolute_difference"]) for row in per_lead_rows
        ),
        "claim_boundary": "The mV values use held-out target scaling coefficients and are not deployable generation metrics.",
    }
    (output / "verification_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate_results"], indent=2), flush=True)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--preprocessed_dir", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
