#!/usr/bin/env python3
"""Compare three-seed epoch-400 RDDM with matched RCFM-OT endpoints.

wFD follows the historical CAT-compatible convention: calculate it for each
contiguous batch, average batches without size weighting, then average target
leads without weighting. MIMIC-AFib, WESAD, and mmECG are target-informed
oracle aligned to the common central 480-sample support before every metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cpsc2018_multiseed import _holm_adjust, _paired_test
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_ptbxl_fourway import _metric_summary
from scripts.recalculate_legacy_batch_fd import low_rank_literal_fd
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


SEEDS = (31, 32, 33)
METRICS = ("rmse", "mae", "waveform_fd", "pearson_r_median")
DATASET_SPECS = {
    "ptbxl": {
        "rddm": "evaluation/ptbxl_random_window_rddm_e400_s{seed}_raw_v1",
        "rcfm_ot": "evaluation/ptbxl_random_window_rcfm_ot_e200_s{seed}_raw_v1",
        "phase_mode": "raw_full_512", "sampling_rate": 128,
    },
    "cpsc2018": {
        "rddm": "evaluation/cpsc2018_random_window_rddm_e400_s{seed}_raw_v1",
        "rcfm_ot": "evaluation/cpsc2018_random_window_rcfm_ot_e200_s{seed}_raw_v1",
        "phase_mode": "raw_full_512", "sampling_rate": 128,
    },
    "mimic_afib": {
        "rddm": "evaluation/mimic_afib_random_window_rddm_e400_s{seed}_raw_v1",
        "rcfm_ot": "clinical/mimic_afib_random_window_rcfm_ot_e200_s{seed}_phase_clinical_v1/phase_corrected_predictions.npz",
        "phase_mode": "oracle_aligned_fixed_480", "sampling_rate": 128,
    },
    "wesad": {
        "rddm": "evaluation/wesad_random_window_record_minmax_rddm_e400_s{seed}_raw_v1",
        "rcfm_ot": "evaluation/wesad_random_window_record_minmax_rcfm_ot_e200_s{seed}_phase_v1/phase_predictions_maxlag16.npz",
        "phase_mode": "oracle_aligned_fixed_480", "sampling_rate": 128,
    },
    "mmecg": {
        "rddm": "evaluation/mmecg_random_window_rddm_e400_s{seed}_raw_v1",
        "rcfm_ot": "evaluation/mmecg_random_window_rcfm_ot_e200_s{seed}_phase_v1/phase_predictions_maxlag16.npz",
        "phase_mode": "oracle_aligned_fixed_480", "sampling_rate": 200,
    },
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_raw_rddm(directory: Path, seed: int) -> tuple[np.ndarray, np.ndarray, Path]:
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    details = protocol.get("protocol", {})
    if (
        protocol.get("status") != "completed"
        or details.get("model") != "rddm"
        or int(details.get("training_seed", -1)) != seed
        or details.get("phase_correction_applied") is not False
        or int(protocol.get("checkpoint", {}).get("epoch", -1)) != 400
    ):
        raise ValueError(f"RDDM endpoint protocol mismatch: {directory}")
    with np.load(directory / "paired_reference.npz", allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float32)
    prediction_path = directory / "rddm_predictions.npy"
    predictions = np.asarray(np.load(prediction_path, mmap_mode="r"), dtype=np.float32)
    if targets.shape != predictions.shape or targets.ndim != 3:
        raise ValueError(f"RDDM target/prediction shape mismatch: {directory}")
    return targets, predictions, protocol_path


def _load_rcfm_ot(runs_root: Path, dataset: str, seed: int) -> tuple[np.ndarray, np.ndarray, Path]:
    spec = DATASET_SPECS[dataset]
    source = runs_root / str(spec["rcfm_ot"]).format(seed=seed)
    if dataset in {"ptbxl", "cpsc2018"}:
        with np.load(source / "paired_reference.npz", allow_pickle=False) as artifact:
            targets = np.asarray(artifact["targets"], dtype=np.float32)
        predictions = np.asarray(np.load(source / "rcfm_ot_predictions.npy", mmap_mode="r"), dtype=np.float32)
        protocol_path = source / "protocol.json"
    else:
        with np.load(source, allow_pickle=False) as artifact:
            targets = np.asarray(artifact["targets"], dtype=np.float32)
            predictions = np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float32)
        protocol_path = source.parent / "protocol.json"
    if targets.shape != predictions.shape or targets.ndim != 3:
        raise ValueError(f"RCFM-OT target/prediction shape mismatch: {source}")
    return targets, predictions, protocol_path


def _phase_correct(targets: np.ndarray, predictions: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    _, lag_values = _lag_diagnostic(targets, predictions, 16, sampling_rate)
    target_fixed, _, aligned = _fixed_support_align(
        targets, predictions, lag_values["best_lag_samples"].astype(np.int32), 16
    )
    return target_fixed, aligned


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    target_leads = tuple(f"target_{index}" for index in range(targets.shape[1]))
    summary, _ = _metric_summary(
        targets, predictions, include_fd=False, target_leads=target_leads
    )
    return {
        "rmse": float(summary["rmse"]),
        "mae": float(summary["mae"]),
        "pearson_r_median": float(summary["per_record_pearson_median"]),
    }


def _batch_wfd(
    dataset: str, model: str, seed: int, targets: np.ndarray,
    predictions: np.ndarray, batch_size: int,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    lead_means = []
    for lead in range(targets.shape[1]):
        values = []
        for batch_index, start in enumerate(range(0, len(targets), batch_size)):
            stop = min(start + batch_size, len(targets))
            if stop - start < 2:
                continue
            value = low_rank_literal_fd(
                targets[start:stop, lead], predictions[start:stop, lead]
            )
            values.append(value)
            rows.append({
                "dataset": dataset, "model": model, "seed": seed,
                "lead_index": lead, "batch_index": batch_index,
                "record_start": start, "record_stop_exclusive": stop,
                "records": stop - start, "waveform_fd": value,
            })
        if not values:
            raise ValueError(f"no valid wFD batches for {dataset}/{model}/seed-{seed}/lead-{lead}")
        lead_means.append(float(np.mean(values)))
    return float(np.mean(lead_means)), rows


def run(args: argparse.Namespace) -> Path:
    runs_root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = tuple(args.datasets)
    seed_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []
    for dataset in selected:
        spec = DATASET_SPECS[dataset]
        for seed in SEEDS:
            rddm_dir = runs_root / str(spec["rddm"]).format(seed=seed)
            rddm_target, rddm_prediction, rddm_protocol = _load_raw_rddm(rddm_dir, seed)
            rcfm_target, rcfm_prediction, rcfm_protocol = _load_rcfm_ot(runs_root, dataset, seed)
            if spec["phase_mode"] == "oracle_aligned_fixed_480":
                rddm_target, rddm_prediction = _phase_correct(
                    rddm_target, rddm_prediction, int(spec["sampling_rate"])
                )
            if rddm_target.shape != rcfm_target.shape or not np.array_equal(rddm_target, rcfm_target):
                raise ValueError(f"RDDM and RCFM-OT targets are not exactly paired for {dataset}/seed-{seed}")
            if seed > SEEDS[0]:
                previous = next(
                    row for row in inputs
                    if row["dataset"] == dataset and row["seed"] == SEEDS[0]
                )
                if previous["target_array_sha256"] != _array_sha256(rddm_target):
                    raise ValueError(f"target rows differ across seeds for {dataset}")
            for model, predictions in (("RDDM", rddm_prediction), ("RCFM-OT", rcfm_prediction)):
                metrics = _metrics(rddm_target, predictions)
                metrics["waveform_fd"], rows = _batch_wfd(
                    dataset, model, seed, rddm_target, predictions, args.batch_size
                )
                batch_rows.extend(rows)
                seed_rows.append({
                    "dataset": dataset, "model": model, "seed": seed,
                    "phase_mode": spec["phase_mode"], **metrics,
                })
            inputs.append({
                "dataset": dataset, "seed": seed,
                "rddm_protocol": str(rddm_protocol),
                "rddm_protocol_sha256": _sha256(rddm_protocol),
                "rcfm_ot_protocol": str(rcfm_protocol),
                "rcfm_ot_protocol_sha256": _sha256(rcfm_protocol),
                "target_array_sha256": _array_sha256(rddm_target),
            })

    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    raw_p: dict[str, float] = {}
    for dataset in selected:
        for model in ("RDDM", "RCFM-OT"):
            rows = [row for row in seed_rows if row["dataset"] == dataset and row["model"] == model]
            for metric in METRICS:
                values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                summary_rows.append({
                    "dataset": dataset, "model": model, "metric": metric,
                    "phase_mode": rows[0]["phase_mode"], "n_training_seeds": len(values),
                    "mean": float(values.mean()), "sample_sd_ddof1": float(values.std(ddof=1)),
                })
        for metric in METRICS:
            rddm = np.asarray([
                row[metric] for row in seed_rows
                if row["dataset"] == dataset and row["model"] == "RDDM"
            ], dtype=np.float64)
            rcfm_ot = np.asarray([
                row[metric] for row in seed_rows
                if row["dataset"] == dataset and row["model"] == "RCFM-OT"
            ], dtype=np.float64)
            test = _paired_test(rddm, rcfm_ot)
            row = {
                "dataset": dataset, "metric": metric,
                "comparison": "RDDM_minus_RCFM-OT",
                "lower_is_better": metric != "pearson_r_median", **test,
            }
            test_rows.append(row)
            raw_p[f"{dataset}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p)
    for row in test_rows:
        row["holm_adjusted_p_value_all_dataset_metrics"] = adjusted[
            f"{row['dataset']}/{row['metric']}"
        ]

    paths = {
        "per_seed": output / "per_seed_metrics.csv",
        "summary": output / "model_mean_sd.csv",
        "tests": output / "rddm_vs_rcfm_ot_paired_tests.csv",
        "batches": output / "per_batch_waveform_fd.csv",
    }
    for key, rows in (
        ("per_seed", seed_rows), ("summary", summary_rows),
        ("tests", test_rows), ("batches", batch_rows),
    ):
        _write_csv(paths[key], rows)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1, "status": "completed", "datasets": list(selected),
        "models": ["RDDM", "RCFM-OT"], "training_seeds": list(SEEDS),
        "mean_sd_definition": "mean and sample SD (ddof=1) across independently trained seeds",
        "paired_test_definition": "two-sided paired t-test and exact two-sided sign-flip test over matched training seeds",
        "multiplicity": f"Holm adjustment across {len(selected) * len(METRICS)} dataset-metric paired t-tests",
        "wfd_batch_rule": (
            f"contiguous deterministic batches of at most {args.batch_size}; final batch retained "
            "when n>=2; unweighted mean over batches then unweighted mean over target leads"
        ),
        "phase_rule": "raw 512 for PTB-XL/CPSC2018; target-informed +/-16-sample oracle alignment on central 480 support for MIMIC-AFib/WESAD/mmECG",
        "claim_boundary": "Random-window identity overlap remains; phase-corrected results are oracle morphology diagnostics. With n=3 seeds the exact sign-flip p-value cannot be below 0.25.",
        "model_mean_sd": summary_rows, "paired_tests": test_rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "inputs": inputs,
        "outputs": {path.name: _sha256(path) for path in (*paths.values(), summary_path)},
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
    )
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
