#!/usr/bin/env python3
"""Aggregate epoch-200 random-window RCFM-OT waveform metrics across seeds.

PTB-XL and CPSC2018 consume raw simultaneous-ECG summaries.  MIMIC-AFib,
WESAD, and mmECG consume only the explicitly target-informed, +/-16-sample
oracle-aligned summaries evaluated on the common 480-sample support.
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

from scripts.recalculate_legacy_batch_fd import low_rank_literal_fd


SEEDS = (31, 32, 33)
METRICS = ("rmse", "mae", "waveform_fd", "pearson_r_median")
DATASET_SPECS = {
    "ptbxl": {
        "path": "evaluation/ptbxl_random_window_rcfm_ot_e200_s{seed}_raw_v1/waveform_summary.json",
        "node": ("models", "rcfm_ot"),
        "phase_mode": "raw_full_512",
        "fd": "waveform_fd_macro_lead",
        "pearson": "per_record_pearson_median",
        "waveforms": "evaluation/ptbxl_random_window_rcfm_ot_e200_s{seed}_raw_v1",
        "waveform_kind": "raw_npy",
    },
    "cpsc2018": {
        "path": "evaluation/cpsc2018_random_window_rcfm_ot_e200_s{seed}_raw_v1/waveform_summary.json",
        "node": ("models", "rcfm_ot"),
        "phase_mode": "raw_full_512",
        "fd": "waveform_fd_macro_lead",
        "pearson": "per_record_pearson_median",
        "waveforms": "evaluation/cpsc2018_random_window_rcfm_ot_e200_s{seed}_raw_v1",
        "waveform_kind": "raw_npy",
    },
    "mimic_afib": {
        "path": "clinical/mimic_afib_random_window_rcfm_ot_e200_s{seed}_phase_clinical_v1/summary.json",
        "node": ("waveform", "oracle_aligned_fixed_support_480"),
        "phase_mode": "oracle_aligned_fixed_480",
        "fd": "waveform_fd",
        "pearson": ("per_record_pearson", "median"),
        "waveforms": "clinical/mimic_afib_random_window_rcfm_ot_e200_s{seed}_phase_clinical_v1/phase_corrected_predictions.npz",
        "waveform_kind": "phase_npz",
    },
    "wesad": {
        "path": "evaluation/wesad_random_window_record_minmax_rcfm_ot_e200_s{seed}_phase_v1/waveform_phase_summary.json",
        "node": ("oracle_aligned_fixed_480_samples",),
        "phase_mode": "oracle_aligned_fixed_480",
        "fd": "waveform_fd",
        "pearson": "per_window_pearson_median",
        "waveforms": "evaluation/wesad_random_window_record_minmax_rcfm_ot_e200_s{seed}_phase_v1/phase_predictions_maxlag16.npz",
        "waveform_kind": "phase_npz",
    },
    "mmecg": {
        "path": "evaluation/mmecg_random_window_rcfm_ot_e200_s{seed}_phase_v1/waveform_phase_summary.json",
        "node": ("oracle_aligned_fixed_480_samples",),
        "phase_mode": "oracle_aligned_fixed_480",
        "fd": "waveform_fd",
        "pearson": "per_window_pearson_median",
        "waveforms": "evaluation/mmecg_random_window_rcfm_ot_e200_s{seed}_phase_v1/phase_predictions_maxlag16.npz",
        "waveform_kind": "phase_npz",
    },
}


def _nested(value: object, keys: tuple[str, ...]) -> object:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def extract_metrics(payload: dict[str, object], spec: dict[str, object]) -> dict[str, float]:
    node = _nested(payload, spec["node"])  # type: ignore[arg-type]
    if not isinstance(node, dict):
        raise TypeError("metric node must be an object")
    pearson_key = spec["pearson"]
    pearson = _nested(node, pearson_key) if isinstance(pearson_key, tuple) else node[pearson_key]
    values = {
        "rmse": float(node["rmse"]),
        "mae": float(node["mae"]),
        "waveform_fd": float(node[spec["fd"]]),
        "pearson_r_median": float(pearson),
    }
    if not all(np.isfinite(list(values.values()))):
        raise ValueError("all waveform metrics must be finite")
    return values


def calculate_batch_wfd(
    runs_root: Path, dataset: str, seed: int, batch_size: int
) -> tuple[float, list[dict[str, object]]]:
    """Return macro-lead mean of contiguous per-batch CAT-compatible wFD."""

    spec = DATASET_SPECS[dataset]
    source = runs_root / str(spec["waveforms"]).format(seed=seed)
    if spec["waveform_kind"] == "raw_npy":
        with np.load(source / "paired_reference.npz", allow_pickle=False) as artifact:
            reference = np.asarray(artifact["targets"], dtype=np.float64)
        generated = np.asarray(np.load(source / "rcfm_ot_predictions.npy", mmap_mode="r"), dtype=np.float64)
    else:
        with np.load(source, allow_pickle=False) as artifact:
            reference = np.asarray(artifact["targets"], dtype=np.float64)
            generated = np.asarray(artifact["oracle_aligned_predictions"], dtype=np.float64)
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError(f"waveform shape mismatch for {dataset}/seed-{seed}")
    rows: list[dict[str, object]] = []
    lead_means = []
    for lead in range(reference.shape[1]):
        values = []
        for batch_index, start in enumerate(range(0, len(reference), batch_size)):
            stop = min(start + batch_size, len(reference))
            if stop - start < 2:
                continue
            value = low_rank_literal_fd(reference[start:stop, lead], generated[start:stop, lead])
            values.append(value)
            rows.append({
                "dataset": dataset, "seed": seed, "lead_index": lead,
                "batch_index": batch_index, "record_start": start,
                "record_stop_exclusive": stop, "records": stop - start,
                "waveform_fd": value,
            })
        if not values:
            raise ValueError(f"no valid wFD batches for {dataset}/seed-{seed}/lead-{lead}")
        lead_means.append(float(np.mean(values)))
    return float(np.mean(lead_means)), rows


def aggregate_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for dataset in DATASET_SPECS:
        selected = [row for row in rows if row["dataset"] == dataset]
        if [row["seed"] for row in selected] != list(SEEDS):
            raise ValueError(f"expected seeds {SEEDS} for {dataset}")
        result: dict[str, object] = {
            "dataset": dataset,
            "model": "RCFM-OT",
            "seeds": 3,
            "phase_mode": selected[0]["phase_mode"],
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_sample_sd"] = float(np.std(values, ddof=1))
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    runs_root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    inputs = []
    for dataset, spec in DATASET_SPECS.items():
        for seed in SEEDS:
            source = runs_root / str(spec["path"]).format(seed=seed)
            waveform_source = runs_root / str(spec["waveforms"]).format(seed=seed)
            payload = json.loads(source.read_text(encoding="utf-8"))
            metrics = extract_metrics(payload, spec)
            metrics["waveform_fd"], details = calculate_batch_wfd(
                runs_root, dataset, seed, args.batch_size
            )
            batch_rows.extend(details)
            seed_rows.append({
                "dataset": dataset,
                "model": "RCFM-OT",
                "seed": seed,
                "phase_mode": spec["phase_mode"],
                **metrics,
                "source_artifact": str(source),
                "waveform_fd_source_artifact": str(waveform_source),
            })
            inputs.append({"summary": str(source), "waveforms": str(waveform_source)})
    summary_rows = aggregate_seed_rows(seed_rows)
    _write_csv(output / "per_seed_waveform_metrics.csv", seed_rows)
    _write_csv(output / "per_batch_waveform_fd.csv", batch_rows)
    _write_csv(output / "waveform_mean_sd.csv", summary_rows)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "model": "RCFM-OT",
        "training_seeds": list(SEEDS),
        "metrics": {
            "rmse_mae_domain": "normalized waveform domain",
            "waveform_fd": f"CAT-compatible arithmetic mean of contiguous batches of at most {args.batch_size} records; macro target-lead mean for PTB-XL/CPSC2018; single target otherwise",
            "waveform_fd_formula": "||mu_r-mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r*Sigma_g)^(1/2)); NumPy sample covariance (ddof=1)",
            "pearson_r_median": "median per-record Pearson for PTB-XL/CPSC2018/MIMIC-AFib; median per-window Pearson for WESAD/mmECG",
            "uncertainty": "sample SD across independently trained seeds",
        },
        "wfd_batch_rule": "saved deterministic row order; final batch retained when it has at least two records; unweighted mean over batches and then unweighted mean over target leads",
        "phase_rule": "raw full 512 for PTB-XL/CPSC2018; target-informed +/-16-sample oracle alignment on common 480 support for MIMIC-AFib/WESAD/mmECG",
        "claim_boundary": "random-window validation splits permit identity overlap; phase-corrected rows are morphology diagnostics and are not deployable or subject-independent estimates",
        "rows": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(args.command),
        "inputs": inputs,
        "outputs": ["per_seed_waveform_metrics.csv", "per_batch_waveform_fd.csv", "waveform_mean_sd.csv", "summary.json"],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    args.command = list(__import__("sys").argv)
    return args


if __name__ == "__main__":
    run(parse_args())
