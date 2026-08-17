"""Assemble and analyze the frozen PTB-XL DiagMask comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _json, _sha256
from scripts.evaluate_ptbxl_fourway import TARGET_LEADS, _lag_diagnostic, _metric_summary
from src.rcfm.metrics.statistics import holm_adjust


MODEL_ORDER = ("cfm", "cfm_ot", "pan", "pan_ot", "diag", "diag_ot")
PRIMARY_COMPARISONS = (
    ("diag", "cfm"),
    ("diag_ot", "cfm_ot"),
    ("diag", "pan"),
    ("diag_ot", "pan_ot"),
)
METRICS = ("rmse", "mae", "pearson_r")


def _patient_means(values: np.ndarray, patient_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    valid = np.isfinite(values)
    patients, inverse = np.unique(patient_ids[valid], return_inverse=True)
    sums = np.bincount(inverse, weights=values[valid])
    counts = np.bincount(inverse)
    return patients, sums / counts


def _paired_test(
    comparison: np.ndarray,
    reference: np.ndarray,
    patient_ids: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    comparison_patients, comparison_values = _patient_means(comparison, patient_ids)
    reference_patients, reference_values = _patient_means(reference, patient_ids)
    common, comparison_index, reference_index = np.intersect1d(
        comparison_patients, reference_patients, return_indices=True
    )
    comparison_values = comparison_values[comparison_index]
    reference_values = reference_values[reference_index]
    difference = comparison_values - reference_values
    if len(common) < 2 or replicates <= 0:
        raise ValueError("paired patient inference requires at least two patients and positive replicates")
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(common), size=(replicates, len(common)))
    bootstrap = np.mean(difference[draws], axis=1)
    if np.allclose(difference, 0):
        statistic, p_value, effect = 0.0, 1.0, 0.0
    else:
        result = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(result.pvalue)
        nonzero = difference[difference != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        effect = float((np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / np.sum(ranks))
    return {
        "patients": int(len(common)),
        "difference_definition": "comparison_minus_reference_after_mean_within_patient",
        "mean_difference": float(np.mean(difference)),
        "paired_difference_std": float(np.std(difference, ddof=1)),
        "median_difference": float(np.median(difference)),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "test": "paired_wilcoxon_signed_rank",
        "statistic": statistic,
        "raw_p_value": p_value,
        "effect_size": {"name": "rank_biserial_comparison_minus_reference", "value": effect},
        "inference_scope": "fixed_training_seed_test_patient_uncertainty_only",
    }


def _load_single_prediction(directory: Path, model: str, reference_hash: str) -> tuple[Path, dict[str, object]]:
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "completed" or protocol.get("model_name") != model:
        raise ValueError(f"incomplete or mismatched {model} generation artifact")
    if protocol.get("paired_reference", {}).get("sha256") != reference_hash:
        raise ValueError(f"{model} did not use the frozen paired reference")
    prediction_path = directory / "predictions.npy"
    if _sha256(prediction_path) != protocol.get("prediction", {}).get("sha256"):
        raise ValueError(f"{model} prediction hash changed")
    return prediction_path, protocol


def _link(source: Path, destination: Path) -> None:
    relative = os.path.relpath(source.resolve(), start=destination.parent)
    destination.symlink_to(relative)


def _named_directories(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("extra predictions must use MODEL=PATH")
        model, path = value.split("=", 1)
        if not model or model in MODEL_ORDER or model in output:
            raise ValueError(f"invalid or duplicate extra model: {model!r}")
        output[model] = Path(path)
    return output


def _extra_comparisons(values: list[str], models: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    output = []
    for value in values:
        if ":" not in value:
            raise ValueError("extra comparisons must use COMPARISON:REFERENCE")
        comparison, reference = value.split(":", 1)
        if comparison not in models or reference not in models or comparison == reference:
            raise ValueError(f"invalid extra comparison: {value!r}")
        pair = (comparison, reference)
        if pair in PRIMARY_COMPARISONS or pair in output:
            raise ValueError(f"duplicate comparison: {value!r}")
        output.append(pair)
    return tuple(output)


def run(args: argparse.Namespace) -> Path:
    source_dir = args.fourway_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((source_dir / "protocol.json").read_text(encoding="utf-8"))
    if source_protocol.get("status") != "completed" or source_protocol.get("protocol", {}).get("split") != "official_fold_10":
        raise ValueError("four-way source is not the completed official fold-10 artifact")
    reference_source = source_dir / "paired_reference.npz"
    reference_hash = _sha256(reference_source)
    if reference_hash != source_protocol.get("artifacts", {}).get("paired_reference_sha256"):
        raise ValueError("four-way paired reference hash changed")
    _link(reference_source, output_dir / "paired_reference.npz")
    extra_directories = _named_directories(args.extra_prediction)
    model_order = MODEL_ORDER + tuple(extra_directories)
    primary_comparisons = PRIMARY_COMPARISONS + _extra_comparisons(
        args.extra_comparison, model_order
    )
    prediction_sources = {
        "cfm": source_dir / "cfm_predictions.npy",
        "pan": source_dir / "rcfm_predictions.npy",
        "pan_ot": source_dir / "rcfm_ot_predictions.npy",
    }
    generated_protocols = {}
    for model, directory in (("cfm_ot", args.cfm_ot_dir), ("diag", args.diag_dir), ("diag_ot", args.diag_ot_dir)):
        prediction_sources[model], generated_protocols[model] = _load_single_prediction(
            directory.resolve(), model, reference_hash
        )
    for model, directory in extra_directories.items():
        prediction_sources[model], generated_protocols[model] = _load_single_prediction(
            directory.resolve(), model, reference_hash
        )
    for model in model_order:
        _link(prediction_sources[model], output_dir / f"{model}_predictions.npy")
    with np.load(reference_source, allow_pickle=False) as artifact:
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        record_ids = np.asarray(artifact["record_ids"])
        patient_ids = np.asarray(artifact["patient_ids"])
    summaries: dict[str, object] = {}
    per_record: dict[str, Mapping[str, np.ndarray]] = {}
    lag = {}
    for model in model_order:
        prediction = np.asarray(np.load(prediction_sources[model], mmap_mode="r"))
        if prediction.shape != targets.shape or not np.all(np.isfinite(prediction)):
            raise ValueError(f"{model} prediction shape or finiteness changed")
        summaries[model], per_record[model] = _metric_summary(targets, prediction)
        lag[model] = _lag_diagnostic(targets, prediction, args.max_lag_samples)
    waveform_path = output_dir / "waveform_summary.json"
    _json(waveform_path, {"models": summaries, "lag_diagnostic": lag})
    per_record_path = output_dir / "per_record_metrics.csv"
    fields = ["record_id", "patient_id"] + [f"{model}_{metric}" for model in model_order for metric in ("rmse", "mae", "bias", "pearson_r")]
    with per_record_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record_id in enumerate(record_ids):
            row: dict[str, object] = {"record_id": str(record_id), "patient_id": str(patient_ids[index])}
            for model in model_order:
                for metric in ("rmse", "mae", "bias", "pearson_r"):
                    row[f"{model}_{metric}"] = float(per_record[model][metric][index])
            writer.writerow(row)
    per_lead_path = output_dir / "per_lead_waveform_metrics.csv"
    with per_lead_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["model", "lead", "rmse", "mae", "bias", "waveform_fd"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in model_order:
            for lead in TARGET_LEADS:
                row = summaries[model]["per_lead"][lead]
                writer.writerow({"model": model, "lead": lead, **{field: row[field] for field in fields[2:]}})
    tests = {}
    raw_p = {}
    for comparison_index, (comparison, reference) in enumerate(primary_comparisons):
        pair_name = f"{comparison}_vs_{reference}"
        tests[pair_name] = {}
        for metric_index, metric in enumerate(METRICS):
            result = _paired_test(
                per_record[comparison][metric], per_record[reference][metric], patient_ids,
                args.bootstrap_seed + comparison_index * 10 + metric_index,
                args.bootstrap_replicates,
            )
            tests[pair_name][metric] = result
            raw_p[f"{pair_name}/{metric}"] = float(result["raw_p_value"])
    adjusted = holm_adjust(raw_p)
    test_count = len(raw_p)
    for name, value in adjusted.items():
        pair_name, metric = name.split("/")
        tests[pair_name][metric][f"holm_adjusted_p_value_{test_count}_tests"] = value
    statistics_path = output_dir / "patient_paired_significance.json"
    _json(
        statistics_path,
        {
            "schema_version": 1,
            "comparisons": tests,
            "multiplicity": f"Holm adjustment jointly across {len(primary_comparisons)} pre-registered comparisons x 3 metrics",
            "waveform_fd_inference": "blocked_no_record_level_decomposition; full-test-set 11-lead macro wFD is descriptive",
            "claim_boundary": "One training seed: intervals and p-values quantify fold-10 patient sampling uncertainty, not retraining variability.",
        },
    )
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            **source_protocol["protocol"],
            "models": list(model_order),
            "same_flow_noise_across_all_models": True,
            **({"same_flow_noise_across_all_six_models": True} if len(model_order) == 6 else {}),
            "mask_used_at_inference": False,
            "ot_used_at_inference": False,
        },
        "analysis_comparisons": [list(pair) for pair in primary_comparisons],
        "source_fourway": {"path": str(source_dir), "protocol_sha256": _sha256(source_dir / "protocol.json")},
        "single_generation_protocols": {
            model: {"path": str(directory.resolve()), "protocol_sha256": _sha256(directory.resolve() / "protocol.json")}
            for model, directory in {
                "cfm_ot": args.cfm_ot_dir,
                "diag": args.diag_dir,
                "diag_ot": args.diag_ot_dir,
                **extra_directories,
            }.items()
        },
        "artifacts": {
            "paired_reference_sha256": reference_hash,
            **{f"{model}_predictions_sha256": _sha256(prediction_sources[model]) for model in model_order},
            "waveform_summary_sha256": _sha256(waveform_path),
            "per_record_metrics_sha256": _sha256(per_record_path),
            "per_lead_waveform_metrics_sha256": _sha256(per_lead_path),
            "patient_paired_significance_sha256": _sha256(statistics_path),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _json(output_dir / "protocol.json", protocol)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourway_dir", type=Path, required=True)
    parser.add_argument("--cfm_ot_dir", type=Path, required=True)
    parser.add_argument("--diag_dir", type=Path, required=True)
    parser.add_argument("--diag_ot_dir", type=Path, required=True)
    parser.add_argument(
        "--extra_prediction", action="append", default=[], metavar="MODEL=PATH",
        help="Add a provenance-checked prediction artifact to the paired comparison.",
    )
    parser.add_argument(
        "--extra_comparison", action="append", default=[], metavar="COMPARISON:REFERENCE",
        help="Add a pre-registered paired comparison to the joint Holm family.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL DiagMask artifact saved to {output}")
