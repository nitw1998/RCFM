"""Analyze the pre-registered three-seed PTB-XL mask-by-OT factorial."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
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
from scripts.evaluate_ptbxl_flow_checkpoint import _validate_reference
from scripts.evaluate_ptbxl_fourway import _metric_summary
from src.rcfm.metrics.statistics import holm_adjust


MODELS = ("cfm", "cfm_ot", "diag", "diag_ot")
SEEDS = (31, 32, 33)
COMPARISONS = (
    ("diag", "cfm"),
    ("diag_ot", "cfm_ot"),
    ("cfm_ot", "cfm"),
    ("diag_ot", "diag"),
)
INFERENTIAL_METRICS = ("rmse", "mae", "pearson_r")
SUMMARY_METRICS = ("rmse", "mae", "waveform_fd_macro_lead", "per_record_pearson_median")


def _parse_prediction_specs(values: list[str]) -> dict[tuple[str, int], Path]:
    parsed: dict[tuple[str, int], Path] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise ValueError("prediction specs must use MODEL:SEED=PATH")
        identity, raw_path = value.split("=", 1)
        model, raw_seed = identity.split(":", 1)
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError(f"invalid training seed in prediction spec: {value}") from error
        key = (model, seed)
        if model not in MODELS or seed not in SEEDS or key in parsed:
            raise ValueError(f"invalid or duplicate prediction spec: {value}")
        parsed[key] = Path(raw_path)
    expected = set(itertools.product(MODELS, SEEDS))
    missing = sorted(expected - set(parsed))
    if missing:
        raise ValueError(f"missing model/seed prediction artifacts: {missing}")
    return parsed


def _load_prediction(
    directory: Path,
    model: str,
    seed: int,
    reference_hash: str,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, object], Path]:
    protocol_path = directory.resolve() / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_inference = {
        "records": expected_shape[0],
        "split": "official_fold_10",
        "flow_nfe": 50,
        "shared_initial_noise_seed": 2025,
        "deterministic_seed": 31,
        "mask_used_at_inference": False,
        "ot_used_at_inference": False,
        "phase_correction_applied": False,
    }
    inference = protocol.get("inference", {})
    bad = [name for name, expected in expected_inference.items() if inference.get(name) != expected]
    if (
        protocol.get("status") != "completed"
        or protocol.get("model_name") != model
        or protocol.get("training_seed") != seed
        or protocol.get("selection") != "fold_9_best_rmse_checkpoint"
        or protocol.get("paired_reference", {}).get("sha256") != reference_hash
        or bad
    ):
        raise ValueError(f"{model} seed {seed} prediction protocol mismatch: {bad}")
    prediction_path = directory.resolve() / "predictions.npy"
    if _sha256(prediction_path) != protocol.get("prediction", {}).get("sha256"):
        raise ValueError(f"{model} seed {seed} prediction hash changed")
    prediction = np.asarray(np.load(prediction_path, mmap_mode="r"))
    if prediction.shape != expected_shape or not np.all(np.isfinite(prediction)):
        raise ValueError(f"{model} seed {seed} prediction shape/finiteness changed")
    return prediction, protocol, protocol_path


def _patient_map(values: np.ndarray, patient_ids: np.ndarray) -> dict[object, float]:
    values = np.asarray(values, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    output = {}
    for patient in np.unique(patient_ids):
        selected = values[patient_ids == patient]
        selected = selected[np.isfinite(selected)]
        if selected.size:
            output[patient.item() if hasattr(patient, "item") else patient] = float(selected.mean())
    return output


def _exact_seed_sign_flip_p_value(seed_differences: np.ndarray) -> float:
    values = np.asarray(seed_differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("seed sign-flip test requires at least two finite paired seed differences")
    observed = abs(float(values.mean()))
    permutations = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.abs((permutations * values[None]).mean(axis=1))
    return float(np.mean(null >= observed - np.finfo(np.float64).eps * 16))


def _hierarchical_paired_test(
    comparison_by_seed: list[Mapping[object, float]],
    reference_by_seed: list[Mapping[object, float]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, object]:
    if len(comparison_by_seed) != len(reference_by_seed) or len(comparison_by_seed) < 2:
        raise ValueError("hierarchical comparison requires paired maps for at least two seeds")
    common = set(comparison_by_seed[0]) & set(reference_by_seed[0])
    for mapping in (*comparison_by_seed[1:], *reference_by_seed[1:]):
        common &= set(mapping)
    patients = sorted(common, key=str)
    if len(patients) < 2 or bootstrap_replicates <= 0:
        raise ValueError("hierarchical comparison requires patients and positive bootstraps")
    differences = np.asarray(
        [
            [comparison[patient] - reference[patient] for patient in patients]
            for comparison, reference in zip(comparison_by_seed, reference_by_seed)
        ],
        dtype=np.float64,
    )
    patient_differences = differences.mean(axis=0)
    seed_differences = differences.mean(axis=1)
    if np.allclose(patient_differences, 0):
        statistic, raw_p, effect = 0.0, 1.0, 0.0
    else:
        result = stats.wilcoxon(patient_differences, zero_method="wilcox", alternative="two-sided")
        statistic, raw_p = float(result.statistic), float(result.pvalue)
        nonzero = patient_differences[patient_differences != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        effect = float(
            (np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / np.sum(ranks)
        )
    generator = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        sampled_seeds = generator.integers(0, differences.shape[0], differences.shape[0])
        sampled_patients = generator.integers(0, differences.shape[1], differences.shape[1])
        bootstrap[index] = differences[np.ix_(sampled_seeds, sampled_patients)].mean()
    return {
        "training_seeds": int(differences.shape[0]),
        "patients": int(differences.shape[1]),
        "difference_definition": "comparison_minus_reference; records averaged within patient, then averaged across training seeds",
        "mean_difference": float(differences.mean()),
        "patient_averaged_difference_std": float(np.std(patient_differences, ddof=1)),
        "seed_level_differences": [float(value) for value in seed_differences],
        "seed_level_difference_std": float(np.std(seed_differences, ddof=1)),
        "hierarchical_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "patient_test": "paired_wilcoxon_on_patient_differences_averaged_across_three_seeds",
        "patient_test_statistic": statistic,
        "raw_patient_p_value": raw_p,
        "patient_rank_biserial": effect,
        "exact_seed_sign_flip_p_value": _exact_seed_sign_flip_p_value(seed_differences),
        "seed_p_value_resolution_note": "With three seeds, the minimum attainable two-sided exact sign-flip p-value is 0.25.",
        "inference_scope": (
            "Patient p-value quantifies fold-10 patient sampling after averaging the three fixed "
            "training seeds; the hierarchical CI resamples both axes but only three seeds are "
            "available, so it does not establish broad retraining-population significance."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    prediction_dirs = _parse_prediction_specs(args.prediction)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_protocol_path = args.source_protocol.resolve()
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    reference_path = args.reference_artifact.resolve()
    arrays = _validate_reference(reference_path, source_protocol, args.expected_records)
    targets = arrays["targets"].astype(np.float32, copy=False)
    patient_ids = arrays["patient_ids"]
    reference_hash = _sha256(reference_path)

    summaries: dict[tuple[str, int], Mapping[str, object]] = {}
    per_record: dict[tuple[str, int], Mapping[str, np.ndarray]] = {}
    protocols = {}
    seed_rows = []
    for model in MODELS:
        for seed in SEEDS:
            prediction, protocol, protocol_path = _load_prediction(
                prediction_dirs[(model, seed)], model, seed, reference_hash, targets.shape
            )
            summary, records = _metric_summary(targets, prediction)
            summaries[(model, seed)] = summary
            per_record[(model, seed)] = records
            protocols[f"{model}_s{seed}"] = {
                "path": str(protocol_path),
                "sha256": _sha256(protocol_path),
                "prediction_sha256": protocol["prediction"]["sha256"],
                "checkpoint_sha256": protocol["checkpoint"]["sha256"],
            }
            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "rmse": summary["rmse"],
                    "mae": summary["mae"],
                    "waveform_fd_macro_lead": summary["waveform_fd_macro_lead"],
                    "per_record_pearson_median": summary["per_record_pearson_median"],
                }
            )
    seed_metrics_path = output_dir / "seed_metrics.csv"
    _write_csv(seed_metrics_path, seed_rows)

    model_rows = []
    for model in MODELS:
        row: dict[str, object] = {"model": model, "training_seeds": len(SEEDS)}
        for metric in SUMMARY_METRICS:
            values = np.asarray([summaries[(model, seed)][metric] for seed in SEEDS])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        model_rows.append(row)
    model_summary_path = output_dir / "model_mean_sd.csv"
    _write_csv(model_summary_path, model_rows)

    tests = {}
    raw_p = {}
    for comparison_index, (comparison, reference) in enumerate(COMPARISONS):
        pair_name = f"{comparison}_vs_{reference}"
        tests[pair_name] = {}
        for metric_index, metric in enumerate(INFERENTIAL_METRICS):
            result = _hierarchical_paired_test(
                [
                    _patient_map(per_record[(comparison, seed)][metric], patient_ids)
                    for seed in SEEDS
                ],
                [
                    _patient_map(per_record[(reference, seed)][metric], patient_ids)
                    for seed in SEEDS
                ],
                bootstrap_seed=args.bootstrap_seed + comparison_index * 10 + metric_index,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            tests[pair_name][metric] = result
            raw_p[f"{pair_name}/{metric}"] = float(result["raw_patient_p_value"])
    adjusted = holm_adjust(raw_p)
    for name, value in adjusted.items():
        pair_name, metric = name.split("/")
        tests[pair_name][metric]["holm_adjusted_patient_p_value_12_tests"] = value
    significance_path = output_dir / "paired_multiseed_significance.json"
    _json(
        significance_path,
        {
            "schema_version": 1,
            "training_seeds": list(SEEDS),
            "comparisons": tests,
            "multiplicity": "Holm adjustment across 4 pre-registered factorial contrasts x RMSE/MAE/Pearson = 12 patient-level tests",
            "waveform_fd_inference": "descriptive mean and sample SD across three training seeds; no patient-level decomposition or p-value",
            "claim_boundary": "Three training seeds provide limited retraining-variability resolution; do not claim seed-level significance from n=3.",
        },
    )
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "dataset": "PTB-XL official fold 10",
        "records": int(len(targets)),
        "patients": int(len(np.unique(patient_ids))),
        "training_seeds": list(SEEDS),
        "models": list(MODELS),
        "comparisons": [list(value) for value in COMPARISONS],
        "same_reference_and_seed2025_initial_noise": True,
        "checkpoint_selection": "fold-9 best RMSE independently within each model and seed",
        "reference": {
            "path": str(reference_path),
            "sha256": reference_hash,
            "source_protocol_path": str(source_protocol_path),
            "source_protocol_sha256": _sha256(source_protocol_path),
        },
        "prediction_protocols": protocols,
        "artifacts": {
            "seed_metrics_sha256": _sha256(seed_metrics_path),
            "model_mean_sd_sha256": _sha256(model_summary_path),
            "paired_significance_sha256": _sha256(significance_path),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _json(output_dir / "protocol.json", protocol)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="MODEL:SEED=DIR",
        help="Repeat exactly once for all four models and seeds 31/32/33.",
    )
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--source_protocol", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_records", type=int, default=2203)
    parser.add_argument("--bootstrap_seed", type=int, default=3100)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    output = run(build_argparser().parse_args())
    print(f"PTB-XL multiseed factorial analysis saved to {output}")
