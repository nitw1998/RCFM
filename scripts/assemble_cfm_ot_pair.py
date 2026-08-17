"""Assemble paired CFM/CFM+OT waveform artifacts on one frozen test split."""

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

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cfm_ot_frozen_reference import DATASETS
from scripts.evaluate_cpsc_zscore_paired import _json, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_ptbxl_fourway import _metric_summary
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


MODELS = ("cfm", "cfm_ot")


def _load_source(dataset: str, path: Path) -> dict[str, np.ndarray]:
    if dataset in {"cpsc2018", "ptbxl"}:
        directory = path
        with np.load(directory / "paired_reference.npz", allow_pickle=False) as artifact:
            output = {key: np.asarray(artifact[key]) for key in artifact.files}
        output["cfm_predictions"] = np.asarray(np.load(directory / "cfm_predictions.npy", mmap_mode="r"))
        if dataset == "ptbxl":
            output["cfm_ot_predictions"] = np.asarray(np.load(directory / "cfm_ot_predictions.npy", mmap_mode="r"))
        return output
    with np.load(path, allow_pickle=False) as artifact:
        return {key: np.asarray(artifact[key]) for key in artifact.files}


def _identity(dataset: str, arrays: dict[str, np.ndarray]) -> tuple[str | None, np.ndarray | None]:
    if dataset == "ptbxl":
        return "patient_id", arrays["patient_ids"].astype(str)
    if dataset == "cpsc2018":
        return "record_id", arrays["record_ids"].astype(str)
    if dataset in {"wesad", "mmecg"}:
        return "subject_id", arrays["subject_ids"].astype(str)
    return None, None


def _patient_means(values: np.ndarray, identities: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    _, inverse = np.unique(identities[valid], return_inverse=True)
    return np.bincount(inverse, weights=values[valid]) / np.bincount(inverse)


def _paired_inference(
    comparison: np.ndarray, reference: np.ndarray, identities: np.ndarray,
    seed: int, replicates: int, status: str,
) -> dict[str, object]:
    comparison_means = _patient_means(comparison, identities)
    reference_means = _patient_means(reference, identities)
    difference = comparison_means - reference_means
    generator = np.random.default_rng(seed)
    bootstrap = np.mean(
        difference[generator.integers(0, len(difference), size=(replicates, len(difference)))], axis=1
    )
    result: dict[str, object] = {
        "identity_count": int(len(difference)), "difference_definition": "cfm_ot_minus_cfm_after_identity_mean",
        "mean_difference": float(np.mean(difference)), "paired_difference_std": float(np.std(difference, ddof=1)),
        "bootstrap_95_ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "bootstrap_replicates": replicates, "bootstrap_seed": seed, "inference_status": status,
    }
    if status != "confirmatory_eligible" or len(difference) < 10:
        result.update({"test": None, "raw_p_value": None, "effect_size": None})
        return result
    if np.allclose(difference, 0):
        result.update({"test": "exact_no_difference", "raw_p_value": 1.0, "effect_size": 0.0})
        return result
    test = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    nonzero = difference[difference != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    effect = (np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / np.sum(ranks)
    result.update({"test": "paired_wilcoxon_signed_rank", "raw_p_value": float(test.pvalue),
                   "effect_size": {"name": "rank_biserial_cfm_ot_minus_cfm", "value": float(effect)}})
    return result


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    arrays = _load_source(args.dataset, args.source.resolve())
    targets = np.asarray(arrays["targets"], dtype=np.float32)
    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    cfm = np.asarray(arrays["cfm_predictions"], dtype=np.float32)
    if args.dataset == "ptbxl":
        cfm_ot = np.asarray(arrays["cfm_ot_predictions"], dtype=np.float32)
        generation_protocol = None
    else:
        generation_protocol = json.loads((args.cfm_ot_dir / "protocol.json").read_text(encoding="utf-8"))
        if generation_protocol.get("status") != "completed" or generation_protocol.get("dataset") != args.dataset:
            raise ValueError("CFM+OT generation protocol is incomplete or mismatched")
        cfm_ot_path = args.cfm_ot_dir / "cfm_ot_predictions.npy"
        if _sha256(cfm_ot_path) != generation_protocol["prediction"]["sha256"]:
            raise ValueError("CFM+OT prediction hash changed")
        cfm_ot = np.asarray(np.load(cfm_ot_path, mmap_mode="r"), dtype=np.float32)
    if not (targets.shape == conditions.shape[:1] + (targets.shape[1], 512) == cfm.shape == cfm_ot.shape):
        raise ValueError("paired waveform arrays disagree in shape")
    if any(not np.all(np.isfinite(value)) for value in (targets, conditions, cfm, cfm_ot)):
        raise FloatingPointError("paired waveform arrays contain NaN or Inf")
    identity_name, identities = _identity(args.dataset, arrays)
    summaries, rows = {}, {}
    metric = _metric_summary if targets.shape[1] == 11 else _waveform_metrics
    for model, prediction in (("cfm", cfm), ("cfm_ot", cfm_ot)):
        summaries[model], rows[model] = metric(targets, prediction)
    raw_path = output / "paired_predictions.npz"
    payload = {"targets": targets, "conditions": conditions, "cfm_predictions": cfm,
               "cfm_ot_predictions": cfm_ot}
    for key in ("record_ids", "patient_ids", "subject_ids", "labels", "source_files"):
        if key in arrays:
            payload[key] = arrays[key]
    np.savez_compressed(raw_path, **payload)
    phase_path = None
    phase_summary = None
    aligned_rows = None
    if args.dataset in {"mimic_afib", "wesad", "mmecg"}:
        center = targets[:, :, args.max_lag_samples : -args.max_lag_samples]
        phase_payload = {"targets": center}
        for key in ("subject_ids", "labels", "source_files"):
            if key in arrays:
                phase_payload[key] = arrays[key]
        phase_summary, aligned_rows = {}, {}
        for model, prediction in (("cfm", cfm), ("cfm_ot", cfm_ot)):
            lag_summary, lag_values = _lag_diagnostic(targets, prediction, args.max_lag_samples, 128)
            shifts = lag_values["best_lag_samples"].astype(np.int32)
            aligned_target, unshifted, aligned = _fixed_support_align(
                targets, prediction, shifts, args.max_lag_samples
            )
            if not np.array_equal(center, aligned_target):
                raise ValueError("phase alignment changed the common target support")
            before_summary, _ = _waveform_metrics(center, unshifted)
            after_summary, aligned_rows[model] = _waveform_metrics(center, aligned)
            phase_summary[model] = {"lag": lag_summary, "unshifted_fixed_support": before_summary,
                                    "oracle_aligned_fixed_support": after_summary}
            phase_payload[f"{model}_oracle_shifts"] = shifts
            phase_payload[f"{model}_unshifted_predictions"] = unshifted
            phase_payload[f"{model}_oracle_aligned_predictions"] = aligned
        phase_path = output / "phase_predictions_maxlag16.npz"
        np.savez_compressed(phase_path, **phase_payload)
    waveform_path = output / "waveform_summary.json"
    _json(waveform_path, {"models": summaries, "phase_sensitivity": phase_summary})
    per_record_path = output / "per_record_waveform_metrics.csv"
    with per_record_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["row", identity_name or "identity_status"] + [f"{model}_{metric_name}" for model in MODELS for metric_name in ("rmse", "mae", "bias", "pearson_r")]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(len(targets)):
            row: dict[str, object] = {"row": index, identity_name or "identity_status": identities[index] if identities is not None else "unavailable"}
            for model in MODELS:
                for metric_name in ("rmse", "mae", "bias", "pearson_r"):
                    row[f"{model}_{metric_name}"] = float(rows[model][metric_name][index])
            writer.writerow(row)
    statistics = {"identity": identity_name, "comparisons": {}, "waveform_fd_inference": "descriptive_full_distribution_only"}
    if identities is None:
        inference_status = "blocked_identity_unavailable"
    elif args.dataset in {"wesad", "mmecg"}:
        inference_status = "descriptive_only_extremely_underpowered_n3"
    elif args.dataset == "cpsc2018":
        inference_status = "confirmatory_eligible_record_level_patient_ids_unavailable"
    else:
        inference_status = "confirmatory_eligible"
    for metric_index, metric_name in enumerate(("rmse", "mae", "pearson_r")):
        if identities is None:
            statistics["comparisons"][metric_name] = {"inference_status": inference_status, "raw_p_value": None}
        else:
            status = "confirmatory_eligible" if inference_status.startswith("confirmatory_eligible") else inference_status
            statistics["comparisons"][metric_name] = _paired_inference(
                rows["cfm_ot"][metric_name], rows["cfm"][metric_name], identities,
                args.bootstrap_seed + metric_index, args.bootstrap_replicates, status,
            )
            statistics["comparisons"][metric_name]["identity_caveat"] = inference_status
    statistics_path = output / "paired_significance.json"
    _json(statistics_path, statistics)
    protocol = {
        "schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "dataset": args.dataset,
        "protocol": {"raw_full_window_primary": True, "phase_correction_applied": False,
                     "phase_correction_applied_to_primary": False,
                     "oracle_phase_diagnostic": args.dataset in {"mimic_afib", "wesad", "mmecg"},
                     "models": list(MODELS)},
        "source": {"path": str(args.source.resolve()), "sha256": _sha256(args.source) if args.source.is_file() else _sha256(args.source / "protocol.json")},
        "generation_protocol_sha256": _sha256(args.cfm_ot_dir / "protocol.json") if generation_protocol else None,
        "outputs": {path.name: _sha256(path) for path in (raw_path, waveform_path, per_record_path, statistics_path, *([phase_path] if phase_path else []))},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    _json(output / "protocol.json", protocol)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic_afib", "cpsc2018", "wesad", "mmecg", "ptbxl"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cfm_ot_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument("--bootstrap_seed", type=int, default=4041)
    parser.add_argument("--bootstrap_replicates", type=int, default=10000)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"CFM/CFM+OT paired artifact saved to {result}")
