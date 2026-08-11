"""Deterministically compare CPSC2018 min-max CFM, RCFM, and RCFM-OT."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import neurokit2 as nk
import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_clinical import PARAMETERS, evaluate as evaluate_clinical
from scripts.evaluate_cpsc_zscore_paired import (
    AMPLITUDE_PARAMETERS,
    _array_sha256,
    _checkpoint_contract,
    _fixed_noise,
    _generate,
    _json,
    _save_waveform_outputs,
    _sha256,
)
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from train_rcfm import build_datasets


MATCHED_FIELDS = (
    "task",
    "datasets",
    "dataset_version",
    "split_hash",
    "normalization_id",
    "condition_unit",
    "target_unit",
    "alignment_id",
    "condition_lead",
    "target_lead",
    "condition_lead_index",
    "target_lead_index",
    "window_size",
    "attention_heads",
    "flow_matcher",
    "sigma",
    "seed",
)


def _validate_threeway_contracts(contracts: Mapping[str, Mapping[str, object]]) -> None:
    if set(contracts) != {"cfm", "rcfm", "rcfm_ot"}:
        raise ValueError("three-way comparison requires cfm, rcfm, and rcfm_ot contracts")
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    for name, expected in expected_kinds.items():
        if contracts[name]["kind"] != expected:
            raise ValueError(f"{name} checkpoint kind must be {expected}")
    reference = contracts["cfm"]
    for name in ("rcfm", "rcfm_ot"):
        mismatched = [
            field
            for field in MATCHED_FIELDS
            if reference["config"].get(field) != contracts[name]["config"].get(field)
        ]
        if mismatched:
            raise ValueError(f"{name} checkpoint disagrees on: " + ", ".join(mismatched))
        if reference["normalization"] != contracts[name]["normalization"]:
            raise ValueError(f"{name} checkpoint normalization metadata differs")
        if reference["output_spec"] != contracts[name]["output_spec"]:
            raise ValueError(f"{name} checkpoint output specification differs")
    config = reference["config"]
    if int(reference["output_spec"].get("channels", 0)) != 1:
        raise ValueError(
            "this historical clinical evaluator supports only single-target-lead checkpoints; "
            "multi-lead ECG parameters must be evaluated separately per lead"
        )
    if config.get("normalization_id") != "record_minmax_neg1_1_v1":
        raise ValueError("three-way entry requires record_minmax_neg1_1_v1")
    if config.get("task") != "ecg2ecg" or config.get("datasets") != ["CPSC2018"]:
        raise ValueError("three-way entry requires the CPSC2018 ECG-to-ECG task")
    cfm = contracts["cfm"]["config"]
    rcfm = contracts["rcfm"]["config"]
    rcfm_ot = contracts["rcfm_ot"]["config"]
    if float(cfm.get("region_weight", -1)) != 0 or bool(cfm.get("use_minibatch_ot")):
        raise ValueError("CFM must have region_weight=0 and OT disabled")
    if float(rcfm.get("region_weight", 0)) <= 0 or bool(rcfm.get("use_minibatch_ot")):
        raise ValueError("RCFM must have positive region weight and OT disabled")
    if float(rcfm_ot.get("region_weight", 0)) != float(rcfm["region_weight"]):
        raise ValueError("RCFM and RCFM-OT region weights must match")
    if not bool(rcfm_ot.get("use_minibatch_ot")):
        raise ValueError("RCFM-OT must enable minibatch OT")
    if rcfm_ot.get("ot_method") != "exact" or rcfm_ot.get("ot_sampling_strategy") != "assignment":
        raise ValueError("RCFM-OT must use exact assignment coupling")


def _include_all_p_wave_applicability(count: int) -> tuple[np.ndarray, dict[str, object]]:
    if count <= 0:
        raise ValueError("record count must be positive")
    return np.ones(count, dtype=bool), {
        "policy": "all CPSC records eligible for PR and P-wave delineation by author decision",
        "af_labels_consulted": False,
        "applicable_records": count,
        "not_applicable_records": 0,
    }


def _clinical_namespace(
    real_path: Path,
    generated_path: Path,
    record_ids_path: Path,
    p_wave_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    target_lead: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        real=real_path,
        generated=generated_path,
        subject_ids=record_ids_path,
        p_wave_applicability=p_wave_path,
        output=output_path,
        sampling_rate=args.sampling_rate,
        lead=target_lead,
        analysis_unit="record",
        normalization_id="record_minmax_neg1_1_v1",
        unit="normalized",
        inverse_transformed=False,
        allow_normalized_amplitudes=True,
        continuous=False,
        minimum_hrv_seconds=args.minimum_hrv_seconds,
        qtc_formula=args.qtc_formula,
        st_offset_ms=args.st_offset_ms,
        delineation_method=args.delineation_method,
        clean_method=args.clean_method,
    )


def _parameter_rows(result: Mapping[str, object], parameter: str) -> dict[str, tuple[float, float]]:
    real = result["unit_summaries"]["real"]
    generated = result["unit_summaries"]["generated"]
    rows = {}
    for record_id in set(real) & set(generated):
        pair = (real[record_id].get(parameter), generated[record_id].get(parameter))
        if all(value is not None and np.isfinite(value) for value in pair):
            rows[str(record_id)] = (float(pair[0]), float(pair[1]))
    return rows


def _clinical_threeway_comparison(
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    models = tuple(results)
    output: dict[str, object] = {
        "analysis_unit": "record",
        "subset_policy": "same record has real and generated values for all three models",
        "p_wave_policy": "include_all_records_without_AF_exclusion",
        "inference_status": "single_seed_record_level_descriptive_only",
        "parameters": {},
    }
    for parameter in PARAMETERS:
        rows = {model: _parameter_rows(results[model], parameter) for model in models}
        common = sorted(set.intersection(*(set(values) for values in rows.values())))
        if len(common) < 2:
            output["parameters"][parameter] = {
                "status": "insufficient_common_record_pairs",
                "n": len(common),
            }
            continue
        real_by_model = np.asarray(
            [[rows[model][record_id][0] for record_id in common] for model in models]
        )
        if not np.allclose(real_by_model, real_by_model[0], atol=1e-10, rtol=1e-10):
            raise ValueError("real clinical summaries changed between model evaluations")
        real = real_by_model[0]
        model_metrics = {}
        for model in models:
            generated = np.asarray([rows[model][record_id][1] for record_id in common])
            error = generated - real
            correlation = paired_correlation(real, generated)
            agreement = bland_altman(real, generated)
            agreement.pop("pair_means")
            agreement.pop("differences")
            model_metrics[model] = {
                "generated_mean": float(np.mean(generated)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "pearson": correlation,
                "bland_altman": agreement,
            }
        output["parameters"][parameter] = {
            "status": "ok",
            "n": len(common),
            "unit": "normalized_record_minmax_neg1_1" if parameter in AMPLITUDE_PARAMETERS else "ms",
            "claim_status": "exploratory_not_physical" if parameter in AMPLITUDE_PARAMETERS else "interval_metric",
            "real_mean": float(np.mean(real)),
            "models": model_metrics,
        }
    return output


def _write_clinical_comparison(payload: Mapping[str, object], output_dir: Path) -> None:
    _json(output_dir / "threeway_clinical_comparison.json", payload)
    fields = ["parameter", "status", "n", "unit", "real_mean"]
    for model in ("cfm", "rcfm", "rcfm_ot"):
        fields.extend(
            [
                f"{model}_generated_mean",
                f"{model}_mae",
                f"{model}_rmse",
                f"{model}_pearson_r",
                f"{model}_ba_bias",
                f"{model}_ba_lower",
                f"{model}_ba_upper",
            ]
        )
    with (output_dir / "threeway_clinical_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for parameter in PARAMETERS:
            result = payload["parameters"][parameter]
            row = {
                "parameter": parameter,
                "status": result["status"],
                "n": result.get("n", 0),
                "unit": result.get("unit", ""),
                "real_mean": result.get("real_mean", ""),
            }
            for model, metrics in result.get("models", {}).items():
                row.update(
                    {
                        f"{model}_generated_mean": metrics["generated_mean"],
                        f"{model}_mae": metrics["mae"],
                        f"{model}_rmse": metrics["rmse"],
                        f"{model}_pearson_r": metrics["pearson"]["r"],
                        f"{model}_ba_bias": metrics["bland_altman"]["bias"],
                        f"{model}_ba_lower": metrics["bland_altman"]["lower_limit"],
                        f"{model}_ba_upper": metrics["bland_altman"]["upper_limit"],
                    }
                )
            writer.writerow(row)


def _pairwise_waveform_comparison(
    per_record: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    output = {"inference_status": "single_seed_descriptive_only", "pairs": {}}
    for first, second in (("cfm", "rcfm"), ("cfm", "rcfm_ot"), ("rcfm", "rcfm_ot")):
        pair = {}
        for metric in ("rmse", "mae"):
            difference = np.asarray(per_record[first][metric]) - np.asarray(
                per_record[second][metric]
            )
            pair[metric] = {
                "difference_definition": f"{first}_minus_{second}",
                "mean_difference": float(np.mean(difference)),
                "median_difference": float(np.median(difference)),
                f"fraction_favoring_{second}": float(np.mean(difference > 0)),
                "ties": int(np.sum(difference == 0)),
            }
        output["pairs"][f"{first}_vs_{second}"] = pair
    return output


def evaluate(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    checkpoint_paths = {
        "cfm": args.cfm_checkpoint,
        "rcfm": args.rcfm_checkpoint,
        "rcfm_ot": args.rcfm_ot_checkpoint,
    }
    contracts = {name: _checkpoint_contract(path) for name, path in checkpoint_paths.items()}
    _validate_threeway_contracts(contracts)
    config = contracts["rcfm"]["config"]
    output_spec = contracts["rcfm"]["output_spec"]
    _, test_set = build_datasets(
        config["task"],
        config["datasets"],
        str(args.data_root),
        int(config["window_size"]),
        normalization_metadata=contracts["rcfm"]["normalization"],
        normalization_id=config["normalization_id"],
        condition_lead_index=int(config["condition_lead_index"]),
        target_lead_index=int(config["target_lead_index"]),
        load_train=False,
        heldout_split="test",
    )
    total_records = len(test_set)
    if args.expected_records is not None and total_records != args.expected_records:
        raise ValueError(f"expected {args.expected_records} records, found {total_records}")
    count = total_records if args.max_records is None else min(args.max_records, total_records)
    targets = np.asarray(test_set.target_ecg[:count, None, :], dtype=np.float32)
    conditions = np.asarray(test_set.condition_signal[:count, None, :], dtype=np.float32)
    record_ids = np.asarray(test_set.record_ids[:count])
    if len(np.unique(record_ids)) != count:
        raise ValueError("test record IDs must be unique")
    record_ids_path = output_dir / "record_ids.npy"
    targets_path = output_dir / "targets_normalized.npy"
    conditions_path = output_dir / "conditions_normalized.npy"
    np.save(record_ids_path, record_ids, allow_pickle=False)
    np.save(targets_path, targets, allow_pickle=False)
    np.save(conditions_path, conditions, allow_pickle=False)
    p_wave_applicable, p_wave_policy = _include_all_p_wave_applicability(count)
    p_wave_path = output_dir / "p_wave_applicability.npy"
    np.save(p_wave_path, p_wave_applicable, allow_pickle=False)
    initial_noise = _fixed_noise(
        count, int(output_spec["channels"]), int(output_spec["length"]), args.noise_seed
    )
    np.save(output_dir / "initial_noise.npy", initial_noise, allow_pickle=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch cannot access CUDA")

    predictions = {}
    generation_metadata = {}
    expected_kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    for name in ("cfm", "rcfm", "rcfm_ot"):
        prediction, metadata = _generate(
            checkpoint_paths[name], expected_kinds[name], conditions, initial_noise,
            args.batch_size, args.steps, device, args.deterministic_seed,
        )
        prediction_path = output_dir / f"{name}_predictions_normalized.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        predictions[name] = prediction
        generation_metadata[name] = {
            **metadata,
            "prediction_file": prediction_path.name,
            "prediction_sha256": _array_sha256(prediction),
        }

    waveform_summaries = {}
    per_record_waveform = {}
    for name in ("cfm", "rcfm", "rcfm_ot"):
        waveform_summaries[name], per_record_waveform[name] = _save_waveform_outputs(
            name, record_ids, targets, predictions[name], output_dir
        )
    waveform_comparison = _pairwise_waveform_comparison(per_record_waveform)
    _json(output_dir / "threeway_waveform_comparison.json", waveform_comparison)

    clinical_dir = output_dir / "clinical"
    clinical_dir.mkdir()
    clinical_results = {}
    for name in ("cfm", "rcfm", "rcfm_ot"):
        model_dir = clinical_dir / name
        model_dir.mkdir()
        namespace = _clinical_namespace(
            targets_path,
            output_dir / f"{name}_predictions_normalized.npy",
            record_ids_path,
            p_wave_path,
            model_dir / "clinical_metrics.json",
            args,
            str(output_spec["target_lead"]),
        )
        print(f"clinical delineation: {name}", flush=True)
        clinical_results[name] = evaluate_clinical(namespace)
        _json(model_dir / "clinical_metrics.json", clinical_results[name])
    clinical_comparison = _clinical_threeway_comparison(clinical_results)
    _write_clinical_comparison(clinical_comparison, clinical_dir)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "dataset": "CPSC2018",
            "split": "test",
            "split_hash": config["split_hash"],
            "record_count": count,
            "full_test_record_count": total_records,
            "normalization_id": config["normalization_id"],
            "condition_lead": config["condition_lead"],
            "target_lead": output_spec["target_lead"],
            "sampling_rate_hz": args.sampling_rate,
            "nfe": args.steps,
            "batch_size": args.batch_size,
            "noise_seed": args.noise_seed,
            "deterministic_seed": args.deterministic_seed,
            "noise_sha256": _array_sha256(initial_noise),
            "same_initial_noise_for_all_models": True,
            "qtc_formula": args.qtc_formula,
            "st_offset_ms": args.st_offset_ms,
            "p_wave_policy": p_wave_policy,
        },
        "applicability": {
            "rr_pr_qrs_qt_qtc": "eligible_after_independent_delineation",
            "hrv_sdnn_rmssd": "blocked_4_second_records_not_continuous",
            "physical_amplitudes_and_st": "blocked_unknown_source_physical_unit",
            "normalized_amplitudes_and_st": "exploratory_record_minmax_not_mV",
            "subject_level_inference": "blocked_no_subject_identifiers",
            "single_seed_significance": "blocked_requires_additional_training_seeds",
        },
        "checkpoints": {
            name: {
                "path": str(checkpoint_paths[name].resolve()),
                "sha256": _sha256(checkpoint_paths[name]),
                **generation_metadata[name],
            }
            for name in checkpoint_paths
        },
        "array_hashes": {
            "targets": _array_sha256(targets),
            "conditions": _array_sha256(conditions),
            "record_ids": _array_sha256(record_ids),
        },
        "waveform_metrics": waveform_summaries,
        "waveform_comparison_file": "threeway_waveform_comparison.json",
        "clinical_comparison_file": "clinical/threeway_clinical_comparison.json",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _json(output_dir / "paired_evaluation_metadata.json", metadata)
    print(f"three-way paired evaluation complete: {output_dir}", flush=True)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=688)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--minimum_hrv_seconds", type=float, default=30.0)
    parser.add_argument(
        "--qtc_formula",
        choices=["bazett", "fridericia", "framingham", "hodges"],
        default="fridericia",
    )
    parser.add_argument("--st_offset_ms", type=float, default=60.0)
    parser.add_argument("--delineation_method", choices=["dwt", "cwt", "peak"], default="dwt")
    parser.add_argument("--clean_method", default="neurokit")
    return parser


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
