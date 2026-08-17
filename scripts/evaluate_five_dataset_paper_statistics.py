"""Build frozen paper-level waveform statistics for five physiological datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.metrics.paper_statistics import (
    ELEVEN_TARGET_LEADS,
    apply_fixed_lag,
    estimate_training_fixed_lag,
    raw_waveform_summary,
)
from data import PairedSignalDataset, _rddm_window_minmax_metadata, _rddm_window_minmax_neg1_1


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
DATASETS = ("MIMIC-AFib", "WESAD", "MMECG", "PTB-XL", "CPSC2018")
DEFAULT_DATASETS = ("WESAD", "MMECG", "PTB-XL", "CPSC2018")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        return {name: np.asarray(artifact[name]) for name in required}


def _load_mimic(workspace: Path) -> dict[str, object]:
    flow_path = workspace / "runs/clinical/mimic_afib_flow_waveform_check_seed2025_v2/mimic_flow_predictions.npz"
    rddm_path = workspace / "runs/clinical/mimic_afib_rddm_deterministic_seed2025_v1/mimic_rddm_predictions.npz"
    identity = workspace / "runs/preprocessing/mimic_afib_identity_v1"
    flow = _load_npz(
        flow_path,
        {"targets", "conditions", *(f"{model}_predictions" for model in MODELS[:-1])},
    )
    rddm = _load_npz(
        rddm_path,
        {"targets", "conditions", "rddm_predictions", "source_test_rows_before_zero_filter"},
    )
    if not np.array_equal(flow["targets"], rddm["targets"]) or not np.array_equal(
        flow["conditions"], rddm["conditions"]
    ):
        raise ValueError("MIMIC flow and RDDM rows are not identical")
    subjects = np.load(identity / "subject_ids_test.npy", allow_pickle=False).astype(str)
    records = np.load(identity / "record_ids_test.npy", allow_pickle=False).astype(str)
    afib = np.load(identity / "afib_labels_test.npy", allow_pickle=False).astype(bool)
    source_rows = np.load(identity / "source_rows_before_qc_test.npy", allow_pickle=False)
    if not np.array_equal(source_rows, rddm["source_test_rows_before_zero_filter"]):
        raise ValueError("MIMIC identity sidecar row mapping differs from frozen predictions")
    predictions = {
        model: np.asarray(flow[f"{model}_predictions"], dtype=np.float32)
        for model in MODELS[:-1]
    }
    predictions["rddm"] = np.asarray(rddm["rddm_predictions"], dtype=np.float32)
    return {
        "targets": np.asarray(flow["targets"], dtype=np.float32),
        "conditions": np.asarray(flow["conditions"], dtype=np.float32),
        "predictions": predictions,
        "groups": subjects,
        "identity": {"subject_ids": subjects, "record_ids": records, "afib": afib},
        "inputs": [flow_path, rddm_path, identity / "identity_manifest.json"],
    }


def _load_single_npz(
    path: Path,
    group_key: str,
    expected_records: int,
) -> dict[str, object]:
    arrays = _load_npz(
        path,
        {"targets", "conditions", group_key, *(f"{model}_predictions" for model in MODELS)},
    )
    targets = np.asarray(arrays["targets"], dtype=np.float32)
    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    groups = np.asarray(arrays[group_key]).astype(str)
    if targets.shape != (expected_records, 1, 512) or conditions.shape != targets.shape:
        raise ValueError(f"{path} violates its frozen single-lead shape")
    return {
        "targets": targets,
        "conditions": conditions,
        "predictions": {
            model: np.asarray(arrays[f"{model}_predictions"], dtype=np.float32)
            for model in MODELS
        },
        "groups": groups,
        "identity": {"subject_ids": groups},
        "inputs": [path],
    }


def _load_multilead(
    directory: Path,
    group_key: str,
    expected_records: int,
) -> dict[str, object]:
    reference_path = directory / "paired_reference.npz"
    arrays = _load_npz(reference_path, {"targets", "conditions", group_key})
    targets = np.asarray(arrays["targets"], dtype=np.float32)
    conditions = np.asarray(arrays["conditions"], dtype=np.float32)
    groups = np.asarray(arrays[group_key]).astype(str)
    if targets.shape != (expected_records, 11, 512) or conditions.shape != (
        expected_records,
        1,
        512,
    ):
        raise ValueError(f"{directory} violates the frozen 11-lead shape")
    paths = [reference_path]
    predictions = {}
    for model in MODELS:
        path = directory / f"{model}_predictions.npy"
        predictions[model] = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        paths.append(path)
    return {
        "targets": targets,
        "conditions": conditions,
        "predictions": predictions,
        "groups": groups,
        "identity": {group_key: groups},
        "inputs": paths,
    }


def _validate_manifest_coverage(
    data: Mapping[str, object], manifest_path: Path, expected_subjects: tuple[str, ...]
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = np.asarray(data["groups"]).astype(str)
    actual = tuple(sorted(set(groups.tolist())))
    declared = tuple(sorted(str(value) for value in manifest["test_subject_ids"]))
    if actual != declared or actual != tuple(sorted(expected_subjects)):
        raise ValueError(f"test subject coverage differs for {manifest_path}")
    if int(manifest["test_windows"]) != len(groups):
        raise ValueError(f"test window count differs for {manifest_path}")
    return {
        "status": "complete_frozen_test_split",
        "split_method": manifest["split_method"],
        "test_subject_ids": list(actual),
        "test_windows": len(groups),
        "not_an_analysis_subset": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _training_pair(
    workspace: Path, dataset: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if dataset == "MIMIC-AFib":
        root = workspace / "runs/preprocessing/mimic_afib_rddm_zero_ppg_qc_v1/MIMIC-AFib"
        raw_target = np.load(root / "ecg_train_4sec.npy", mmap_mode="r", allow_pickle=False)
        raw_source = np.load(root / "ppg_train_4sec.npy", mmap_mode="r", allow_pickle=False)
        calibration_count = min(512, len(raw_target))
        indices = np.linspace(0, len(raw_target) - 1, calibration_count, dtype=np.int64)
        target = _rddm_window_minmax_neg1_1(np.asarray(raw_target[indices]))
        source = _rddm_window_minmax_neg1_1(np.asarray(raw_source[indices]))
        calibration = PairedSignalDataset(
            target,
            source,
            clean_target=True,
            clean_condition_ppg=True,
            normalization_metadata=_rddm_window_minmax_metadata(),
            return_region_mask=False,
        )
        return (
            np.asarray(calibration.condition_signal, dtype=np.float32),
            np.asarray(calibration.target_ecg, dtype=np.float32),
            {
                "training_windows_available": int(len(raw_target)),
                "calibration_windows": int(calibration_count),
                "selection": "integer_linspace_over_complete_training_row_order",
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                "prediction_or_test_data_used_for_selection": False,
                "preprocessing": "exact_training_minmax_then_neurokit_cleaning",
            },
        )
    elif dataset == "WESAD":
        root = workspace / "runs/preprocessing/wesad_subject_fold1_v1/WESAD"
        target_path, source_path = root / "ecg_train_4sec.npy", root / "ppg_train_4sec.npy"
        raw_target = np.load(target_path, allow_pickle=False)
        raw_source = np.load(source_path, allow_pickle=False)
    elif dataset == "MMECG":
        root = workspace / "runs/preprocessing/mmecg_subject_split_v1/mmECG"
        target_path, source_path = root / "ecg_train_4sec.npy", root / "ppg_train_4sec.npy"
        raw_target = np.load(target_path, allow_pickle=False)
        raw_source = np.load(source_path, allow_pickle=False)
    elif dataset == "PTB-XL":
        root = workspace / "runs/preprocessing/ptbxl_official_minmax_v1/PTBXL"
        records_path = root / "X_train_resampled.npy"
        records = np.load(records_path, allow_pickle=False)
    else:
        root = workspace / "runs/preprocessing/cpsc2018_multilead_qc_v3/CPSC2018"
        records_path = root / "X_train_resampled.npy"
        records = np.load(records_path, allow_pickle=False)
    if dataset in {"WESAD", "MMECG"}:
        target = _rddm_window_minmax_neg1_1(raw_target)
        source = _rddm_window_minmax_neg1_1(raw_source)
        preprocessing = "finite_per_window_minmax_neg1_1_no_cleaning"
        training_inputs = {
            str(target_path.resolve()): _sha256(target_path),
            str(source_path.resolve()): _sha256(source_path),
        }
    else:
        target_indices = tuple(index for index in range(12) if index != 1)
        source = np.asarray(records[:, :512, 1], dtype=np.float32)
        target = np.transpose(
            np.asarray(records[:, :512, target_indices], dtype=np.float32), (0, 2, 1)
        )
        source_min = source.min(axis=-1, keepdims=True)
        source_range = source.max(axis=-1, keepdims=True) - source_min
        target_min = target.min(axis=-1, keepdims=True)
        target_range = target.max(axis=-1, keepdims=True) - target_min
        if np.any(source_range <= 0) or np.any(target_range <= 0):
            raise ValueError(f"{dataset} training records contain zero-range selected leads")
        source = (2.0 * (source - source_min) / source_range - 1.0).astype(np.float32)
        target = (2.0 * (target - target_min) / target_range - 1.0).astype(np.float32)
        preprocessing = "per_record_per_lead_minmax_neg1_1"
        training_inputs = {str(records_path.resolve()): _sha256(records_path)}
    return (
        np.asarray(source, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        {
            "training_windows_available": int(len(target)),
            "calibration_windows": int(len(target)),
            "selection": "complete_training_split",
            "prediction_or_test_data_used_for_selection": False,
            "preprocessing": preprocessing,
            "training_input_sha256": training_inputs,
        },
    )


def _flatten_summary(dataset: str, model: str, summary: Mapping[str, object]) -> dict[str, object]:
    pearson = summary["pearson"]
    wfd = summary["wfd"]
    return {
        "dataset": dataset,
        "model": model,
        "support": summary["support"],
        "samples": summary["samples"],
        "groups": pearson["group_count"],
        "rmse": summary["rmse"],
        "mae_native_synchronized_normalized_domain": summary["mae"],
        "wfd_full_test_set": wfd["value"],
        "wfd_aggregation": wfd["aggregation"],
        "pearson_group_median_primary": pearson["group_median_primary"],
        "pearson_group_fisher_mean": pearson["group_fisher_mean"],
        "pearson_sample_median_secondary": pearson["sample_median"],
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    workspace = args.workspace.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    requested = tuple(args.datasets)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("datasets must be nonempty and unique")
    unknown = set(requested) - set(DATASETS)
    if unknown:
        raise ValueError(f"unknown datasets: {sorted(unknown)}")
    loaders = {
        "MIMIC-AFib": lambda: _load_mimic(workspace),
        "WESAD": _load_single_npz(
            workspace / "runs/evaluation/wesad_fourway_raw_seed2025_v1/raw_predictions.npz",
            "subject_ids",
            4213,
        ),
        "MMECG": _load_single_npz(
            workspace / "runs/evaluation/mmecg_fourway_raw_seed2025_v1/raw_predictions.npz",
            "subject_ids",
            2877,
        ),
        "PTB-XL": _load_multilead(
            workspace / "runs/evaluation/ptbxl_fourway_fold10_raw_seed2025_v1",
            "patient_ids",
            2203,
        ),
        "CPSC2018": _load_multilead(
            workspace / "runs/evaluation/cpsc2018_fourway_raw_seed2025_v1",
            "record_ids",
            686,
        ),
    }
    data = {
        dataset: (loader() if callable(loader) else loader)
        for dataset, loader in loaders.items()
        if dataset in requested
    }
    split_audit = {
        "WESAD": _validate_manifest_coverage(
            data["WESAD"],
            workspace / "runs/preprocessing/wesad_subject_fold1_v1/WESAD/dataset_manifest.json",
            ("S001", "S010", "S012"),
        ) if "WESAD" in data else {"status": "not_requested"},
        "MMECG": _validate_manifest_coverage(
            data["MMECG"],
            workspace / "runs/preprocessing/mmecg_subject_split_v1/mmECG/dataset_manifest.json",
            ("S005", "S007", "S009"),
        ) if "MMECG" in data else {"status": "not_requested"},
    }

    result: dict[str, object] = {}
    detail: dict[str, object] = {}
    table_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    for dataset in requested:
        item = data[dataset]
        targets = np.asarray(item["targets"], dtype=np.float32)
        conditions = np.asarray(item["conditions"], dtype=np.float32)
        groups = np.asarray(item["groups"]).astype(str)
        lead_names = ELEVEN_TARGET_LEADS if targets.shape[1] == 11 else None
        if groups.shape != (len(targets),) or conditions.shape[0] != len(targets):
            raise ValueError(f"{dataset} identities do not align with predictions")
        if any(
            np.asarray(item["predictions"][model]).shape != targets.shape
            or not np.all(np.isfinite(item["predictions"][model]))
            for model in MODELS
        ):
            raise ValueError(f"{dataset} predictions violate shape/finite requirements")

        train_source, train_target, calibration = _training_pair(workspace, dataset)
        lag = estimate_training_fixed_lag(train_source, train_target, args.max_lag_samples)
        lag["calibration"] = calibration
        dataset_result = {"raw_full_window": {}, "training_derived_fixed_lag": {}}
        dataset_detail = {
            "identity": {
                key: np.asarray(values).tolist()
                for key, values in item["identity"].items()
            },
            "models": {},
        }
        for model in MODELS:
            prediction = np.asarray(item["predictions"][model], dtype=np.float32)
            raw, raw_detail = raw_waveform_summary(targets, prediction, groups, lead_names)
            table_rows.append(_flatten_summary(dataset, model, raw))
            center, before, aligned = apply_fixed_lag(
                targets, prediction, int(lag["lag_samples"]), args.max_lag_samples
            )
            before_summary, _ = raw_waveform_summary(center, before, groups, lead_names)
            aligned_summary, _ = raw_waveform_summary(center, aligned, groups, lead_names)
            dataset_result["raw_full_window"][model] = raw
            dataset_result["training_derived_fixed_lag"][model] = {
                "unshifted_common_support": before_summary,
                "fixed_lag_common_support": aligned_summary,
                "rmse_change": aligned_summary["rmse"] - before_summary["rmse"],
                "mae_change": aligned_summary["mae"] - before_summary["mae"],
                "pearson_group_median_change": (
                    aligned_summary["pearson"]["group_median_primary"]
                    - before_summary["pearson"]["group_median_primary"]
                ),
            }
            lag_rows.append({
                "dataset": dataset,
                "model": model,
                "training_selected_lag_samples": lag["lag_samples"],
                "training_selected_lag_ms": lag["lag_samples"] * 1000.0 / 128.0,
                "support_samples": center.shape[-1],
                "unshifted_rmse": before_summary["rmse"],
                "fixed_lag_rmse": aligned_summary["rmse"],
                "unshifted_mae": before_summary["mae"],
                "fixed_lag_mae": aligned_summary["mae"],
                "unshifted_pearson_group_median": before_summary["pearson"]["group_median_primary"],
                "fixed_lag_pearson_group_median": aligned_summary["pearson"]["group_median_primary"],
            })
            dataset_detail["models"][model] = {
                "per_sample_pearson": raw_detail["per_sample_pearson"].tolist(),
                "per_group_pearson": dict(raw_detail["per_group_pearson"]),
            }
        dataset_result["training_lag_estimation"] = lag
        result[dataset] = dataset_result
        detail[dataset] = dataset_detail

    if "MIMIC-AFib" in data:
        mimic_identity = data["MIMIC-AFib"]["identity"]
        identity_summary = {
            "status": "completed",
            "subjects": int(len(set(mimic_identity["subject_ids"].tolist()))),
            "records": int(len(set(mimic_identity["record_ids"].tolist()))),
            "windows": int(len(mimic_identity["subject_ids"])),
            "afib_windows": int(np.sum(mimic_identity["afib"])),
            "non_afib_windows": int(np.sum(~mimic_identity["afib"])),
            "subject_disjoint_verified": True,
            "afib_subgroup_evaluation_available": bool(np.any(mimic_identity["afib"])),
        }
    else:
        identity_summary = {"status": "skipped_by_author_request"}
    summary_path = output / "five_dataset_waveform_statistics.json"
    detail_path = output / "pearson_group_details.json"
    table_path = output / "paper_waveform_table.csv"
    lag_path = output / "training_derived_fixed_lag_sensitivity.csv"
    _json(summary_path, {
        "schema_version": 1,
        "requested_datasets": list(requested),
        "metric_contract": {
            "support_primary": "raw_full_window_without_phase_correction",
            "mae_native_definition": (
                "native synchronized model-input normalized domain; not physical mV"
            ),
            "wfd_single_lead": "full-test-set waveform-vector Frechet distance",
            "wfd_multilead": (
                "independent full-test-set wFD per frozen target lead, then unweighted mean over "
                + ", ".join(ELEVEN_TARGET_LEADS)
            ),
            "pearson_primary": (
                "per four-second channel-time-flattened sample Pearson; Fisher-z mean within "
                "subject/patient/record; median over equal-weight groups"
            ),
            "fixed_lag": (
                "one global lag selected from training source-target pairs only and applied unchanged "
                "to test predictions on a common central support"
            ),
            "qrs_peak_to_peak_amplitude": "available offline in src.rcfm.metrics.clinical",
        },
        "split_audit": split_audit,
        "mimic_identity": identity_summary,
        "datasets": result,
    })
    _json(detail_path, detail)
    _write_rows(table_path, table_rows)
    _write_rows(lag_path, lag_rows)
    outputs = (summary_path, detail_path, table_path, lag_path)
    input_paths = sorted(
        {path.resolve() for item in data.values() for path in item["inputs"]}, key=str
    )
    _json(output / "protocol.json", {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "script_sha256": _sha256(Path(__file__)),
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": (
            "One-seed descriptive results. MIMIC is omitted unless explicitly requested; "
            "WESAD/MMECG have only three complete held-out subjects; CPSC lacks patient IDs. Fixed-lag sensitivity "
            "does not use test targets for lag selection and does not replace raw metrics."
        ),
    })
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DATASETS,
        help="Datasets to evaluate; MIMIC-AFib is excluded by default",
    )
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
