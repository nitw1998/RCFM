"""Evaluate an epoch-200 random-window CFM-family checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _json,
    _sha256,
)
from scripts.evaluate_ptbxl_fourway import TARGET_INDICES, TARGET_LEADS, _metric_summary
from train_rcfm import build_datasets


SPECS = {
    "ptbxl": {
        "dataset": "PTBXL",
        "dataset_version": "ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2",
        "split_hash": "9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c",
        "records": 8735,
        "task": "ecg2ecg",
        "normalization_id": "source_record_joint12_minmax_neg1_1_v1",
        "split": "val",
        "target_leads": TARGET_LEADS,
        "condition_lead": "II",
    },
    "cpsc2018": {
        "dataset": "CPSC2018",
        "dataset_version": "cpsc2018-source-fullrecord-joint12-minmax-all-nonoverlap4s-random80-20-record-overlap-v1",
        "split_hash": "b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8",
        "records": 4842,
        "task": "ecg2ecg",
        "normalization_id": "source_record_joint12_minmax_neg1_1_v1",
        "split": "val",
        "target_leads": TARGET_LEADS,
        "condition_lead": "II",
    },
    "mmecg": {
        "dataset": "mmECG",
        "dataset_version": "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1",
        "split_hash": "6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2",
        "records": 2494,
        "task": "rcg2ecg",
        "normalization_id": "window_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("single_channel_ECG",),
        "condition_lead": "energy_weighted_RCG",
    },
    "wesad": {
        "dataset": "WESAD",
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-window-minmax-v1",
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "records": 4342,
        "task": "ppg2ecg",
        "normalization_id": "window_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("chest_ECG",),
        "condition_lead": "wrist_BVP",
    },
    "wesad_record_minmax": {
        "dataset": "WESAD",
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "records": 4342,
        "task": "ppg2ecg",
        "normalization_id": "source_record_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("chest_ECG",),
        "condition_lead": "wrist_BVP",
    },
    "mimic_afib": {
        "dataset": "MIMIC-AFib",
        "dataset_version": "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1",
        "split_hash": "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232",
        "records": 2040,
        "task": "ppg2ecg",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "split": "test",
        "target_leads": ("upstream_artifact_ecg_channel",),
        "condition_lead": "PPG",
    },
}

FLOW_MODELS = {
    "cfm": {
        "kind": "canonical_multistep_cfm",
        "region_weight": 0.0,
        "use_minibatch_ot": False,
    },
    "rcfm": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": False,
    },
    "rcfm_ot": {
        "kind": "canonical_multistep_rcfm",
        "region_weight": 0.01,
        "use_minibatch_ot": True,
    },
}


def fixed_validation_noise(
    records: int, channels: int, length: int, batch_size: int, seed: int
) -> np.ndarray:
    """Reproduce validate_epoch's independently seeded noise for every batch."""

    if min(records, channels, length, batch_size) <= 0:
        raise ValueError("noise dimensions and batch size must be positive")
    output = np.empty((records, channels, length), dtype=np.float32)
    for batch_index, start in enumerate(range(0, records, batch_size)):
        stop = min(start + batch_size, records)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + batch_index)
        output[start:stop] = torch.randn(
            (stop - start, channels, length), generator=generator, dtype=torch.float32
        ).numpy()
    return output


def _validate_contract(
    contract: dict[str, object], dataset_key: str, model_key: str = "cfm",
    expected_training_seed: int = 31,
) -> None:
    spec = SPECS[dataset_key]
    model_spec = FLOW_MODELS[model_key]
    config = contract["config"]
    expected = {
        "task": spec["task"],
        "datasets": [spec["dataset"]],
        "dataset_version": spec["dataset_version"],
        "split_hash": spec["split_hash"],
        "normalization_id": spec["normalization_id"],
        "flow_matcher": "conditional",
        "sigma": 0.0,
        "region_weight": model_spec["region_weight"],
        "use_minibatch_ot": model_spec["use_minibatch_ot"],
        "seed": expected_training_seed,
    }
    if dataset_key in {"ptbxl", "cpsc2018"}:
        expected.update({
            "condition_lead_index": 1,
            "target_lead_indices": list(TARGET_INDICES),
        })
    bad = [key for key, value in expected.items() if config.get(key) != value]
    output = contract["output_spec"]
    if contract.get("kind") != model_spec["kind"]:
        bad.append("kind")
    if int(contract.get("epoch", -1)) != 200:
        bad.append("epoch")
    if output.get("channels") != len(spec["target_leads"]) or output.get("length") != 512:
        bad.append("output_shape")
    if tuple(output.get("target_leads", ())) != tuple(spec["target_leads"]):
        bad.append("target_leads")
    if bad:
        raise ValueError(
            f"checkpoint violates random-window {model_key} contract: " + ", ".join(bad)
        )


def run(args: argparse.Namespace) -> Path:
    spec = SPECS[args.dataset]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(args.checkpoint.resolve())
    _validate_contract(contract, args.dataset, args.model, args.expected_training_seed)
    dataset_dir = args.data_root.resolve() / str(spec["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("dataset_version") != spec["dataset_version"]
        or manifest.get("split_hash") != spec["split_hash"]
        or int(manifest.get("splits", {}).get(spec["split"], {}).get("windows", -1)) != spec["records"]
    ):
        raise ValueError("dataset manifest violates random-window evaluation contract")
    config = contract["config"]
    dataset_kwargs = {}
    if args.dataset in {"ptbxl", "cpsc2018"}:
        dataset_kwargs = {
            "condition_lead_index": 1,
            "target_lead_indices": list(TARGET_INDICES),
        }
    _, heldout = build_datasets(
        config["task"], config["datasets"], str(args.data_root.resolve()), 4,
        normalization_metadata=contract["normalization"],
        normalization_id=config["normalization_id"], load_train=False,
        heldout_split=spec["split"], **dataset_kwargs,
    )
    expected_records = int(spec["records"])
    if len(heldout) != expected_records:
        raise ValueError(f"expected {expected_records} validation windows, found {len(heldout)}")
    targets = np.asarray(heldout.target_ecg, dtype=np.float32)
    if targets.ndim == 2:
        targets = targets[:, None, :]
    conditions = np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32)
    channels = len(spec["target_leads"])
    noise = fixed_validation_noise(len(targets), channels, 512, args.batch_size, args.noise_seed)
    if args.dataset == "mmecg":
        record_ids = np.load(dataset_dir / "source_files_test.npy", allow_pickle=False).astype(str)
        window_starts = np.load(
            dataset_dir / "source_record_window_ordinals_test.npy", allow_pickle=False
        ) * 256
    elif args.dataset in {"wesad", "wesad_record_minmax"}:
        subject_ids = np.load(dataset_dir / "subject_ids_test.npy", allow_pickle=False).astype(str)
        window_ordinals = np.load(
            dataset_dir / "subject_window_ordinals_test.npy", allow_pickle=False
        )
        record_ids = np.asarray([
            f"{subject}:window_{int(ordinal):06d}"
            for subject, ordinal in zip(subject_ids, window_ordinals)
        ])
        window_starts = window_ordinals * 512
    elif args.dataset == "mimic_afib":
        record_ids = np.load(dataset_dir / "record_ids_test.npy", allow_pickle=False).astype(str)
        window_starts = np.load(
            dataset_dir / "start_samples_128hz_test.npy", allow_pickle=False
        ).astype(np.int64)
    else:
        record_ids = np.load(dataset_dir / "record_ids_val.npy", allow_pickle=False).astype(str)
        window_starts = np.load(dataset_dir / "window_start_samples_val.npy", allow_pickle=False)
    reference = {
        "targets": targets, "conditions": conditions, "initial_flow_noise": noise,
        "record_ids": record_ids, "window_start_samples": window_starts,
    }
    if args.dataset == "ptbxl":
        reference["patient_ids"] = np.load(dataset_dir / "patient_ids_val.npy", allow_pickle=False).astype(str)
    if args.dataset == "mmecg":
        reference["subject_ids"] = np.load(
            dataset_dir / "subject_ids_test.npy", allow_pickle=False
        ).astype(str)
    if args.dataset in {"wesad", "wesad_record_minmax"}:
        reference["subject_ids"] = subject_ids
        reference["labels"] = np.load(
            dataset_dir / "labels_test.npy", allow_pickle=False
        ).astype(np.int16)
    if args.dataset == "mimic_afib":
        reference["subject_ids"] = np.load(
            dataset_dir / "subject_ids_test.npy", allow_pickle=False
        ).astype(str)
        reference["afib_labels"] = np.load(
            dataset_dir / "afib_labels_test.npy", allow_pickle=False
        ).astype(bool)
        reference["source_rows"] = np.load(
            dataset_dir / "source_rows_test.npy", allow_pickle=False
        ).astype(np.int32)
        reference["source_split_codes"] = np.load(
            dataset_dir / "source_split_codes_test.npy", allow_pickle=False
        ).astype(np.uint8)
    reference_path = output / "paired_reference.npz"
    np.savez_compressed(reference_path, **reference)
    device = torch.device(args.device)
    predictions, generation = _generate(
        args.checkpoint.resolve(), str(FLOW_MODELS[args.model]["kind"]), conditions, noise,
        args.batch_size, args.steps, device, args.deterministic_seed,
    )
    prediction_path = output / f"{args.model}_predictions.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    summary, per_window = _metric_summary(
        targets, predictions, target_leads=tuple(spec["target_leads"])
    )
    summary["waveform_fd_sum_channels"] = channels * summary["waveform_fd_macro_lead"]
    summary["waveform_fd_sum_definition"] = (
        f"sum of {channels} independent full-validation lead/channel FDs"
    )
    if channels == 11:
        summary["waveform_fd_sum_11_lead"] = summary["waveform_fd_sum_channels"]
    logged = contract["best_metrics"]
    summary["logged_checkpoint_best_metrics"] = logged
    _json(output / "waveform_summary.json", {"models": {args.model: summary}})
    np.savez_compressed(output / "per_window_waveform_metrics.npz", **per_window)
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "protocol": {
            "model": args.model,
            "training_seed": args.expected_training_seed,
            "dataset": spec["dataset"], "dataset_version": spec["dataset_version"],
            "split": "random_window_validation", "split_hash": spec["split_hash"],
            "available_records": expected_records, "evaluated_records": len(targets),
            "normalization_id": config["normalization_id"], "flow_nfe": args.steps,
            "flow_noise_seed": args.noise_seed, "noise_schedule": "seed_plus_batch_index",
            "batch_size": args.batch_size, "phase_correction_applied": False,
            "target_leads": list(spec["target_leads"]),
            "condition_lead": spec["condition_lead"],
        },
        "claim_boundaries": {
            "validation_only": True, "non_grouped_random_windows": True,
            "target_informed_joint12_scaling": args.dataset in {"ptbxl", "cpsc2018"},
            "patient_independent_generalization": False,
            "raw_sample_overlap_across_train_validation": args.dataset == "mmecg",
            "same_subject_in_train_and_validation": args.dataset in {"mmecg", "wesad", "wesad_record_minmax", "mimic_afib"},
            "same_continuous_recording_in_train_and_validation": args.dataset in {"wesad", "wesad_record_minmax", "mimic_afib"},
            "heldout_target_statistics_used_for_normalization": args.dataset == "wesad_record_minmax",
        },
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint), **generation},
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "artifacts": {
            "paired_reference_sha256": _sha256(reference_path),
            "prediction_file_sha256": _sha256(prediction_path),
            "prediction_array_sha256": _array_sha256(predictions),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    _json(output / "protocol.json", protocol)
    del predictions
    gc.collect()
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--model", choices=tuple(FLOW_MODELS), default="cfm")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument(
        "--expected_training_seed", type=int, choices=(31, 32, 33, 34, 35), default=31,
        help="Training seed required in the structured checkpoint contract.",
    )
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
