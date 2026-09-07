"""Evaluate a frozen random-window RDDM checkpoint on one of five datasets."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _json, _sha256
from scripts.evaluate_mimic_rddm_predictions import _generate_batches, _set_deterministic
from scripts.evaluate_ptbxl_fourway import TARGET_INDICES, TARGET_LEADS, _metric_summary
from train_rcfm import build_datasets


UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"
DATASET_SPECS = {
    "ptbxl": {
        "dataset": "PTBXL", "task": "ecg2ecg",
        "dataset_version": "ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2",
        "split_hash": "9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c",
        "normalization_id": "source_record_joint12_minmax_neg1_1_v1",
        "alignment_id": "ptbxl_all_records_two_nonoverlap_4s_windows_random80_20_seed31_lead_II_to_other11_v1",
        "train_rows": 34939, "heldout_rows": 8735, "global_step_epoch_200": 54600,
        "reproduction_label": "RDDM-ECG (random-window adapted)",
        "target_lead": TARGET_LEADS, "condition_lead": "II", "heldout_split": "val",
    },
    "cpsc2018": {
        "dataset": "CPSC2018", "task": "ecg2ecg",
        "dataset_version": "cpsc2018-source-fullrecord-joint12-minmax-all-nonoverlap4s-random80-20-record-overlap-v1",
        "split_hash": "b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8",
        "normalization_id": "source_record_joint12_minmax_neg1_1_v1",
        "alignment_id": "cpsc2018_all_complete_nonoverlap_4s_windows_random80_20_seed31_lead_II_to_other11_v1",
        "train_rows": 19364, "heldout_rows": 4842, "global_step_epoch_200": 30400,
        "reproduction_label": "RDDM-ECG (random-window adapted)",
        "target_lead": TARGET_LEADS, "condition_lead": "II", "heldout_split": "val",
    },
    "mimic_afib": {
        "dataset": "MIMIC-AFib", "task": "ppg2ecg",
        "dataset_version": "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1",
        "split_hash": "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232",
        "normalization_id": "rddm_window_minmax_neg1_1_v1",
        "alignment_id": "paired_source_row_no_phase_correction_random80_20_v1",
        "train_rows": 8160, "heldout_rows": 2040, "global_step_epoch_200": 12800,
        "reproduction_label": "RDDM (reproduced)",
        "target_lead": ("upstream_artifact_ecg_channel",), "condition_lead": "PPG",
        "heldout_split": "test",
    },
    "wesad_record_minmax": {
        "dataset": "WESAD", "task": "ppg2ecg",
        "dataset_version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "split_hash": "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
        "normalization_id": "source_record_minmax_neg1_1_v1",
        "alignment_id": "native_common_start_same_window_no_delay_correction_random80_20_v1",
        "train_rows": 17365, "heldout_rows": 4342, "global_step_epoch_200": 27200,
        "reproduction_label": "RDDM-PPG (random-window record-minmax adaptation)",
        "target_lead": ("chest_ECG",), "condition_lead": "wrist_BVP",
        "heldout_split": "test",
    },
    "mmecg": {
        "dataset": "mmECG", "task": "rcg2ecg",
        "dataset_version": "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1",
        "split_hash": "6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2",
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": "same_record_same_window_no_delay_correction_random80_20_v1",
        "train_rows": 9973, "heldout_rows": 2494, "global_step_epoch_200": 15600,
        "reproduction_label": "RDDM-RCG (random-window adapted)",
        "target_lead": ("single_channel_ECG",), "condition_lead": "energy_weighted_RCG",
        "heldout_split": "test",
    },
}
DATASET_VERSION = DATASET_SPECS["mimic_afib"]["dataset_version"]
SPLIT_HASH = DATASET_SPECS["mimic_afib"]["split_hash"]
EXPECTED_ROWS = DATASET_SPECS["mimic_afib"]["heldout_rows"]


def validate_checkpoint(
    payload: Mapping[str, object], dataset_key: str = "mimic_afib",
    expected_training_seed: int = 31, expected_epoch: int = 200,
) -> None:
    spec = DATASET_SPECS[dataset_key]
    required = {
        "kind", "epoch", "global_step", "rddm_state", "condition_1_state",
        "condition_2_state", "config", "normalization", "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("RDDM checkpoint is missing: " + ", ".join(missing))
    config = payload["config"]
    if not isinstance(config, Mapping):
        raise ValueError("RDDM checkpoint config must be a mapping")
    expected = {
        "task": spec["task"],
        "datasets": [spec["dataset"]],
        "dataset_version": spec["dataset_version"],
        "split_hash": spec["split_hash"],
        "normalization_id": spec["normalization_id"],
        "alignment_id": spec["alignment_id"],
        "expected_train_windows": spec["train_rows"],
        "expected_test_windows": spec["heldout_rows"],
        "heldout_split": spec["heldout_split"],
        "nT": 10,
        "attention_heads": 8,
        "seed": expected_training_seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "reproduction_label": spec["reproduction_label"],
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if payload.get("kind") != "independent_rddm_reproduction":
        bad.append("kind")
    expected_global_step = int(spec["global_step_epoch_200"]) * expected_epoch // 200
    if (
        expected_epoch not in (200, 400)
        or int(payload.get("epoch", -1)) != expected_epoch
        or int(payload.get("global_step", -1)) != expected_global_step
    ):
        bad.append("epoch/global_step")
    target_channels = len(spec["target_lead"])
    if int(config.get("target_channels", -1)) != target_channels:
        bad.append("target_channels")
    if dataset_key in {"ptbxl", "cpsc2018"} and (
        config.get("condition_lead_index") != 1
        or config.get("target_lead_indices") != list(TARGET_INDICES)
    ):
        bad.append("lead_indices")
    if float(config.get("beta_start", -1)) != 1e-4 or float(config.get("beta_end", -1)) != 0.2:
        bad.append("betas")
    if bad:
        raise ValueError("checkpoint violates random-window RDDM contract: " + ", ".join(bad))


def run(args: argparse.Namespace) -> Path:
    spec = DATASET_SPECS[args.dataset]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint(
        checkpoint, args.dataset, args.expected_training_seed, args.expected_epoch
    )
    config = dict(checkpoint["config"])
    dataset_dir = args.data_root.resolve() / str(spec["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("dataset_version") != spec["dataset_version"]
        or manifest.get("split_hash") != spec["split_hash"]
        or manifest.get("splits", {}).get(str(spec["heldout_split"]), {}).get("windows") != spec["heldout_rows"]
    ):
        raise ValueError("dataset manifest violates random-window RDDM contract")
    dataset_kwargs = {}
    if args.dataset in {"ptbxl", "cpsc2018"}:
        dataset_kwargs = {
            "condition_lead_index": 1,
            "target_lead_indices": list(TARGET_INDICES),
        }
    _, heldout = build_datasets(
        spec["task"], [spec["dataset"]], str(args.data_root.resolve()), 4,
        normalization_metadata=dict(checkpoint["normalization"]),
        normalization_id=spec["normalization_id"],
        load_train=False,
        heldout_split=str(spec["heldout_split"]), **dataset_kwargs,
    )
    targets = np.asarray(heldout.target_ecg, dtype=np.float32)
    if targets.ndim == 2:
        targets = targets[:, None, :]
    conditions = np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32)
    expected_shape = (int(spec["heldout_rows"]), len(spec["target_lead"]), 512)
    if targets.shape != expected_shape or conditions.shape != (expected_shape[0], 1, 512):
        raise ValueError(
            f"target/condition shapes must be {expected_shape} and {(expected_shape[0], 1, 512)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    _set_deterministic(args.deterministic_seed, device)
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(
            512, len(spec["target_lead"]), str(device), num_heads=8
        ),
        region_model=DiffusionUNetCrossAttention(
            512, len(spec["target_lead"]), str(device), num_heads=8
        ),
        betas=(1e-4, 0.2),
        n_T=10,
    ).to(device)
    condition_1 = ConditionNet().to(device)
    condition_2 = ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval()
    condition_1.eval()
    condition_2.eval()
    del checkpoint
    gc.collect()
    predictions, batch_indices, batch_seeds = _generate_batches(
        model, condition_1, condition_2, conditions, args.batch_size,
        args.sampling_seed, device, output_channels=len(spec["target_lead"]),
    )
    del model, condition_1, condition_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "ptbxl":
        reference = {
            "record_ids": np.load(dataset_dir / "record_ids_val.npy", allow_pickle=False).astype(str),
            "patient_ids": np.load(dataset_dir / "patient_ids_val.npy", allow_pickle=False).astype(str),
            "window_start_samples": np.load(dataset_dir / "window_start_samples_val.npy", allow_pickle=False).astype(np.int64),
        }
    elif args.dataset == "cpsc2018":
        reference = {
            "record_ids": np.load(dataset_dir / "record_ids_val.npy", allow_pickle=False).astype(str),
            "window_start_samples": np.load(dataset_dir / "window_start_samples_val.npy", allow_pickle=False).astype(np.int64),
        }
    elif args.dataset == "mimic_afib":
        reference = {
            "record_ids": np.load(dataset_dir / "record_ids_test.npy", allow_pickle=False).astype(str),
            "subject_ids": np.load(dataset_dir / "subject_ids_test.npy", allow_pickle=False).astype(str),
            "window_start_samples": np.load(dataset_dir / "start_samples_128hz_test.npy", allow_pickle=False).astype(np.int64),
            "afib_labels": np.load(dataset_dir / "afib_labels_test.npy", allow_pickle=False).astype(bool),
            "source_rows": np.load(dataset_dir / "source_rows_test.npy", allow_pickle=False).astype(np.int32),
            "source_split_codes": np.load(dataset_dir / "source_split_codes_test.npy", allow_pickle=False).astype(np.uint8),
        }
    elif args.dataset == "wesad_record_minmax":
        subjects = np.load(dataset_dir / "subject_ids_test.npy", allow_pickle=False).astype(str)
        ordinals = np.load(dataset_dir / "subject_window_ordinals_test.npy", allow_pickle=False).astype(np.int64)
        reference = {
            "subject_ids": subjects,
            "record_ids": np.asarray([f"{s}:window_{o:06d}" for s, o in zip(subjects, ordinals)]),
            "window_start_samples": ordinals * 512,
            "labels": np.load(dataset_dir / "labels_test.npy", allow_pickle=False).astype(np.int16),
        }
    else:
        ordinals = np.load(dataset_dir / "source_record_window_ordinals_test.npy", allow_pickle=False).astype(np.int64)
        reference = {
            "record_ids": np.load(dataset_dir / "source_files_test.npy", allow_pickle=False).astype(str),
            "subject_ids": np.load(dataset_dir / "subject_ids_test.npy", allow_pickle=False).astype(str),
            "window_start_samples": ordinals * 256,
        }
    reference.update({"targets": targets, "conditions": conditions})
    reference_path = output / "paired_reference.npz"
    np.savez_compressed(reference_path, **reference)
    prediction_path = output / "rddm_predictions.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    summary, per_window = _metric_summary(
        targets, predictions, target_leads=tuple(spec["target_lead"])
    )
    _json(output / "waveform_summary.json", {"models": {"rddm": summary}})
    np.savez_compressed(
        output / "per_window_waveform_metrics.npz",
        **per_window,
        sampling_batch_index=batch_indices,
        sampling_batch_seeds=batch_seeds,
    )
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "model": "rddm",
            "training_seed": args.expected_training_seed,
            "dataset": spec["dataset"],
            "dataset_version": spec["dataset_version"],
            "split": "random_window_validation",
            "split_hash": spec["split_hash"],
            "available_records": spec["heldout_rows"],
            "evaluated_records": spec["heldout_rows"],
            "normalization_id": spec["normalization_id"],
            "diffusion_steps": 10,
            "sampling_seed_base": args.sampling_seed,
            "batch_seed_rule": "sampling_seed_base + zero_based_batch_index",
            "batch_size": args.batch_size,
            "phase_correction_applied": False,
            "target_leads": list(spec["target_lead"]),
            "condition_lead": spec["condition_lead"],
        },
        "claim_boundaries": {
            "validation_only": True,
            "non_grouped_random_windows": True,
            "same_subject_in_train_and_validation": args.dataset not in {"ptbxl", "cpsc2018"},
            "same_continuous_recording_in_train_and_validation": args.dataset in {"mimic_afib", "wesad_record_minmax"},
            "raw_sample_overlap_across_train_validation": args.dataset == "mmecg",
            "patient_independent_generalization": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": args.expected_epoch,
            "global_step": int(spec["global_step_epoch_200"]) * args.expected_epoch // 200,
        },
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "artifacts": {
            "paired_reference_sha256": _sha256(reference_path),
            "prediction_file_sha256": _sha256(prediction_path),
            "prediction_array_sha256": _array_sha256(predictions),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _json(output / "protocol.json", protocol)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), default="mimic_afib")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sampling_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument(
        "--expected_training_seed", type=int, choices=(31, 32, 33, 34, 35), default=31
    )
    parser.add_argument("--expected_epoch", type=int, choices=(200, 400), default=200)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
