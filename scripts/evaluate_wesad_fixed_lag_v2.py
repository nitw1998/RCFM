#!/usr/bin/env python3
"""Evaluate WESAD fixed-lag-v2 RCFM and RCFM-OT on the frozen test rows."""

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
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (  # noqa: E402
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _sha256,
    _waveform_metrics,
)
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align  # noqa: E402
from scripts.evaluate_ptbxl_fourway import _set_deterministic  # noqa: E402
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic  # noqa: E402
from train_rcfm import build_datasets  # noqa: E402


MODELS = ("rcfm", "rcfm_ot")
EXPECTED_VERSION = "wesad-subject-fold1-train-fixed-lag-aligned-v2"
EXPECTED_SPLIT = "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd"
EXPECTED_ALIGNMENT = "train_subjects_peak_median_fixed_lag_crop_before_window_subject_fold1_v2"


def validate_contracts(contracts: Mapping[str, Mapping[str, object]]) -> None:
    if set(contracts) != set(MODELS):
        raise ValueError("contracts must contain exactly rcfm and rcfm_ot")
    expected = {
        "task": "ppg2ecg", "datasets": ["WESAD"], "dataset_version": EXPECTED_VERSION,
        "split_hash": EXPECTED_SPLIT, "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": EXPECTED_ALIGNMENT, "window_size": 4, "attention_heads": 8,
        "flow_matcher": "conditional", "sigma": 0.0, "region_weight": 0.01, "seed": 31,
    }
    reference = contracts["rcfm"]
    for model, contract in contracts.items():
        config = contract.get("config", {})
        bad = [key for key, value in expected.items() if config.get(key) != value]
        if contract.get("kind") != "canonical_multistep_rcfm":
            bad.append("kind")
        if int(contract.get("epoch", -1)) != 500 or int(contract.get("global_step", -1)) != 68500:
            bad.append("endpoint")
        if contract.get("output_spec", {}).get("channels") != 1 or contract.get("output_spec", {}).get("length") != 512:
            bad.append("output_spec")
        if contract.get("normalization") != reference.get("normalization"):
            bad.append("normalization")
        if bad:
            raise ValueError(f"{model} violates fixed-lag-v2 contract: {', '.join(bad)}")
    if bool(contracts["rcfm"]["config"].get("use_minibatch_ot")):
        raise ValueError("RCFM must have OT disabled")
    ot_config = contracts["rcfm_ot"]["config"]
    if not bool(ot_config.get("use_minibatch_ot")) or ot_config.get("ot_method") != "exact" or ot_config.get("ot_sampling_strategy") != "assignment":
        raise ValueError("RCFM-OT must use exact assignment OT")


def _load_v2_rows(data_root: Path, normalization: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    _, heldout = build_datasets(
        task="ppg2ecg", datasets=["WESAD"], data_root=str(data_root), window_size=4,
        normalization_metadata=dict(normalization), normalization_id="window_minmax_neg1_1_v1",
        load_train=False, heldout_split="test",
    )
    root = data_root / "WESAD"
    values = (
        np.asarray(heldout.target_ecg[:, None, :], dtype=np.float32),
        np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32),
        np.load(root / "subject_ids_test.npy", allow_pickle=False).astype(str),
        np.load(root / "labels_test.npy", allow_pickle=False).astype(np.int16),
    )
    if values[0].shape != (4213, 1, 512) or values[1].shape != values[0].shape:
        raise ValueError("fixed-lag-v2 waveform rows violate the frozen shape")
    return values


def _load_shared_noise(reference_path: Path, targets: np.ndarray, subjects: np.ndarray,
                       labels: np.ndarray) -> np.ndarray:
    with np.load(reference_path, allow_pickle=False) as artifact:
        required = {"targets", "subject_ids", "labels", "initial_flow_noise"}
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("shared-noise reference is missing: " + ", ".join(missing))
        reference_target = np.asarray(artifact["targets"], dtype=np.float32)
        reference_subjects = np.asarray(artifact["subject_ids"]).astype(str)
        reference_labels = np.asarray(artifact["labels"], dtype=np.int16)
        noise = np.asarray(artifact["initial_flow_noise"], dtype=np.float32)
    if not (np.array_equal(reference_target, targets) and np.array_equal(reference_subjects, subjects)
            and np.array_equal(reference_labels, labels)):
        raise ValueError("v2 target/identity rows do not match the frozen WESAD reference")
    if noise.shape != targets.shape or not np.all(np.isfinite(noise)):
        raise ValueError("shared flow noise violates the frozen shape or finite-value contract")
    return noise


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.inference_steps != 50 or args.max_lag_samples != 16 or args.sampling_rate != 128:
        raise ValueError("fixed-lag-v2 evaluation requires NFE=50 and +/-16 samples at 128 Hz")
    checkpoint_paths = {model: getattr(args, f"{model}_checkpoint").resolve() for model in MODELS}
    contracts = {model: _checkpoint_contract(path) for model, path in checkpoint_paths.items()}
    validate_contracts(contracts)
    targets, conditions, subjects, labels = _load_v2_rows(args.data_root.resolve(), contracts["rcfm"]["normalization"])
    noise = _load_shared_noise(args.reference_artifact.resolve(), targets, subjects, labels)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal fixed-lag-v2 evaluation requires CUDA")
    _set_deterministic(args.deterministic_seed, device)
    predictions, generation = {}, {}
    for model in MODELS:
        predictions[model], generation[model] = _generate(
            checkpoint_paths[model], "canonical_multistep_rcfm", conditions, noise,
            args.batch_size, args.inference_steps, device, args.deterministic_seed,
        )
        torch.cuda.empty_cache(); gc.collect()

    raw_summary, phase_summary = {}, {}
    phase_arrays: dict[str, np.ndarray] = {
        "targets": targets[:, :, 16:-16], "subject_ids": subjects, "labels": labels,
    }
    for model in MODELS:
        raw_summary[model], _ = _waveform_metrics(targets, predictions[model])
        lag_summary, lag_values = _lag_diagnostic(targets, predictions[model], 16, 128)
        shifts = lag_values["best_lag_samples"].astype(np.int32)
        center, before, after = _fixed_support_align(targets, predictions[model], shifts, 16)
        before_summary, _ = _waveform_metrics(center, before)
        after_summary, _ = _waveform_metrics(center, after)
        phase_summary[model] = {
            "lag": lag_summary, "unshifted_fixed_support": before_summary,
            "oracle_aligned_fixed_support": after_summary,
            "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
            "boundary_hit_fraction": float(np.mean(np.abs(shifts) == 16)),
        }
        phase_arrays[f"{model}_oracle_shifts"] = shifts
        phase_arrays[f"{model}_unshifted_predictions"] = before
        phase_arrays[f"{model}_oracle_aligned_predictions"] = after

    raw_path = output / "raw_predictions.npz"
    phase_path = output / "phase_predictions_maxlag16.npz"
    summary_path = output / "waveform_summary.json"
    np.savez_compressed(
        raw_path, targets=targets, conditions=conditions, subject_ids=subjects, labels=labels,
        initial_flow_noise=noise, **{f"{model}_predictions": predictions[model] for model in MODELS},
    )
    np.savez_compressed(phase_path, **phase_arrays)
    summary_path.write_text(json.dumps({
        "schema_version": 1, "training_seed": 31, "generation": generation,
        "raw_full_window": raw_summary, "phase_sensitivity": phase_summary,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs = (raw_path, phase_path, summary_path)
    protocol = {
        "schema_version": 1, "status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv), "models": list(MODELS),
        "inputs": {"reference": {"path": str(args.reference_artifact.resolve()), "sha256": _sha256(args.reference_artifact)},
                   "checkpoints": {model: {"path": str(path), "sha256": _sha256(path)} for model, path in checkpoint_paths.items()}},
        "array_sha256": {"targets": _array_sha256(targets), "conditions": _array_sha256(conditions),
                         "subject_ids": _array_sha256(subjects), "labels": _array_sha256(labels),
                         "initial_flow_noise": _array_sha256(noise)},
        "protocol": {"dataset": "WESAD", "dataset_version": EXPECTED_VERSION,
                     "alignment_id": EXPECTED_ALIGNMENT, "training_only_fixed_lag_samples": 36,
                     "windows": 4213, "heldout_subjects": sorted(set(subjects)), "flow_nfe": 50,
                     "flow_noise_seed": 2025, "phase_correction_target_informed": True,
                     "max_lag_samples": 16, "fixed_support_samples": 480},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                      "torch": torch.__version__, "device": str(device)},
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": "Three held-out subjects; oracle phase correction is diagnostic and target-informed; v2 is not interchangeable with v1.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
