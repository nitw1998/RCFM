#!/usr/bin/env python3
"""Evaluate the completed seed-31 WESAD fixed-lag-v2 RDDM endpoint."""

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

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_ptbxl_fourway import _sample_rddm, _set_deterministic
from scripts.evaluate_wesad_fixed_lag_v2 import (
    EXPECTED_ALIGNMENT,
    EXPECTED_SPLIT,
    EXPECTED_VERSION,
    _load_v2_rows,
)
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


UPSTREAM_COMMIT = "7d5348843c3985c211a23ae5105a2d9497d5156a"


def validate_rddm_checkpoint(checkpoint: Mapping[str, object]) -> None:
    config = checkpoint.get("config", {})
    expected = {
        "task": "ppg2ecg",
        "datasets": ["WESAD"],
        "dataset_version": EXPECTED_VERSION,
        "split_hash": EXPECTED_SPLIT,
        "normalization_id": "window_minmax_neg1_1_v1",
        "alignment_id": EXPECTED_ALIGNMENT,
        "heldout_split": "test",
        "expected_train_windows": 17494,
        "expected_test_windows": 4213,
        "window_size": 4,
        "target_channels": 1,
        "attention_heads": 8,
        "nT": 10,
        "seed": 31,
        "reproduction_label": "RDDM-PPG (matched-protocol reproduction)",
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    provenance = checkpoint.get("provenance", {})
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != "independent_rddm_reproduction"
        or int(checkpoint.get("epoch", -1)) != 500
        or int(checkpoint.get("global_step", -1)) != 68500
        or provenance.get("upstream_commit") != UPSTREAM_COMMIT
        or bad
    ):
        raise ValueError("RDDM checkpoint violates fixed-lag-v2 contract: " + ", ".join(bad))
    if float(config.get("beta_start", -1)) != 1e-4 or float(config.get("beta_end", -1)) != 0.2:
        raise ValueError("RDDM checkpoint has the wrong beta schedule")


@torch.no_grad()
def generate(
    checkpoint: Mapping[str, object],
    conditions: np.ndarray,
    batch_size: int,
    sampling_seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    config = checkpoint["config"]
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=8),
        region_model=DiffusionUNetCrossAttention(512, 1, str(device), num_heads=8),
        betas=(float(config["beta_start"]), float(config["beta_end"])),
        n_T=int(config["nT"]),
    ).to(device)
    condition_1, condition_2 = ConditionNet().to(device), ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval(); condition_1.eval(); condition_2.eval()
    predictions = np.empty((len(conditions), 1, 512), dtype=np.float32)
    seeds: list[int] = []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        seed = sampling_seed + batch_index
        seeds.append(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        source = torch.as_tensor(conditions[start:stop], device=device)
        generated = _sample_rddm(
            model, condition_1(source), condition_2(source), (stop - start, 1, 512)
        )
        predictions[start:stop] = generated.cpu().numpy()
    del model, condition_1, condition_2
    torch.cuda.empty_cache(); gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    return predictions, seeds


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.max_lag_samples != 16 or args.sampling_rate != 128:
        raise ValueError("fixed-lag-v2 evaluation requires +/-16 samples at 128 Hz")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_rddm_checkpoint(checkpoint)
    normalization = checkpoint["normalization"]
    targets, conditions, subjects, labels = _load_v2_rows(args.data_root.resolve(), normalization)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal fixed-lag-v2 RDDM evaluation requires CUDA")
    _set_deterministic(args.deterministic_seed, device)
    predictions, sampling_seeds = generate(
        checkpoint, conditions, args.batch_size, args.sampling_seed, device
    )
    checkpoint_metadata = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "rddm_steps": int(checkpoint["config"]["nT"]),
        "sampling_batch_seeds": sampling_seeds,
    }
    del checkpoint

    raw_summary, _ = _waveform_metrics(targets, predictions)
    lag_summary, lag_values = _lag_diagnostic(targets, predictions, 16, 128)
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    center, before, after = _fixed_support_align(targets, predictions, shifts, 16)
    before_summary, _ = _waveform_metrics(center, before)
    after_summary, _ = _waveform_metrics(center, after)
    phase_summary = {
        "lag": lag_summary,
        "unshifted_fixed_support": before_summary,
        "oracle_aligned_fixed_support": after_summary,
        "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
        "boundary_hit_fraction": float(np.mean(np.abs(shifts) == 16)),
    }

    raw_path = output / "raw_predictions.npz"
    phase_path = output / "phase_predictions_maxlag16.npz"
    summary_path = output / "waveform_summary.json"
    np.savez_compressed(
        raw_path,
        targets=targets,
        conditions=conditions,
        subject_ids=subjects,
        labels=labels,
        rddm_predictions=predictions,
    )
    np.savez_compressed(
        phase_path,
        targets=center,
        subject_ids=subjects,
        labels=labels,
        rddm_oracle_shifts=shifts,
        rddm_unshifted_predictions=before,
        rddm_oracle_aligned_predictions=after,
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "training_seed": 31,
                "generation": checkpoint_metadata,
                "raw_full_window": {"rddm": raw_summary},
                "phase_sensitivity": {"rddm": phase_summary},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    outputs = (raw_path, phase_path, summary_path)
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "input": {"checkpoint": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "array_sha256": {
            "targets": _array_sha256(targets),
            "conditions": _array_sha256(conditions),
            "subject_ids": _array_sha256(subjects),
            "labels": _array_sha256(labels),
        },
        "protocol": {
            "dataset": "WESAD",
            "dataset_version": EXPECTED_VERSION,
            "alignment_id": EXPECTED_ALIGNMENT,
            "training_only_fixed_lag_samples": 36,
            "windows": 4213,
            "heldout_subjects": sorted(set(subjects)),
            "rddm_steps": 10,
            "sampling_seed_schedule": "2025 + batch_index",
            "phase_correction_target_informed": True,
            "max_lag_samples": 16,
            "fixed_support_samples": 480,
        },
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "device": str(device),
        },
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": (
            "Single training seed and three held-out subjects; oracle phase correction is "
            "target-informed; fixed-lag-v2 is not interchangeable with unaligned v1."
        ),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--sampling_seed", type=int, default=2025)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
