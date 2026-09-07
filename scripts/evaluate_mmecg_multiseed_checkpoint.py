"""Evaluate one mmECG training seed on the frozen paired test reference."""

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
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _sha256,
    _waveform_metrics,
)
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_mmecg_fourway import (
    MODELS,
    _subject_rows,
    _validate_flow_contracts,
    _validate_rddm_checkpoint,
    _write_csv,
)
from scripts.evaluate_ptbxl_fourway import _sample_rddm, _set_deterministic
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic
from train_rcfm import build_datasets


@torch.no_grad()
def _generate_rddm(
    checkpoint_path: Path,
    conditions: np.ndarray,
    batch_size: int,
    sampling_seed: int,
    device: torch.device,
    expected_training_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_rddm_checkpoint(checkpoint, expected_training_seed)
    config = checkpoint["config"]
    channels = int(config["target_channels"])
    length = int(config["window_size"]) * 128
    model = RDDM(
        eps_model=DiffusionUNetCrossAttention(
            length, channels, str(device), num_heads=int(config["attention_heads"])
        ),
        region_model=DiffusionUNetCrossAttention(
            length, channels, str(device), num_heads=int(config["attention_heads"])
        ),
        betas=(float(config["beta_start"]), float(config["beta_end"])),
        n_T=int(config["nT"]),
    ).to(device)
    condition_1, condition_2 = ConditionNet().to(device), ConditionNet().to(device)
    model.load_state_dict(checkpoint["rddm_state"], strict=True)
    condition_1.load_state_dict(checkpoint["condition_1_state"], strict=True)
    condition_2.load_state_dict(checkpoint["condition_2_state"], strict=True)
    model.eval(); condition_1.eval(); condition_2.eval()
    metadata = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
    }
    del checkpoint
    gc.collect()
    predictions = np.empty((len(conditions), channels, length), dtype=np.float32)
    batch_seeds = []
    for batch_index, start in enumerate(range(0, len(conditions), batch_size)):
        stop = min(start + batch_size, len(conditions))
        seed = sampling_seed + batch_index
        batch_seeds.append(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        source = torch.as_tensor(conditions[start:stop], device=device)
        generated = _sample_rddm(
            model,
            condition_1(source),
            condition_2(source),
            (stop - start, channels, length),
        )
        predictions[start:stop] = generated.cpu().numpy()
    del model, condition_1, condition_2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("RDDM predictions contain NaN or Inf")
    metadata["sampling_batch_seeds"] = batch_seeds
    return predictions, metadata


def _load_reference(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "targets", "conditions", "subject_ids", "source_files", "initial_flow_noise"
        }
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("frozen reference is missing: " + ", ".join(missing))
        arrays = tuple(np.asarray(artifact[key]) for key in (
            "targets", "conditions", "subject_ids", "source_files", "initial_flow_noise"
        ))
    targets, conditions, subjects, sources, noise = arrays
    expected = (2877, 1, 512)
    if any(value.shape != expected for value in (targets, conditions, noise)):
        raise ValueError("frozen mmECG reference arrays have the wrong shape")
    if subjects.shape != (2877,) or sources.shape != (2877,):
        raise ValueError("frozen mmECG reference identities have the wrong shape")
    if any(not np.all(np.isfinite(value)) for value in (targets, conditions, noise)):
        raise FloatingPointError("frozen mmECG reference contains NaN or Inf")
    return targets.astype(np.float32), conditions.astype(np.float32), subjects.astype(str), sources.astype(str), noise.astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.max_lag_samples != 16 or args.sampling_rate != 128:
        raise ValueError("response protocol requires +/-16 samples at 128 Hz")
    paths = {name: getattr(args, f"{name}_checkpoint").resolve() for name in MODELS}
    contracts = {name: _checkpoint_contract(paths[name]) for name in ("cfm", "rcfm", "rcfm_ot")}
    _validate_flow_contracts(contracts, args.expected_training_seed)
    rddm = torch.load(paths["rddm"], map_location="cpu")
    _validate_rddm_checkpoint(rddm, args.expected_training_seed)
    rddm_steps = int(rddm["config"]["nT"])
    del rddm

    targets, conditions, subjects, sources, noise = _load_reference(args.reference_artifact.resolve())
    _, heldout = build_datasets(
        task="rcg2ecg", datasets=["mmECG"], data_root=str(args.data_root.resolve()),
        window_size=4, normalization_metadata=contracts["cfm"]["normalization"],
        normalization_id="window_minmax_neg1_1_v1", load_train=False, heldout_split="test",
    )
    loaded = (
        np.asarray(heldout.target_ecg[:, None, :], dtype=np.float32),
        np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32),
        np.load(args.data_root / "mmECG/subject_ids_test.npy", allow_pickle=False).astype(str),
        np.load(args.data_root / "mmECG/source_files_test.npy", allow_pickle=False).astype(str),
    )
    if not all(np.array_equal(a, b) for a, b in zip((targets, conditions, subjects, sources), loaded)):
        raise ValueError("frozen reference rows do not match the current held-out dataset")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    _set_deterministic(args.deterministic_seed, device)
    predictions: dict[str, np.ndarray] = {}
    generation: dict[str, object] = {}
    kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    for model in ("cfm", "rcfm", "rcfm_ot"):
        predictions[model], generation[model] = _generate(
            paths[model], kinds[model], conditions, noise, args.batch_size,
            args.inference_steps, device, args.deterministic_seed,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    predictions["rddm"], generation["rddm"] = _generate_rddm(
        paths["rddm"], conditions, args.batch_size, args.rddm_sampling_seed,
        device, args.expected_training_seed,
    )

    raw_summary, raw_rows, aligned_summary, aligned_rows = {}, {}, {}, {}
    phase_summary: dict[str, object] = {}
    phase_arrays: dict[str, np.ndarray] = {
        "targets": targets[:, :, args.max_lag_samples:-args.max_lag_samples],
        "subject_ids": subjects,
        "source_files": sources,
    }
    per_window_rows = []
    for model in MODELS:
        raw_summary[model], raw_rows[model] = _waveform_metrics(targets, predictions[model])
        lag_summary, lag_values = _lag_diagnostic(
            targets, predictions[model], args.max_lag_samples, args.sampling_rate
        )
        shifts = lag_values["best_lag_samples"].astype(np.int32)
        center, before, after = _fixed_support_align(
            targets, predictions[model], shifts, args.max_lag_samples
        )
        before_summary, _ = _waveform_metrics(center, before)
        after_summary, aligned_rows[model] = _waveform_metrics(center, after)
        aligned_summary[model] = after_summary
        phase_summary[model] = {
            "lag": lag_summary,
            "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
            "boundary_hit_fraction": float(np.mean(np.abs(shifts) == args.max_lag_samples)),
            "unshifted_fixed_support": before_summary,
            "oracle_aligned_fixed_support": after_summary,
        }
        phase_arrays[f"{model}_oracle_shifts"] = shifts
        phase_arrays[f"{model}_unshifted_predictions"] = before
        phase_arrays[f"{model}_oracle_aligned_predictions"] = after
        for index in range(len(targets)):
            per_window_rows.append({
                "window": index, "subject_id": subjects[index], "source_file": sources[index],
                "model": model, "oracle_shift_samples": int(shifts[index]),
                "aligned_rmse": float(aligned_rows[model]["rmse"][index]),
                "aligned_mae": float(aligned_rows[model]["mae"][index]),
                "aligned_pearson_r": float(aligned_rows[model]["pearson_r"][index]),
            })

    raw_path = output / "raw_predictions.npz"
    phase_path = output / "phase_predictions_maxlag16.npz"
    np.savez_compressed(
        raw_path, targets=targets, conditions=conditions, subject_ids=subjects,
        source_files=sources, initial_flow_noise=noise,
        **{f"{model}_predictions": predictions[model] for model in MODELS},
    )
    np.savez_compressed(phase_path, **phase_arrays)
    per_window_path = output / "per_window_phase_metrics.csv"
    subject_path = output / "per_subject_waveform_metrics.csv"
    _write_csv(per_window_path, per_window_rows)
    _write_csv(subject_path, _subject_rows(subjects, raw_rows, aligned_rows))
    summary_path = output / "waveform_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "training_seed": args.expected_training_seed,
        "protocol": {
            "windows": 2877, "subjects": sorted(set(subjects)), "sampling_rate_hz": 128,
            "phase_correction": "target-informed Pearson-maximizing +/-16-sample shift on common 480-sample support",
        },
        "generation": generation,
        "raw_full_window": raw_summary,
        "phase_sensitivity": phase_summary,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs = (raw_path, phase_path, per_window_path, subject_path, summary_path)
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "training_seed": args.expected_training_seed,
        "inputs": {
            "reference": {"path": str(args.reference_artifact.resolve()), "sha256": _sha256(args.reference_artifact)},
            "checkpoints": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        },
        "array_sha256": {
            "targets": _array_sha256(targets), "conditions": _array_sha256(conditions),
            "initial_flow_noise": _array_sha256(noise),
        },
        "protocol": {
            "dataset": "mmECG", "split": "heldout_subject_test", "windows": 2877,
            "sampling_rate_hz": 128, "phase_correction_applied": True,
            "phase_correction_target_informed": True, "max_lag_samples": 16,
            "fixed_support_samples": 480, "flow_nfe": args.inference_steps,
            "rddm_steps": rddm_steps, "flow_noise_seed": 2025,
            "rddm_sampling_seed": args.rddm_sampling_seed,
        },
        "execution": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__, "device": str(device),
        },
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
        "claim_boundary": "Oracle phase correction uses each held-out ECG target. Overlapping windows and only three subjects are not independent replicates.",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    for model in MODELS:
        parser.add_argument(f"--{model}_checkpoint", type=Path, required=True)
    parser.add_argument("--reference_artifact", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_training_seed", type=int, choices=(32, 33), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--rddm_sampling_seed", type=int, default=2025)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
