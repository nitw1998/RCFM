"""Complete deterministic four-model evaluation on the frozen mmECG split."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import (
    _array_sha256,
    _checkpoint_contract,
    _generate,
    _sha256,
    _waveform_metrics,
)
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_ptbxl_fourway import _set_deterministic
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic
from train_rcfm import build_datasets


MODELS = ("cfm", "rcfm", "rcfm_ot", "rddm")
EXPECTED_DATASET_VERSION = "mmecg-public-20221108-subject-split-window-minmax-v1"
EXPECTED_SPLIT_HASH = "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f"
EXPECTED_ALIGNMENT = "same_record_same_window_no_additional_phase_correction_subject_split_v1"


def _validate_flow_contracts(
    contracts: Mapping[str, Mapping[str, object]], expected_training_seed: int = 31
) -> None:
    if set(contracts) != {"cfm", "rcfm", "rcfm_ot"}:
        raise ValueError("flow contracts must contain cfm, rcfm, and rcfm_ot")
    kinds = {
        "cfm": "canonical_multistep_cfm",
        "rcfm": "canonical_multistep_rcfm",
        "rcfm_ot": "canonical_multistep_rcfm",
    }
    reference = contracts["cfm"]
    matched = (
        "task", "datasets", "dataset_version", "split_hash", "normalization_id",
        "alignment_id", "window_size", "attention_heads", "flow_matcher", "sigma", "seed",
    )
    for model, contract in contracts.items():
        if contract.get("kind") != kinds[model]:
            raise ValueError(f"{model} has the wrong checkpoint kind")
        if int(contract.get("epoch", -1)) != 500 or int(contract.get("global_step", -1)) != 37500:
            raise ValueError(f"{model} is not the frozen epoch-500 endpoint")
        config = contract["config"]
        mismatched = [key for key in matched if config.get(key) != reference["config"].get(key)]
        if mismatched:
            raise ValueError(f"{model} differs on frozen fields: {', '.join(mismatched)}")
        if contract["normalization"] != reference["normalization"]:
            raise ValueError(f"{model} normalization metadata differs")
        if contract["output_spec"] != reference["output_spec"]:
            raise ValueError(f"{model} output specification differs")
    expected = {
        "task": "rcg2ecg", "datasets": ["mmECG"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "window_size": 4, "seed": expected_training_seed,
    }
    bad = [key for key, value in expected.items() if reference["config"].get(key) != value]
    output = reference["output_spec"]
    if bad or output.get("channels") != 1 or output.get("length") != 512:
        raise ValueError("flow checkpoints violate the frozen mmECG contract: " + ", ".join(bad))
    cfm, rcfm, rcfm_ot = (contracts[name]["config"] for name in ("cfm", "rcfm", "rcfm_ot"))
    if float(cfm.get("region_weight", -1)) != 0 or bool(cfm.get("use_minibatch_ot")):
        raise ValueError("CFM must have region_weight=0 and OT disabled")
    if float(rcfm.get("region_weight", 0)) <= 0 or bool(rcfm.get("use_minibatch_ot")):
        raise ValueError("RCFM must have positive region weight and OT disabled")
    if float(rcfm_ot.get("region_weight", 0)) != float(rcfm.get("region_weight")):
        raise ValueError("RCFM and RCFM-OT region weights must match")
    if not bool(rcfm_ot.get("use_minibatch_ot")) or rcfm_ot.get("ot_method") != "exact":
        raise ValueError("RCFM-OT must use exact minibatch OT")


def _validate_rddm_checkpoint(
    checkpoint: Mapping[str, object], expected_training_seed: int = 31
) -> None:
    required = {
        "schema_version", "kind", "epoch", "global_step", "config", "normalization", "provenance"
    }
    if missing := sorted(required - set(checkpoint)):
        raise ValueError("RDDM checkpoint is missing: " + ", ".join(missing))
    config = checkpoint["config"]
    expected = {
        "task": "rcg2ecg", "datasets": ["mmECG"],
        "dataset_version": EXPECTED_DATASET_VERSION, "split_hash": EXPECTED_SPLIT_HASH,
        "normalization_id": "window_minmax_neg1_1_v1", "alignment_id": EXPECTED_ALIGNMENT,
        "heldout_split": "test", "expected_train_windows": 9590,
        "expected_test_windows": 2877, "window_size": 4, "target_channels": 1,
        "nT": 10, "seed": expected_training_seed, "reproduction_label": "RDDM-RCG (adapted)",
    }
    bad = [key for key, value in expected.items() if config.get(key) != value]
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("kind") != "independent_rddm_reproduction"
        or int(checkpoint.get("epoch", -1)) != 500
        or int(checkpoint.get("global_step", -1)) != 37500
        or bad
    ):
        raise ValueError("RDDM checkpoint violates the frozen mmECG contract: " + ", ".join(bad))
    if checkpoint["provenance"].get("upstream_commit") != "7d5348843c3985c211a23ae5105a2d9497d5156a":
        raise ValueError("RDDM upstream commit changed")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _subject_rows(
    subjects: np.ndarray,
    raw_rows: Mapping[str, Mapping[str, np.ndarray]],
    aligned_rows: Mapping[str, Mapping[str, np.ndarray]],
) -> list[dict[str, object]]:
    rows = []
    for model in MODELS:
        for subject in sorted(set(subjects)):
            take = subjects == subject
            for phase, values in (("raw", raw_rows[model]), ("oracle_aligned", aligned_rows[model])):
                rows.append(
                    {
                        "model": model,
                        "subject_id": subject,
                        "phase_mode": phase,
                        "windows": int(np.sum(take)),
                        "rmse_mean": float(np.mean(values["rmse"][take])),
                        "mae_mean": float(np.mean(values["mae"][take])),
                        "pearson_median": float(np.nanmedian(values["pearson_r"][take])),
                        "inference_status": "descriptive_only_n3_subjects_overlapping_windows",
                    }
                )
    return rows


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: getattr(args, f"{name}_checkpoint").resolve() for name in MODELS}
    contracts = {name: _checkpoint_contract(paths[name]) for name in ("cfm", "rcfm", "rcfm_ot")}
    _validate_flow_contracts(contracts)
    rddm_checkpoint = torch.load(paths["rddm"], map_location="cpu")
    _validate_rddm_checkpoint(rddm_checkpoint)
    del rddm_checkpoint

    with np.load(args.frozen_two_model_predictions, allow_pickle=False) as old:
        required = {
            "targets", "conditions", "subject_ids", "source_files", "initial_flow_noise",
            "rcfm_predictions", "rddm_predictions",
        }
        if missing := sorted(required - set(old.files)):
            raise ValueError("frozen two-model artifact is missing: " + ", ".join(missing))
        targets = np.asarray(old["targets"], dtype=np.float32)
        conditions = np.asarray(old["conditions"], dtype=np.float32)
        subjects = np.asarray(old["subject_ids"]).astype(str)
        sources = np.asarray(old["source_files"]).astype(str)
        noise = np.asarray(old["initial_flow_noise"], dtype=np.float32)
        frozen = {name: np.asarray(old[f"{name}_predictions"], dtype=np.float32) for name in ("rcfm", "rddm")}
    expected_shape = (2877, 1, 512)
    arrays = [targets, conditions, noise, *frozen.values()]
    if any(value.shape != expected_shape for value in arrays) or subjects.shape != (2877,) or sources.shape != (2877,):
        raise ValueError("frozen mmECG arrays violate the expected 2877x1x512 contract")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("frozen mmECG arrays contain NaN or Inf")

    _, heldout = build_datasets(
        task="rcg2ecg", datasets=["mmECG"], data_root=str(args.data_root.resolve()),
        window_size=4, normalization_metadata=contracts["cfm"]["normalization"],
        normalization_id="window_minmax_neg1_1_v1", load_train=False, heldout_split="test",
    )
    loaded_targets = np.asarray(heldout.target_ecg[:, None, :], dtype=np.float32)
    loaded_conditions = np.asarray(heldout.condition_signal[:, None, :], dtype=np.float32)
    loaded_subjects = np.load(args.data_root / "mmECG/subject_ids_test.npy", allow_pickle=False).astype(str)
    loaded_sources = np.load(args.data_root / "mmECG/source_files_test.npy", allow_pickle=False).astype(str)
    if not all(
        np.array_equal(a, b)
        for a, b in (
            (targets, loaded_targets), (conditions, loaded_conditions),
            (subjects, loaded_subjects), (sources, loaded_sources),
        )
    ):
        raise ValueError("frozen prediction rows do not match the current held-out artifact")

    device = torch.device(args.device)
    _set_deterministic(31, device)
    predictions = dict(frozen)
    generation = {}
    for model in ("cfm", "rcfm_ot"):
        predictions[model], generation[model] = _generate(
            paths[model], contracts[model]["kind"], conditions, noise,
            args.batch_size, args.inference_steps, device, 31,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    generation.update(
        {
            "rcfm": {"source": "reused_frozen_two_model_artifact", "sampling_seed": 2025},
            "rddm": {"source": "reused_frozen_two_model_artifact", "batch_seed_start": 2025},
        }
    )

    raw_summary, raw_rows, aligned_summary, aligned_rows = {}, {}, {}, {}
    phase_summary: dict[str, object] = {}
    phase_arrays: dict[str, np.ndarray] = {
        "targets": targets[:, :, args.max_lag_samples : -args.max_lag_samples],
        "subject_ids": subjects,
        "source_files": sources,
    }
    per_window_rows = []
    for model in MODELS:
        raw_summary[model], raw_rows[model] = _waveform_metrics(targets, predictions[model])
        lag_summary, lag_values = _lag_diagnostic(targets, predictions[model], args.max_lag_samples, 128)
        shifts = lag_values["best_lag_samples"].astype(np.int32)
        center, before, after = _fixed_support_align(targets, predictions[model], shifts, args.max_lag_samples)
        before_summary, before_rows = _waveform_metrics(center, before)
        after_summary, after_rows = _waveform_metrics(center, after)
        aligned_summary[model], aligned_rows[model] = after_summary, after_rows
        phase_summary[model] = {
            "lag": lag_summary,
            "median_absolute_shift_samples": float(np.median(np.abs(shifts))),
            "boundary_hit_fraction": float(np.mean(np.abs(shifts) == args.max_lag_samples)),
            "fraction_rmse_improved": float(np.mean(after_rows["rmse"] < before_rows["rmse"])),
            "unshifted_fixed_support": before_summary,
            "oracle_aligned_fixed_support": after_summary,
        }
        phase_arrays[f"{model}_oracle_shifts"] = shifts
        phase_arrays[f"{model}_unshifted_predictions"] = before
        phase_arrays[f"{model}_oracle_aligned_predictions"] = after
        for index in range(len(targets)):
            per_window_rows.append(
                {
                    "window": index, "subject_id": subjects[index], "source_file": sources[index],
                    "model": model, "raw_rmse": float(raw_rows[model]["rmse"][index]),
                    "raw_mae": float(raw_rows[model]["mae"][index]),
                    "raw_pearson_r": float(raw_rows[model]["pearson_r"][index]),
                    "oracle_shift_samples": int(shifts[index]),
                    "aligned_rmse": float(after_rows["rmse"][index]),
                    "aligned_mae": float(after_rows["mae"][index]),
                    "aligned_pearson_r": float(after_rows["pearson_r"][index]),
                }
            )

    raw_path = output / "raw_predictions.npz"
    phase_path = output / "phase_predictions_maxlag16.npz"
    np.savez_compressed(
        raw_path, targets=targets, conditions=conditions, subject_ids=subjects,
        source_files=sources, initial_flow_noise=noise,
        **{f"{model}_predictions": predictions[model] for model in MODELS},
    )
    np.savez_compressed(phase_path, **phase_arrays)
    per_window_path = output / "per_window_waveform_metrics.csv"
    subject_path = output / "per_subject_waveform_metrics.csv"
    _write_csv(per_window_path, per_window_rows)
    _write_csv(subject_path, _subject_rows(subjects, raw_rows, aligned_rows))
    summary_path = output / "waveform_summary.json"
    summary = {
        "schema_version": 1,
        "protocol": {
            "windows": 2877, "subjects": sorted(set(subjects)), "sampling_rate_hz": 128,
            "normalization": "window_minmax_neg1_1_v1", "raw_full_window_is_primary": True,
            "phase_sensitivity": "target-informed Pearson-maximizing +/-16-sample shift on common 480-sample support",
            "window_inference": "blocked_50_percent_overlap_and_subject_clustering",
            "subject_inference": "descriptive_only_extremely_underpowered_n3",
        },
        "generation": generation,
        "raw_full_window": raw_summary,
        "phase_sensitivity": phase_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    outputs = (raw_path, phase_path, per_window_path, subject_path, summary_path)
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "inputs": {
            "frozen_predictions": {"path": str(args.frozen_two_model_predictions.resolve()), "sha256": _sha256(args.frozen_two_model_predictions)},
            "checkpoints": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        },
        "array_sha256": {"targets": _array_sha256(targets), "conditions": _array_sha256(conditions), "initial_flow_noise": _array_sha256(noise)},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "script_sha256": _sha256(Path(__file__))},
        "claim_boundary": "Raw synchronized results are primary. Oracle phase correction is target-informed and diagnostic only. Overlapping windows and only three held-out subjects prohibit window-level or powered subject-level inference.",
        "outputs": [path.name for path in outputs],
        "artifact_sha256": {path.name: _sha256(path) for path in outputs},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--cfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_ot_checkpoint", type=Path, required=True)
    parser.add_argument("--rddm_checkpoint", type=Path, required=True)
    parser.add_argument("--frozen_two_model_predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    result = run(build_argparser().parse_args())
    print(f"mmECG four-model evaluation saved to {result}")
