#!/usr/bin/env python3
"""Deterministically evaluate the lightweight direct-CNN baseline."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data import get_ppg2ecg_datasets
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.evaluate_ptbxl_fourway import TARGET_INDICES, TARGET_LEADS, _metric_summary
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic
from src.rcfm.baselines import DirectRegressionCNN
from src.rcfm.metrics.paper_statistics import raw_waveform_summary
from train_rcfm import build_datasets


SINGLE_OUTPUT_CONTRACTS = {
    "WESAD": {"records": 4213, "task": "ppg2ecg"},
    "mmECG": {"records": 2877, "task": "rcg2ecg"},
}


def _json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_checkpoint(path: Path, dataset: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("kind") != "direct_cnn_regression":
        raise ValueError("checkpoint must be a structured direct-CNN regression checkpoint")
    if int(payload.get("epoch", -1)) != 500:
        raise ValueError("primary direct-CNN evaluation requires epoch 500")
    config = payload.get("config", {})
    if config.get("datasets") != [dataset] or config.get("model_family") != "DirectCNN":
        raise ValueError("checkpoint dataset/model contract mismatch")
    if config.get("stochastic") is not False or config.get("sampling_steps") != 0:
        raise ValueError("direct-CNN checkpoint must declare deterministic one-pass inference")
    return dict(payload)


@torch.inference_mode()
def _predict(payload: Mapping[str, object], conditions: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    config = payload["config"]
    model = DirectRegressionCNN(
        int(config["input_channels"]), int(config["output_channels"]), int(config["width"]),
        tuple(int(item) for item in config["dilations"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    loader = DataLoader(torch.from_numpy(conditions), batch_size=batch_size, shuffle=False, num_workers=0)
    return np.concatenate([model(batch.float().to(device)).cpu().numpy() for batch in loader]).astype(np.float32)


def _mimic(args: argparse.Namespace, payload: Mapping[str, object], output: Path) -> dict[str, object]:
    config, normalization = payload["config"], payload["normalization"]
    _, dataset = get_ppg2ecg_datasets(
        DATA_PATH=str(args.data_root.resolve()), datasets=["MIMIC-AFib"], window_size=4,
        normalization_metadata=normalization, normalization_id=config["normalization_id"], load_train=False,
    )
    if len(dataset) != 1800:
        raise ValueError("MIMIC direct-CNN evaluation requires all 1,800 QC test windows")
    targets = np.asarray(dataset.target_ecg[:, None, :], dtype=np.float32)
    conditions = np.asarray(dataset.condition_signal[:, None, :], dtype=np.float32)
    identity = args.identity_root.resolve()
    subjects = np.load(identity / "subject_ids_test.npy", allow_pickle=False).astype(str)
    records = np.load(identity / "record_ids_test.npy", allow_pickle=False).astype(str)
    afib = np.load(identity / "afib_labels_test.npy", allow_pickle=False).astype(bool)
    original_rows = np.load(args.data_root.resolve() / "MIMIC-AFib/kept_indices_test.npy", allow_pickle=False)
    if any(len(values) != 1800 for values in (subjects, records, afib, original_rows)):
        raise ValueError("MIMIC identity arrays must align with all retained test windows")
    predictions = _predict(payload, conditions, args.batch_size, torch.device(args.device))
    legacy, _ = _waveform_metrics(targets, predictions)
    paper, detail = raw_waveform_summary(targets, predictions, subjects, lead_names=("ECG",))
    np.savez_compressed(
        output / "direct_cnn_predictions.npz", targets=targets, conditions=conditions,
        direct_cnn_predictions=predictions, subject_ids=subjects, record_ids=records,
        afib_labels=afib, source_test_rows_before_zero_filter=original_rows,
    )
    max_lag = 16
    lag_summary, lag_values = _lag_diagnostic(targets, predictions, max_lag, 128)
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    centered_target, unshifted, aligned = _fixed_support_align(targets, predictions, shifts, max_lag)
    np.savez_compressed(
        output / "phase_predictions_maxlag16.npz", targets=centered_target,
        direct_cnn_unshifted_predictions=unshifted,
        direct_cnn_oracle_aligned_predictions=aligned,
        direct_cnn_oracle_shifts=shifts,
    )
    unshifted_summary, _ = _waveform_metrics(centered_target, unshifted)
    aligned_summary, _ = _waveform_metrics(centered_target, aligned)
    summary = {
        "model": {"direct_cnn": {"legacy_waveform": legacy, "paper_waveform": paper}},
        "phase_sensitivity_maxlag16": {
            "status": "target_informed_oracle_diagnostic_only", "lag": lag_summary,
            "unshifted_fixed_480": unshifted_summary, "oracle_aligned_fixed_480": aligned_summary,
        },
        "identity": {"subjects": int(len(np.unique(subjects))), "records": int(len(np.unique(records))), "afib_windows": int(afib.sum())},
    }
    _json(output / "waveform_summary.json", summary)
    return {"records": 1800, "split": "frozen_qc_test", "summary": summary}


def _ptbxl(args: argparse.Namespace, payload: Mapping[str, object], output: Path) -> dict[str, object]:
    config = payload["config"]
    _, dataset = build_datasets(
        "ecg2ecg", ["PTBXL"], str(args.data_root.resolve()), 4,
        normalization_metadata=payload["normalization"], normalization_id=config["normalization_id"],
        condition_lead_index=1, target_lead_indices=list(TARGET_INDICES), load_train=False, heldout_split="test",
    )
    if len(dataset) != 2203:
        raise ValueError("PTB-XL direct-CNN evaluation requires all 2,203 fold10 records")
    targets = np.asarray(dataset.target_ecg, dtype=np.float32)
    conditions = np.asarray(dataset.condition_signal[:, None, :], dtype=np.float32)
    records = np.asarray(dataset.record_ids)
    patients = np.load(args.data_root.resolve() / "PTBXL/patient_ids_test.npy", allow_pickle=False)
    predictions = _predict(payload, conditions, args.batch_size, torch.device(args.device))
    legacy, per_record = _metric_summary(targets, predictions)
    paper, detail = raw_waveform_summary(targets, predictions, patients, lead_names=TARGET_LEADS)
    np.savez_compressed(
        output / "paired_reference.npz", targets=targets, conditions=conditions,
        record_ids=records, patient_ids=patients,
    )
    np.save(output / "direct_cnn_predictions.npy", predictions, allow_pickle=False)
    summary = {"models": {"direct_cnn": {**legacy, "paper_waveform": paper}}}
    _json(output / "waveform_summary.json", summary)
    return {"records": 2203, "split": "official_fold_10", "summary": summary}


def _cpsc(args: argparse.Namespace, payload: Mapping[str, object], output: Path) -> dict[str, object]:
    config = payload["config"]
    _, dataset = build_datasets(
        "ecg2ecg", ["CPSC2018"], str(args.data_root.resolve()), 4,
        normalization_metadata=payload["normalization"], normalization_id=config["normalization_id"],
        condition_lead_index=1, target_lead_indices=list(TARGET_INDICES), load_train=False,
        heldout_split="val",
    )
    if len(dataset) != 686:
        raise ValueError("CPSC2018 direct-CNN evaluation requires all 686 validation records")
    targets = np.asarray(dataset.target_ecg, dtype=np.float32)
    conditions = np.asarray(dataset.condition_signal[:, None, :], dtype=np.float32)
    records = np.asarray(dataset.record_ids)
    predictions = _predict(payload, conditions, args.batch_size, torch.device(args.device))
    legacy, _ = _metric_summary(targets, predictions)
    paper, _ = raw_waveform_summary(targets, predictions, records, lead_names=TARGET_LEADS)
    np.savez_compressed(output / "paired_reference.npz", targets=targets, conditions=conditions,
                        record_ids=records)
    np.save(output / "direct_cnn_predictions.npy", predictions, allow_pickle=False)
    summary = {"models": {"direct_cnn": {**legacy, "paper_waveform": paper}}}
    _json(output / "waveform_summary.json", summary)
    return {"records": 686, "split": "source_derived_validation", "summary": summary}


def _single_output(args: argparse.Namespace, payload: Mapping[str, object], output: Path) -> dict[str, object]:
    config = payload["config"]
    contract = SINGLE_OUTPUT_CONTRACTS[args.dataset]
    _, dataset = build_datasets(
        contract["task"], [args.dataset], str(args.data_root.resolve()), 4,
        normalization_metadata=payload["normalization"], normalization_id=config["normalization_id"],
        load_train=False, heldout_split="test",
    )
    if len(dataset) != contract["records"]:
        raise ValueError(f"{args.dataset} direct-CNN evaluation requires all {contract['records']} test windows")
    targets = np.asarray(dataset.target_ecg[:, None, :] if dataset.target_ecg.ndim == 2 else dataset.target_ecg,
                         dtype=np.float32)
    conditions = np.asarray(dataset.condition_signal[:, None, :] if dataset.condition_signal.ndim == 2
                            else dataset.condition_signal, dtype=np.float32)
    subjects = np.load(args.data_root.resolve() / args.dataset / "subject_ids_test.npy", allow_pickle=False).astype(str)
    if subjects.shape != (contract["records"],):
        raise ValueError(f"{args.dataset} subject IDs do not match held-out rows")
    predictions = _predict(payload, conditions, args.batch_size, torch.device(args.device))
    raw, _ = _waveform_metrics(targets, predictions)
    paper, _ = raw_waveform_summary(targets, predictions, subjects, lead_names=("ECG",))
    lag_summary, lag_values = _lag_diagnostic(targets, predictions, 16, 128)
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    center, unshifted, aligned = _fixed_support_align(targets, predictions, shifts, 16)
    before, _ = _waveform_metrics(center, unshifted)
    after, _ = _waveform_metrics(center, aligned)
    np.savez_compressed(output / "direct_cnn_predictions.npz", targets=targets, predictions=predictions,
                        conditions=conditions, subject_ids=subjects)
    np.savez_compressed(output / "phase_predictions_maxlag16.npz", targets=center,
                        direct_cnn_unshifted_predictions=unshifted,
                        direct_cnn_oracle_aligned_predictions=aligned,
                        direct_cnn_oracle_shifts=shifts, subject_ids=subjects)
    summary = {
        "dataset": args.dataset, "model": "DirectCNN", "raw_full_window": raw,
        "paper_waveform": paper, "unshifted_fixed_support": before,
        "oracle_aligned_fixed_support": after,
        "oracle_lag_diagnostic": {**lag_summary,
            "boundary_fraction": float(np.mean(np.abs(shifts) == 16))},
        "interpretation": "Raw full-window results are primary; oracle alignment uses test targets and is diagnostic only.",
    }
    _json(output / "waveform_summary.json", summary)
    return {"records": contract["records"], "split": "frozen_subject_test", "summary": summary}


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    payload = _load_checkpoint(checkpoint, args.dataset)
    if args.dataset == "MIMIC-AFib":
        result = _mimic(args, payload, output)
    elif args.dataset == "PTBXL":
        result = _ptbxl(args, payload, output)
    elif args.dataset == "CPSC2018":
        result = _cpsc(args, payload, output)
    else:
        result = _single_output(args, payload, output)
    artifact_names = [path.name for path in output.iterdir() if path.is_file()]
    protocol = {
        "schema_version": 1, "status": "completed", "command": shlex.join(sys.argv),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "dataset": args.dataset, "split": result["split"], "evaluated_records": result["records"],
            "sampling_rate_hz": 128, "window_seconds": 4, "phase_correction_applied": False,
            "model": "DirectCNN", "deterministic": True, "neural_forward_passes": 1,
            "selection": "predeclared_epoch_500_endpoint",
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint), "epoch": 500},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": args.device},
        "claim_boundary": "Normalized-domain waveform results; MIMIC oracle phase sensitivity is diagnostic only; PTB-XL physical inversion uses target scalers only in downstream descriptive clinical analysis.",
        "artifacts": {name: _sha256(output / name) for name in artifact_names},
    }
    _json(output / "protocol.json", protocol)
    print(json.dumps({"status": "completed", "output": str(output), "summary": result["summary"]}, allow_nan=False))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MIMIC-AFib", "PTBXL", "CPSC2018", "WESAD", "mmECG"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--identity_root", type=Path, default=REPO.parent / "runs/preprocessing/mimic_afib_identity_v1")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
