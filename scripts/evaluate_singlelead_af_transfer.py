#!/usr/bin/env python3
"""Evaluate the frozen PTB-XL single-lead AF classifier without target fitting."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.evaluation.diagnostic_transfer import binary_metrics, sha256, stratified_bootstrap_auc


def aggregate_records(logits, labels, groups):
    group_values = np.asarray(groups).astype(str)
    names = np.unique(group_values)
    output_logits, output_labels = [], []
    for name in names:
        selected = group_values == name
        values = np.unique(labels[selected])
        if len(values) != 1:
            raise ValueError(f"conflicting labels within {name}")
        output_logits.append(float(np.mean(logits[selected])))
        output_labels.append(int(values[0]))
    return names, np.asarray(output_logits), np.asarray(output_labels, dtype=np.uint8)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.model_code_root.resolve()))
    from LibMTL.model.singlelead_af import load_singlelead_af_checkpoint

    device = torch.device(args.device)
    model, checkpoint = load_singlelead_af_checkpoint(args.checkpoint.resolve(), map_location=device)
    model.to(device).eval()
    waveforms = np.load(args.input_dir / "ecg_ptb_normalized_100hz.npy")
    labels = np.load(args.input_dir / "afib_labels.npy")
    groups = np.load(args.input_dir / "source_record_names.npy").astype(str)
    if waveforms.shape != (len(labels), 1, 400) or len(groups) != len(labels):
        raise ValueError("single-lead waveforms, labels, and record names are not aligned")
    logits = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), args.batch_size):
            batch = torch.from_numpy(waveforms[start : start + args.batch_size]).to(device)
            logits.append(model(batch).float().cpu().numpy())
    logits = np.concatenate(logits).astype(np.float64)
    names, record_logits, record_labels = aggregate_records(logits, labels, groups)
    fixed = binary_metrics(record_labels, record_logits, threshold=0.5)
    source_threshold = binary_metrics(record_labels, record_logits, threshold=args.source_threshold)
    bootstrap = stratified_bootstrap_auc(
        record_labels, record_logits, seed=args.bootstrap_seed, replicates=args.bootstrap_replicates
    )
    gate = {
        "minimum_record_auroc": args.minimum_auroc,
        "require_bootstrap_auroc_ci_lower_above": args.minimum_ci_lower,
        "point_criterion_pass": bool(fixed["auroc"] >= args.minimum_auroc),
        "ci_criterion_pass": bool(bootstrap["auroc_95_ci"][0] > args.minimum_ci_lower),
    }
    gate["passed"] = bool(gate["point_criterion_pass"] and gate["ci_criterion_pass"])

    with (output / "per_record_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_record_name", "label", "mean_logit", "probability", "windows"])
        writer.writeheader()
        for name, label, logit in zip(names, record_labels, record_logits):
            writer.writerow({
                "source_record_name": name, "label": int(label), "mean_logit": float(logit),
                "probability": float(torch.sigmoid(torch.tensor(logit)).item()),
                "windows": int(np.sum(groups == name)),
            })
    np.savez_compressed(
        output / "scores.npz", window_logits=logits.astype(np.float32), window_labels=labels,
        record_names=names, record_logits=record_logits.astype(np.float32), record_labels=record_labels,
    )
    summary = {
        "schema_version": 1, "status": "completed_zero_shot_singlelead_transfer",
        "dataset": "MIMIC PERform AF Lead II", "target_domain_training_or_calibration": False,
        "analysis_unit": "mean logit within each source record",
        "record_metrics_fixed_0.5": fixed,
        "record_metrics_source_validation_threshold": source_threshold,
        "source_validation_threshold": args.source_threshold,
        "bootstrap": bootstrap, "prespecified_gradcam_gate": gate,
        "checkpoint_metadata": {
            "epoch": checkpoint["epoch"], "class_names": checkpoint["class_names"],
            "input_spec": checkpoint["input_spec"], "model_config": checkpoint["model_config"],
        },
        "inputs_sha256": {
            "checkpoint": sha256(args.checkpoint.resolve()),
            "preprocessing_manifest": sha256(args.input_dir.resolve() / "manifest.json"),
            "waveforms": sha256(args.input_dir.resolve() / "ecg_ptb_normalized_100hz.npy"),
            "labels": sha256(args.input_dir.resolve() / "afib_labels.npy"),
            "groups": sha256(args.input_dir.resolve() / "source_record_names.npy"),
        },
        "execution": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv), "device": str(device)},
        "claim_boundary": "Threshold-free discrimination is primary; source thresholds are descriptive and were not calibrated on MIMIC.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": fixed, "bootstrap": bootstrap, "gate": gate}, indent=2))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_code_root", type=Path, required=True)
    parser.add_argument("--source_threshold", type=float, required=True)
    parser.add_argument("--minimum_auroc", type=float, default=0.70)
    parser.add_argument("--minimum_ci_lower", type=float, default=0.50)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
