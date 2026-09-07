#!/usr/bin/env python3
"""Evaluate a frozen diagnostic classifier without target-domain fitting."""

from __future__ import annotations

import argparse
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

from src.rcfm.evaluation.diagnostic_models import (
    load_torchscript_adapter,
    load_xresnet_adapter,
    predict_record,
)
from src.rcfm.evaluation.diagnostic_transfer import (
    aggregate_by_group,
    as_time_leads,
    binary_metrics,
    load_array_spec,
    sha256,
    stratified_bootstrap_auc,
)


def _path_from_spec(specification: str) -> Path:
    return Path(specification.rsplit(":", 1)[0] if ".npz:" in specification else specification).resolve()


def _concatenate(specifications: list[str]) -> np.ndarray:
    arrays = [load_array_spec(value) for value in specifications]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)


def _load_labels(specifications: list[str], column: int | None) -> np.ndarray:
    labels = _concatenate(specifications)
    if labels.ndim == 2:
        if column is None or column < 0 or column >= labels.shape[1]:
            raise ValueError("two-dimensional labels require a valid --label_column")
        labels = labels[:, column]
    elif labels.ndim != 1 or column is not None:
        raise ValueError("labels must be one-dimensional, or two-dimensional with --label_column")
    unique = set(np.unique(labels).tolist())
    if not unique <= {0, 1, False, True}:
        raise ValueError("selected diagnostic labels are not binary")
    return labels.astype(bool)


def _load_adapter(args: argparse.Namespace, device: torch.device):
    if args.backend == "xresnet":
        required = (args.benchmark_code_root, args.mlb, args.scaler)
        if any(value is None for value in required):
            raise ValueError("xresnet backend requires --benchmark_code_root, --mlb, and --scaler")
        return load_xresnet_adapter(
            checkpoint=args.checkpoint.resolve(), benchmark_code_root=args.benchmark_code_root.resolve(),
            mlb_path=args.mlb.resolve(), scaler_path=args.scaler.resolve(), device=device,
        )
    if args.class_names is None:
        raise ValueError("torchscript backend requires --class_names")
    return load_torchscript_adapter(
        checkpoint=args.checkpoint.resolve(), class_names_path=args.class_names.resolve(), device=device,
        input_rate_hz=args.model_rate_hz, crop_samples=args.crop_samples,
        crop_stride=args.crop_stride, normalization=args.normalization,
    )


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    adapter = _load_adapter(args, device)
    target_index = adapter.class_index(args.target_class)
    raw_waveforms = _concatenate(args.waveforms)
    waveforms = as_time_leads(raw_waveforms)
    labels = _load_labels(args.labels, args.label_column)
    groups = _concatenate(args.groups).astype(str) if args.groups else None
    if len(waveforms) != len(labels) or (groups is not None and len(groups) != len(labels)):
        raise ValueError("waveforms, labels, and groups are not row-aligned")
    records = len(waveforms) if args.max_records is None else min(args.max_records, len(waveforms))
    if records < 2:
        raise ValueError("at least two records are required")
    waveforms, labels = waveforms[:records], labels[:records]
    groups = groups[:records] if groups is not None else None

    logits = np.empty(records, dtype=np.float64)
    for index, waveform in enumerate(waveforms):
        logits[index] = predict_record(
            adapter, waveform, source_rate_hz=args.source_rate_hz,
            aggregation=args.crop_aggregation,
        )[target_index]
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"classified {index + 1}/{records}", flush=True)

    analysis_logits, analysis_labels = logits, labels
    unit = "record"
    pseudo_groups = np.arange(records, dtype=np.int64)
    if groups is not None:
        analysis_logits, analysis_labels, pseudo_groups = aggregate_by_group(logits, labels, groups)
        unit = "group_mean_logit"
    metrics = binary_metrics(analysis_labels, analysis_logits, threshold=args.probability_threshold)
    bootstrap = stratified_bootstrap_auc(
        analysis_labels, analysis_logits, seed=args.bootstrap_seed,
        replicates=args.bootstrap_replicates,
    )
    np.savez_compressed(
        output / "diagnostic_scores.npz",
        row_logits=logits.astype(np.float32), row_labels=labels.astype(np.uint8),
        analysis_logits=analysis_logits.astype(np.float32),
        analysis_labels=analysis_labels.astype(np.uint8), pseudo_group_indices=pseudo_groups,
    )
    inputs = {
        "waveforms": [sha256(_path_from_spec(value)) for value in args.waveforms],
        "labels": [sha256(_path_from_spec(value)) for value in args.labels],
        "checkpoint": sha256(args.checkpoint.resolve()),
    }
    for name in ("mlb", "scaler", "class_names"):
        value = getattr(args, name)
        if value is not None:
            inputs[name] = sha256(value.resolve())
    if args.groups:
        inputs["groups"] = [sha256(_path_from_spec(value)) for value in args.groups]
    summary = {
        "schema_version": 1,
        "status": "completed_zero_shot_diagnostic_transfer",
        "dataset": args.dataset,
        "target_class": args.target_class,
        "classifier_backend": args.backend,
        "target_domain_training_or_calibration": False,
        "source_rate_hz": args.source_rate_hz,
        "input_adapter": "single lead repeated to 12" if raw_waveforms.ndim == 2 else "12 lead",
        "crop_aggregation": args.crop_aggregation,
        "analysis_unit": unit,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "claim_boundary": (
            "This experiment measures frozen-classifier cross-domain discrimination. It does not by itself "
            "establish temporal Grad-CAM localization accuracy or clinical validity."
        ),
        "inputs_sha256": inputs,
        "execution": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "device": str(device),
            "torch_version": torch.__version__,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "metrics": metrics}, sort_keys=True))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--waveforms", action="append", required=True, help="Repeat for concatenated .npy/.npz:key inputs.")
    parser.add_argument("--labels", action="append", required=True, help="Repeat in the same order as --waveforms.")
    parser.add_argument("--label_column", type=int)
    parser.add_argument("--groups", action="append", help="Repeat group/subject arrays in waveform order.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=["xresnet", "torchscript"], default="xresnet")
    parser.add_argument("--benchmark_code_root", type=Path)
    parser.add_argument("--mlb", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--class_names", type=Path)
    parser.add_argument("--target_class", default="AFIB")
    parser.add_argument("--source_rate_hz", type=int, default=128)
    parser.add_argument("--model_rate_hz", type=int, default=100)
    parser.add_argument("--crop_samples", type=int, default=250)
    parser.add_argument("--crop_stride", type=int, default=125)
    parser.add_argument("--normalization", choices=["ptb_scalar", "record_zscore", "none"], default="record_zscore")
    parser.add_argument("--crop_aggregation", choices=["mean_logit", "max_logit"], default="mean_logit")
    parser.add_argument("--probability_threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
