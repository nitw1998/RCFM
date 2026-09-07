#!/usr/bin/env python3
"""Evaluate generated ECGs with a classifier independent of the mask generator."""

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
    binary_metrics,
    load_array_spec,
    paired_stratified_bootstrap_auc_difference,
    reconstruct_twelve_leads,
    sha256,
    stratified_bootstrap_auc,
)


def _prediction_specs(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--prediction must use NAME=PATH or NAME=PATH.npz:KEY")
        name, specification = value.split("=", 1)
        if not name or name in output:
            raise ValueError("prediction names must be nonempty and unique")
        output[name] = specification
    if not output:
        raise ValueError("at least one generated prediction is required")
    return output


def _path_from_spec(specification: str) -> Path:
    return Path(
        specification.rsplit(":", 1)[0] if ".npz:" in specification else specification
    ).resolve()


def _load_labels(specification: str, column: int | None) -> np.ndarray:
    labels = load_array_spec(specification)
    if labels.ndim == 2:
        if column is None or not 0 <= column < labels.shape[1]:
            raise ValueError("two-dimensional labels require --label_column")
        labels = labels[:, column]
    if labels.ndim != 1 or not set(np.unique(labels).tolist()) <= {0, 1, False, True}:
        raise ValueError("selected downstream labels must be binary")
    return labels.astype(bool)


def _load_adapter(args: argparse.Namespace, device: torch.device):
    if args.backend == "xresnet":
        if any(value is None for value in (args.benchmark_code_root, args.mlb, args.scaler)):
            raise ValueError("xresnet evaluator requires benchmark root, mlb, and scaler")
        return load_xresnet_adapter(
            checkpoint=args.evaluator_checkpoint.resolve(), benchmark_code_root=args.benchmark_code_root.resolve(),
            mlb_path=args.mlb.resolve(), scaler_path=args.scaler.resolve(), device=device,
        )
    if args.class_names is None:
        raise ValueError("torchscript evaluator requires --class_names")
    return load_torchscript_adapter(
        checkpoint=args.evaluator_checkpoint.resolve(), class_names_path=args.class_names.resolve(),
        device=device, input_rate_hz=args.model_rate_hz, crop_samples=args.crop_samples,
        crop_stride=args.crop_stride, normalization=args.normalization,
    )


def _inverse_record_minmax(values: np.ndarray, minima: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low = np.asarray(minima, dtype=np.float32)
    span = np.asarray(ranges, dtype=np.float32)
    if array.ndim != 3 or low.shape != (len(array), 12) or span.shape != low.shape or np.any(span <= 0):
        raise ValueError("inverse normalization requires arrays (N,12,T), minima (N,12), ranges (N,12)")
    return ((array + 1.0) * 0.5 * span[:, :, None] + low[:, :, None]).astype(np.float32)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    mask_hash = sha256(args.mask_generator_checkpoint.resolve())
    evaluator_hash = sha256(args.evaluator_checkpoint.resolve())
    if mask_hash == evaluator_hash:
        raise ValueError("independent-classifier audit forbids reusing the mask-generator checkpoint")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    adapter = _load_adapter(args, device)
    target_index = adapter.class_index(args.target_class)

    condition = load_array_spec(args.condition)
    labels = _load_labels(args.labels, args.label_column)
    prediction_specs = _prediction_specs(args.prediction)
    targets = {"real": load_array_spec(args.real_targets)}
    targets.update({name: load_array_spec(spec) for name, spec in prediction_specs.items()})
    groups = load_array_spec(args.groups).astype(str) if args.groups else None
    if len(condition) != len(labels) or (groups is not None and len(groups) != len(labels)):
        raise ValueError("condition, labels, and groups are not aligned")
    twelve_lead = {
        name: reconstruct_twelve_leads(condition, values, condition_lead=args.condition_lead)
        for name, values in targets.items()
    }
    if args.input_domain == "record_minmax_neg1_1":
        if args.record_minima is None or args.record_ranges is None:
            raise ValueError("record-minmax inputs require --record_minima and --record_ranges")
        minima, ranges = load_array_spec(args.record_minima), load_array_spec(args.record_ranges)
        twelve_lead = {
            name: _inverse_record_minmax(values, minima, ranges) for name, values in twelve_lead.items()
        }
    elif args.input_domain != "classifier_ready":
        raise ValueError("unsupported classifier input domain")

    count = len(labels) if args.max_records is None else min(args.max_records, len(labels))
    labels = labels[:count]
    groups = groups[:count] if groups is not None else None
    logits: dict[str, np.ndarray] = {}
    analysis_logits: dict[str, np.ndarray] = {}
    common_analysis_labels: np.ndarray | None = None
    summaries: dict[str, object] = {}
    for model_name, waveforms in twelve_lead.items():
        values = np.empty(count, dtype=np.float64)
        for index, waveform in enumerate(waveforms[:count]):
            values[index] = predict_record(
                adapter, waveform.T, source_rate_hz=args.source_rate_hz,
                aggregation=args.crop_aggregation,
            )[target_index]
        logits[model_name] = values
        analysis_values, analysis_labels = values, labels
        if groups is not None:
            analysis_values, analysis_labels, _ = aggregate_by_group(values, labels, groups)
        analysis_logits[model_name] = analysis_values
        common_analysis_labels = analysis_labels
        summaries[model_name] = {
            "metrics": binary_metrics(analysis_labels, analysis_values, args.probability_threshold),
            "bootstrap": stratified_bootstrap_auc(
                analysis_labels, analysis_values, seed=args.bootstrap_seed,
                replicates=args.bootstrap_replicates,
            ),
        }
        print(f"evaluated {model_name}: {count} records", flush=True)

    if args.reference_model not in analysis_logits:
        raise ValueError(f"reference model {args.reference_model!r} is absent")
    assert common_analysis_labels is not None
    comparisons = {}
    for index, (model_name, values) in enumerate(analysis_logits.items()):
        if model_name == args.reference_model:
            continue
        comparisons[f"{model_name}_minus_{args.reference_model}"] = (
            paired_stratified_bootstrap_auc_difference(
                common_analysis_labels, values, analysis_logits[args.reference_model],
                seed=args.bootstrap_seed + 100 + index, replicates=args.bootstrap_replicates,
            )
        )

    np.savez_compressed(
        output / "independent_classifier_logits.npz",
        labels=labels.astype(np.uint8),
        analysis_labels=common_analysis_labels.astype(np.uint8),
        **{f"{name}_logits": values.astype(np.float32) for name, values in logits.items()},
        **{
            f"{name}_analysis_logits": values.astype(np.float32)
            for name, values in analysis_logits.items()
        },
    )
    input_hashes = {
        "condition": sha256(_path_from_spec(args.condition)),
        "real_targets": sha256(_path_from_spec(args.real_targets)),
        "labels": sha256(_path_from_spec(args.labels)),
        "predictions": {
            name: sha256(_path_from_spec(specification))
            for name, specification in prediction_specs.items()
        },
    }
    for name in ("groups", "record_minima", "record_ranges", "class_names", "mlb", "scaler"):
        value = getattr(args, name)
        if value is not None:
            input_hashes[name] = sha256(_path_from_spec(str(value)))
    summary = {
        "schema_version": 1,
        "status": "completed_independent_downstream_classifier_audit",
        "dataset": args.dataset,
        "target_class": args.target_class,
        "analysis_unit": "group mean logit" if groups is not None else "record",
        "mask_generator_checkpoint_sha256": mask_hash,
        "downstream_evaluator": {"backend": args.backend, "checkpoint_sha256": evaluator_hash},
        "same_checkpoint_as_mask_generator": False,
        "input_domain": args.input_domain,
        "inputs_sha256": input_hashes,
        "models": summaries,
        "paired_comparisons": comparisons,
        "claim_boundary": (
            "This evaluates diagnostic information with a checkpoint independent of the mask generator. "
            "A different architecture (TorchScript backend) is stronger mismatch evidence than a second XResNet checkpoint."
        ),
        "execution": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv)},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "models": sorted(summaries)}))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", required=True, help="Lead-II condition, .npy or .npz:key")
    parser.add_argument("--real_targets", required=True, help="Eleven real target leads")
    parser.add_argument("--prediction", action="append", default=[], help="NAME=array specification")
    parser.add_argument("--reference_model", default="cfm", help="Baseline prediction name for paired differences.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label_column", type=int)
    parser.add_argument("--groups")
    parser.add_argument("--condition_lead", type=int, default=1)
    parser.add_argument("--input_domain", choices=["record_minmax_neg1_1", "classifier_ready"], required=True)
    parser.add_argument("--record_minima")
    parser.add_argument("--record_ranges")
    parser.add_argument("--mask_generator_checkpoint", type=Path, required=True)
    parser.add_argument("--evaluator_checkpoint", type=Path, required=True)
    parser.add_argument("--backend", choices=["xresnet", "torchscript"], default="torchscript")
    parser.add_argument("--benchmark_code_root", type=Path)
    parser.add_argument("--mlb", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--class_names", type=Path)
    parser.add_argument("--target_class", default="AFIB")
    parser.add_argument("--source_rate_hz", type=int, default=128)
    parser.add_argument("--model_rate_hz", type=int, default=100)
    parser.add_argument("--crop_samples", type=int, default=250)
    parser.add_argument("--crop_stride", type=int, default=125)
    parser.add_argument("--normalization", choices=["record_zscore", "none"], default="record_zscore")
    parser.add_argument("--crop_aggregation", choices=["mean_logit", "max_logit"], default="mean_logit")
    parser.add_argument("--probability_threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
