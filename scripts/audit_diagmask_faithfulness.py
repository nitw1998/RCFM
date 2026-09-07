#!/usr/bin/env python3
"""Audit transferred AFIB Grad-CAM faithfulness against matched controls."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.evaluation.diagnostic_models import (
    aggregate_crop_logits,
    load_xresnet_adapter,
    predict_crops,
    prepare_record,
    record_crops,
)
from src.rcfm.evaluation.diagnostic_transfer import (
    as_time_leads,
    load_array_spec,
    paired_bootstrap_mean,
    sha256,
)
from src.rcfm.interpretability.gradcam import gradcam_1d, normalize_soft_mask, stitch_temporal_cams


def _load_labels(specification: str, column: int | None) -> np.ndarray:
    labels = load_array_spec(specification)
    if labels.ndim == 2:
        if column is None or not 0 <= column < labels.shape[1]:
            raise ValueError("two-dimensional labels require --label_column")
        labels = labels[:, column]
    if labels.ndim != 1 or not set(np.unique(labels).tolist()) <= {0, 1, False, True}:
        raise ValueError("faithfulness labels must be binary")
    return labels.astype(bool)


def _cam_for_prepared(adapter, prepared: np.ndarray, target_index: int) -> tuple[np.ndarray, bool]:
    crops, starts = record_crops(prepared, adapter)
    crop_cams: list[np.ndarray] = []
    target_layer = adapter.model[7][-1]
    for crop in crops:
        tensor = torch.from_numpy(crop).unsqueeze(0).to(adapter.device, dtype=torch.float32)
        cam, _ = gradcam_1d(adapter.model, target_layer, tensor, target_index)
        crop_cams.append(cam)
    stitched = stitch_temporal_cams(crop_cams, starts, len(prepared))
    return normalize_soft_mask(stitched)


def _target_score(adapter, prepared: np.ndarray, target_index: int, aggregation: str) -> float:
    crops, _ = record_crops(prepared, adapter)
    logits = predict_crops(adapter, crops)
    return float(aggregate_crop_logits(logits, aggregation)[target_index])


def _selected(mask: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(round(len(mask) * fraction)))
    indices = np.argpartition(np.asarray(mask), len(mask) - count)[-count:]
    selected = np.zeros(len(mask), dtype=bool)
    selected[indices] = True
    return selected


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _perturbed_score(adapter, prepared, selected, target_index, aggregation, insertion=False) -> float:
    values = np.zeros_like(prepared) if insertion else prepared.copy()
    if insertion:
        values[selected] = prepared[selected]
    else:
        values[selected] = 0.0
    return _target_score(adapter, values, target_index, aggregation)


def _aggregate_rows(rows: list[dict[str, object]], groups: np.ndarray | None) -> list[dict[str, object]]:
    if groups is None:
        return rows
    numeric = [key for key in rows[0] if key not in {"row_index", "label", "cam_degenerate"}]
    output: list[dict[str, object]] = []
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    for group_index in range(len(unique)):
        selected = np.flatnonzero(inverse == group_index)
        labels = {bool(rows[index]["label"]) for index in selected}
        if len(labels) != 1:
            raise ValueError("group contains conflicting labels")
        output.append({
            "row_index": group_index,
            "label": int(labels.pop()),
            "cam_degenerate": float(np.mean([bool(rows[index]["cam_degenerate"]) for index in selected])),
            **{key: float(np.mean([float(rows[index][key]) for index in selected])) for key in numeric},
        })
    return output


def _summarize_analysis_rows(
    rows: list[dict[str, object]], *, bootstrap_seed: int, bootstrap_replicates: int
) -> dict[str, object]:
    estimand_keys = [
        key
        for key in rows[0]
        if key.startswith(("deletion_advantage_", "insertion_advantage_"))
    ]
    cohorts = {
        "AF_positive": [row for row in rows if row["label"] == 1],
        "non_AF": [row for row in rows if row["label"] == 0],
        "overall": rows,
    }
    output: dict[str, object] = {}
    seed_offset = 0
    for cohort_name, selected in cohorts.items():
        paired_estimands: dict[str, object] = {}
        for key in estimand_keys:
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            if len(values) >= 2:
                paired_estimands[key] = paired_bootstrap_mean(
                    values,
                    seed=bootstrap_seed + seed_offset,
                    replicates=bootstrap_replicates,
                )
                seed_offset += 1
            else:
                paired_estimands[key] = {
                    "observations": int(len(values)),
                    "status": "insufficient_for_bootstrap",
                }
        output[cohort_name] = {
            "observations": len(selected),
            "paired_estimands": paired_estimands,
            "descriptive": {
                "mean_occupancy": (
                    float(np.mean([row["cam_occupancy"] for row in selected]))
                    if selected else None
                ),
                "degenerate_fraction": (
                    float(np.mean([row["cam_degenerate"] for row in selected]))
                    if selected else None
                ),
                "mean_amplitude_spearman": (
                    float(np.mean([row["amplitude_cam_spearman"] for row in selected]))
                    if selected else None
                ),
                "mean_noise_spearman": (
                    float(np.mean([row["noise_cam_spearman"] for row in selected]))
                    if selected else None
                ),
            },
        }
    return output


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (
        not args.fractions
        or any(not 0 < value < 1 for value in args.fractions)
        or args.random_replicates < 1
        or args.bootstrap_replicates < 1
    ):
        raise ValueError("fractions, random replicates, and bootstrap replicates are invalid")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    adapter = load_xresnet_adapter(
        checkpoint=args.checkpoint.resolve(), benchmark_code_root=args.benchmark_code_root.resolve(),
        mlb_path=args.mlb.resolve(), scaler_path=args.scaler.resolve(), device=device,
    )
    target_index = adapter.class_index(args.target_class)
    waveforms = as_time_leads(load_array_spec(args.waveforms))
    labels = _load_labels(args.labels, args.label_column)
    groups = load_array_spec(args.groups).astype(str) if args.groups else None
    if len(waveforms) != len(labels) or (groups is not None and len(groups) != len(labels)):
        raise ValueError("waveforms, labels, and groups are not aligned")
    count = len(waveforms) if args.max_records is None else min(args.max_records, len(waveforms))
    waveforms, labels = waveforms[:count], labels[:count]
    groups = groups[:count] if groups is not None else None
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for row_index, (waveform, label) in enumerate(zip(waveforms, labels)):
        prepared = prepare_record(waveform, source_rate_hz=args.source_rate_hz, adapter=adapter)
        mask, degenerate = _cam_for_prepared(adapter, prepared, target_index)
        baseline = _target_score(adapter, prepared, target_index, args.crop_aggregation)
        transformed = prepared * args.amplitude_scale
        amplitude_cam, _ = _cam_for_prepared(adapter, transformed, target_index)
        noise_scale = args.noise_std_fraction * max(float(np.std(prepared)), 1e-8)
        noise_cam, _ = _cam_for_prepared(
            adapter, prepared + rng.normal(0.0, noise_scale, prepared.shape).astype(np.float32), target_index
        )
        row: dict[str, object] = {
            "row_index": row_index, "label": int(label), "cam_degenerate": bool(degenerate),
            "cam_occupancy": float(np.mean(mask)),
            "amplitude_cam_spearman": _finite_spearman(mask, amplitude_cam),
            "noise_cam_spearman": _finite_spearman(mask, noise_cam),
        }
        for fraction in args.fractions:
            suffix = f"top{int(round(100 * fraction))}"
            selected = _selected(mask, fraction)
            shifted = np.roll(selected, len(selected) // 2)
            deletion = baseline - _perturbed_score(
                adapter, prepared, selected, target_index, args.crop_aggregation
            )
            shifted_drop = baseline - _perturbed_score(
                adapter, prepared, shifted, target_index, args.crop_aggregation
            )
            null = _target_score(adapter, np.zeros_like(prepared), target_index, args.crop_aggregation)
            insertion = _perturbed_score(
                adapter, prepared, selected, target_index, args.crop_aggregation, insertion=True
            ) - null
            random_deletion, random_insertion = [], []
            selected_count = int(np.sum(selected))
            for _ in range(args.random_replicates):
                random_selected = np.zeros(len(prepared), dtype=bool)
                random_selected[rng.choice(len(prepared), selected_count, replace=False)] = True
                random_deletion.append(baseline - _perturbed_score(
                    adapter, prepared, random_selected, target_index, args.crop_aggregation
                ))
                random_insertion.append(_perturbed_score(
                    adapter, prepared, random_selected, target_index, args.crop_aggregation, insertion=True
                ) - null)
            row.update({
                f"deletion_drop_{suffix}": deletion,
                f"random_deletion_drop_{suffix}": float(np.mean(random_deletion)),
                f"deletion_advantage_{suffix}": deletion - float(np.mean(random_deletion)),
                f"shifted_deletion_drop_{suffix}": shifted_drop,
                f"insertion_gain_{suffix}": insertion,
                f"random_insertion_gain_{suffix}": float(np.mean(random_insertion)),
                f"insertion_advantage_{suffix}": insertion - float(np.mean(random_insertion)),
            })
        rows.append(row)
        if args.progress_every and (row_index + 1) % args.progress_every == 0:
            print(f"audited {row_index + 1}/{count}", flush=True)

    analysis_rows = _aggregate_rows(rows, groups)
    fields = list(rows[0])
    with (output / "per_record_faithfulness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    cohort_summaries = _summarize_analysis_rows(
        analysis_rows,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    summary = {
        "schema_version": 2,
        "status": "completed_gradcam_faithfulness_audit",
        "dataset": args.dataset,
        "target": f"{args.target_class} pre-sigmoid logit",
        "analysis_unit": "group mean" if groups is not None else "record",
        "perturbation": "zero replacement in frozen-classifier standardized input domain",
        "primary_cohort": "AF_positive",
        "cohort_summaries": cohort_summaries,
        "claim_boundary": (
            "Deletion/insertion tests classifier faithfulness, not anatomical localization or clinical validity. "
            "AF records are not required to overlap a canonical P-wave interval."
        ),
        "inputs_sha256": {
            "checkpoint": sha256(args.checkpoint.resolve()),
            "mlb": sha256(args.mlb.resolve()),
            "scaler": sha256(args.scaler.resolve()),
            "waveforms": sha256(Path(args.waveforms.rsplit(":", 1)[0] if ".npz:" in args.waveforms else args.waveforms).resolve()),
            "labels": sha256(Path(args.labels.rsplit(":", 1)[0] if ".npz:" in args.labels else args.labels).resolve()),
            **(
                {"groups": sha256(Path(args.groups.rsplit(":", 1)[0] if ".npz:" in args.groups else args.groups).resolve())}
                if args.groups else {}
            ),
        },
        "execution": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv)},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "analysis_observations": len(analysis_rows)}))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--waveforms", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--label_column", type=int)
    parser.add_argument("--groups")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark_code_root", type=Path, required=True)
    parser.add_argument("--mlb", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--target_class", default="AFIB")
    parser.add_argument("--source_rate_hz", type=int, default=128)
    parser.add_argument("--crop_aggregation", choices=["mean_logit", "max_logit"], default="mean_logit")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--random_replicates", type=int, default=10)
    parser.add_argument("--amplitude_scale", type=float, default=1.05)
    parser.add_argument("--noise_std_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
