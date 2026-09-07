#!/usr/bin/env python3
"""Audit Grad-CAM faithfulness for the frozen single-lead AF classifier."""

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
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.evaluation.diagnostic_transfer import paired_bootstrap_mean, sha256


def aggregate_rows(rows, groups):
    numeric = [key for key in rows[0] if key not in {"row_index", "label", "cam_degenerate"}]
    output = []
    group_values = np.asarray(groups).astype(str)
    for group_index, name in enumerate(np.unique(group_values)):
        selected = np.flatnonzero(group_values == name)
        labels = {bool(rows[index]["label"]) for index in selected}
        if len(labels) != 1:
            raise ValueError("group contains conflicting labels")
        output.append({
            "row_index": group_index, "label": int(labels.pop()),
            "cam_degenerate": float(np.mean([bool(rows[index]["cam_degenerate"]) for index in selected])),
            **{key: float(np.mean([float(rows[index][key]) for index in selected])) for key in numeric},
        })
    return output


def summarize_rows(rows, bootstrap_seed, bootstrap_replicates):
    estimands = [key for key in rows[0] if key.startswith(("deletion_advantage_", "insertion_advantage_"))]
    cohorts = {
        "AF_positive": [row for row in rows if row["label"] == 1],
        "non_AF": [row for row in rows if row["label"] == 0],
        "overall": rows,
    }
    output, offset = {}, 0
    for cohort_name, selected in cohorts.items():
        paired = {}
        for key in estimands:
            values = np.asarray([row[key] for row in selected], dtype=np.float64)
            if len(values) >= 2:
                paired[key] = paired_bootstrap_mean(
                    values, seed=bootstrap_seed + offset, replicates=bootstrap_replicates
                )
                offset += 1
            else:
                paired[key] = {"observations": int(len(values)), "status": "insufficient_for_bootstrap"}
        output[cohort_name] = {
            "observations": len(selected), "paired_estimands": paired,
            "descriptive": {
                "mean_occupancy": float(np.mean([row["cam_occupancy"] for row in selected])) if selected else None,
                "degenerate_fraction": float(np.mean([row["cam_degenerate"] for row in selected])) if selected else None,
                "mean_amplitude_spearman": float(np.mean([row["amplitude_cam_spearman"] for row in selected])) if selected else None,
                "mean_noise_spearman": float(np.mean([row["noise_cam_spearman"] for row in selected])) if selected else None,
            },
        }
    return output


def batch_gradcam(model, target_layer, values: torch.Tensor):
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(values)
        logits.sum().backward()
        activation = captured["activation"]
        weights = activation.grad.mean(dim=-1, keepdim=True)
        raw = torch.relu((weights * activation).sum(dim=1))
        minimum = raw.amin(dim=1, keepdim=True)
        span = raw.amax(dim=1, keepdim=True) - minimum
        degenerate = span.squeeze(1) <= 1e-12
        masks = torch.where(span > 1e-12, (raw - minimum) / span.clamp_min(1e-12), torch.zeros_like(raw))
        return logits.detach(), masks.detach(), degenerate.detach()
    finally:
        hook.remove()


@torch.no_grad()
def predict(model, values: torch.Tensor, batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(values), batch_size):
        output.append(model(values[start : start + batch_size]).float().cpu().numpy())
    return np.concatenate(output)


def selected_mask(mask: np.ndarray, fraction: float) -> np.ndarray:
    count = max(1, int(round(mask.shape[-1] * fraction)))
    indices = np.argpartition(mask, mask.shape[-1] - count, axis=1)[:, -count:]
    selected = np.zeros(mask.shape, dtype=bool)
    np.put_along_axis(selected, indices, True, axis=1)
    return selected


def finite_spearman(left, right):
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    transfer = json.loads(args.transfer_summary.read_text(encoding="utf-8"))
    if not transfer.get("prespecified_gradcam_gate", {}).get("passed"):
        raise ValueError("zero-shot transfer gate did not pass")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.model_code_root.resolve()))
    from LibMTL.model.singlelead_af import load_singlelead_af_checkpoint

    device = torch.device(args.device)
    model, checkpoint = load_singlelead_af_checkpoint(args.checkpoint.resolve(), map_location=device)
    model.to(device).eval()
    waveforms = np.load(args.input_dir / "ecg_ptb_normalized_100hz.npy")
    labels = np.load(args.input_dir / "afib_labels.npy").astype(np.uint8)
    groups = np.load(args.input_dir / "source_record_names.npy").astype(str)
    if waveforms.shape != (len(labels), 1, 400) or len(groups) != len(labels):
        raise ValueError("faithfulness inputs are not aligned single-lead windows")
    if args.max_windows is not None:
        waveforms, labels, groups = waveforms[: args.max_windows], labels[: args.max_windows], groups[: args.max_windows]
    if not args.fractions or any(not 0 < value < 1 for value in args.fractions):
        raise ValueError("fractions must lie strictly between zero and one")

    rng = np.random.default_rng(args.seed)
    rows = []
    zero = torch.zeros((1, 1, 400), device=device)
    with torch.no_grad():
        null_logit = float(model(zero).item())
    for start in range(0, len(waveforms), args.cam_batch_size):
        stop = min(start + args.cam_batch_size, len(waveforms))
        values_np = waveforms[start:stop, 0].astype(np.float32)
        values = torch.from_numpy(values_np[:, None, :]).to(device)
        baseline, masks_tensor, degenerate_tensor = batch_gradcam(model, model.gradcam_target_layer, values)
        masks = masks_tensor.cpu().numpy()
        degenerate = degenerate_tensor.cpu().numpy()

        amplitude_values = values * args.amplitude_scale
        _, amplitude_masks_tensor, _ = batch_gradcam(model, model.gradcam_target_layer, amplitude_values)
        noise_scale = np.maximum(np.std(values_np, axis=1, keepdims=True), 1e-8) * args.noise_std_fraction
        noisy_np = values_np + rng.normal(size=values_np.shape).astype(np.float32) * noise_scale
        noisy_values = torch.from_numpy(noisy_np[:, None, :]).to(device)
        _, noise_masks_tensor, _ = batch_gradcam(model, model.gradcam_target_layer, noisy_values)
        amplitude_masks = amplitude_masks_tensor.cpu().numpy()
        noise_masks = noise_masks_tensor.cpu().numpy()
        baseline_np = baseline.cpu().numpy()

        batch_rows = []
        for local in range(stop - start):
            batch_rows.append({
                "row_index": start + local,
                "label": int(labels[start + local]),
                "cam_degenerate": bool(degenerate[local]),
                "cam_occupancy": float(np.mean(masks[local])),
                "amplitude_cam_spearman": finite_spearman(masks[local], amplitude_masks[local]),
                "noise_cam_spearman": finite_spearman(masks[local], noise_masks[local]),
            })

        for fraction in args.fractions:
            suffix = f"top{int(round(100 * fraction))}"
            selected = selected_mask(masks, fraction)
            shifted = np.roll(selected, selected.shape[1] // 2, axis=1)
            deleted = values_np.copy(); deleted[selected] = 0.0
            shifted_deleted = values_np.copy(); shifted_deleted[shifted] = 0.0
            inserted = np.zeros_like(values_np); inserted[selected] = values_np[selected]
            deletion_score = predict(model, torch.from_numpy(deleted[:, None, :]).to(device), args.score_batch_size)
            shifted_score = predict(model, torch.from_numpy(shifted_deleted[:, None, :]).to(device), args.score_batch_size)
            insertion_score = predict(model, torch.from_numpy(inserted[:, None, :]).to(device), args.score_batch_size)
            random_deletion = np.empty((len(values_np), args.random_replicates), dtype=np.float32)
            random_insertion = np.empty_like(random_deletion)
            selected_count = int(selected.sum(axis=1)[0])
            for replicate in range(args.random_replicates):
                random_selected = np.zeros_like(selected)
                for row_index in range(len(values_np)):
                    random_selected[row_index, rng.choice(400, selected_count, replace=False)] = True
                random_deleted = values_np.copy(); random_deleted[random_selected] = 0.0
                random_inserted = np.zeros_like(values_np); random_inserted[random_selected] = values_np[random_selected]
                random_deletion[:, replicate] = predict(
                    model, torch.from_numpy(random_deleted[:, None, :]).to(device), args.score_batch_size
                )
                random_insertion[:, replicate] = predict(
                    model, torch.from_numpy(random_inserted[:, None, :]).to(device), args.score_batch_size
                )
            deletion = baseline_np - deletion_score
            shifted_drop = baseline_np - shifted_score
            insertion = insertion_score - null_logit
            random_deletion_drop = baseline_np[:, None] - random_deletion
            random_insertion_gain = random_insertion - null_logit
            for local, row in enumerate(batch_rows):
                random_drop = float(np.mean(random_deletion_drop[local]))
                random_gain = float(np.mean(random_insertion_gain[local]))
                row.update({
                    f"deletion_drop_{suffix}": float(deletion[local]),
                    f"random_deletion_drop_{suffix}": random_drop,
                    f"deletion_advantage_{suffix}": float(deletion[local] - random_drop),
                    f"shifted_deletion_drop_{suffix}": float(shifted_drop[local]),
                    f"insertion_gain_{suffix}": float(insertion[local]),
                    f"random_insertion_gain_{suffix}": random_gain,
                    f"insertion_advantage_{suffix}": float(insertion[local] - random_gain),
                })
        rows.extend(batch_rows)
        if args.progress_every and stop % args.progress_every < args.cam_batch_size:
            print(f"audited {stop}/{len(waveforms)}", flush=True)

    with (output / "per_window_faithfulness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    analysis_rows = aggregate_rows(rows, groups)
    cohorts = summarize_rows(analysis_rows, args.bootstrap_seed, args.bootstrap_replicates)
    summary = {
        "schema_version": 1, "status": "completed_singlelead_gradcam_faithfulness",
        "dataset": "MIMIC PERform AF Lead II", "target": "AFIB pre-sigmoid logit",
        "analysis_unit": "source-record mean", "primary_cohort": "AF_positive",
        "cohort_summaries": cohorts,
        "protocol": {
            "fractions": args.fractions, "random_replicates": args.random_replicates,
            "replacement": "zero in PTB-XL training-standardized input domain",
            "amplitude_scale": args.amplitude_scale, "noise_std_fraction": args.noise_std_fraction,
            "gradcam_target_layer": "feature_projection.0",
        },
        "checkpoint_metadata": {"epoch": checkpoint["epoch"], "input_spec": checkpoint["input_spec"]},
        "inputs_sha256": {
            "checkpoint": sha256(args.checkpoint.resolve()),
            "preprocessing_manifest": sha256(args.input_dir.resolve() / "manifest.json"),
            "transfer_summary": sha256(args.transfer_summary.resolve()),
            "waveforms": sha256(args.input_dir.resolve() / "ecg_ptb_normalized_100hz.npy"),
            "labels": sha256(args.input_dir.resolve() / "afib_labels.npy"),
            "groups": sha256(args.input_dir.resolve() / "source_record_names.npy"),
        },
        "execution": {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv), "device": str(device)},
        "claim_boundary": "Deletion tests frozen-classifier temporal faithfulness, not anatomical localization or clinical validity; insertion is retained as a separate off-manifold sensitivity result.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_records": len(analysis_rows), "AF_positive": cohorts["AF_positive"]}, indent=2))
    return output


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_code_root", type=Path, required=True)
    parser.add_argument("--transfer_summary", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--random_replicates", type=int, default=10)
    parser.add_argument("--amplitude_scale", type=float, default=1.05)
    parser.add_argument("--noise_std_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--bootstrap_seed", type=int, default=2031)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--cam_batch_size", type=int, default=32)
    parser.add_argument("--score_batch_size", type=int, default=256)
    parser.add_argument("--max_windows", type=int)
    parser.add_argument("--progress_every", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
