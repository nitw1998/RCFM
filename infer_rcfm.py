"""Run RCFM inference from saved checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_rcfm import build_datasets
from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching
from src.rcfm.checkpoint import load_checkpoint


def _record_coefficients_for_signals(
    values: np.ndarray,
    signals: np.ndarray,
    name: str,
) -> np.ndarray:
    """Align per-record or per-record/per-lead coefficients with (N, C, T)."""

    values = np.asarray(values, dtype=np.float32)
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim != 3:
        raise ValueError("record inverse transform requires (records, channels, samples)")
    expected = signals.shape[:-1]
    if values.shape == (signals.shape[0],) and signals.shape[1] == 1:
        values = values[:, None]
    if values.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
    return values[..., None]


def _inverse_record_zscore(
    signals: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    means = _record_coefficients_for_signals(means, signals, "record means")
    scales = _record_coefficients_for_signals(scales, signals, "record scales")
    return np.asarray(signals, dtype=np.float32) * scales + means


def _inverse_record_minmax_neg1_1(
    signals: np.ndarray,
    offsets: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    offsets = _record_coefficients_for_signals(offsets, signals, "record offsets")
    ranges = _record_coefficients_for_signals(ranges, signals, "record ranges")
    return (np.asarray(signals, dtype=np.float32) + 1.0) * ranges / 2.0 + offsets


def plot_prediction(condition: np.ndarray, target: np.ndarray, prediction: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
    for ax, signal, title in zip(
        axes,
        [condition, target, prediction],
        ["Condition signal", "Target ECG", "RCFM prediction"],
    ):
        ax.plot(signal, linewidth=1.0)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Sample")
    fig.savefig(path, dpi=300)
    plt.close(fig)


@torch.no_grad()
def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    datasets = config["datasets"]
    _, test_set = build_datasets(
        config["task"],
        datasets,
        args.data_root,
        int(config["window_size"]),
        normalization_metadata=checkpoint["normalization"],
        normalization_id=config["normalization_id"],
        condition_lead_index=config["condition_lead_index"],
        target_lead_index=config["target_lead_index"],
        target_lead_indices=config.get("target_lead_indices"),
        load_train=False,
        heldout_split=args.split,
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_spec = checkpoint["output_spec"]
    signal_length = int(output_spec["length"])
    output_channels = int(output_spec["channels"])
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(
        signal_length,
        output_channels,
        device=str(device),
        num_heads=int(config["attention_heads"]),
    ).to(device)
    rcfm = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=config["flow_matcher"],
        sigma=float(config["sigma"]),
        region_weight=float(config["region_weight"]),
        use_minibatch_ot=False,
    ).to(device)

    rcfm.load_state_dict(checkpoint["model_state"])
    condition_net.load_state_dict(checkpoint["condition_state"])
    rcfm.eval()
    condition_net.eval()

    predictions, targets, conditions_np = [], [], []
    for target_ecg, condition_signal in loader:
        target_ecg = target_ecg.float().to(device)
        condition_signal = condition_signal.float().to(device)
        conditions = condition_net(condition_signal)
        pred = rcfm.sample(
            conditions=conditions,
            shape=(target_ecg.shape[0], output_channels, signal_length),
            steps=args.steps,
            device=device,
        )
        predictions.append(pred.cpu().numpy())
        targets.append(target_ecg.cpu().numpy())
        conditions_np.append(condition_signal.cpu().numpy())
        if sum(batch.shape[0] for batch in predictions) >= args.num_samples:
            break

    predictions_np = np.concatenate(predictions, axis=0)[: args.num_samples]
    targets_np = np.concatenate(targets, axis=0)[: args.num_samples]
    conditions_np = np.concatenate(conditions_np, axis=0)[: args.num_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rcfm_predictions.npy", predictions_np)
    np.save(output_dir / "rcfm_targets_normalized.npy", targets_np)
    np.save(output_dir / "condition_signals_normalized.npy", conditions_np)
    inverse_metadata: dict[str, object] = {
        "record_scalers_saved": False,
        "real_signals_inverse_transformed": False,
        "generated_signal_inverse_transform": "not_available_for_global_normalization",
    }
    if config["normalization_id"] == "record_zscore_v1":
        required_attributes = (
            "record_ids",
            "target_means",
            "target_scales",
            "condition_means",
            "condition_scales",
        )
        if any(getattr(test_set, name, None) is None for name in required_attributes):
            raise ValueError("record-zscore inference dataset is missing scaler metadata")
        count = len(predictions_np)
        record_ids = np.asarray(test_set.record_ids[:count])
        target_means = np.asarray(test_set.target_means[:count], dtype=np.float32)
        target_scales = np.asarray(test_set.target_scales[:count], dtype=np.float32)
        condition_means = np.asarray(test_set.condition_means[:count], dtype=np.float32)
        condition_scales = np.asarray(test_set.condition_scales[:count], dtype=np.float32)
        np.save(output_dir / "record_ids.npy", record_ids, allow_pickle=False)
        np.savez(
            output_dir / "record_zscore_coefficients.npz",
            record_ids=record_ids,
            target_means=target_means,
            target_scales=target_scales,
            condition_means=condition_means,
            condition_scales=condition_scales,
        )
        target_raw = _inverse_record_zscore(targets_np, target_means, target_scales)
        condition_raw = _inverse_record_zscore(
            conditions_np, condition_means, condition_scales
        )
        np.save(output_dir / "rcfm_targets_source_values.npy", target_raw.astype(np.float32))
        np.save(
            output_dir / "condition_signals_source_values.npy",
            condition_raw.astype(np.float32),
        )
        generated_policy = "normalized_only_target_scaler_withheld"
        if args.allow_oracle_target_inverse:
            prediction_oracle = _inverse_record_zscore(
                predictions_np, target_means, target_scales
            )
            np.save(
                output_dir / "rcfm_predictions_oracle_target_scale.npy",
                prediction_oracle.astype(np.float32),
            )
            generated_policy = "oracle_ground_truth_target_scaler_explicitly_enabled"
        inverse_metadata = {
            "record_scalers_saved": True,
            "scaler_file": "record_zscore_coefficients.npz",
            "real_signals_inverse_transformed": True,
            "generated_signal_inverse_transform": generated_policy,
            "oracle_generated_file": (
                "rcfm_predictions_oracle_target_scale.npy"
                if args.allow_oracle_target_inverse
                else None
            ),
            "amplitude_claim_warning": (
                "CPSC source units are unknown; source-value outputs must not be labelled mV"
            ),
        }
    elif config["normalization_id"] == "record_minmax_neg1_1_v1":
        required_attributes = (
            "record_ids",
            "target_offsets",
            "target_scales",
            "condition_offsets",
            "condition_scales",
        )
        if any(getattr(test_set, name, None) is None for name in required_attributes):
            raise ValueError("record-minmax inference dataset is missing scaler metadata")
        count = len(predictions_np)
        record_ids = np.asarray(test_set.record_ids[:count])
        target_offsets = np.asarray(test_set.target_offsets[:count], dtype=np.float32)
        target_ranges = np.asarray(test_set.target_scales[:count], dtype=np.float32)
        condition_offsets = np.asarray(test_set.condition_offsets[:count], dtype=np.float32)
        condition_ranges = np.asarray(test_set.condition_scales[:count], dtype=np.float32)
        np.save(output_dir / "record_ids.npy", record_ids, allow_pickle=False)
        np.savez(
            output_dir / "record_minmax_neg1_1_coefficients.npz",
            record_ids=record_ids,
            target_offsets=target_offsets,
            target_ranges=target_ranges,
            condition_offsets=condition_offsets,
            condition_ranges=condition_ranges,
        )
        target_raw = _inverse_record_minmax_neg1_1(
            targets_np, target_offsets, target_ranges
        )
        condition_raw = _inverse_record_minmax_neg1_1(
            conditions_np, condition_offsets, condition_ranges
        )
        np.save(output_dir / "rcfm_targets_source_values.npy", target_raw.astype(np.float32))
        np.save(
            output_dir / "condition_signals_source_values.npy",
            condition_raw.astype(np.float32),
        )
        generated_policy = "normalized_only_target_scaler_withheld"
        if args.allow_oracle_target_inverse:
            prediction_oracle = _inverse_record_minmax_neg1_1(
                predictions_np, target_offsets, target_ranges
            )
            np.save(
                output_dir / "rcfm_predictions_oracle_target_scale.npy",
                prediction_oracle.astype(np.float32),
            )
            generated_policy = "oracle_ground_truth_target_scaler_explicitly_enabled"
        inverse_metadata = {
            "record_scalers_saved": True,
            "scaler_file": "record_minmax_neg1_1_coefficients.npz",
            "real_signals_inverse_transformed": True,
            "generated_signal_inverse_transform": generated_policy,
            "oracle_generated_file": (
                "rcfm_predictions_oracle_target_scale.npy"
                if args.allow_oracle_target_inverse
                else None
            ),
            "amplitude_claim_warning": (
                "CPSC source units are unknown; source-value outputs must not be labelled mV"
            ),
        }
    (output_dir / "inference_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "checkpoint": checkpoint_path.name,
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_git_commit": checkpoint["provenance"]["git_commit"],
                "checkpoint_git_dirty": checkpoint["provenance"]["git_dirty"],
                "task": config["task"],
                "datasets": datasets,
                "split_hash": config["split_hash"],
                "normalization_id": config["normalization_id"],
                "alignment_id": config["alignment_id"],
                "target_lead": output_spec["target_lead"],
                "steps": args.steps,
                "split": args.split,
                "output_spec": output_spec,
                "prediction_domain": "normalized",
                "inverse_transform": inverse_metadata,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plot_prediction(
        condition=conditions_np[0, 0],
        target=targets_np[0, 0],
        prediction=predictions_np[0, 0],
        path=output_dir / "rcfm_prediction.png",
    )
    print(f"saved predictions to {output_dir / 'rcfm_predictions.npy'}")
    print(f"saved figure to {output_dir / 'rcfm_prediction.png'}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow_oracle_target_inverse",
        action="store_true",
        help=(
            "Explicitly use each held-out ground-truth target mean/std to inverse-transform "
            "predictions. This is oracle analysis and must not be reported as deployable inference."
        ),
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_argparser().parse_args()
    if not parsed_args.data_root:
        raise ValueError("--data_root or RCFM_DATA_ROOT is required")
    if not parsed_args.output_dir:
        raise ValueError("--output_dir or RCFM_RUNS_ROOT is required")
    infer(parsed_args)
