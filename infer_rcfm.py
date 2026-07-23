"""Run RCFM inference from saved checkpoints."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_rcfm import build_datasets, parse_datasets
from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching


def latest_checkpoint(directory: Path, prefix: str) -> Path:
    candidates = list(directory.glob(f"{prefix}*_epoch_*.pth")) + list(directory.glob(f"{prefix}*_epoch*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching {prefix}*_epoch*.pth in {directory}")

    def epoch(path: Path) -> int:
        match = re.search(r"_epoch_?(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch)


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
    datasets = parse_datasets(args.datasets)
    _, test_set = build_datasets(args.task, datasets, args.data_root, args.window_size)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    signal_length = args.window_size * 128
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(signal_length, 1, device=str(device), num_heads=args.attention_heads).to(device)
    rcfm = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=args.flow_matcher,
        sigma=args.sigma,
        region_weight=args.region_weight,
        use_minibatch_ot=False,
    ).to(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    rcfm_path = Path(args.rcfm_checkpoint) if args.rcfm_checkpoint else latest_checkpoint(checkpoint_dir, "rcfm")
    cond_path = Path(args.condition_checkpoint) if args.condition_checkpoint else latest_checkpoint(checkpoint_dir, "condition_net")

    rcfm.load_state_dict(torch.load(rcfm_path, map_location=device))
    condition_net.load_state_dict(torch.load(cond_path, map_location=device))
    rcfm.eval()
    condition_net.eval()

    predictions, targets, conditions_np = [], [], []
    for target_ecg, condition_signal, _ in loader:
        target_ecg = target_ecg.float().to(device)
        condition_signal = condition_signal.float().to(device)
        conditions = condition_net(condition_signal)
        pred = rcfm.sample(
            conditions=conditions,
            shape=tuple(target_ecg.shape),
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
    parser.add_argument("--task", choices=["ecg2ecg", "ppg2ecg", "rcg2ecg"], default="ppg2ecg")
    parser.add_argument("--datasets", default="MIMIC-AFib")
    parser.add_argument("--data_root", default="/data/user/RCFM/data/")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--rcfm_checkpoint", default=None)
    parser.add_argument("--condition_checkpoint", default=None)
    parser.add_argument("--output_dir", default="./outputs/rcfm")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--flow_matcher", choices=["conditional", "target", "sb", "vp"], default="vp")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--region_weight", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    return parser


if __name__ == "__main__":
    infer(build_argparser().parse_args())
