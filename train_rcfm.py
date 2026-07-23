"""Train RCFM for ECG-to-ECG, PPG-to-ECG, or RCG-to-ECG generation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_ecg2ecg_datasets, get_ppg2ecg_datasets
from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching


def set_deterministic(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_datasets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_datasets(task: str, datasets: Iterable[str], data_root: str, window_size: int):
    if task in {"ppg2ecg", "rcg2ecg"}:
        return get_ppg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=list(datasets),
            window_size=window_size,
            clean_condition_ppg=task == "ppg2ecg",
        )
    if task == "ecg2ecg":
        return get_ecg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=list(datasets),
            window_size=window_size,
        )
    raise ValueError(f"Unknown task={task!r}")


def train(args: argparse.Namespace) -> None:
    set_deterministic(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = parse_datasets(args.datasets)
    train_set, _ = build_datasets(args.task, datasets, args.data_root, args.window_size)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=args.use_minibatch_ot,
    )

    signal_length = args.window_size * 128
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(
        signal_length,
        1,
        device=str(device),
        num_heads=args.attention_heads,
    ).to(device)
    rcfm = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=args.flow_matcher,
        sigma=args.sigma,
        region_weight=args.region_weight,
        use_minibatch_ot=args.use_minibatch_ot,
        ot_method=args.ot_method,
        ot_reg=args.ot_reg,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(rcfm.parameters()) + list(condition_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    main_epochs = max(args.epochs - warmup_epochs, 1)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=max(warmup_epochs, 1)),
            CosineAnnealingLR(optimizer, T_max=main_epochs),
        ],
        milestones=[warmup_epochs],
    )

    run_dir = Path(args.output_dir) / args.task / "-".join(datasets)
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        rcfm.train()
        condition_net.train()
        losses: list[float] = []
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")

        for target_ecg, condition_signal, region_mask in pbar:
            target_ecg = target_ecg.float().to(device)
            condition_signal = condition_signal.float().to(device)
            region_mask = region_mask.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            conditions = condition_net(condition_signal)
            output = rcfm(
                target=target_ecg,
                conditions=conditions,
                region_mask=region_mask,
            )
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(rcfm.parameters()) + list(condition_net.parameters()),
                max_norm=args.grad_clip,
            )
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

            if args.max_batches is not None and len(losses) >= args.max_batches:
                break

        scheduler.step()
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"epoch={epoch + 1} loss={mean_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3e}")

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save(rcfm.state_dict(), run_dir / f"rcfm_{args.flow_matcher}_epoch_{epoch + 1}.pth")
            torch.save(condition_net.state_dict(), run_dir / f"condition_net_epoch_{epoch + 1}.pth")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ecg2ecg", "ppg2ecg", "rcg2ecg"], default="ppg2ecg")
    parser.add_argument("--datasets", default="MIMIC-AFib", help="Comma-separated dataset names.")
    parser.add_argument("--data_root", default="/data/user/RCFM/data/")
    parser.add_argument("--output_dir", default="./saved/reviewer")
    parser.add_argument("--window_size", type=int, default=4, help="Window length in seconds at 128 Hz.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--flow_matcher", choices=["conditional", "target", "sb", "vp"], default="vp")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--region_weight", type=float, default=1.0)
    parser.add_argument("--use_minibatch_ot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ot_method", choices=["exact", "sinkhorn", "unbalanced", "partial"], default="sinkhorn")
    parser.add_argument("--ot_reg", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Debug option for short smoke runs.")
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
