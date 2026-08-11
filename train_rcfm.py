"""Train RCFM for ECG-to-ECG, PPG-to-ECG, or RCG-to-ECG generation."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data import get_ecg2ecg_datasets, get_ppg2ecg_datasets
from src.rcfm.training import run_training


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


def parse_lead_indices(value) -> list[int] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    indices = [int(item) for item in items]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("target lead indices must be nonempty, unique, and nonnegative")
    return indices


def build_datasets(
    task: str,
    datasets: Iterable[str],
    data_root: str,
    window_size: int,
    normalization_metadata: dict[str, object] | None = None,
    normalization_id: str = "training_global_zscore_v1",
    condition_lead_index: int | None = None,
    target_lead_index: int | None = None,
    target_lead_indices: list[int] | str | None = None,
    load_train: bool = True,
    heldout_split: str = "val",
    max_train_records: int | None = None,
    max_heldout_records: int | None = None,
):
    if task in {"ppg2ecg", "rcg2ecg"}:
        return get_ppg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=list(datasets),
            window_size=window_size,
            clean_condition_ppg=False,
            normalization_metadata=normalization_metadata,
            normalization_id=normalization_id,
            load_train=load_train,
            max_train_records=max_train_records,
            max_heldout_records=max_heldout_records,
        )
    if task == "ecg2ecg":
        parsed_targets = parse_lead_indices(target_lead_indices)
        if parsed_targets is None and target_lead_index is not None:
            parsed_targets = [target_lead_index]
        if condition_lead_index is None or parsed_targets is None:
            raise ValueError("ecg2ecg requires condition lead and target lead indices")
        if condition_lead_index < 0:
            raise ValueError("ECG lead indices must be nonnegative")
        return get_ecg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=list(datasets),
            window_size=window_size,
            condition_lead=condition_lead_index,
            target_lead=parsed_targets,
            normalization_metadata=normalization_metadata,
            normalization_id=normalization_id,
            load_train=load_train,
            heldout_split=heldout_split,
            max_train_records=max_train_records,
            max_heldout_records=max_heldout_records,
        )
    raise ValueError(f"Unknown task={task!r}")


def train(args: argparse.Namespace) -> None:
    """Run the instrumented canonical trainer."""

    run_training(args, build_datasets)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["ecg2ecg", "ppg2ecg", "rcg2ecg"], default="ppg2ecg")
    parser.add_argument("--datasets", default="MIMIC-AFib", help="Comma-separated dataset names.")
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--config", default=None, help="JSON-formatted resolved-config template.")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--dataset_version", default=None)
    parser.add_argument("--split_hash", default=None)
    parser.add_argument(
        "--normalization_id",
        choices=[
            "training_global_zscore_v1",
            "record_zscore_v1",
            "record_minmax_neg1_1_v1",
            "rddm_window_minmax_neg1_1_v1",
            "window_minmax_neg1_1_v1",
        ],
        default="training_global_zscore_v1",
    )
    parser.add_argument("--condition_unit", default=None)
    parser.add_argument("--target_unit", default=None)
    parser.add_argument("--alignment_id", default=None)
    parser.add_argument("--condition_lead", default=None)
    parser.add_argument("--target_lead", default=None)
    parser.add_argument("--condition_lead_index", type=int)
    parser.add_argument("--target_lead_index", type=int)
    parser.add_argument(
        "--target_lead_indices",
        default=None,
        help="Comma-separated target indices for joint multi-lead generation.",
    )
    parser.add_argument("--window_size", type=int, default=4, help="Window length in seconds at 128 Hz.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument(
        "--checkpoint_policy",
        choices=["full", "latest_only"],
        default="full",
        help="Use latest_only for disposable smoke runs to avoid duplicate large checkpoints.",
    )
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument(
        "--flow_matcher",
        choices=["conditional", "target", "sb", "vp"],
        default="conditional",
    )
    parser.add_argument(
        "--experiment_role",
        choices=["canonical", "path_ablation"],
        default="canonical",
    )
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--region_weight", type=float, default=0.01)
    parser.add_argument("--use_minibatch_ot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ot_method", choices=["exact", "sinkhorn", "unbalanced", "partial"], default="sinkhorn")
    parser.add_argument("--ot_reg", type=float, default=0.05)
    parser.add_argument("--ot_normalize_cost", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ot_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ot_strict_mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ot_sampling_strategy",
        choices=["multinomial", "assignment"],
        default="multinomial",
    )
    parser.add_argument("--ot_log_interval_steps", type=int, default=1)
    parser.add_argument("--ot_histogram_interval_steps", type=int, default=100)
    parser.add_argument("--association_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--validation_interval_epochs", type=int, default=1)
    parser.add_argument(
        "--heldout_role",
        choices=["validation", "upstream_test_final_only"],
        default="validation",
    )
    parser.add_argument("--validation_max_batches", type=int, default=None)
    parser.add_argument("--validation_fixed_noise_seed", type=int, default=2025)
    parser.add_argument(
        "--wandb_mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_job_type", default="train")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="Debug option for short smoke runs.")
    parser.add_argument(
        "--max_train_records",
        type=int,
        default=None,
        help="Smoke-only cap applied after fitting normalization on the full training split.",
    )
    parser.add_argument(
        "--max_heldout_records",
        type=int,
        default=None,
        help="Smoke-only cap on validation records.",
    )
    return parser


def parse_args_with_config(
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    known, _ = preliminary.parse_known_args(argv)
    parser = parser or build_argparser()
    if known.config:
        config_path = Path(known.config)
        defaults = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            raise ValueError("config template must contain a JSON object")
        unknown = sorted(set(defaults) - {action.dest for action in parser._actions})
        if unknown:
            raise ValueError(f"config template has unknown keys: {unknown}")
        parser.set_defaults(**defaults)
    parsed = parser.parse_args(argv)
    required = (
        "dataset_version", "split_hash", "condition_unit", "target_unit",
        "alignment_id", "condition_lead", "target_lead",
    )
    missing = [name for name in required if not getattr(parsed, name)]
    if missing:
        parser.error("missing required experiment metadata: " + ", ".join(missing))
    return parsed


if __name__ == "__main__":
    parsed_args = parse_args_with_config()
    if not parsed_args.data_root:
        raise ValueError("--data_root or RCFM_DATA_ROOT is required")
    if not parsed_args.output_dir:
        raise ValueError("--output_dir or RCFM_RUNS_ROOT is required")
    train(parsed_args)
