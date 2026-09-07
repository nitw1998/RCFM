"""Train RCFM for ECG-to-ECG, PPG-to-ECG, or RCG-to-ECG generation."""

from __future__ import annotations

import argparse
import hashlib
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_external_region_masks(
    *,
    mask_path: str,
    manifest_path: str,
    mask_method: str,
    data_root: str,
    dataset_name: str,
    dataset_version: str,
    split_hash: str,
    window_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    array_path = Path(mask_path).resolve()
    metadata_path = Path(manifest_path).resolve()
    if not array_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("external region mask array and manifest must both exist")
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("schema_version") != 1:
        raise ValueError("region-mask manifest must be a completed schema-version-1 artifact")
    dataset = manifest.get("dataset", {})
    mask = manifest.get("mask", {})
    if dataset.get("dataset_version") != dataset_version:
        raise ValueError("region-mask manifest dataset_version does not match the experiment")
    if dataset.get("split_hash") != split_hash:
        raise ValueError("region-mask manifest split_hash does not match the experiment")
    if mask.get("method") != mask_method:
        raise ValueError("region-mask manifest method does not match --mask_method")
    if mask.get("file_name") != array_path.name:
        raise ValueError("region-mask array file name does not match its manifest")
    if bool(manifest.get("test_mask_generated", True)):
        raise ValueError("region-mask artifact must explicitly state test_mask_generated=false")
    if mask.get("sha256") != _sha256(array_path):
        raise ValueError("region-mask array SHA-256 does not match its manifest")
    source = dataset.get("source_ecg", {})
    source_file_name = source.get("file_name")
    if not isinstance(source_file_name, str) or Path(source_file_name).name != source_file_name:
        raise ValueError("region-mask manifest source ECG file name is invalid")
    source_path = Path(data_root).resolve() / dataset_name / source_file_name
    if not source_path.is_file():
        raise FileNotFoundError("region-mask source ECG training array is missing")
    if source.get("file_name") != source_path.name or source.get("sha256") != _sha256(source_path):
        raise ValueError("region-mask source ECG provenance does not match the training array")
    masks = np.load(array_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(value) for value in mask.get("shape", []))
    if masks.shape != expected_shape or str(masks.dtype) != mask.get("dtype"):
        raise ValueError("region-mask array shape/dtype does not match its manifest")
    if len(masks) != int(dataset.get("train_records", -1)):
        raise ValueError("region-mask row count does not match its manifest")
    provenance = {
        "method": mask_method,
        "manifest_sha256": _sha256(metadata_path),
        "array_sha256": mask["sha256"],
        "source_ecg_sha256": source["sha256"],
        "test_mask_generated": False,
        "claim_boundary": manifest.get("claim_boundary"),
    }
    return masks, provenance


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
    return_region_mask_train: bool = True,
    region_mask_path: str | None = None,
    region_mask_manifest: str | None = None,
    mask_method: str = "cached_target_ecg_r_peak_roi",
    dataset_version: str | None = None,
    split_hash: str | None = None,
):
    dataset_names = list(datasets)
    if region_mask_path is not None and len(dataset_names) != 1:
        raise ValueError("external region masks currently require exactly one dataset")
    if task in {"ppg2ecg", "rcg2ecg"}:
        masks = None
        provenance = None
        if region_mask_path is not None:
            if task != "ppg2ecg" or region_mask_manifest is None:
                raise ValueError("external Grad-CAM masks require ppg2ecg and a manifest")
            masks, provenance = _load_external_region_masks(
                mask_path=region_mask_path,
                manifest_path=region_mask_manifest,
                mask_method=mask_method,
                data_root=data_root,
                dataset_name=dataset_names[0],
                dataset_version=str(dataset_version),
                split_hash=str(split_hash),
                window_size=window_size,
            )
        train_set, heldout_set = get_ppg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=dataset_names,
            window_size=window_size,
            clean_condition_ppg=False,
            normalization_metadata=normalization_metadata,
            normalization_id=normalization_id,
            load_train=load_train,
            max_train_records=max_train_records,
            max_heldout_records=max_heldout_records,
            return_region_mask_train=return_region_mask_train,
            region_masks_train=masks,
        )
        if train_set is not None and provenance is not None:
            train_set.region_mask_provenance = provenance
        return train_set, heldout_set
    if task == "ecg2ecg":
        masks = None
        provenance = None
        if region_mask_path is not None:
            if region_mask_manifest is None:
                raise ValueError("external Grad-CAM masks require a manifest")
            masks, provenance = _load_external_region_masks(
                mask_path=region_mask_path,
                manifest_path=region_mask_manifest,
                mask_method=mask_method,
                data_root=data_root,
                dataset_name=dataset_names[0],
                dataset_version=str(dataset_version),
                split_hash=str(split_hash),
                window_size=window_size,
            )
        parsed_targets = parse_lead_indices(target_lead_indices)
        if parsed_targets is None and target_lead_index is not None:
            parsed_targets = [target_lead_index]
        if condition_lead_index is None or parsed_targets is None:
            raise ValueError("ecg2ecg requires condition lead and target lead indices")
        if condition_lead_index < 0:
            raise ValueError("ECG lead indices must be nonnegative")
        train_set, heldout_set = get_ecg2ecg_datasets(
            DATA_PATH=data_root,
            datasets=dataset_names,
            window_size=window_size,
            condition_lead=condition_lead_index,
            target_lead=parsed_targets,
            normalization_metadata=normalization_metadata,
            normalization_id=normalization_id,
            load_train=load_train,
            heldout_split=heldout_split,
            max_train_records=max_train_records,
            max_heldout_records=max_heldout_records,
            return_region_mask_train=return_region_mask_train,
            region_masks_train=masks,
        )
        if train_set is not None and provenance is not None:
            train_set.region_mask_provenance = provenance
        return train_set, heldout_set
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
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Schema-2 checkpoint used for strict stateful continuation into a new run directory.",
    )
    parser.add_argument(
        "--resume_lr_policy",
        choices=["restart_cosine"],
        default="restart_cosine",
        help="Resume policy: retain AdamW moments but restart base LR over remaining epochs.",
    )
    parser.add_argument(
        "--resume_restart_lr",
        type=float,
        default=None,
        help="Peak LR for restart_cosine; defaults to the original --lr.",
    )
    parser.add_argument("--dataset_version", default=None)
    parser.add_argument("--split_hash", default=None)
    parser.add_argument(
        "--normalization_id",
        choices=[
            "training_global_zscore_v1",
            "record_zscore_v1",
            "record_minmax_neg1_1_v1",
            "record_joint12_minmax_neg1_1_v1",
            "source_record_joint12_minmax_neg1_1_v1",
            "source_record_minmax_neg1_1_v1",
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
    parser.add_argument("--region_mask_path", default=None)
    parser.add_argument("--region_mask_manifest", default=None)
    parser.add_argument("--mask_method", default=None)
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
