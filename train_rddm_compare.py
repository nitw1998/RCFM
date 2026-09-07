"""Train an independently reproduced RDDM comparator on a frozen paired artifact."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import RDDM
from model import ConditionNet, DiffusionUNetCrossAttention
from src.rcfm.checkpoint import capture_rng_states
from src.rcfm.experiment import RunArtifacts, WandbLogger
from train_rcfm import build_datasets, parse_datasets, parse_lead_indices, set_deterministic


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument(
        "--task", choices=["ppg2ecg", "rcg2ecg", "ecg2ecg"], default="ppg2ecg"
    )
    parser.add_argument("--datasets", default="MIMIC-AFib")
    parser.add_argument("--dataset_version", default=None)
    parser.add_argument("--split_hash", default=None)
    parser.add_argument("--normalization_id", default=None)
    parser.add_argument("--alignment_id", default=None)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--condition_lead_index", type=int, default=None)
    parser.add_argument("--target_lead_indices", default=None)
    parser.add_argument("--heldout_split", choices=["val", "test"], default="test")
    parser.add_argument("--expected_train_windows", type=int, default=8400)
    parser.add_argument("--expected_test_windows", type=int, default=1800)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_t_max", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--nT", type=int, default=10)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.2)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--condition_drop_probability", type=float, default=0.0)
    parser.add_argument("--ddpm_loss_weight", type=float, default=100.0)
    parser.add_argument("--region_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--max_train_records", type=int, default=None)
    parser.add_argument("--max_heldout_records", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument(
        "--wandb_mode", choices=["disabled", "offline", "online"], default="disabled"
    )
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="mimic-afib-rddm-reproduced")
    parser.add_argument("--wandb_job_type", default="train")
    parser.add_argument("--upstream_repository", default=None)
    parser.add_argument("--upstream_commit", default=None)
    parser.add_argument("--upstream_license", default=None)
    parser.add_argument("--reproduction_label", default="RDDM (reproduced)")
    return parser


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", required=True)
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    config_path = Path(known.config)
    defaults = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        raise ValueError("RDDM config must contain a JSON object")
    known_fields = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known_fields)
    if unknown:
        raise ValueError(f"RDDM config has unknown keys: {unknown}")
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_config(args)
    return args


def validate_config(args: argparse.Namespace) -> None:
    datasets = parse_datasets(args.datasets)
    targets = parse_lead_indices(args.target_lead_indices)
    if args.task == "ppg2ecg" and datasets == ["MIMIC-AFib"]:
        if args.normalization_id != "rddm_window_minmax_neg1_1_v1":
            raise ValueError("MIMIC RDDM requires the frozen RDDM window normalization")
        if args.heldout_split != "test":
            raise ValueError("MIMIC RDDM requires the artifact's frozen test-named split")
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("MIMIC PPG-to-ECG must not declare ECG lead indices")
        if args.reproduction_label != "RDDM (reproduced)":
            raise ValueError("MIMIC RDDM must retain the reproduction label")
        protocols = {
            "mimic-afib-rddm-upstream-all-zero-ppg-qc-v1": (
                "a7e388293adaa7b48d3493efc505dd8750520730cae9fd7649157866efa86a51",
                "paired_array_row_rddm_contract_zero_ppg_qc_v1",
                8400,
                1800,
            ),
            "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1": (
                "8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232",
                "paired_source_row_no_phase_correction_random80_20_v1",
                8160,
                2040,
            ),
        }
        expected = protocols.get(args.dataset_version)
        if expected is None or (
            args.split_hash,
            args.alignment_id,
            args.expected_train_windows,
            args.expected_test_windows,
        ) != expected:
            raise ValueError("MIMIC RDDM provenance or window counts do not match a frozen protocol")
    elif args.task == "ecg2ecg" and datasets == ["PTBXL"]:
        expected_targets = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if args.heldout_split != "val":
            raise ValueError("PTB-XL RDDM requires a validation-named held-out split")
        if args.condition_lead_index != 1 or targets != expected_targets:
            raise ValueError("PTB-XL RDDM requires Lead II to the other 11 leads")
        protocols = {
            "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1": (
                "7784c98a2c8daccc23fc7cb0d47dc1933eeee77f120c05f8a24ac149cd8474f7",
                "record_minmax_neg1_1_v1",
                "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
                17440,
                2193,
                "RDDM-ECG (adapted)",
            ),
            "ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2": (
                "9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c",
                "source_record_joint12_minmax_neg1_1_v1",
                "ptbxl_all_records_two_nonoverlap_4s_windows_random80_20_seed31_lead_II_to_other11_v1",
                34939,
                8735,
                "RDDM-ECG (random-window adapted)",
            ),
        }
        expected = protocols.get(args.dataset_version)
        if expected is None or (
            args.split_hash,
            args.normalization_id,
            args.alignment_id,
            args.expected_train_windows,
            args.expected_test_windows,
            args.reproduction_label,
        ) != expected:
            raise ValueError("PTB-XL RDDM provenance, counts, normalization, or label are invalid")
    elif args.task == "ecg2ecg" and datasets == ["CPSC2018"]:
        expected_targets = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        if args.heldout_split != "val":
            raise ValueError("CPSC2018 RDDM requires a validation-named held-out split")
        if args.condition_lead_index != 1 or targets != expected_targets:
            raise ValueError("CPSC2018 RDDM requires Lead II to the other 11 leads")
        protocols = {
            "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1": (
                "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223",
                "record_minmax_neg1_1_v1",
                "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
                5487,
                686,
                "RDDM-ECG (adapted)",
            ),
            "cpsc2018-source-fullrecord-joint12-minmax-all-nonoverlap4s-random80-20-record-overlap-v1": (
                "b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8",
                "source_record_joint12_minmax_neg1_1_v1",
                "cpsc2018_all_complete_nonoverlap_4s_windows_random80_20_seed31_lead_II_to_other11_v1",
                19364,
                4842,
                "RDDM-ECG (random-window adapted)",
            ),
        }
        expected = protocols.get(args.dataset_version)
        if expected is None or (
            args.split_hash,
            args.normalization_id,
            args.alignment_id,
            args.expected_train_windows,
            args.expected_test_windows,
            args.reproduction_label,
        ) != expected:
            raise ValueError("CPSC2018 RDDM provenance, counts, normalization, or label are invalid")
    elif args.task == "rcg2ecg" and datasets == ["mmECG"]:
        if args.normalization_id != "window_minmax_neg1_1_v1":
            raise ValueError("mmECG RDDM requires the frozen window min-max normalization")
        if args.heldout_split != "test":
            raise ValueError("mmECG RDDM requires the artifact's frozen test-named split")
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("mmECG RCG-to-ECG must not declare ECG lead indices")
        protocols = {
            "mmecg-public-20221108-subject-split-window-minmax-v1": (
                "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f",
                "same_record_same_window_no_additional_phase_correction_subject_split_v1",
                9590,
                2877,
                "RDDM-RCG (adapted)",
            ),
            "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1": (
                "6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2",
                "same_record_same_window_no_delay_correction_random80_20_v1",
                9973,
                2494,
                "RDDM-RCG (random-window adapted)",
            ),
        }
        expected = protocols.get(args.dataset_version)
        if expected is None or (
            args.split_hash,
            args.alignment_id,
            args.expected_train_windows,
            args.expected_test_windows,
            args.reproduction_label,
        ) != expected:
            raise ValueError("mmECG RDDM provenance, counts, or adaptation label are invalid")
    elif args.task == "ppg2ecg" and datasets == ["WESAD"]:
        if args.heldout_split != "test":
            raise ValueError("WESAD RDDM requires the artifact's frozen test-named split")
        if args.condition_lead_index is not None or targets is not None:
            raise ValueError("WESAD PPG-to-ECG must not declare ECG lead indices")
        protocols = {
            "wesad-subject-fold1-linear-resample-window-minmax-v1": (
                "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd",
                "window_minmax_neg1_1_v1",
                "native_common_start_same_window_no_delay_correction_subject_fold1_v1",
                17494,
                4213,
                "RDDM-PPG (matched-protocol reproduction)",
            ),
            "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2": (
                "ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c",
                "source_record_minmax_neg1_1_v1",
                "native_common_start_same_window_no_delay_correction_random80_20_v1",
                17365,
                4342,
                "RDDM-PPG (random-window record-minmax adaptation)",
            ),
            "wesad-subject-fold1-train-fixed-lag-aligned-v2": (
                "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd",
                "window_minmax_neg1_1_v1",
                "train_subjects_peak_median_fixed_lag_crop_before_window_subject_fold1_v2",
                17494,
                4213,
                "RDDM-PPG (matched-protocol reproduction)",
            ),
        }
        expected = protocols.get(args.dataset_version)
        if expected is None or (
            args.split_hash,
            args.normalization_id,
            args.alignment_id,
            args.expected_train_windows,
            args.expected_test_windows,
            args.reproduction_label,
        ) != expected:
            raise ValueError("WESAD RDDM provenance, counts, normalization, or label are invalid")
    else:
        raise ValueError(
            "RDDM comparator supports only frozen MIMIC PPG-to-ECG, "
            "PTB-XL or CPSC2018 ECG-to-ECG, mmECG RCG-to-ECG, "
            "or WESAD PPG-to-ECG"
        )
    if args.nT != 10 or args.beta_start != 1e-4 or args.beta_end != 0.2:
        raise ValueError("RDDM reproduction requires nT=10 and betas=(1e-4, 0.2)")
    if args.ddpm_loss_weight != 100.0 or args.region_loss_weight != 1.0:
        raise ValueError("RDDM reproduction requires loss weights alpha1=100 and alpha2=1")
    if args.condition_drop_probability != 0.0:
        raise ValueError("RDDM reproduction requires condition_drop_probability=0")
    if args.epochs <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("epochs, batch_size, and save_every must be positive")
    for name in ("max_train_records", "max_heldout_records", "max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when supplied")
    if args.epochs % args.save_every != 0:
        raise ValueError("epochs must be divisible by save_every for a frozen checkpoint schedule")
    if args.scheduler_t_max != 1000:
        raise ValueError("RDDM reproduction keeps the upstream scheduler T_max=1000")
    if not args.dataset_version or not args.split_hash or not args.alignment_id:
        raise ValueError("RDDM dataset provenance fields are required")
    if args.upstream_commit != "7d5348843c3985c211a23ae5105a2d9497d5156a":
        raise ValueError("RDDM upstream commit must match the pinned reproduction source")


def checkpoint_epochs(epochs: int, save_every: int) -> list[int]:
    if epochs <= 0 or save_every <= 0 or epochs % save_every != 0:
        raise ValueError("checkpoint schedule requires divisible positive epochs")
    return list(range(save_every, epochs + 1, save_every))


def _repository_state(root: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip()), status


def train(args: argparse.Namespace) -> Path:
    validate_config(args)
    if not args.data_root:
        raise ValueError("--data_root or RCFM_DATA_ROOT is required")
    if not args.output_dir:
        raise ValueError("--output_dir or RCFM_RUNS_ROOT is required")
    set_deterministic(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    datasets = parse_datasets(args.datasets)
    target_leads = parse_lead_indices(args.target_lead_indices)
    train_set, heldout_set = build_datasets(
        task=args.task,
        datasets=datasets,
        data_root=args.data_root,
        window_size=args.window_size,
        normalization_id=args.normalization_id,
        condition_lead_index=args.condition_lead_index,
        target_lead_indices=target_leads,
        heldout_split=args.heldout_split,
        max_train_records=args.max_train_records,
        max_heldout_records=args.max_heldout_records,
    )
    expected_train = (
        min(args.expected_train_windows, args.max_train_records)
        if args.max_train_records is not None
        else args.expected_train_windows
    )
    expected_heldout = (
        min(args.expected_test_windows, args.max_heldout_records)
        if args.max_heldout_records is not None
        else args.expected_test_windows
    )
    if len(train_set) != expected_train or len(heldout_set) != expected_heldout:
        raise ValueError(
            "RDDM artifact sizes disagree with the frozen train/held-out protocol"
        )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    signal_length = args.window_size * 128
    target_channels = 1 if target_leads is None else len(target_leads)
    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(
            signal_length, target_channels, str(device), num_heads=args.attention_heads
        ),
        region_model=DiffusionUNetCrossAttention(
            signal_length, target_channels, str(device), num_heads=args.attention_heads
        ),
        betas=(args.beta_start, args.beta_end),
        n_T=args.nT,
    ).to(device)
    condition_net_1 = ConditionNet().to(device)
    condition_net_2 = ConditionNet().to(device)
    parameters = [
        *rddm.parameters(),
        *condition_net_1.parameters(),
        *condition_net_2.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_t_max)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_window_protocol = "random80-20" in args.dataset_version
    reproduction_labels = {
        "RDDM (reproduced)",
        "RDDM-PPG (matched-protocol reproduction)",
    }
    resolved = vars(args).copy()
    resolved.update(
        {
            "run_id": run_id,
            "datasets": datasets,
            "sample_rate_hz": 128,
            "signal_length": signal_length,
            "target_channels": target_channels,
            "model_family": "RDDM",
            "comparison_role": (
                "independent_upstream_paper_reproduction"
                if args.reproduction_label in reproduction_labels
                else "conditional_modality_adaptation"
            ),
            "checkpoint_epochs": checkpoint_epochs(args.epochs, args.save_every),
            "model_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
            "heldout_role": (
                "random_window_validation_not_evaluated_during_training"
                if random_window_protocol
                else "upstream_test_not_evaluated_during_training"
                if args.heldout_split == "test"
                else "official_fold9_validation_not_evaluated_during_training"
            ),
        }
    )
    repository_root = Path(__file__).resolve().parent
    git_commit, git_dirty, git_status = _repository_state(repository_root)
    resolved.update({"git_commit": git_commit, "git_dirty": git_dirty})
    run_dir = Path(args.output_dir) / args.task / datasets[0] / run_id
    gpu_model = torch.cuda.get_device_name(device) if device.type == "cuda" else "unavailable"
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "heldout_evaluated_during_training": False,
        },
        "\n".join(
            [
                f"python={platform.python_version()}",
                f"numpy={np.__version__}",
                f"torch={torch.__version__}",
                f"cuda={torch.version.cuda}",
                f"device={device}",
                f"gpu={gpu_model}",
            ]
        ),
        f"commit={git_commit}\ndirty={git_dirty}\n{git_status}",
    )
    logger = WandbLogger(
        mode=args.wandb_mode,
        run_dir=run_dir,
        config={**resolved, "hostname": socket.gethostname(), "gpu_model": gpu_model},
        project=args.wandb_project,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        run_name=run_id,
    )
    print("resolved RDDM configuration:")
    for key in (
        "datasets", "dataset_version", "nT", "batch_size", "epochs", "save_every",
        "ddpm_loss_weight", "region_loss_weight", "seed", "device",
    ):
        print(f"  {key}={resolved[key]}")

    global_step = 0
    start_time = time.monotonic()
    try:
        for epoch in range(1, args.epochs + 1):
            rddm.train()
            condition_net_1.train()
            condition_net_2.train()
            epoch_values: dict[str, list[float]] = {
                "train/ddpm_loss_weighted": [],
                "train/region_loss_weighted": [],
                "train/total_loss": [],
            }
            pbar = tqdm(loader, desc=f"RDDM epoch {epoch}/{args.epochs}")
            for batch_index, (target, condition, region_mask) in enumerate(pbar):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                target = target.float().to(device)
                condition = condition.float().to(device)
                region_mask = region_mask.float().to(device)
                optimizer.zero_grad(set_to_none=True)
                conditions_1 = condition_net_1(
                    condition, drop_prob=args.condition_drop_probability
                )
                conditions_2 = condition_net_2(
                    condition, drop_prob=args.condition_drop_probability
                )
                ddpm_loss, region_loss = rddm(
                    x=target,
                    cond1=conditions_1,
                    cond2=conditions_2,
                    patch_labels=region_mask,
                )
                weighted_ddpm = args.ddpm_loss_weight * ddpm_loss
                weighted_region = args.region_loss_weight * region_loss
                loss = weighted_ddpm + weighted_region
                loss.backward()
                optimizer.step()
                metrics = {
                    "train/ddpm_loss_weighted": float(weighted_ddpm.detach().cpu()),
                    "train/region_loss_weighted": float(weighted_region.detach().cpu()),
                    "train/total_loss": float(loss.detach().cpu()),
                }
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("RDDM training metrics contain NaN or Inf")
                for key, value in metrics.items():
                    epoch_values[key].append(value)
                if global_step % args.log_interval_steps == 0:
                    logger.log({**metrics, "train/epoch": float(epoch)}, step=global_step)
                global_step += 1
                pbar.set_postfix(loss=f"{metrics['train/total_loss']:.4f}")
            scheduler.step()
            epoch_metrics = {
                key: float(np.mean(values)) for key, values in epoch_values.items()
            }
            epoch_metrics["train/learning_rate"] = float(optimizer.param_groups[0]["lr"])
            artifacts.append_metrics("epoch_metrics.csv", epoch, global_step, epoch_metrics)
            logger.log(epoch_metrics, step=global_step)
            print(
                f"epoch={epoch} total_loss={epoch_metrics['train/total_loss']:.6f} "
                f"lr={epoch_metrics['train/learning_rate']:.3e}"
            )
            if epoch % args.save_every == 0:
                checkpoint = {
                    "schema_version": 1,
                    "kind": "independent_rddm_reproduction",
                    "epoch": epoch,
                    "global_step": global_step,
                    "rddm_state": rddm.state_dict(),
                    "condition_1_state": condition_net_1.state_dict(),
                    "condition_2_state": condition_net_2.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "rng_states": capture_rng_states(),
                    "config": resolved,
                    "normalization": dict(train_set.normalization_metadata),
                    "provenance": {
                        "git_commit": git_commit,
                        "git_dirty": git_dirty,
                        "command": shlex.join(sys.argv),
                        "upstream_commit": args.upstream_commit,
                    },
                }
                checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.pt"
                temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary_path)
                os.replace(temporary_path, checkpoint_path)
                metadata = {
                    "file": checkpoint_path.name,
                    "epoch": epoch,
                    "global_step": global_step,
                }
                artifacts.update_checkpoint_manifest(f"epoch_{epoch}", metadata)
                artifacts.update_checkpoint_manifest("latest", metadata)

        summary = {
            "status": "completed",
            "final_epoch": args.epochs,
            "global_step": global_step,
            "final_train_total_loss": epoch_metrics["train/total_loss"],
            "training_duration_seconds": time.monotonic() - start_time,
            "heldout_evaluated_during_training": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(summary)
        logger.finish(summary)
    except BaseException as error:
        failure = {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "exception_type": type(error).__name__,
            "exception_message": str(error).splitlines()[0],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        artifacts.update_run_metadata(failure)
        logger.finish(failure, exit_code=1)
        raise
    return run_dir


if __name__ == "__main__":
    train(parse_args_with_config())
