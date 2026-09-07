"""Train the independent CAT-PPG reproduction on a frozen paired dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rcfm.baselines.cat_checkpoint import (
    CAT_CHECKPOINT_KIND,
    load_cat_checkpoint,
    save_cat_checkpoint,
)
from src.rcfm.baselines.cat_data import (
    CPSC2018_SPLIT_HASH,
    MIMIC_DATASET_VERSION,
    MIMIC_SPLIT_HASH,
    MMECG_SPLIT_HASH,
    OTHER_11_LEADS,
    PTBXL_SPLIT_HASH,
    WESAD_SPLIT_HASH,
    load_cat_datasets,
)
from src.rcfm.baselines.catransformer import CATECGAdapter, CATLoss, CATransformer
from src.rcfm.checkpoint import capture_rng_states, restore_rng_states
from src.rcfm.experiment import RunArtifacts, WandbLogger
from src.rcfm.runtime import exception_summary, gradients_are_finite


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=os.environ.get("RCFM_DATA_ROOT"))
    parser.add_argument("--output_dir", default=os.environ.get("RCFM_RUNS_ROOT"))
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--task", default="ppg2ecg")
    parser.add_argument("--datasets", default="MIMIC-AFib")
    parser.add_argument("--dataset_version", default=None)
    parser.add_argument("--split_hash", default=None)
    parser.add_argument("--normalization_id", default=None)
    parser.add_argument("--alignment_id", default=None)
    parser.add_argument("--condition_lead", default=None)
    parser.add_argument("--target_lead", default=None)
    parser.add_argument("--condition_lead_index", type=int, default=None)
    parser.add_argument("--target_lead_indices", type=int, nargs="+", default=None)
    parser.add_argument("--heldout_split", choices=("val", "test"), default="test")
    parser.add_argument("--output_channels", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--expected_train_windows", type=int, default=8400)
    parser.add_argument("--expected_test_windows", type=int, default=1800)
    parser.add_argument("--heldout_role", default="upstream_test_final_only")
    parser.add_argument("--reproduction_label", default="CAT-PPG (reproduced)")
    parser.add_argument("--cycle_source", default="source_ppg_fft_only")
    parser.add_argument("--ecg_adapter_version", default="not_applicable")
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument("--cat_layers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--patch_width", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--log_interval_steps", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_train_records", type=int, default=None)
    parser.add_argument("--max_test_records", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb_project", default="RCFM")
    parser.add_argument("--wandb_group", default="mimic-afib-cat-ppg-paper-reproduction")
    parser.add_argument("--wandb_job_type", default="train")
    return parser


def parse_args_with_config(argv: list[str] | None = None) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", required=True)
    known, _ = preliminary.parse_known_args(argv)
    parser = build_argparser()
    defaults = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(defaults, dict):
        raise ValueError("CAT config must contain a JSON object")
    fields = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - fields)
    if unknown:
        raise ValueError("CAT config has unknown keys: " + ", ".join(unknown))
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    validate_config(args)
    return args


def validate_config(args: argparse.Namespace) -> None:
    common = {"window_size": 4, "sampling_rate": 128, "nfe": 1,
              "cat_layers": 2, "top_k": 2, "encoder_layers": 4,
              "ecg_adapter_version": "not_applicable"}
    protocols = {
        ("ppg2ecg", "MIMIC-AFib"): {
            "dataset_version": MIMIC_DATASET_VERSION, "split_hash": MIMIC_SPLIT_HASH,
            "normalization_id": "rddm_window_minmax_neg1_1_v1",
            "alignment_id": "paired_array_row_rddm_contract_zero_ppg_qc_v1",
            "expected_train_windows": 8400, "expected_test_windows": 1800,
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "reproduction_label": "CAT-PPG (reproduced)",
            "cycle_source": "source_ppg_fft_only", "output_channels": 1,
        },
        ("ppg2ecg", "WESAD"): {
            "dataset_version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
            "split_hash": WESAD_SPLIT_HASH, "normalization_id": "window_minmax_neg1_1_v1",
            "alignment_id": "native_common_start_same_window_no_delay_correction_subject_fold1_v1",
            "expected_train_windows": 17494, "expected_test_windows": 4213,
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "reproduction_label": "CAT-PPG (reproduced)",
            "cycle_source": "source_bvp_fft_only", "output_channels": 1,
        },
        ("rcg2ecg", "mmECG"): {
            "dataset_version": "mmecg-public-20221108-subject-split-window-minmax-v1",
            "split_hash": MMECG_SPLIT_HASH, "normalization_id": "window_minmax_neg1_1_v1",
            "alignment_id": "same_record_same_window_no_additional_phase_correction_subject_split_v1",
            "expected_train_windows": 9590, "expected_test_windows": 2877,
            "heldout_split": "test", "heldout_role": "upstream_test_final_only",
            "reproduction_label": "CAT-RCG (adapted)",
            "cycle_source": "source_rcg_fft_only", "output_channels": 1,
        },
        ("ecg2ecg", "PTBXL"): {
            "dataset_version": "ptbxl-1.0.1-official-folds-record-minmax-neg1-1-v1",
            "split_hash": PTBXL_SPLIT_HASH, "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "ptbxl_official_folds_first4s_same_record_lead_II_to_other11_v1",
            "expected_train_windows": 17440, "expected_test_windows": 2193,
            "heldout_split": "val", "heldout_role": "validation_endpoint_only",
            "reproduction_label": "CAT-ECG (adapted)",
            "cycle_source": "source_ecg_lead_II_fft_only", "output_channels": 11,
            "ecg_adapter_version": "shared_first_lead_specific_second_cat_v2",
        },
        ("ecg2ecg", "CPSC2018"): {
            "dataset_version": "cpsc2018-source-derived-all12lead-qc-v3-record-minmax-neg1-1",
            "split_hash": CPSC2018_SPLIT_HASH, "normalization_id": "record_minmax_neg1_1_v1",
            "alignment_id": "same_record_simultaneous_channels_first_4s_lead_II_to_other_11_v3",
            "expected_train_windows": 5487, "expected_test_windows": 686,
            "heldout_split": "val", "heldout_role": "validation_endpoint_only",
            "reproduction_label": "CAT-ECG (adapted)",
            "cycle_source": "source_ecg_lead_II_fft_only", "output_channels": 11,
            "ecg_adapter_version": "shared_first_lead_specific_second_cat_v2",
        },
    }
    protocol = protocols.get((args.task, args.datasets))
    if protocol is None:
        raise ValueError("unsupported CAT dataset/task protocol")
    expected = {**common, **protocol}
    changed = [name for name, value in expected.items() if getattr(args, name) != value]
    if changed:
        raise ValueError("CAT frozen protocol fields changed: " + ", ".join(changed))
    if args.task == "ecg2ecg":
        if args.condition_lead_index != 1 or args.target_lead_indices != OTHER_11_LEADS:
            raise ValueError("CAT-ECG requires lead II to the other 11 leads")
    elif args.condition_lead_index is not None or args.target_lead_indices is not None:
        raise ValueError("single-source CAT must not declare ECG lead indices")
    positive = (
        "patch_width", "d_model", "n_heads", "ff_dim", "kl_temperature", "epochs",
        "batch_size", "learning_rate", "save_every", "log_interval_steps", "grad_clip",
    )
    if any(float(getattr(args, name)) <= 0 for name in positive):
        raise ValueError("CAT numeric architecture/training fields must be positive")
    if args.d_model % args.n_heads or not 0 <= args.dropout < 1 or args.kl_weight < 0:
        raise ValueError("invalid CAT attention, dropout, or KL configuration")
    if args.epochs % args.save_every:
        raise ValueError("CAT epochs must be divisible by save_every")
    for name in ("max_train_records", "max_test_records", "max_batches"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")


def _set_deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_state() -> tuple[str, bool, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
                            capture_output=True, text=True).stdout
    return commit, bool(status.strip()), status


def _model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs = dict(
        input_length=args.window_size * args.sampling_rate,
        cat_layers=args.cat_layers,
        top_k=args.top_k,
        patch_width=args.patch_width,
        d_model=args.d_model,
        n_heads=args.n_heads,
        encoder_layers=args.encoder_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    if args.reproduction_label == "CAT-ECG (adapted)":
        return CATECGAdapter(output_channels=args.output_channels, **kwargs)
    return CATransformer(output_channels=1, **kwargs)


RESUME_MATCH_FIELDS = (
    "task", "datasets", "dataset_version", "split_hash", "normalization_id",
    "alignment_id", "window_size", "sampling_rate", "expected_train_windows",
    "expected_test_windows", "heldout_role", "reproduction_label", "cycle_source",
    "condition_lead", "target_lead", "condition_lead_index", "target_lead_indices",
    "heldout_split", "output_channels",
    "nfe", "cat_layers", "top_k", "patch_width", "d_model", "n_heads",
    "encoder_layers", "ff_dim", "dropout", "kl_weight", "kl_temperature",
    "batch_size", "learning_rate", "weight_decay", "grad_clip", "amp", "seed",
    "num_workers", "max_train_records", "max_batches",
)


def _resolved_config(args: argparse.Namespace) -> dict[str, object]:
    local_only = {"config", "data_root", "output_dir", "run_id", "resume"}
    return {key: value for key, value in vars(args).items() if key not in local_only}


def _validate_resume_contract(
    args: argparse.Namespace, checkpoint: dict[str, object]
) -> None:
    saved = checkpoint["config"]
    mismatched = [
        field for field in RESUME_MATCH_FIELDS if saved.get(field) != getattr(args, field)
    ]
    if (
        args.reproduction_label == "CAT-ECG (adapted)"
        and saved.get("ecg_adapter_version") != args.ecg_adapter_version
    ):
        mismatched.append("ecg_adapter_version")
    if mismatched:
        raise ValueError("CAT resume checkpoint differs on: " + ", ".join(mismatched))
    if int(checkpoint["epoch"]) >= args.epochs:
        raise ValueError("CAT resume checkpoint already reached the configured final epoch")


def train(args: argparse.Namespace) -> Path:
    validate_config(args)
    if not args.data_root or not args.output_dir:
        raise ValueError("CAT training requires --data_root and --output_dir")
    _set_deterministic(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    train_set, test_set, normalization = load_cat_datasets(args)
    expected_train = min(args.expected_train_windows, args.max_train_records or args.expected_train_windows)
    expected_test = min(args.expected_test_windows, args.max_test_records or args.expected_test_windows)
    if len(train_set) != expected_train or len(test_set) != expected_test:
        raise ValueError("CAT data sizes disagree with the frozen protocol")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    model = _model(args).to(device)
    criterion = CATLoss(args.kl_weight, args.kl_temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    resume_checkpoint = None
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        resume_checkpoint = load_cat_checkpoint(args.resume.resolve(), map_location="cpu")
        _validate_resume_contract(args, resume_checkpoint)
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scaler.load_state_dict(resume_checkpoint["scaler_state"])
        generator.set_state(resume_checkpoint["data_loader_generator_state"].cpu())
        start_epoch = int(resume_checkpoint["epoch"])
        global_step = int(resume_checkpoint["global_step"])
    default_suffix = f"cat_ppg_resume_e{start_epoch}" if resume_checkpoint else "cat_ppg"
    run_id = args.run_id or datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ_{default_suffix}")
    run_dir = Path(args.output_dir) / args.task / args.datasets / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing CAT run: {run_dir}")
    commit, dirty, status = _git_state()
    resolved = _resolved_config(args)
    started = datetime.now(timezone.utc)
    artifacts = RunArtifacts(
        run_dir,
        resolved,
        {"schema_version": 1, "run_id": run_id, "status": "running",
         "started_at_utc": started.isoformat(), "heldout_evaluated_during_training": False,
         "resumed_from": str(args.resume.resolve()) if args.resume else None,
         "start_epoch": start_epoch, "start_global_step": global_step},
        f"python={platform.python_version()}\ntorch={torch.__version__}\nnumpy={np.__version__}\nhost={socket.gethostname()}\ndevice={device}\n",
        f"commit={commit}\ndirty={dirty}\n{status}",
    )
    logger = WandbLogger(args.wandb_mode, run_dir, resolved, args.wandb_project,
                         args.wandb_group, args.wandb_job_type, run_id)
    output_spec = {"channels": args.output_channels, "length": 512,
                   "sampling_rate_hz": 128, "target_lead": args.target_lead}

    def checkpoint(epoch: int, label: str) -> None:
        payload = {
            "schema_version": 1, "kind": CAT_CHECKPOINT_KIND, "epoch": epoch,
            "global_step": global_step, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
            "config": resolved, "normalization": normalization, "output_spec": output_spec,
            "rng_states": capture_rng_states(),
            "data_loader_generator_state": generator.get_state(),
            "provenance": {"git_commit": commit, "git_dirty": dirty,
                           "command": shlex.join(sys.argv),
                           "paper_doi": "10.1109/JBHI.2024.3482853",
                           "implementation_status": (
                               "independent_paper_based_adaptation"
                               if args.reproduction_label in {"CAT-ECG (adapted)", "CAT-RCG (adapted)"}
                               else "independent_paper_based_reproduction"
                           )},
        }
        path = run_dir / f"checkpoint_{label}.pt"
        save_cat_checkpoint(payload, path)
        artifacts.update_checkpoint_manifest(
            label, {"file": path.name, "epoch": epoch, "global_step": global_step}
        )

    try:
        if resume_checkpoint is not None:
            restore_rng_states(resume_checkpoint["rng_states"])
            del resume_checkpoint
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            totals: dict[str, float] = {"total_loss": 0.0, "mse_loss": 0.0, "kl_loss": 0.0,
                                        "cycle_fallback_rate": 0.0, "reconstruction_fallback_rate": 0.0,
                                        "amp_overflow_rate": 0.0}
            batches = 0
            progress = tqdm(loader, desc=f"CAT epoch {epoch}/{args.epochs}")
            for batch_index, (target, source) in enumerate(progress):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                target = target.to(device, non_blocking=True); source = source.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction, diagnostics = model(source, return_diagnostics=True)
                    loss, components = criterion(prediction, target)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("CAT forward loss contains NaN or Inf")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                finite_gradients = gradients_are_finite(model.parameters())
                amp_overflow = bool(amp_enabled and not finite_gradients)
                if not finite_gradients and not amp_overflow:
                    raise FloatingPointError("CAT gradients contain NaN or Inf without AMP")
                gradient_norm = (
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if finite_gradients
                    else None
                )
                scaler.step(optimizer); scaler.update()
                global_step += 1; batches += 1
                step_metrics = {name: float(value) for name, value in components.items()}
                cycle_flags = torch.stack([value.float() for key, value in diagnostics.items() if key.endswith("cycle_fallback")])
                reconstruction_flags = torch.stack([value.float() for key, value in diagnostics.items() if key.endswith("reconstruction_fallback")])
                step_metrics.update({
                    "cycle_fallback_rate": float(cycle_flags.mean()),
                    "reconstruction_fallback_rate": float(reconstruction_flags.mean()),
                    "amp_overflow_rate": float(amp_overflow),
                })
                if gradient_norm is not None:
                    step_metrics["gradient_norm"] = float(gradient_norm)
                if not all(np.isfinite(value) for value in step_metrics.values()):
                    raise FloatingPointError("CAT training metrics contain NaN or Inf")
                for name in totals:
                    totals[name] += step_metrics[name]
                if global_step % args.log_interval_steps == 0:
                    logger.log({f"train/{name}": value for name, value in step_metrics.items()}, global_step)
                progress.set_postfix(loss=f"{step_metrics['total_loss']:.4f}")
            if batches == 0:
                raise RuntimeError("CAT training epoch processed no batches")
            epoch_metrics = {name: value / batches for name, value in totals.items()}
            artifacts.append_metrics("epoch_metrics.csv", epoch, global_step, epoch_metrics)
            logger.log({f"epoch/{name}": value for name, value in epoch_metrics.items()}, global_step)
            checkpoint(epoch, "latest")
            if epoch % args.save_every == 0 or epoch == args.epochs:
                checkpoint(epoch, f"epoch_{epoch}")
        finished = datetime.now(timezone.utc)
        summary = {"final_epoch": args.epochs, "global_step": global_step,
                   "training_duration_seconds": (finished - started).total_seconds(),
                   "heldout_evaluated_during_training": False}
        artifacts.update_run_metadata({**summary, "status": "completed", "finished_at_utc": finished.isoformat()})
        logger.finish(summary)
    except BaseException as error:
        artifacts.update_run_metadata({
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception_type": type(error).__name__,
            "exception_message": exception_summary(error),
        })
        logger.finish(exit_code=1)
        raise
    return run_dir


if __name__ == "__main__":
    output = train(parse_args_with_config())
    print(f"CAT training complete: {output}")
