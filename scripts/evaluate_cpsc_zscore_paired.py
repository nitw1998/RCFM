"""Deterministically compare matched CPSC2018 z-score CFM and RCFM checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcfm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import ConditionNet, DiffusionUNetCrossAttention
from rcfm import RegionAwareConditionalFlowMatching
from scripts.evaluate_clinical import PARAMETERS, evaluate as evaluate_clinical
from src.rcfm.checkpoint import load_checkpoint
from src.rcfm.metrics.bland_altman import bland_altman, paired_correlation
from src.rcfm.metrics.waveform import waveform_frechet_distance
from train_rcfm import build_datasets


MATCHED_CONFIG_FIELDS = (
    "task",
    "datasets",
    "dataset_version",
    "split_hash",
    "normalization_id",
    "condition_unit",
    "target_unit",
    "alignment_id",
    "condition_lead",
    "target_lead",
    "condition_lead_index",
    "target_lead_index",
    "window_size",
    "attention_heads",
    "flow_matcher",
    "sigma",
    "use_minibatch_ot",
)

AMPLITUDE_PARAMETERS = {
    "p_amplitude",
    "r_amplitude",
    "t_amplitude",
    "st_deviation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_contract(path: Path) -> dict[str, object]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    contract = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "config": dict(checkpoint["config"]),
        "normalization": dict(checkpoint["normalization"]),
        "output_spec": dict(checkpoint["output_spec"]),
        "provenance": dict(checkpoint["provenance"]),
        "best_metrics": dict(checkpoint["best_metrics"]),
    }
    del checkpoint
    gc.collect()
    return contract


def _validate_contracts(
    cfm: Mapping[str, object],
    rcfm: Mapping[str, object],
) -> None:
    if cfm["kind"] != "canonical_multistep_cfm":
        raise ValueError("--cfm_checkpoint must be a canonical_multistep_cfm checkpoint")
    if rcfm["kind"] != "canonical_multistep_rcfm":
        raise ValueError("--rcfm_checkpoint must be a canonical_multistep_rcfm checkpoint")
    cfm_config = cfm["config"]
    rcfm_config = rcfm["config"]
    mismatched = [
        field
        for field in MATCHED_CONFIG_FIELDS
        if cfm_config.get(field) != rcfm_config.get(field)
    ]
    if mismatched:
        raise ValueError("paired checkpoints disagree on: " + ", ".join(mismatched))
    if cfm["normalization"] != rcfm["normalization"]:
        raise ValueError("paired checkpoints contain different normalization metadata")
    if cfm["output_spec"] != rcfm["output_spec"]:
        raise ValueError("paired checkpoints contain different output specifications")
    if int(rcfm["output_spec"].get("channels", 0)) != 1:
        raise ValueError(
            "this historical clinical evaluator supports only single-target-lead checkpoints; "
            "multi-lead ECG parameters must be evaluated separately per lead"
        )
    if cfm_config.get("normalization_id") != "record_zscore_v1":
        raise ValueError("this entry is restricted to the record-zscore comparison")
    if cfm_config.get("task") != "ecg2ecg" or cfm_config.get("datasets") != ["CPSC2018"]:
        raise ValueError("this entry is restricted to the CPSC2018 ECG-to-ECG experiment")
    if float(cfm_config.get("region_weight", -1)) != 0.0:
        raise ValueError("CFM checkpoint must have region_weight=0")
    if float(rcfm_config.get("region_weight", 0)) <= 0.0:
        raise ValueError("RCFM checkpoint must have a positive region_weight")


def _fixed_noise(
    record_count: int,
    channels: int,
    length: int,
    seed: int,
) -> np.ndarray:
    if record_count <= 0 or channels <= 0 or length <= 0:
        raise ValueError("fixed-noise dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        (record_count, channels, length),
        generator=generator,
        dtype=torch.float32,
    ).numpy()


def _set_deterministic(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def _generate(
    checkpoint_path: Path,
    expected_kind: str,
    conditions: np.ndarray,
    initial_noise: np.ndarray,
    batch_size: int,
    steps: int,
    device: torch.device,
    deterministic_seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint["kind"] != expected_kind:
        raise ValueError(f"checkpoint kind changed while loading {checkpoint_path.name}")
    config = checkpoint["config"]
    output_spec = checkpoint["output_spec"]
    signal_length = int(output_spec["length"])
    output_channels = int(output_spec["channels"])
    if initial_noise.shape != (len(conditions), output_channels, signal_length):
        raise ValueError("saved initial noise disagrees with checkpoint output shape")

    _set_deterministic(deterministic_seed, device)
    condition_net = ConditionNet().to(device)
    flow_network = DiffusionUNetCrossAttention(
        signal_length,
        output_channels,
        device=str(device),
        num_heads=int(config["attention_heads"]),
    ).to(device)
    model = RegionAwareConditionalFlowMatching(
        flow_model=flow_network,
        flow_matcher_type=str(config["flow_matcher"]),
        sigma=float(config["sigma"]),
        region_weight=float(config["region_weight"]),
        use_minibatch_ot=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    condition_net.load_state_dict(checkpoint["condition_state"], strict=True)
    model.eval()
    condition_net.eval()
    metadata = {
        "kind": checkpoint["kind"],
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "best_metrics": dict(checkpoint["best_metrics"]),
    }
    del checkpoint
    gc.collect()

    predictions = np.empty_like(initial_noise)
    for start in range(0, len(conditions), batch_size):
        stop = min(start + batch_size, len(conditions))
        condition_batch = torch.from_numpy(conditions[start:stop]).to(device=device)
        noise_batch = torch.from_numpy(initial_noise[start:stop]).to(device=device)
        encoded = condition_net(condition_batch)
        prediction = model.sample(
            conditions=encoded,
            shape=tuple(noise_batch.shape),
            steps=steps,
            device=device,
            initial_noise=noise_batch,
        )
        predictions[start:stop] = prediction.cpu().numpy()
        print(f"{expected_kind}: generated {stop}/{len(conditions)}", flush=True)

    del model, flow_network, condition_net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("generated predictions contain NaN or Inf")
    return predictions, metadata


def _pearson_rows(reference: np.ndarray, generated: np.ndarray) -> np.ndarray:
    reference = reference.reshape(len(reference), -1).astype(np.float64)
    generated = generated.reshape(len(generated), -1).astype(np.float64)
    reference = reference - reference.mean(axis=1, keepdims=True)
    generated = generated - generated.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(generated, axis=1)
    correlations = np.full(len(reference), np.nan, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = np.sum(reference[valid] * generated[valid], axis=1) / denominator[valid]
    return correlations


def _waveform_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError("waveform metrics require matching (records, channels, samples) arrays")
    error = generated.astype(np.float64) - reference.astype(np.float64)
    per_record = {
        "rmse": np.sqrt(np.mean(error**2, axis=(1, 2))),
        "mae": np.mean(np.abs(error), axis=(1, 2)),
        "bias": np.mean(error, axis=(1, 2)),
        "pearson_r": _pearson_rows(reference, generated),
    }
    flat_reference = reference.reshape(-1)
    flat_generated = generated.reshape(-1)
    correlation = paired_correlation(flat_reference, flat_generated)
    correlation["p_value"] = None
    correlation["inference_status"] = "blocked_autocorrelated_time_samples"
    agreement = bland_altman(flat_reference, flat_generated)
    agreement.pop("pair_means")
    agreement.pop("differences")
    if len(reference) >= 2:
        waveform_fd = waveform_frechet_distance(reference, generated)
        waveform_fd_status = "ok"
    else:
        waveform_fd = None
        waveform_fd_status = "insufficient_records"
    summary = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "waveform_fd": waveform_fd,
        "waveform_fd_status": waveform_fd_status,
        "pointwise_correlation_descriptive_only": correlation,
        "pointwise_bland_altman_descriptive_only": agreement,
        "per_record_pearson": {
            "usable_records": int(np.isfinite(per_record["pearson_r"]).sum()),
            "mean": float(np.nanmean(per_record["pearson_r"])),
            "median": float(np.nanmedian(per_record["pearson_r"])),
        },
    }
    return summary, per_record


def _save_waveform_outputs(
    model_name: str,
    record_ids: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    output_dir: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    summary, per_record = _waveform_metrics(reference, generated)
    _json(output_dir / f"{model_name}_waveform_metrics.json", summary)
    with (output_dir / f"{model_name}_per_record_waveform_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_id", "rmse", "mae", "bias", "pearson_r"])
        for index, record_id in enumerate(record_ids):
            writer.writerow(
                [
                    str(record_id),
                    per_record["rmse"][index],
                    per_record["mae"][index],
                    per_record["bias"][index],
                    per_record["pearson_r"][index],
                ]
            )
    _plot_waveform_agreement(model_name, reference, generated, output_dir)
    return summary, per_record


def _plot_waveform_agreement(
    model_name: str,
    reference: np.ndarray,
    generated: np.ndarray,
    output_dir: Path,
    maximum_points: int = 20_000,
) -> None:
    reference_flat = reference.reshape(-1).astype(np.float64)
    generated_flat = generated.reshape(-1).astype(np.float64)
    if len(reference_flat) > maximum_points:
        indices = np.linspace(0, len(reference_flat) - 1, maximum_points, dtype=np.int64)
        reference_flat = reference_flat[indices]
        generated_flat = generated_flat[indices]
    pair_mean = (reference_flat + generated_flat) / 2.0
    difference = generated_flat - reference_flat
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(reference_flat, generated_flat, s=3, alpha=0.18, rasterized=True)
    lower = min(float(reference_flat.min()), float(generated_flat.min()))
    upper = max(float(reference_flat.max()), float(generated_flat.max()))
    axes[0].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[0].set(xlabel="Real normalized ECG", ylabel="Generated normalized ECG", title="Correlation")
    axes[1].scatter(pair_mean, difference, s=3, alpha=0.18, rasterized=True)
    for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        axes[1].axhline(value, color="black", linestyle=style, linewidth=1)
    axes[1].set(
        xlabel="Pair mean (normalized)",
        ylabel="Generated - real (normalized)",
        title="Bland-Altman (descriptive)",
    )
    fig.savefig(output_dir / f"{model_name}_waveform_agreement.png", dpi=220)
    plt.close(fig)


def _p_wave_applicability(dataset_root: Path, count: int) -> tuple[np.ndarray, dict[str, object]]:
    labels = np.load(dataset_root / "labels_test.npy", allow_pickle=False)
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    label_names = manifest.get("label_names", {})
    af_keys = [int(key) for key, value in label_names.items() if value == "atrial_fibrillation"]
    if len(af_keys) != 1:
        raise ValueError("dataset manifest must identify exactly one atrial_fibrillation label")
    af_column = af_keys[0] - 1
    if labels.ndim != 2 or labels.shape[1] <= af_column or len(labels) < count:
        raise ValueError("test labels do not align with requested CPSC records")
    applicable = labels[:count, af_column] == 0
    return applicable, {
        "policy": "PR and P-wave amplitude excluded for records labelled atrial_fibrillation",
        "af_label_column_zero_based": af_column,
        "applicable_records": int(applicable.sum()),
        "not_applicable_af_records": int((~applicable).sum()),
    }


def _clinical_namespace(
    real_path: Path,
    generated_path: Path,
    record_ids_path: Path,
    p_wave_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    target_lead: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        real=real_path,
        generated=generated_path,
        subject_ids=record_ids_path,
        p_wave_applicability=p_wave_path,
        output=output_path,
        sampling_rate=args.sampling_rate,
        lead=target_lead,
        analysis_unit="record",
        normalization_id="record_zscore_v1",
        unit="normalized",
        inverse_transformed=False,
        allow_normalized_amplitudes=True,
        continuous=False,
        minimum_hrv_seconds=args.minimum_hrv_seconds,
        qtc_formula=args.qtc_formula,
        st_offset_ms=args.st_offset_ms,
        delineation_method=args.delineation_method,
        clean_method=args.clean_method,
    )


def _write_clinical_outputs(
    model_name: str,
    result: dict[str, object],
    output_dir: Path,
) -> None:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    _json(model_dir / "clinical_metrics.json", result)
    rows: list[list[object]] = []
    for parameter in PARAMETERS:
        agreement = result["agreement"][parameter]
        if agreement["status"] != "ok":
            rows.append([parameter, agreement["status"], agreement.get("n", 0)] + [""] * 14)
            continue
        real_values, generated_values = _parameter_pairs(result, parameter)
        correlation = agreement["correlation"]
        bland = agreement["bland_altman"]
        units = "normalized_record_zscore" if parameter in AMPLITUDE_PARAMETERS else "ms"
        claim = "exploratory_not_physical" if parameter in AMPLITUDE_PARAMETERS else "interval_metric"
        error = generated_values - real_values
        rows.append(
            [
                parameter,
                "ok",
                bland["n"],
                float(np.mean(real_values)),
                float(np.std(real_values, ddof=1)),
                float(np.mean(generated_values)),
                float(np.std(generated_values, ddof=1)),
                float(np.mean(np.abs(error))),
                float(np.sqrt(np.mean(error**2))),
                correlation["r"],
                correlation["p_value"],
                bland["bias"],
                bland["difference_sd"],
                bland["lower_limit"],
                bland["upper_limit"],
                units,
                claim,
            ]
        )
        _plot_parameter_agreement(model_name, parameter, result, model_dir)
    with (model_dir / "agreement_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "parameter",
                "status",
                "n_record_pairs",
                "real_mean",
                "real_sd",
                "generated_mean",
                "generated_sd",
                "mean_absolute_error",
                "root_mean_squared_error",
                "pearson_r",
                "pearson_p_value",
                "bland_altman_bias",
                "difference_sd",
                "lower_limit",
                "upper_limit",
                "unit",
                "claim_status",
            ]
        )
        writer.writerows(rows)


def _parameter_pairs(
    result: Mapping[str, object],
    parameter: str,
) -> tuple[np.ndarray, np.ndarray]:
    real_units = result["unit_summaries"]["real"]
    generated_units = result["unit_summaries"]["generated"]
    common = sorted(set(real_units) & set(generated_units))
    pairs = [
        (real_units[unit].get(parameter), generated_units[unit].get(parameter))
        for unit in common
    ]
    pairs = [(real, generated) for real, generated in pairs if real is not None and generated is not None]
    if not pairs:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    real, generated = zip(*pairs)
    return np.asarray(real, dtype=np.float64), np.asarray(generated, dtype=np.float64)


def _plot_parameter_agreement(
    model_name: str,
    parameter: str,
    result: Mapping[str, object],
    output_dir: Path,
) -> None:
    real, generated = _parameter_pairs(result, parameter)
    if len(real) < 2:
        return
    pair_mean = (real + generated) / 2.0
    difference = generated - real
    bias = float(np.mean(difference))
    sd = float(np.std(difference, ddof=1))
    unit = "normalized z-score units" if parameter in AMPLITUDE_PARAMETERS else "ms"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(real, generated, s=14, alpha=0.6)
    lower = min(float(real.min()), float(generated.min()))
    upper = max(float(real.max()), float(generated.max()))
    axes[0].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[0].set(xlabel=f"Real ({unit})", ylabel=f"Generated ({unit})", title="Correlation")
    axes[1].scatter(pair_mean, difference, s=14, alpha=0.6)
    for value, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        axes[1].axhline(value, color="black", linestyle=style, linewidth=1)
    axes[1].set(
        xlabel=f"Pair mean ({unit})",
        ylabel=f"Generated - real ({unit})",
        title="Bland-Altman",
    )
    fig.suptitle(f"{model_name.upper()} {parameter}")
    fig.savefig(output_dir / f"{parameter}_agreement.png", dpi=220)
    plt.close(fig)


def _paired_model_comparison(
    cfm: Mapping[str, np.ndarray],
    rcfm: Mapping[str, np.ndarray],
) -> dict[str, object]:
    output: dict[str, object] = {
        "difference_definition": "CFM_minus_RCFM; positive favors RCFM for error metrics",
        "inference_status": "single_seed_descriptive_only",
    }
    for metric in ("rmse", "mae"):
        difference = np.asarray(cfm[metric]) - np.asarray(rcfm[metric])
        output[metric] = {
            "record_count": int(len(difference)),
            "mean_difference": float(np.mean(difference)),
            "median_difference": float(np.median(difference)),
            "fraction_favoring_rcfm": float(np.mean(difference > 0)),
        }
    return output


def _clinical_model_comparison(
    cfm: Mapping[str, object],
    rcfm: Mapping[str, object],
) -> dict[str, object]:
    output: dict[str, object] = {
        "analysis_unit": "record",
        "subset_policy": "same record has real, CFM, and RCFM values for the parameter",
        "inference_status": "single_seed_record_level_descriptive_only",
        "parameters": {},
    }
    cfm_real = cfm["unit_summaries"]["real"]
    cfm_generated = cfm["unit_summaries"]["generated"]
    rcfm_real = rcfm["unit_summaries"]["real"]
    rcfm_generated = rcfm["unit_summaries"]["generated"]
    common_units = sorted(
        set(cfm_real) & set(cfm_generated) & set(rcfm_real) & set(rcfm_generated)
    )
    for parameter in PARAMETERS:
        rows = []
        for unit in common_units:
            values = (
                cfm_real[unit].get(parameter),
                cfm_generated[unit].get(parameter),
                rcfm_real[unit].get(parameter),
                rcfm_generated[unit].get(parameter),
            )
            if all(value is not None and np.isfinite(value) for value in values):
                rows.append(values)
        if len(rows) < 2:
            output["parameters"][parameter] = {
                "status": "insufficient_common_record_pairs",
                "n": len(rows),
            }
            continue
        values = np.asarray(rows, dtype=np.float64)
        real = (values[:, 0] + values[:, 2]) / 2.0
        if not np.allclose(values[:, 0], values[:, 2], atol=1e-10, rtol=1e-10):
            raise ValueError("real clinical summaries changed between paired model evaluations")
        cfm_generated_values = values[:, 1]
        rcfm_generated_values = values[:, 3]
        cfm_error = cfm_generated_values - real
        rcfm_error = rcfm_generated_values - real
        cfm_abs = np.abs(cfm_error)
        rcfm_abs = np.abs(rcfm_error)
        output["parameters"][parameter] = {
            "status": "ok",
            "n": len(rows),
            "unit": (
                "normalized_record_zscore"
                if parameter in AMPLITUDE_PARAMETERS
                else "ms"
            ),
            "claim_status": (
                "exploratory_not_physical"
                if parameter in AMPLITUDE_PARAMETERS
                else "interval_metric"
            ),
            "real_mean": float(np.mean(real)),
            "cfm_generated_mean": float(np.mean(cfm_generated_values)),
            "rcfm_generated_mean": float(np.mean(rcfm_generated_values)),
            "cfm_mean_absolute_error": float(np.mean(cfm_abs)),
            "rcfm_mean_absolute_error": float(np.mean(rcfm_abs)),
            "cfm_root_mean_squared_error": float(np.sqrt(np.mean(cfm_error**2))),
            "rcfm_root_mean_squared_error": float(np.sqrt(np.mean(rcfm_error**2))),
            "fraction_absolute_error_favoring_rcfm": float(np.mean(rcfm_abs < cfm_abs)),
            "ties": int(np.sum(rcfm_abs == cfm_abs)),
        }
    return output


def _write_clinical_model_comparison(
    comparison: Mapping[str, object],
    clinical_dir: Path,
) -> None:
    _json(clinical_dir / "paired_clinical_model_comparison.json", comparison)
    fields = (
        "parameter",
        "status",
        "n",
        "unit",
        "claim_status",
        "real_mean",
        "cfm_generated_mean",
        "rcfm_generated_mean",
        "cfm_mean_absolute_error",
        "rcfm_mean_absolute_error",
        "cfm_root_mean_squared_error",
        "rcfm_root_mean_squared_error",
        "fraction_absolute_error_favoring_rcfm",
        "ties",
    )
    with (clinical_dir / "paired_clinical_model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for parameter in PARAMETERS:
            row = {"parameter": parameter, **comparison["parameters"][parameter]}
            writer.writerow({field: row.get(field, "") for field in fields})


def evaluate(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    cfm_contract = _checkpoint_contract(args.cfm_checkpoint)
    rcfm_contract = _checkpoint_contract(args.rcfm_checkpoint)
    _validate_contracts(cfm_contract, rcfm_contract)
    config = rcfm_contract["config"]
    output_spec = rcfm_contract["output_spec"]
    _, test_set = build_datasets(
        config["task"],
        config["datasets"],
        str(args.data_root),
        int(config["window_size"]),
        normalization_metadata=rcfm_contract["normalization"],
        normalization_id=config["normalization_id"],
        condition_lead_index=int(config["condition_lead_index"]),
        target_lead_index=int(config["target_lead_index"]),
        load_train=False,
        heldout_split="test",
    )
    total_records = len(test_set)
    if args.expected_records is not None and total_records != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} test records but loader returned {total_records}"
        )
    count = total_records if args.max_records is None else min(args.max_records, total_records)
    targets = np.asarray(test_set.target_ecg[:count, None, :], dtype=np.float32)
    conditions = np.asarray(test_set.condition_signal[:count, None, :], dtype=np.float32)
    record_ids = np.asarray(test_set.record_ids[:count])
    if len(np.unique(record_ids)) != count:
        raise ValueError("test record IDs must be unique")

    record_ids_path = output_dir / "record_ids.npy"
    targets_path = output_dir / "targets_normalized.npy"
    conditions_path = output_dir / "conditions_normalized.npy"
    np.save(record_ids_path, record_ids, allow_pickle=False)
    np.save(targets_path, targets, allow_pickle=False)
    np.save(conditions_path, conditions, allow_pickle=False)

    dataset_root = args.data_root / "CPSC2018"
    p_wave_applicable, p_wave_policy = _p_wave_applicability(dataset_root, count)
    p_wave_path = output_dir / "p_wave_applicability.npy"
    np.save(p_wave_path, p_wave_applicable, allow_pickle=False)

    initial_noise = _fixed_noise(
        count,
        int(output_spec["channels"]),
        int(output_spec["length"]),
        args.noise_seed,
    )
    noise_path = output_dir / "initial_noise.npy"
    np.save(noise_path, initial_noise, allow_pickle=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch cannot access CUDA")

    predictions: dict[str, np.ndarray] = {}
    generation_metadata: dict[str, object] = {}
    for model_name, checkpoint_path, kind in (
        ("cfm", args.cfm_checkpoint, "canonical_multistep_cfm"),
        ("rcfm", args.rcfm_checkpoint, "canonical_multistep_rcfm"),
    ):
        prediction, model_metadata = _generate(
            checkpoint_path,
            kind,
            conditions,
            initial_noise,
            args.batch_size,
            args.steps,
            device,
            args.deterministic_seed,
        )
        prediction_path = output_dir / f"{model_name}_predictions_normalized.npy"
        np.save(prediction_path, prediction, allow_pickle=False)
        predictions[model_name] = prediction
        generation_metadata[model_name] = {
            **model_metadata,
            "prediction_file": prediction_path.name,
            "prediction_sha256": _array_sha256(prediction),
        }

    waveform_summaries: dict[str, object] = {}
    per_record_waveform: dict[str, dict[str, np.ndarray]] = {}
    for model_name in ("cfm", "rcfm"):
        waveform_summaries[model_name], per_record_waveform[model_name] = _save_waveform_outputs(
            model_name,
            record_ids,
            targets,
            predictions[model_name],
            output_dir,
        )
    paired_comparison = _paired_model_comparison(
        per_record_waveform["cfm"], per_record_waveform["rcfm"]
    )
    _json(output_dir / "paired_model_comparison.json", paired_comparison)

    clinical_dir = output_dir / "clinical"
    clinical_dir.mkdir()
    clinical_results: dict[str, object] = {}
    for model_name in ("cfm", "rcfm"):
        namespace = _clinical_namespace(
            targets_path,
            output_dir / f"{model_name}_predictions_normalized.npy",
            record_ids_path,
            p_wave_path,
            clinical_dir / model_name / "clinical_metrics.json",
            args,
            str(output_spec["target_lead"]),
        )
        print(f"clinical delineation: {model_name}", flush=True)
        clinical_results[model_name] = evaluate_clinical(namespace)
        _write_clinical_outputs(model_name, clinical_results[model_name], clinical_dir)
    clinical_comparison = _clinical_model_comparison(
        clinical_results["cfm"], clinical_results["rcfm"]
    )
    _write_clinical_model_comparison(clinical_comparison, clinical_dir)

    metadata = {
        "schema_version": 1,
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "protocol": {
            "dataset": "CPSC2018",
            "split": "test",
            "split_hash": config["split_hash"],
            "record_count": count,
            "full_test_record_count": total_records,
            "analysis_unit": "record_no_subject_identifiers_available",
            "normalization_id": config["normalization_id"],
            "target_lead": output_spec["target_lead"],
            "sampling_rate_hz": args.sampling_rate,
            "nfe": args.steps,
            "batch_size": args.batch_size,
            "noise_seed": args.noise_seed,
            "deterministic_seed": args.deterministic_seed,
            "noise_sha256": _array_sha256(initial_noise),
            "same_initial_noise_for_both_models": True,
            "determinism_scope": (
                "bitwise repeatability requires unchanged software, hardware, batch size, and arguments"
            ),
            "qtc_formula": args.qtc_formula,
            "delineation_method": f"neurokit2.ecg_delineate:{args.delineation_method}",
            "st_offset_ms": args.st_offset_ms,
            "p_wave_policy": p_wave_policy,
        },
        "applicability": {
            "rr_pr_qrs_qt_qtc": "eligible_after_independent_delineation",
            "hrv_sdnn_rmssd": "blocked_4_second_records_not_continuous_minimum_duration",
            "physical_p_r_t_amplitudes_and_st": "blocked_unknown_source_physical_unit",
            "normalized_p_r_t_amplitudes_and_st": (
                "exploratory_record_zscore_morphology_only_not_mV_or_clinical_amplitude"
            ),
            "subject_level_inference": "blocked_no_subject_identifiers",
            "single_seed_significance": "blocked_requires_additional_training_seeds",
        },
        "checkpoints": {
            "cfm": {
                "path": str(args.cfm_checkpoint.resolve()),
                "sha256": _sha256(args.cfm_checkpoint),
                **generation_metadata["cfm"],
            },
            "rcfm": {
                "path": str(args.rcfm_checkpoint.resolve()),
                "sha256": _sha256(args.rcfm_checkpoint),
                **generation_metadata["rcfm"],
            },
        },
        "array_hashes": {
            "targets": _array_sha256(targets),
            "conditions": _array_sha256(conditions),
            "record_ids": _array_sha256(record_ids),
        },
        "waveform_metrics": waveform_summaries,
        "paired_model_comparison": paired_comparison,
        "clinical_model_comparison_file": "clinical/paired_clinical_model_comparison.json",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "neurokit2": getattr(nk, "__version__", "unknown"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    _json(output_dir / "paired_evaluation_metadata.json", metadata)
    print(f"paired evaluation complete: {output_dir}", flush=True)
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm_checkpoint", type=Path, required=True)
    parser.add_argument("--rcfm_checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--noise_seed", type=int, default=2025)
    parser.add_argument("--deterministic_seed", type=int, default=31)
    parser.add_argument("--expected_records", type=int, default=688)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--sampling_rate", type=float, default=128.0)
    parser.add_argument("--minimum_hrv_seconds", type=float, default=30.0)
    parser.add_argument(
        "--qtc_formula",
        choices=["bazett", "fridericia", "framingham", "hodges"],
        default="fridericia",
    )
    parser.add_argument("--st_offset_ms", type=float, default=60.0)
    parser.add_argument("--delineation_method", choices=["dwt", "cwt", "peak"], default="dwt")
    parser.add_argument("--clean_method", default="neurokit")
    return parser


if __name__ == "__main__":
    evaluate(build_argparser().parse_args())
