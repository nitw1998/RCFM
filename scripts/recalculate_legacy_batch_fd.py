#!/usr/bin/env python3
"""Recalculate CAT-paper-form FD as a deterministic mean over record batches."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from src.rcfm.metrics.waveform import waveform_frechet_distance


def literal_fd(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    """Use the covariance-product expression printed in CATransformer."""

    real = np.asarray(reference, dtype=np.float64)
    fake = np.asarray(generated, dtype=np.float64)
    if real.shape != fake.shape or real.ndim != 2 or len(real) < 2:
        raise ValueError("literal FD requires matching (observations, features) arrays")
    if not np.all(np.isfinite(real)) or not np.all(np.isfinite(fake)):
        raise FloatingPointError("literal FD inputs contain NaN or Inf")
    mean_difference = real.mean(axis=0) - fake.mean(axis=0)
    covariance_real = np.atleast_2d(np.cov(real, rowvar=False))
    covariance_generated = np.atleast_2d(np.cov(fake, rowvar=False))
    product_eigenvalues = np.linalg.eigvals(covariance_real @ covariance_generated).astype(
        np.complex128
    )
    trace_product_sqrt = float(np.sqrt(product_eigenvalues).real.sum())
    mean_term = float(mean_difference @ mean_difference)
    trace_real = float(np.trace(covariance_real))
    trace_generated = float(np.trace(covariance_generated))
    fd = mean_term + trace_real + trace_generated - 2.0 * trace_product_sqrt
    if not np.isfinite(fd):
        raise FloatingPointError("literal FD is nonfinite")
    return {
        "fd": float(max(fd, 0.0)),
        "mean_term": mean_term,
        "trace_reference": trace_real,
        "trace_generated": trace_generated,
        "twice_trace_product_sqrt": 2.0 * trace_product_sqrt,
        "maximum_product_eigenvalue_imaginary_magnitude": float(
            np.max(np.abs(product_eigenvalues.imag))
        ),
    }


def batch_fd_rows(
    reference: np.ndarray, generated: np.ndarray, batch_size: int
) -> list[dict[str, object]]:
    """Pool channels as observations within each deterministic record batch."""

    if reference.shape != generated.shape or reference.ndim != 3 or batch_size < 2:
        raise ValueError("batch FD requires matching (records, channels, samples) arrays")
    rows = []
    for batch_index, start in enumerate(range(0, len(reference), batch_size)):
        stop = min(start + batch_size, len(reference))
        if stop - start < 2:
            continue
        real = reference[start:stop].reshape(-1, reference.shape[-1])
        fake = generated[start:stop].reshape(-1, generated.shape[-1])
        result = literal_fd(real, fake)
        rows.append({
            "batch_index": batch_index,
            "record_start": start,
            "record_stop_exclusive": stop,
            "records": stop - start,
            "pooled_channel_observations": len(real),
            **result,
        })
    if not rows:
        raise ValueError("no complete FD batch is available")
    return rows


def low_rank_literal_fd(reference: np.ndarray, generated: np.ndarray) -> float:
    """Equivalent CAT FD using the smaller observation-space SVD."""

    real = np.asarray(reference, dtype=np.float64)
    fake = np.asarray(generated, dtype=np.float64)
    if real.shape != fake.shape or real.ndim != 2 or len(real) < 2:
        raise ValueError("low-rank FD requires matching (observations, features) arrays")
    real_centered = real - real.mean(axis=0)
    fake_centered = fake - fake.mean(axis=0)
    denominator = len(real) - 1
    mean_difference = real.mean(axis=0) - fake.mean(axis=0)
    trace_product_sqrt = float(
        np.linalg.svd(real_centered @ fake_centered.T, compute_uv=False).sum() / denominator
    )
    value = (
        mean_difference @ mean_difference
        + np.sum(real_centered**2) / denominator
        + np.sum(fake_centered**2) / denominator
        - 2.0 * trace_product_sqrt
    )
    return float(max(value, 0.0))


def leadwise_batch_fd(
    reference: np.ndarray, generated: np.ndarray, batch_size: int
) -> list[float]:
    """Calculate per-lead batch-mean FD without pooling lead distributions."""

    values = []
    for lead in range(reference.shape[1]):
        batches = []
        for start in range(0, len(reference), batch_size):
            stop = min(start + batch_size, len(reference))
            if stop - start >= 2:
                batches.append(low_rank_literal_fd(
                    reference[start:stop, lead], generated[start:stop, lead]
                ))
        values.append(float(np.mean(batches)))
    return values


def _parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be DATASET=EVALUATION_DIR")
    dataset, path = value.split("=", 1)
    if not dataset or not path:
        raise argparse.ArgumentTypeError("--input must be DATASET=EVALUATION_DIR")
    return dataset, Path(path)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    for dataset, source_value in args.input:
        source = source_value.resolve()
        protocol_path = source / "protocol.json"
        prediction_path = source / "cfm_predictions.npy"
        reference_path = source / "paired_reference.npz"
        waveform_path = source / "waveform_summary.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        contract = protocol.get("protocol", {})
        if (
            protocol.get("status") != "completed"
            or contract.get("phase_correction_applied") is not False
            or str(contract.get("dataset", "")).lower() != dataset.lower()
        ):
            raise ValueError(f"{dataset} source violates the completed raw prediction contract")
        with np.load(reference_path, allow_pickle=False) as artifact:
            reference = np.asarray(artifact["targets"], dtype=np.float64)
        generated = np.asarray(np.load(prediction_path, mmap_mode="r"), dtype=np.float64)
        if reference.shape != generated.shape or reference.ndim != 3:
            raise ValueError(f"{dataset} prediction/reference shapes do not match")
        rows = batch_fd_rows(reference, generated, args.batch_size)
        values = np.asarray([row["fd"] for row in rows], dtype=np.float64)
        lead_values = (
            [float(np.mean(values))]
            if reference.shape[1] == 1
            else leadwise_batch_fd(reference, generated, args.batch_size)
        )
        target_leads = list(contract.get("target_leads", ()))
        if len(target_leads) != reference.shape[1]:
            raise ValueError(f"{dataset} target-lead labels do not match channel count")
        pooled_reference = reference.reshape(-1, reference.shape[-1])
        pooled_generated = generated.reshape(-1, generated.shape[-1])
        full_literal = literal_fd(pooled_reference, pooled_generated)
        full_stable = waveform_frechet_distance(pooled_reference, pooled_generated)
        current = json.loads(waveform_path.read_text(encoding="utf-8"))["models"]["cfm"]
        summaries.append({
            "dataset": dataset,
            "records": len(reference),
            "channels": reference.shape[1],
            "samples": reference.shape[2],
            "batch_size_records": args.batch_size,
            "batch_count": len(rows),
            "legacy_contiguous_batch256_fd_mean": float(np.mean(values)),
            "legacy_contiguous_batch256_fd_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "legacy_contiguous_batch256_fd_min": float(np.min(values)),
            "legacy_contiguous_batch256_fd_max": float(np.max(values)),
            "leadwise_contiguous_batch256_fd": dict(zip(target_leads, lead_values)),
            "leadwise_contiguous_batch256_fd_macro_mean": float(np.mean(lead_values)),
            "pooled_full_split_literal_fd": full_literal["fd"],
            "pooled_full_split_stable_fd": full_stable,
            "literal_minus_stable_full_split": full_literal["fd"] - full_stable,
            "current_full_split_macro_lead_wfd": current["waveform_fd_macro_lead"],
            "dataset_version": contract.get("dataset_version"),
            "normalization_id": contract.get("normalization_id"),
            "checkpoint_sha256": protocol.get("checkpoint", {}).get("sha256"),
        })
        all_rows.extend({"dataset": dataset, **row} for row in rows)
        inputs[dataset] = {
            "source_protocol_sha256": _sha256(protocol_path),
            "paired_reference_sha256": _sha256(reference_path),
            "prediction_sha256": _sha256(prediction_path),
            "waveform_summary_sha256": _sha256(waveform_path),
        }
    summary_path = output / "legacy_batch256_fd_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "metric": {
            "formula": "||mu_p-mu_q||^2 + Tr(Sigma_p+Sigma_q-2*(Sigma_p*Sigma_q)^(1/2))",
            "covariance": "numpy sample covariance ddof=1",
            "batch_aggregation": "arithmetic mean of per-batch FD",
            "batch_order": "saved deterministic prediction row order; no shuffle",
            "multichannel_policy": "pool record-channel waveforms as observations within each record batch",
            "multichannel_alternative": "also report per-lead batch-mean FD followed by an unweighted lead macro mean",
            "compatibility_boundary": "legacy-compatible deterministic estimator; not an exact replay of unrecorded shuffled batch membership",
        },
        "results": summaries,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    rows_path = output / "per_batch_fd.csv"
    fields = list(all_rows[0])
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps({
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "inputs": inputs,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {path.name: _sha256(path) for path in (summary_path, rows_path)},
        "script_sha256": _sha256(Path(__file__)),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_parse_input, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
