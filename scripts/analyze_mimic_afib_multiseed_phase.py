"""Aggregate MIMIC-AFib three-seed bounded oracle phase diagnostics.

The phase-corrected estimand follows the response-letter protocol exactly: for
each frozen generated/target pair, select the Pearson-maximizing integer lag in
[-16, 16] samples, then evaluate both unshifted and shifted predictions on the
same central 480-sample support.  The target-informed result is a morphology
diagnostic and is never a deployable alignment score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_mimic_afib_multiseed import (
    COMPARISONS,
    SEEDS,
    _holm_adjust,
    _paired_test,
)
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _waveform_metrics
from scripts.evaluate_mimic_phase_corrected import _fixed_support_align
from scripts.visualize_mimic_flow_predictions import _lag_diagnostic


MODELS = ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM")
METRICS = ("rmse", "mae", "waveform_fd", "pearson_window_median")
PHASE_MODES = ("unshifted_fixed_support", "oracle_aligned_fixed_support")
FLOW_KEYS = {
    "CFM": "cfm_predictions",
    "RCFM-Pan": "rcfm_predictions",
    "RCFM-Pan+OT": "rcfm_ot_predictions",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_values(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    summary, _ = _waveform_metrics(reference, generated)
    return {
        "rmse": float(summary["rmse"]),
        "mae": float(summary["mae"]),
        "waveform_fd": float(summary["waveform_fd"]),
        "pearson_window_median": float(summary["per_record_pearson"]["median"]),
    }


def _phase_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    max_lag: int,
    sampling_rate: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    lag_summary, lag_values = _lag_diagnostic(
        targets, predictions, max_lag, sampling_rate
    )
    shifts = lag_values["best_lag_samples"].astype(np.int32)
    target_fixed, unshifted, aligned = _fixed_support_align(
        targets, predictions, shifts, max_lag
    )
    _, unshifted_per = _waveform_metrics(target_fixed, unshifted)
    _, aligned_per = _waveform_metrics(target_fixed, aligned)
    return (
        {
            "unshifted_fixed_support": _metric_values(target_fixed, unshifted),
            "oracle_aligned_fixed_support": _metric_values(target_fixed, aligned),
        },
        {
            "shift_samples_mean": float(np.mean(shifts)),
            "shift_samples_median": float(np.median(shifts)),
            "absolute_shift_samples_median": float(np.median(np.abs(shifts))),
            "fraction_at_search_boundary": float(np.mean(np.abs(shifts) == max_lag)),
            "fraction_rmse_improved": float(
                np.mean(aligned_per["rmse"] < unshifted_per["rmse"])
            ),
            "oracle_pearson_median_over_lag_search_support": float(
                lag_summary["best_correlation_median"]
            ),
        },
    )


def _load_seed(
    flow_path: Path, rddm_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    with np.load(flow_path, allow_pickle=False) as flow, np.load(
        rddm_path, allow_pickle=False
    ) as rddm:
        required_flow = {"targets", "conditions", *FLOW_KEYS.values()}
        required_rddm = {
            "targets",
            "conditions",
            "rddm_predictions",
            "source_test_rows_before_zero_filter",
        }
        if required_flow - set(flow.files) or required_rddm - set(rddm.files):
            raise ValueError("prediction artifacts do not contain the required arrays")
        targets = np.asarray(flow["targets"], dtype=np.float32)
        conditions = np.asarray(flow["conditions"], dtype=np.float32)
        if not np.array_equal(targets, rddm["targets"]):
            raise ValueError("flow and RDDM targets are not row-aligned")
        if not np.array_equal(conditions, rddm["conditions"]):
            raise ValueError("flow and RDDM conditions are not row-aligned")
        predictions = {
            model: np.asarray(flow[key], dtype=np.float32)
            for model, key in FLOW_KEYS.items()
        }
        predictions["RDDM"] = np.asarray(rddm["rddm_predictions"], dtype=np.float32)
        rows = np.asarray(rddm["source_test_rows_before_zero_filter"], dtype=np.int64)
    expected_shape = (1800, 1, 512)
    if any(array.shape != expected_shape for array in (targets, conditions, *predictions.values())):
        raise ValueError("all MIMIC arrays must have shape (1800,1,512)")
    if rows.shape != (1800,) or len(np.unique(rows)) != 1800:
        raise ValueError("invalid frozen test-row mapping")
    if any(not np.all(np.isfinite(array)) for array in (targets, conditions, *predictions.values())):
        raise FloatingPointError("phase-analysis arrays must be finite")
    return targets, conditions, predictions, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_lag_samples != 16 or args.sampling_rate != 128:
        raise ValueError("response-letter protocol requires max lag 16 at 128 Hz")

    flow_paths = {31: args.flow_seed31, 32: args.flow_seed32, 33: args.flow_seed33}
    rddm_paths = {31: args.rddm_seed31, 32: args.rddm_seed32, 33: args.rddm_seed33}
    values: dict[str, dict[int, dict[str, dict[str, float]]]] = {
        model: {} for model in MODELS
    }
    lag_diagnostics: dict[str, dict[int, dict[str, float]]] = {
        model: {} for model in MODELS
    }
    frozen_hashes = None
    seed_rows = []
    for seed in SEEDS:
        targets, conditions, predictions, rows = _load_seed(
            flow_paths[seed].resolve(), rddm_paths[seed].resolve()
        )
        hashes = (
            _array_sha256(targets),
            _array_sha256(conditions),
            _array_sha256(rows),
        )
        if frozen_hashes is None:
            frozen_hashes = hashes
        elif hashes != frozen_hashes:
            raise ValueError("training seeds do not share identical targets, conditions, and rows")
        for model in MODELS:
            phase_values, diagnostics = _phase_metrics(
                targets, predictions[model], args.max_lag_samples, args.sampling_rate
            )
            values[model][seed] = phase_values
            lag_diagnostics[model][seed] = diagnostics
            for phase_mode in PHASE_MODES:
                seed_rows.append(
                    {"model": model, "seed": seed, "phase_mode": phase_mode, **phase_values[phase_mode]}
                )

    with (output_dir / "per_seed_phase_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=seed_rows[0])
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_rows = []
    for model in MODELS:
        for phase_mode in PHASE_MODES:
            for metric in METRICS:
                samples = np.asarray(
                    [values[model][seed][phase_mode][metric] for seed in SEEDS],
                    dtype=np.float64,
                )
                summary_rows.append(
                    {
                        "model": model,
                        "phase_mode": phase_mode,
                        "metric": metric,
                        "n_training_seeds": len(SEEDS),
                        "mean": float(samples.mean()),
                        "sample_sd_ddof1": float(samples.std(ddof=1)),
                    }
                )
    with (output_dir / "model_phase_mean_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)

    tests = {}
    raw_p = {}
    for comparison, first, second in COMPARISONS:
        tests[comparison] = {}
        for metric in METRICS:
            first_values = np.asarray(
                [values[first][seed]["oracle_aligned_fixed_support"][metric] for seed in SEEDS]
            )
            second_values = np.asarray(
                [values[second][seed]["oracle_aligned_fixed_support"][metric] for seed in SEEDS]
            )
            test = _paired_test(first_values, second_values)
            test.update(
                {
                    "first_model": first,
                    "second_model": second,
                    "lower_is_better": metric != "pearson_window_median",
                }
            )
            tests[comparison][metric] = test
            raw_p[f"{comparison}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p)
    test_rows = []
    for comparison, metric_tests in tests.items():
        for metric, test in metric_tests.items():
            test["holm_adjusted_p_value_16_tests"] = adjusted[f"{comparison}/{metric}"]
            test_rows.append({"comparison": comparison, "metric": metric, **test})
    with (output_dir / "paired_seed_phase_tests.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [key for key in test_rows[0] if key != "seed_differences_first_minus_second"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields} for row in test_rows
        )

    summary = {
        "status": "completed",
        "dataset": "MIMIC-AFib",
        "seeds": list(SEEDS),
        "phase_protocol": {
            "sampling_rate_hz": args.sampling_rate,
            "lag_search_samples": [-args.max_lag_samples, args.max_lag_samples],
            "lag_objective": "maximize per-window Pearson against the held-out ECG target",
            "fixed_support_rule": "target[16:496] versus generated[16-shift:496-shift]",
            "support_samples": 480,
            "positive_shift_definition": "delay the generated waveform",
            "circular_wrap": False,
            "target_informed": True,
        },
        "mean_and_sample_sd": summary_rows,
        "lag_diagnostics": lag_diagnostics,
        "paired_seed_tests_oracle_aligned": tests,
        "multiplicity_family": "Holm adjustment across 4 predeclared comparisons x 4 phase-corrected waveform metrics",
        "claim_boundary": (
            "The aligned estimand is a target-informed oracle morphology diagnostic, not "
            "deployable post-processing. SD is across three independently trained models, not "
            "across windows. The 1,800 windows lack certified identities and are not independent."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs = [*(path.resolve() for path in flow_paths.values()), *(path.resolve() for path in rddm_paths.values())]
    outputs = [
        "per_seed_phase_metrics.csv",
        "model_phase_mean_sd.csv",
        "paired_seed_phase_tests.csv",
        "summary.json",
        "protocol.json",
    ]
    protocol = {
        "status": "completed",
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "frozen_array_sha256": {
            "targets": frozen_hashes[0],
            "conditions": frozen_hashes[1],
            "source_rows": frozen_hashes[2],
        },
        "script_sha256": _sha256(Path(__file__).resolve()),
        "outputs": outputs,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--flow_seed{seed}", type=Path, required=True)
        parser.add_argument(f"--rddm_seed{seed}", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--max_lag_samples", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
