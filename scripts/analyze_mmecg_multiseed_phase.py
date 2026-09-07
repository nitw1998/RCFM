"""Aggregate mmECG three-seed target-informed phase-corrected results."""

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

from scripts.analyze_cpsc2018_multiseed import (
    COMPARISONS,
    SEEDS,
    _holm_adjust,
    _paired_test,
)
from scripts.evaluate_cpsc_zscore_paired import _array_sha256, _waveform_metrics


MODELS = ("CFM", "RCFM-Pan", "RCFM-Pan+OT", "RDDM")
MODEL_KEYS = dict(zip(MODELS, ("cfm", "rcfm", "rcfm_ot", "rddm")))
METRICS = ("rmse", "mae", "waveform_fd", "pearson_window_median")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {"targets", "subject_ids", "source_files"}
        required.update(f"{key}_oracle_aligned_predictions" for key in MODEL_KEYS.values())
        if missing := sorted(required - set(artifact.files)):
            raise ValueError("mmECG phase artifact is missing: " + ", ".join(missing))
        targets = np.asarray(artifact["targets"], dtype=np.float32)
        subjects = np.asarray(artifact["subject_ids"]).astype(str)
        sources = np.asarray(artifact["source_files"]).astype(str)
        predictions = {
            model: np.asarray(artifact[f"{key}_oracle_aligned_predictions"], dtype=np.float32)
            for model, key in MODEL_KEYS.items()
        }
    expected = (2877, 1, 480)
    if targets.shape != expected or any(value.shape != expected for value in predictions.values()):
        raise ValueError("mmECG phase arrays must have shape (2877,1,480)")
    if subjects.shape != (2877,) or sources.shape != (2877,):
        raise ValueError("mmECG phase identities have the wrong shape")
    if any(not np.all(np.isfinite(value)) for value in (targets, *predictions.values())):
        raise FloatingPointError("mmECG phase arrays contain NaN or Inf")
    return targets, subjects, sources, predictions


def _metric_values(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    summary, _ = _waveform_metrics(targets, predictions)
    return {
        "rmse": float(summary["rmse"]),
        "mae": float(summary["mae"]),
        "waveform_fd": float(summary["waveform_fd"]),
        "pearson_window_median": float(summary["per_record_pearson"]["median"]),
    }


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    paths = {seed: getattr(args, f"seed{seed}_phase").resolve() for seed in SEEDS}
    values: dict[int, dict[str, dict[str, float]]] = {}
    frozen_hashes = None
    for seed, path in paths.items():
        targets, subjects, sources, predictions = _load(path)
        hashes = (_array_sha256(targets), _array_sha256(subjects), _array_sha256(sources))
        if frozen_hashes is None:
            frozen_hashes = hashes
        elif hashes != frozen_hashes:
            raise ValueError("mmECG seeds do not share identical target rows and identities")
        values[seed] = {model: _metric_values(targets, predictions[model]) for model in MODELS}

    per_seed_rows = [
        {"model": model, "seed": seed, **values[seed][model]}
        for model in MODELS for seed in SEEDS
    ]
    with (output / "per_seed_phase_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed_rows[0])
        writer.writeheader(); writer.writerows(per_seed_rows)

    summary_rows = []
    for model in MODELS:
        for metric in METRICS:
            samples = np.asarray([values[seed][model][metric] for seed in SEEDS])
            summary_rows.append({
                "model": model, "metric": metric, "n_training_seeds": 3,
                "mean": float(samples.mean()), "sample_sd_ddof1": float(samples.std(ddof=1)),
            })
    with (output / "model_phase_mean_sd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0])
        writer.writeheader(); writer.writerows(summary_rows)

    tests, raw_p = {}, {}
    for comparison, first, second in COMPARISONS:
        tests[comparison] = {}
        for metric in METRICS:
            first_values = np.asarray([values[seed][first][metric] for seed in SEEDS])
            second_values = np.asarray([values[seed][second][metric] for seed in SEEDS])
            test = _paired_test(first_values, second_values)
            test.update({
                "first_model": first, "second_model": second,
                "lower_is_better": metric != "pearson_window_median",
            })
            tests[comparison][metric] = test
            raw_p[f"{comparison}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p)
    test_rows = []
    for comparison, metric_tests in tests.items():
        for metric, test in metric_tests.items():
            test["holm_adjusted_p_value_16_tests"] = adjusted[f"{comparison}/{metric}"]
            test_rows.append({"comparison": comparison, "metric": metric, **test})
    with (output / "paired_seed_phase_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in test_rows[0] if key != "seed_differences_first_minus_second"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in test_rows)

    summary = {
        "status": "completed", "dataset": "mmECG", "seeds": list(SEEDS),
        "phase_protocol": {
            "target_informed": True, "lag_search_samples": [-16, 16],
            "sampling_rate_hz": 128, "support_samples": 480,
            "lag_objective": "maximize per-window Pearson against held-out ECG target",
        },
        "mean_and_sample_sd": summary_rows,
        "paired_seed_tests_oracle_aligned": tests,
        "multiplicity_family": "Holm adjustment across 4 predeclared comparisons x 4 phase-corrected waveform metrics",
        "claim_boundary": (
            "SD and paired tests use three independently trained seeds. The exact sign-flip test "
            "has minimum attainable p=0.25. The 2,877 overlapping windows and three held-out "
            "subjects are not treated as independent replicates. Phase correction is target-informed."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = ["per_seed_phase_metrics.csv", "model_phase_mean_sd.csv", "paired_seed_phase_tests.csv", "summary.json", "protocol.json"]
    protocol = {
        "status": "completed", "input_sha256": {str(path): _sha256(path) for path in paths.values()},
        "frozen_array_sha256": {"targets": frozen_hashes[0], "subject_ids": frozen_hashes[1], "source_files": frozen_hashes[2]},
        "script_sha256": _sha256(Path(__file__).resolve()), "outputs": outputs,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in SEEDS:
        parser.add_argument(f"--seed{seed}_phase", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
