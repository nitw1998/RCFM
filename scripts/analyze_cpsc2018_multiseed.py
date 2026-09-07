"""Aggregate frozen CPSC2018 seed-31/32/33 four-model waveform results.

The validation records do not carry certified subject identities. Inference is
therefore restricted to paired training-seed summaries; records are not treated
as independent subjects.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (31, 32, 33)
MODEL_KEYS = {
    "CFM": "cfm",
    "RCFM-Pan": "rcfm",
    "RCFM-Pan+OT": "rcfm_ot",
    "RDDM": "rddm",
}
METRICS = ("rmse", "mae", "waveform_fd", "per_record_pearson_median")
SUMMARY_KEYS = {
    "rmse": "rmse",
    "mae": "mae",
    "waveform_fd": "waveform_fd_macro_lead",
    "per_record_pearson_median": "per_record_pearson_median",
}
COMPARISONS = (
    ("rcfm_pan_vs_cfm", "RCFM-Pan", "CFM"),
    ("rcfm_pan_ot_vs_rcfm_pan", "RCFM-Pan+OT", "RCFM-Pan"),
    ("rcfm_pan_ot_vs_cfm", "RCFM-Pan+OT", "CFM"),
    ("rcfm_pan_ot_vs_rddm", "RCFM-Pan+OT", "RDDM"),
)
EXPECTED_SPLIT_HASH = "35e0a796a60e4d6979b1f050048495fdf1826eedda7d40e47a56d4dcd5874223"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_metrics(eval_dir: Path, expected_seed: int) -> tuple[dict[str, dict[str, float]], str]:
    protocol = _json(eval_dir / "protocol.json")
    frozen = protocol.get("protocol", {})
    if protocol.get("status") != "completed":
        raise ValueError(f"incomplete CPSC2018 evaluation: {eval_dir}")
    if frozen.get("dataset") != "CPSC2018" or frozen.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError(f"CPSC2018 dataset contract mismatch: {eval_dir}")
    if int(frozen.get("evaluated_records", -1)) != 686:
        raise ValueError(f"CPSC2018 evaluation must contain all 686 records: {eval_dir}")
    recorded_seed = frozen.get("training_seed", 31)
    if int(recorded_seed) != expected_seed:
        raise ValueError(f"training seed mismatch in {eval_dir}: {recorded_seed}")
    if set(protocol.get("selection", {}).values()) != {"epoch_500_endpoint_no_validation_metric_selection"}:
        raise ValueError(f"CPSC2018 checkpoint selection contract changed: {eval_dir}")
    summary = _json(eval_dir / "waveform_summary.json").get("models", {})
    if set(summary) != set(MODEL_KEYS.values()):
        raise ValueError(f"CPSC2018 model set mismatch: {eval_dir}")
    values = {
        label: {metric: float(summary[key][SUMMARY_KEYS[metric]]) for metric in METRICS}
        for label, key in MODEL_KEYS.items()
    }
    reference_hash = str(protocol.get("artifacts", {}).get("paired_reference_sha256", ""))
    if not reference_hash:
        raise ValueError(f"missing paired reference hash: {eval_dir}")
    return values, reference_hash


def _exact_sign_flip_p_value(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    observed = abs(float(values.mean()))
    permuted = [
        abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    previous = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, raw[key] * (count - rank)))
        adjusted[key] = previous
    return adjusted


def _paired_test(first: np.ndarray, second: np.ndarray) -> dict[str, float | list[float]]:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    result = stats.ttest_rel(first, second)
    mean = float(difference.mean())
    sd = float(difference.std(ddof=1))
    sem = sd / np.sqrt(len(difference))
    critical = float(stats.t.ppf(0.975, df=len(difference) - 1))
    return {
        "seed_differences_first_minus_second": difference.tolist(),
        "mean_difference": mean,
        "difference_sample_sd": sd,
        "ci95_lower": mean - critical * sem,
        "ci95_upper": mean + critical * sem,
        "paired_t_statistic": float(result.statistic),
        "raw_paired_t_p_value": float(result.pvalue),
        "paired_cohens_dz": mean / sd if sd > 0 else float("nan"),
        "exact_two_sided_sign_flip_p_value": _exact_sign_flip_p_value(difference),
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dirs = {
        31: args.seed31_eval.resolve(),
        32: args.seed32_eval.resolve(),
        33: args.seed33_eval.resolve(),
    }
    by_seed = {}
    reference_hashes = set()
    for seed, eval_dir in eval_dirs.items():
        by_seed[seed], reference_hash = _evaluation_metrics(eval_dir, seed)
        reference_hashes.add(reference_hash)
    if len(reference_hashes) != 1:
        raise ValueError("CPSC2018 seeds do not share the same frozen paired reference")

    with (output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "seed", *METRICS))
        writer.writeheader()
        for model in MODEL_KEYS:
            for seed in SEEDS:
                writer.writerow({"model": model, "seed": seed, **by_seed[seed][model]})

    summary_rows = []
    for model in MODEL_KEYS:
        for metric in METRICS:
            metric_values = np.asarray([by_seed[seed][model][metric] for seed in SEEDS])
            summary_rows.append({
                "model": model,
                "metric": metric,
                "n_seeds": len(SEEDS),
                "mean": float(metric_values.mean()),
                "sample_sd_ddof1": float(metric_values.std(ddof=1)),
            })
    with (output_dir / "model_mean_sd.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)

    tests: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for comparison, first, second in COMPARISONS:
        tests[comparison] = {}
        for metric in METRICS:
            first_values = np.asarray([by_seed[seed][first][metric] for seed in SEEDS])
            second_values = np.asarray([by_seed[seed][second][metric] for seed in SEEDS])
            test = _paired_test(first_values, second_values)
            test.update({
                "first_model": first,
                "second_model": second,
                "higher_is_better": metric == "per_record_pearson_median",
            })
            tests[comparison][metric] = test
            raw_p[f"{comparison}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p)
    test_rows = []
    for comparison, metric_tests in tests.items():
        for metric, test in metric_tests.items():
            test["holm_adjusted_p_value_16_tests"] = adjusted[f"{comparison}/{metric}"]
            test_rows.append({"comparison": comparison, "metric": metric, **test})
    with (output_dir / "paired_seed_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in test_rows[0] if key != "seed_differences_first_minus_second"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in test_rows)

    summary = {
        "status": "completed",
        "dataset": "CPSC2018",
        "seeds": list(SEEDS),
        "mean_and_sample_sd": summary_rows,
        "paired_seed_tests": tests,
        "multiplicity_family": "Holm adjustment across 4 predeclared comparisons x 4 waveform metrics",
        "metric_definition": {"waveform_fd": "fixed macro-average across the 11 generated target leads"},
        "claim_boundary": (
            "Three matched training seeds provide low-resolution retraining evidence. Paired t-tests "
            "have df=2; the exact two-sided sign-flip test has minimum attainable p=0.25. The 686 "
            "validation records lack certified subject identities and are not treated as independent subjects."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inputs = [path for eval_dir in eval_dirs.values() for path in (
        eval_dir / "protocol.json", eval_dir / "waveform_summary.json"
    )]
    protocol = {
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "paired_reference_sha256": next(iter(reference_hashes)),
        "outputs": ["per_seed_metrics.csv", "model_mean_sd.csv", "paired_seed_tests.csv", "summary.json", "protocol.json"],
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed31_eval", type=Path, required=True)
    parser.add_argument("--seed32_eval", type=Path, required=True)
    parser.add_argument("--seed33_eval", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
