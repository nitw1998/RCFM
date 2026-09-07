"""Aggregate frozen MIMIC-AFib seed-31/32/33 waveform results.

Inference is performed only across matched training seeds. The MIMIC artifact does
not expose certified subject identities, so this entry deliberately does not test
the 1,800 windows as independent observations.
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
METRICS = ("rmse", "mae", "waveform_fd")
COMPARISONS = (
    ("rcfm_pan_vs_cfm", "RCFM-Pan", "CFM"),
    ("rcfm_pan_ot_vs_rcfm_pan", "RCFM-Pan+OT", "RCFM-Pan"),
    ("rcfm_pan_ot_vs_cfm", "RCFM-Pan+OT", "CFM"),
    ("rcfm_pan_ot_vs_rddm", "RCFM-Pan+OT", "RDDM"),
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flow_metrics(run_dir: Path) -> dict[str, float]:
    metadata = _json(run_dir / "run_metadata.json")
    if metadata.get("status") != "completed" or metadata.get("subject_metadata_available") is not False:
        raise ValueError(f"invalid completed MIMIC flow run: {run_dir}")
    rows: dict[str, float] = {}
    with (run_dir / "validation_metrics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) == 500 and row["metric"] in {"val/rmse", "val/mae", "val/waveform_fd"}:
                rows[row["metric"].removeprefix("val/")] = float(row["value"])
    if set(rows) != set(METRICS):
        raise ValueError(f"missing epoch-500 waveform metrics in {run_dir}")
    return rows


def _rddm_metrics(eval_dir: Path, expected_seed: int) -> tuple[dict[str, float], dict[str, str]]:
    protocol = _json(eval_dir / "protocol.json")
    summary = _json(eval_dir / "waveform_summary.json")["model"]["rddm"]
    frozen = protocol["protocol"]
    if protocol.get("status") != "completed" or frozen.get("heldout_windows_evaluated") != 1800:
        raise ValueError(f"invalid completed RDDM evaluation: {eval_dir}")
    recorded_seed = frozen.get("training_seed", 31)
    if recorded_seed != expected_seed:
        raise ValueError(f"RDDM training seed mismatch in {eval_dir}: {recorded_seed}")
    metrics = {metric: float(summary[metric]) for metric in METRICS}
    hashes = {key: str(frozen[key]) for key in ("target_sha256", "condition_sha256")}
    return metrics, hashes


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

    flow_roots = {
        "CFM": {31: args.cfm_seed31, 32: args.multiseed_root / "mimic_afib_cfm_s32_v1", 33: args.multiseed_root / "mimic_afib_cfm_s33_v1"},
        "RCFM-Pan": {31: args.rcfm_seed31, 32: args.multiseed_root / "mimic_afib_rcfm_pan_s32_v1", 33: args.multiseed_root / "mimic_afib_rcfm_pan_s33_v1"},
        "RCFM-Pan+OT": {31: args.rcfm_ot_seed31, 32: args.multiseed_root / "mimic_afib_rcfm_pan_exact_ot_s32_v1", 33: args.multiseed_root / "mimic_afib_rcfm_pan_exact_ot_s33_v1"},
    }
    values: dict[str, dict[int, dict[str, float]]] = {
        model: {seed: _flow_metrics(path.resolve()) for seed, path in roots.items()}
        for model, roots in flow_roots.items()
    }
    rddm_roots = {31: args.rddm_seed31_eval, 32: args.rddm_multiseed_eval_root / "rddm_s32", 33: args.rddm_multiseed_eval_root / "rddm_s33"}
    rddm_hashes = []
    values["RDDM"] = {}
    for seed, root in rddm_roots.items():
        values["RDDM"][seed], hashes = _rddm_metrics(root.resolve(), seed)
        rddm_hashes.append(hashes)
    if len({item["target_sha256"] for item in rddm_hashes}) != 1 or len({item["condition_sha256"] for item in rddm_hashes}) != 1:
        raise ValueError("RDDM seeds do not share the same frozen target and condition arrays")

    with (output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "seed", *METRICS))
        writer.writeheader()
        for model in values:
            for seed in SEEDS:
                writer.writerow({"model": model, "seed": seed, **values[model][seed]})

    summary_rows = []
    for model in values:
        for metric in METRICS:
            metric_values = np.asarray([values[model][seed][metric] for seed in SEEDS])
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
            first_values = np.asarray([values[first][seed][metric] for seed in SEEDS])
            second_values = np.asarray([values[second][seed][metric] for seed in SEEDS])
            test = _paired_test(first_values, second_values)
            test.update({"first_model": first, "second_model": second, "lower_is_better": True})
            tests[comparison][metric] = test
            raw_p[f"{comparison}/{metric}"] = float(test["raw_paired_t_p_value"])
    adjusted = _holm_adjust(raw_p)
    test_rows = []
    for comparison, metric_tests in tests.items():
        for metric, test in metric_tests.items():
            test["holm_adjusted_p_value_12_tests"] = adjusted[f"{comparison}/{metric}"]
            test_rows.append({"comparison": comparison, "metric": metric, **test})
    with (output_dir / "paired_seed_tests.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in test_rows[0] if key != "seed_differences_first_minus_second"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key in fields} for row in test_rows)

    summary = {
        "status": "completed",
        "dataset": "MIMIC-AFib",
        "seeds": list(SEEDS),
        "models": values,
        "mean_and_sample_sd": summary_rows,
        "paired_seed_tests": tests,
        "multiplicity_family": "Holm adjustment across 4 predeclared comparisons x 3 waveform metrics",
        "claim_boundary": (
            "Three training seeds provide low-resolution retraining evidence. Paired t-tests have "
            "df=2; the exact two-sided sign-flip test has minimum attainable p=0.25. The frozen "
            "1,800 windows lack certified subject identities and are not treated as independent."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inputs = [
        *(root / "run_metadata.json" for roots in flow_roots.values() for root in roots.values()),
        *(root / "validation_metrics.csv" for roots in flow_roots.values() for root in roots.values()),
        *(root / "protocol.json" for root in rddm_roots.values()),
        *(root / "waveform_summary.json" for root in rddm_roots.values()),
    ]
    protocol = {"input_sha256": {str(path.resolve()): _sha256(path.resolve()) for path in inputs}, "outputs": ["per_seed_metrics.csv", "model_mean_sd.csv", "paired_seed_tests.csv", "summary.json", "protocol.json"]}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfm_seed31", type=Path, required=True)
    parser.add_argument("--rcfm_seed31", type=Path, required=True)
    parser.add_argument("--rcfm_ot_seed31", type=Path, required=True)
    parser.add_argument("--multiseed_root", type=Path, required=True)
    parser.add_argument("--rddm_seed31_eval", type=Path, required=True)
    parser.add_argument("--rddm_multiseed_eval_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
