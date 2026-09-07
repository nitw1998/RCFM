#!/usr/bin/env python3
"""Aggregate three-seed RCFM-OT ECG parameters and group-level paired tests.

The inferential unit is patient, subject, or source record.  For every ECG
parameter, rows must be jointly measurable for the reference and all three
training seeds.  Generated group means are averaged across seeds before the
paired test against the reference group mean; seeds are never treated as
patients.  Phase-corrected datasets consume only their explicitly labelled
target-informed oracle-aligned rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_cpsc_zscore_paired import _sha256
from scripts.evaluate_random_window_ecg_clinical import PARAMETERS, _unit


SEEDS = (31, 32, 33)
DATASETS = ("ptbxl", "cpsc2018", "mimic_afib", "wesad", "mmecg")
DATASET_SPECS = {
    "ptbxl": {"phase": "raw", "group_level": "patient", "physical": True},
    "cpsc2018": {"phase": "raw", "group_level": "source_record", "physical": False},
    "mimic_afib": {"phase": "oracle_aligned", "group_level": "source_record", "physical": False},
    "wesad": {"phase": "oracle_aligned", "group_level": "subject", "physical": False},
    "mmecg": {"phase": "oracle_aligned", "group_level": "subject", "physical": False},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str | None) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _load_rows(dataset: str, directory: Path) -> dict[tuple[str, str], dict[str, tuple[float, float]]]:
    """Return row key -> parameter -> (reference, generated)."""

    spec = DATASET_SPECS[dataset]
    if dataset in {"ptbxl", "cpsc2018"}:
        path = directory / "per_window_parameters.csv"
        raw = _read_csv(path)
        output: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
        for row in raw:
            key = (row["window_index"], row["lead"])
            output[key] = {
                parameter: pair
                for parameter in PARAMETERS
                if (pair := (_finite(row.get(f"reference_{parameter}")),
                             _finite(row.get(f"generated_{parameter}"))))
                and pair[0] is not None and pair[1] is not None
            }
            output[key]["__group__"] = (row["group_id"], row["group_id"])  # type: ignore[assignment]
        return output

    path = directory / "per_window_ecg_parameters.csv"
    raw = _read_csv(path)
    output = {}
    for row in raw:
        if dataset in {"wesad", "mmecg"} and row.get("phase_mode") != "oracle_aligned":
            continue
        key = (row["window_index"], "single")
        group = row["record_id"] if dataset == "mimic_afib" else row["subject_id"]
        generated_prefix = "oracle_aligned" if dataset == "mimic_afib" else "generated"
        output[key] = {
            parameter: pair
            for parameter in PARAMETERS
            if (pair := (_finite(row.get(f"reference_{parameter}")),
                         _finite(row.get(f"{generated_prefix}_{parameter}"))))
            and pair[0] is not None and pair[1] is not None
        }
        output[key]["__group__"] = (group, group)  # type: ignore[assignment]
    return output


def _group_common_pairs(
    seed_rows: dict[int, dict[tuple[str, str], dict[str, tuple[float, float]]]],
    parameter: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], np.ndarray]:
    common = set.intersection(*(set(rows) for rows in seed_rows.values()))
    common = {
        key for key in common
        if all(parameter in seed_rows[seed][key] for seed in SEEDS)
    }
    grouped: dict[str, list[tuple[float, dict[int, float]]]] = defaultdict(list)
    for key in sorted(common):
        groups = [str(seed_rows[seed][key]["__group__"][0]) for seed in SEEDS]
        if len(set(groups)) != 1:
            raise ValueError(f"group identity differs across seeds for row {key}")
        references = np.asarray([seed_rows[seed][key][parameter][0] for seed in SEEDS])
        if not np.allclose(references, references[0], atol=1e-10, rtol=1e-10):
            raise ValueError(f"reference parameter differs across seeds for row {key}")
        grouped[groups[0]].append((
            float(references[0]),
            {seed: float(seed_rows[seed][key][parameter][1]) for seed in SEEDS},
        ))
    group_ids = np.asarray(sorted(grouped))
    reference = np.asarray([
        np.mean([row[0] for row in grouped[group]]) for group in group_ids
    ], dtype=np.float64)
    generated = {
        seed: np.asarray([
            np.mean([row[1][seed] for row in grouped[group]]) for group in group_ids
        ], dtype=np.float64)
        for seed in SEEDS
    }
    counts = np.asarray([len(grouped[group]) for group in group_ids], dtype=np.int64)
    return group_ids, reference, generated, counts


def _pearson(reference: np.ndarray, generated: np.ndarray) -> float | None:
    if len(reference) < 2 or np.std(reference) == 0 or np.std(generated) == 0:
        return None
    return float(stats.pearsonr(reference, generated).statistic)


def _seed_metrics(reference: np.ndarray, generated: np.ndarray) -> dict[str, float | None]:
    difference = generated - reference
    return {
        "reference_mean": float(np.mean(reference)),
        "generated_mean": float(np.mean(generated)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "bias": float(np.mean(difference)),
        "difference_sd": float(np.std(difference, ddof=1)) if len(difference) > 1 else None,
        "pearson_r": _pearson(reference, generated),
    }


def _bootstrap_ci(differences: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    # Bound peak memory for PTB-XL, where the random-window split contains
    # several thousand patient groups.
    means = np.empty(draws, dtype=np.float64)
    chunk = max(1, min(draws, 1_000_000 // len(differences)))
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = generator.integers(
            0, len(differences), size=(stop - start, len(differences))
        )
        means[start:stop] = np.mean(differences[indices], axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _paired_test(reference: np.ndarray, generated: np.ndarray, bootstrap_seed: int) -> dict[str, object]:
    difference = generated - reference
    n = len(difference)
    if n < 3 or np.all(difference == 0):
        return {"status": "insufficient_or_all_zero", "n_groups": n}
    if n <= 5000:
        normality = stats.shapiro(difference)
        normality_name = "Shapiro-Wilk"
    else:
        # SciPy warns that the Shapiro p-value is inaccurate above 5,000.
        normality = stats.normaltest(difference)
        normality_name = "D'Agostino-Pearson K2"
    ci_low, ci_high = _bootstrap_ci(difference, bootstrap_seed)
    common = {
        "status": "ok", "n_groups": n,
        "difference_definition": "three_seed_mean_generated_minus_reference",
        "mean_difference": float(np.mean(difference)),
        "difference_sd": float(np.std(difference, ddof=1)),
        "bootstrap_95_ci_lower": ci_low, "bootstrap_95_ci_upper": ci_high,
        "normality_test": normality_name, "normality_p_value": float(normality.pvalue),
    }
    if normality.pvalue >= 0.05:
        result = stats.ttest_rel(generated, reference)
        dz = float(np.mean(difference) / np.std(difference, ddof=1))
        return {**common, "test": "two-sided paired t-test", "raw_p_value": float(result.pvalue),
                "effect_name": "paired_Cohen_dz", "effect_value": dz}
    wilcoxon_method = "approx" if n > 50 else "auto"
    result = stats.wilcoxon(
        generated, reference, alternative="two-sided", method=wilcoxon_method
    )
    nonzero = difference[difference != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    rank_biserial = float(np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0])) / float(np.sum(ranks))
    raw_p = float(result.pvalue)
    extra: dict[str, object] = {"wilcoxon_method": wilcoxon_method}
    if wilcoxon_method == "approx":
        z = float(result.zstatistic)
        log_p = float(math.log(2.0) + stats.norm.logsf(abs(z)))
        log10_p = log_p / math.log(10.0)
        minimum = float(np.finfo(np.float64).tiny)
        if raw_p == 0.0:
            raw_p = minimum
            extra["p_value_underflow_capped"] = True
        extra.update({"wilcoxon_z": z, "raw_log10_p_value": log10_p})
    return {**common, "test": "two-sided Wilcoxon signed-rank", "raw_p_value": raw_p,
            "effect_name": "matched_pairs_rank_biserial", "effect_value": rank_biserial,
            **extra}


def _holm(rows: list[dict[str, object]]) -> None:
    eligible = [(index, float(row["raw_p_value"])) for index, row in enumerate(rows) if row.get("status") == "ok"]
    ordered = sorted(eligible, key=lambda item: item[1])
    running = 0.0
    m = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        rows[index]["holm_adjusted_p_value_dataset_11_parameters"] = running


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        dataset: {seed: getattr(args, f"{dataset}_seed{seed}").resolve() for seed in SEEDS}
        for dataset in DATASETS
    }
    per_seed: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    group_details: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(DATASETS):
        loaded = {seed: _load_rows(dataset, path) for seed, path in inputs[dataset].items()}
        dataset_tests = []
        for parameter_index, parameter in enumerate(PARAMETERS):
            groups, reference, generated, counts = _group_common_pairs(loaded, parameter)
            if not len(groups):
                raise ValueError(f"no common jointly measurable groups for {dataset}/{parameter}")
            metric_rows = []
            for seed in SEEDS:
                metrics = _seed_metrics(reference, generated[seed])
                row = {
                    "dataset": dataset, "parameter": parameter, "seed": seed,
                    "group_level": DATASET_SPECS[dataset]["group_level"],
                    "phase_mode": DATASET_SPECS[dataset]["phase"],
                    "n_groups": len(groups), "n_common_rows": int(np.sum(counts)),
                    "unit": _unit(parameter, DATASET_SPECS[dataset]["physical"]), **metrics,
                }
                per_seed.append(row); metric_rows.append(row)
            for metric in ("generated_mean", "mae", "rmse", "bias", "difference_sd", "pearson_r"):
                values = np.asarray([row[metric] for row in metric_rows if row[metric] is not None], dtype=np.float64)
                summaries.append({
                    "dataset": dataset, "parameter": parameter, "metric": metric,
                    "n_training_seeds": len(values), "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "seed31": metric_rows[0][metric], "seed32": metric_rows[1][metric],
                    "seed33": metric_rows[2][metric], "unit": metric_rows[0]["unit"],
                    "phase_mode": DATASET_SPECS[dataset]["phase"],
                })
            mean_generated = np.mean(np.stack([generated[seed] for seed in SEEDS]), axis=0)
            test = {
                "dataset": dataset, "parameter": parameter,
                "group_level": DATASET_SPECS[dataset]["group_level"],
                "phase_mode": DATASET_SPECS[dataset]["phase"],
                "unit": metric_rows[0]["unit"],
                **_paired_test(reference, mean_generated, 31000 + dataset_index * 100 + parameter_index),
            }
            dataset_tests.append(test)
            for group, real, count, values in zip(
                groups, reference, counts, zip(*(generated[seed] for seed in SEEDS))
            ):
                group_details.append({
                    "dataset": dataset, "parameter": parameter, "group_id": group,
                    "common_rows": int(count), "reference_mean": float(real),
                    "seed31_generated_mean": float(values[0]),
                    "seed32_generated_mean": float(values[1]),
                    "seed33_generated_mean": float(values[2]),
                    "three_seed_generated_mean": float(np.mean(values)),
                    "difference": float(np.mean(values) - real),
                })
        _holm(dataset_tests)
        tests.extend(dataset_tests)

    paths = {
        "per_seed": output / "per_seed_ecg_parameter_metrics.csv",
        "summary": output / "ecg_parameter_mean_sd.csv",
        "tests": output / "group_paired_tests.csv",
        "groups": output / "per_group_three_seed_parameters.csv",
    }
    for key, rows in (("per_seed", per_seed), ("summary", summaries), ("tests", tests), ("groups", group_details)):
        _write_csv(paths[key], rows)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1, "status": "completed", "model": "RCFM-OT",
        "training_seeds": list(SEEDS), "mean_sd_definition": "sample SD across training seeds (ddof=1)",
        "paired_test_definition": (
            "Group-level reference versus the mean generated parameter across seeds 31/32/33; "
            "paired t-test when the prespecified normality test p>=0.05, otherwise Wilcoxon "
            "signed-rank; Shapiro-Wilk is used through n=5000 and D'Agostino-Pearson K2 above it"
        ),
        "multiplicity": "Holm adjustment separately within each dataset across 11 ECG parameters",
        "phase_policy": {dataset: DATASET_SPECS[dataset]["phase"] for dataset in DATASETS},
        "claim_boundary": (
            "MIMIC-AFib, WESAD, and mmECG use target-informed oracle phase correction. "
            "Random-window overlap limitations remain; normalized amplitudes are not physical units. "
            "These tests assess RCFM-OT generated-versus-reference parameter bias, not a pairwise "
            "RCFM-OT-versus-baseline advantage."
        ),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1, "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "command": shlex.join(sys.argv),
        "inputs": {
            f"{dataset}_seed{seed}": {
                "path": str(path),
                "parameter_csv_sha256": _sha256(path / (
                    "per_window_parameters.csv" if dataset in {"ptbxl", "cpsc2018"}
                    else "per_window_ecg_parameters.csv"
                )),
                "protocol_sha256": _sha256(path / "protocol.json"),
            }
            for dataset, seed_paths in inputs.items() for seed, path in seed_paths.items()
        },
        "outputs": {path.name: _sha256(path) for path in (*paths.values(), summary_path)},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        for seed in SEEDS:
            parser.add_argument(f"--{dataset}_seed{seed}", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
