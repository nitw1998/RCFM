"""Export PTB-XL per-lead waveform metrics from a completed evaluation summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_ptbxl_fourway import TARGET_LEADS


FIELDS = (
    "model",
    "lead",
    "rmse",
    "mae",
    "bias",
    "waveform_fd",
    "per_record_pearson_mean",
    "per_record_pearson_median",
    "overall_rmse_all_11_leads",
    "rmse_aggregation",
)


def rows_from_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    models = summary.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("waveform summary must contain completed model metrics")
    rows = []
    for model, metrics in models.items():
        per_lead = metrics.get("per_lead")
        if not isinstance(per_lead, dict) or set(per_lead) != set(TARGET_LEADS):
            raise ValueError(f"{model} does not contain the frozen 11 target leads")
        for lead in TARGET_LEADS:
            values = per_lead[lead]
            rows.append(
                {
                    "model": model,
                    "lead": lead,
                    "rmse": values["rmse"],
                    "mae": values["mae"],
                    "bias": values["bias"],
                    "waveform_fd": values["waveform_fd"],
                    "per_record_pearson_mean": values["per_record_pearson_mean"],
                    "per_record_pearson_median": values["per_record_pearson_median"],
                    "overall_rmse_all_11_leads": metrics["rmse"],
                    "rmse_aggregation": "per_lead_root_mean_square_over_all_records_and_time_samples",
                }
            )
    return rows


def run(input_path: Path, output_path: Path) -> Path:
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    rows = rows_from_summary(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(run(args.input.resolve(), args.output.resolve()))
