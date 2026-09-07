"""Plot separate window- and subject-level MIMIC comparator B--A figures."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_crossmodal_cfm_bland_altman_levels import (
    _plot_dataset,
    _sha256,
    _write_csv,
    compute_views,
    load_mimic,
)


MODELS = {
    "rcfm": "RCFM",
    "rcfm_ot": "RCFM-OT",
    "rddm": "RDDM",
}


def run(args: argparse.Namespace) -> Path:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "rcfm": args.rcfm_dir.resolve(),
        "rcfm_ot": args.rcfm_ot_dir.resolve(),
        "rddm": args.rddm_dir.resolve(),
    }
    generated: list[Path] = []
    all_rows: list[dict[str, object]] = []
    counts: dict[str, object] = {}
    for model, directory in inputs.items():
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if protocol.get("status") != "completed" or summary.get("model") != MODELS[model]:
            raise ValueError(f"invalid completed clinical input for {model}: {directory}")
        rows = load_mimic(directory)
        identities = {(str(row["phase"]), str(row["window_id"])) for row in rows}
        if len(identities) != len(rows):
            raise ValueError(f"duplicate phase/window identities for {model}")
        views, summaries = compute_views(rows, "MIMIC-AFib")
        for row in summaries:
            row["model"] = model
            row["model_label"] = MODELS[model]
        all_rows.extend(summaries)
        counts[model] = {
            "windows_per_phase": {
                phase: sum(row["phase"] == phase for row in rows)
                for phase in views["window"]
            },
            "subjects": len({str(row["subject_id"]) for row in rows}),
        }
        for level in ("window", "subject"):
            stem = output / f"mimic_afib_{model}_{level}_level_bland_altman"
            _plot_dataset("MIMIC-AFib", views, level, stem, model_label=MODELS[model])
            generated.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    csv_path = output / "bland_altman_level_summary.csv"
    _write_csv(csv_path, all_rows)
    generated.append(csv_path)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "difference_definition": "generated_minus_reference",
                "counts": counts,
                "rows": all_rows,
                "window_level": "all finite paired windows; correlated repeated measurements",
                "subject_level": "within-subject paired means before Bland-Altman analysis",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated.append(summary_path)
    protocol = {
        "schema_version": 1,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "method": {
            "window_level": "all phase-specific finite pairs without downsampling",
            "subject_level": "arithmetic paired means within subject",
            "limits": "bias +/- 1.96 sample SD of generated-minus-reference differences",
        },
        "claim_boundary": "Window observations are correlated; all 34 subjects occur in training and validation. Oracle alignment is target-informed and not deployable.",
        "execution": {"python": platform.python_version(), "script_sha256": _sha256(Path(__file__))},
        "inputs": {
            model: {
                "directory": str(directory),
                "protocol_sha256": _sha256(directory / "protocol.json"),
                "clinical_csv_sha256": _sha256(directory / "per_window_ecg_parameters.csv"),
            }
            for model, directory in inputs.items()
        },
        "outputs": {path.name: _sha256(path) for path in generated},
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcfm_dir", type=Path, required=True)
    parser.add_argument("--rcfm_ot_dir", type=Path, required=True)
    parser.add_argument("--rddm_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
