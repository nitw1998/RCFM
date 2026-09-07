#!/usr/bin/env python3
"""Create aligned PTB-XL binary labels from record IDs and SCP metadata."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    record_ids = np.load(args.record_ids.resolve(), allow_pickle=False).astype(np.int64)
    metadata = pd.read_csv(args.metadata.resolve(), usecols=["ecg_id", "strat_fold", "scp_codes"])
    if metadata.ecg_id.duplicated().any():
        raise ValueError("PTB-XL metadata contains duplicate ECG IDs")
    metadata = metadata.set_index("ecg_id")
    missing = sorted(set(record_ids.tolist()) - set(metadata.index.astype(int).tolist()))
    if missing:
        raise ValueError(f"record IDs are missing from PTB-XL metadata: {missing[:10]}")
    codes = tuple(dict.fromkeys(args.positive_code))
    labels = np.zeros(len(record_ids), dtype=np.uint8)
    folds = np.empty(len(record_ids), dtype=np.int16)
    per_code = {code: 0 for code in codes}
    for index, record_id in enumerate(record_ids):
        row = metadata.loc[int(record_id)]
        statements = ast.literal_eval(str(row.scp_codes))
        if not isinstance(statements, dict):
            raise ValueError("scp_codes did not decode to a dictionary")
        present = [code for code in codes if code in statements]
        labels[index] = bool(present)
        folds[index] = int(row.strat_fold)
        for code in present:
            per_code[code] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, labels, allow_pickle=False)
    summary = {
        "status": "completed",
        "positive_codes_union": list(codes),
        "records": int(len(labels)),
        "positive_records": int(labels.sum()),
        "negative_records": int(len(labels) - labels.sum()),
        "per_code_records": per_code,
        "strat_fold_counts": {
            str(fold): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds).tolist())
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--record_ids", type=Path, required=True)
    parser.add_argument("--positive_code", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_argparser().parse_args())
