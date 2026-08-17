from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_five_dataset_paper_statistics import (
    DEFAULT_DATASETS,
    build_argparser,
    _validate_manifest_coverage,
)


def test_manifest_coverage_requires_complete_declared_test_split(tmp_path: Path):
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        '{"test_subject_ids":["S1","S2"],"test_windows":3,"split_method":"group"}',
        encoding="utf-8",
    )
    data = {"groups": np.asarray(["S1", "S1", "S2"])}
    result = _validate_manifest_coverage(data, manifest, ("S1", "S2"))
    assert result["status"] == "complete_frozen_test_split"
    assert result["not_an_analysis_subset"] is True


def test_manifest_coverage_rejects_subject_subset(tmp_path: Path):
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        '{"test_subject_ids":["S1","S2"],"test_windows":3,"split_method":"group"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage differs"):
        _validate_manifest_coverage({"groups": np.asarray(["S1", "S1"])}, manifest, ("S1", "S2"))


def test_default_entry_skips_mimic():
    args = build_argparser().parse_args(["--workspace", "/tmp", "--output_dir", "/tmp/out"])
    assert tuple(args.datasets) == DEFAULT_DATASETS
    assert "MIMIC-AFib" not in args.datasets
