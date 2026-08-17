import csv
from pathlib import Path

import numpy as np

from scripts.evaluate_delineation_unet import (
    _classification_metrics,
    _merge_training_metrics,
    _select_visualization_indices,
)


def test_region_classification_metrics_are_exact():
    metrics = _classification_metrics(tp=8, fp=2, fn=4, valid=20)
    assert metrics["dice"] == 16 / 22
    assert metrics["iou"] == 8 / 14
    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 8 / 12
    assert metrics["valid_samples"] == 20


def test_visualization_indices_are_preselected_from_all_wave_validity():
    class Dataset:
        samples = np.asarray([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
        wave_valid = np.asarray([[[1, 1, 1], [1, 0, 1]], [[1, 1, 1], [1, 1, 1]]])

    assert _select_visualization_indices(Dataset(), 3) == [0, 2, 3]


def test_training_metrics_merge_continues_across_resume_files(tmp_path: Path):
    paths = []
    for name, rows in (
        ("initial.csv", [(1, "val/region_macro_dice", 0.7), (2, "val/region_macro_dice", 0.8)]),
        ("resume.csv", [(2, "val/region_macro_dice", 0.81), (3, "val/region_macro_dice", 0.82)]),
    ):
        path = tmp_path / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "global_step", "split", "metric", "value"])
            writer.writeheader()
            for epoch, metric, value in rows:
                writer.writerow({"epoch": epoch, "global_step": epoch, "split": "val", "metric": metric, "value": value})
        paths.append(path)

    merged = _merge_training_metrics(paths)
    assert merged["val/region_macro_dice"] == {1: 0.7, 2: 0.81, 3: 0.82}
