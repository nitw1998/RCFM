import csv

import numpy as np

from scripts.analyze_ptbxl_ot_training_dynamics import (
    _first_epoch_at_or_below,
    _read_long_metric,
)


def test_read_long_metric_sorts_and_filters(tmp_path):
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "global_step", "metric", "value"))
        writer.writeheader()
        writer.writerows(
            (
                {"epoch": 50, "global_step": 2, "metric": "val/rmse", "value": 0.34},
                {"epoch": 25, "global_step": 1, "metric": "val/mae", "value": 0.20},
                {"epoch": 25, "global_step": 1, "metric": "val/rmse", "value": 0.36},
            )
        )
    epochs, values = _read_long_metric(path, "val/rmse")
    np.testing.assert_array_equal(epochs, [25, 50])
    np.testing.assert_allclose(values, [0.36, 0.34])


def test_first_threshold_epoch_is_not_post_selected():
    epochs = np.asarray([25, 50, 75, 100])
    values = np.asarray([0.40, 0.36, 0.37, 0.34])
    assert _first_epoch_at_or_below(epochs, values, 0.38) == 50
    assert _first_epoch_at_or_below(epochs, values, 0.35) == 100
    assert _first_epoch_at_or_below(epochs, values, 0.30) is None
