import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_ptbxl_ecgmamba_taskhead_masks import (
    DIAG_METHOD,
    SEMANTIC_METHOD,
    _positive_targets,
    _project,
)


def test_taskhead_mask_methods_are_distinct_and_configs_match():
    root = Path(__file__).resolve().parents[1]
    diag = json.loads(
        (root / "configs/ptbxl/rcfm_ecgmamba_diaghead_gradcam_no_ot_seed31.yaml").read_text()
    )
    semantic = json.loads(
        (root / "configs/ptbxl/rcfm_ecgmamba_semantichead_gradcam_no_ot_seed31.yaml").read_text()
    )
    assert diag["mask_method"] == DIAG_METHOD
    assert semantic["mask_method"] == SEMANTIC_METHOD
    for config in (diag, semantic):
        assert config["region_weight"] == 0.01
        assert config["use_minibatch_ot"] is False
        assert config["seed"] == 31
        assert config["batch_size"] == 128
        assert config["epochs"] == 500


def test_positive_targets_do_not_fallback_for_unlabelled_records():
    metadata = pd.DataFrame(
        {"ecg_id": [10, 11], "scp_codes": ["{'NORM': 100.0}", "{'AFIB': 100.0}"]}
    ).set_index("ecg_id")
    assert _positive_targets(metadata, np.array([10, 11]), ["NORM", "LVH"]) == [(0,), ()]


def test_full_resolution_cam_projection_is_finite_and_aligned():
    values = np.zeros(5000, dtype=np.float32)
    values[500:750] = np.linspace(0, 1, 250)
    mask, degenerate = _project(values)
    assert mask.shape == (512,)
    assert np.isfinite(mask).all()
    assert mask.min() == 0 and mask.max() == 1
    assert not degenerate
