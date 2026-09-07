from pathlib import Path

import pytest

from scripts.analyze_multilead_rddm_multiseed import _parse_predictions


def test_rddm_result_specs_require_exact_three_seed_set():
    values = [f"{seed}=/tmp/rddm_s{seed}" for seed in (31, 32, 33)]
    parsed = _parse_predictions(values)
    assert parsed[33] == Path("/tmp/rddm_s33")
    with pytest.raises(ValueError, match="exactly training seeds"):
        _parse_predictions(values[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_predictions(values + ["33=/tmp/duplicate"])
