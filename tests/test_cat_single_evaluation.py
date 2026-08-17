from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_cat_single_clinical import EXPECTED, _write_csv
from scripts.evaluate_cat_single_dataset import CONTRACTS, _model


def test_single_cat_contracts_exclude_invalid_multilead_adapters():
    assert set(CONTRACTS) == {"WESAD", "mmECG"}
    assert all(contract["records"] == EXPECTED[name] for name, contract in CONTRACTS.items())
    assert all("CPSC" not in name and "PTB" not in name for name in CONTRACTS)


def test_single_cat_model_is_one_output_and_source_only():
    config = {"window_size": 4, "sampling_rate": 128, "cat_layers": 2, "top_k": 2,
              "patch_width": 8, "d_model": 16, "n_heads": 4,
              "encoder_layers": 1, "ff_dim": 32, "dropout": 0.0}
    model = _model(config)
    assert model.output_channels == 1
    with pytest.raises(TypeError):
        model(np.zeros((1, 1, 512), dtype=np.float32), target=None)


def test_clinical_csv_union_columns(tmp_path: Path):
    path = tmp_path / "rows.csv"
    _write_csv(path, [{"a": 1}, {"a": 2, "b": 3}])
    assert path.read_text(encoding="utf-8").splitlines()[0] == "a,b"
