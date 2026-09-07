import numpy as np
import pytest

from scripts.recalculate_legacy_batch_fd import (
    batch_fd_rows,
    literal_fd,
    low_rank_literal_fd,
)
from src.rcfm.metrics.waveform import waveform_frechet_distance


def test_literal_product_formula_matches_stable_bures_trace():
    generator = np.random.default_rng(9)
    reference = generator.normal(size=(40, 8))
    generated = reference * 0.8 + generator.normal(scale=0.1, size=reference.shape)
    assert literal_fd(reference, generated)["fd"] == pytest.approx(
        waveform_frechet_distance(reference, generated), abs=1e-10
    )


def test_batch_fd_pools_channels_as_independent_waveform_observations():
    generator = np.random.default_rng(11)
    reference = generator.normal(size=(5, 3, 8))
    generated = reference + 0.2
    rows = batch_fd_rows(reference, generated, batch_size=4)
    assert [row["records"] for row in rows] == [4]
    assert rows[0]["pooled_channel_observations"] == 12
    expected = literal_fd(reference[:4].reshape(-1, 8), generated[:4].reshape(-1, 8))["fd"]
    assert rows[0]["fd"] == pytest.approx(expected)


def test_low_rank_observation_space_formula_matches_literal_product():
    generator = np.random.default_rng(13)
    reference = generator.normal(size=(12, 24))
    generated = reference * 0.7 + generator.normal(scale=0.2, size=reference.shape)
    assert low_rank_literal_fd(reference, generated) == pytest.approx(
        literal_fd(reference, generated)["fd"], abs=1e-6
    )
