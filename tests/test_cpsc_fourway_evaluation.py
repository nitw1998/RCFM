import json

import pytest

from scripts.evaluate_cpsc_fourway import (
    EXPECTED_SPLIT_HASH,
    _validate_manifest,
)


def test_cpsc_manifest_is_bound_by_schema_split_and_validation_count(tmp_path):
    path = tmp_path / "dataset_manifest.json"
    payload = {
        "schema_version": 2,
        "split_hash": EXPECTED_SPLIT_HASH,
        "splits": {"val": {"records": 686}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_manifest(path, 686)["splits"]["val"]["records"] == 686
    payload["splits"]["val"]["records"] = 685
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record count"):
        _validate_manifest(path, 686)
