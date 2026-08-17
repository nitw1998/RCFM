import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data import PairedSignalDataset, get_ppg2ecg_datasets
from scripts.prepare_mimic_gradcam_masks import _mask_for_record
from train_rcfm import _load_external_region_masks


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cached_region_masks_are_validated_and_returned_without_heldout_leakage(tmp_path):
    dataset_root = tmp_path / "synthetic"
    dataset_root.mkdir()
    ramp = np.arange(128, dtype=np.float32)
    for split in ("train", "test"):
        np.save(dataset_root / f"ecg_{split}_1sec.npy", np.stack([ramp, ramp + 1]))
        np.save(dataset_root / f"ppg_{split}_1sec.npy", np.stack([ramp + 2, ramp + 3]))
    masks = np.stack(
        [np.linspace(0, 1, 128, dtype=np.float32), np.linspace(1, 0, 128, dtype=np.float32)]
    )[:, None, :]

    train_set, heldout_set = get_ppg2ecg_datasets(
        DATA_PATH=str(tmp_path),
        datasets=["synthetic"],
        window_size=1,
        normalization_id="window_minmax_neg1_1_v1",
        region_masks_train=masks,
        max_train_records=1,
    )

    assert len(train_set) == 1
    np.testing.assert_array_equal(train_set[0][2], masks[0])
    assert len(heldout_set[0]) == 2


@pytest.mark.parametrize(
    "masks, message",
    [
        (np.zeros((2, 2, 8), dtype=np.float32), "expected"),
        (np.full((2, 1, 8), np.nan, dtype=np.float32), "finite"),
        (np.full((2, 1, 8), 1.1, dtype=np.float32), r"\[0, 1\]"),
    ],
)
def test_cached_region_masks_reject_invalid_arrays(masks, message):
    with pytest.raises(ValueError, match=message):
        PairedSignalDataset(
            np.zeros((2, 8), dtype=np.float32),
            np.ones((2, 8), dtype=np.float32),
            cached_region_masks=masks,
        )


def test_external_manifest_binds_dataset_split_source_and_mask(tmp_path):
    dataset_root = tmp_path / "MIMIC-AFib"
    dataset_root.mkdir()
    source_path = dataset_root / "ecg_train_4sec.npy"
    np.save(source_path, np.arange(1024, dtype=np.float32).reshape(2, 512))
    mask_path = tmp_path / "region_masks_train.npy"
    np.save(mask_path, np.zeros((2, 1, 512), dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "dataset": {
            "dataset_version": "version-1",
            "split_hash": "split-1",
            "train_records": 2,
            "source_ecg": {"file_name": source_path.name, "sha256": _file_sha256(source_path)},
        },
        "mask": {
            "method": "gradcam-test",
            "file_name": mask_path.name,
            "shape": [2, 1, 512],
            "dtype": "float32",
            "sha256": _file_sha256(mask_path),
        },
        "test_mask_generated": False,
        "claim_boundary": "synthetic test",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    masks, provenance = _load_external_region_masks(
        mask_path=str(mask_path),
        manifest_path=str(manifest_path),
        mask_method="gradcam-test",
        data_root=str(tmp_path),
        dataset_name="MIMIC-AFib",
        dataset_version="version-1",
        split_hash="split-1",
        window_size=4,
    )

    assert masks.shape == (2, 1, 512)
    assert provenance["test_mask_generated"] is False
    manifest["dataset"]["split_hash"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split_hash"):
        _load_external_region_masks(
            mask_path=str(mask_path), manifest_path=str(manifest_path),
            mask_method="gradcam-test", data_root=str(tmp_path),
            dataset_name="MIMIC-AFib",
            dataset_version="version-1", split_hash="split-1", window_size=4,
        )


class _TinyGradCamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(12, 4, kernel_size=8, stride=8, bias=False),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(4, 71, bias=False)
        torch.nn.init.constant_(self.features[0].weight, 0.05)
        torch.nn.init.constant_(self.classifier.weight, 0.1)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).mean(dim=-1))


def test_gradcam_record_projection_produces_finite_512_sample_soft_mask():
    model = _TinyGradCamModel().eval()
    signal = 2.0 + np.sin(np.linspace(0, 12 * np.pi, 512, dtype=np.float32))

    mask, degenerate = _mask_for_record(
        model, model.features[1], signal, scaler_mean=0.0, scaler_scale=1.0,
        device=torch.device("cpu"),
    )

    assert mask.shape == (512,)
    assert np.isfinite(mask).all()
    assert 0 <= mask.min() <= mask.max() <= 1
    assert degenerate is False


def test_gradcam_rcfm_ot_config_freezes_training_contract():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "mimic_afib"
         / "rcfm_ot_xresnet_gradcam_l5_seed31.yaml").read_text(encoding="utf-8")
    )
    assert config["mask_method"] == "xresnet1d101_afib_gradcam_l5_sample_center_v1"
    assert config["use_minibatch_ot"] is True
    assert config["ot_method"] == "exact"
    assert config["ot_sampling_strategy"] == "assignment"
    assert config["batch_size"] == 128
    assert config["epochs"] == 500
    assert config["save_every"] == 25
    assert config["validation_interval_epochs"] == 500
