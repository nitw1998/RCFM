import json
from pathlib import Path

import pytest

from train_rcfm import parse_args_with_config as parse_flow_config
from train_rddm_compare import parse_args_with_config as parse_rddm_config


ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    "ptbxl": {
        "suffix": "random_window80_20_joint12_seed31.yaml",
        "baseline": "cfm_compare_random_window80_20_joint12_seed31.yaml",
        "version": "ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2",
        "normalization": "source_record_joint12_minmax_neg1_1_v1",
        "counts": (34939, 8735),
        "heldout_split": "val",
    },
    "cpsc2018": {
        "suffix": "random_window80_20_joint12_seed31.yaml",
        "baseline": "cfm_compare_random_window80_20_joint12_seed31.yaml",
        "version": "cpsc2018-source-fullrecord-joint12-minmax-all-nonoverlap4s-random80-20-record-overlap-v1",
        "normalization": "source_record_joint12_minmax_neg1_1_v1",
        "counts": (19364, 4842),
        "heldout_split": "val",
    },
    "mimic_afib": {
        "suffix": "random_window80_20_seed31.yaml",
        "version": "mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1",
        "normalization": "rddm_window_minmax_neg1_1_v1",
        "counts": (8160, 2040),
        "heldout_split": "test",
    },
    "wesad": {
        "suffix": "random_window80_20_record_minmax_seed31.yaml",
        "version": "wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2",
        "normalization": "source_record_minmax_neg1_1_v1",
        "counts": (17365, 4342),
        "heldout_split": "test",
    },
    "mmecg": {
        "suffix": "random_window80_20_seed31.yaml",
        "version": "mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1",
        "normalization": "window_minmax_neg1_1_v1",
        "counts": (9973, 2494),
        "heldout_split": "test",
    },
}


@pytest.mark.parametrize("dataset", CELLS)
def test_flow_comparator_configs_match_random_window_cfm(dataset):
    spec = CELLS[dataset]
    baseline_name = spec.get("baseline", f"cfm_{spec['suffix']}")
    baseline = json.loads((ROOT / "configs" / dataset / baseline_name).read_text())
    for variant, use_ot in (("rcfm", False), ("rcfm_ot", True)):
        path = ROOT / "configs" / dataset / f"{variant}_{spec['suffix']}"
        args = parse_flow_config(["--config", str(path)])
        assert args.dataset_version == baseline["dataset_version"] == spec["version"]
        assert args.split_hash == baseline["split_hash"]
        assert args.normalization_id == baseline["normalization_id"] == spec["normalization"]
        assert (args.epochs, args.batch_size, args.seed) == (200, 128, 31)
        assert args.region_weight == 0.01
        assert args.use_minibatch_ot is use_ot
        assert args.ot_method == "exact"
        assert args.ot_sampling_strategy == "assignment"


@pytest.mark.parametrize("dataset", CELLS)
def test_rddm_random_window_configs_are_strictly_accepted(dataset):
    spec = CELLS[dataset]
    path = ROOT / "configs" / dataset / f"rddm_{spec['suffix']}"
    args = parse_rddm_config(["--config", str(path)])
    assert args.dataset_version == spec["version"]
    assert args.normalization_id == spec["normalization"]
    assert (args.expected_train_windows, args.expected_test_windows) == spec["counts"]
    assert args.heldout_split == spec["heldout_split"]
    assert (args.epochs, args.batch_size, args.seed) == (200, 128, 31)
    assert args.nT == 10


def test_unified_launcher_exposes_exactly_the_requested_cells():
    script = (ROOT / "scripts" / "train_random_window80_20_comparator.sh").read_text()
    assert "ptbxl|cpsc2018|mimic-afib|wesad|mmecg" in script
    assert "rcfm|rcfm-ot|rddm" in script
    assert 'VARIANT="cfm"' not in script


def test_wesad_mmecg_rcfm_ot_e500_resume_launcher_is_checkpoint_pinned():
    script = (ROOT / "scripts" / "resume_random_window_rcfm_ot_to_e500.sh").read_text()
    assert "wesad|mmecg" in script
    assert "--epochs 500" in script
    assert "--resume_lr_policy restart_cosine" in script
    assert "--resume_restart_lr 1e-5" in script
    assert "checkpoint[\"epoch\"] == 200" in script
    assert "checkpoint[\"optimizer_state\"][\"param_groups\"][0][\"lr\"] == 0.0" in script
    assert "29b74f64c8880365813f450bd7bf15bced2bd38611367a14ab91fcdbe22f7809" in script
    assert "023e33132dadd8bbb8e346f7c848fdf9b4900d836e2c78b381880e1806f4ce54" in script
