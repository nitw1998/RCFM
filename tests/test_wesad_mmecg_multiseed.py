from pathlib import Path

import pytest

from scripts.train_cat import parse_args_with_config as parse_cat
from train_cfm_compare import parse_args_with_config as parse_cfm
from train_cfm_ot import parse_args_with_config as parse_cfm_ot
from train_direct_cnn import parse_args_with_config as parse_direct
from train_rcfm import parse_args_with_config as parse_rcfm
from train_rddm_compare import parse_args_with_config as parse_rddm


REPO = Path(__file__).resolve().parents[1]
CONTRACTS = {
    "wesad": {
        "dataset": "WESAD",
        "task": "ppg2ecg",
        "version": "wesad-subject-fold1-linear-resample-window-minmax-v1",
        "split": "0b90bffe7c3032243c803e534de3592618284d26792c2b612ce7af17a81a85cd",
    },
    "mmecg": {
        "dataset": "mmECG",
        "task": "rcg2ecg",
        "version": "mmecg-public-20221108-subject-split-window-minmax-v1",
        "split": "e26fc81121cfd3b0e457608e37a7aa496ac7c48a0e0e7583a465bd069cc9da9f",
    },
}


@pytest.mark.parametrize("dataset_key", tuple(CONTRACTS))
@pytest.mark.parametrize("seed", (32, 33))
def test_all_wesad_mmecg_additional_seed_configs(dataset_key: str, seed: int) -> None:
    root = REPO / "configs" / dataset_key
    rddm_name = (
        "rddm_matched_window_minmax_seed31.yaml"
        if dataset_key == "wesad"
        else "rddm_adapted_window_minmax_seed31.yaml"
    )
    cat_name = (
        "cat_ppg_reproduced_window_minmax_seed31.yaml"
        if dataset_key == "wesad"
        else "cat_rcg_adapted_window_minmax_seed31.yaml"
    )
    variants = (
        parse_cfm(["--config", str(root / "cfm_window_minmax_no_ot_seed31.yaml"), "--seed", str(seed)]),
        parse_cfm_ot(["--config", str(root / "cfm_ot_window_minmax_seed31.yaml"), "--seed", str(seed)]),
        parse_rcfm(["--config", str(root / "rcfm_window_minmax_no_ot_seed31.yaml"), "--seed", str(seed)]),
        parse_rcfm(["--config", str(root / "rcfm_window_minmax_exact_ot_seed31.yaml"), "--seed", str(seed)]),
        parse_rddm(["--config", str(root / rddm_name), "--seed", str(seed)]),
        parse_cat(["--config", str(root / cat_name), "--seed", str(seed)]),
        parse_direct(["--config", str(root / "direct_cnn_regression_window_minmax_seed31.yaml"), "--seed", str(seed)]),
    )
    expected = CONTRACTS[dataset_key]
    for args in variants:
        assert args.datasets == expected["dataset"]
        assert args.task == expected["task"]
        assert args.dataset_version == expected["version"]
        assert args.split_hash == expected["split"]
        assert args.normalization_id == "window_minmax_neg1_1_v1"
        assert args.seed == seed
        assert args.epochs == 500
        assert args.batch_size == 128


def test_public_and_local_multiseed_entries_are_path_safe_and_complete() -> None:
    launcher = (REPO / "scripts/launch_wesad_mmecg_multiseed.sh").read_text()
    worker = (REPO / "scripts/run_wesad_mmecg_multiseed_worker.sh").read_text()
    for source in (launcher, worker):
        assert "/data/user" not in source
        assert "/home/user" not in source
        for variant in ("cfm", "cfm_ot", "rcfm", "rcfm_ot", "rddm", "cat", "direct_cnn"):
            assert variant in source
    assert "WESAD_DATA_ROOT" in launcher and "MMECG_DATA_ROOT" in launcher
    assert 'ENTRY="train_rddm_compare.py"' in worker
    assert 'ENTRY="scripts/train_cat.py"' in worker
    assert 'ENTRY="train_direct_cnn.py"' in worker
    assert "Formal additional seeds are restricted to 32 or 33" in worker
