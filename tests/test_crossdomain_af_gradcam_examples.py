from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_crossdomain_af_gradcam_examples.py"
SPEC = importlib.util.spec_from_file_location("plot_crossdomain_af_gradcam_examples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_group_examples_is_record_stratified_and_score_free() -> None:
    groups = np.asarray(["af_a"] * 3 + ["af_b"] * 2 + ["af_c"] * 4 + ["n_a"] * 2 + ["n_b"] * 3 + ["n_c"] * 2)
    labels = np.asarray([1] * 9 + [0] * 7)
    selected = MODULE.select_group_examples(groups, labels, 3)
    assert selected == [1, 4, 7, 10, 12, 15]


def test_plot_examples_writes_reference_style_outputs(tmp_path: Path) -> None:
    examples = []
    for group in MODULE.GROUP_ORDER:
        for number in range(1, 4):
            examples.append({
                "group": group, "example_number": number,
                "signal": np.sin(np.linspace(0, 8 * np.pi, 400)),
                "mask": np.linspace(0, 1, 400), "probability": 0.75,
                "degenerate": False,
            })
    outputs = MODULE.plot_examples(
        examples, {"sampling_rate_hz": 100}, tmp_path / "examples", "Frozen Grad-CAM examples"
    )
    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.stat().st_size > 1000 for path in outputs)
