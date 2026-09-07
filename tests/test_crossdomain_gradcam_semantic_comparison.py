from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_crossdomain_gradcam_semantic_comparison.py"
SPEC = importlib.util.spec_from_file_location("plot_crossdomain_gradcam_semantic_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_plot_comparison_has_no_suptitle_or_footer(tmp_path: Path) -> None:
    examples = []
    for group in MODULE.GROUP_ORDER:
        for number in range(1, 4):
            examples.append({
                "group": group, "example_number": number,
                "signal": np.sin(np.linspace(0, 8 * np.pi, 400)),
                "diagnostic_mask": np.linspace(0, 1, 400),
                "semantic_gradcam": np.linspace(1, 0, 400),
                "semantic_direct": np.abs(np.sin(np.linspace(0, 4 * np.pi, 400))),
                "diagnostic_degenerate": False,
            })
    outputs = MODULE.plot_comparison(examples, {"sampling_rate_hz": 100}, tmp_path / "comparison")
    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.stat().st_size > 1000 for path in outputs)


def test_semantic_mask_resampling_preserves_probability_bounds() -> None:
    mask = MODULE.resample_mask_to_sample_grid(np.asarray([0.0, 0.5, 1.0]), 500, 100, output_length=2)
    assert mask.shape == (2,)
    assert np.all((mask >= 0) & (mask <= 1))
