"""Run the frozen PTB-XL clinical agreement protocol for path/coupling ablations."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import evaluate_ptbxl_clinical_agreement as clinical
from scripts.evaluate_ptbxl_path_ablation import MODEL_ORDER


clinical.MODEL_ORDER = MODEL_ORDER
clinical.MODEL_LABELS.update(
    {
        "conditional_ot": "Conditional-OT",
        "vp": "VP",
        "target": "Target",
        "sb": "SB",
    }
)
clinical.MODEL_COLORS.update(
    {
        "conditional_ot": "#2878b5",
        "vp": "#2f8f5b",
        "target": "#c43d4b",
        "sb": "#d17a00",
    }
)


if __name__ == "__main__":
    output = clinical.run(clinical.build_argparser().parse_args())
    print(f"PTB-XL path/coupling clinical agreement saved to {output}")
