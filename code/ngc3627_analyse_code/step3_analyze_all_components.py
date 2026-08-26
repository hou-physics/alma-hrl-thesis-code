"""NGC 3627 sensitivity-check rerun with mask_components_mode='all'.

Loads canonical config and overrides only the component-selection mode.
Output → results/NGC3627_all_components/ to keep the original "center" run
intact for comparison.

Motivation: NGC 3627 (M66) is a PHANGS face-on barred spiral with prominent
spiral arms + bar; its CO morphology is highly distributed (clear arms +
bar visible in Panel C) but the default "center" mode + scan-winning
sthresh=32σ collapses the mask to ~60 beams at the nucleus, ignoring all
arm structure. Toma + Ada both reported H30α detections — a fair test of
whether the disk-mode mask captures additional spiral-arm signal.
"""
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "_base_cfg", str(Path(__file__).parent / "step3_analyze.py"))
_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_base)

from analyze import analyze

config = replace(
    _base.config,
    mask_components_mode="all",
    output_dir="/Volumes/HouAstro/master/results/NGC3627_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
