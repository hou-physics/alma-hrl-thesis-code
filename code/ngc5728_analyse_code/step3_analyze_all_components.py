"""NGC 5728 sensitivity-check rerun with mask_components_mode='all'.

Loads canonical config and overrides only the component-selection mode.
Output → results/NGC5728_all_components/ to keep the original "center" run intact.

Motivation: NGC 5728 is a Sy2 with prominent circumnuclear ring + extended bar.
fixed-env Nbeam=135.8 vs scan Nbeam=1.9 (factor 71×) — the most dramatic
center-vs-distributed mismatch among the four. Default center-anchored picks
a 1.9-beam knot; mode='all' lets the ring + bar components contribute, which
is the morphologically expected H30α geometry for Sy2 ring starbursts.
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
    output_dir="/Volumes/HouAstro/master/results/NGC5728_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
