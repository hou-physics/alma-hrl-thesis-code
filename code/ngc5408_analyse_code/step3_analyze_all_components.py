"""NGC 5408 sensitivity-check rerun with mask_components_mode='all'.

Loads canonical config and overrides only the component-selection mode.
Output → results/NGC5408_all_components/ to keep the original "center" run intact.

Motivation: NGC 5408 is a low-metallicity dwarf irregular (~5 Mpc); CO is
weak (fixed-env Nbeam=4.1 vs scan 1.9, factor 2× — small CO footprint).
mode='all' is unlikely to expand the mask substantially because CO itself is
compact, but it's a cheap consistency check given the tight 15" signal box.
Note: noise box only 4" (small dwarf, narrow PB envelope), so noise statistics
are based on ~16 beams — already verified line-free in the original run.
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
    output_dir="/Volumes/HouAstro/master/results/NGC5408_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
