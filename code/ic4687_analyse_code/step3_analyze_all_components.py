"""IC 4687 sensitivity-check rerun with mask_components_mode='all'.

Loads canonical config and overrides only the component-selection mode.
Output → results/IC4687_all_components/ to keep the original "center" run intact.

Motivation: IC 4687 sits in an interacting LIRG pair with IC 4686 (~5" SE)
inside the same CO 3σ envelope. The default center-anchored selection picks
only the IC 4687 nucleus knot (scan Nbeam=6.0); fixed-env Nbeam=82.9 shows the
joint CO footprint is ~14× larger. mode='all' lets both nuclei (and any
residual envelope) contribute when their components survive sthresh.
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
    output_dir="/Volumes/HouAstro/master/results/IC4687_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
