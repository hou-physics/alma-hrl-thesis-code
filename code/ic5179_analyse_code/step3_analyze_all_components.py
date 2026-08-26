"""IC 5179 sensitivity-check rerun with mask_components_mode='all'.

Loads canonical config and overrides only the component-selection mode.
Output → results/IC5179_all_components/ to keep the original "center" run intact.

Motivation: IC 5179 is a face-on Sc disk; CO archival cube has sub-arcsec
long-baseline beam (per caveats-todo.md "Sub-arcsec long-baseline" entry).
Scan Nbeam=64.0 vs fixed-env Nbeam=460.5 (factor 7×) — CO footprint is highly
distributed and the center-anchored default may collapse the mask onto a
single arm knot. mode='all' tests whether disk-wide weak emission stacks up
above null threshold, OR whether short-spacing filtering rules it out.
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
    output_dir="/Volumes/HouAstro/master/results/IC5179_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
