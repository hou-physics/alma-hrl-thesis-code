"""NGC 5248 sensitivity-check rerun with mask_components_mode='all'.

Loads the canonical config from `step3_analyze.py` and overrides only the
component-selection mode. Output goes to `results/NGC5248_all_components/`
to keep the original "center" run in `results/NGC5248/` for comparison.

Motivation: NGC 5248 is a face-on barred spiral with CO concentrated in the
spiral arms and bar ends, NOT at the geometric center. The default "center"
mode falls back to the largest single connected component above threshold,
which captures only one knot (the lower-right arm) and excludes the upper-
left arm + central bar region. "all" keeps every component within the CO
footprint so the mask integrates across all genuinely CO-bright regions.
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
    output_dir="/Volumes/HouAstro/master/results/NGC5248_all_components",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
