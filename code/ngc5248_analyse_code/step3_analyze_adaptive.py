"""NGC 5248 with full S1+N1 adaptive mode."""
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
    signal_region_mode="dilated_footprint",
    signal_buffer_arcsec=5.0,
    signal_center_anchor_arcsec=8.0,
    noise_region_mode="auto",
    noise_pb_threshold=0.5,
    nbeam_min=5,
    n_components_max=10,
    output_dir="/Volumes/HouAstro/master/results/NGC5248_adaptive",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
