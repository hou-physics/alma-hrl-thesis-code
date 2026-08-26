"""NGC 4945 control verification — S1+N1 adaptive on a strong detection.

NGC 4945 is a known clean H30α detection (S/N 23.5, flux 8.84 Jy·km/s vs
Toma 9.27, 4.7% diff). Compact-starburst nucleus → mode='center' is the
correct prior. Verification: under adaptive signal/noise (S1+N1), the
detection should remain stable within ~5% (the Sun+/Leroy framework is
designed to be conservative on compact bright sources).

If S/N drops by >20% or flux disagrees by >10% from the canonical run,
the adaptive mode has a regression and must be debugged before claiming
methodology improvement on disk-class targets.
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
    # Keep mask_components_mode="center" — NGC 4945 is compact starburst, no morphology change
    signal_region_mode="dilated_footprint",
    signal_buffer_arcsec=5.0,
    signal_center_anchor_arcsec=8.0,
    noise_region_mode="auto",
    noise_pb_threshold=0.5,
    output_dir="/Volumes/HouAstro/master/results/NGC4945_adaptive",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
