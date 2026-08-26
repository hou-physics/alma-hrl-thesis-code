"""NGC 7793 sensitivity rerun with signal_region enlarged to cover the full
disk, mode='all' kept. Tests whether headline S/N changes when manual box no
longer truncates CO emission. Cube is ~3.5'×3.5' (1700×1400 pix at ~0.13"/pix);
we use a 200" box to cover most of the visible CO disk while staying inside
PB > 0.5 except at edges."""
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
from config import SkyRegion

# Inflate signal_region from 45" to 200" — enough to cover the visible CO
# fragmented disk in the displayed cube (~3 arcmin).
new_signal = SkyRegion(
    ra_deg=_base.config.signal_region.ra_deg,
    dec_deg=_base.config.signal_region.dec_deg,
    size_arcsec=200.0,
)

config = replace(
    _base.config,
    signal_region=new_signal,
    mask_components_mode="all",
    output_dir="/Volumes/HouAstro/master/results/NGC7793_full_field",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
