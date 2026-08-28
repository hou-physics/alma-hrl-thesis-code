"""NGC 3628 step3 config (Phase Y dual-cube pipeline).

Modernized 2026-05-02: switched from custom-imaged v5 single-cube to
ALMA pipeline standard product (project 2013.1.00087.S, mousid
uid://A001/X144/X23b). spw0 contains CO(2-1), spw1 contains H30α —
auto-detected by step-1_download.py and exposed as canonical symlinks
NGC3628_H30a_*.fits / NGC3628_CO21_*.fits. The change removes
custom-imaging quality as a confound in per-galaxy null testing
(see docs/paper-notes/null_test_2026-05-02.md for motivation).

Toma+ (2018) and Bittner (2022) both report NGC 3628 H30α detection
on the same project's data — so this is a literature-validated cross-
check, not a novel candidate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628"

# Sky-coord signal/noise regions inherited from v5 analysis (galaxy
# nucleus + edge-on disk geometry): NGC 3628 nucleus J2000
# 11h20m17.0s +13d35m22s = (170.0708°, +13.5888°). The legacy boxes
# (40″ × 12″ signal, 15″ × 13″ noise) were tuned for the edge-on
# starburst disk and should still apply to the pipeline product —
# verify no edge clipping after first run on the new cube.
config = AnalysisConfig(
    galaxy="NGC 3628",
    z=0.002772,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC3628_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC3628_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC3628_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC3628_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC3628_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC3628_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        ra_deg=170.0708, dec_deg=13.5888,
        size_arcsec=40.0, dec_size_arcsec=12.0,
    ),
    noise_region=SkyRegion(
        ra_deg=170.0707, dec_deg=13.5944,
        size_arcsec=15.0, dec_size_arcsec=13.0,
    ),
    strong_line_window_kms=460.0,
    weak_line_window_kms=760.0,
    output_dir="/Volumes/HouAstro/master/results/NGC3628",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
