"""NGC 253 step3 config (Phase Y dual-cube pipeline).

NGC 253 is a single-cube case: H40α and CS(2-1) are both in the same
Band 3 SPW cube (only ~1 GHz apart in frequency). Strong tracer is **CS(2-1)**,
not CO — legacy picked this because CO(1-0) is at 115 GHz, outside the SPW.

Prerequisite: run `reconstruct_pbcor.py` once to create NGC253_H40a_pbcor.fits
(work_dir only has nonpbcor + 2D PB, no pre-computed pbcor cube).

Legacy v1 flux: 4.85 Jy·km/s (raw, no baseline sub; integrated on nonpbcor).
Toma: 6.39 Jy·km/s (re-imaged single observation → better sensitivity).
Legacy script preserved as step3_analyze_v1_legacy.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC253"

# Sky coords derived from legacy pixel boxes via cube WCS (0.16"/pix):
#   SIGNAL_BOX [275, 295, 470, 420] → center (372, 357), 31.2"×20.0" (edge-on disk)
#   NOISE_BOX  [500, 500, 580, 580] → center (540, 540), 12.8"×12.8"
config = AnalysisConfig(
    galaxy="NGC 253",
    z=0.00080,
    weak_line="H40a",
    strong_line="CS21",           # CS(2-1) at 97.98 GHz, same SPW as H40α (99.02 GHz)
    hrl_pbcor_path=f"{WORKDIR}/NGC253_H40a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC253_H40a_spw31_Xf05_v1.fits",
    co_pbcor_path=f"{WORKDIR}/NGC253_H40a_pbcor.fits",      # same cube, single-cube case
    co_nonpbcor_path=f"{WORKDIR}/NGC253_H40a_spw31_Xf05_v1.fits",
    hrl_pb_path=f"{WORKDIR}/NGC253_spw31_pb_mfs.fits",
    co_pb_path=f"{WORKDIR}/NGC253_spw31_pb_mfs.fits",
    signal_region=SkyRegion(
        ra_deg=11.88808, dec_deg=-25.28838,
        size_arcsec=31.2, dec_size_arcsec=20.0,   # edge-on nuclear starburst
    ),
    noise_region=SkyRegion(
        ra_deg=11.87983, dec_deg=-25.28024,
        size_arcsec=12.8, dec_size_arcsec=12.8,
    ),
    strong_line_window_kms=300.0,   # CS FWHM ~77 km/s, window = 4× FWHM
    weak_line_window_kms=760.0,
    output_dir="/Volumes/HouAstro/master/results/NGC253",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
