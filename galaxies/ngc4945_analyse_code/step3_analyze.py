"""NGC 4945 step3 config (Phase Y dual-cube pipeline).

Prerequisite: run `reconstruct_nonpbcor.py` once to create the two
*_nonpbcor.fits files (they don't exist in the original work_dir).

Legacy v1 implementation kept as step3_analyze_v1_legacy.py for reference.
Legacy v1 flux was 9.117 Jy·km/s (raw, no baseline subtraction).
Phase Y baseline-subtracted flux expected to be smaller by ~30-50% due to
residual baseline removal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945"

# Sky coords derived from legacy pixel boxes via cube WCS (0.1"/pix scale):
#   SIGNAL_BOX [270, 180, 440, 370] → center (355, 275), 17"×19" rectangle
#   NOISE_BOX  [170, 420, 240, 490] → center (205, 455), 7"×7"
config = AnalysisConfig(
    galaxy="NGC 4945",
    z=0.00188,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC4945_H30a_spw1_v1_contsub.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC4945_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC4945_CO_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC4945_CO_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC4945_H30a_spw1_flux.fits",
    co_pb_path=f"{WORKDIR}/NGC4945_CO_flux.fits",
    signal_region=SkyRegion(
        ra_deg=196.36260, dec_deg=-49.46942,
        size_arcsec=17.0, dec_size_arcsec=19.0,   # edge-on disk
    ),
    noise_region=SkyRegion(
        ra_deg=196.36901, dec_deg=-49.46442,
        size_arcsec=7.0, dec_size_arcsec=7.0,
    ),
    strong_line_window_kms=460.0,
    weak_line_window_kms=760.0,
    output_dir="/Volumes/HouAstro/master/results/NGC4945",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
