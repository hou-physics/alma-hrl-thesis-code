"""IRAS F18293-3413 step3 config (Phase Y dual-cube pipeline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/IRASF18293-3413"

config = AnalysisConfig(
    galaxy="IRAS F18293-3413",
    z=0.018176,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/IRASF18293-3413_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/IRASF18293-3413_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/IRASF18293-3413_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/IRASF18293-3413_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/IRASF18293-3413_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/IRASF18293-3413_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED position (WISEA J183241.12-341127.3). CO21 radial profile in
        # non-pbcor drops to ~2% of peak by r=5" and to noise by r=7"; a
        # compact LIRG core. 20" safely encloses the full emission with
        # room for mask scanning, well inside PB>0.5 region (radius ~12").
        ra_deg=278.17134, dec_deg=-34.19094,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # 15" offset W of NED center (pixel (344, 481) in 960x960 grid).
        # PB at center 0.846, PB_min across box 0.343; mean CO mom0 in
        # box = 0.03 sigma — cleanest available offset given the
        # diffuse CO + compact primary beam (PB FWHM ~25" @226 GHz).
        # Search over the full PB-covered annulus r=250-400 px ranked
        # this position as having the lowest |<CO>| / sigma_mom0.
        ra_deg=278.173855, dec_deg=-34.190947,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/IRASF18293-3413",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
