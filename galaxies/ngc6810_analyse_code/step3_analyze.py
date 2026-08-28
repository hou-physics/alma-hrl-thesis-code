"""NGC 6810 step3 config (Phase Y dual-cube pipeline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC6810"

config = AnalysisConfig(
    galaxy="NGC 6810",
    z=0.006775,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC6810_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC6810_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC6810_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC6810_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC6810_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC6810_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        ra_deg=295.89351, dec_deg=-58.65559,
        # CO emission extends ~48" (CO21 mom-0 bbox 43"x44" at 5-sigma),
        # so 40" safely encloses the galaxy without touching edges.
        size_arcsec=40.0,
    ),
    noise_region=SkyRegion(
        # Placed NE of galaxy (pixel ~591,145 in 784x784 cube): RA=295.88284,
        # Dec=-58.66114. CO mom-0 here: mean=-0.01, std=0.10, max=0.78, PB~0.28
        # (well above the 0.2 cutoff). Original proposed noise (~29" E, 14" N)
        # overlapped galaxy emission, so moved further out diagonally.
        ra_deg=295.88284, dec_deg=-58.66114,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC6810",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
