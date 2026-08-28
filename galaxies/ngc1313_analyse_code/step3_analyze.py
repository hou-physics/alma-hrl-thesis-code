"""NGC 1313 step3 config (Phase Y dual-cube pipeline)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC1313"

# Pre-flight verification (done prior to config creation, 2026-04-23):
#   HRL cube freq range 230.901-232.770 GHz, H30a obs = 231.900928/(1+z) = 231.537 GHz ✓
#   CO  cube freq range 229.254-231.126 GHz, CO21 obs = 230.538/(1+z)    = 230.177 GHz ✓
#   Beams match (HRL 0.608"x0.529", CO 0.610"x0.530"), same WCS grid.
#   NED center: RA=49.56688°, Dec=-66.49825°  (pixel (805,897) in 1792^2 cube).
#   CO inspection: galaxy is a face-on SBd at d=4.2 Mpc with extended disk;
#   CO emission diffuse. Unrestricted ±150 km/s mom-0 peak at pixel (582,1475),
#   offset (-24.5", +63.6") from NED — on the disk north of NED nucleus.
#   Choosing a signal_region large enough to encompass NED nucleus + CO peak.
#   PB ≈ 0.98 at NED center, falls to ~0.3 at 40" offset; noise region kept
#   well inside PB>0.9 zone.

config = AnalysisConfig(
    galaxy="NGC 1313",
    z=0.001568,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC1313_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC1313_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC1313_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC1313_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC1313_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC1313_CO21_pb.fits.gz",
    # NGC 1313 is face-on, so use a square region. NED nucleus + brightest
    # off-nucleus CO mom0 peak lies 67" to the NNW. A 90" signal box centered
    # on NED covers both while keeping PB>0.5 everywhere (PB falls to 0.28
    # only beyond ~55" radius). Threshold scan filters noise voxels in the
    # CO-weak portions of the signal_region.
    signal_region=SkyRegion(
        ra_deg=49.56688, dec_deg=-66.49825,
        size_arcsec=90.0,
    ),
    # Noise region NW of nucleus at ~25" offset in each axis (pixel ~1032,1124).
    # PB=0.995, CO mom0 mean=-0.001, std=0.095 — comfortably line-free, clean.
    noise_region=SkyRegion(
        ra_deg=49.552949, dec_deg=-66.492694,
        size_arcsec=20.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC1313",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
