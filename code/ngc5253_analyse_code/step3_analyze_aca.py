"""NGC 5253 step3 config — ACA (X1e, 7-m Morita) tier sanity test.

Companion to step3_analyze.py (X1c 12-m, 0.91" beam, S/N 3.08 non-detection).
After X1c fell below null p95 = 3.54, this ACA run tests whether the compact
nuclear starburst — concentrated into a single resolution element by the 7-m
array's larger beam — produces a cleaner single-dish-style total-flux
measurement of Bendo+2017's ~0.5–1 Jy·km/s detection.

Pre-flight verification (2026-04-24, X1e cubes):
  HRL cube freq 230.656-232.524 GHz, H30a obs 231.577 GHz -> IN RANGE.
  CO  cube freq 229.295-231.162 GHz, CO21 obs 230.216 GHz -> IN RANGE (at edge).
  Beams elongated: HRL 10.25"x4.12" @ PA 78.5 deg; CO 8.19"x3.87" @ PA 82.0 deg.
    The advertised "4.95"" ACA resolution is the geometric mean; the actual
    synthesized beam is highly elliptical at this low declination (-31.6 deg)
    because of the limited short-baseline coverage in one direction.
  Cube 120x112 pixels at 0.760"/pix = 91.2"x85.1" field.
  ACA PB HWHM ~ 19" at 230 GHz (PB FWHM ~38"), so the cube spans ~2 PB FWHMs.

Region choices (sky coords inherited from X1c CO peak — the source position
doesn't change between tiers):
  - Signal: 40" square centered on CO peak. Larger than X1c's 20" because
    a 10x4 beam needs multiple beam diameters of region to do mask scanning,
    and the source PSF itself will span ~10-15" in the elongated axis.
  - Noise: 25" square offset 25" W, outside the CO source + PSF skirt,
    still inside PB>0.5 (checked post hoc).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_aca"

config = AnalysisConfig(
    galaxy="NGC 5253",
    z=0.0014,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5253_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5253_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5253_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5253_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5253_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5253_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # CO peak coords inherited from X1c pre-flight. 40" square accommodates
        # the 10"x4" elongated ACA beam + compact source (unresolved in a beam
        # this size) with several beam diameters of buffer for mask scanning.
        ra_deg=204.982993, dec_deg=-31.640281,
        size_arcsec=40.0,
    ),
    noise_region=SkyRegion(
        # Offset ~25" W of CO peak (same direction as X1c noise region).
        # PB>0.5 extends well beyond 25" for ACA so PB coverage is not a
        # concern at this offset — the limiting factor is staying outside
        # the PSF skirt of the bright compact source.
        ra_deg=204.991250, dec_deg=-31.640681,
        size_arcsec=25.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5253_aca",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
