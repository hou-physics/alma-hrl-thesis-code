"""IC 4687 step3 config (Phase Y dual-cube pipeline).

IC 4687 is part of an interacting LIRG triplet with IC 4686 (~5" SE) and
IC 4689 (~80" S). IC 4689 lies well outside the cube field; only IC 4687
and IC 4686 are within the primary-beam-corrected area. Both nuclei sit
within a single ~6"x6" CO(2-1) emission complex centered on the IC 4687
nucleus.

Cube geometry (Project 2013.1.00271.S, 0.34"x0.27" beam, 0.049"/pix):
- 882x882 grid, full extent ~43" total, ~26" PB>0.5, ~38" PB>0.2
- HRL beam 0.342"x0.263", PA 35.9°
- CO  beam 0.344"x0.265", PA 36.6°  (essentially identical -> common beam ~ identity)
- HRL freq 227.7585-229.4461 GHz; expected H30a observed 227.9472 GHz IN RANGE
- CO  freq 225.5061-227.3798 GHz; expected CO(2-1) observed 226.6075 GHz IN RANGE
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/IC4687"

config = AnalysisConfig(
    galaxy="IC 4687",
    z=0.017345,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/IC4687_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/IC4687_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/IC4687_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/IC4687_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/IC4687_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/IC4687_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # Centered on CO(2-1) mom-0 peak at pixel (445, 442) =
        # RA=273.41586, Dec=-57.72534, which lies 0.4" from the NED IC 4687
        # nucleus position (273.41514, -57.72537) — well within one beam
        # (0.34") so the CO peak is the IC 4687 starburst nucleus itself.
        # CO emission >1 Jy/beam km/s spans only 6.1"x5.5" centered here,
        # and a secondary CO peak at offset (0.1", -3.0") is consistent
        # with IC 4686 contributing within the same envelope. A 16"
        # box generously encloses the pair plus surrounding CO envelope
        # while leaving >>4" to the noise region 12" north.
        ra_deg=273.41586, dec_deg=-57.72534,
        size_arcsec=16.0,
    ),
    noise_region=SkyRegion(
        # Placed 12" due north of the nucleus at pixel (445, 687).
        # The cube is small (PB>0.5 only spans ~26" diameter, PB>0.2 ~38"),
        # so the brief's 40-60" offset cannot fit the field at all. 12"
        # is the maximum offset that still keeps the noise box (8" half-
        # width = ±80 pix) inside PB>0.32. Within this box the CO mom-0
        # mean is +0.0005 Jy/beam km/s (≈0) and max 0.470 Jy/beam km/s,
        # i.e. cleanly outside the >1 Jy/beam km/s galaxy emission.
        # PB minimum 0.319 satisfies pipeline pb_cutoff=0.2 with margin.
        ra_deg=273.41586, dec_deg=-57.72200,
        size_arcsec=8.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/IC4687",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
