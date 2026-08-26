"""NGC 4826 (M64, "Black Eye galaxy") step3 config — H30a / CO(2-1) ACA 7m.

Project 2016.2.00053.S, mousid uid://A001/X124a/X1b4 (ACA 7m only).
Single pointing 1.2" from NED center. Sensitivity 8.73 mJy/bm at 10 km/s,
~10x worse than NGC 5253 X1c. Cube is 70x70 pix at 1.1"/pix (77"x77" field),
PB>0.5 region ~45" diameter (radius 22.4"). 109 channels in HRL spw1, 1917
in CO spw0.

Pre-flight verification (2026-04-25):
  HRL cube freq 231.369-233.057 GHz; H30a obs 231.577 GHz -> IN RANGE.
  CO  cube freq 229.285-231.156 GHz; CO21 obs 230.216 GHz -> IN RANGE.
  Beams nearly matched: HRL 6.333"x5.638"; CO 6.421"x5.787".
    Common-beam convolution will short-circuit to identity (<1.5% mismatch).
  Note: actual beam ~6", not the ~5.05" archive-listed angular resolution
  (which is the geometric mean of the deconvolved restoring beam from a
  partial track).

CO morphology (from non-pbcor mom0, |v|<300 km/s window):
  - Peak at pixel (36, 36) = RA=194.181738, Dec=21.682964 (NED-sep=0.7").
  - Emission falls to noise level beyond ~12" along both major axes.
  - 5"-6" beam smears M64's ~24"-diameter inner ring into a roughly
    centrally-concentrated source. Outer disk emission is below noise
    in this single-pointing cube.

Region choices:
  - Signal: 30" square centered on CO peak (NED-coincident). Covers full
    detected CO emission (~24" diameter) plus a ~3" margin for mask scan
    smoothing. PB range across the 30" box: 0.52-1.00 (fully PB>0.5).
  - Noise: 16" square offset 14.3" S of CO peak (RA same as CO peak,
    Dec offset = -14.3"). PB range across box: 0.50-0.99. Position chosen
    by scanning candidate 16" boxes inside PB>0.5 and minimizing CO
    moment-0 magnitude: this box has CO mean=-2.5, std=16.7 (lowest of
    all PB-compliant offsets tried). North/East/West offsets either fall
    outside PB>0.5 corners or hit ring residuals.
  - The cramped layout (30"+16" boxes barely fit inside 45"-diameter PB>0.5
    region) is forced by NGC 4826's small ACA single-pointing field, and is
    a known limitation for ACA-only targets — see NGC 5253 X1e ACA tier
    notes for analogous geometry.

Sensitivity expectation: 8.73 mJy/bm/(10 km/s) noise floor predicts
sub-threshold (S/N < 5) result for any sensible HRL flux at this beam,
analogous to NGC 5253 X1e (S/N 3.22 marginal). A clean S/N > 5 detection
would warrant double-checking continuum subtraction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4826"

config = AnalysisConfig(
    galaxy="NGC 4826",
    z=0.0014,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC4826_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC4826_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC4826_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC4826_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC4826_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC4826_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # Center: CO non-pbcor mom-0 peak, pixel (36,36) on the 70x70 grid.
        # This is 0.7" from NED center (194.181960, +21.682990) - the small
        # offset is sub-pixel and consistent with single-pointing astrometry.
        # Size 30": covers the full detected CO emission (~24" extent in the
        # ACA 5"-beam-convolved view of M64's inner ring) + ~3" smoothing
        # margin for the s_f={3,5,10}*theta_CO mask scan. PB range across
        # the 30" box is 0.52-1.00 (fully PB>0.5; constraint satisfied).
        ra_deg=194.181738, dec_deg=21.682964,
        size_arcsec=30.0,
    ),
    noise_region=SkyRegion(
        # Offset ~14.3" S of CO peak (Dec offset = -14.3"; RA same column).
        # Position chosen by scanning all 16"-box candidates inside the
        # PB>0.5 region and selecting the one with lowest |CO mom-0| mean.
        # This box: CO mean=-2.51, std=16.70 (lowest among compliant offsets;
        # E/W/N alternatives have CO std 25-40 due to ring-edge residuals).
        # PB range across the 16" box: 0.50-0.99 (constraint satisfied;
        # min PB = 0.503 at the SW corner). The 16" size is forced down
        # from the 20" project default by NGC 4826's small ACA field
        # (PB>0.5 region is only ~45" diameter, so a 30" signal box +
        # 20" noise box won't both fit clear of the PB drop).
        # Distance from CO peak: 14.3" = ~2.4 beam diameters, sufficient
        # to be uncorrelated with source emission.
        ra_deg=194.182067, dec_deg=21.678992,
        size_arcsec=16.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC4826",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
