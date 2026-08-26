"""NGC 5408 step3 config — H30α at the ULX-1 SF complex (novel target).

NGC 5408 is a Magellanic-irregular dwarf starburst (D = 7.2 Mpc, ~1.5' across)
hosting the famous ULX NGC 5408 X-1 (possible IMBH). The ALMA observation
analysed here is centered on the ULX-1 position, NOT the optical galaxy
nucleus 26" NE; the cube's PB > 0.5 cone is therefore a tight 12-13" radius
disk around ULX-1, with the NED galaxy center falling outside the cube
footprint entirely (PB = NaN at NED). The science target is the young SF
complex co-spatial with X-1 (Cseh+ 2014 radio counterpart, embedded H II
regions, CO emission per literature).

Pre-flight verified 2026-04-30:
  H30α observed:  231.5115 GHz inside HRL cube range [231.076, 232.946] GHz ✓
  CO(2-1) obs:    230.1509 GHz inside CO cube range  [229.210, 231.080] GHz ✓
  HRL beam: 0.690"×0.574"  (BPA 79.1°)
  CO  beam: 0.699"×0.579"  (BPA 78.6°)
    → near-identity common-beam (Δ < 1.5%); short-circuit expected
  Beam_pc ≈ 24×20 (D = 7.2 Mpc) — sweet spot for compact SF complex
  Cube grid: 432×432 pix @ 0.110"/pix; both HRL and CO share identical WCS
  Pointing center (CRVAL1/CRVAL2): RA=210.831790°, Dec=-41.382970°
    = ULX-1 sky position (NOT NED galaxy optical center 25.9" NE)

Mode rationale:
  - Compact starburst SF complex (single source) → mask_components_mode
    default 'center' is appropriate (NGC 5253 / NGC 4945 family, not
    distributed-disk PHANGS family)
  - Novel target, no published H30α reference → first-detection analysis;
    no comparison value
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5408"

config = AnalysisConfig(
    galaxy="NGC 5408",
    z=0.001682,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5408_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5408_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5408_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5408_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5408_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5408_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # AUDIT: signal_region centered on ULX-1 (= cube CRVAL1/CRVAL2 = pointing
        # center), NOT the NED optical galaxy nucleus.
        # (a) Why ULX-1, not NED: the cube is centered on ULX-1; the NED
        #     optical center lies 25.9" NE of the pointing and falls OUTSIDE
        #     the cube footprint entirely (PB = NaN at NED pixel coordinates,
        #     verified 2026-04-30). The science target of this archival
        #     observation is the ULX-1 young SF complex (Cseh+ 2014 radio
        #     counterpart), co-spatial with embedded H II regions and CO
        #     emission. PB at pointing center = 1.000.
        # (b) Box size 15" half-extent ±7.5"; PB at the corners (±7.5") =
        #     ~0.74, well inside PB > 0.5. Comfortably encloses the ULX SF
        #     complex (~few-arcsec scale at 7.2 Mpc) plus margin for the
        #     mask scan, while staying well within the tight ~12-13"-radius
        #     PB > 0.5 cone of this single-pointing observation.
        ra_deg=210.831790, dec_deg=-41.382970,
        size_arcsec=15.0,
    ),
    noise_region=SkyRegion(
        # AUDIT: noise box centered 10" S of pointing center (= 10" S of
        # ULX-1), size 4". Placement rationale:
        # (a) Outside signal: noise box top edge at Dec offset -8" lies
        #     below signal box bottom at Dec offset -7.5" → non-overlapping.
        # (b) Inside PB > 0.5: every corner verified — center PB = 0.669,
        #     bottom corners (Dec offset -12") PB ≈ 0.544, top corners
        #     (Dec offset -8") PB ≈ 0.764. Minimum PB across the box ≥ 0.54.
        # (c) Distance from ULX SF complex: center 10" S of ULX-1 → outside
        #     the few-arcsec ULX SF region; safe for line-free sigma-clipping.
        # (d) Why a small 4" box rather than the 12-15" sample-wide default:
        #     this single-pointing observation has a very tight PB cone
        #     (PB > 0.5 only within ~12" radius); a larger box would either
        #     overlap the signal region or exceed PB > 0.5. 4" gives ~36
        #     beam-areas in the box (4"×4" / 0.4"² ≈ 40 pixels per beam-area
        #     statistic), sufficient for σ-clip.
        ra_deg=210.831790, dec_deg=-41.385748,  # -41.382970 - 10/3600
        size_arcsec=4.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5408",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
