"""IRAS 17208-0014 step3 config (Phase Y dual-cube pipeline).

Novel ULIRG target. Advanced merger, log L_IR ~12.4, compact obscured
nuclear starburst (+ possible AGN). Project 2018.1.00486.S, 0.11" beam.
First HRL analysis for this source — no precursor reference value.

Pre-flight verified 2026-04-30:
- HRL cube spans 221.928-223.772 GHz; H30a observed 222.381 GHz (z=0.04281) IN RANGE.
- CO  cube spans 220.127-221.994 GHz; CO(2-1) observed 221.074 GHz IN RANGE.
- Both cubes share the same 2880x2880 grid at 0.016"/pix, phase center
  (RA=260.8414, Dec=-0.2836), and ~0.118" x 0.086" beams (cross-mousid
  identity: common-beam convolution will be near-identity, <1%).

PB structure (H30a cube): Gaussian-like, peak=1.0 at image center.
- PB > 0.9 contained within +/- ~6.7" of center (pixels ~440)
- PB > 0.5 contained within +/- ~13.4" of center (pixels ~840)
- PB > 0.2 contained within +/- ~19.5" of center (pixels ~1220)
This means the "noise box at 30"+ offset" suggestion in the task brief
is geometrically impossible at PB > 0.5 — the maximum useable offset is
~13" before PB drops below 0.5. We document this constraint and pick a
noise-box position ~8" west of nucleus where PB ~0.79 and CO mom-0 is
flat at the noise level (selected from a 32-position scout grid; see
audit comment on noise_region below).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/IRAS17208-0014"

config = AnalysisConfig(
    galaxy="IRAS17208-0014",
    z=0.04281,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/IRAS17208-0014_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/IRAS17208-0014_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/IRAS17208-0014_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/IRAS17208-0014_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/IRAS17208-0014_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/IRAS17208-0014_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # IRAS 17208-0014 is a ULIRG advanced merger; CO(2-1) emission is
        # confined to a compact (~1-2 kpc -> <2.5") nuclear disk + tidal
        # debris extending to a few arcsec. NED RA=260.8414 Dec=-0.2836
        # matches the cube phase center exactly. CO mom-0 peak located at
        # full-pixel (1440,1424) = 0.26" west of center, consistent with
        # the catalogued nucleus to within the 0.12" beam. A 10" box (=
        # ~8.5 kpc at z=0.0428) safely encloses the merger system + any
        # tidal-arm CO debris while staying inside the high-PB (>0.79)
        # core of the primary beam, avoiding edge noise amplification on
        # this 46"-wide FOV.
        ra_deg=260.8414, dec_deg=-0.2836,
        size_arcsec=10.0,
    ),
    noise_region=SkyRegion(
        # Selected from a 32-position scout grid at offsets 500/600/700/800
        # pixels (8/9.6/11.2/12.8") in 8 cardinal/inter-cardinal directions
        # from nucleus. Selection criterion: PB > 0.5 AND |CO mom-0 mean|
        # < 1e4 (i.e. flat at the noise floor) AND no neighbouring bright
        # CO companion. Picked r=500 west: pixel (1440, 940), corresponding
        # to RA=260.84365, Dec=-0.2836. Diagnostics at this position:
        #   PB(H30a) = 0.79 (well above the 0.2 cutoff)
        #   CO mom-0 mean = +2.7e3 (consistent with zero given std 6.5e4)
        #   CO mom-0 |median| = 1.3e4 (lowest among all PB>0.7 candidates)
        # The 30"+ offset suggested in the task brief is geometrically
        # impossible: the H30a primary beam falls below PB=0.5 at ~13.4"
        # from the phase center. r=500 (8") at PB=0.79 is the largest
        # offset with no detectable CO contamination and no PB-amplified
        # edge noise. Box size 6" gives a 375x375 pixel noise patch — large
        # enough for stable sigma-clipping yet small enough to stay clear
        # of the signal box's outer edge (closest approach 3" from signal
        # box edge to noise box edge).
        ra_deg=260.84365, dec_deg=-0.2836,
        size_arcsec=6.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/IRAS17208-0014",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
