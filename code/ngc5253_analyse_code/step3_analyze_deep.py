"""NGC 5253 step3 config — deep tier (X1a H30α + X1c CO21 dual-mousid).

Option A path to verify the X1c marginal S/N 3.31 weak-signal hypothesis at
Bendo-equivalent sensitivity. X1a is the 12-m long-baseline extended config
that Bendo+2017 used (0.19″ beam, 0.92 mJy/bm, 18σ on a 0.15″ compact source).
We download only X1a spw1 (H30α, 25.6 GB) and reuse X1c CO21 spw0 as the
strong-tracer cube — Phase Y dual-cube pipeline convolves both to a common
beam (0.91″ = larger of the two) and reprojects CO mask onto HRL grid.

Pre-flight (will be verified by the pipeline on first run):
  HRL cube (X1a): beam 0.19″, sens 0.92 mJy/bm, 16.46 GB pbcor
  CO cube  (X1c): beam 0.91″, sens 2.05 mJy/bm, 0.86 GB pbcor (same source,
    same observing project, different array config)
  Common beam: 0.91″ (X1c dominates; X1a smoothed up to match)
  Expected S/N improvement vs X1c alone: ~2.2× (factor = 2.05/0.92 sensitivity)
  Expected flux under this pipeline: Bendo-consistent 0.86 Jy·km/s if real,
    or noise-level ~0.1 Jy·km/s if the X1c 3.31 signal was statistical fluctuation

The test: does flux measured here significantly exceed the X1c 0.328 Jy·km/s
and approach Bendo's 0.86? If yes → detection confirmed + X1c was real marginal.
If no (flux stays ≲ 0.4) → X1c 3.31 was selector/noise and deep data agrees
with sensitivity-limited non-detection.

Signal/noise regions inherited from X1c config (same CO peak coords, same
sky geometry — sky positions invariant across mousids).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

HRL_DIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep"
CO_DIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253"

config = AnalysisConfig(
    galaxy="NGC 5253",
    z=0.0014,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{HRL_DIR}/NGC5253_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{HRL_DIR}/NGC5253_H30a_nonpbcor.fits",
    co_pbcor_path=f"{CO_DIR}/NGC5253_CO21_pbcor.fits",
    co_nonpbcor_path=f"{CO_DIR}/NGC5253_CO21_nonpbcor.fits",
    hrl_pb_path=f"{HRL_DIR}/NGC5253_H30a_pb.fits.gz",
    co_pb_path=f"{CO_DIR}/NGC5253_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # CO peak in X1c (same physical position across mousids). 20" box
        # comfortably encloses the 5σ CO double-core with headroom for
        # mask scanning; well inside PB>0.5 for both X1a and X1c.
        ra_deg=204.982993, dec_deg=-31.640281,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # Same 15" SE-offset box used for X1c — physics invariant across mousids.
        ra_deg=204.987952, dec_deg=-31.640681,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5253_deep",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
