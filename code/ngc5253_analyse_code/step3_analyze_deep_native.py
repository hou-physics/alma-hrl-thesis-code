"""NGC 5253 step3 config — deep tier NATIVE 0.19" (X1a H30α + X1a CO21 same-mousid).

Discovered 2026-04-25: X1a project's spw0 covers CO(2-1) at 230.228 GHz,
matching NGC 5253's redshifted CO(2-1) (230.215 GHz, 17 km/s offset).
Earlier alma-query missed this because spw0 was treated as continuum sideband
in the project metadata. With both H30α (spw1) and CO21 (spw0) from the same
X1a mousid, we run native 0.19" same-mousid analysis instead of cross-mousid
(which forced common-beam convolution to 0.91" = 5× resolution loss).

Pre-flight expectations:
  HRL cube (X1a spw1): beam 0.19″, sens 0.92 mJy/bm
  CO cube  (X1a spw0): beam 0.19″, sens ~similar
  Common beam: identity (same mousid → no convolution)
  Mask resolution: 0.19" = 5× tighter than cross-mousid version
  Predicted S/N: 12-16 (vs cross-mousid 8.01)
  Predicted flux: ~0.787 Jy·km/s (similar; only S/N improves via mask matching)

Compared to step3_analyze_deep.py (cross-mousid):
  - Same physical sky regions (CO peak position is mousid-invariant)
  - Same redshift, same line definitions
  - DIFFERENT output_dir to keep both versions for comparison
  - DIFFERENT CO inputs (X1a same-mousid instead of X1c)

If this run gives S/N substantially > 8 + flux similar to 0.787 → confirms
matched-filter benefit of native resolution. If S/N drops or flux changes
much → suggests source is more extended than the 0.19" beam can resolve
optimally (less likely given Bendo's 0.3" imaging showed compact emission).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep"

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
        # AUDIT: CO mom-0 peak from X1c imaging at world pixel (212, 186), which
        # lies 0.65" from NED center and 6.6" NE of mosaic pointing center. This
        # sky position is invariant across mousids, so we reuse the X1c-derived
        # coordinate for the X1a same-mousid analysis. 20" box covers the 5"
        # circumnuclear starburst with margin; well inside PB > 0.6 at the X1a
        # 0.19" beam (12m long-baseline mosaic envelope is large).
        ra_deg=204.982993, dec_deg=-31.640281,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # AUDIT: 15" SE-offset box, 25" from CO peak — well outside 5" source
        # and outside 3σ CO footprint by ~10". PB > 0.5 verified at X1c (same
        # mosaic geometry). σ-clip on this box gives line-free noise estimate.
        ra_deg=204.987952, dec_deg=-31.640681,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5253_deep_native",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
