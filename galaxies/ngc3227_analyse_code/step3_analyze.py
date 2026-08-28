"""NGC 3227 step3 config (Phase Y dual-cube pipeline).

Seyfert 1 at ~20 Mpc. ALMA long-baseline Band 6: 0.02"/pixel, synthesized beam
~0.14"x0.095" (HRL) / 0.15"x0.094" (CO21). Cube field only 43.2"x43.2" total;
PB>0.5 within r~13", PB>0.2 within r~19". CO(2-1) and H30a are in separate
SPWs (spw19 for CO, spw21 for H30a) but share identical pixel grid, WCS
reference, and frequency axis shape.

CO non-pbcor moment-0 over |v|<300 km/s (pb>0.5, sigma_clip sigma = 0.115
Jy/beam km/s per pixel):
  - >10sigma compact nuclear peak within r<=1.1"
  - >5sigma connected emission extends to r~9" (primarily along Dec, asymmetric
    with bright lobe to +Dec out to ~9", faint/lopsided toward -Dec)
  - The y-axis (Dec) extent is -1.3" to +8.9"; bright emission mostly within
    2-3" of nucleus, faint ring to the north. Total source footprint ~10".

Signal region 20" centered on NED NGC 3227 J2000 (155.877375, +19.865083):
  pb_min=0.44, pb_mean=0.78; fully encloses the r~9" CO-bright zone with
  comfortable margin for threshold scanning and is entirely inside PB>0.44.

Noise region: analogous to NGC 7130, this compact single-pointing field
cannot simultaneously host a 15" offset box AND satisfy PB>0.5 (PB>0.5 only
extends to r=13"). Options:
  - 15" box: geometrically impossible anywhere offset from nucleus (would
    require >=15"+source_radius offset, which crosses the PB=0.2 field edge).
  - 12" box centered 15.5" due south at RA=155.877375, Dec=+19.860778:
    pb_min=0.20 (= pb_cutoff default), pb_mean=0.42, mean(mom0)/sigma = -0.015
    (essentially zero within the per-pixel mom0 sigma of 0.12 Jy/km/s/beam).
    NaN fraction 0% inside box; 274k/360k pixels satisfy pb>0.15. This is
    clearly line-free and has ample statistics for sigma_clip_noise.
Adopted: 12" noise box at 15.5" south of nucleus. This mirrors NGC 7130's
compact-cube exception documented in that script.

High-resolution caveat: At 0.11"-0.14" angular resolution this long-baseline
observation resolves out extended emission. A typical HRL CMZ in a nearby
galaxy is 5-10" in extent, so HRL flux is spread across many (hundreds of)
beams. Per-beam surface brightness may fall below detection threshold even
for a genuine bulk HRL source. A low-S/N or non-detection here is NOT
diagnostic of an absent HRL; it reports only that per-beam HRL emission at
this resolution is <sigma_rms. Report this caveat in the sanity check.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC3227"

config = AnalysisConfig(
    galaxy="NGC 3227",
    z=0.003756,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC3227_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC3227_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC3227_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC3227_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC3227_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC3227_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED NGC 3227 J2000: 10h23m30.57s +19d51m54.30s -> 155.877375, +19.865083.
        # CO21 non-pbcor mom-0 >5sigma connected emission spans r<=9"; brightest
        # >10sigma core is within r<=1.1". 20" side length encloses the r~9" bright
        # zone with margin for threshold scanning, stays inside PB>0.44.
        ra_deg=155.877375, dec_deg=19.865083,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # Offset 15.5" due south of NED center (pixel (1080,305) in 2160x2160
        # CO cube). 12" box: pb_min=0.20 (at pb_cutoff default), pb_mean=0.42;
        # mean(mom0)/sigma = -0.015 (line-free), NaN fraction 0%.
        # A 15" box is geometrically impossible with pb_min > 0.2 given the
        # compact single-pointing field (PB>0.2 only r<19", source footprint
        # reaches r~9", leaves <10" clearance for a noise box). Follows the
        # same compact-cube compromise adopted for NGC 7130.
        ra_deg=155.877375, dec_deg=19.860778,
        size_arcsec=12.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC3227",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
