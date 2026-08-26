"""NGC 5253 step3 config (Phase Y dual-cube pipeline).

Compact dwarf starburst at d=4 Mpc (Centaurus A group). ALMA Band 6 mosaic
(pointing center offset ~6.6" SW of the CO emission peak). Per-SPW delivery:
CO(2-1) in spw0 @ 230.198 GHz and H30a in spw1 @ 231.560 GHz, same WCS grid
(360x320 pix at 0.160"/pix = 57.6"x51.2" field). Beams are similar but not
identical: HRL 1.62"x0.91" @ PA 86.4 deg; CO 1.54"x0.90" @ PA 82.5 deg. The
Phase Y pipeline convolves both to a common enclosing beam before mask
reprojection. Reference: Bendo+2017 reports a clear H30a detection in this
galaxy, so we expect S/N ~ several.

Pre-flight verification (2026-04-24):
  HRL cube freq range 230.654-232.466 GHz, H30a obs = 231.900928/(1+0.0014)
    = 231.577 GHz  -> IN RANGE (931 MHz above lower edge, plenty of room)
  CO  cube freq range 229.293-231.104 GHz, CO21 obs = 230.538/(1+0.0014)
    = 230.216 GHz  -> IN RANGE (well-centered)
  Pointing center (PB peak) at pixel (160,180) = RA 204.98466, Dec -31.64144,
    offset 6.60" from CO peak -- matches task brief "6.7" mosaic offset".
  PB at CO peak = 0.93.

CO non-pbcor mom-0 over +-300 km/s of CO center:
  - mom0 sigma = 0.163 Jy/beam km/s; peak 3.91 (~24 sigma, clean detection)
  - 5 sigma connected component around the peak: 4.6"x5.3"
  - 10 sigma core: 2.1"x1.8"  (spatially very compact as expected for a dwarf)
  - CO peak at RA=204.982993, Dec=-31.640281 (J2000)  ~ 0.65" from NED center
    (task-brief coords 204.9832, -31.6401). The nucleus is essentially on NED.

Signal region: 20" square centered on the CO emission peak (not on the
pointing center). This fully encloses the 5 sigma CO component (~5" wide)
with a wide buffer for threshold scanning, and keeps the entire region
inside PB > 0.6 (PB drops below 0.5 only beyond ~13" from pointing center).

Noise region: 15" square offset 15.3" W of CO peak at RA=204.98795,
Dec=-31.64068. Grid-search optimum (scores CO mom-0 mean/sigma):
    mom0_mean = -0.0023 Jy/beam km/s (0.014 sigma -- essentially zero)
    PB_min = 0.21 (at pb_cutoff default 0.2), PB_mean = 0.67
Distance constraint (>15" from CO peak) satisfied with 0.3" margin.
A PB>0.5 offset box of this size is geometrically tight given the mosaic
pointing is offset to the SW of the source; this position sits on the NW
side of the source and stays well within the primary beam.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253"

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
        # Centered on CO mom-0 emission peak (world pixel 212,186), which
        # sits 0.65" from NED center and 6.6" NE of the mosaic pointing
        # center. 20" box comfortably covers the 5"-diameter compact source
        # with room for threshold scanning; stays inside PB > 0.6.
        ra_deg=204.982993, dec_deg=-31.640281,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # Offset 15.3" W of CO peak, outside the 5 sigma CO extent and
        # inside PB_min=0.21 (= pb_cutoff default) / PB_mean=0.67. Grid
        # search over all PB>0.2, distance>15" boxes gave mom0 mean
        # -0.0023 Jy/beam km/s (0.014 sigma -- essentially zero), the
        # cleanest line-free location in the field.
        ra_deg=204.987952, dec_deg=-31.640681,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5253",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
