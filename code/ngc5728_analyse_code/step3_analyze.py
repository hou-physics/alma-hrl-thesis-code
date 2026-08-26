"""NGC 5728 step3 config (Phase Y dual-cube pipeline).

NGC 5728: Sy2 barred galaxy at z=0.009186, D ~ 39 Mpc.
- Native HRL beam 0.563" x 0.487" (~95 pc at 39 Mpc) -- sweet spot.
- Pixel scale 0.084"/pix, cube 490x490 (~41" x 41" FOV).
- Single pointing, NED center coincides with cube CRPIX (245,245).
- PB radial profile: PB=1.0 at center, PB=0.5 at r~13", cube edge at PB~0.05.
- AGN-host caveat: HRL signal includes Sy2 NLR contribution; not purely SF.

Pre-flight verification (frequencies confirmed in cube):
- H30a observed = 231.900928 / 1.009186 = 229.7926 GHz
  HRL cube range 229.505 -- 231.193 GHz   --> INSIDE
- CO(2-1) observed = 230.538 / 1.009186 = 228.4422 GHz
  CO  cube range 227.436 -- 229.305 GHz   --> INSIDE
- HRL/CO native beams nearly identical (0.563x0.487 vs 0.562x0.486),
  common-beam convolution will be ~identity (<0.1% relative).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5728"

config = AnalysisConfig(
    galaxy="NGC 5728",
    z=0.009186,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5728_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5728_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5728_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5728_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5728_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5728_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED center for NGC 5728: RA=220.59970, Dec=-17.25317.
        # Cube CRVAL1/CRVAL2 coincide with NED to <0.001" -- single pointing.
        # CO mom-0 (5-sigma over +/-400 km/s) reveals a circumnuclear ring +
        # bar of radial extent ~5" (95% percentile 4.3", max 4.9"). Adopting
        # 22" signal box gives >4x margin around CO emission, keeps the box
        # entirely inside the cube (490x490 = 41.2"), and stays well within
        # PB>0.5 (PB=0.5 contour at r~13"). 22" was preferred over 25"
        # because the box must remain inside the cube edges with margin
        # for the threshold scan to land on coherent CO morphology.
        ra_deg=220.599700, dec_deg=-17.253169,
        size_arcsec=22.0,
    ),
    noise_region=SkyRegion(
        # Noise box centered 10" west of NED center, on the major-axis-free
        # side (no detected CO above 5-sigma in any direction at r=10").
        # Pixel position (364, 245): PB=0.674 (well above 0.5 cutoff), CO
        # mom-0 box-max 0.15 vs 5-sigma threshold 0.70 (i.e. the entire 10"
        # box is line-free at >5x margin). Mean -0.028, std 0.063 in box.
        # Note: task spec asked for "30"+ offset within PB>0.5", but this
        # cube's PB=0.5 radius is only ~13" so 30" lies outside the
        # measurable field. Adopted 10" offset as the largest line-free
        # offset that keeps PB>0.6, fully within cube footprint, and clear
        # of the ~5" CO ring + bar emission. RA decreases east on this
        # archive image (negative CDELT1), so px=(364,245) corresponds to
        # +10" pixel-x offset = -10" RA offset = 10" WEST of NED center.
        ra_deg=220.596793, dec_deg=-17.253169,
        size_arcsec=10.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5728",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
