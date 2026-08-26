"""NGC 3627 (M66) step3 config (Phase Y, single-cube case).

Project 2015.1.01538.S, mousid X717, ari_l deliverable. spw3 (centered
~230.547 GHz, ~1.875 GHz wide) covers BOTH the redshifted CO(2-1) line
(229.986 GHz at z=0.0024) AND the redshifted H30α line (231.346 GHz).
The same FITS cube is therefore used for both HRL and CO inputs;
Phase Y supports this (single-cube degenerate mode: identity convolve +
identity reproject).

Pre-flight verification (FITS header):
- Cube freq range: 229.6105 – 231.4832 GHz (NAXIS3=3836)
  - H30α observed (231.901 / 1.0024 = 231.3457 GHz) → in cube ✓
  - CO(2-1) observed (230.538 / 1.0024 = 229.9860 GHz) → in cube ✓
- Beam: BMAJ=0.7053″, BMIN=0.5752″ (sub-arcsec, well-resolved)
- Pixel scale: 0.12″/pix; field 180″×207″
- NED nucleus: RA=170.06254°, Dec=+12.99161° (J2000)

Toma (T2.1, T4.4) AND Ada (T4.1, §4.5) both report H30α DETECTION for
NGC 3627 — strong positive control.
"""
import sys
from pathlib import Path

# Add _step3 shared package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC3627"
PBCOR = f"{WORKDIR}/NGC3627_H30a_pbcor.fits"
NONPBCOR = f"{WORKDIR}/NGC3627_H30a_nonpbcor.fits"
PB = f"{WORKDIR}/NGC3627_H30a_pb.fits.gz"

config = AnalysisConfig(
    galaxy="NGC 3627",
    z=0.0024,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=PBCOR,
    hrl_nonpbcor_path=NONPBCOR,
    co_pbcor_path=PBCOR,        # single-cube: same file as HRL (spw3 covers both lines)
    co_nonpbcor_path=NONPBCOR,
    hrl_pb_path=PB,
    co_pb_path=PB,              # single-cube: same file
    # SIGNAL REGION:
    # Centered on NGC 3627 NED nucleus (RA=170.06254°, Dec=+12.99161°).
    # M66 is an SBb spiral with: nuclear ring radius ~10-15″, a stellar bar
    # extending ~30″ along PA ≈ 165-180°, and embedded HRL emission expected
    # to track the nuclear ring + bar end-points (where dense ionized-gas
    # tracers concentrate). A 40″×40″ square box covers the full nuclear
    # ring + bar within margins. Square (no dec_size_arcsec) since the bar PA
    # is roughly N-S but the bar length and ring radius are comparable; we
    # do NOT need an edge-on rectangle like NGC 3628 (M66 is i ~57°, not
    # edge-on). The CO mom-0 confirms emission is confined to ~30″ × 25″
    # around the nucleus, well inside this 40″ box.
    signal_region=SkyRegion(
        ra_deg=170.06254, dec_deg=12.99161,
        size_arcsec=40.0,
        dec_size_arcsec=None,
    ),
    # NOISE REGION:
    # Probed CO mom-0 (±300 km/s around redshifted CO) on a 20″ box at
    # multiple offset positions; chose the cleanest position with PB > 0.5
    # required and far from M66's bright disk emission. M66 has extended
    # CO across ~1' along the major axis (PA ≈ 170°) so the cleanest noise
    # box is in the NW quadrant at (-55″ east, +50″ north) i.e. 55″ WEST
    # and 50″ NORTH of the NED nucleus → RA=170.04688°, Dec=+13.00549°.
    # Verified: PB_med = 0.959 (well above 0.5 cutoff), CO mom-0 max=2.3
    # mJy/bm·Hz, mom-0 median = -0.02 mJy/bm·Hz (consistent with noise);
    # well outside the bright nuclear ring + bar + spiral arms.
    noise_region=SkyRegion(
        ra_deg=170.04688, dec_deg=13.00549,
        size_arcsec=20.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC3627",
    # NGC 3627 is a face-on PHANGS barred spiral — CO emission is distributed
    # across nucleus + bar + multiple spiral arms. Default "center" mode falls
    # back to the largest single connected component, collapsing the mask to
    # the nucleus core (~130 beams at sthresh=32σ) and excluding genuine arm
    # signal. "all" mode retains every connected component within the CO
    # footprint; verified 2026-04-28 to lift S/N 2.89 → 3.20, fixed-env flux
    # 0.40 → 1.23 (sign-flip cleared, fluxes converge to within 15%).
    mask_components_mode="all",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
