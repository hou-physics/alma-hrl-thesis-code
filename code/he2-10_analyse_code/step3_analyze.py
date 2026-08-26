"""He 2-10 (Henize 2-10 / ESO 495-G021) step3 config (Phase Y).

Single-cube case: H40α (rest 99.022952 GHz) AND CS(2-1) (rest 97.980953 GHz)
both fall in spw29 of project uid___A001_X1465_X67d (frequency coverage
96.915–98.780 GHz). At z=0.002912 the observed lines are:
  H40α:    99.022952 / 1.002912 = 98.7354 GHz   ✓ inside cube
  CS(2-1): 97.980953 / 1.002912 = 97.6965 GHz   ✓ inside cube

CO(1-0) is NOT in this archive tier for He 2-10, so CS(2-1) is used as the
strong (mass-tracing) line. Phase Y common-beam machinery runs as identity
since HRL and CO paths point to the same FITS.

He 2-10 is a famous dwarf compact starburst Blue Compact Galaxy (BCG),
hosting compact super star clusters within ~5″ + a radio nucleus.
Closest analog in our sample is NGC 5253. Both are "compact-starburst"
sources where the ionized gas is concentrated at the dynamical center —
mask_components_mode='center' (default) is therefore the appropriate prior
(consistent with NGC 5253 / NGC 4945 / NGC 253 setup; see
docs/paper-notes/mask-baseline.md §"When 'all' is not the right choice").

Cube geometry (verified pre-flight 2026-05-01):
  CRVAL = NED center (RA 129.06329°, Dec -26.40935°)
  Image: 350 × 350 pix at 0.290″/pix → 101.5″ × 101.5″
  Beam: BMAJ=2.226″, BMIN=1.646″, BPA=83.5°
  PB > 0.5 footprint: ~30″ radius (single pointing)
"""
import sys
from pathlib import Path

# Add _step3 shared package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/He2-10"
# Single-cube case: same FITS holds both H40α and CS(2-1).
COMBINED_PBCOR = f"{WORKDIR}/He2-10_H40a_pbcor.fits"
COMBINED_NONPBCOR = f"{WORKDIR}/He2-10_H40a_nonpbcor.fits"
COMBINED_PB = f"{WORKDIR}/He2-10_H40a_pb.fits.gz"

config = AnalysisConfig(
    galaxy="He2-10",
    z=0.002912,
    weak_line="H40a",
    strong_line="CS21",  # CO(1-0) not in this archive product; CS(2-1) used as tracer.
    hrl_pbcor_path=COMBINED_PBCOR,
    hrl_nonpbcor_path=COMBINED_NONPBCOR,
    co_pbcor_path=COMBINED_PBCOR,       # same cube → identity convolve + reproject
    co_nonpbcor_path=COMBINED_NONPBCOR,
    hrl_pb_path=COMBINED_PB,
    co_pb_path=COMBINED_PB,
    # SIGNAL REGION:
    # Centered on NED coords for He 2-10 (RA 129.06329°, Dec -26.40935°),
    # which also coincides with cube CRVAL and pixel (175, 175). He 2-10's
    # stellar mass, ionized gas, super star clusters, and radio nucleus
    # are all confined to the inner ~5″, but we use a 16″ box to give the
    # CO-footprint scan ample margin around the compact core (similar to
    # NGC 5253 setup) while staying well inside the PB > 0.7 region. PB
    # at center = 1.0; PB at the ±8″ corners ≳ 0.92 (verified). Square
    # box (no dec_size_arcsec) since He 2-10 is morphologically near-round
    # on arcsec scales — no edge-on disk constraint.
    signal_region=SkyRegion(
        ra_deg=129.06329, dec_deg=-26.40935,
        size_arcsec=16.0,
    ),
    # NOISE REGION:
    # Offset 16″ NW from NED center on the sky, where NW = west on the
    # equatorial axis (smaller RA) + north (larger Dec). With CDELT1<0 in
    # the cube, this lands at pixel (~230, 230) where PB=0.70. The 11″
    # box keeps every corner above the PB=0.5 floor required by the task
    # spec (verified: all 4 corners PB ∈ [0.506, 0.864]). The box thus
    # spans roughly 10″–22″ from the source, well outside both the
    # source's <5″ extent and the 16″ signal box (no overlap). NW chosen
    # rather than SE to stay clear of any extended diffuse emission on
    # the SE side of the galaxy. Inside PB > 0.5 footprint everywhere.
    # Sky-coord arithmetic:
    #   ΔRA = -16″/cos(26.4°)/3600 = -16/(3600·0.8957) = -0.004962°
    #     → RA_noise = 129.06329 - 0.004962 = 129.05833°
    #   ΔDec = +16″/3600 = +0.004444°
    #     → Dec_noise = -26.40935 + 0.004444 = -26.40491°
    noise_region=SkyRegion(
        ra_deg=129.05833, dec_deg=-26.40491,
        size_arcsec=11.0,
    ),
    # Default ±300 km/s windows; CS(2-1) and H40α are both narrow lines
    # for a dwarf galaxy (He 2-10 CO FWHM ~50 km/s reported in literature).
    # Pipeline's adaptive CS mom-0 window will narrow further if FW(10%) is
    # smaller than 300 km/s.
    output_dir="/Volumes/HouAstro/master/results/He2-10",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
