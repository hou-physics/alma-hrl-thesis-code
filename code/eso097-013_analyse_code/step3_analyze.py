"""Circinus Galaxy (ESO 097-013) step3 config (Phase Y).

Single-cube degenerate case (analogous to Hen 2-10 / He 2-10): H40α
(rest 99.022952 GHz) AND CS(2-1) (rest 97.980953 GHz) both fall inside
spw3 of project 2015.1.01286.S (member.uid___A001_X2f7_X201, frequency
coverage 97.5813 - 99.4367 GHz). At z=0.001448 the observed lines are:
  H40α:    99.022952 / 1.001448 = 98.8798 GHz   ✓ inside cube
  CS(2-1): 97.980953 / 1.001448 = 97.8393 GHz   ✓ inside cube

CO(1-0) is not present in this archive product (would lie at 115.2 GHz,
outside Band 3 tuning). CS(2-1) is used as the strong (mass-tracing)
line. Phase Y common-beam machinery + reproject are identity operations
since HRL and CO paths point to the same FITS cube.

Circinus is a Sy2 host with a compact circumnuclear starburst ring at
~10-20″ scale embedded in a moderately inclined (i ~65°) host disk.
Beam in this archive is 2.585″ × 2.407″ at PA -20.5°, ~85 pc per beam
at D ≈ 6.2 Mpc — the nuclear SB ring is partially resolved.
mask_components_mode='center' is the appropriate prior: Circinus's
ionized-gas peak is concentrated at the dynamical nucleus (Sy2 +
circumnuclear SB), so we want the center-connected component of the
CO-thresholded mask, not the whole disk. Same compact-starburst
prior as NGC 5253 / He 2-10 / NGC 4945.

Cube geometry (verified pre-flight 2026-05-13):
  CRVAL = (RA 213.29127°, Dec -65.33902°), within 1″ of NED center
  Image: 240 × 240 pix at 0.470″/pix → 112.8″ × 112.8″ FOV
  Beam: BMAJ=2.585″, BMIN=2.407″, BPA=-20.46°
  PB(center)=1.0; PB=0.5 at ~21.5″ diagonal radius; usable PB>0.2 out
  to ~30″ diagonal.
  CTYPE3=FREQ, RESTFRQ=98.65 GHz (header convention, not used by step3),
  SPECSYS=LSRK.

No prior published mm-RRL ALMA analysis of Circinus. cm-RRL detection
exists (Roy & Goss 2003 via VLA, cited Bittner thesis) but this is the
first H40α / mm archival analysis.
"""
import sys
from pathlib import Path

# Add _step3 shared package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/ESO097-013"
# Single-cube degenerate case: same FITS holds both H40α and CS(2-1).
COMBINED_PBCOR = f"{WORKDIR}/ESO097-013_H40a_pbcor.fits"
COMBINED_NONPBCOR = f"{WORKDIR}/ESO097-013_H40a_nonpbcor.fits"
COMBINED_PB = f"{WORKDIR}/ESO097-013_H40a_pb.fits.gz"

config = AnalysisConfig(
    galaxy="ESO097-013",
    z=0.001448,
    weak_line="H40a",
    strong_line="CS21",  # CO not in this archive product; CS(2-1) used as tracer.
    hrl_pbcor_path=COMBINED_PBCOR,
    hrl_nonpbcor_path=COMBINED_NONPBCOR,
    co_pbcor_path=COMBINED_PBCOR,       # same cube → identity convolve + reproject
    co_nonpbcor_path=COMBINED_NONPBCOR,
    hrl_pb_path=COMBINED_PB,
    co_pb_path=COMBINED_PB,
    # ---- SIGNAL REGION AUDIT ----
    # Centered on NED coordinates for Circinus (RA 213.29152°, Dec
    # -65.33909°), within 1″ of the cube reference pixel (CRVAL =
    # 213.29127°, -65.33902°). Circinus's compact circumnuclear
    # starburst ring lies at r ≲ 10″ from the nucleus and the disk
    # CO emission extends to ~15-20″; using a 35″ side box gives the
    # CO-footprint scan ample margin around the nuclear ring + inner
    # disk while staying inside the usable PB region. PB at box
    # center = 1.0; PB at the four corners (±17.5″ from center,
    # diagonal ≈ 24.7″) ≈ 0.40, which is above the pb_cutoff=0.2
    # floor — sufficient for the CO mom-0 mask construction, although
    # the corner pixels are noisier; the scan will naturally avoid
    # them through the σ-clipped CO threshold. Square box (no
    # dec_size_arcsec) since Circinus is moderately inclined (i~65°)
    # and the ALMA Band-3 beam doesn't strongly favour the disk minor
    # axis; the CO footprint constraint will select the actual disk
    # ellipse within the box.
    signal_region=SkyRegion(
        ra_deg=213.29152, dec_deg=-65.33909,
        size_arcsec=35.0,
    ),
    # ---- NOISE REGION AUDIT ----
    # Off-source NW from NED center. Single-pointing Band 3 at 98 GHz
    # has PB HWHM ≈ 25″; PB > 0.5 out to ~21.5″ diagonal radius. We
    # center the noise box 14″ NW of the nucleus (PB ≈ 0.76 at the
    # box center) with a 7″ side. Sky-coordinate arithmetic:
    #   ΔRA  = -14″ / cos(65.339°) / 3600 = -14 / (3600 × 0.4170) =
    #          -0.009323° → RA  = 213.29152 - 0.009323 = 213.282197°
    #   ΔDec = +14″ / 3600                = +0.003889°
    #          → Dec = -65.33909 + 0.003889 = -65.335201°
    # The 7″ box thus spans RA-Dec corners at PB ∈ ~[0.59, 0.86]
    # (corner pixel diagonals: ~10.5″ inner / ~19.8″ outer from
    # nucleus). All four corners stay above PB = 0.5 → satisfies the
    # noise-region PB > 0.5 floor stated in the task. Distance from
    # the signal box (35″ side, half-extent 17.5″ in RA × cos(dec)):
    # the noise-box outer corner (~19.8″ from nucleus) sits just
    # outside the signal box, ensuring CO emission from the nuclear
    # ring does not bleed into the noise statistics. NW chosen to
    # stay clear of the prominent SE outflow / extended emission
    # axis that Circinus's Sy2 wind drives (the H I + Hα cones lie
    # along PA ≈ 100°, so SE-NW orthogonally avoids it).
    noise_region=SkyRegion(
        ra_deg=213.282197, dec_deg=-65.335201,
        size_arcsec=7.0,
    ),
    # Default ±300 km/s windows. Circinus CO FWHM ~150-200 km/s; the
    # adaptive strong-line window will narrow further if FW(10%) is
    # smaller than 300 km/s.
    output_dir="/Volumes/HouAstro/master/results/ESO097-013",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
