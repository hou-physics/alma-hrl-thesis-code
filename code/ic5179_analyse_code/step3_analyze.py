"""IC 5179 step3 config (Phase Y dual-cube pipeline).

IC 5179: face-on LIRG (D~50 Mpc), classic IRAS RBGS member with strong
compact nuclear starburst. Pure starburst (no AGN dominance), so HRL→SFR
should be clean. NOVEL HRL target — never analyzed by Lau / Bittner /
Toma / Ada.

Observation: ALMA project 2017.1.00255.S, 0.29" synthesized beam,
sub-arcsecond resolution. H30α and CO(2-1) live in separate per-SPW cubes
sharing the same WCS grid (784 x 784 pix at 0.055"/pix), centered on the
NED galaxy position; same execution block, so common-beam convolution will
short-circuit to identity (HRL beam 0.342"x0.289", CO beam 0.341"x0.286"
— differ by <1%). Field of view spans only ~43" total with PB > 0.5
within a ~13" radius — much smaller than the 30"/40-60" envelope assumed
by the generic config template. Signal/noise regions are scaled to the
actual field below.

Pre-flight (2026-04-30):
  H30α observed freq (rest 231.900928 GHz, z=0.011405): 229.286 GHz —
    inside HRL cube (228.837–230.701 GHz). OK.
  CO(2-1) observed freq (rest 230.538 GHz, z=0.011405): 227.938 GHz —
    inside CO cube (227.005–228.874 GHz). OK.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/IC5179"

config = AnalysisConfig(
    galaxy="IC 5179",
    z=0.011405,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/IC5179_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/IC5179_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/IC5179_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/IC5179_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/IC5179_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/IC5179_CO21_pb.fits.gz",
    # ---------- SIGNAL REGION ----------
    # Centered on the NED galaxy position (RA=334.0379°, Dec=-36.8438°),
    # which matches the cube's CRVAL1/CRVAL2 (the ALMA pointing center is
    # at the galaxy nucleus). Box size is 12" (square half-width 6"),
    # chosen to enclose the PB > 0.5 high-fidelity zone (PB drops to 0.5
    # at ~10" radius and to 0.4 at ~13"; outside that the noise is
    # amplified beyond usefulness). This 12" box covers the compact
    # nuclear starburst that dominates IC 5179's IR emission and the
    # innermost CO(2-1) disk — the only region of the cube where a
    # mass-loaded HRL signal is expected. The 30" central box requested
    # by the spec template is impossible: the entire field of view spans
    # only ~43", so a 30" box would fall partly into the PB < 0.4
    # high-noise corona. Morphology-wise, IC 5179 is a face-on LIRG
    # whose CO(2-1) emission fills the inner disk (bbox at threshold
    # ~38"x38" in mom-0), so the mask scan will have plenty of CO
    # structure to follow within the 12" signal box; no need to extend.
    signal_region=SkyRegion(
        ra_deg=334.037871, dec_deg=-36.843758,
        size_arcsec=12.0,
    ),
    # ---------- NOISE REGION ----------
    # Best feasible noise spot in this small-field cube. Source CO
    # emission fills a large fraction of the inner field (the bright-CO
    # bbox spans ~38"x38" of the 43" field at low threshold), so a 40-60"
    # offset is geometrically impossible. After scanning the cube on a
    # 25-pix grid for windows with PB > 0.5 and minimum |mom-0| max,
    # the cleanest position is at pixel (600, 425) → RA=334.033900°,
    # Dec=-36.843254°: offset 11.6" SW of nucleus, PB=0.589 (well above
    # the 0.2 cutoff and above the requested 0.5 floor), and |mom-0|
    # extremes a factor ~20× smaller than the corner regions which suffer
    # from PB-amplified edge noise. Box size 5" (half-width 2.5") to
    # keep the window inside the locally-clean patch and away from the
    # nearest CO-bright pixels. The compromise (closer than the 40-60"
    # spec) is forced by the small archival field; null-test calibration
    # will use a velocity-offset baseline anyway.
    noise_region=SkyRegion(
        ra_deg=334.033900, dec_deg=-36.843254,
        size_arcsec=5.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/IC5179",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
