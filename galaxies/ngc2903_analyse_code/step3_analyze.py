"""Config for NGC2903 step3 analysis (Phase Y dual-cube pipeline).

NGC 2903 — IRAS-pivot deep-tier candidate, 2018.1.00517.S Bendo-class,
single pointing, mousid uid://A001/X133d/Xdca, sens 0.72 mJy/bm at 10 km/s.
Toma reports a non-detection / upper limit for H30α (Table C reference);
this analysis tests whether the deep 0.05″-class data reveal compact
nuclear emission below Toma's threshold.

Pre-flight (verified 2026-04-27):
  HRL obs freq 231.484 GHz inside HRL cube range [231.204, 232.985] GHz ✓
  CO  obs freq 230.124 GHz inside CO  cube range [229.181, 231.051] GHz ✓
  HRL native beam 0.098″ × 0.075″ (announced 0.05″, actual ~0.09″)
  CO  native beam 0.089″ × 0.074″
  Same mousid (X133d/Xdca, spw19=H30α, spw25=CO21) → common-beam should be identity
  CO mom-0 peak @ (143.04333°, 21.50199°) — 4.2″ NE of NED phase center

Footprint-fragmentation audit (per task brief): native beam ~0.09″ is
sub-arcsecond, so the 1″ co_footprint_smoothing_arcsec default applies
the same fragmentation fix used on NGC 5253 (142 → 5 components).
Inspect step3 stdout / footprint plot for component count.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC2903"

config = AnalysisConfig(
    galaxy="NGC2903",
    z=0.0018,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC2903_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC2903_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC2903_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC2903_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC2903_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC2903_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # AUDIT: Centred on NED RA/Dec (143.04213°, 21.50083°), which
        # coincides with the single-pointing mosaic phase center (PB = 1.0).
        # CO mom-0 peak sits 4.2″ NE of this center (143.04333°, 21.50199°),
        # consistent with the compact circumnuclear starburst in NGC 2903.
        # 20″ box (full width) covers the CO bright zone with comfortable
        # scan margin while staying inside PB > 0.65 along all four edges
        # (PB=0.91 at ±5″, PB=0.67 at ±10″ in the cardinal directions).
        # This matches the NGC 5253 deep-native sub-arcsec config geometry
        # used as the reference template (single pointing, 20″ box, compact
        # nuclear starburst expected per IRAS classification).
        ra_deg=143.04213, dec_deg=21.50083,
        size_arcsec=20.0,
        dec_size_arcsec=None,
    ),
    noise_region=SkyRegion(
        # AUDIT: 15″ box centred 10″ south of phase center, at
        # (143.04213°, 21.49805°). PB = 0.67 verified at this position
        # — comfortably above the 0.5 floor required for clean noise
        # estimation. The 15″ aperture clears the CO 3σ envelope (CO
        # emission is concentrated within ~5″ of the nucleus per the
        # mom-0 inspection); minimum distance from CO peak ≈ 14″ so the
        # box edge is ≥ 4″ outside any plausible CO footprint at the
        # native sub-arcsec resolution. The location is also well outside
        # the HRL line channel range, so σ-clip on noise channels there
        # gives a line-free per-pixel σ.
        ra_deg=143.04213, dec_deg=21.49805,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC2903",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
