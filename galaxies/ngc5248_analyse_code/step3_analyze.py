"""Config for NGC5248 step3 analysis (Phase Y dual-cube pipeline).

NGC 5248 — Table D #2 (PHANGS-ALMA) candidate, 2017.1.00277.S Bendo-class
project, mousid X1296/Xdef, single pointing 1.0″ from NED center, sens
0.81 mJy/bm at 10 km/s. **First-time HRL detection attempt** for this
galaxy (no published reference; not in Toma/Ada Table C). Predicted log
S/N ~ -1.86 (very shallow); detection probability LOW per ranker. SAB(rs)bc
spiral, D ~ 14.9 Mpc, logSFR=+0.36, logM*=10.41 — main-sequence, not
extreme starburst.

Pre-flight (verified 2026-04-27):
  HRL obs freq 231.014 GHz inside HRL cube range [230.780, 232.467] GHz ✓
  CO  obs freq 229.656 GHz inside CO  cube range [228.700, 230.572] GHz ✓
  HRL native beam 0.0928″ × 0.0589″ (sub-arcsec, footprint smoothing applies)
  CO  native beam 0.0922″ × 0.0590″ (essentially identical)
  Same mousid (X1296/Xdef, spw19=H30α, spw25=CO21) → common-beam = identity
  Pixel scale 0.0110″, NAXIS=1920×1920 (single pointing field, ~21″ across)

Field geometry:
  NED center: RA=204.38343°, Dec=8.88525° (13h37m32.0s, +08d53m07.0s)
  HRL pb peak at (204.383630°, 8.885034°) ≈ 1.06″ from NED
  CO mom-0 5σ peak at (204.383355°, 8.885532°) ≈ 1.05″ from NED (NW)
  CO mom-0 5σ extent ~3-5″ in central 6″ box → compact nuclear CO
  PB > 0.5 only within ~12″ radius of pb peak (small single-pointing field)
  PB at ±5″: 0.91 / PB at ±10″: 0.67 / image edges PB → 0

Footprint-fragmentation audit (per task brief): native beam ~0.06-0.09″,
strongly sub-arcsec, so the 1″ co_footprint_smoothing_arcsec default
applies the same fragmentation fix used on NGC 5253 (142 → 5 components)
and NGC 2903 (~0.09″). Inspect step3 stdout / footprint plot for component
count; expected dominant component dominant by similar margin.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5248"

config = AnalysisConfig(
    galaxy="NGC5248",
    z=0.003839,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5248_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5248_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5248_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5248_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5248_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5248_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # AUDIT: Centred on NED RA/Dec (204.38343°, 8.88525°).  HRL pb peak
        # and CO mom-0 5σ peak both fall within ~1.1″ of this center, so a
        # NED-anchored box captures the inner emission. The single-pointing
        # field is small: PB > 0.5 only within ~12″ of pb peak (image is
        # 21.1″ × 21.1″ at 0.011″/pix), so a 30-40″ box that would cover
        # the full inner ring is NOT possible here. An 18″ box (full width)
        # keeps all four edges at PB > 0.55 (verified PB(±9″) ≈ 0.73 along
        # cardinal directions) while comfortably enclosing the compact CO
        # nucleus (5σ mom-0 extent ~3-5″ within the central 6″ region).
        # This box is slightly smaller than NGC 2903's 20″ at the same
        # ~0.09″ beam class because NGC 5248's pb field-of-view is tighter
        # (image is exactly 21″ wide, leaving no room beyond 10″ before
        # PB → 0). NGC 5248 has an inner ring ~30″ across per Table D
        # notes, but only the inner ~18-20″ of that ring lies within the
        # PB > 0.5 region — this matches the matched-filter scope of the
        # mask scan, which selects the brightest CO morphology.
        ra_deg=204.38343, dec_deg=8.88525,
        size_arcsec=18.0,
        dec_size_arcsec=None,
    ),
    noise_region=SkyRegion(
        # AUDIT: 6″ box centred 8″ south of NED phase center, at
        # (204.38343°, 8.88303°). PB min = 0.62 verified within this box
        # (median 0.80) — comfortably above the 0.5 floor. Distance from
        # the CO mom-0 5σ peak is 9.0″, which is well outside the compact
        # CO nucleus (5σ extent ~5″ in radius). A larger 8″ box pushes
        # PB min to 0.58 (still > 0.5) but adds little statistical
        # advantage; 6″ keeps the noise sample compact and squarely within
        # the PB > 0.5 region. The box also avoids the CO 5σ contour by
        # ≥ 4″ at all edges, ensuring sigma-clip on noise channels there
        # gives a line-free per-pixel σ.  No usable larger off-source
        # region exists in this 21″ field — this is the standard
        # constraint at sub-arcsec single-pointing data.
        ra_deg=204.38343, dec_deg=8.88303,
        size_arcsec=6.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5248",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
