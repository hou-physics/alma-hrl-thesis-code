"""NGC 7130 step3 config (Phase Y dual-cube pipeline).

Seyfert 2 / starburst composite LIRG. Compact high-resolution ALMA Band 6
pointing (0.088" pixels, cube spans ~45" on a side). CO(2-1) and H30a live
in separate SPWs (spw2 at 226.842 GHz and spw3 at 228.184 GHz respectively).
Primary beam is tight: PB>0.5 only within r~12", PB>0.2 only within r~20".

CO morphology from non-pbcor moment-0 over ±400 km/s:
  - >10sigma connected emission extends to r~6.5" (compact nuclear source)
  - >20sigma (bright core) only r~3.5"
So a 20" signal region around NED center fully encloses the CO-bright zone
with ample margin for mask scanning while remaining inside PB>0.5.

Noise region is more constrained because the field is so compact. The
cleanest >=15" offset box available has PB_min=0.20 (equals pipeline
pb_cutoff default). This is a geometry-of-cube constraint, not a choice:
we cannot simultaneously be offset beyond the r~7-10" CO extent AND keep
PB>0.5. The 15" box at PB_min=0.20, PB_mean=0.42 has mom0 mean consistent
with 0 within the moment-0 sigma (0.065 Jy/km/s/beam per pixel), which is
what sigma_clip_noise needs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7130"

config = AnalysisConfig(
    galaxy="NGC 7130",
    z=0.016151,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC7130_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC7130_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC7130_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC7130_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC7130_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC7130_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED NGC 7130 J2000: 21h48m19.53s -34d57m04.50s -> 327.08137, -34.95125.
        # CO21 non-pbcor mom-0 >10sigma connected component spans r<=6.5"; the
        # >5sigma extent (dominated by noise peaks across 262k-pixel field)
        # reaches 19" but is not real emission. 20" side length comfortably
        # encloses the real compact source with room for threshold scanning,
        # and sits entirely inside PB>0.5 (r<12").
        ra_deg=327.08137, dec_deg=-34.95125,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # Offset 17.6" SW of NED center (pixel (421, 155) in 512x512 CO cube).
        # Grid search for the 15" box minimizing |<CO mom-0>|/sigma across
        # positions satisfying PB_min >= 0.2 and dist > 17.5": this location
        # gave mean = -0.0002 Jy/km/s/beam (essentially 0 vs sigma 0.065).
        # PB_min=0.20 (at pb_cutoff default), PB_mean=0.42, NaN fraction 38%
        # (outside primary beam cutoff). sigma_clip_noise uses only valid
        # pixels, which still number ~17k within the box -- ample statistics.
        # PB>0.5 offset box of this size is geometrically impossible given
        # the compact single-pointing field (PB>0.5 only r<12").
        ra_deg=327.07629, dec_deg=-34.95378,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC7130",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
