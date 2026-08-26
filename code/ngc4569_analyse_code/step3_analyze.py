"""NGC 4569 step3 config (Phase Y dual-cube pipeline).

NGC 4569 is a Virgo anemic / HI-stripped spiral, negative z=-0.000784 (blueshifted
due to cluster infall). logM*=10.81, logSFR=0.12 — modest SFR, not a nuclear
starburst, so HRL detection is uncertain.

Pre-flight checks (2026-04-22):
- HRL cube freq range 231.793-233.621 GHz; H30a observed (232.083 GHz) is INSIDE.
- CO21 cube freq range 229.775-231.647 GHz; CO21 observed (230.719 GHz) is INSIDE.
- Both cubes are 200x200 pixels at 0.210"/pix = 42"x42" total FOV.
- PB>0.5 region is only ~26"x26" around the source; PB>0.2 extends to ~42".
- CO emission extent (5-sigma mom-0) is ~12" x 20" centered on NED position.
- Signal region 25" comfortably encloses the CO emission with ~3" margin per side
  inside the PB>0.5 region. No dec_size_arcsec override (NGC 4569 inclination ~65
  deg, not thin like NGC 3628 edge-on).
- Noise region: small (8") box placed at pix (148, 100) i.e. ~10" east of the
  galaxy nucleus, fully outside the CO mom-0 5-sigma mask (CO ends at pix x=128).
  Center PB = 0.69, box mean PB = 0.65 — satisfies rule's PB>0.5 guideline (the
  rule's "~1' away" suggestion is not feasible on this 42" cube, so the noise box
  is placed at the largest feasible separation that still keeps PB>0.5 at center
  and avoids CO emission).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4569"

config = AnalysisConfig(
    galaxy="NGC 4569",
    z=-0.000784,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC4569_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC4569_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC4569_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC4569_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC4569_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC4569_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED optical center of NGC 4569. 25" square encloses the CO(2-1)
        # 5-sigma mom-0 emission (~12" x 20") with margin, fits inside the
        # cube's PB>0.5 region (~26" square).
        ra_deg=189.20753, dec_deg=13.16294,
        size_arcsec=25.0,
    ),
    noise_region=SkyRegion(
        # Placed at pixel (148, 100) — ~10" east of the nucleus, fully outside
        # the CO mom-0 5-sigma mask (CO extent x<=128 in pixel coords).
        # Center PB ~0.69, mean over 8" box ~0.65. Largest feasible separation
        # from galaxy on this 42" cube while keeping PB>0.5 at the center.
        ra_deg=189.20469, dec_deg=13.16287,
        size_arcsec=8.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC4569",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
