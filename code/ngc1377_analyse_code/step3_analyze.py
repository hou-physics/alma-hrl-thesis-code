"""NGC 1377 step3 config (Phase Y dual-cube pipeline).

NGC 1377 — "silent quasar" — is a NOVEL HRL target (no precursor in
Lau / Bittner / Toma / Ada). Aalto+ studies of CS/HCN/HCO+ show the
molecular gas is concentrated in a sub-arcsecond circumnuclear core,
with an outflow possibly powered by a deeply obscured AGN.

The ALMA cube delivered for this analysis is at very high angular
resolution (BMAJ ~ 0.044", BMIN ~ 0.038" — i.e. ~0.04" beam) on a
1024 x 1024 grid at 0.01"/pix, giving a 10.24" x 10.24" field. Primary
beam coverage is uniform across the field (PB > 0.82 everywhere) so any
in-field location is usable. CO(2-1) mom0 peak is at the geometric
centre pixel (512, 512), and emission is unresolved — i.e. concentrated
within a few beams of phase centre, consistent with the Aalto+ compact
core morphology. There is no diffuse extended emission to recover at
this depth, so the 0.04" beam is appropriate (no resolving-out concern
flagged in CLAUDE.md context for this specific target).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC1377"

config = AnalysisConfig(
    galaxy="NGC 1377",
    z=0.005977,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC1377_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC1377_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC1377_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC1377_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC1377_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC1377_CO21_pb.fits.gz",
    # ----- signal_region -----
    # Coordinates: phase centre of the ALMA observation,
    #   RA = 54.16281 deg, Dec = -20.90196 deg (= 03h36m39.07s, -20d54m07.05s).
    #   This matches NED's NGC 1377 nucleus position to within 0.1" and is
    #   confirmed as the dynamical centre by the CO(2-1) mom-0 peak landing
    #   exactly on the central pixel (512, 512) of the 1024x1024 cube.
    # Size: 6.0" — NGC 1377's molecular gas (Aalto+2012 SMA + ALMA studies)
    #   is concentrated in a sub-arcsecond circumnuclear core (~0.5"-1"
    #   diameter for the bright CS/HCN/HCO+ core), with a fainter
    #   ~5" extent in low-J CO. With a 10.24"x10.24" total field, a 6"
    #   signal box leaves ~2" margin on each side for clean noise sampling
    #   and is comfortably larger than the entire CO emission envelope.
    # Galaxy-specific morphology: compact obscured AGN host with potential
    #   molecular outflow; emission expected to be unimodal and centred
    #   (CO mom-0 peak coincides with phase centre, value 0.87 Jy/beam km/s
    #   at peak vs noise std 0.04 in clean field — strong, compact, central).
    signal_region=SkyRegion(
        ra_deg=54.16281, dec_deg=-20.90196,
        size_arcsec=6.0,
    ),
    # ----- noise_region -----
    # Coordinates: NW corner of the field, RA = 54.16380, Dec = -20.90288
    #   (= pixel (180, 180) in the 1024x1024 grid; ~3.3" SW of the
    #   geometric centre converted to sky offsets gives a ~4.7" diagonal
    #   separation from the nucleus).
    # Size: 1.5" — large enough to give robust sigma-clip statistics
    #   (~150x150 pix = ~22500 pixels) but small enough to fit comfortably
    #   inside the field without crossing the central CO emission.
    # Galaxy-specific morphology: NGC 1377's molecular emission is confined
    #   to a sub-arcsecond core, so a region 4.7" off-centre is well clear
    #   of any line-emitting structure. PB > 0.82 throughout, so flux scale
    #   is uniform. Verified empirically on CO mom-0 over +-300 km/s window
    #   centred on the redshifted CO line: NW box mean ~ 0, std ~ 0.044
    #   Jy/beam km/s, max 0.15 (vs centre peak 0.87) — clean noise region.
    noise_region=SkyRegion(
        ra_deg=54.16380, dec_deg=-20.90288,
        size_arcsec=1.5,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC1377",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
