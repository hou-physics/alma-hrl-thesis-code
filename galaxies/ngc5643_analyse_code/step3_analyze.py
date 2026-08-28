"""Config for NGC 5643 step3 analysis (Phase Y dual-cube pipeline).

Galaxy context (Ada-reproduction batch):
  NGC 5643 — SAB(rs)c Seyfert 2 with prominent stellar bar and circumnuclear
  starburst surrounding the AGN. Distance ~17 Mpc.
  z = 0.003999. Project 2016.1.00254.S, native beam ~0.13" (HRL: 0.198x0.120",
  CO: 0.193x0.120"; same-mousid sister of NGC 7582 — common-beam convolution
  near-identity). Pixel scale 0.024"/pix, image 1764x1764 (PB>0.5 region
  extends to ~14" radius from nucleus).

Pre-flight verification (passed):
  H30a observed 230.9772 GHz inside HRL cube range 230.146--232.009 GHz.
  CO21 observed 229.6198 GHz inside CO cube range 228.690--230.553 GHz.
  Channel width 5.10 km/s; 117 channels span the +/-300 km/s line window.
  NED coords RA=218.169630, Dec=-44.174430 (J2000).
  CO peak at (RA=218.169592, Dec=-44.174451), 0.12" S-W of NED nucleus —
    essentially at the AGN/pointing centre. Peak Jy/beam·km/s = 4.22 in the
    +/-300 km/s window (vs 4.05 for NGC 7582 — CO brighter here).
  Empirical mom-0 noise (sigma-clipped over 5 random off-line realisations)
    = 0.0495 Jy/beam·km/s.
  CO emission extents (empirical noise):
    20σ envelope:  1.4" x  2.7" (compact AGN-illuminated nucleus)
    10σ envelope: 23.4" x 32.4" (bar + circumnuclear disk; max_sep 18.8")
     5σ envelope: 37.7" x 35.8" (low-level extended disk; reaches PB<0.5)
  PB>0.5 region: max separation 14.0" from NED.

Literature reference (sanity bounds):
  Ada Bittner 2022: 3σ upper limit (non-detection) — nuclear extraction.
  NGC 5643 is a Seyfert 2 (LINER/AGN dominates), so even if circumnuclear
    star formation is present, ionised-gas RRL emission is expected weak.
  Expectation: non-detection or marginal (S/N <= 5). >5x literature upper
    limit (i.e. anticipated S/N > 10) triggers Conflict Report per
    docs/rules/_conflict-protocol.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5643"

config = AnalysisConfig(
    galaxy="NGC5643",
    z=0.003999,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5643_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5643_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5643_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5643_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5643_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5643_CO21_pb.fits.gz",
    # ──── Signal region ─────────────────────────────────────────────────
    # Centred on the NED nuclear position (RA=218.169630, Dec=-44.174430).
    # The empirical CO peak sits 0.12" away — essentially identical, well
    # within the 0.13" archival beam — so NED coords are used as canonical
    # centre (Ada used the same convention for sister galaxies).
    # Box size 24" chosen to encompass the 10σ CO bar/disk envelope
    # (23.4" x 32.4" bbox) while staying inside the PB>0.5 region (which
    # extends to ~14" radius from nucleus = 28" full diameter). 24" gives
    # ~1.7x linear margin around the bright bar so the smoothing+threshold
    # scan has working room without leaking into the PB<0.5 mosaic ring.
    # NGC 5643 has a more extended molecular bar than NGC 7582's compact
    # ring (~5" envelope) — hence the larger signal box. Square geometry
    # is appropriate (face-on barred spiral, no strong projection).
    signal_region=SkyRegion(
        ra_deg=218.169630,
        dec_deg=-44.174430,
        size_arcsec=24.0,
    ),
    # ──── Noise region ──────────────────────────────────────────────────
    # 12" East of NED nucleus at (RA=218.174278, Dec=-44.174430). Diagnostic
    # sweep over eight 8" cardinal/diagonal offsets at 12" radius picked
    # East as the location with the lowest CO mom-0 contamination
    # (mean=-0.105, max=0.278 Jy/beam·km/s) while keeping PB highest among
    # candidates (PB_mean=0.601). Western offset has slightly cleaner CO
    # but PB=0.517 (too close to PB>0.5 cutoff for safety margin). 12"
    # separation places the noise box on the edge of the 5σ extended disk
    # but entirely outside the brighter 10σ bar emission — acceptable
    # because the noise estimator uses sigma-clipping which rejects any
    # residual CO line voxels. Square 8" box matches NGC 7582 noise
    # geometry for direct sister comparability.
    noise_region=SkyRegion(
        ra_deg=218.174278,
        dec_deg=-44.174430,
        size_arcsec=8.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5643",
)

# Mode selection: NGC 5643 is a Seyfert 2 with a compact AGN-dominated
# nucleus (20σ CO envelope only ~1.4"x2.7"), embedded in a more extended
# bar/disk (10σ ~23"x32"). The default `box` signal-region mode and
# `center` mask-components mode are appropriate (matches Ada's
# pointing-based analysis and the NGC 7582 sister script). Do NOT switch
# to `dilated_footprint` or `all` modes unless box clearly fails — the
# Phase Y baseline subtraction + tracer mask are designed for compact
# circumnuclear extraction, which is the relevant geometry here.

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
