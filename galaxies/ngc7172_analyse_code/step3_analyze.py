"""NGC 7172 step3 config (Phase Y dual-cube pipeline).

NGC 7172 is a Sy2 LIRG with a circumnuclear starburst, edge-on Sa
host. NOVEL HRL target — Lau 2019 Table 2.1 listed it but did not
analyse it (no precursor in Lau / Bittner / Toma / Ada). AGN-host
caveat: the H30a signal may include a small NLR contribution; the
thesis discussion focuses on whether the SB component dominates.

Pre-flight:
  H30a observed = 231.900928 / (1+0.008726) = 229.897 GHz -> inside
    HRL cube range 229.514--231.366 GHz                   OK
  CO21 observed = 230.538    / (1+0.008726) = 228.546 GHz -> inside
    CO  cube range 227.630--229.499 GHz                   OK
  Both cubes share grid (2160x2160 at 0.02"/pix, 43.2"x43.2") and
  near-identical synthesised beams (HRL 0.125"x0.106" BPA -84.4;
  CO  0.135"x0.107"). Project 2018.1.00248.S, very long baseline
  (0.10" resolution) -- structurally tight PB footprint.

PB footprint is very tight at this resolution:
  PB >= 0.5 reaches r ~= 13.0" only (cube center pixel 1080,1080)
  PB >= 0.3 reaches r ~= 16.7"
  PB >= 0.2 reaches r ~= 18.9"
The user-prompt's "40-60" offset noise region" is geometrically
impossible -- the entire cube is only 43" across and PB >= 0.5 has
only a 26"-diameter disk. The noise box is therefore placed as far
from the disk as the PB >= 0.5 envelope allows; this constraint is
documented in the audit comment on `noise_region` below.

CO morphology from CO21 non-pbcor mom-0 over +/-300 km/s:
  Edge-on disk emission concentrated along E-W direction at the
  galaxy nucleus. >5sigma bbox 10.6" x 1.06" centered at pixel
  (1152, 1136) -- i.e. ~1.7" E and ~1.5" N of the cube/WCS centre
  (1080, 1080), close to the NED nucleus (330.508, -31.870)
  ~= pixel (1065, 1060). The disk is thus oriented essentially
  east-west; clean off-disk noise must sit north or south.
  Real CO is confined to r<~5" along the major axis and <~1"
  perpendicular.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7172"

config = AnalysisConfig(
    galaxy="NGC 7172",
    z=0.008726,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC7172_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC7172_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC7172_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC7172_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC7172_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC7172_CO21_pb.fits.gz",
    # ----- signal_region -----
    # Coordinates: NED nucleus position for NGC 7172,
    #   RA = 330.508 deg, Dec = -31.870 deg (= 22h02m01.92s, -31d52m12s).
    #   This is consistent with the cube WCS reference (CRVAL1=330.5079,
    #   CRVAL2=-31.86989) to within 0.1" and lies essentially at the PB
    #   maximum.
    # Size: 20" — CO(2-1) bright emission spans ~10.6" along the edge-on
    #   major axis (E-W) with ~1" minor-axis extent at the >5 sigma level
    #   (CO mom-0 bbox check: x pixels 888-1416, y pixels 1109-1162). A
    #   20" signal box (radius 10") fully encloses the CO-bright disk plus
    #   ~5" margin for the threshold scan to explore lower-S/N envelope
    #   contours. The user's prompt suggested ~30" central, but the
    #   PB >= 0.5 envelope has radius only 13" at this 0.10" beam, so a
    #   20" box (half-side 10") is the largest box that still keeps the
    #   majority of its area within PB >= 0.5 (corners at radius ~14"
    #   sit just outside the PB >= 0.5 disk but well within PB >= 0.3,
    #   which is the actual flux integration cutoff via pb_cutoff=0.2).
    #   Going larger (30") would push corners to PB ~0.2 where pbcor
    #   noise amplification hurts threshold determination.
    # Galaxy-specific morphology: edge-on Sa with circumnuclear starburst;
    #   emission expected to be unimodal and concentrated at the dynamical
    #   centre. mask_components_mode left at default "center" because the
    #   morphology is compact (single emission knot at the nucleus, not
    #   distributed face-on disk like NGC 3627).
    signal_region=SkyRegion(
        ra_deg=330.508, dec_deg=-31.870,
        size_arcsec=20.0,
    ),
    # ----- noise_region -----
    # Coordinates: SE of nucleus, RA = 330.5055, Dec = -31.8717
    #   (= pixel (1450, 750) in the 2160x2160 grid; ~7.7" E and ~6.2" S
    #   of the galaxy centre, total separation 9.9"). Placed below the
    #   edge-on disk plane (which runs E-W at y~1135) to avoid any
    #   extra-planar gas contamination (the NE candidate at matching
    #   distance had CO mom-0 mean +0.0056 with max 0.39 Jy/beam km/s,
    #   suggesting residual diffuse emission; the SE candidate has
    #   mean -0.0007 with max 0.055, clean).
    # Size: 10" — large enough to give robust sigma-clip statistics
    #   (~250000 valid pixels) and to fit entirely within PB >= 0.5
    #   (PB at the box centre is 0.683; 5" half-extent in any direction
    #   keeps PB > 0.5 across the box).
    # Constraint note: the user's prompt suggested 40-60" offset, but
    #   the cube field is only 43" across and PB >= 0.5 has radius 13";
    #   noise was placed as far as the PB envelope allows while staying
    #   off the edge-on disk plane and inside PB >= 0.5. This is the
    #   PB-limited equivalent of the "as far as possible" fallback noted
    #   in the analyse-galaxy task spec.
    # Verified empirically on CO mom-0 over +/-300 km/s: SE box mean
    #   ~ -0.001, std ~ 0.016 Jy/beam km/s, max 0.055 — i.e. clean noise
    #   region, no detectable CO signal contamination, no PB edge effects.
    noise_region=SkyRegion(
        ra_deg=330.5055, dec_deg=-31.8717,
        size_arcsec=10.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC7172",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
