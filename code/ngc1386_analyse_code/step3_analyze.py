"""NGC 1386 step3 config (Phase Y dual-cube pipeline).

Seyfert 2 galaxy in the Fornax cluster (z=0.003052). ALMA Band 6 mosaic
(project 2016.1.01279.S -- same project as NGC 4945, 5643, 6810, IC 5063),
0.46" angular resolution. CO(2-1) and H30a live in separate SPWs (CO in
spw0 at 228.97--230.79 GHz, H30a in spw1 at 230.53--232.35 GHz). Cubes
are 750 x 750 x 234 (0.085" px, ~64" x 64"). Edge-on (inclination ~65),
narrow N-S disk.

Pre-flight (verified 2026-04-21):
  H30a observed = 231.900928 / (1+0.003052) = 231.1953 GHz -> inside
    HRL cube range 230.5301--232.3506 GHz  OK
  CO21 observed = 230.538    / (1+0.003052) = 229.8365 GHz -> inside
    CO  cube range 228.9731--230.7936 GHz  OK
  Both cubes share the same pixel grid (750x750, CRVAL1=54.191869,
  CRVAL2=-36.000265) and similar beams (HRL 0.468"x0.429" BPA -27.3;
  CO 0.490"x0.442" BPA -38.3) -> common-beam convolution is near-identity.
  NED NGC 1386 J2000: 03h36m46.3s -35d59m58s -> 54.19292, -35.99944
  (matches cube reference closely; PB peak at NED center = 0.983).

Mosaic PB footprint is tight for this single-galaxy pointing:
  PB>=0.5 reaches r=19.9", PB>=0.2 reaches r=27.1".

CO morphology from CO21 non-pbcor mom-0 integrated over +/-400 km/s
(mom0 sigma = 0.050 Jy/beam*km/s):
  >20 sigma: r <= 2.5" bbox 2.4" x 2.6" (bright sub-arcsec nuclear core)
  >10 sigma: r <= 14.4" bbox 15.7"(RA) x 21.7"(Dec) -- clear Dec elongation
    consistent with edge-on disk (disk position angle roughly N-S)
  >5  sigma: r <= 24.7" bbox 39.3" x 38.0" -- low-level extended wings
  >3  sigma: r <= 26.8" -- largely noise-dominated at cube edges.
The >10 sigma edge-on disk fits inside a 20" square with margin, and
20" stays entirely inside PB>=0.5 (PB min=0.548, mean=0.848). The wider
>5 sigma extension (~39" E-W, ~38" N-S) is already outside PB>=0.5 and
dominated by residual noise + beam wings, so including it would not
help the mask.

A rectangular signal region (18" RA x 26" Dec) was considered to match
the Dec-elongated disk shape, but its corners dip to PB=0.47, and the
square 20" region already fully encloses the >10 sigma CO emission and
matches the IC 5063 (same project) template that worked on an
analogous compact-core + jet-driven Seyfert source. Using 20" square.

Noise region: tight PB footprint (r<=19.9" at PB=0.5) limits placement
options. Grid search across the cube for 12"-side boxes with PB_min>=0.5,
valid_frac>=95%, offset>=10" from nucleus returned 41 candidates. The
winning position is centered ~10" east of the nucleus at pixel (459,419)
-> RA=54.18942, Dec=-35.99923, giving offset 10.2" (beyond the >10sigma
CO emission which extends r<=14.4" in Dec but only ~8" in RA east of
nucleus at >=10 sigma threshold), PB_min=0.565, PB_mean=0.822, mean CO
mom0 essentially zero (-0.0000 Jy/beam*km/s). The east direction is
chosen because the edge-on CO disk elongation is along N/S (Dec axis),
so an eastward offset places the noise box perpendicular to the disk
long-axis in the cleanest off-source direction.

Box size reduced from the usual 15" to 12" for NGC 1386 because the
19.9" PB>=0.5 radius cannot accommodate a 15" box at >=10" offset from
nucleus (half-diagonal of 15" box is 10.6", pushing corners to r>=20.6").
12" box keeps corners inside PB=0.5 and still provides a robust noise
estimate (~19000 pixels after beam-convolution).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC1386"

config = AnalysisConfig(
    galaxy="NGC 1386",
    z=0.003052,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC1386_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC1386_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC1386_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC1386_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC1386_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC1386_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED NGC 1386 J2000: 03h36m46.3s -35d59m58s -> 54.19292, -35.99944.
        # CO(2-1) emission: compact sub-arcsec core (>20sigma r<=2.5") with
        # Dec-elongated disk (>10sigma bbox 15.7" RA x 21.7" Dec, rmax 14.4").
        # 20" square fully encloses >10sigma emission with margin for
        # nbeam=3--7 threshold scanning, and stays inside PB>=0.5
        # (PB min=0.548, mean=0.848). Same size as IC 5063 (same project).
        ra_deg=54.19292, dec_deg=-35.99944,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # 12" box ~10" east of nucleus (pixel (459,419) in 750x750 grid).
        # Grid-search winner by |<CO mom0>| subject to PB_min>=0.5,
        # valid_frac>=95%, offset>=10". PB_min=0.565, PB_mean=0.822, mean
        # CO mom0 -0.0000 Jy/beam*km/s (zero within 1-pixel sigma 0.050).
        # Eastward direction chosen to avoid the N-S edge-on CO disk.
        # Box size 12" (not 15") because PB>=0.5 radius is only 19.9" --
        # a 15" box at >=10" offset would push corners outside PB>=0.5.
        ra_deg=54.18942, dec_deg=-35.99923,
        size_arcsec=12.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC1386",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
