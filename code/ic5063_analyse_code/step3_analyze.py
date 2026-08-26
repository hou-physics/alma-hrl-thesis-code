"""IC 5063 step3 config (Phase Y dual-cube pipeline).

Seyfert 2 AGN with a dusty jet-ISM interaction. ALMA Band 6 **mosaic**
(project 2016.1.01279.S), 0.38" angular resolution. CO(2-1) and H30a
live in separate SPWs (spw0 at 227.969 GHz and spw1 at 229.601 GHz
respectively). Cubes are 972 x 980 x 234 (0.063" px, 61" x 62").

Pre-flight:
  H30a observed = 231.900928 / (1+0.011348) = 229.299 GHz -> inside
    HRL cube range 228.691--230.512 GHz  OK
  CO21 observed = 230.538    / (1+0.011348) = 227.951 GHz -> inside
    CO  cube range 227.059--228.879 GHz  OK
  Both cubes share the same pixel grid (972x980, CRVAL1=313.00665,
  CRVAL2=-57.06747) and similar beams (HRL 0.500"x0.315" BPA 54.4;
  CO 0.502"x0.320" BPA 53.1) -> common-beam convolution is near-identity.

Mosaic PB footprint is modest: PB>=0.5 reaches r=22.6", PB>=0.2 reaches
r=29.9" (despite being a mosaic, the pointing layout is compact around
the Seyfert nucleus). This constrains the noise-region geometry more
than I initially expected, though a mosaic still gives more room than a
single-pointing would.

CO morphology from CO21 non-pbcor mom-0 integrated over +/-400 km/s:
  >20 sigma: r <= 2.5" (sub-arcsec nuclear core -- matches the known
    jet-driven emission line structure)
  >10 sigma: r <= 2.6" (still compact)
  >5  sigma: bbox 4.85" x 2.71" (r<=2.7")
  >3  sigma: extended out to r=28.6" but this is dominated by noise
    peaks, not connected emission.
Real CO emission is thus very compact (r~3"). A 20" signal region
around NED center (313.00982, -57.06876) fully encloses it with ample
margin for threshold scanning and stays entirely inside PB>=0.5 (r=22.6).

Noise region: the compact PB footprint limits how far out a clean 15"
box can sit. Grid search across the cube for 15"-side boxes with
PB_min>=0.5, valid_frac>=95%, offset>=10" from nucleus returned 131
candidates; the winning position is centered NE of the nucleus at
RA=313.00417, Dec=-57.06672 (cx=563, cy=533 in 972x980 grid), giving
offset 13.3" (beyond even the 3-sigma nuclear emission), PB_min=0.513,
PB_mean=0.842, valid_frac=1.00, CO mom0 mean 1.3e-4 Jy/beam*km/s
(zero within sigma=0.068). 13" is ample given the real CO emission is
confined to r<=3".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/IC5063"

config = AnalysisConfig(
    galaxy="IC 5063",
    z=0.011348,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/IC5063_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/IC5063_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/IC5063_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/IC5063_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/IC5063_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/IC5063_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # NED IC 5063 J2000: 20h52m02.36s -57d04m07.54s -> 313.00982, -57.06876.
        # Real CO(2-1) emission from this cube is compact: >5sigma bbox
        # 4.85" x 2.71" (radial <=2.7"), >10sigma radial <=2.6". A 20"
        # signal region encloses the CO-bright nucleus with 8"+ of
        # margin on every side for the mask threshold scan, and sits
        # entirely inside PB>=0.5 (PB>=0.5 radius is 22.6").
        ra_deg=313.00982, dec_deg=-57.06876,
        size_arcsec=20.0,
    ),
    noise_region=SkyRegion(
        # 15" box centered 13.3" NE of nucleus (pixel (563, 533) in
        # 972x980 grid). Grid-search winner by |<CO mom0>| subject to
        # PB_min>=0.5, valid_frac>=95%, offset>=10". PB_min=0.513,
        # PB_mean=0.842, CO mom0 mean +0.00013 Jy/beam*km/s (essentially
        # 0 vs per-pixel sigma 0.068). 13" offset is well beyond the
        # real CO emission extent (r<=3" at >=5sigma), so no galaxy
        # signal contaminates the noise estimate.
        ra_deg=313.00417, dec_deg=-57.06672,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/IC5063",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
