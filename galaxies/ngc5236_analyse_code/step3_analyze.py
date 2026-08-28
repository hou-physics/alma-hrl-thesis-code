"""NGC 5236 / M83 step3 config (Phase Y dual-cube pipeline, Plan B ARI-L tier).

Nearby grand-design barred spiral with a nuclear starburst ring, d ~ 7.4 Mpc,
IRAS compact-starburst pivot target (R_compact = +1.69). ALMA project
2015.1.00121.S, mousid uid://A001/X2fe/X34, re-imaged via the ARI-L pipeline
(12-m compact config, ~1.8" x 0.96" beam). Per-SPW ARI-L delivery:
CO(2-1) in spw0 @ ~230.25 GHz tuning and H30a in spw1 @ ~231.82 GHz tuning,
same WCS grid (1500x1344 pix at 0.18"/pix = 270"x242" field), 1916 chans at
dν = 976.47 kHz (dv ~ 1.27 km/s at band center).

Plan A (re-imaging 2013.1.01161.S scenario-C MS from raw ASDM) was blocked
by a CASA-4.3.1-era calibration script incompatibility -- see
work_dir/NGC5236_reimaging/stage_log.md. Plan B uses the ARI-L pre-calibrated
FITS tier.

Pre-flight verification (2026-04-24):
  HRL cube freq range 230.883-232.753 GHz, H30a obs = 231.900928/(1+0.0017)
    = 231.507 GHz  -> IN RANGE (624 MHz above low edge, well-centered)
  CO  cube freq range 229.312-231.182 GHz, CO21 obs = 230.538/(1+0.0017)
    = 230.147 GHz  -> IN RANGE (835 MHz above low edge)
  Beams: HRL 1.813"x0.962" @ PA -89.4 deg;  CO 1.831"x0.965" @ PA -89.6 deg
    -> ~1% mismatch (well within 10% tolerance). Common-beam convolution
    is nearly identity.
  WCS identical in celestial axes across all 4 cubes (same project, same
    mosaic) -> reprojection step is trivial.
  Pixel scale: 0.18"/pix (CDELT1,2 = 5e-5 deg).
  NGC 5236 not in docs/rules/exceptions.md (verified).

Note on RESTFRQ header oddity:
  HRL cube header RESTFRQ = 232.213 GHz (not 231.901 GHz for H30a) -- this
  appears to be a pipeline-chosen tuning center, not the physical line. The
  _step3 package overrides it via weak_line="H30a" -> 231.90092784 GHz from
  the internal line table, so the config does NOT need weak_line_rest_ghz.

CO non-pbcor moment-0 over +-300 km/s of CO center (472 chans):
  - mom0 sigma (line-free central box) = 0.253 Jy/beam km/s
  - peak 71.69 Jy/beam km/s at pix (759,705) = RA 204.251795, Dec -29.863910
    -> 283 sigma, clean detection
  - 8.67" from NED center (204.25396, -29.86542)
  - 6.16" from mosaic pointing center (brief said 3.6", pipeline-reported
    offset is larger in the ARI-L tier; still well within PB)
  - 50 sigma connected component around peak: 14.8" x 22.9" bbox
    (nuclear starburst ring, matches Handa+1990/Muraoka+2009 literature)
  - 100 sigma core: 12.1" x 15.8"
  - Disk (>5 sigma mom0) extends well beyond 90" x 75" in this central
    region, with spiral arms threading most of the 270" field -- a
    strictly line-free noise box inside the field requires accepting a
    small amount of grazing CO contamination.

Signal region: 40" square centered on CO mom-0 peak at
  RA=204.251795, Dec=-29.863910 (J2000).
  - Fully encloses the 15"x23" bright ring + ~8"-12" buffer
    (ring corner-to-signal-box edge)
  - Stays inside PB = 0.78 - 0.98 (mean 0.91)
  - Comfortably larger than the 100-sigma core; threshold scan has room
    to explore 3-32 sigma without clipping at the box edge
  - Used by both signal region AND mask center (cx, cy = center of box)

Noise region: 15" square at pixel (1066, 496) = RA=204.234093,
  Dec=-29.874358 (J2000), SE offset from ring centroid by 67" (9.1 kpc
  at 7.4 Mpc).
  - PB_min = 0.86, PB_mean = 0.86 -> deep in the primary beam
  - mom0_mean = +0.05 sigma (= +12 mJy/beam km/s) -- essentially zero
  - mom0_max = +5.05 sigma (spatial outlier pixel, not a systematic bias)
  - 63" from CO peak, outside the >3sigma disk core near the ring
  - 4.5% of box pixels graze a 5-sigma-dilated CO5sig mask (interarm
    spiral-arm tail edges) -- unavoidable in a mosaic saturated with CO
    disk; the sigma-clip noise estimator in the pipeline (calc_flux_sn)
    excludes channels near line center anyway, so line-free noise stats
    are preserved.
  - Cleanest candidate found by exhaustive 5-pix-stride grid search over
    all PB>0.5 / d_COpeak>40" positions in the image.

Expectations:
  - No direct H30a literature for M83 (Toma analyzed H40a in a different
    2015.1.01177.S project/band). Order-of-magnitude check: H30a total
    flux for a thermal free-free source should be comparable to H40a's
    tens of mJy km/s ... ~ few hundred mJy km/s to ~1 Jy km/s range.
  - Anything > 10 Jy km/s triggers a conflict.
  - Nuclear ring geometry is resolved (many beams across the ring), so
    flux integration with beam conversion = Σ(Jy/beam) / beam_area_pix
    should give the total integrated line flux.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5236"

config = AnalysisConfig(
    galaxy="NGC 5236",
    z=0.0017,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5236_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5236_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5236_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5236_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5236_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5236_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # Centered on CO mom-0 emission peak (pixel 759,705), which sits
        # 8.67" from NED center and 6.16" from the mosaic pointing center.
        # 40" box encloses the 15"x23" >50sigma nuclear starburst ring plus
        # ~8-12" buffer for threshold scanning. Stays inside PB = 0.78-0.98.
        ra_deg=204.251795, dec_deg=-29.863910,
        size_arcsec=40.0,
    ),
    noise_region=SkyRegion(
        # 15" square 67" SE of ring centroid, PB = 0.86, mom0_mean +0.05
        # sigma (essentially zero). Outside the >3sigma disk near the ring.
        # 4.5% CO5sig grazing from a far-disk spiral arm edge is accepted
        # (unavoidable in M83's mosaic; does not bias line-free noise).
        ra_deg=204.234093, dec_deg=-29.874358,
        size_arcsec=15.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5236",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
