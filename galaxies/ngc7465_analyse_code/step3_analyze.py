"""Config for NGC 7465 step3 analysis (Phase Y dual-cube pipeline).

Ada (2022) §4.8 reproduction. Sy 2 (Markarian 1290), z=0.006538, D~27 Mpc.
Strong tracer = HCO+(1-0) (CO not in this Band 3 SPW set; spw0=HCO+,
spw2=H40α). Same project / same execution block; HRL beam 1.28"x1.00",
HCO+ beam 1.41"x1.10" — common-beam convolution will be near-identity.

Pre-flight checks (2026-04-29):
- H40α observed = 99.022952 / 1.006538 = 98.380 GHz, inside spw2
  (97.450 - 99.318 GHz). OK.
- HCO+(1-0) observed = 89.188523 / 1.006538 = 88.609 GHz, inside spw0
  (87.408 - 89.275 GHz). OK.
- NED nucleus: RA = 345.504°, Dec = +15.96477° (matches cube CRVAL).
- HCO+ smoothed-1" 5σ nuclear component is 4.4" x 3.8" (max radius
  ~3" from nucleus); compact starburst/Sy 2 morphology, M0 mode is
  the right prior.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))
from analyze import analyze
from config import AnalysisConfig, SkyRegion

config = AnalysisConfig(
    galaxy="NGC7465",
    z=0.006538,
    weak_line="H40a",
    strong_line="HCO+10",
    hrl_pbcor_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_H40a_pbcor.fits",
    hrl_nonpbcor_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_H40a_nonpbcor.fits",
    co_pbcor_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_HCO+10_pbcor.fits",
    co_nonpbcor_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_HCO+10_nonpbcor.fits",
    hrl_pb_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_H40a_pb.fits.gz",
    co_pb_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7465/NGC7465_HCO+10_pb.fits.gz",
    signal_region=SkyRegion(
        # AUDIT COMMENT: NGC 7465 = Sy 2 (Markarian 1290), compact Seyfert
        # nucleus. NED nucleus 345.504°, +15.96477° (= cube pointing center,
        # PB = 1.0 here). HCO+ smoothed-1" 5σ nuclear connected component
        # measures only 4.4" x 3.8" (max radius ~3" from nucleus); a 24"
        # box gives ~10" margin all around — comfortable scan latitude
        # without including outer noise pixels. Single-pointing 12m PB
        # FWHM at 88.6 GHz is ~67"; 24" stays well within PB > 0.5
        # (PB ~1.0 at nucleus, ~0.7 at 25" radius). Compact morphology
        # ⇒ M0 (box + mask_components_mode='center') is the correct
        # prior for a Sy 2.
        ra_deg=345.504, dec_deg=15.96477,
        size_arcsec=24.0,
        dec_size_arcsec=None,
    ),
    noise_region=SkyRegion(
        # AUDIT COMMENT: noise box placed 25" north of nucleus —
        # cleanest line-free patch found in 8-direction sweep:
        # mom0_med = 0.0003, mom0_std = 0.008 (vs east-25" mom0_med
        # 0.0024 = residual disk emission and SW-30" mom0_std 0.019 =
        # PB edge). Position RA = 345.504°, Dec = +15.97171° (Δdec
        # = +25"/3600). PB ~ 0.69 here, well above 0.5 cutoff and
        # outside the ~3"-radius HCO+ nuclear component (25" >> 3").
        # 8" box at 0.19"/pix ≈ 42 pix — enough samples for σ-clip
        # noise stat without nudging into adjacent structure.
        ra_deg=345.504, dec_deg=15.97171,
        size_arcsec=8.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC7465",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
