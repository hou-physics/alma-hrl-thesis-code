"""Config for NGC5135 step3 analysis (Phase Y dual-cube pipeline).

NGC 5135: Sy2 + circumnuclear starburst LIRG, D ~ 60 Mpc (Lau Table 2.1).
Last viable archival candidate from the multi-line/audit pool.
ALMA project 2019.1.00373.S — Band 6, single-pointing 12m, sens 0.44 mJy/bm.
HRL on spw1 (228.844 GHz tuning), CO(2-1) on spw0 (227.410 GHz tuning).

Pre-flight:
  H30a obs = 231.900928 / 1.013693 = 228.768 GHz, in HRL cube [228.000, 229.688]
  CO21 obs = 230.538    / 1.013693 = 227.424 GHz, in CO  cube [226.473, 228.347]
HRL beam 0.230"x0.186"; CO beam 0.255"x0.194" — same project, ~10% mismatch
will be handled by common-beam convolution.
NED: RA=201.43345 deg, Dec=-29.83344 deg. CO peak at (RA=201.43392,
Dec=-29.83283) is ~3" from cube center / NED — circumnuclear ring.
Pixel scale 0.035"/pix; cube FoV ~44" full side; PB > 0.5 within ~12-13"
of pointing center (long-baseline ari_l product, narrow PB).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5135"

config = AnalysisConfig(
    galaxy="NGC5135",
    z=0.013693,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC5135_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC5135_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC5135_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC5135_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC5135_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC5135_CO21_pb.fits.gz",
    signal_region=SkyRegion(
        # AUDIT COMMENT: NED center (RA=201.43345, Dec=-29.83344) matches
        # cube center to within 1 px; CO peak ~3" from NED. NGC 5135 hosts
        # a compact circumnuclear starburst ring (~5") embedded in a Sy2
        # disk, so 20" box generously encloses the ring + scan margin while
        # remaining inside the PB > 0.5 envelope (single-pointing 12m
        # Band 6 ari_l product has narrow PB; PB > 0.5 only within ~12-13"
        # of pointing centre, so 20" box reaches PB ~ 0.4 at corners but
        # well-into PB > 0.5 across the body of the source). A larger 30"
        # box would extend beyond the science-usable PB envelope of this
        # narrow-PB long-baseline product.
        ra_deg=201.43345, dec_deg=-29.83344,
        size_arcsec=20.0,
        dec_size_arcsec=None,
    ),
    noise_region=SkyRegion(
        # AUDIT COMMENT: clean line-free region offset ~10.5" west of CO
        # peak (pixel 294,713 in 1250x1250 cube), at PB_center=0.56,
        # PB_box_min=0.39 over a 5" box — comfortably above the 0.2 PB
        # cutoff. CO mom-0 in this 5" box: mean=-58 std=106 max=262
        # (compare CO peak max=3711) — confirms the box does not touch
        # the circumnuclear ring or its diffuse skirt. Smaller box (5")
        # used because the narrow PB envelope leaves little room between
        # the source and the PB > 0.4 horizon.
        ra_deg=201.43728, dec_deg=-29.83283,
        size_arcsec=5.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC5135",
)

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
