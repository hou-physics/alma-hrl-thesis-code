"""Config for NGC 7582 step3 analysis (Phase Y dual-cube pipeline).

Galaxy context (Ada-reproduction batch):
  NGC 7582 — SBab Seyfert 2 with circumnuclear ionised outflow.
  z = 0.00541. Project 2016.1.00254.S, native beam ~0.13" (HRL: 0.155x0.125",
  CO: 0.161x0.128"; same-mousid → common-beam convolution is identity).
  Pixel scale 0.018"/pix, image 2304x2304 (PB>0.5 region 26x26 arcsec).

Pre-flight verification (passed):
  H30a observed 230.6531 GHz inside HRL cube range 229.857--231.720 GHz
  CO21 observed 229.2975 GHz inside CO cube range 228.402--230.266 GHz
  NED coords RA=349.59845, Dec=-42.37042 (matches CRVAL → at pointing centre).
  CO peak observed at (RA=349.59855, Dec=-42.37080), 1.4" S of NED centre,
    peak Jy/beam·km/s = 4.05 in a fixed +/-300 km/s window.
  CO emission >5sigma extent ~4.9" (RA) x 4.7" (Dec) — compact circumnuclear.

Literature reference (sanity bounds):
  Ada Bittner 2022: peak HRL S/N ~3 (borderline detection-by-technicality),
    peak CO S/N ~2.1.
  Toma 2024 reanalysis: flux <0.4 Jy·km/s (non-detection).
  Expectation: S/N ~2-4 marginal/non-detection. >10x sigma vs literature
  triggers Conflict Report per docs/rules/_conflict-protocol.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7582"

config = AnalysisConfig(
    galaxy="NGC7582",
    z=0.00541,
    weak_line="H30a",
    strong_line="CO21",
    hrl_pbcor_path=f"{WORKDIR}/NGC7582_H30a_pbcor.fits",
    hrl_nonpbcor_path=f"{WORKDIR}/NGC7582_H30a_nonpbcor.fits",
    co_pbcor_path=f"{WORKDIR}/NGC7582_CO21_pbcor.fits",
    co_nonpbcor_path=f"{WORKDIR}/NGC7582_CO21_nonpbcor.fits",
    hrl_pb_path=f"{WORKDIR}/NGC7582_H30a_pb.fits.gz",
    co_pb_path=f"{WORKDIR}/NGC7582_CO21_pb.fits.gz",
    # ──── Signal region ─────────────────────────────────────────────────
    # Center placed on the empirical CO peak (RA=349.59855, Dec=-42.37080),
    # which sits 1.4" South of the NED nuclear position (NED gives 349.59845,
    # -42.37042). The CO moment-0 5σ contour spans only ~4.9"x4.7"
    # (circumnuclear ring of NGC 7582 — the sub-kpc starburst surrounding the
    # Sy 2 core). The 0.13" archival beam fully resolves this region but
    # filters out the larger-scale spiral disk (consistent with Ada's
    # peak-only S/N ≈ 3 result and Toma's non-detection upper limit).
    # Box size 14" gives ~3x linear margin around the bright CO emission so
    # the smoothing+threshold scan has working room without leaking into
    # the PB<0.5 mosaic ring (PB>0.5 limit is ±13" from centre).
    # Square (no `dec_size_arcsec`) since the circumnuclear ring is
    # geometrically isotropic at this resolution.
    signal_region=SkyRegion(
        ra_deg=349.598549,
        dec_deg=-42.370802,
        size_arcsec=14.0,
    ),
    # ──── Noise region ──────────────────────────────────────────────────
    # 8" North of NED centre at (RA=349.59845, Dec=-42.36820). Diagnostic
    # sweep over six 8" offset candidates picked North as the location with
    # minimum CO contamination (mom-0 mean=-0.057, max=0.479 Jy/beam·km/s)
    # while keeping PB sufficiently high (PB_mean=0.73, well above the
    # PB>0.5 threshold rule). 8" separation puts the noise box clearly
    # outside the ~5" CO 5σ envelope. Square 8" box matches typical
    # pipeline noise-statistic area for this beam scale.
    noise_region=SkyRegion(
        ra_deg=349.598450,
        dec_deg=-42.368200,
        size_arcsec=8.0,
    ),
    output_dir="/Volumes/HouAstro/master/results/NGC7582",
)

# Mode selection: NGC 7582 is a compact Seyfert nucleus (CO 5σ envelope only
# 4.9"x4.7"), so the default `box` signal-region mode and `center`
# mask-components mode are appropriate (matches Ada's pointing-based
# analysis). Do NOT switch to dilated_footprint or `all` modes unless box
# fails clearly.

if __name__ == "__main__":
    result = analyze(config)
    print(result.summary())
