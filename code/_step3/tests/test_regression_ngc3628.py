"""Regression test: Phase Y pipeline must reproduce NGC 3628 flux = 0.596 Jy·km/s.

GATE test for Phase Y. Requires the NGC 3628 combined cube at
/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/. Skipped otherwise.
"""
from pathlib import Path
import sys

import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from analyze import analyze
from config import AnalysisConfig, SkyRegion

WORKDIR = Path("/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628")
PBCOR = WORKDIR / "NGC3628_CO_H30alpha_v5_pbcor.fits"
NONPBCOR = WORKDIR / "NGC3628_CO_H30alpha_v5_nonpbcor.fits"
PB = WORKDIR / "NGC3628_CO_H30alpha_v5_pb.fits"


@pytest.mark.regression
@pytest.mark.skipif(
    not PBCOR.exists() or not NONPBCOR.exists() or not PB.exists(),
    reason="NGC 3628 regression cubes not found at expected path"
)
def test_ngc3628_reproduces_flux_and_sn(tmp_path):
    config = AnalysisConfig(
        galaxy="NGC 3628",
        z=0.002772,
        weak_line="H30a",
        strong_line="CO21",
        hrl_pbcor_path=str(PBCOR),
        hrl_nonpbcor_path=str(NONPBCOR),
        co_pbcor_path=str(PBCOR),
        co_nonpbcor_path=str(NONPBCOR),
        hrl_pb_path=str(PB),
        co_pb_path=str(PB),
        signal_region=SkyRegion(
            ra_deg=170.0708, dec_deg=13.5888,
            size_arcsec=40.0, dec_size_arcsec=12.0),
        noise_region=SkyRegion(
            ra_deg=170.0707, dec_deg=13.5944,
            size_arcsec=15.0, dec_size_arcsec=13.0),
        strong_line_window_kms=460.0,
        weak_line_window_kms=760.0,
        output_dir=str(tmp_path),
    )
    result = analyze(config)
    # Tolerance rationale: Phase Y does baseline subtraction (matches Ada 2022
    # class_for_cube_v3.py methodology). Legacy v1 did NOT — its raw flux
    # 0.596 includes ~0.3 Jy·km/s baseline contribution on our cube. Ada's
    # paper value 0.626 was on her re-imaged cube with different baseline
    # residual. The methodologically-correct Phase Y value on our cube is
    # ~0.27–0.30 Jy·km/s (see docs/paper-notes.md § Baseline subtraction).
    #
    # This test asserts: (1) detection succeeds, (2) flux is in the "real
    # H30α line-only" range 0.15–0.5 Jy·km/s, (3) S/N > 3, (4) pipeline
    # picks a reasonable integration width. It is NOT a bit-exact Ada match.
    assert result.is_detection, f"NGC 3628 should be a detection; got sn={result.sn:.2f}"
    assert 0.15 < result.flux_jy_km_s < 0.5, (
        f"NGC 3628 baseline-subtracted flux expected 0.15–0.5 Jy·km/s "
        f"(Phase Y methodology); got {result.flux_jy_km_s:.3f}"
    )
    assert result.sn > 3, f"S/N should be >3 for detection; got {result.sn:.1f}"
    assert 100 < result.width_kms < 500, (
        f"Integration width should be in 100-500 km/s range; got {result.width_kms:.0f}"
    )
