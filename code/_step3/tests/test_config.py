"""Tests for config.py dataclasses."""
import math
from pathlib import Path
import sys

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from config import AnalysisConfig, AnalysisResult, SkyRegion


def test_sky_region_instantiation():
    r = SkyRegion(ra_deg=295.89, dec_deg=-58.65, size_arcsec=30)
    assert r.ra_deg == 295.89
    assert r.size_arcsec == 30


def test_analysis_config_defaults():
    c = AnalysisConfig(
        galaxy="NGC6810",
        z=0.006775,
        weak_line="H30a",
        strong_line="CO21",
        hrl_pbcor_path="/a",
        hrl_nonpbcor_path="/b",
        co_pbcor_path="/c",
        co_nonpbcor_path="/d",
        signal_region=SkyRegion(1, 2, 3),
        noise_region=SkyRegion(4, 5, 6),
        output_dir="/results/NGC6810",
    )
    assert c.pb_cutoff == 0.2
    assert c.weak_line_window_kms == 300.0
    assert c.smoothing_factors == [3, 5, 10]
    assert c.require_connected is True


def test_analysis_result_summary_detection():
    r = AnalysisResult(
        galaxy="NGC6810", weak_line="H30a",
        sn=5.2, flux_jy_km_s=0.234, width_kms=240, nbeam=3.5, peak_jy=0.0015,
        best_smoothing=3, best_spatial_threshold=15.0, best_velocity_width_kms=240,
        script_path="/x", summary_md_path="/y", plots_dir="/p", tables_dir="/t",
        is_detection=True,
    )
    s = r.summary()
    assert "Galaxy: NGC6810" in s
    assert "S/N: 5.2" in s
    assert "Detection" in s


def test_analysis_result_summary_nondetection():
    r = AnalysisResult(
        galaxy="X", weak_line="H30a",
        sn=1.2, flux_jy_km_s=0.01, width_kms=100, nbeam=1.0, peak_jy=0.0001,
        best_smoothing=3, best_spatial_threshold=3.0, best_velocity_width_kms=100,
        script_path="/x", summary_md_path="/y", plots_dir="/p", tables_dir="/t",
        is_detection=False,
    )
    assert "Non-detection" in r.summary()
