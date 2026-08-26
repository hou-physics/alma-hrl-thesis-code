"""Tests for null-test offset injection (`null_test_offset_kms`)."""
from pathlib import Path
import sys

import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from analyze import _shift_center_for_null, ConflictReportError
from config import AnalysisConfig, SkyRegion


def test_shift_center_exact_channel_math():
    """offset=+1000 km/s at chan_width=10 km/s → +100 chan exactly."""
    new_center = _shift_center_for_null(
        hrl_center=500, offset_kms=1000.0,
        chan_width_kms=10.0, nchan=2000,
    )
    assert new_center == 600


def test_shift_center_negative_offset():
    new_center = _shift_center_for_null(
        hrl_center=500, offset_kms=-750.0,
        chan_width_kms=10.0, nchan=2000,
    )
    assert new_center == 500 - 75


def test_shift_center_rounds_half_channel():
    """offset=+55 km/s at chan_width=10 → +5.5 → rounds to +6 (banker: even)."""
    new_center = _shift_center_for_null(
        hrl_center=100, offset_kms=55.0,
        chan_width_kms=10.0, nchan=500,
    )
    assert new_center in (105, 106)


def test_shift_center_too_close_to_edge_raises():
    """buffer_kms=550 at chan_width=10 → 55 chan buffer. new_center=50 < 55 → raise."""
    with pytest.raises(ConflictReportError, match="pushes HRL center"):
        _shift_center_for_null(
            hrl_center=100, offset_kms=-500.0,
            chan_width_kms=10.0, nchan=500,
        )


def test_shift_center_beyond_cube_right_edge_raises():
    with pytest.raises(ConflictReportError):
        _shift_center_for_null(
            hrl_center=450, offset_kms=+500.0,
            chan_width_kms=10.0, nchan=500,
        )


def test_config_defaults_null_fields():
    """AnalysisConfig must have new null-test fields with backwards-compat defaults."""
    c = AnalysisConfig(
        galaxy="X", z=0.01, weak_line="H30a", strong_line="CO21",
        hrl_pbcor_path="/a", hrl_nonpbcor_path="/b",
        co_pbcor_path="/c", co_nonpbcor_path="/d",
        signal_region=SkyRegion(1, 2, 3),
        noise_region=SkyRegion(4, 5, 6),
        output_dir="/out",
    )
    assert c.null_test_offset_kms == 0.0
    assert c.skip_plots is False
    assert c.results_csv_row is None
