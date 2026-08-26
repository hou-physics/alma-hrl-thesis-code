"""Tests for analyze.py (line channel resolution; full orchestration tested in regression)."""
from pathlib import Path
import sys

import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from analyze import _resolve_line_channels, ConflictReportError
from cube_io import load_cube

MINI = str(PKG_DIR / "tests/fixtures/mini_cube.fits")


def test_resolve_channels_line_at_center():
    """Line at rest 230 GHz, z=0 → observed 230 GHz → lands at CRPIX3=1 (ch 0)."""
    b = load_cube(MINI)
    ch_min, ch_max, center = _resolve_line_channels(
        b, rest_ghz=230.0, z=0.0, window_kms=40.0
    )
    assert center == 0 or center == 1
    assert ch_min <= center <= ch_max


def test_resolve_channels_line_outside_raises():
    b = load_cube(MINI)
    with pytest.raises(ConflictReportError):
        _resolve_line_channels(b, rest_ghz=250.0, z=0.0, window_kms=40.0)
