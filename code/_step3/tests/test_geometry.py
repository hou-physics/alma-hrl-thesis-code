"""Tests for geometry.py sky-to-pixel conversion."""
from pathlib import Path
import sys

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from config import SkyRegion
from cube_io import load_cube
from geometry import sky_region_to_pixel_box

MINI = str(PKG_DIR / "tests/fixtures/mini_cube.fits")


def test_center_region_returns_center_box():
    """SkyRegion at cube reference coord should produce box centered on CRPIX."""
    b = load_cube(MINI)
    # mini_cube: CRVAL1=220 RA, CRVAL2=-58 Dec, CRPIX=26, CDELT=0.2"/pix
    r = SkyRegion(ra_deg=220.0, dec_deg=-58.0, size_arcsec=4.0)  # 20 pix
    x1, y1, x2, y2 = sky_region_to_pixel_box(r, b.wcs, cube_shape=(20, 50, 50))
    # Box centered at CRPIX 25.5 (0-indexed), half-size 10 pix → [15, 15, 35, 35]±1
    assert 14 <= x1 <= 16
    assert 34 <= x2 <= 36
    assert 14 <= y1 <= 16
    assert 34 <= y2 <= 36


def test_box_clipped_to_cube_shape():
    """If sky region would exceed cube bounds, box is clipped."""
    b = load_cube(MINI)
    r = SkyRegion(ra_deg=220.0, dec_deg=-58.0, size_arcsec=400.0)   # much larger than cube
    x1, y1, x2, y2 = sky_region_to_pixel_box(r, b.wcs, cube_shape=(20, 50, 50))
    assert x1 == 0 and y1 == 0
    assert x2 == 49 and y2 == 49


def test_box_outside_cube_raises():
    """SkyRegion far from cube should raise ValueError."""
    import pytest
    b = load_cube(MINI)
    r = SkyRegion(ra_deg=0.0, dec_deg=0.0, size_arcsec=5.0)  # far from cube center
    with pytest.raises(ValueError, match="outside cube"):
        sky_region_to_pixel_box(r, b.wcs, cube_shape=(20, 50, 50))
