"""Tests for mask.py — smoothing, thresholding, beam matching, reprojection."""
import math
from pathlib import Path
import sys

import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from cube_io import load_cube
from mask import (
    smooth, build_mask,
    compute_common_beam, convolve_to_beam,
    reproject_mask,
)

MINI = str(PKG_DIR / "tests/fixtures/mini_cube.fits")


def test_smooth_preserves_flux():
    img = np.ones((30, 30))
    smoothed = smooth(img, kernel_fwhm_pix=3.0)
    assert math.isclose(np.nanmean(smoothed[5:25, 5:25]), 1.0, abs_tol=0.01)


def test_build_mask_applies_threshold_in_signal_box():
    img = np.ones((30, 30)) * 5.0
    mask = build_mask(
        img, threshold=3.0, signal_box=(10, 10, 20, 20),
        require_connected=False, galaxy_center_pix=None,
    )
    assert mask[15, 15] == True
    assert mask[5, 5] == False
    assert mask[25, 25] == False


def test_build_mask_connected_component():
    img = np.zeros((30, 30))
    img[10:15, 10:15] = 10.0
    img[20:23, 20:23] = 10.0
    mask = build_mask(
        img, threshold=5.0, signal_box=(0, 0, 29, 29),
        require_connected=True, galaxy_center_pix=(12, 12),
    )
    assert mask[12, 12] == True
    assert mask[21, 21] == False


def test_build_mask_footprint_hard_constraint():
    """co_footprint AND-restricts the mask to physically-confirmed CO emission.

    Smoothed CO exceeds threshold in two regions; CO footprint only covers one.
    The mask should land only in the overlap region, never outside the footprint.
    """
    img = np.zeros((30, 30))
    img[10:15, 10:15] = 10.0   # "real" CO peak
    img[20:25, 20:25] = 10.0   # smoothed-noise peak OUTSIDE the real CO footprint

    # Footprint: only the first (physical) peak is confirmed by unsmoothed CO.
    footprint = np.zeros((30, 30), dtype=bool)
    footprint[9:16, 9:16] = True

    mask = build_mask(
        img, threshold=5.0, signal_box=(0, 0, 29, 29),
        require_connected=False, galaxy_center_pix=None,
        co_footprint=footprint,
    )
    # Real peak passes
    assert mask[12, 12] == True
    # Outside-footprint smoothed-noise peak blocked
    assert mask[22, 22] == False
    # Nothing outside the footprint at all
    assert np.all(mask[~footprint] == False)


def test_build_mask_footprint_none_is_backward_compatible():
    """Omitting co_footprint preserves pre-2026-04-24 behavior."""
    img = np.zeros((30, 30))
    img[10:15, 10:15] = 10.0
    img[20:25, 20:25] = 10.0
    mask_no_fp = build_mask(
        img, threshold=5.0, signal_box=(0, 0, 29, 29),
        require_connected=False, galaxy_center_pix=None,
    )
    # Both peaks pass without the constraint
    assert mask_no_fp[12, 12] == True
    assert mask_no_fp[22, 22] == True


def test_compute_common_beam_equal_inputs():
    b = load_cube(MINI)
    common = compute_common_beam(b, b)
    assert math.isclose(common.major.to("arcsec").value, 2.0, abs_tol=0.01)


def test_convolve_to_beam_identity_shortcut():
    b = load_cube(MINI)
    common = compute_common_beam(b)
    out = convolve_to_beam(b, common)
    assert np.array_equal(out.data, b.data)


def test_reproject_mask_identity_when_same_wcs():
    b = load_cube(MINI)
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:30, 20:30] = True
    out = reproject_mask(mask, b.wcs, b.wcs, target_shape=(50, 50))
    agreement = np.sum(mask == out) / mask.size
    assert agreement > 0.99
