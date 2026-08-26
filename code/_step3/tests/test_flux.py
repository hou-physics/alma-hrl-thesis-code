"""Tests for flux.py."""
import math
from pathlib import Path
import sys

import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from flux import make_moment0, extract_spectrum, sigma_clip_noise, calc_flux_sn


def test_moment0_sums_channels_times_width():
    cube = np.ones((10, 5, 5))
    mom0 = make_moment0(cube, 2, 5, chan_width_kms=10.0)
    # ch 2..5 inclusive = 4 channels × 10 km/s = 40 per pix
    assert np.allclose(mom0, 40.0)


def test_moment0_ignores_nan():
    cube = np.ones((10, 5, 5))
    cube[2:5, :, :] = np.nan
    mom0 = make_moment0(cube, 2, 5, chan_width_kms=10.0)
    # Only ch 5 contributes (1 × 10 = 10 per pix)
    assert np.allclose(mom0, 10.0)


def test_extract_spectrum_sums_masked_pixels_per_channel():
    cube = np.ones((3, 4, 4)) * 2.0
    mask = np.array([[True]*4]*4)
    beam_area = 4.0
    spec, npix, nbeam = extract_spectrum(cube, mask, beam_area)
    # 16 pix × 2.0 / beam_area(4) = 8.0 per channel
    assert np.allclose(spec, 8.0)
    assert npix == 16
    assert math.isclose(nbeam, 4.0)


def test_extract_spectrum_empty_mask_returns_zeros():
    cube = np.ones((3, 4, 4))
    mask = np.zeros((4, 4), dtype=bool)
    spec, npix, nbeam = extract_spectrum(cube, mask, beam_area_pix=4.0)
    assert np.allclose(spec, 0.0)
    assert npix == 0
    assert nbeam == 0


def test_sigma_clip_noise_returns_std_of_clipped():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 0.5, size=(100, 100))
    sigma = sigma_clip_noise(img, (10, 10, 90, 90))
    assert math.isclose(sigma, 0.5, abs_tol=0.05)


def test_calc_flux_sn_on_known_input():
    rng = np.random.default_rng(1)
    spec = rng.normal(0, 0.001, size=100)
    spec[48:52] = 0.010
    flux, sn, width_kms, peak = calc_flux_sn(
        spec, center_chan=50, width_kms=40, chan_width_kms=10.0,
        exclude_line_chans=None,
    )
    assert 0.3 < flux < 0.5
    assert sn > 5
    assert math.isclose(width_kms, 50.0, abs_tol=15)
    assert math.isclose(peak, 0.010, abs_tol=0.003)
