"""Tests for cube_io.py (mini_cube-based unit tests)."""
import math
from pathlib import Path
import sys

import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from cube_io import (
    CubeBundle, load_cube, load_cube_bundle,
    apply_pb_cutoff, get_beam_area_pix, channel_width_kms,
)

MINI = str(PKG_DIR / "tests/fixtures/mini_cube.fits")


def test_load_cube_returns_bundle():
    b = load_cube(MINI)
    assert isinstance(b, CubeBundle)
    assert b.data.shape == (20, 50, 50)
    assert math.isclose(b.bmaj_deg * 3600, 2.0, abs_tol=0.001)
    assert math.isclose(b.bmin_deg * 3600, 1.0, abs_tol=0.001)
    assert b.crval3_hz == 230e9
    assert b.cdelt3_hz == -10e6


def test_load_cube_bundle_six_paths_returns_dict():
    d = load_cube_bundle(
        hrl_pbcor_path=MINI, hrl_nonpbcor_path=MINI, hrl_pb_path=MINI,
        co_pbcor_path=MINI, co_nonpbcor_path=MINI, co_pb_path=MINI,
    )
    assert set(d.keys()) == {
        "hrl_pbcor", "hrl_nonpbcor", "hrl_pb",
        "co_pbcor", "co_nonpbcor", "co_pb",
    }
    assert all(isinstance(v, CubeBundle) for v in d.values())


def test_apply_pb_cutoff_zeros_low_pb():
    """PB-cut pixels set to 0 (not NaN) — match legacy v1 behavior."""
    bundle = load_cube(MINI)
    pb_like = load_cube(MINI)
    pb_like.data = np.abs(pb_like.data)
    cut = apply_pb_cutoff(bundle, pb_like, cutoff=0.0008)
    n_zeroed = np.sum(cut.data == 0.0)
    assert n_zeroed > 0
    assert cut.data[6, 25, 25] != 0.0  # signal peak survives


def test_get_beam_area_pix_formula():
    b = load_cube(MINI)
    area = get_beam_area_pix(b)
    bmaj_pix = (2.0 / 3600) / (0.2 / 3600)   # 10 pix
    bmin_pix = (1.0 / 3600) / (0.2 / 3600)   #  5 pix
    expected = math.pi * bmaj_pix * bmin_pix / (4 * math.log(2))
    assert math.isclose(area, expected, rel_tol=1e-6)


def test_channel_width_kms_from_rest_freq():
    b = load_cube(MINI)
    # |dv| = c * |df|/f = 299792.458 * 10/230000 = 13.034 km/s
    dv = channel_width_kms(b, rest_ghz=230.0)
    assert math.isclose(abs(dv), 13.034, abs_tol=0.01)
