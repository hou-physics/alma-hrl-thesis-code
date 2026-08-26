"""Tests for known_lines.py (catalog + per-cube matcher)."""
from pathlib import Path
import sys

import numpy as np

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

from known_lines import (
    LINES_REST_HZ,
    line_free_mask_for_cube,
)


def test_catalog_has_required_lines():
    """Catalog includes the lines that matter for our SPWs."""
    for key in ["H30a", "H39b", "H40a", "H50b",
                "CS2-1", "SO_23-12", "CO2-1"]:
        assert key in LINES_REST_HZ, f"catalog missing {key}"


def test_catalog_frequencies_match_rydberg():
    """Every hydrogen line in the catalog must be reproducible from Rydberg.

    This test exists because two beta lines were wrong for months (H39b off by
    +26 GHz, H51b by +115 GHz) and the old H39b test below asserted the wrong
    value as expected behavior, which froze the error in place.  Completeness
    of the catalog is not enough; correctness has to be checked too.
    """
    R_HC = 3.288052e15                      # R_H * c, Hz
    for key, (n, dn) in {
        "H30a": (30, 1), "H40a": (40, 1),
        "H39b": (39, 2), "H50b": (50, 2), "H51b": (51, 2),
    }.items():
        want = R_HC * (1.0 / n**2 - 1.0 / (n + dn) ** 2)
        got = LINES_REST_HZ[key]
        assert abs(got - want) / want < 1e-3, (
            f"{key}: catalog {got/1e9:.4f} GHz vs Rydberg {want/1e9:.4f} GHz")


def test_line_free_mask_excludes_h30a_only_when_alone():
    """Cube around 231.5 GHz at z=0, W=120 km/s → only H30α excluded.

    H39β is at 205.76 GHz, far outside a Band 6 window tuned on H30α, so it
    must NOT produce an exclusion here.  Before 2026-08-26 the catalog carried
    a wrong H39β (232.025 GHz, 124 MHz from H30α) and this test asserted the
    resulting phantom exclusion as correct."""
    v_chan = np.linspace(-1000.0, +1000.0, 401)   # km/s
    rest_obs_hz = 231.900928e9                      # H30α at z=0
    # primary line zone exclusion (|v| <= 60) is done outside known_lines;
    # known_lines.line_free_mask_for_cube only returns extra KNOWN_LINES exclusions
    mask = line_free_mask_for_cube(
        v_chan=v_chan,
        primary_line_rest_hz=rest_obs_hz,
        cube_freq_range_hz=(230.5e9, 232.5e9),
        z=0.0,
        primary_line_key="H30a",
        fwhm_kms=120.0,
    )
    # All-True mask means every channel passes the KNOWN_LINES filter
    # (the primary H30α exclusion is the caller's job; this filter is
    # for *other* lines that happen to be in the SPW).
    assert mask.dtype == bool
    assert mask.shape == v_chan.shape
    # H39β sits at 205.76 GHz — 26 GHz below the band edge of an H30α tuning,
    # so it cannot contaminate this cube and must leave the mask untouched.
    # The window below is where the old wrong catalog value put its phantom
    # exclusion; nothing may be excised there any more.
    h39b_phantom = (v_chan > -220) & (v_chan < -100)
    assert mask[h39b_phantom].all(), \
        "H39β is out of band here; no channel may be excised near -160 km/s"


def test_line_free_mask_skips_lines_outside_cube_band():
    """CO(2-1) is far from H40α SPW; channels near its v_offset are NOT excluded.

    The He2-10 H40α cube (96.9–98.8 GHz) does contain CS2-1 (97.70 GHz) which
    IS correctly excluded.  This test verifies that CO2-1 (obs ~229.9 GHz, way
    outside the band) does NOT cause any exclusion inside v_chan's range.
    """
    v_chan = np.linspace(-1000.0, +5000.0, 1201)
    rest_obs_hz = 99.022952e9 / (1 + 0.002912)    # H40α at He2-10 z (obs)
    mask = line_free_mask_for_cube(
        v_chan=v_chan,
        primary_line_rest_hz=rest_obs_hz,
        cube_freq_range_hz=(96.9e9, 98.8e9),
        z=0.002912,
        primary_line_key="H40a",
        fwhm_kms=120.0,
    )
    # CO2-1 v_offset ≈ -398 000 km/s — completely outside v_chan range.
    # Its obs freq (229.9 GHz) is outside the cube band, so the mask function
    # must not exclude any channels due to CO2-1.
    # Verify by checking no exclusion exists near v=0 except what CS2-1 causes
    # (CS2-1 is at ~+3155 km/s, so channels near 0 are all True).
    near_zero = (v_chan > -500) & (v_chan < +500)
    assert mask[near_zero].all(), \
        "Channels near v=0 should be kept (CO2-1 is out of band; CS2-1 is at +3155 km/s)"
    # The mask is NOT entirely True because CS2-1 (97.70 GHz) IS in the band.
    assert not mask.all(), \
        "CS2-1 IS in the He2-10 H40α band and should be excluded at ~+3155 km/s"


def test_primary_line_not_excluded_when_in_band():
    """H30a sitting in its own SPW with `primary_line_key='H30a'` must NOT
    be excluded — the function only filters contaminants, not the primary.

    Velocity range tight (±30 km/s) so no other catalog line falls in;
    a passing test means H30a was correctly skipped despite being in band.
    """
    v_chan = np.linspace(-30.0, +30.0, 21)
    rest_hz = 231.900928e9
    mask = line_free_mask_for_cube(
        v_chan=v_chan,
        primary_line_rest_hz=rest_hz,
        cube_freq_range_hz=(230.5e9, 232.5e9),
        z=0.0,
        primary_line_key="H30a",
        fwhm_kms=120.0,
    )
    assert mask.all(), \
        "primary line must be skipped even when in band"
