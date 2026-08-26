"""KNOWN_LINES catalog and per-cube line-free-mask matcher.

Catalog covers all spectral transitions that appear in the H30α / H40α
SPWs used by the v1 6-galaxy validation set: hydrogen recombination
α and β series, plus the molecular tracers we mask with (CO2-1, CS2-1)
and contaminants observed in our cubes (SO 2_3-1_2, HC3N 24-23).

Reference: spec §4.1 KNOWN_LINES. Treat this file as the code source of
truth; docs/rules/methods.md §K mirrors this in a table for readers.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


C_KMS = 299792.458

# Rest frequencies in Hz. Keys match the conventions in step3 configs.
#
# Every entry must be reproducible from the Rydberg formula
# nu = R_H*c * (1/n^2 - 1/(n+dn)^2), R_H*c = 3.288052e15 Hz.
# Two beta lines were wrong here until 2026-08-26 (see docs/decisions.md):
# H39b read 232.025 GHz (+26 GHz off) and H51b read 208.290 GHz (+115 GHz off).
# The H39b error mattered: 232.025 sits only 124 MHz from H30a, so the phantom
# line fell inside every H30a spectral window and excised real baseline
# channels near -160 km/s. Its measured effect on the fluxes was quantified
# before the fix and found negligible (median 0.08 sigma_F, no source crossing
# a tier boundary), so no re-run was required.
LINES_REST_HZ = {
    # Hydrogen recombination lines (Mezger-Henderson)
    "H30a":   231.900928e9,
    "H39b":   205.760370e9,    # was 232.025000e9 — outside Band 6 entirely
    "H40a":    99.022952e9,
    "H50b":    99.225000e9,
    "H51b":    93.607344e9,    # was 208.290000e9; documented for completeness
    # Molecular tracers (commonly co-detected with HRLs in our SPWs)
    "CO2-1":  230.538000e9,
    "CS2-1":   97.980953e9,
    "SO_23-12": 99.299870e9,
    "HC3N_24-23": 218.324723e9,
    # wave-1 tracer additions (2026-08-06 audit): previously stage-1-only
    # literals, invisible to baseline excision
    "CO1-0":  115.271202e9,
    "13CO2-1": 220.398684e9,
    "HCO+1-0": 89.188523e9,
    "HCN1-0":  88.631847e9,
}


def line_free_mask_for_cube(
    *,
    v_chan: np.ndarray,
    primary_line_rest_hz: float,
    cube_freq_range_hz: Tuple[float, float],
    z: float,
    primary_line_key: str,
    fwhm_kms: float = 120.0,
) -> np.ndarray:
    """Boolean mask over v_chan (True = keep) that excludes every catalog
    line OTHER than `primary_line_key` whose observed frequency falls
    inside `cube_freq_range_hz`.

    The primary line's own line zone exclusion is the caller's
    responsibility (this function only handles contaminants).

    v_chan is the per-channel velocity in the *source rest frame of the
    primary line* (so primary-line center sits at v ≈ 0).

    Each excluded line gets a window of ±fwhm_kms/2 around its
    redshifted velocity offset from the primary line.

    Because the velocity offset between two lines is z-independent, the
    `primary_line_rest_hz` parameter accepts either the actual rest
    frequency or the redshifted observed frequency of the primary line —
    the ratio gives the same v_other in either case.
    """
    keep = np.ones_like(v_chan, dtype=bool)
    band_lo, band_hi = cube_freq_range_hz
    for key, rest_hz in LINES_REST_HZ.items():
        if key == primary_line_key:
            continue
        obs_hz = rest_hz / (1.0 + z)
        if not (band_lo <= obs_hz <= band_hi):
            continue
        # v_other (in primary line's rest frame) =
        #   C_KMS * (primary_line_rest_hz - obs_hz) / primary_line_rest_hz
        v_other = (
            (primary_line_rest_hz - obs_hz) / primary_line_rest_hz
        ) * C_KMS
        half = fwhm_kms / 2.0
        keep &= ~((v_chan > v_other - half) & (v_chan < v_other + half))
    return keep
