"""Moment-0, spectrum extraction, noise estimation, S/N calculation."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from astropy.stats import sigma_clip


def make_moment0(
    cube: np.ndarray, ch_min: int, ch_max: int, chan_width_kms: float
) -> np.ndarray:
    """Sum channels [ch_min, ch_max] (inclusive) and multiply by channel width."""
    subcube = cube[ch_min:ch_max + 1, :, :]
    return np.nansum(subcube, axis=0) * chan_width_kms


def extract_spectrum(
    cube: np.ndarray, mask_2d: np.ndarray, beam_area_pix: float,
    ch_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, int, float]:
    """Sum masked pixels per channel, divide by beam_area_pix.

    `ch_range = (ch_min, ch_max)` — if given, fill only [ch_min, ch_max]
    inclusive; channels outside are NaN. Use to restrict per-channel
    masked-sum work to the line + σ-baseline window for I/O-bound speedup
    (~2× on large cubes). `calc_flux_sn` uses `np.nanstd` / `np.nanmedian`
    on the noise window so NaN channels don't poison σ estimation, but the
    range MUST contain enough off-line channels for the σ-clip statistic
    (recommended: HRL_center ± (W_max/2 + baseline_exclude_kms + 200 km/s)).

    Returns (spectrum, npix, nbeam).
    """
    npix = int(np.sum(mask_2d))
    if npix == 0:
        return np.zeros(cube.shape[0]), 0, 0.0
    nbeam = npix / beam_area_pix
    if ch_range is None:
        spec = np.zeros(cube.shape[0])
        for ch in range(cube.shape[0]):
            spec[ch] = np.nansum(cube[ch][mask_2d]) / beam_area_pix
        return spec, npix, nbeam
    n_chan = cube.shape[0]
    ch_min = max(0, int(ch_range[0]))
    ch_max = min(n_chan - 1, int(ch_range[1]))
    spec = np.full(n_chan, np.nan)
    for ch in range(ch_min, ch_max + 1):
        spec[ch] = np.nansum(cube[ch][mask_2d]) / beam_area_pix
    return spec, npix, nbeam


def sigma_clip_noise(
    image: np.ndarray,
    noise_box,
    sigma: float = 3.0,
    maxiters: int = 5,
) -> float:
    """Return clipped std of image within noise_box.

    `noise_box` accepts either:
      - rectangular box (x1, y1, x2, y2) — legacy usage
      - 2D boolean mask (np.ndarray, same shape as image) — adaptive usage
    """
    if isinstance(noise_box, np.ndarray) and noise_box.dtype == bool:
        sub = image[noise_box]
    else:
        x1, y1, x2, y2 = noise_box
        sub = image[y1:y2, x1:x2]
    clipped = sigma_clip(sub, sigma=sigma, maxiters=maxiters)
    return float(np.nanstd(clipped))


def calc_flux_sn(
    spectrum: np.ndarray,
    center_chan: int,
    width_kms: float,
    chan_width_kms: float,
    exclude_line_chans: Optional[Tuple[int, int]] = None,
    baseline_exclude_half_width_kms: float = 300.0,
    baseline_degree: int = 1,
) -> Tuple[float, float, float, float]:
    """Compute (flux_jy_km_s, sn, mask_width_kms, peak_jy).

    Baseline and per-channel RMS are estimated from a FIXED exclusion zone
    around the HRL line (default ±300 km/s), independent of the integration
    width being evaluated. The polynomial of `baseline_degree` is fit to
    the σ-clipped line-free channels and SUBTRACTED from the line region
    before integration.

      - baseline_degree=0: constant (median) — legacy behaviour
      - baseline_degree=1: linear (a + b·channel) — catches across-band
        slope from imperfect continuum subtraction
      - baseline_degree=2: quadratic — for curved residuals (use sparingly)

    Fixing the exclusion zone decouples baseline from integration width:
    if it changed with W, baseline would be biased (at small W line wings
    leak into noise; at large W zone near the line is mostly excluded).
    The fixed zone (default 300 km/s) should exceed realistic HRL FWHM.
    """
    nchan = len(spectrum)
    half = int(round(width_kms / chan_width_kms / 2.0))
    ch_left = max(0, center_chan - half)
    ch_right = min(nchan - 1, center_chan + half)
    n_chan_line = ch_right - ch_left + 1
    mask_width_kms = n_chan_line * chan_width_kms

    fixed_half_ch = int(round(baseline_exclude_half_width_kms / chan_width_kms))
    fix_left = max(0, center_chan - fixed_half_ch)
    fix_right = min(nchan - 1, center_chan + fixed_half_ch)
    idx_all = np.arange(nchan)
    noise_keep = (idx_all < fix_left) | (idx_all > fix_right)
    if exclude_line_chans is not None:
        e1, e2 = exclude_line_chans
        noise_keep &= (idx_all < e1) | (idx_all > e2)
    noise_idx = idx_all[noise_keep]
    noise_values = spectrum[noise_idx]
    finite = np.isfinite(noise_values)
    noise_idx = noise_idx[finite]
    noise_values = noise_values[finite]

    if len(noise_values) >= max(5, baseline_degree + 2):
        clipped = sigma_clip(noise_values, sigma=3, maxiters=5)
        keep_mask = ~np.asarray(clipped.mask, dtype=bool) \
            if hasattr(clipped, "mask") else np.ones_like(noise_values,
                                                          dtype=bool)
        fit_idx = noise_idx[keep_mask]
        fit_vals = noise_values[keep_mask]
        if len(fit_vals) >= baseline_degree + 2:
            coeffs = np.polyfit(fit_idx.astype(float), fit_vals,
                                deg=baseline_degree)
            baseline_func = np.poly1d(coeffs)
            residuals = fit_vals - baseline_func(fit_idx)
            per_chan_rms = float(np.nanstd(residuals))
        else:
            baseline_func = np.poly1d([float(np.nanmedian(clipped))])
            per_chan_rms = float(np.nanstd(clipped))
    else:
        baseline_func = np.poly1d([0.0])
        per_chan_rms = 0.0

    line_chans = np.arange(ch_left, ch_right + 1, dtype=float)
    line_slice = spectrum[ch_left:ch_right + 1] - baseline_func(line_chans)
    flux = float(np.nansum(line_slice)) * chan_width_kms
    peak = float(np.nanmax(line_slice)) if n_chan_line > 0 else 0.0

    if per_chan_rms == 0.0:
        return flux, 0.0, mask_width_kms, peak
    sigma_flux = per_chan_rms * np.sqrt(n_chan_line) * chan_width_kms
    sn = flux / sigma_flux if sigma_flux > 0 else 0.0
    return flux, sn, mask_width_kms, peak
