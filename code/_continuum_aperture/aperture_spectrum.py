"""Extract an aperture-integrated 1D spectrum from a cube.

Given a cube and a 2D aperture mask, this returns a per-channel sum (Jy units
after correcting for beam area).

Implementation:
- Cube is in Jy/beam.
- Spatial sum over Npix aperture pixels: Σ S_pix [Jy/beam].
- Convert to Jy: divide by beam area (in pixel units) since
  beam_area_pix = π·bmaj·bmin / (4·ln 2) / pixel_area, and the integrated flux
  equals (Σ pixels) × pixel_area / beam_area = (Σ pixels) / beam_area_pix.
- Result has units Jy per channel.

Returns velocity axis + Jy spectrum + per-channel rms (computed off-line).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

C_KMS = 299792.458


@dataclass
class ApertureSpectrum:
    velocity_kms: np.ndarray
    flux_jy: np.ndarray
    sigma_chan_jy: float           # off-line per-channel rms (one number)
    sigma_per_chan_jy: np.ndarray  # spectral-noise (alternative estimator, sigma-clipped per channel)
    rest_freq_hz: float
    z: float
    npix_aperture: int
    beam_area_pix: float
    chan_width_kms: float
    cube_path: str


def beam_area_in_pixels(bmaj_arcsec: float, bmin_arcsec: float, pix_arcsec: float) -> float:
    return np.pi * bmaj_arcsec * bmin_arcsec / (4.0 * np.log(2)) / pix_arcsec ** 2


def extract_aperture_spectrum(
    cube_path: str,
    aperture_mask_2d: np.ndarray,
    rest_freq_hz: float,
    z: float,
    line_centers_kms: tuple[float, ...] = (0.0,),
    line_window_kms: float = 300.0,
) -> ApertureSpectrum:
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        if data.ndim == 4:
            data = data[0]
        nchan, ny, nx = data.shape
        wcs = WCS(hdr)
        freq_hz = wcs.spectral.array_index_to_world_values(np.arange(nchan)).astype(float)

        bmaj = float(hdr['BMAJ']) * 3600
        bmin = float(hdr['BMIN']) * 3600
        try:
            pix_arcsec = abs(float(hdr['CDELT1'])) * 3600
        except KeyError:
            pix_arcsec = abs(float(hdr['CD1_1'])) * 3600

        # Stream-sum only inside the aperture
        if aperture_mask_2d.shape != (ny, nx):
            raise ValueError(
                f"aperture mask shape {aperture_mask_2d.shape} != cube spatial {(ny, nx)}"
            )
        npix = int(aperture_mask_2d.sum())
        if npix < 1:
            raise RuntimeError("aperture mask is empty — no pixels above continuum threshold")

        # Per-channel sequential read for 40 GB cubes (NGC 3627). Slicing
        # into the aperture during the read avoids holding a full chan-block
        # in memory; the page cache then drops behind the read head.
        spec_jybeam = np.zeros(nchan, dtype=np.float64)
        mask_2d = aperture_mask_2d.astype(bool)
        for c in range(nchan):
            chan = np.asarray(data[c, :, :], dtype=np.float32)
            vals = chan[mask_2d]
            vals = vals[np.isfinite(vals)]
            spec_jybeam[c] = float(vals.sum(dtype=np.float64))
            del chan, vals

    beam_area_pix = beam_area_in_pixels(bmaj, bmin, pix_arcsec)
    spec_jy = spec_jybeam / beam_area_pix

    # Velocity axis
    f_rest_obs = rest_freq_hz / (1.0 + z)
    vel_kms = -C_KMS * (freq_hz - f_rest_obs) / f_rest_obs
    df_hz = abs(np.median(np.diff(freq_hz)))
    dv_kms = abs(C_KMS * df_hz / f_rest_obs)

    # Noise: sigma-clipped std of off-line channels of the aperture spectrum
    in_line = np.zeros(nchan, dtype=bool)
    for vc in line_centers_kms:
        in_line |= np.abs(vel_kms - vc) <= line_window_kms
    off = spec_jy[~in_line]
    off = off[np.isfinite(off)]
    for _ in range(5):
        m, s = np.median(off), np.std(off)
        keep = np.abs(off - m) < 3 * s
        if keep.sum() == off.size:
            break
        off = off[keep]
    sigma_chan = float(np.std(off))

    return ApertureSpectrum(
        velocity_kms=vel_kms,
        flux_jy=spec_jy,
        sigma_chan_jy=sigma_chan,
        sigma_per_chan_jy=np.full(nchan, sigma_chan, dtype=np.float64),  # placeholder; per-chan would need cube
        rest_freq_hz=rest_freq_hz,
        z=z,
        npix_aperture=npix,
        beam_area_pix=beam_area_pix,
        chan_width_kms=dv_kms,
        cube_path=cube_path,
    )


def measure_line(
    spec: ApertureSpectrum,
    integration_window_kms: float = 200.0,
    line_center_kms: float = 0.0,
    poly_order: int = 1,
    excluded_velocities_kms: tuple[tuple[float, float], ...] = (),
) -> dict:
    """Toma-style fixed-window line measurement.

    1. Linear baseline fit on off-line channels (excluding the integration
       window itself with a buffer, plus any user-supplied excluded ranges
       — e.g. another bright line in a combined cube).
    2. Subtract baseline.
    3. Integrate in [line_center - W/2, line_center + W/2].
    4. σ_int = σ_chan × √N_chan_in_window × dv (Toma eq.: 1σ = RMS √(Δv/200) Δv
       at W=200 km/s; this is the same expression).
    5. S/N = flux / σ_int.

    `excluded_velocities_kms`: extra (center_kms, half_width_kms) ranges to
    drop from the baseline fit. Used when the cube contains a second bright
    line (e.g. CO(2-1) at +1764 km/s relative to H30α in a single-spw cube)
    that would otherwise pull the linear baseline up at one end.

    Output: dict with flux_jy_kms, sigma_int_jy_kms, sn, n_int_chan,
    sigma_chan_jy, baseline_coef, baseline_subtracted_flux.
    """
    v = spec.velocity_kms
    dv = abs(spec.chan_width_kms)

    # Off-line for baseline fit: outside [line_center - W/2 - buffer, +W/2 + buffer]
    buffer_kms = max(50.0, 0.5 * integration_window_kms)
    half = 0.5 * integration_window_kms
    off = (np.abs(v - line_center_kms) > half + buffer_kms) & np.isfinite(spec.flux_jy)
    for vc, vhw in excluded_velocities_kms:
        off &= np.abs(v - vc) > vhw
    if off.sum() < 20:
        raise RuntimeError(f"baseline fit needs >=20 off-line channels, got {off.sum()}")
    coef = np.polyfit(v[off], spec.flux_jy[off], poly_order)
    baseline = np.polyval(coef, v)
    flux_sub = spec.flux_jy - baseline

    # Recompute σ_chan after baseline subtraction (small adjustment)
    off2 = flux_sub[off]
    for _ in range(5):
        m, s = np.median(off2), np.std(off2)
        keep = np.abs(off2 - m) < 3 * s
        if keep.sum() == off2.size:
            break
        off2 = off2[keep]
    sigma_chan = float(np.std(off2))

    # Integrate
    in_mask = (v >= line_center_kms - half) & (v <= line_center_kms + half) & np.isfinite(flux_sub)
    n_int = int(in_mask.sum())
    if n_int < 2:
        raise RuntimeError(f"integration window {integration_window_kms} km/s yields {n_int} channels")

    flux_int = float(np.sum(flux_sub[in_mask]) * dv)  # Jy·km/s
    sigma_int = float(sigma_chan * np.sqrt(n_int) * dv)
    sn = flux_int / sigma_int if sigma_int > 0 else float('nan')

    return dict(
        flux_jy_kms=flux_int,
        sigma_int_jy_kms=sigma_int,
        sn=sn,
        n_int_chan=n_int,
        sigma_chan_jy=sigma_chan,
        baseline_coef=coef.tolist(),
        baseline_subtracted_flux=flux_sub,
        integration_window_kms=integration_window_kms,
    )
