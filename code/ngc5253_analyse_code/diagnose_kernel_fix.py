"""Test: does NGC 5253 native analysis succeed with larger smoothing?

Pipeline scans s_f in {3, 5, 10} × beam = {0.57", 0.95", 1.9"} kernels at
native 0.19" beam. NGC 5253 CO disk is ~10" — need kernel ≥ 5" to envelope.
This script extends s_f to {30, 50} (kernels 5.7", 9.5") and verifies S/N.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/Volumes/HouAstro/master/master_thesis/my_code/_step3")

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import label

from cube_io import load_cube
from mask import smooth, build_mask, reproject_mask
from flux import make_moment0, sigma_clip_noise, calc_flux_sn

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep"

# Same regions as native config
SIGNAL_RA = 204.982993
SIGNAL_DEC = -31.640281
SIGNAL_SIZE = 20.0  # arcsec full width
NOISE_RA = 204.987952
NOISE_DEC = -31.640681
NOISE_SIZE = 15.0


def sky_to_box(wcs, ra, dec, size, shape):
    c = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
    xc, yc = wcs.celestial.world_to_pixel(c)
    cdelt = abs(wcs.wcs.cdelt[0]) * 3600
    half = size / 2 / cdelt
    nx, ny = shape[-1], shape[-2]
    x1 = max(0, int(xc - half)); x2 = min(nx-1, int(xc + half))
    y1 = max(0, int(yc - half)); y2 = min(ny-1, int(yc + half))
    return (x1, y1, x2, y2)


def main():
    print("=" * 60)
    print("NGC 5253 native test: scan extended s_f range")
    print("=" * 60)

    co_pbcor = load_cube(f"{WORKDIR}/NGC5253_CO21_pbcor.fits")
    co_nonpbcor = load_cube(f"{WORKDIR}/NGC5253_CO21_nonpbcor.fits")
    hrl_pbcor = load_cube(f"{WORKDIR}/NGC5253_H30a_pbcor.fits")

    # CO mom-0 over ±300 km/s
    co_chan_center = 945
    co_chan_width = abs(co_nonpbcor.cdelt3_hz) * 299792.458 / co_nonpbcor.crval3_hz
    co_half = int(round(300 / co_chan_width))
    co_mom0 = make_moment0(co_nonpbcor.data, co_chan_center - co_half,
                           co_chan_center + co_half, co_chan_width)

    # Boxes
    co_signal_box = sky_to_box(co_nonpbcor.wcs, SIGNAL_RA, SIGNAL_DEC, SIGNAL_SIZE,
                               co_nonpbcor.data.shape)
    co_noise_box = sky_to_box(co_nonpbcor.wcs, NOISE_RA, NOISE_DEC, NOISE_SIZE,
                              co_nonpbcor.data.shape)
    print(f"Signal box: {co_signal_box}")

    # CO unsmoothed sigma + footprint
    sigma_unsm = sigma_clip_noise(co_mom0, co_noise_box)
    footprint = (co_mom0 > 3 * sigma_unsm)
    fp_box = np.zeros_like(footprint)
    fp_box[co_signal_box[1]:co_signal_box[3]+1, co_signal_box[0]:co_signal_box[2]+1] = True
    footprint = footprint & fp_box
    print(f"Footprint: {footprint.sum()} pix")

    # CO beam in pixels
    co_pixel_arcsec = abs(co_nonpbcor.wcs.wcs.cdelt[0]) * 3600
    co_bmaj_arcsec = co_nonpbcor.bmaj_deg * 3600
    co_bmaj_pix = co_bmaj_arcsec / co_pixel_arcsec
    print(f"CO beam: {co_bmaj_arcsec:.3f}\" = {co_bmaj_pix:.1f} pix")

    # HRL setup
    hrl_chan_width = abs(hrl_pbcor.cdelt3_hz) * 299792.458 / hrl_pbcor.crval3_hz
    hrl_chan_center = 945
    beam_area_pix_hrl = (np.pi * hrl_pbcor.bmaj_deg * hrl_pbcor.bmin_deg *
                        (3600**2) / (4 * np.log(2)) /
                        (abs(hrl_pbcor.wcs.wcs.cdelt[0]) * 3600)**2)

    # Test: vary s_f from {3, 5, 10, 20, 30, 50} at threshold 5σ, width 140 km/s
    cx = (co_signal_box[0] + co_signal_box[2]) // 2
    cy = (co_signal_box[1] + co_signal_box[3]) // 2

    print(f"\n{'s_f':>4} {'kernel':>8} {'σ_sm':>10} {'mask_pix':>10} {'mask_beams':>11} {'flux':>10} {'S/N':>8}")
    print("-" * 70)
    for s_f in [3, 5, 10, 20, 30, 50, 75]:
        kernel_fwhm_pix = s_f * co_bmaj_pix
        kernel_fwhm_arcsec = kernel_fwhm_pix * co_pixel_arcsec
        smoothed = smooth(co_mom0, kernel_fwhm_pix)
        sigma_sm = sigma_clip_noise(smoothed, co_noise_box)

        # Try threshold 5σ (cross-mousid optimum)
        thresh = 5.0 * sigma_sm
        mask_co = build_mask(smoothed, thresh, co_signal_box, True, (cx, cy),
                             co_footprint=footprint)

        # Reproject to HRL grid (identity for same-mousid)
        mask_hrl = reproject_mask(mask_co, co_nonpbcor.wcs, hrl_pbcor.wcs,
                                  target_shape=hrl_pbcor.data.shape[1:])
        n_mask = int(mask_hrl.sum())
        n_beams = n_mask / beam_area_pix_hrl

        # Extract spectrum + integrate
        # Mean over mask per channel
        nchan = hrl_pbcor.data.shape[0]
        spec = np.array([np.nanmean(hrl_pbcor.data[c, mask_hrl]) if n_mask > 0 else 0
                        for c in range(nchan)])

        # Width = 140 km/s (cross-mousid optimum)
        half = int(round(140 / 2 / hrl_chan_width))
        flux_jybkm, sn, _, _ = calc_flux_sn(
            spec, hrl_chan_center, 140, hrl_chan_width,
            exclude_line_chans=None,
        )
        print(f"{s_f:>4d} {kernel_fwhm_arcsec:>7.2f}\" {sigma_sm:>10.5f} "
              f"{n_mask:>10d} {n_beams:>10.1f} {flux_jybkm:>10.4f} {sn:>8.2f}")

    print()
    print("Cross-mousid reference (X1a→0.91, X1c CO 0.91): S/N 8.01, flux 0.787")


if __name__ == "__main__":
    main()
