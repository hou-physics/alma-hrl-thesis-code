"""Definitive test: apply cross-mousid CO mask to NATIVE HRL data.

Cross-mousid analysis (X1a HRL convolved to 0.91" + X1c CO at 0.91") gave
S/N 8.01, flux 0.787 Jy·km/s. Native analysis (X1a HRL + X1a CO both at
0.19") gave S/N -0.26.

This script:
1. Loads cross-mousid cache → gets the working mask (X1c CO 0.91"-derived,
   reprojected to X1a grid at common beam scale)
2. Applies it to the NATIVE X1a HRL pbcor cube (no convolution)
3. Integrates flux + computes S/N
4. Compares with cross-mousid result

If native flux ≈ 0.787 with this mask → data is fine, native CO mask
is the problem (mask SHAPE issue).
If native flux ≈ 0 → flux is genuinely lost in native HRL data
(zero-spacing / missing extended flux problem).
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

sys.path.insert(0, "/Volumes/HouAstro/master/master_thesis/my_code/_step3")

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

CROSS_CACHE = "/Volumes/HouAstro/master/results/NGC5253_deep/_plot_cache.pkl"
NATIVE_HRL_PBCOR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep/NGC5253_H30a_pbcor.fits"
NATIVE_HRL_NONPBCOR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep/NGC5253_H30a_nonpbcor.fits"

HRL_CHAN_CENTER = 945
LINE_HALF_WIDTH_KMS = 70  # ±70 km/s = 140 km/s window (matches cross-mousid result)


def main():
    # === Load cross-mousid mask + best params ===
    with open(CROSS_CACHE, "rb") as f:
        cache = pickle.load(f)
    plot_args = cache["plot_args"]
    mask_cross = plot_args["spatial_mask_hrl"]   # mask on X1a grid at 0.91" common beam
    print(f"Cross-mousid mask:")
    print(f"  shape: {mask_cross.shape}")
    print(f"  N_pixels: {mask_cross.sum()}")
    print(f"  cross-mousid result: S/N {plot_args['best_sn']:.2f}, flux {plot_args['best_flux']:.4f}")

    # === Load native X1a HRL pbcor (NOT convolved) ===
    with fits.open(NATIVE_HRL_PBCOR, memmap=True) as h:
        hdr = h[0].header
        bmaj = hdr["BMAJ"] * 3600  # arcsec
        bmin = hdr["BMIN"] * 3600
        cdelt = abs(hdr["CDELT1"]) * 3600  # arcsec/pix
        crval3 = hdr["CRVAL3"]
        cdelt3 = hdr["CDELT3"]
        chan_width_kms = abs(cdelt3) * 299792.458 / crval3
        nchan = hdr["NAXIS3"]
        data = h[0].data
        if data.ndim == 4:
            data = data[0]
        # Compute mom-0 over line window
        half = int(round(LINE_HALF_WIDTH_KMS / chan_width_kms))
        line_start = HRL_CHAN_CENTER - half
        line_end = HRL_CHAN_CENTER + half
        n_chans = line_end - line_start + 1
        # Spectrum integrated over mask
        masked = data[line_start:line_end + 1, mask_cross]   # (n_chan, n_pix_in_mask)
        spec_sum_chan = np.nansum(masked, axis=1)   # (n_chan,)

        # Off-line spectrum for noise estimate
        off_chans = list(range(0, line_start - 200)) + list(range(line_end + 200, nchan))
        off_chans = [c for c in off_chans if 0 <= c < nchan]
        off_data = data[off_chans][:, mask_cross]
        off_spec_sum_chan = np.nansum(off_data, axis=1)

    print(f"\nNative HRL cube: beam={bmaj:.3f}x{bmin:.3f}\", pix={cdelt:.5f}\"/pix, chan_width={chan_width_kms:.3f} km/s")
    beam_area_arcsec2 = np.pi * bmaj * bmin / (4 * np.log(2))
    beam_area_pix = beam_area_arcsec2 / cdelt**2
    print(f"  beam area: {beam_area_arcsec2:.4f} arcsec² = {beam_area_pix:.1f} pixels")
    print(f"  N_pix in mask: {mask_cross.sum()} = {mask_cross.sum() / beam_area_pix:.1f} beams")

    # Convert spec_sum_chan → mean per beam (Jy/beam) by dividing by N_pix_per_beam
    # Actually, for flux integration: F = Σ_pix(Jy/bm) × pix_area / beam_area = Σ_pix / beam_area_pix
    # Per channel: mean Jy/bm in mask = sum / N_pix_in_mask (this is per beam)
    n_pix_in_mask = int(mask_cross.sum())
    spec_mean_per_beam = spec_sum_chan / n_pix_in_mask   # Jy/beam
    off_mean_per_beam = off_spec_sum_chan / n_pix_in_mask

    # Noise
    arr = off_mean_per_beam.copy()
    arr = arr[np.isfinite(arr)]
    for _ in range(5):
        m, s = np.nanmean(arr), np.nanstd(arr)
        arr = arr[np.abs(arr - m) < 3 * s]
    sigma_per_chan = np.nanstd(arr)
    print(f"\n  σ per channel (off-line, mask-mean per beam): {sigma_per_chan*1000:.3f} mJy/bm")

    # Integrated flux: sum over channels × dv / beam_area_pix
    # spec_sum_chan is in Jy/bm summed over pixels; to get Jy: divide by beam_area_pix
    flux_jybkm = np.nansum(spec_sum_chan) * chan_width_kms / beam_area_pix
    print(f"  Integrated flux (using cross-mousid mask on NATIVE HRL): {flux_jybkm:.4f} Jy·km/s")

    # Noise on integrated flux
    n_beams = mask_cross.sum() / beam_area_pix
    noise_jybkm = sigma_per_chan * np.sqrt(n_chans) * chan_width_kms * np.sqrt(n_beams)
    sn = flux_jybkm / noise_jybkm if noise_jybkm > 0 else 0
    print(f"  Noise: {noise_jybkm:.4f} Jy·km/s")
    print(f"  S/N: {sn:.2f}")

    print("\n=== Comparison ===")
    cs_sn = plot_args['best_sn']
    cs_flux = plot_args['best_flux']
    print(f"  Cross-mousid (X1a->0.91, X1c CO 0.91):    S/N {cs_sn:.2f}, flux {cs_flux:.4f}")
    print(f"  Native data + cross-mousid mask:           S/N {sn:.2f}, flux {flux_jybkm:.4f}")
    print(f"  Native pipeline + native CO mask:          S/N -0.26, flux -0.006 (failed)")
    print()
    if abs(flux_jybkm) > 0.5:
        print("  -> Native data HAS the flux! Failure = native CO mask shape is WRONG.")
    else:
        print("  -> Native data LOST the flux! Failure = X1a long-baseline zero-spacing problem.")


if __name__ == "__main__":
    main()
