"""Diagnostic for NGC 5253 native 0.19" failure.

Hypotheses to test:
1. WCS misalignment between spw0 (CO) and spw1 (HRL)
2. CO peak position differs from H30α peak position at native res
3. CO 3σ unsmoothed footprint too tight (excludes source)
4. Smoothing scales (3, 5, 10 × 0.19") wrong for the source

Output: numerical diagnostics printed; saves diagnostic plot.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/Volumes/HouAstro/master/master_thesis/my_code/_step3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

WORKDIR = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253_deep"
HRL_PBCOR = f"{WORKDIR}/NGC5253_H30a_pbcor.fits"
CO_PBCOR = f"{WORKDIR}/NGC5253_CO21_pbcor.fits"
CO_NONPBCOR = f"{WORKDIR}/NGC5253_CO21_nonpbcor.fits"
HRL_NONPBCOR = f"{WORKDIR}/NGC5253_H30a_nonpbcor.fits"

# Known signal_region center (NGC 5253 CO peak from X1c imaging, world-coord invariant)
SIGNAL_RA_DEG = 204.982993
SIGNAL_DEC_DEG = -31.640281
SIGNAL_BOX_ARCSEC = 20.0  # half-width in each direction

# Line center channels (from earlier diagnostic)
HRL_CHAN_CENTER = 945
CO_CHAN_CENTER = 945
LINE_HALF_WINDOW_KMS = 70  # ~140 km/s integration window (matches deep result)


def load_cube_meta(path):
    with fits.open(path, memmap=True) as h:
        hdr = h[0].header
        wcs = WCS(hdr).celestial
        ra0 = hdr.get("CRVAL1")
        dec0 = hdr.get("CRVAL2")
        crpix1 = hdr.get("CRPIX1")
        crpix2 = hdr.get("CRPIX2")
        cdelt1 = abs(hdr.get("CDELT1")) * 3600  # arcsec/pix
        cdelt2 = abs(hdr.get("CDELT2")) * 3600
        bmaj = hdr.get("BMAJ", 0) * 3600
        bmin = hdr.get("BMIN", 0) * 3600
        nchan = hdr.get("NAXIS3")
        crval3 = hdr.get("CRVAL3")
        cdelt3 = hdr.get("CDELT3")
        crpix3 = hdr.get("CRPIX3")
        chan_width_kms = abs(cdelt3) * 299792.458 / crval3
    return dict(
        wcs=wcs, ra0=ra0, dec0=dec0, crpix1=crpix1, crpix2=crpix2,
        cdelt_arcsec=cdelt1, bmaj=bmaj, bmin=bmin, nchan=nchan,
        chan_width_kms=chan_width_kms,
    )


def make_mom0(cube_path, chan_min, chan_max, chan_width_kms):
    with fits.open(cube_path, memmap=True) as h:
        data = h[0].data
        if data.ndim == 4:
            data = data[0]
        mom0 = np.nansum(data[chan_min:chan_max + 1], axis=0) * chan_width_kms
    return mom0


def main():
    print("=" * 60)
    print("NGC 5253 native 0.19\" failure diagnostic")
    print("=" * 60)

    # === Hypothesis 1: WCS alignment ===
    print("\n[1] WCS alignment (spw0 CO vs spw1 HRL):")
    co_meta = load_cube_meta(CO_NONPBCOR)
    hrl_meta = load_cube_meta(HRL_PBCOR)
    print(f"  CO  ra0={co_meta['ra0']:.7f}, dec0={co_meta['dec0']:.7f}, "
          f"crpix=({co_meta['crpix1']:.3f},{co_meta['crpix2']:.3f}), "
          f"cdelt={co_meta['cdelt_arcsec']:.5f}\"/pix, beam={co_meta['bmaj']:.3f}x{co_meta['bmin']:.3f}\"")
    print(f"  HRL ra0={hrl_meta['ra0']:.7f}, dec0={hrl_meta['dec0']:.7f}, "
          f"crpix=({hrl_meta['crpix1']:.3f},{hrl_meta['crpix2']:.3f}), "
          f"cdelt={hrl_meta['cdelt_arcsec']:.5f}\"/pix, beam={hrl_meta['bmaj']:.3f}x{hrl_meta['bmin']:.3f}\"")
    ra_diff_arcsec = abs(co_meta['ra0'] - hrl_meta['ra0']) * 3600
    dec_diff_arcsec = abs(co_meta['dec0'] - hrl_meta['dec0']) * 3600
    print(f"  Reference RA/Dec diff: {ra_diff_arcsec:.5f}\" / {dec_diff_arcsec:.5f}\"")

    # === Hypothesis 2: peak position comparison ===
    print(f"\n[2] Peak positions at native 0.19\":")
    half = int(round(LINE_HALF_WINDOW_KMS / co_meta['chan_width_kms']))
    print(f"  Integration window: ±{LINE_HALF_WINDOW_KMS} km/s = ±{half} channels")

    co_mom0 = make_mom0(CO_NONPBCOR, CO_CHAN_CENTER - half, CO_CHAN_CENTER + half,
                       co_meta['chan_width_kms'])
    hrl_mom0 = make_mom0(HRL_PBCOR, HRL_CHAN_CENTER - half, HRL_CHAN_CENTER + half,
                        hrl_meta['chan_width_kms'])

    # Find peaks within signal_box
    center_sky = SkyCoord(SIGNAL_RA_DEG * u.deg, SIGNAL_DEC_DEG * u.deg, frame="icrs")
    co_xc, co_yc = co_meta['wcs'].world_to_pixel(center_sky)
    hrl_xc, hrl_yc = hrl_meta['wcs'].world_to_pixel(center_sky)
    half_pix_co = SIGNAL_BOX_ARCSEC / 2 / co_meta['cdelt_arcsec']
    half_pix_hrl = SIGNAL_BOX_ARCSEC / 2 / hrl_meta['cdelt_arcsec']

    co_x1, co_x2 = max(0, int(co_xc - half_pix_co)), min(co_mom0.shape[1] - 1, int(co_xc + half_pix_co))
    co_y1, co_y2 = max(0, int(co_yc - half_pix_co)), min(co_mom0.shape[0] - 1, int(co_yc + half_pix_co))
    hrl_x1, hrl_x2 = max(0, int(hrl_xc - half_pix_hrl)), min(hrl_mom0.shape[1] - 1, int(hrl_xc + half_pix_hrl))
    hrl_y1, hrl_y2 = max(0, int(hrl_yc - half_pix_hrl)), min(hrl_mom0.shape[0] - 1, int(hrl_yc + half_pix_hrl))

    co_box = co_mom0[co_y1:co_y2 + 1, co_x1:co_x2 + 1]
    hrl_box = hrl_mom0[hrl_y1:hrl_y2 + 1, hrl_x1:hrl_x2 + 1]
    co_pk_idx = np.unravel_index(np.nanargmax(co_box), co_box.shape)
    hrl_pk_idx = np.unravel_index(np.nanargmax(hrl_box), hrl_box.shape)
    co_pk_pix = (co_x1 + co_pk_idx[1], co_y1 + co_pk_idx[0])
    hrl_pk_pix = (hrl_x1 + hrl_pk_idx[1], hrl_y1 + hrl_pk_idx[0])
    co_pk_sky = co_meta['wcs'].pixel_to_world(co_pk_pix[0], co_pk_pix[1])
    hrl_pk_sky = hrl_meta['wcs'].pixel_to_world(hrl_pk_pix[0], hrl_pk_pix[1])

    sep = co_pk_sky.separation(hrl_pk_sky).arcsec
    print(f"  CO  peak: pix={co_pk_pix}, sky=({co_pk_sky.ra.deg:.6f},{co_pk_sky.dec.deg:.6f}), value={co_box[co_pk_idx]:.4f} Jy*km/s/bm")
    print(f"  HRL peak: pix={hrl_pk_pix}, sky=({hrl_pk_sky.ra.deg:.6f},{hrl_pk_sky.dec.deg:.6f}), value={hrl_box[hrl_pk_idx]:.4f} Jy*km/s/bm")
    print(f"  CO-HRL peak separation: {sep:.3f}\" (beam = 0.20\", so ratio = {sep/0.20:.2f} beams)")

    # === Hypothesis 3: footprint coverage ===
    print(f"\n[3] CO 3σ unsmoothed footprint at native 0.19\":")
    # noise box (from step3 config: SE-offset, ra=204.987952 dec=-31.640681 size=15)
    n_ra, n_dec, n_size = 204.987952, -31.640681, 15.0
    n_center = SkyCoord(n_ra * u.deg, n_dec * u.deg, frame="icrs")
    n_xc, n_yc = co_meta['wcs'].world_to_pixel(n_center)
    n_half = n_size / 2 / co_meta['cdelt_arcsec']
    n_x1, n_x2 = max(0, int(n_xc - n_half)), min(co_mom0.shape[1] - 1, int(n_xc + n_half))
    n_y1, n_y2 = max(0, int(n_yc - n_half)), min(co_mom0.shape[0] - 1, int(n_yc + n_half))
    noise_pixels = co_mom0[n_y1:n_y2 + 1, n_x1:n_x2 + 1]
    noise_pixels = noise_pixels[np.isfinite(noise_pixels)]

    # sigma-clip iter 5 sigma 3
    arr = noise_pixels.copy()
    for _ in range(5):
        m, s = np.nanmean(arr), np.nanstd(arr)
        arr = arr[np.abs(arr - m) < 3 * s]
    co_sigma = np.nanstd(arr)
    print(f"  CO unsmoothed σ (sigma-clip on noise box {n_x1}:{n_x2},{n_y1}:{n_y2}): {co_sigma:.5f} Jy*km/s/bm")
    print(f"  3σ threshold: {3*co_sigma:.5f}")
    print(f"  CO peak in signal box: {co_box[co_pk_idx]:.4f}")
    print(f"  Peak / 3σ ratio: {co_box[co_pk_idx]/(3*co_sigma):.2f}x")

    # Footprint area
    footprint = (co_mom0 > 3 * co_sigma)
    fp_box = np.zeros_like(footprint)
    fp_box[co_y1:co_y2 + 1, co_x1:co_x2 + 1] = True
    footprint = footprint & fp_box
    fp_npix = footprint.sum()
    fp_arcsec_sq = fp_npix * co_meta['cdelt_arcsec']**2
    beam_area_arcsec_sq = np.pi * co_meta['bmaj'] * co_meta['bmin'] / (4 * np.log(2))
    fp_nbeams = fp_arcsec_sq / beam_area_arcsec_sq
    print(f"  Footprint area: {fp_npix} pix = {fp_arcsec_sq:.3f}\"² = {fp_nbeams:.1f} beams")

    # === Hypothesis 4: extract HRL spectrum at CO peak ===
    print(f"\n[4] HRL spectrum at CO peak position (the key test):")
    # CO peak in HRL grid (use CO sky coord, project to HRL grid)
    hrl_co_xc, hrl_co_yc = hrl_meta['wcs'].world_to_pixel(co_pk_sky)
    hrl_co_xc, hrl_co_yc = int(round(float(hrl_co_xc))), int(round(float(hrl_co_yc)))
    print(f"  CO peak in HRL pixel coords: ({hrl_co_xc}, {hrl_co_yc})")

    # Extract spectrum at +/- 3 pixel box around CO peak (1 beam)
    with fits.open(HRL_PBCOR, memmap=True) as h:
        hrl_data = h[0].data
        if hrl_data.ndim == 4:
            hrl_data = hrl_data[0]
        ext_x1, ext_x2 = max(0, hrl_co_xc - 3), min(hrl_data.shape[2] - 1, hrl_co_xc + 3)
        ext_y1, ext_y2 = max(0, hrl_co_yc - 3), min(hrl_data.shape[1] - 1, hrl_co_yc + 3)
        # mean spectrum over extraction box
        spec_box = hrl_data[:, ext_y1:ext_y2+1, ext_x1:ext_x2+1]
        spec = np.nanmean(spec_box, axis=(1, 2))
        print(f"  Extraction box: {ext_x1}:{ext_x2+1}, {ext_y1}:{ext_y2+1} ({ext_x2-ext_x1+1}x{ext_y2-ext_y1+1} pix)")

    # Integrated flux in window
    line_lo = HRL_CHAN_CENTER - half
    line_hi = HRL_CHAN_CENTER + half + 1
    line_flux_per_chan = spec[line_lo:line_hi]
    n_box_pix = (ext_x2 - ext_x1 + 1) * (ext_y2 - ext_y1 + 1)
    n_box_beams = n_box_pix * hrl_meta['cdelt_arcsec']**2 / (np.pi * hrl_meta['bmaj'] * hrl_meta['bmin'] / (4 * np.log(2)))

    # noise on spec excluding line
    mask_off = np.ones(len(spec), dtype=bool)
    mask_off[max(0, HRL_CHAN_CENTER - 250):min(len(spec), HRL_CHAN_CENTER + 250)] = False
    spec_off = spec[mask_off]
    spec_off = spec_off[np.isfinite(spec_off)]
    arr = spec_off.copy()
    for _ in range(5):
        m, s = np.nanmean(arr), np.nanstd(arr)
        arr = arr[np.abs(arr - m) < 3 * s]
    sigma_chan = np.nanstd(arr)

    integrated_jybkm = np.sum(line_flux_per_chan) * hrl_meta['chan_width_kms']
    flux_jy = integrated_jybkm / (np.pi * hrl_meta['bmaj'] * hrl_meta['bmin'] / (4 * np.log(2)) / hrl_meta['cdelt_arcsec']**2) * n_box_pix

    npix = len(line_flux_per_chan)
    noise_window = sigma_chan * np.sqrt(npix) * hrl_meta['chan_width_kms'] / np.sqrt(n_box_beams)
    sn = integrated_jybkm / noise_window if noise_window > 0 else 0

    print(f"  Per-channel σ (off-line): {sigma_chan*1000:.3f} mJy/bm")
    print(f"  Spectrum mean in line window (channels {line_lo}:{line_hi}): "
          f"{np.mean(line_flux_per_chan)*1000:.3f} mJy/bm")
    print(f"  Integrated mean Jy*km/s/bm: {integrated_jybkm:.4f}")
    print(f"  Approx S/N: {sn:.2f}")
    print(f"  → If S/N >> 0 here, HRL signal IS at CO peak (mask shape problem)")
    print(f"  → If S/N ~ 0, HRL signal NOT at CO peak (genuinely offset)")

    # === Save diagnostic plot ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # CO mom0 + peaks
    ax = axes[0]
    im = ax.imshow(co_box, origin="lower", cmap="cividis",
                   vmin=0, vmax=np.nanpercentile(co_box[np.isfinite(co_box)], 99),
                   extent=[co_x1, co_x2, co_y1, co_y2])
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.scatter(*co_pk_pix, color="red", marker="x", s=100, label="CO peak")
    ax.scatter(hrl_co_xc, hrl_co_yc, color="cyan", marker="+", s=100,
               label=f"HRL peak (sep={sep:.2f}\")")
    ax.set_title(f"CO mom-0 (native, σ={co_sigma:.4f}, fp={fp_nbeams:.0f} beams)")
    ax.legend()

    # HRL mom0 + peaks
    ax = axes[1]
    im = ax.imshow(hrl_box, origin="lower", cmap="cividis",
                   vmin=np.nanpercentile(hrl_box[np.isfinite(hrl_box)], 50),
                   vmax=np.nanpercentile(hrl_box[np.isfinite(hrl_box)], 99),
                   extent=[hrl_x1, hrl_x2, hrl_y1, hrl_y2])
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.scatter(*hrl_pk_pix, color="cyan", marker="+", s=100, label="HRL peak")
    ax.scatter(hrl_co_xc, hrl_co_yc, color="red", marker="x", s=100,
               label="CO peak (projected)")
    ax.set_title(f"HRL mom-0 ±70 km/s")
    ax.legend()

    # Spectrum at CO peak
    ax = axes[2]
    vels = (np.arange(len(spec)) - HRL_CHAN_CENTER) * hrl_meta['chan_width_kms']
    ax.plot(vels, spec * 1000, color="black", lw=0.5, drawstyle="steps-mid")
    ax.axvspan(-LINE_HALF_WINDOW_KMS, LINE_HALF_WINDOW_KMS, alpha=0.2, color="red",
               label="line window")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axhline(sigma_chan*1000, color="red", lw=0.5, ls="--", label=f"±1σ ({sigma_chan*1000:.2f} mJy)")
    ax.axhline(-sigma_chan*1000, color="red", lw=0.5, ls="--")
    ax.set_xlim(-500, 500)
    ax.set_xlabel("Velocity offset (km/s)")
    ax.set_ylabel("Flux (mJy/bm)")
    ax.set_title(f"HRL spectrum at CO peak ({n_box_beams:.1f} beam aperture)\nApprox S/N = {sn:.1f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = "/Volumes/HouAstro/master/results/NGC5253_deep_native/diagnostic.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nDiagnostic plot saved to: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
