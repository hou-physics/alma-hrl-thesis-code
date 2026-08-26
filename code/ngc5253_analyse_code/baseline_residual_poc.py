"""
Proof-of-concept: baseline-residual fingerprint subtraction for NGC 5253
dilation CoG.

Idea:
  F_obs(N)      observed flux along dilation CoG (= phase2 result)
  ε(x,y)        per-pixel mean over line-free channels (DC residual map)
  F_residual(N) = (Σ_{pix in M(N)} ε) × W × (pix_area / beam_area)
  F_clean(N)    = F_obs(N) − F_residual(N)

If F_clean(N) shows clean plateau in 5..10 beam range → bowl explained by
baseline residual, F_clean plateau = true flux.
If F_clean is still weird → baseline not the cause.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cmasher  # noqa: F401  -- registers cmr.* colormaps
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))
from cube_io import load_cube  # noqa: E402

# -- inputs (matches step3_analyze.py + phase2_raw_step3) -------------------
HRL_PATH = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_H30a_pbcor.fits"
SEED_MASK = "/Volumes/HouAstro/master/result_v2/NGC5253/phase2_raw_step3/best_mask_hrl.npy"

H30A_REST_HZ = 231.900928e9  # H30alpha rest frequency
Z = 0.0014                    # NGC 5253 systemic redshift
W_KMS = 135.0                 # scan-best velocity width
C_KMS = 299792.458

N_DILATION = 41
OUT_PNG = "/Volumes/HouAstro/master/result_v2/NGC5253/poc_baseline_residual_dilation.png"
OUT_EPSMAP = "/Volumes/HouAstro/master/result_v2/NGC5253/poc_baseline_eps_map.png"


def channel_velocities(header, rest_hz_obs):
    """Return per-channel velocity (km/s) in source-rest frame."""
    n = header["NAXIS3"]
    cdelt = header["CDELT3"]
    crval = header["CRVAL3"]
    crpix = header["CRPIX3"]
    freqs = crval + (np.arange(n) - (crpix - 1)) * cdelt
    return (rest_hz_obs - freqs) / rest_hz_obs * C_KMS


def beam_pixels_per_beam(header):
    bmaj_deg = header["BMAJ"]
    bmin_deg = header["BMIN"]
    pix_deg = abs(header["CDELT1"])
    bmaj_pix = bmaj_deg / pix_deg
    bmin_pix = bmin_deg / pix_deg
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_pix * bmin_pix


def main():
    print("loading HRL cube ...")
    bundle = load_cube(HRL_PATH)
    cube = bundle.data  # shape (nchan, ny, nx) float32
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    print(f"  cube shape (nchan, ny, nx) = {cube.shape}")

    # observed H30a center frequency (redshifted)
    rest_obs_hz = H30A_REST_HZ / (1.0 + Z)
    v_chan = channel_velocities(hdr, rest_obs_hz)  # km/s in source rest frame
    dv_kms = abs(np.median(np.diff(v_chan)))
    print(f"  channel width dv = {dv_kms:.2f} km/s, nchan = {nchan}")
    print(f"  v range = [{v_chan.min():+.1f}, {v_chan.max():+.1f}] km/s")

    pix_per_beam = beam_pixels_per_beam(hdr)
    pix_area_per_beam_area = 1.0 / pix_per_beam
    print(f"  pix per beam = {pix_per_beam:.2f}")

    # -- line zone vs line-free zone ---------------------------------------
    # Line zone: |v| <= W/2 (we are in source rest frame; H30a centered at 0)
    line_zone = np.abs(v_chan) <= W_KMS / 2.0
    # Line-free zone: everything else, with a buffer of one channel
    line_free = np.abs(v_chan) > (W_KMS / 2.0 + dv_kms)
    n_line = int(line_zone.sum())
    n_free = int(line_free.sum())
    print(f"  line zone channels: {n_line}, line-free channels: {n_free}")

    # -- eps map: per-pixel mean over line-free channels --------------------
    print("computing eps map ...")
    eps = np.nanmean(cube[line_free, :, :], axis=0)  # (ny, nx) Jy/beam
    sigma_eps = np.nanstd(cube[line_free, :, :], axis=0) / np.sqrt(n_free)
    eps_finite = eps[np.isfinite(eps)]
    print(f"  eps stats: median={np.nanmedian(eps_finite):+.4e} Jy/beam")
    print(f"             mean  ={np.nanmean(eps_finite):+.4e} Jy/beam")
    print(f"             std   ={np.nanstd(eps_finite):.4e} Jy/beam")
    print(f"             |eps| > 3*sigma_eps fraction: "
          f"{np.nanmean(np.abs(eps) > 3*sigma_eps):.3f}")

    # -- line-zone "moment-0 / W" map for sanity ---------------------------
    line_avg = np.nanmean(cube[line_zone, :, :], axis=0)  # Jy/beam

    # -- seed mask + dilation sequence -------------------------------------
    print("loading seed mask ...")
    seed = np.load(SEED_MASK).astype(bool)
    assert seed.shape == (ny, nx), f"seed shape {seed.shape} != cube xy {(ny, nx)}"
    print(f"  seed Npix={int(seed.sum())}, Nbeam={seed.sum()/pix_per_beam:.2f}")

    masks = [seed.copy()]
    cur = seed.copy()
    for i in range(N_DILATION):
        cur = ndimage.binary_dilation(cur, iterations=1)
        masks.append(cur.copy())

    # -- per-step F_obs, F_residual ----------------------------------------
    # F_obs(N) = sum_{pix in M} sum_{v in line_zone} I(pix, v) * dv * (1/pix_per_beam)
    # F_residual(N) = (sum_{pix in M} eps(pix)) * W * (1/pix_per_beam)
    line_int_per_pix = np.nansum(cube[line_zone, :, :], axis=0) * dv_kms  # Jy/beam · km/s per pixel-spectrum
    eps_x_W = eps * W_KMS  # Jy/beam · km/s per pixel (DC contribution to line zone integral)

    Npix_arr = np.array([int(m.sum()) for m in masks])
    Nbeam_arr = Npix_arr / pix_per_beam
    F_obs = np.array([np.nansum(line_int_per_pix[m]) * pix_area_per_beam_area
                      for m in masks])
    F_res = np.array([np.nansum(eps_x_W[m]) * pix_area_per_beam_area
                      for m in masks])
    F_clean = F_obs - F_res

    # -- print summary -----------------------------------------------------
    print("\nstep   Nbeam   F_obs    F_residual   F_clean")
    for i in range(len(masks)):
        print(f" {i:3d}   {Nbeam_arr[i]:6.2f}   {F_obs[i]:+6.3f}   "
              f"{F_res[i]:+6.3f}      {F_clean[i]:+6.3f}")

    # -- plot 1: 3 curves --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Nbeam_arr, F_obs, "o-", color="C3", label="F_obs (dilation CoG)")
    ax.plot(Nbeam_arr, F_res, "s-", color="C2",
            label=r"F_residual = ($\Sigma$$\varepsilon$) $\times$ W")
    ax.plot(Nbeam_arr, F_clean, "^-", color="C0",
            label="F_clean = F_obs − F_residual")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mask size — Nbeam")
    ax.set_ylabel("Flux (Jy·km/s)")
    ax.set_title(
        "NGC 5253 — baseline-residual subtraction along dilation CoG\n"
        f"(W = {W_KMS:.0f} km/s, seed Nbeam = {Nbeam_arr[0]:.2f}, "
        f"{N_DILATION} dilation steps)"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nsaved {OUT_PNG}")

    # -- plot 2: eps map vs line-zone-mean map -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vlim = 3 * np.nanstd(eps_finite)
    im0 = axes[0].imshow(eps, origin="lower", cmap="RdBu_r",
                         vmin=-vlim, vmax=+vlim)
    axes[0].set_title(
        r"$\varepsilon$(x,y): mean over line-free channels"
        + f" (n={n_free})"
    )
    plt.colorbar(im0, ax=axes[0], label="Jy/beam")

    im1 = axes[1].imshow(line_avg, origin="lower", cmap="cmr.cosmic")
    axes[1].set_title("line-zone mean (sanity)")
    plt.colorbar(im1, ax=axes[1], label="Jy/beam")

    # overlay all dilation mask outlines on eps map
    for i in [0, 5, 10, 20, 30, 40]:
        if i < len(masks):
            axes[0].contour(masks[i], levels=[0.5], colors="black",
                            linewidths=0.5, alpha=0.4)
    axes[0].contour(masks[0], levels=[0.5], colors="yellow",
                    linewidths=1.0, label="seed")

    plt.tight_layout()
    plt.savefig(OUT_EPSMAP, dpi=150)
    print(f"saved {OUT_EPSMAP}")


if __name__ == "__main__":
    main()
