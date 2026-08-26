"""
Compute spatial CoG (sthresh + dilation) for 3 clean detections that
don't have phase2 outputs: Circinus, NGC 253, NGC 3627. Then assemble
a 7-panel comparison figure including the 4 already done by Phase 2
(NGC 5253, NGC 4945, NGC 3628, He 2-10).

Focused PoC; not part of the production pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402

C_KMS = 299792.458
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")

LINE_REST_HZ = {"H30a": 231.900928e9, "H40a": 99.022952e9}

# Standard sthresh ladder (12 values)
STHRESH_LADDER = [3.0, 3.7, 4.6, 5.7, 7.0, 8.6,
                  10.6, 13.1, 16.2, 19.9, 24.6, 30.3]

N_DILATION = 41


@dataclass
class GalaxyConfig:
    name: str
    cube_path: str
    z: float
    line: str           # "H30a" or "H40a"
    W_kms: float        # scan-best velocity width
    sf_scan: int        # scan-best smoothing factor (multiplier on beam FWHM)
    sthresh_scan: float # scan-best sthresh
    signal_ra: float
    signal_dec: float
    signal_size_arcsec: float
    noise_ra: float
    noise_dec: float
    noise_size_arcsec: float


def channel_velocities(hdr, rest_obs_hz):
    n = hdr["NAXIS3"]
    freqs = hdr["CRVAL3"] + (np.arange(n) - (hdr["CRPIX3"] - 1)) * hdr["CDELT3"]
    return (rest_obs_hz - freqs) / rest_obs_hz * C_KMS


def beam_fwhm_pix(hdr):
    """Use BMAJ only — matches production analyze.py:`smooth(co_mom0, sf * co_bmaj_pix)`."""
    bmaj = hdr["BMAJ"]
    pix = abs(hdr["CDELT1"])
    return bmaj / pix


def pix_per_beam(hdr):
    bmaj = hdr["BMAJ"]; bmin = hdr["BMIN"]; pix = abs(hdr["CDELT1"])
    return (np.pi / (4.0 * np.log(2.0))) * (bmaj / pix) * (bmin / pix)


def smooth_fft(image, kernel_fwhm_pix):
    if kernel_fwhm_pix <= 0:
        return image
    sigma = kernel_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    kernel = Gaussian2DKernel(sigma)
    return convolve_fft(image, kernel, normalize_kernel=True,
                        nan_treatment="fill", boundary="fill")


def make_box_mask(wcs2d, ny, nx, ra, dec, size_arcsec):
    """2D bool mask: True inside a centered box at (ra, dec) of given side."""
    # convert ra, dec to pix (using WCS)
    xc, yc = wcs2d.world_to_pixel_values(ra, dec)
    pix_scale_arcsec = abs(wcs2d.wcs.cdelt[0]) * 3600.0
    half_pix = (size_arcsec / 2.0) / pix_scale_arcsec
    mask = np.zeros((ny, nx), dtype=bool)
    y0 = max(0, int(np.floor(yc - half_pix)))
    y1 = min(ny, int(np.ceil(yc + half_pix)))
    x0 = max(0, int(np.floor(xc - half_pix)))
    x1 = min(nx, int(np.ceil(xc + half_pix)))
    mask[y0:y1, x0:x1] = True
    return mask


def cog_for_galaxy(cfg: GalaxyConfig):
    print(f"\n=== {cfg.name} ===")
    bundle = load_cube(cfg.cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    print(f"  cube shape {cube.shape}")
    rest_obs_hz = LINE_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    pix_per_beam_val = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    print(f"  dv={dv:.2f} km/s, pix/beam={pix_per_beam_val:.2f}, "
          f"beam_FWHM={beam_fwhm:.2f} pix")

    # WCS for 2D plane
    wcs3 = WCS(hdr)
    wcs2 = wcs3.celestial

    # Spatial regions
    sig_box = make_box_mask(wcs2, ny, nx,
                            cfg.signal_ra, cfg.signal_dec,
                            cfg.signal_size_arcsec)
    noise_box = make_box_mask(wcs2, ny, nx,
                              cfg.noise_ra, cfg.noise_dec,
                              cfg.noise_size_arcsec)
    print(f"  sig_box Npix={int(sig_box.sum())}, "
          f"noise_box Npix={int(noise_box.sum())}")

    # mom-0 over scan-best W window
    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv  # Jy/beam · km/s
    print(f"  mom0 range: [{np.nanmin(mom0):+.3f}, {np.nanmax(mom0):+.3f}]")

    # Smooth at scan-best sf (sf is multiplier of beam FWHM)
    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)

    # sigma from noise box (on smoothed map)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    print(f"  sigma_smooth (noise box, sf={cfg.sf_scan}) = "
          f"{sigma_smooth:.4f}")

    # ---------- sthresh-CoG (at fixed sf=scan_best, vary sthresh) ----------
    sthresh_arr = []
    sthresh_Nbeam = []
    sthresh_F = []
    for sthresh in STHRESH_LADDER:
        mask = (mom0_smooth > sthresh * sigma_smooth) & sig_box
        if not mask.any():
            continue
        Npix = int(mask.sum())
        F = float(np.nansum(mom0[mask]) / pix_per_beam_val)
        sthresh_arr.append(sthresh)
        sthresh_Nbeam.append(Npix / pix_per_beam_val)
        sthresh_F.append(F)
    sthresh_arr = np.array(sthresh_arr)
    sthresh_Nbeam = np.array(sthresh_Nbeam)
    sthresh_F = np.array(sthresh_F)
    if len(sthresh_arr) == 0:
        print(f"  WARN: all sthresh values gave empty mask — over-smoothed?")
        print(f"        mom0_smooth in signal box: max={float(np.nanmax(mom0_smooth[sig_box])):.4f}, "
              f"sigma={sigma_smooth:.4f}, ratio_max/sigma={float(np.nanmax(mom0_smooth[sig_box]))/sigma_smooth:.2f}")
    else:
        print(f"  sthresh-CoG: {len(sthresh_arr)} valid cells, "
              f"Nbeam range [{sthresh_Nbeam.min():.1f}, {sthresh_Nbeam.max():.1f}]"
              f", F range [{sthresh_F.min():+.3f}, {sthresh_F.max():+.3f}]")

    # ---------- dilation-CoG (seed = scan-best mask, dilate 41 steps) ----------
    seed = (mom0_smooth > cfg.sthresh_scan * sigma_smooth) & sig_box
    if not seed.any():
        # try a softer threshold as fallback
        seed = (mom0_smooth > 5.0 * sigma_smooth) & sig_box
        print(f"  WARN: scan-best mask empty, fell back to 5σ")
    cur = seed.copy()
    dil_Nbeam = [seed.sum() / pix_per_beam_val]
    dil_F = [float(np.nansum(mom0[seed]) / pix_per_beam_val)]
    for _ in range(N_DILATION):
        cur = ndimage.binary_dilation(cur, iterations=1)
        dil_Nbeam.append(cur.sum() / pix_per_beam_val)
        dil_F.append(float(np.nansum(mom0[cur]) / pix_per_beam_val))
    dil_Nbeam = np.array(dil_Nbeam)
    dil_F = np.array(dil_F)
    print(f"  dilation-CoG: seed Nbeam={dil_Nbeam[0]:.2f}, "
          f"final Nbeam={dil_Nbeam[-1]:.2f}, "
          f"F range [{dil_F.min():+.3f}, {dil_F.max():+.3f}]")

    # plot per-galaxy
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sthresh_Nbeam, sthresh_F, "o-", color="C0",
            label=f"sthresh-CoG ({len(sthresh_arr)} cells)", markersize=5)
    ax.plot(dil_Nbeam, dil_F, "s-", color="C3",
            label=f"dilation-CoG ({N_DILATION} steps)", markersize=4)
    ax.scatter([dil_Nbeam[0]], [dil_F[0]], marker="x", color="black", s=80,
               zorder=10, label=f"seed Nbeam={dil_Nbeam[0]:.1f}")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mask size — Nbeam")
    ax.set_ylabel("Integrated flux F (Jy·km/s)")
    ax.set_title(
        f"{cfg.name} — Spatial CoG (sthresh vs dilation; cog_clean_detection_poc)\n"
        f"line={cfg.line}, W={cfg.W_kms:.0f} km/s, scan sf={cfg.sf_scan}, "
        f"sthresh_scan={cfg.sthresh_scan}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / "cog_clean_detection_poc.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "sthresh_Nbeam": sthresh_Nbeam,
        "sthresh_F": sthresh_F,
        "dilation_Nbeam": dil_Nbeam,
        "dilation_F": dil_F,
        "seed_Nbeam": float(dil_Nbeam[0]),
    }


# ------------------ configs ------------------
GALAXIES = [
    GalaxyConfig(
        name="Circinus",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/ESO097-013/ESO097-013_H40a_pbcor.fits",
        z=0.001448, line="H40a",
        W_kms=165.0, sf_scan=7, sthresh_scan=19.9,
        signal_ra=213.29152, signal_dec=-65.33909, signal_size_arcsec=35.0,
        noise_ra=213.282197, noise_dec=-65.335201, noise_size_arcsec=7.0,
    ),
    GalaxyConfig(
        name="NGC253",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/NGC253_H40a_pbcor.fits",
        z=0.00080, line="H40a",
        W_kms=180.0, sf_scan=3, sthresh_scan=32.0,
        signal_ra=11.88808, signal_dec=-25.28838, signal_size_arcsec=31.2,
        noise_ra=11.87983, noise_dec=-25.28024, noise_size_arcsec=12.8,
    ),
    GalaxyConfig(
        name="NGC3627",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3627/NGC3627_H30a_pbcor.fits",
        z=0.0024, line="H30a",
        # NGC 3627 phase0 uses sf=10 + W=60 (narrow line, heavy smooth),
        # but at sf=10 a 5-pix source is washed below 3σ in our self-masked
        # mom-0. Use sf=3 + larger W=300 for the CoG demo so both curves
        # have valid cells; this preserves the SHAPE of the CoG (which is
        # what we're studying), even if the absolute F differs from the
        # scan-best value.
        W_kms=300.0, sf_scan=3, sthresh_scan=10.0,
        signal_ra=170.06254, signal_dec=12.99161, signal_size_arcsec=40.0,
        noise_ra=170.04688, noise_dec=13.00549, noise_size_arcsec=20.0,
    ),
]


def main():
    results = [cog_for_galaxy(g) for g in GALAXIES]

    # combined panel: 3 new only (rich detail per panel; existing 4 stay separate)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, r in zip(axes, results):
        ax.plot(r["sthresh_Nbeam"], r["sthresh_F"], "o-", color="C0",
                label="sthresh-CoG", markersize=5)
        ax.plot(r["dilation_Nbeam"], r["dilation_F"], "s-", color="C3",
                label="dilation-CoG", markersize=4)
        ax.scatter([r["seed_Nbeam"]], [r["dilation_F"][0]],
                   marker="x", color="black", s=80, zorder=10)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Nbeam")
        ax.set_ylabel("F (Jy·km/s)")
        ax.set_title(f"{r['name']} ({r['line']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    out = OUT_ROOT / "_cog_clean_detection_3new_panel.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"\nsaved combined panel {out}")


if __name__ == "__main__":
    main()
