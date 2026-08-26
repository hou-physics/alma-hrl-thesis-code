"""
Apply baseline_diag's ε-prediction to the ellipse-scale CoG.

For each galaxy, compute:
  F_obs(scale)       = sum(mom0 over ellipse_mask) / pix_per_beam
  F_residual(scale)  = sum(ε map over ellipse_mask) * W / pix_per_beam
                       (with ±σ band from beam-correlated noise model)
  F_clean(scale)     = F_obs − F_residual

Hypothesis: if F_obs(scale) declines because of baseline residual
accumulating over larger ellipse area, F_clean(scale) should plateau.

Output:
  - per-galaxy 3-curve plot (F_obs / F_residual / F_clean with σ band)
  - 7-panel composite: BEFORE (F_obs) vs AFTER (F_clean) overlay
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy import ndimage
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402
from baseline_diag import (
    compute_eps_map,
    predict_F_residual,
    line_free_mask_for_cube,
)
from known_lines import LINES_REST_HZ
from ellipse_cog_poc import (
    GalaxyConfig,
    GALAXIES,
    keep_component_containing,
    fit_ellipse_weighted,
    ellipse_pixel_mask,
    beam_fwhm_pix,
    pix_per_beam,
    smooth_fft,
    make_box_mask,
    channel_velocities,
    N_SCALES,
    SCALE_MIN,
    SCALE_MAX,
)

OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")


def run_galaxy(cfg: GalaxyConfig):
    print(f"\n=== {cfg.name} ===")
    bundle = load_cube(cfg.cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)

    wcs2 = WCS(hdr).celestial
    sig_dec = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    noi_dec = cfg.noise_dec_size_arcsec or cfg.noise_size_arcsec
    sig_box, _ = make_box_mask(wcs2, ny, nx,
                               cfg.signal_ra, cfg.signal_dec,
                               cfg.signal_size_arcsec, sig_dec)
    noise_box, _ = make_box_mask(wcs2, ny, nx,
                                 cfg.noise_ra, cfg.noise_dec,
                                 cfg.noise_size_arcsec, noi_dec)

    # ---- line zone / line-free zone ----
    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    outside_line_buffer = np.abs(v_chan) > cfg.W_kms / 2.0 + dv
    cube_freq_lo = hdr["CRVAL3"] + (1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    cube_freq_hi = hdr["CRVAL3"] + (nchan - hdr["CRPIX3"]) * hdr["CDELT3"]
    cube_band = (min(cube_freq_lo, cube_freq_hi),
                 max(cube_freq_lo, cube_freq_hi))
    keep_known = line_free_mask_for_cube(
        v_chan=v_chan,
        primary_line_rest_hz=rest_obs_hz,
        cube_freq_range_hz=cube_band,
        z=cfg.z,
        primary_line_key=cfg.line,
        fwhm_kms=cfg.W_kms,
    )
    line_free = outside_line_buffer & keep_known
    n_free = int(line_free.sum())
    print(f"  line-free channels = {n_free}/{nchan}")

    # ---- mom-0 (line zone) and ε map (line-free zone) ----
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    eps = compute_eps_map(cube, line_free)
    eps_finite = eps[np.isfinite(eps)]
    sigma_eps = float(np.nanstd(eps_finite))
    print(f"  ε map: mean={np.nanmean(eps_finite):+.4e}, "
          f"per-beam σ={sigma_eps:.4e} Jy/beam")

    # ---- ellipse seed (3σ + connected component at brightest pixel) ----
    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    raw_mask = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw_mask, bx, by)
    if not seed.any():
        for try_st in [2.5, 2.0]:
            raw_mask = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw_mask, bx, by)
            if seed.any():
                break
    if not seed.any():
        print(f"  WARN: no seed, skipping")
        bundle.data = None  # type: ignore
        return None
    print(f"  seed Npix={int(seed.sum())} = {seed.sum()/ppb:.1f} beams")

    # ---- ellipse fit ----
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    fit = fit_ellipse_weighted((ys, xs), weights)
    if fit is None:
        bundle.data = None  # type: ignore
        return None
    xc, yc, a, b, PA = fit
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°")

    # ---- scale sweep ----
    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    masks = [ellipse_pixel_mask((ny, nx), xc, yc, a, b, PA, s)
             for s in scale_arr]

    # F_obs: integrate mom-0 over each ellipse
    F_obs = np.array([
        float(np.nansum(mom0[m]) / ppb) for m in masks
    ])
    # F_residual + σ via baseline_diag
    F_res, sigma_F_res = predict_F_residual(
        eps_map=eps,
        masks=masks,
        W_kms=cfg.W_kms,
        pix_per_beam=ppb,
    )
    F_clean = F_obs - F_res

    Nbeam_arr = np.array([int(m.sum()) / ppb for m in masks])
    F_obs_peak_idx = int(np.argmax(F_obs))
    F_clean_peak_idx = int(np.argmax(F_clean))

    print(f"  F_obs   range: [{F_obs.min():+.3f}, {F_obs.max():+.3f}], "
          f"peak {F_obs.max():+.3f} at scale={scale_arr[F_obs_peak_idx]:.2f}")
    print(f"  F_res   range: [{F_res.min():+.3f}, {F_res.max():+.3f}]")
    print(f"  F_clean range: [{F_clean.min():+.3f}, {F_clean.max():+.3f}], "
          f"peak {F_clean.max():+.3f} at scale={scale_arr[F_clean_peak_idx]:.2f}")

    # ---- plot per-galaxy (3 curves overlaid + before/after panels) ----
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: 3 curves
    ax = axes[0]
    ax.plot(scale_arr, F_obs, "o-", color="C3", label="F_obs", markersize=4)
    ax.plot(scale_arr, F_res, "s-", color="C2", label="F_residual (ε × W × Σ)",
            markersize=4)
    ax.fill_between(scale_arr, F_res - sigma_F_res, F_res + sigma_F_res,
                    color="C2", alpha=0.20, label="F_res ±1σ")
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label="F_clean = F_obs − F_residual", markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale (× seed)")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — 3 curves overlay")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: BEFORE vs AFTER (just F_obs vs F_clean, no F_res)
    ax = axes[1]
    ax.plot(scale_arr, F_obs, "o-", color="C3",
            label=f"BEFORE: F_obs (peak={F_obs.max():+.2f} @ s={scale_arr[F_obs_peak_idx]:.1f})",
            markersize=4)
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label=f"AFTER:  F_clean (peak={F_clean.max():+.2f} @ s={scale_arr[F_clean_peak_idx]:.1f})",
            markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale (× seed)")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — baseline correction effect")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"{cfg.name}  line={cfg.line}  W={cfg.W_kms:.0f} km/s  "
        f"sf={cfg.sf_scan}  ε_perBeam_σ={sigma_eps:.1e}",
        fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "ellipse_cog_baseline_corrected.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "scale": scale_arr,
        "Nbeam": Nbeam_arr,
        "F_obs": F_obs,
        "F_res": F_res,
        "sigma_F_res": sigma_F_res,
        "F_clean": F_clean,
        "F_obs_peak": F_obs.max(),
        "F_clean_peak": F_clean.max(),
        "scale_obs_peak": scale_arr[F_obs_peak_idx],
        "scale_clean_peak": scale_arr[F_clean_peak_idx],
    }


def main():
    results = []
    for g in GALAXIES:
        # NGC 3627 cube is ~33 GB to load — skip to avoid OOM.
        # (Its CoG was already shown to be noise-dominated; baseline
        # subtraction won't recover it given the W=300 vs scan-best W=60
        # mismatch upstream.)
        if g.name == "NGC3627":
            print(f"\n=== {g.name} SKIPPED (cube too large for current RAM budget) ===")
            continue
        try:
            r = run_galaxy(g)
        except Exception as e:
            print(f"  ERROR on {g.name}: {e}")
            r = None
        if r is not None:
            results.append(r)

    # 7-panel composite: BEFORE (red) vs AFTER (blue)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["scale"], r["F_obs"], "o-", color="C3",
                label=f"BEFORE peak {r['F_obs_peak']:+.2f}",
                markersize=3)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"AFTER  peak {r['F_clean_peak']:+.2f}",
                markersize=3)
        ax.fill_between(r["scale"],
                        r["F_clean"] - r["sigma_F_res"],
                        r["F_clean"] + r["sigma_F_res"],
                        color="C0", alpha=0.15)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("ellipse scale")
        ax.set_ylabel("F (Jy·km/s)")
        delta = r["F_clean_peak"] - r["F_obs_peak"]
        ax.set_title(
            f"{r['name']} ({r['line']})\n"
            f"Δpeak = {delta:+.2f}, "
            f"clean plateau flatness improved?"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Ellipse CoG: BEFORE (F_obs, red) vs AFTER baseline subtraction (F_clean, blue)\n"
        "Hypothesis: F_clean should plateau where F_obs declines",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_ellipse_cog_before_after_baseline.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 7-panel composite {out}")


if __name__ == "__main__":
    main()
