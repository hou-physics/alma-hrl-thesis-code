"""
Generate mom-0 image of NGC 7793 with seed mask + multiple scale ellipses
overlaid, to visualize whether the F(scale) plateau at scale 4-5 is
supported by visible extended emission or is just noise integration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Ellipse, Rectangle
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402
from ellipse_cog_poc import (
    GalaxyConfig,
    keep_component_containing,
    fit_ellipse_weighted,
    ellipse_pixel_mask,
    beam_fwhm_pix,
    pix_per_beam,
    smooth_fft,
    make_box_mask,
    channel_velocities,
)
from known_lines import LINES_REST_HZ

OUT = Path("/Volumes/HouAstro/master/result_v2/NGC7793/mom0_ellipse_overlay.png")
OUT_TMP = Path("/Volumes/HouAstro/master/result_v2/tmp/NGC7793__mom0_overlay.png")

cfg = GalaxyConfig(
    name="NGC7793",
    cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pbcor.fits",
    z=0.000749, line="H30a",
    W_kms=300.0, sf_scan=3,
    signal_ra=359.457210, signal_dec=-32.590990,
    signal_size_arcsec=45.0, signal_dec_size_arcsec=None,
    noise_ra=359.437428, noise_dec=-32.588212,
    noise_size_arcsec=20.0, noise_dec_size_arcsec=None,
)

OVERLAY_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]
COLORS = ["red", "orange", "yellow", "white", "magenta"]


def main():
    print(f"loading {cfg.cube_path}")
    bundle = load_cube(cfg.cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    print(f"  cube {cube.shape}, dv={dv:.2f}, pix/beam={ppb:.2f}")

    wcs2 = WCS(hdr).celestial
    sig_box, (cx_box, cy_box) = make_box_mask(
        wcs2, ny, nx,
        cfg.signal_ra, cfg.signal_dec,
        cfg.signal_size_arcsec, cfg.signal_size_arcsec)
    noise_box, _ = make_box_mask(
        wcs2, ny, nx,
        cfg.noise_ra, cfg.noise_dec,
        cfg.noise_size_arcsec, cfg.noise_size_arcsec)

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    print(f"  σ_smooth (in noise box) = {sigma_smooth:.5f}")

    # seed at 3σ + CC at brightest pixel in sig_box
    raw_mask = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw_mask, bx, by)
    print(f"  seed: {int(seed.sum())} pix = {seed.sum()/ppb:.2f} beams")

    # ellipse fit
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°, b/a={b/max(a,1e-9):.2f}")

    # display window: scale=5 ellipse + 30% padding
    max_ext = a * 5.0
    pad = max(int(max_ext * 1.3), 30)
    x_lo = max(0, int(xc - pad)); x_hi = min(nx, int(xc + pad))
    y_lo = max(0, int(yc - pad)); y_hi = min(ny, int(yc + pad))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ---- left: mom-0 (line-zone integrated) with sig_box + seed + ellipses ----
    ax = axes[0]
    finite_mask = np.isfinite(mom0) & sig_box
    if finite_mask.any():
        vmax = float(np.nanpercentile(mom0[finite_mask], 99))
        vmin = float(np.nanpercentile(mom0[finite_mask], 1))
    else:
        vmax = float(np.nanmax(mom0)); vmin = -vmax / 5
    im = ax.imshow(mom0, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    plt.colorbar(im, ax=ax, label="mom-0 (Jy/beam · km/s)")

    # signal box outline (dotted white)
    sb_y, sb_x = np.where(sig_box)
    rect = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                     sb_x.max() - sb_x.min() + 1,
                     sb_y.max() - sb_y.min() + 1,
                     fill=False, edgecolor="white", linestyle=":",
                     linewidth=1.0, label="signal box (45″)")
    ax.add_patch(rect)
    # seed contour (cyan)
    ax.contour(seed.astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.plot([], [], color="cyan", linewidth=1.5,
            label=f"seed @ 3σ (CC, {int(seed.sum())} pix, "
                  f"{seed.sum()/ppb:.1f} beams)")
    # ellipses at the 5 overlay scales
    for s, c in zip(OVERLAY_SCALES, COLORS):
        ell = Ellipse((xc, yc), 2*a*s, 2*b*s,
                      angle=np.degrees(PA),
                      fill=False, edgecolor=c, linewidth=1.5)
        ax.add_patch(ell)
        ax.plot([], [], color=c, linewidth=1.5, label=f"scale={s}")
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(
        "NGC 7793 — HRL mom-0 over W=300 km/s\n"
        "seed (cyan) + ellipse scale overlays + signal box (white dotted)"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ---- right: smoothed mom-0 with same overlays (sees structure clearer) ----
    ax = axes[1]
    finite_mask_sm = np.isfinite(mom0_smooth) & sig_box
    vmax_s = float(np.nanpercentile(mom0_smooth[finite_mask_sm], 99))
    vmin_s = float(np.nanpercentile(mom0_smooth[finite_mask_sm], 1))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin_s, vmax=vmax_s)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")

    rect2 = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                      sb_x.max() - sb_x.min() + 1,
                      sb_y.max() - sb_y.min() + 1,
                      fill=False, edgecolor="white", linestyle=":",
                      linewidth=1.0)
    ax.add_patch(rect2)
    ax.contour(seed.astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.contour(mom0_smooth, levels=[sigma_smooth * s for s in [3, 5, 7]],
               colors=["orange", "yellow", "white"],
               linewidths=0.8, alpha=0.7)
    for s, c in zip(OVERLAY_SCALES, COLORS):
        ell = Ellipse((xc, yc), 2*a*s, 2*b*s,
                      angle=np.degrees(PA),
                      fill=False, edgecolor=c, linewidth=1.5)
        ax.add_patch(ell)
    ax.plot([], [], color="orange", lw=0.8, label="3σ contour")
    ax.plot([], [], color="yellow", lw=0.8, label="5σ contour")
    ax.plot([], [], color="white", lw=0.8, label="7σ contour")
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(
        f"smoothed (sf={cfg.sf_scan}×beam_FWHM) — "
        f"3/5/7σ contours (σ={sigma_smooth:.4f})"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"NGC 7793 [smooth-disk candidate detection] — W={cfg.W_kms:.0f} km/s, "
        f"sf={cfg.sf_scan}, b/a={b/max(a,1e-9):.2f}, PA={np.degrees(PA):.0f}°",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")
    # copy to tmp for easy viewing
    import shutil
    shutil.copy(OUT, OUT_TMP)
    print(f"copied to {OUT_TMP}")
    bundle.data = None  # type: ignore


if __name__ == "__main__":
    main()
