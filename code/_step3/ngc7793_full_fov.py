"""
Wider FOV view of NGC 7793 — see where the compact source is relative
to the galaxy disk + NED center + signal box.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Ellipse, Rectangle, Circle
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402
from ellipse_cog_poc import (
    keep_component_containing,
    fit_ellipse_weighted,
    beam_fwhm_pix,
    pix_per_beam,
    smooth_fft,
    make_box_mask,
    channel_velocities,
)
from known_lines import LINES_REST_HZ


OUT = Path("/Volumes/HouAstro/master/result_v2/NGC7793/full_fov_overlay.png")
OUT_TMP = Path("/Volumes/HouAstro/master/result_v2/tmp/NGC7793__full_fov.png")

CUBE_PATH = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pbcor.fits"
PB_PATH = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pb.fits"

z = 0.000749
line = "H30a"
W_kms = 300.0
sf_scan = 3
signal_ra = 359.457210
signal_dec = -32.590990
signal_size_arcsec = 45.0
noise_ra = 359.437428
noise_dec = -32.588212
noise_size_arcsec = 20.0

# NGC 7793 known properties (optical, NED)
opt_diam_arcmin = 9.3  # D25 from RC3, NED
opt_pa_deg = 99.0  # major-axis PA (approximate)
opt_ba = 0.65  # axis ratio


def main():
    print(f"loading {CUBE_PATH}")
    bundle = load_cube(CUBE_PATH)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ[line] / (1.0 + z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    pix_arcsec = abs(hdr["CDELT1"]) * 3600
    print(f"  cube {cube.shape}, dv={dv:.2f}, pix/beam={ppb:.2f}, "
          f"pixel scale={pix_arcsec:.3f}\"/pix")

    wcs2 = WCS(hdr).celestial
    sig_box, _ = make_box_mask(wcs2, ny, nx, signal_ra, signal_dec,
                               signal_size_arcsec, signal_size_arcsec)
    noise_box, _ = make_box_mask(wcs2, ny, nx, noise_ra, noise_dec,
                                 noise_size_arcsec, noise_size_arcsec)

    in_line = np.abs(v_chan) <= W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    kernel_fwhm = sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))

    raw_mask = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw_mask, bx, by)
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)

    # NED center pixel
    ned_xc, ned_yc = wcs2.world_to_pixel_values(signal_ra, signal_dec)
    print(f"  NED center pixel: ({ned_xc:.1f}, {ned_yc:.1f})")
    print(f"  seed best pixel: ({xc:.1f}, {yc:.1f}), "
          f"offset {((xc-ned_xc)**2+(yc-ned_yc)**2)**0.5*pix_arcsec:.1f}\" from NED")

    # Load PB for context
    pb = None
    try:
        pb_bundle = load_cube(PB_PATH)
        # PB cube might be 3D (per-channel); take median across channels
        if pb_bundle.data.ndim == 3:
            pb = np.nanmedian(pb_bundle.data, axis=0)
        else:
            pb = pb_bundle.data
        pb_bundle.data = None  # type: ignore
        print(f"  loaded PB shape {pb.shape}")
    except Exception as e:
        print(f"  no PB: {e}")

    # ---------- Figure: 1 wide FOV + 1 zoom ----------
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # ---- LEFT: wide FOV (entire cube extent) ----
    ax = axes[0]
    finite = np.isfinite(mom0_smooth)
    if finite.any():
        vmax = float(np.nanpercentile(mom0_smooth[finite], 99.5))
        vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    else:
        vmax = float(np.nanmax(mom0_smooth)); vmin = -vmax / 5
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")

    # PB > 0.5 contour
    if pb is not None:
        ax.contour(pb, levels=[0.5], colors="white", linewidths=1.2,
                   linestyles="-", alpha=0.7)
        ax.plot([], [], color="white", linestyle="-", linewidth=1.2,
                label="PB > 0.5 (ALMA pointing)")

    # Optical D25 ellipse (centered at NED, PA from RC3)
    opt_diam_pix = opt_diam_arcmin * 60 / pix_arcsec
    opt_ell = Ellipse((ned_xc, ned_yc),
                      opt_diam_pix, opt_diam_pix * opt_ba,
                      angle=opt_pa_deg,
                      fill=False, edgecolor="lime", linewidth=2,
                      linestyle="--")
    ax.add_patch(opt_ell)
    ax.plot([], [], color="lime", linestyle="--", linewidth=2,
            label=f"NGC 7793 optical D25 (≈ {opt_diam_arcmin}', PA={opt_pa_deg}°)")

    # NED center marker
    ax.plot(ned_xc, ned_yc, "+", color="lime", markersize=20, markeredgewidth=2,
            label="NED center")

    # signal box
    sb_y, sb_x = np.where(sig_box)
    rect = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                     sb_x.max() - sb_x.min() + 1,
                     sb_y.max() - sb_y.min() + 1,
                     fill=False, edgecolor="cyan", linestyle=":",
                     linewidth=1.5, label="signal box (45″)")
    ax.add_patch(rect)

    # Seed centroid marker
    ax.plot(xc, yc, "x", color="red", markersize=15, markeredgewidth=2,
            label=f"detected peak (off NED by 5.2″)")

    # 3σ contour everywhere (not just in sig_box) to look for other sources
    ax.contour(mom0_smooth, levels=[3 * sigma_smooth, 5 * sigma_smooth],
               colors=["orange", "yellow"], linewidths=0.7, alpha=0.6)
    ax.plot([], [], color="orange", linewidth=0.7, label="3σ contour (whole cube)")
    ax.plot([], [], color="yellow", linewidth=0.7, label="5σ contour")

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(
        f"NGC 7793 — full ALMA cube FOV ({nx}×{ny} pix = "
        f"{nx*pix_arcsec/60:.1f}'×{ny*pix_arcsec/60:.1f}')\n"
        f"vs optical D25 = {opt_diam_arcmin}' (lime), signal box = "
        f"{signal_size_arcsec}″ (cyan)"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ---- RIGHT: zoom on detection ----
    ax = axes[1]
    pad = max(int(a * 6), 40)
    zx_lo = max(0, int(xc - pad)); zx_hi = min(nx, int(xc + pad))
    zy_lo = max(0, int(yc - pad)); zy_hi = min(ny, int(yc + pad))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    ax.set_xlim(zx_lo, zx_hi)
    ax.set_ylim(zy_lo, zy_hi)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")
    ax.contour(seed.astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.plot([], [], color="cyan", lw=1.5,
            label=f"seed @ 3σ (CC, {seed.sum()/ppb:.1f} beams)")
    ax.contour(mom0_smooth, levels=[3 * sigma_smooth, 5 * sigma_smooth,
                                    7 * sigma_smooth],
               colors=["orange", "yellow", "white"],
               linewidths=0.8, alpha=0.8)
    ax.plot([], [], color="orange", lw=0.8, label="3σ")
    ax.plot([], [], color="yellow", lw=0.8, label="5σ")
    ax.plot([], [], color="white", lw=0.8, label="7σ")

    # NED center marker
    ax.plot(ned_xc, ned_yc, "+", color="lime", markersize=20,
            markeredgewidth=2, label="NED center")
    ax.plot(xc, yc, "x", color="red", markersize=15, markeredgewidth=2)

    # ellipses
    OVERLAY_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]
    COLORS = ["red", "orange", "yellow", "white", "magenta"]
    for s, c in zip(OVERLAY_SCALES, COLORS):
        ell = Ellipse((xc, yc), 2*a*s, 2*b*s, angle=np.degrees(PA),
                      fill=False, edgecolor=c, linewidth=1.5)
        ax.add_patch(ell)
        ax.plot([], [], color=c, linewidth=1.5, label=f"scale={s}")

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(
        f"zoom @ detected peak — offset {((xc-ned_xc)**2+(yc-ned_yc)**2)**0.5*pix_arcsec:.1f}″ from NED center"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"NGC 7793 (D=3.6 Mpc, Sd, optical 9.3'×6.3') — "
        f"compact peak detected inside small 45″ signal box\n"
        f"σ_smooth = {sigma_smooth:.4f}, peak brightness = "
        f"{float(np.nanmax(mom0_smooth[sig_box]))/sigma_smooth:.1f}σ",
        fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")
    import shutil
    shutil.copy(OUT, OUT_TMP)
    print(f"copied to {OUT_TMP}")
    bundle.data = None  # type: ignore


if __name__ == "__main__":
    main()
