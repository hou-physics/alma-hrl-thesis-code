"""
Visualize manual vs auto signal_box + noise_box on 6 detection galaxies.

For each galaxy:
  - HRL mom-0 (smoothed) as background
  - All photutils detected segments (white contours, dim)
  - Target source segment (red contour)
  - AUTO signal_box (red rectangle, from bbox of target segment)
  - AUTO noise_box (cyan rectangle, from grid search)
  - MANUAL signal_box (yellow dashed, from step3_analyze.py)
  - MANUAL noise_box (orange dashed, from step3_analyze.py)
  - NED center marker (lime +)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Rectangle
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube
from ellipse_cog_poc import (
    GalaxyConfig, GALAXIES,
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
)
from known_lines import LINES_REST_HZ
from matched_null_test import load_pb_mask
from auto_regions_poc import (
    detect_signal_segments, select_target_segment, find_best_noise_box,
)

OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2/tmp")


def viz_galaxy(cfg: GalaxyConfig):
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
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    mom0_smooth = smooth_fft(mom0, cfg.sf_scan * beam_fwhm)

    # PB mask
    pb_path = cfg.cube_path.replace("_pbcor.fits", "_pb.fits")
    if not Path(pb_path).exists():
        pb_path = pb_path + ".gz"
    pb_mask = load_pb_mask(pb_path) if Path(pb_path).exists() else None

    # photutils source detection on PB-masked HRL mom-0
    mom0_for_det = mom0_smooth.copy()
    if pb_mask is not None:
        mom0_for_det[~pb_mask] = np.nan
    segm, _ = detect_signal_segments(mom0_for_det, nsigma=3.0, npixels=10)

    # match to NED
    target_label = None
    target_mask = None
    auto_sig_bbox = None
    if segm is not None and segm.nlabels > 0:
        target_label, _ = select_target_segment(segm, wcs2,
                                                 cfg.signal_ra, cfg.signal_dec,
                                                 max_dist_arcsec=30)
        if target_label is not None:
            target_mask = (segm.data == target_label)
            ys, xs = np.where(target_mask)
            auto_sig_bbox = (xs.min(), ys.min(),
                             xs.max() - xs.min() + 1,
                             ys.max() - ys.min() + 1)
            print(f"  AUTO sig bbox: {auto_sig_bbox[2]*pix_arcsec:.1f}\" × "
                  f"{auto_sig_bbox[3]*pix_arcsec:.1f}\"")

    # auto noise box (using cfg manual noise size for comparison)
    noise_xc = noise_yc = None
    noise_score = None
    if segm is not None and segm.nlabels > 0:
        source_mask = (segm.data > 0)
        noise_box_pix = int(cfg.noise_size_arcsec / pix_arcsec)
        exclusion_pix = int(3 * beam_fwhm)
        nbox = find_best_noise_box(mom0, pb_mask, source_mask, noise_box_pix,
                                    min_dist_from_source_pix=exclusion_pix)
        if nbox is not None:
            noise_xc, noise_yc, _, noise_score, _, _ = nbox
            print(f"  AUTO noise box: ({noise_xc}, {noise_yc}), score={noise_score:.4f}")

    # manual region pixel coords
    manual_sig_xc, manual_sig_yc = wcs2.world_to_pixel_values(
        cfg.signal_ra, cfg.signal_dec)
    sig_dec_size = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    manual_sig_half_x = int(cfg.signal_size_arcsec / 2 / pix_arcsec)
    manual_sig_half_y = int(sig_dec_size / 2 / pix_arcsec)
    manual_noise_xc, manual_noise_yc = wcs2.world_to_pixel_values(
        cfg.noise_ra, cfg.noise_dec)
    manual_noise_half = int(cfg.noise_size_arcsec / 2 / pix_arcsec)
    ned_xc, ned_yc = wcs2.world_to_pixel_values(cfg.signal_ra, cfg.signal_dec)

    # full FOV plot
    fig, ax = plt.subplots(figsize=(11, 10))
    finite = np.isfinite(mom0_smooth)
    vmax = float(np.nanpercentile(mom0_smooth[finite], 99))
    vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="HRL mom-0 smoothed (Jy/beam · km/s)")

    # PB > 0.5 outline
    if pb_mask is not None:
        ax.contour(pb_mask.astype(int), levels=[0.5], colors="white",
                   linewidths=0.8, alpha=0.7)
        ax.plot([], [], color="white", lw=0.8, label="PB > 0.5")

    # all detected segments (light contours)
    if segm is not None and segm.nlabels > 0:
        all_src = (segm.data > 0).astype(int)
        ax.contour(all_src, levels=[0.5], colors="white",
                   linewidths=0.4, alpha=0.5)
        ax.plot([], [], color="white", lw=0.4, alpha=0.6,
                label=f"all {segm.nlabels} photutils detections (3σ)")

    # target segment + auto sig bbox
    if target_mask is not None:
        ax.contour(target_mask.astype(int), levels=[0.5], colors="red",
                   linewidths=2.0)
        rect_auto_sig = Rectangle(
            (auto_sig_bbox[0] - 0.5, auto_sig_bbox[1] - 0.5),
            auto_sig_bbox[2], auto_sig_bbox[3],
            fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect_auto_sig)
        ax.plot([], [], color="red", lw=2,
                label=f"AUTO sig_box {auto_sig_bbox[2]*pix_arcsec:.1f}\""
                      f"×{auto_sig_bbox[3]*pix_arcsec:.1f}\"")

    # auto noise box
    if noise_xc is not None:
        noise_box_pix = int(cfg.noise_size_arcsec / pix_arcsec)
        rect_auto_n = Rectangle(
            (noise_xc - noise_box_pix // 2 - 0.5,
             noise_yc - noise_box_pix // 2 - 0.5),
            noise_box_pix, noise_box_pix,
            fill=False, edgecolor="cyan", linewidth=1.8)
        ax.add_patch(rect_auto_n)
        ax.plot([], [], color="cyan", lw=1.8,
                label=f"AUTO noise_box (score={noise_score:.4f})")

    # MANUAL sig box
    rect_msig = Rectangle(
        (manual_sig_xc - manual_sig_half_x - 0.5,
         manual_sig_yc - manual_sig_half_y - 0.5),
        2 * manual_sig_half_x, 2 * manual_sig_half_y,
        fill=False, edgecolor="yellow", linestyle="--", linewidth=1.5)
    ax.add_patch(rect_msig)
    ax.plot([], [], color="yellow", ls="--", lw=1.5,
            label=f"MANUAL sig_box {cfg.signal_size_arcsec:.0f}\""
                  f"×{sig_dec_size:.0f}\"")

    # MANUAL noise box
    rect_mn = Rectangle(
        (manual_noise_xc - manual_noise_half - 0.5,
         manual_noise_yc - manual_noise_half - 0.5),
        2 * manual_noise_half, 2 * manual_noise_half,
        fill=False, edgecolor="orange", linestyle="--", linewidth=1.5)
    ax.add_patch(rect_mn)
    ax.plot([], [], color="orange", ls="--", lw=1.5,
            label=f"MANUAL noise_box {cfg.noise_size_arcsec:.0f}\"")

    # NED marker
    ax.plot(ned_xc, ned_yc, "+", color="lime", markersize=18,
            markeredgewidth=2, label=f"NED ({cfg.signal_ra:.4f}, {cfg.signal_dec:.4f})")

    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(
        f"{cfg.name} — line={cfg.line}, W={cfg.W_kms:.0f} km/s, sf={cfg.sf_scan}\n"
        f"auto regions vs manual config; mom-0 smoothed"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    out = OUT_ROOT / f"{cfg.name}__auto_vs_manual_regions.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")
    bundle.data = None  # type: ignore


def main():
    targets = [g for g in GALAXIES if g.name != "NGC3627"]
    for cfg in targets:
        try:
            viz_galaxy(cfg)
        except Exception as e:
            print(f"  ERROR on {cfg.name}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
