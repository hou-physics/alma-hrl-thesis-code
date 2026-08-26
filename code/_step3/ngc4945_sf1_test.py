"""
Re-run NGC 4945 photutils detection with sf=1 (no extra smoothing beyond
beam) vs sf=2. Visualize side-by-side to confirm sf=2 was the culprit.
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
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
)
from known_lines import LINES_REST_HZ
from matched_null_test import load_pb_mask
from auto_regions_poc import (
    detect_signal_segments, select_target_segment, find_best_noise_box,
)

CUBE = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits"
PB = "/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_nonpbcor.pbmask_0.20.fits"  # 2D bool mask
z = 0.00188
W = 315.0
NED_RA, NED_DEC = 196.36260, -49.46942
MANUAL_SIG_SIZE = (17.0, 19.0)
MANUAL_NOISE_RA, MANUAL_NOISE_DEC = 196.36901, -49.46442
MANUAL_NOISE_SIZE = 7.0
OUT = Path("/Volumes/HouAstro/master/result_v2/tmp/NGC4945__sf_comparison.png")


def run_for_sf(mom0, beam_fwhm, sf, pb_mask, wcs2, pix_arcsec):
    """Smooth at given sf, run photutils, return (smoothed, segm, n_seg, target_label, target_bbox)."""
    if sf == 0:
        mom0_smooth = mom0.copy()
    else:
        mom0_smooth = smooth_fft(mom0, sf * beam_fwhm)
    mom0_for_det = mom0_smooth.copy()
    if pb_mask is not None:
        mom0_for_det[~pb_mask] = np.nan
    segm, _ = detect_signal_segments(mom0_for_det, nsigma=3.0, npixels=10)
    if segm is None:
        return mom0_smooth, None, 0, None, None
    target_label, _ = select_target_segment(segm, wcs2, NED_RA, NED_DEC,
                                              max_dist_arcsec=30)
    bbox = None
    if target_label is not None:
        target_mask = (segm.data == target_label)
        ys, xs = np.where(target_mask)
        bbox = (xs.min(), ys.min(),
                xs.max() - xs.min() + 1,
                ys.max() - ys.min() + 1)
    return mom0_smooth, segm, segm.nlabels, target_label, bbox


def main():
    print("loading NGC 4945 cube...")
    bundle = load_cube(CUBE)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ["H30a"] / (1.0 + z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600
    print(f"  beam_FWHM={beam_fwhm:.2f} pix, pixel scale={pix_arcsec:.3f}\"/pix")

    in_line = np.abs(v_chan) <= W / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    print(f"  mom-0 range: [{np.nanmin(mom0):+.3f}, {np.nanmax(mom0):+.3f}]")

    # pbmask_0.20.fits is a 3D cube with NaN outside PB>0.20 region
    from astropy.io import fits
    with fits.open(PB) as hdul:
        pb_data = hdul[0].data
        # take middle channel, mask = finite = PB > 0.20
        if pb_data.ndim == 3:
            pb_data = pb_data[pb_data.shape[0] // 2]
        pb_mask = np.isfinite(pb_data)
    print(f"  pbmask>0.20: {pb_mask.sum()} pix")

    # Three smoothing levels
    sf_values = [0, 1, 2]
    results = []
    for sf in sf_values:
        smoothed, segm, n, tlabel, bbox = run_for_sf(
            mom0, beam_fwhm, sf, pb_mask, wcs2, pix_arcsec)
        bbox_arc = (bbox[2] * pix_arcsec, bbox[3] * pix_arcsec) if bbox else None
        print(f"  sf={sf}: detected {n} sources, target bbox = {bbox_arc}")
        results.append((sf, smoothed, segm, n, tlabel, bbox))

    bundle.data = None  # type: ignore

    # plot 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, (sf, sm, segm, n, tlabel, bbox) in zip(axes, results):
        finite = np.isfinite(sm)
        # Match production: high vmin to clip noise
        vmax_prod = float(np.nanpercentile(sm[finite], 99))
        vmin_prod = 0.2 * vmax_prod  # rough "show only top 80% range"
        im = ax.imshow(sm, origin="lower", cmap="cmr.cosmic",
                       vmin=vmin_prod, vmax=vmax_prod)
        plt.colorbar(im, ax=ax, label="mom-0 (Jy/beam · km/s)")
        # all detections white outlines
        if segm is not None and n > 0:
            ax.contour((segm.data > 0).astype(int), levels=[0.5],
                       colors="white", linewidths=0.4, alpha=0.5)
        if bbox is not None:
            target_mask = (segm.data == tlabel)
            ax.contour(target_mask.astype(int), levels=[0.5],
                       colors="red", linewidths=2.0)
            rect = Rectangle(
                (bbox[0] - 0.5, bbox[1] - 0.5), bbox[2], bbox[3],
                fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(rect)
        # MANUAL sig box for reference
        mxc, myc = wcs2.world_to_pixel_values(NED_RA, NED_DEC)
        mhx = int(MANUAL_SIG_SIZE[0] / 2 / pix_arcsec)
        mhy = int(MANUAL_SIG_SIZE[1] / 2 / pix_arcsec)
        ax.add_patch(Rectangle(
            (mxc - mhx - 0.5, myc - mhy - 0.5), 2 * mhx, 2 * mhy,
            fill=False, edgecolor="yellow", linestyle="--", linewidth=1.5))
        ax.plot(mxc, myc, "+", color="lime", markersize=15, markeredgewidth=2)
        # title
        sf_label = "sf=0 (no smoothing)" if sf == 0 else f"sf={sf}×beam"
        title = (f"{sf_label}\n"
                 f"photutils: {n} detections, target bbox: "
                 f"{bbox[2]*pix_arcsec:.1f}\"×{bbox[3]*pix_arcsec:.1f}\""
                 if bbox else f"{sf_label}\n{n} detections, no target")
        ax.set_title(title)
        ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    plt.suptitle("NGC 4945 — photutils source detection vs smoothing level",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
