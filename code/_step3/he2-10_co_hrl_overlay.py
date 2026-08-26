"""
He 2-10: HRL (H40α) mom-0 + CS(2-1) contour + HRL contour overlay.

Demonstrates the CO/CS cavity vs HRL central peak geometry that
motivates HRL self-mask (vs CO-mask-based) methodology.

Literature (Johnson+ 2018 ApJ 853, 125; Imara & Faesi 2019 ApJ 882,
162; Vanzi+ 2009 A&A): CO clears around mature SSCs from feedback;
HRL fills the cavity. Our cube directly shows this.

Both H40α and CS(2-1) live in the SAME single-SPW cube (degenerate
mode); we integrate over the line zone of each line separately.

Output: result_v2/tmp/He2-10__CO_HRL_overlay.png + thesis_figure/ copy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube
from ellipse_cog_poc import (
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
)
from known_lines import LINES_REST_HZ

# He 2-10 setup — H40α and CS(2-1) in same cube
CUBE = "/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pbcor.fits"
PB = "/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pb.fits"

z = 0.002912
H40A_REST_HZ = LINES_REST_HZ["H40a"]
CS21_REST_HZ = LINES_REST_HZ["CS2-1"]
W_HRL_KMS = 75.0           # scan-best for H40α
W_CS_KMS = 200.0           # CS is broader

# He 2-10 NED + step3 manual regions
NED_RA, NED_DEC = 129.06329, -26.40935
SIG_SIZE = 16.0
NOISE_RA, NOISE_DEC = 129.05833, -26.40491
NOISE_SIZE = 11.0

OUT = Path("/Volumes/HouAstro/master/result_v2/tmp/He2-10__CO_HRL_overlay.png")


def main():
    print("loading He 2-10 cube...")
    bundle = load_cube(CUBE)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600
    print(f"  cube {cube.shape}, pix/beam={ppb:.1f}, pixel scale={pix_arcsec:.3f}\"/pix")

    # PB
    pb_bundle = load_cube(PB)
    pb = pb_bundle.data
    if pb.ndim == 3:
        pb = pb[pb.shape[0] // 2]
    pb_bundle.data = None  # type: ignore
    pb_mask = (pb > 0.5) & np.isfinite(pb)
    print(f"  PB > 0.5: {pb_mask.sum()} pix")

    # ---- compute velocity grids for each line (independent rest frequencies) ----
    rest_obs_h40a = H40A_REST_HZ / (1.0 + z)
    rest_obs_cs21 = CS21_REST_HZ / (1.0 + z)
    v_h40a = channel_velocities(hdr, rest_obs_h40a)
    v_cs21 = channel_velocities(hdr, rest_obs_cs21)
    dv = float(np.abs(np.median(np.diff(v_h40a))))
    print(f"  dv = {dv:.2f} km/s, "
          f"v_h40a range [{v_h40a.min():.0f}, {v_h40a.max():.0f}] km/s, "
          f"v_cs21 range [{v_cs21.min():.0f}, {v_cs21.max():.0f}] km/s")

    # HRL mom-0
    in_h40a = np.abs(v_h40a) <= W_HRL_KMS / 2.0
    hrl_mom0 = np.nansum(cube[in_h40a, :, :], axis=0) * dv
    # CS mom-0
    in_cs21 = np.abs(v_cs21) <= W_CS_KMS / 2.0
    cs_mom0 = np.nansum(cube[in_cs21, :, :], axis=0) * dv

    # smoothed versions for sigma estimation
    hrl_smooth = smooth_fft(hrl_mom0, 1.0 * beam_fwhm)  # sf=1
    cs_smooth = smooth_fft(cs_mom0, 1.0 * beam_fwhm)

    # noise box for sigma estimation
    noise_box, _ = make_box_mask(wcs2, ny, nx, NOISE_RA, NOISE_DEC,
                                  NOISE_SIZE, NOISE_SIZE)
    sigma_hrl = float(np.nanstd(hrl_smooth[noise_box]))
    sigma_cs = float(np.nanstd(cs_smooth[noise_box]))
    print(f"  σ_HRL = {sigma_hrl:.4f}, σ_CS = {sigma_cs:.4f} Jy/beam·km/s")

    # zoom tight around source
    cx, cy = wcs2.world_to_pixel_values(NED_RA, NED_DEC)
    half_pix = int(8 / pix_arcsec)  # 8" half-width → 16" total view
    x0, x1 = int(cx - half_pix), int(cx + half_pix)
    y0, y1 = int(cy - half_pix), int(cy + half_pix)

    # report peak positions
    sig_box, _ = make_box_mask(wcs2, ny, nx, NED_RA, NED_DEC, SIG_SIZE, SIG_SIZE)
    hrl_in_sig = np.where(sig_box, hrl_smooth, -np.inf)
    cs_in_sig = np.where(sig_box, cs_smooth, -np.inf)
    hrl_pi = np.unravel_index(np.argmax(hrl_in_sig), hrl_in_sig.shape)
    cs_pi = np.unravel_index(np.argmax(cs_in_sig), cs_in_sig.shape)
    hrl_world = wcs2.pixel_to_world_values(hrl_pi[1], hrl_pi[0])
    cs_world = wcs2.pixel_to_world_values(cs_pi[1], cs_pi[0])
    offset_arcsec = (((hrl_pi[1]-cs_pi[1])**2 + (hrl_pi[0]-cs_pi[0])**2)**0.5
                     * pix_arcsec)
    print(f"  HRL peak:  pixel ({hrl_pi[1]:.0f},{hrl_pi[0]:.0f}) "
          f"→ ({hrl_world[0]:.5f}, {hrl_world[1]:.5f}), "
          f"value {hrl_smooth[hrl_pi]:.4f}")
    print(f"  CS peak:   pixel ({cs_pi[1]:.0f},{cs_pi[0]:.0f}) "
          f"→ ({cs_world[0]:.5f}, {cs_world[1]:.5f}), "
          f"value {cs_smooth[cs_pi]:.4f}")
    print(f"  HRL-CS offset: {offset_arcsec:.2f}\" "
          f"(= {offset_arcsec/(beam_fwhm*pix_arcsec):.2f} × beam)")
    print(f"  CS at HRL peak: {cs_smooth[hrl_pi]:.4f} "
          f"({cs_smooth[hrl_pi]/sigma_cs:.1f}σ)")
    print(f"  HRL at CS peak: {hrl_smooth[cs_pi]:.4f} "
          f"({hrl_smooth[cs_pi]/sigma_hrl:.1f}σ)")

    # ---- 3-panel figure ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: HRL mom-0 only
    ax = axes[0]
    finite = np.isfinite(hrl_smooth) & pb_mask
    vmax = float(np.nanpercentile(hrl_smooth[finite], 99.5))
    vmin = float(np.nanpercentile(hrl_smooth[finite], 1))
    im = ax.imshow(hrl_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="H40α mom-0 (Jy/beam · km/s)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.plot(cx, cy, "+", color="lime", markersize=18, markeredgewidth=2.5,
            label="NED center")
    ax.set_title(f"He 2-10 — H40α mom-0 (W={W_HRL_KMS:.0f} km/s)\n"
                 f"σ={sigma_hrl:.4f}, peak={hrl_smooth[finite].max():.4f}")
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.legend(loc="upper right", fontsize=9)

    # Panel 2: CS(2-1) mom-0 only
    ax = axes[1]
    finite_cs = np.isfinite(cs_smooth) & pb_mask
    vmax_cs = float(np.nanpercentile(cs_smooth[finite_cs], 99.5))
    vmin_cs = float(np.nanpercentile(cs_smooth[finite_cs], 1))
    im = ax.imshow(cs_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin_cs, vmax=vmax_cs)
    plt.colorbar(im, ax=ax, label="CS(2-1) mom-0 (Jy/beam · km/s)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.plot(cx, cy, "+", color="lime", markersize=18, markeredgewidth=2.5)
    ax.set_title(f"He 2-10 — CS(2-1) mom-0 (W={W_CS_KMS:.0f} km/s)\n"
                 f"σ={sigma_cs:.4f}, peak={cs_smooth[finite_cs].max():.4f}")
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")

    # Panel 3: HRL mom-0 with CS contours overlaid
    ax = axes[2]
    im = ax.imshow(hrl_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="H40α mom-0 (Jy/beam · km/s)")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    # CS contours at 3σ, 5σ, 8σ (the molecular gas shape)
    cs_levels = np.array([3, 5, 8, 13]) * sigma_cs
    cs_contour = ax.contour(cs_smooth, levels=cs_levels,
                             colors=["yellow", "orange", "red", "magenta"],
                             linewidths=1.5)
    # HRL contours at 3σ, 5σ
    hrl_levels = np.array([3, 5, 7]) * sigma_hrl
    hrl_contour = ax.contour(hrl_smooth, levels=hrl_levels,
                              colors=["cyan", "white", "lime"],
                              linewidths=1.2, linestyles="--")

    ax.plot(cx, cy, "+", color="lime", markersize=18, markeredgewidth=2.5)

    # legend
    ax.plot([], [], color="yellow", lw=1.5, label="CS(2-1) 3σ")
    ax.plot([], [], color="orange", lw=1.5, label="CS(2-1) 5σ")
    ax.plot([], [], color="red", lw=1.5, label="CS(2-1) 8σ")
    ax.plot([], [], color="magenta", lw=1.5, label="CS(2-1) 13σ")
    ax.plot([], [], color="cyan", lw=1.2, ls="--", label="H40α 3σ")
    ax.plot([], [], color="white", lw=1.2, ls="--", label="H40α 5σ")
    ax.plot([], [], color="lime", lw=1.2, ls="--", label="H40α 7σ")
    ax.set_title(
        "Overlay: H40α (dashed) + CS(2-1) contours (solid)\n"
        "→ HRL peak SHIFTED relative to CS ring structure"
    )
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    plt.suptitle(
        "He 2-10 — molecular vs ionised gas geometry\n"
        "(Johnson+ 2018, Imara & Faesi 2019: SSC feedback creates CO cavity; "
        "HRL fills the cavity → CO mask cannot trace HRL)",
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")
    # also copy to results/He2-10/plots/ for thesis access
    thesis_path = Path("/Volumes/HouAstro/master/results/He2-10/plots/he2-10_co_hrl_overlay.png")
    thesis_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(OUT, thesis_path)
    print(f"copied to {thesis_path}")
    bundle.data = None  # type: ignore


if __name__ == "__main__":
    main()
