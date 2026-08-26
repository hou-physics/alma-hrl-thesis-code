"""
Circinus-only spatial null with SCALE_MAX=3 (cube too small for default 5).
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
from cube_io import load_cube
from ellipse_cog_poc import (
    keep_component_containing, fit_ellipse_weighted, ellipse_pixel_mask,
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
)
from ellipse_cog_spatial_null import generate_null_centers
from known_lines import LINES_REST_HZ

N_SCALES = 30
SCALE_MIN = 0.5
SCALE_MAX = 3.0
K_NULLS = 16
NULL_VIZ_SCALE = 1.5
RNG_SEED = 42

CUBE = "/Volumes/HouAstro/master/master_thesis/work_dir/ESO097-013/ESO097-013_H40a_pbcor.fits"
z = 0.001448
W = 165.0
sf = 7
SIG_RA, SIG_DEC, SIG_SZ = 213.29152, -65.33909, 35.0
NOI_RA, NOI_DEC, NOI_SZ = 213.282197, -65.335201, 7.0
OUT = Path("/Volumes/HouAstro/master/result_v2/Circinus/ellipse_cog_spatial_null_smax3.png")
OUT_TMP = Path("/Volumes/HouAstro/master/result_v2/tmp/Circinus__spatial_null_smax3.png")


def main():
    print("loading Circinus cube...")
    bundle = load_cube(CUBE)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ["H40a"] / (1.0 + z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial
    sig_box, _ = make_box_mask(wcs2, ny, nx, SIG_RA, SIG_DEC, SIG_SZ, SIG_SZ)
    noise_box, _ = make_box_mask(wcs2, ny, nx, NOI_RA, NOI_DEC, NOI_SZ, NOI_SZ)

    in_line = np.abs(v_chan) <= W / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    mom0_smooth = smooth_fft(mom0, sf * beam_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    raw = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw, bx, by)
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, PA={np.degrees(PA):.0f}°")

    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    max_a_pix = a * SCALE_MAX
    print(f"  max_a={max_a_pix:.0f} pix; cube {nx}×{ny}")
    rng = np.random.default_rng(RNG_SEED)
    nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx,
                                  min_clearance_pix=2)
    if not nulls:
        # retry with even smaller scale
        max_a_pix = a * 2.0
        nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx,
                                      min_clearance_pix=2)
        print(f"  fell back to scale_max=2 → {len(nulls)} nulls")
    print(f"  placed {len(nulls)} null positions")

    F_sig = np.zeros(N_SCALES)
    F_nulls = np.zeros((len(nulls), N_SCALES))
    for i, s in enumerate(scale_arr):
        m_sig = ellipse_pixel_mask((ny, nx), xc, yc, a, b, PA, s)
        F_sig[i] = float(np.nansum(mom0[m_sig]) / ppb)
        for k, (xn, yn) in enumerate(nulls):
            m_null = ellipse_pixel_mask((ny, nx), xn, yn, a, b, PA, s)
            F_nulls[k, i] = float(np.nansum(mom0[m_null]) / ppb)
    F_null_mean = F_nulls.mean(axis=0)
    F_null_std = F_nulls.std(axis=0)
    F_clean = F_sig - F_null_mean

    print(f"  F_signal range: [{F_sig.min():+.3f}, {F_sig.max():+.3f}], "
          f"peak {F_sig.max():+.3f} at scale {scale_arr[np.argmax(F_sig)]:.2f}")
    print(f"  F_null_mean:   [{F_null_mean.min():+.3f}, {F_null_mean.max():+.3f}]")
    print(f"  F_null_std:    [{F_null_std.min():+.3f}, {F_null_std.max():+.3f}]")
    print(f"  F_clean range: [{F_clean.min():+.3f}, {F_clean.max():+.3f}], "
          f"peak {F_clean.max():+.3f} at scale {scale_arr[np.argmax(F_clean)]:.2f}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax = axes[0]
    for k in range(len(nulls)):
        ax.plot(scale_arr, F_nulls[k], "-", color="gray", alpha=0.25, lw=0.7)
    ax.plot([], [], "-", color="gray", alpha=0.6, lw=0.7,
            label=f"individual nulls (K={len(nulls)})")
    ax.plot(scale_arr, F_null_mean, "s-", color="C2",
            label="⟨F_null⟩", markersize=4)
    ax.fill_between(scale_arr, F_null_mean - F_null_std,
                    F_null_mean + F_null_std,
                    color="C2", alpha=0.20, label="±std(F_null)")
    ax.plot(scale_arr, F_sig, "o-", color="C3",
            label=f"F_signal (peak {F_sig.max():+.3f})", markersize=4)
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label=f"F_clean (peak {F_clean.max():+.3f})", markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"Circinus — F vs scale (signal vs K={len(nulls)} nulls, SCALE_MAX={SCALE_MAX})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    finite = np.isfinite(mom0_smooth)
    vmax = float(np.nanpercentile(mom0_smooth[finite], 99))
    vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")
    ell_sig = Ellipse((xc, yc), 2*a*NULL_VIZ_SCALE, 2*b*NULL_VIZ_SCALE,
                      angle=np.degrees(PA),
                      fill=False, edgecolor="red", linewidth=2.5)
    ax.add_patch(ell_sig)
    ax.plot([], [], color="red", lw=2.5, label=f"signal (scale={NULL_VIZ_SCALE})")
    for (xn, yn) in nulls:
        ell_n = Ellipse((xn, yn), 2*a*NULL_VIZ_SCALE, 2*b*NULL_VIZ_SCALE,
                        angle=np.degrees(PA),
                        fill=False, edgecolor="yellow", linewidth=1.0, alpha=0.7)
        ax.add_patch(ell_n)
    ax.plot([], [], color="yellow", lw=1.0, alpha=0.7,
            label=f"K={len(nulls)} nulls")
    sb_y, sb_x = np.where(sig_box)
    rect = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                     sb_x.max() - sb_x.min() + 1,
                     sb_y.max() - sb_y.min() + 1,
                     fill=False, edgecolor="white", linestyle=":",
                     linewidth=1.0, label="signal box (35″)")
    ax.add_patch(rect)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(f"Circinus mom-0 + signal (red) + {len(nulls)} nulls (yellow), scale={NULL_VIZ_SCALE}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(f"Circinus — spatial null with SCALE_MAX={SCALE_MAX}", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved {OUT}")
    import shutil
    shutil.copy(OUT, OUT_TMP)
    print(f"copied to {OUT_TMP}")
    bundle.data = None  # type: ignore


if __name__ == "__main__":
    main()
