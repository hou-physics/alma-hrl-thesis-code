"""
Ellipse-scale CoG with K=16 spatial-null subtraction.

For each galaxy:
  1. Fit ellipse at signal position (same as ellipse_cog_poc).
  2. Generate K=16 random "null" positions outside the signal box and
     inside the cube (with margin), where the SAME ellipse shape is
     translated to.
  3. At every scale, compute F at signal position + F at each null
     position. F_clean(scale) = F_signal - mean(F_null), with σ from
     std(F_null) / sqrt(K).

Output per galaxy:
  - F vs scale plot (F_signal red, F_null mean green, F_clean blue ±σ)
  - mom-0 image showing signal ellipse + 16 null ellipses at scale=2

Output composite: 6-panel F(scale) overview.
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
from known_lines import LINES_REST_HZ

OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")
K_NULLS = 16
NULL_VIZ_SCALE = 2.0   # which scale's ellipses to overlay on mom-0
RNG_SEED = 42


def generate_null_centers(rng, K, sig_box, max_a_pix, ny, nx,
                          min_clearance_pix=5, valid_mask=None):
    """Pick K random (x, y) centers outside `sig_box`, where an ellipse of
    semi-major `max_a_pix` fits inside the cube with `min_clearance_pix`.

    If `valid_mask` is provided (bool 2D), the ellipse FOOTPRINT must lie
    entirely inside the mask (typically PB > 0.5 region) — otherwise the
    null integrates over masked/zero pbcor pixels and F_null ≈ 0
    trivially, defeating the null-test purpose.
    """
    # Allowed pixel rectangle (margin so ellipse stays in cube)
    margin = int(np.ceil(max_a_pix)) + min_clearance_pix
    y_lo, y_hi = margin, ny - margin
    x_lo, x_hi = margin, nx - margin
    if y_lo >= y_hi or x_lo >= x_hi:
        return None  # cube too small for this ellipse scale

    sig_y, sig_x = np.where(sig_box)
    sx_lo, sx_hi = sig_x.min(), sig_x.max()
    sy_lo, sy_hi = sig_y.min(), sig_y.max()

    # If valid_mask given, pre-erode it by max_a_pix so any center inside
    # the eroded mask guarantees the ellipse footprint fits within valid_mask
    valid_centers = None
    if valid_mask is not None:
        from scipy.ndimage import binary_erosion
        struct_r = int(np.ceil(max_a_pix))
        struct = np.ones((2 * struct_r + 1, 2 * struct_r + 1), dtype=bool)
        # circular erosion structuring element (approx)
        yy, xx = np.indices(struct.shape)
        cy_s, cx_s = struct_r, struct_r
        struct = (yy - cy_s) ** 2 + (xx - cx_s) ** 2 <= struct_r ** 2
        valid_centers = binary_erosion(valid_mask, structure=struct)

    nulls = []
    tries = 0
    while len(nulls) < K and tries < K * 500:
        tries += 1
        y = int(rng.integers(y_lo, y_hi))
        x = int(rng.integers(x_lo, x_hi))
        # require: ellipse bbox (≤ max_a_pix radius) doesn't overlap sig_box
        if (x + max_a_pix >= sx_lo and x - max_a_pix <= sx_hi
                and y + max_a_pix >= sy_lo and y - max_a_pix <= sy_hi):
            continue
        # require: pixel is inside the eroded valid mask
        if valid_centers is not None and not valid_centers[y, x]:
            continue
        # avoid duplicates / near-duplicates
        too_close = False
        for (xp, yp) in nulls:
            if (x - xp) ** 2 + (y - yp) ** 2 < (2 * max_a_pix) ** 2:
                too_close = True
                break
        if not too_close:
            nulls.append((x, y))
    return nulls


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

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv

    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))

    # ---- seed + ellipse fit (same as PoC) ----
    raw = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw, bx, by)
    if not seed.any():
        for try_st in [2.5, 2.0]:
            raw = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw, bx, by)
            if seed.any():
                break
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°")

    # ---- generate K null positions ----
    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    max_a_pix = a * SCALE_MAX
    rng = np.random.default_rng(RNG_SEED)
    nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx)
    if nulls is None or len(nulls) < K_NULLS:
        if nulls is None:
            print(f"  WARN: cube too small for max ellipse — retry with SCALE_MAX=3")
            max_a_pix = a * 3.0
            nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx)
        if nulls is None or len(nulls) < K_NULLS:
            print(f"  WARN: could only place {len(nulls) if nulls else 0} nulls; using what we have")
            if not nulls:
                bundle.data = None  # type: ignore
                return None
    print(f"  placed {len(nulls)} null positions (max_a={max_a_pix:.0f} pix)")

    # ---- F_signal and F_null per scale ----
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
    # σ on the difference: includes random-null + sigma_mean reduction
    sigma_F_clean = F_null_std  # std across realizations; F_signal noise comparable

    F_sig_peak_idx = int(np.argmax(F_sig))
    F_clean_peak_idx = int(np.argmax(F_clean))
    print(f"  F_signal: peak {F_sig.max():+.3f} at scale "
          f"{scale_arr[F_sig_peak_idx]:.2f}")
    print(f"  F_null_mean: [{F_null_mean.min():+.3f}, {F_null_mean.max():+.3f}]")
    print(f"  F_null_std:  [{F_null_std.min():+.3f}, {F_null_std.max():+.3f}]")
    print(f"  F_clean: peak {F_clean.max():+.3f} at scale "
          f"{scale_arr[F_clean_peak_idx]:.2f}")
    print(f"  F_clean / σ_null at peak: "
          f"{F_clean.max()/max(F_null_std[F_clean_peak_idx],1e-9):.2f}")

    # ---- plot per-galaxy: 2 panels (CoG curves + mom-0 overlay) ----
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # LEFT: CoG curves
    ax = axes[0]
    # plot all K null curves in faint grey
    for k in range(len(nulls)):
        ax.plot(scale_arr, F_nulls[k], "-", color="gray", alpha=0.25,
                linewidth=0.7)
    ax.plot([], [], "-", color="gray", alpha=0.6, linewidth=0.7,
            label=f"individual nulls (K={len(nulls)})")
    ax.plot(scale_arr, F_null_mean, "s-", color="C2",
            label=f"⟨F_null⟩ (mean)", markersize=4)
    ax.fill_between(scale_arr, F_null_mean - F_null_std,
                    F_null_mean + F_null_std,
                    color="C2", alpha=0.20, label="±std(F_null)")
    ax.plot(scale_arr, F_sig, "o-", color="C3",
            label=f"F_signal (peak {F_sig.max():+.2f})", markersize=4)
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label=f"F_clean = F_sig − ⟨F_null⟩ (peak {F_clean.max():+.2f})",
            markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — F vs scale (signal vs K={len(nulls)} nulls)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # RIGHT: mom-0 with signal + null ellipses at scale=NULL_VIZ_SCALE
    ax = axes[1]
    finite_mask = np.isfinite(mom0_smooth)
    if finite_mask.any():
        vmax = float(np.nanpercentile(mom0_smooth[finite_mask], 99))
        vmin = float(np.nanpercentile(mom0_smooth[finite_mask], 1))
    else:
        vmax = float(np.nanmax(mom0_smooth)); vmin = -vmax / 5
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")

    # signal ellipse at NULL_VIZ_SCALE
    s_viz = NULL_VIZ_SCALE
    ell_sig = Ellipse((xc, yc), 2 * a * s_viz, 2 * b * s_viz,
                      angle=np.degrees(PA),
                      fill=False, edgecolor="red", linewidth=2.5)
    ax.add_patch(ell_sig)
    ax.plot([], [], color="red", lw=2.5, label=f"signal (scale={s_viz})")

    # 16 null ellipses at same scale
    for (xn, yn) in nulls:
        ell_n = Ellipse((xn, yn), 2 * a * s_viz, 2 * b * s_viz,
                        angle=np.degrees(PA),
                        fill=False, edgecolor="yellow", linewidth=1.0,
                        alpha=0.7)
        ax.add_patch(ell_n)
    ax.plot([], [], color="yellow", lw=1.0, alpha=0.7,
            label=f"K={len(nulls)} nulls (same shape, random positions)")

    # signal box outline
    sb_y, sb_x = np.where(sig_box)
    rect = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                     sb_x.max() - sb_x.min() + 1,
                     sb_y.max() - sb_y.min() + 1,
                     fill=False, edgecolor="white", linestyle=":",
                     linewidth=1.0, label="signal box")
    ax.add_patch(rect)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(
        f"mom-0 + signal (red) + K={len(nulls)} null ellipses (yellow), "
        f"all at scale={s_viz}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"{cfg.name}  line={cfg.line}  W={cfg.W_kms:.0f} km/s  sf={cfg.sf_scan}  "
        f"K={K_NULLS} spatial nulls",
        fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "ellipse_cog_spatial_null.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "scale": scale_arr,
        "F_signal": F_sig,
        "F_null_mean": F_null_mean,
        "F_null_std": F_null_std,
        "F_clean": F_clean,
        "F_sig_peak": float(F_sig.max()),
        "F_clean_peak": float(F_clean.max()),
        "scale_sig_peak": float(scale_arr[F_sig_peak_idx]),
        "scale_clean_peak": float(scale_arr[F_clean_peak_idx]),
        "n_nulls": len(nulls),
    }


def main():
    results = []
    for g in GALAXIES:
        if g.name == "NGC3627":   # OOM skip
            print(f"\n=== {g.name} SKIPPED (cube too large) ===")
            continue
        try:
            r = run_galaxy(g)
        except Exception as e:
            print(f"  ERROR on {g.name}: {e}")
            r = None
        if r is not None:
            results.append(r)

    # 6-panel composite
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["scale"], r["F_signal"], "o-", color="C3",
                label=f"F_sig peak {r['F_sig_peak']:+.2f}", markersize=3)
        ax.plot(r["scale"], r["F_null_mean"], "s-", color="C2",
                label="⟨F_null⟩", markersize=3)
        ax.fill_between(r["scale"],
                        r["F_null_mean"] - r["F_null_std"],
                        r["F_null_mean"] + r["F_null_std"],
                        color="C2", alpha=0.15)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"F_clean peak {r['F_clean_peak']:+.2f}", markersize=3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("ellipse scale")
        ax.set_ylabel("F (Jy·km/s)")
        ax.set_title(
            f"{r['name']} ({r['line']}) — K={r['n_nulls']} nulls"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Ellipse CoG with K=16 spatial null subtraction\n"
        "F_clean = F_signal − ⟨F at 16 random ellipse positions⟩",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_ellipse_cog_spatial_null_6panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 6-panel composite {out}")


if __name__ == "__main__":
    main()
