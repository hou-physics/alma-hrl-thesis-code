"""
Spatial null subtraction on 4 non-detections (excl. NGC 7793 — too
special, large disk).

Validate F_clean ≈ 0 on cubes with no real source. If it holds:
F_clean is production-ready as canonical flux + null-test detection
criterion.

Targets:
  - IC 5063
  - NGC 1386
  - NGC 7130
  - IRAS F18293-3413
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
    GalaxyConfig,
    keep_component_containing, fit_ellipse_weighted, ellipse_pixel_mask,
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
    N_SCALES, SCALE_MIN, SCALE_MAX,
)
from ellipse_cog_spatial_null import generate_null_centers, K_NULLS, RNG_SEED
from ellipse_cog_nondetection_validation import NONDETECTION_GALAXIES
from known_lines import LINES_REST_HZ

OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")
NULL_VIZ_SCALE = 2.0

# Filter: exclude NGC 7793 per user request
GALAXIES = [g for g in NONDETECTION_GALAXIES if g.name != "NGC7793"]


def load_pb_mask(cube_path: str, threshold: float = 0.5):
    """Load PB cube (if available), return 2D bool mask of PB > threshold."""
    base = cube_path.replace("_pbcor.fits", "")
    for cand in [f"{base}_pb.fits", f"{base}_pb.fits.gz"]:
        if Path(cand).exists():
            pb_bundle = load_cube(cand)
            pb = pb_bundle.data
            if pb.ndim == 3:
                pb = np.nanmedian(pb, axis=0)
            pb_bundle.data = None  # type: ignore
            return (pb > threshold) & np.isfinite(pb)
    return None


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

    raw = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw, bx, by)
    fallback_st = None
    if not seed.any():
        for try_st in [2.5, 2.0, 1.5]:
            raw = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw, bx, by)
            if seed.any():
                fallback_st = try_st
                break
    if not seed.any():
        print("  WARN: no seed even at 1.5σ, skipping")
        bundle.data = None  # type: ignore
        return None

    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)
    print(f"  seed Nbeam={seed.sum()/ppb:.2f} (fallback {fallback_st}σ)"
          if fallback_st else f"  seed Nbeam={seed.sum()/ppb:.2f}")
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°")

    # Load PB > 0.5 mask to constrain null positions
    pb_mask = load_pb_mask(cfg.cube_path, threshold=0.5)
    if pb_mask is not None:
        print(f"  PB > 0.5: {int(pb_mask.sum())} valid pix "
              f"({100*pb_mask.sum()/(ny*nx):.1f}% of cube)")
    else:
        print("  WARN: no PB file found — null positions UNconstrained")

    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    max_a_pix = a * SCALE_MAX
    rng = np.random.default_rng(RNG_SEED)
    nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx,
                                  valid_mask=pb_mask)
    if nulls is None or len(nulls) < K_NULLS:
        # try smaller max scale
        max_a_pix = a * 3.0
        nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx,
                                      valid_mask=pb_mask)
        if nulls:
            print(f"  fell back to scale_max=3 → {len(nulls)} nulls")
    if not nulls:
        # last fallback: scale_max=2 + smaller clearance
        max_a_pix = a * 2.0
        nulls = generate_null_centers(rng, K_NULLS, sig_box, max_a_pix, ny, nx,
                                      min_clearance_pix=2,
                                      valid_mask=pb_mask)
    if not nulls:
        print("  WARN: cannot place nulls even with relaxed scale, skipping")
        bundle.data = None  # type: ignore
        return None
    print(f"  placed {len(nulls)} nulls (max_a_pix={max_a_pix:.0f})")

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

    F_clean_peak_idx = int(np.argmax(F_clean))
    F_sig_peak_idx = int(np.argmax(F_sig))
    sig_at_peak = F_null_std[F_clean_peak_idx]
    nsig = F_clean[F_clean_peak_idx] / max(sig_at_peak, 1e-9)
    print(f"  F_signal peak: {F_sig.max():+.3f} at scale {scale_arr[F_sig_peak_idx]:.2f}")
    print(f"  F_null_mean:   [{F_null_mean.min():+.3f}, {F_null_mean.max():+.3f}]")
    print(f"  F_null_std:    [{F_null_std.min():+.3f}, {F_null_std.max():+.3f}]")
    print(f"  F_clean peak:  {F_clean[F_clean_peak_idx]:+.3f} at scale {scale_arr[F_clean_peak_idx]:.2f}")
    print(f"  F_clean / σ_null at peak: {nsig:.2f}")

    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # LEFT: CoG curves
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
    ax.set_title(
        f"{cfg.name} [NON-DET]\n"
        f"F_clean/σ_null = {nsig:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # RIGHT: mom-0 + signal + nulls at scale=NULL_VIZ_SCALE
    ax = axes[1]
    finite = np.isfinite(mom0_smooth)
    if finite.any():
        vmax = float(np.nanpercentile(mom0_smooth[finite], 99))
        vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    else:
        vmax = float(np.nanmax(mom0_smooth)); vmin = -vmax / 5
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed (Jy/beam · km/s)")

    s_viz = NULL_VIZ_SCALE
    ell_sig = Ellipse((xc, yc), 2 * a * s_viz, 2 * b * s_viz,
                      angle=np.degrees(PA),
                      fill=False, edgecolor="red", linewidth=2.5)
    ax.add_patch(ell_sig)
    ax.plot([], [], color="red", lw=2.5, label=f"signal (scale={s_viz})")
    for (xn, yn) in nulls:
        ell_n = Ellipse((xn, yn), 2 * a * s_viz, 2 * b * s_viz,
                        angle=np.degrees(PA),
                        fill=False, edgecolor="yellow", linewidth=1.0,
                        alpha=0.7)
        ax.add_patch(ell_n)
    ax.plot([], [], color="yellow", lw=1.0, alpha=0.7,
            label=f"K={len(nulls)} nulls")
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
        f"mom-0 + signal (red) + K={len(nulls)} nulls (yellow), scale={s_viz}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"{cfg.name} [NON-DETECTION]  line={cfg.line}  W={cfg.W_kms:.0f} km/s",
        fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "spatial_null_nondet.png"
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
        "F_clean_peak": float(F_clean.max()),
        "F_clean_at_peak_nsig": float(nsig),
        "sigma_at_peak": float(sig_at_peak),
        "scale_clean_peak": float(scale_arr[F_clean_peak_idx]),
        "n_nulls": len(nulls),
    }


def main():
    results = []
    for g in GALAXIES:
        try:
            r = run_galaxy(g)
        except Exception as e:
            print(f"  ERROR on {g.name}: {e}")
            r = None
        if r is not None:
            results.append(r)

    # composite
    n = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        for k in range(r["F_null_mean"].size):
            pass  # don't replot individual null curves in composite
        ax.plot(r["scale"], r["F_signal"], "o-", color="C3",
                label=f"F_sig (peak {r['F_signal'].max():+.3f})",
                markersize=3)
        ax.plot(r["scale"], r["F_null_mean"], "s-", color="C2",
                label="⟨F_null⟩", markersize=3)
        ax.fill_between(r["scale"],
                        r["F_null_mean"] - r["F_null_std"],
                        r["F_null_mean"] + r["F_null_std"],
                        color="C2", alpha=0.15)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"F_clean (peak {r['F_clean_peak']:+.3f}, "
                      f"{r['F_clean_at_peak_nsig']:.2f}σ)",
                markersize=3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("ellipse scale")
        ax.set_ylabel("F (Jy·km/s)")
        ax.set_title(
            f"{r['name']} [NON-DET] — K={r['n_nulls']} nulls"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Spatial null on 4 non-detections — F_clean should be ≈ 0\n"
        "(if F_clean / σ_null < 3 across all panels → method passes validation)",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_spatial_null_nondet_4panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 4-panel composite {out}")

    # summary table
    print("\n" + "=" * 92)
    print(f"{'galaxy':<22} {'F_sig_peak':>11} {'F_clean_peak':>13} "
          f"{'σ_null':>8} {'F_clean/σ':>10} {'verdict':>20}")
    print("-" * 92)
    for r in results:
        nsig = r["F_clean_at_peak_nsig"]
        verdict = "PASS (F_clean ≈ 0)" if nsig < 3 else "MARGINAL/DETECTION?"
        print(f"{r['name']:<22} {r['F_signal'].max():>+11.3f} "
              f"{r['F_clean_peak']:>+13.3f} "
              f"{r['sigma_at_peak']:>8.3f} {nsig:>10.2f} {verdict:>20}")
    print("=" * 92)


if __name__ == "__main__":
    main()
