"""
Matched-procedure null test for ellipse-scale CoG.

The bug in plain spatial-null: F_signal uses argmax(mom0_smooth in
signal box) as ellipse center, but F_null uses RANDOM positions. The
asymmetry causes F_signal to inherit a positive "look-elsewhere" bias
that F_null can't cancel.

Fix: at each of K=16 null positions, build a LOCAL BOX of the same size
as the real signal box, find argmax(mom0_smooth) WITHIN that local box,
use it as the null ellipse center. Now both signal and null procedures
are matched → F_null distribution properly samples the selection bias.

Expected:
  - Non-detection cubes: F_clean ≈ 0 (selection bias cancels)
  - Detection cubes: F_clean ≈ true source flux (real signal >> noise argmax)
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Ellipse, Rectangle
import cmasher  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube
from ellipse_cog_poc import (
    GalaxyConfig, GALAXIES,
    keep_component_containing, fit_ellipse_weighted, ellipse_pixel_mask,
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
    N_SCALES, SCALE_MIN, SCALE_MAX,
)
from ellipse_cog_nondetection_validation import NONDETECTION_GALAXIES
from known_lines import LINES_REST_HZ

K_NULLS = 16
RNG_SEED = 42
NULL_VIZ_SCALE = 2.0
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")


def load_pb_mask(cube_path: str, threshold: float = 0.5):
    """Load PB and return PB > threshold mask (2D).

    Tries in order:
      1. {base}_pb.fits / .fits.gz  (PB cube, threshold cut applied)
      2. *pbmask_0.20.fits in same dir (3D nonpbcor cube with NaN
         outside PB > 0.20 — return finite mask, ignores threshold arg)

    Use the central channel only — PB is nearly constant across channels,
    avoids OOM from nanmedian over big PB cubes."""
    # Strip trailing suffix variants
    base = cube_path
    for suffix in ["_pbcor.fits", "_contsub.fits", "_v1_contsub.fits",
                   "_spw1_v1_contsub.fits"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    # 1. standard PB files
    for cand in [f"{base}_pb.fits", f"{base}_pb.fits.gz"]:
        if Path(cand).exists():
            pb_bundle = load_cube(cand)
            pb = pb_bundle.data
            if pb.ndim == 3:
                pb = np.asarray(pb[pb.shape[0] // 2, :, :])
            pb_bundle.data = None  # type: ignore
            return (pb > threshold) & np.isfinite(pb)
    # 2. fallback: pbmask_0.20.fits (NaN outside PB > 0.20)
    # exclude convolved/derived variants (`*.conv_b*.fits`)
    cube_dir = Path(cube_path).parent
    candidates = [
        p for p in cube_dir.glob("*pbmask_0.*.fits")
        if ".conv_b" not in p.name
    ]
    candidates.sort()
    for cand in candidates:
        try:
            import astropy.io.fits as fits_io
            with fits_io.open(str(cand)) as hdul:
                d = hdul[0].data
                if d is None:
                    continue
                if d.ndim == 3:
                    d = d[d.shape[0] // 2]
                return np.isfinite(d)
        except Exception:
            continue
    return None


def generate_matched_null_centers(
    rng, K, sig_box, sig_box_size_pix_xy,
    max_a_pix, ny, nx, mom0_smooth,
    valid_mask=None, min_clearance_pix=5,
):
    """Pick K null centers. Each null = argmax(mom0_smooth) within a
    LOCAL BOX of size `sig_box_size_pix_xy` centered on a random
    position outside the real sig_box and inside valid_mask.

    Returns list of (null_box_center_x, null_box_center_y, null_ellipse_xc,
    null_ellipse_yc) tuples.
    """
    box_w_x, box_w_y = sig_box_size_pix_xy
    half_x, half_y = box_w_x // 2, box_w_y // 2
    margin = max(int(np.ceil(max_a_pix)), half_x, half_y) + min_clearance_pix
    y_lo, y_hi = margin, ny - margin
    x_lo, x_hi = margin, nx - margin
    if y_lo >= y_hi or x_lo >= x_hi:
        return None

    sig_y, sig_x = np.where(sig_box)
    sx_lo, sx_hi = sig_x.min(), sig_x.max()
    sy_lo, sy_hi = sig_y.min(), sig_y.max()

    # Require only the null CENTER to be inside PB > 0.5 (not the full box).
    # The local box can extend slightly outside PB — that's fine because we
    # only need pixels inside PB for the argmax (masked with -inf outside).
    valid_centers = valid_mask

    # Two-pass: first try non-overlapping nulls, then if not enough,
    # allow overlap (still non-overlapping with sig_box + inside valid mask).
    nulls = []
    for pass_strict in (True, False):
        if len(nulls) >= K:
            break
        tries = 0
        max_tries = K * 500 if pass_strict else K * 2000
        while len(nulls) < K and tries < max_tries:
            tries += 1
            y = int(rng.integers(y_lo, y_hi))
            x = int(rng.integers(x_lo, x_hi))
            # require: local box doesn't overlap sig_box
            bx0, bx1 = x - half_x, x + half_x
            by0, by1 = y - half_y, y + half_y
            if not (bx1 < sx_lo or bx0 > sx_hi or by1 < sy_lo or by0 > sy_hi):
                continue
            # require: center inside eroded valid mask
            if valid_centers is not None and not valid_centers[y, x]:
                continue
            # strict pass: avoid duplicates / overlaps with prior nulls
            if pass_strict:
                too_close = False
                for (xp, yp, _, _) in nulls:
                    if (x - xp) ** 2 + (y - yp) ** 2 < (max(half_x, half_y)) ** 2:
                        too_close = True
                        break
                if too_close:
                    continue
            else:
                # relaxed pass: just avoid exact duplicates
                if any((x, y) == (xp, yp) for (xp, yp, _, _) in nulls):
                    continue
            local = mom0_smooth[by0:by1, bx0:bx1]
            local_valid = local
            if valid_mask is not None:
                local_valid = local.copy()
                local_valid[~valid_mask[by0:by1, bx0:bx1]] = -np.inf
            amax = int(np.argmax(local_valid))
            ay, ax_local = np.unravel_index(amax, local.shape)
            null_xc = bx0 + int(ax_local)
            null_yc = by0 + int(ay)
            nulls.append((x, y, null_xc, null_yc))
    return nulls


def run_galaxy(cfg: GalaxyConfig, label: str = ""):
    print(f"\n=== {cfg.name} {label} ===")
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
    pix_arcsec = abs(hdr["CDELT1"]) * 3600

    sig_dec = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    noi_dec = cfg.noise_dec_size_arcsec or cfg.noise_size_arcsec
    sig_box, _ = make_box_mask(wcs2, ny, nx,
                               cfg.signal_ra, cfg.signal_dec,
                               cfg.signal_size_arcsec, sig_dec)
    noise_box, _ = make_box_mask(wcs2, ny, nx,
                                 cfg.noise_ra, cfg.noise_dec,
                                 cfg.noise_size_arcsec, noi_dec)
    sig_box_w_x = int(round(cfg.signal_size_arcsec / pix_arcsec))
    sig_box_w_y = int(round(sig_dec / pix_arcsec))

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))

    # signal ellipse (same procedure as before)
    raw = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by_sig, bx_sig = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw, bx_sig, by_sig)
    fallback = None
    if not seed.any():
        for try_st in [2.5, 2.0, 1.5]:
            raw = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw, bx_sig, by_sig)
            if seed.any():
                fallback = try_st
                break
    if not seed.any():
        print("  WARN: no seed, skipping")
        bundle.data = None  # type: ignore
        return None
    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    xc, yc, a, b, PA = fit_ellipse_weighted((ys, xs), weights)
    print(f"  signal ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°, seed Nbeam={seed.sum()/ppb:.2f}")

    # PB constraint
    pb_mask = load_pb_mask(cfg.cube_path, threshold=0.5)

    # MATCHED null centers: each null is argmax within a local box
    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    max_a_pix = a * SCALE_MAX
    rng = np.random.default_rng(RNG_SEED)
    nulls = generate_matched_null_centers(
        rng, K_NULLS, sig_box, (sig_box_w_x, sig_box_w_y),
        max_a_pix, ny, nx, mom0_smooth, valid_mask=pb_mask)
    if nulls is None or len(nulls) < K_NULLS:
        # fallback smaller scale
        max_a_pix = a * 3.0
        nulls = generate_matched_null_centers(
            rng, K_NULLS, sig_box, (sig_box_w_x, sig_box_w_y),
            max_a_pix, ny, nx, mom0_smooth, valid_mask=pb_mask)
    if not nulls:
        max_a_pix = a * 2.0
        nulls = generate_matched_null_centers(
            rng, K_NULLS, sig_box, (sig_box_w_x, sig_box_w_y),
            max_a_pix, ny, nx, mom0_smooth, valid_mask=pb_mask,
            min_clearance_pix=2)
    if not nulls:
        print("  WARN: no nulls; skipping")
        bundle.data = None  # type: ignore
        return None
    print(f"  placed {len(nulls)} matched-procedure nulls (max_a_pix={max_a_pix:.0f}, "
          f"local box={sig_box_w_x}×{sig_box_w_y} pix)")

    # F_signal and F_null per scale (each null uses its OWN argmax as center)
    F_sig = np.zeros(N_SCALES)
    F_nulls = np.zeros((len(nulls), N_SCALES))
    for i, s in enumerate(scale_arr):
        m_sig = ellipse_pixel_mask((ny, nx), xc, yc, a, b, PA, s)
        F_sig[i] = float(np.nansum(mom0[m_sig]) / ppb)
        for k, (_, _, nxc, nyc) in enumerate(nulls):
            m_null = ellipse_pixel_mask((ny, nx), nxc, nyc, a, b, PA, s)
            F_nulls[k, i] = float(np.nansum(mom0[m_null]) / ppb)
    F_null_mean = F_nulls.mean(axis=0)
    F_null_std = F_nulls.std(axis=0)
    F_clean = F_sig - F_null_mean

    F_clean_peak_idx = int(np.argmax(F_clean))
    F_sig_peak_idx = int(np.argmax(F_sig))
    nsig_at_peak = F_clean[F_clean_peak_idx] / max(F_null_std[F_clean_peak_idx], 1e-9)
    print(f"  F_signal peak: {F_sig.max():+.3f} at scale {scale_arr[F_sig_peak_idx]:.2f}")
    print(f"  F_null_mean (matched): [{F_null_mean.min():+.3f}, {F_null_mean.max():+.3f}]")
    print(f"  F_null_std:            [{F_null_std.min():+.3f}, {F_null_std.max():+.3f}]")
    print(f"  F_clean peak: {F_clean.max():+.3f} at scale {scale_arr[F_clean_peak_idx]:.2f}, "
          f"{nsig_at_peak:.2f}σ")

    # plot per-galaxy
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax = axes[0]
    for k in range(len(nulls)):
        ax.plot(scale_arr, F_nulls[k], "-", color="gray", alpha=0.25, lw=0.7)
    ax.plot([], [], "-", color="gray", alpha=0.6, lw=0.7,
            label=f"individual nulls (K={len(nulls)})")
    ax.plot(scale_arr, F_null_mean, "s-", color="C2",
            label="⟨F_null⟩ (matched)", markersize=4)
    ax.fill_between(scale_arr, F_null_mean - F_null_std,
                    F_null_mean + F_null_std,
                    color="C2", alpha=0.20, label="±std")
    ax.plot(scale_arr, F_sig, "o-", color="C3",
            label=f"F_signal (peak {F_sig.max():+.3f})", markersize=4)
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label=f"F_clean (peak {F_clean.max():+.3f}, {nsig_at_peak:.2f}σ)",
            markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} {label}\nMATCHED null procedure")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    finite = np.isfinite(mom0_smooth)
    vmax = float(np.nanpercentile(mom0_smooth[finite], 99))
    vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mom-0 smoothed")
    s_viz = NULL_VIZ_SCALE
    # signal ellipse + box
    ell_sig = Ellipse((xc, yc), 2 * a * s_viz, 2 * b * s_viz,
                      angle=np.degrees(PA),
                      fill=False, edgecolor="red", linewidth=2.5)
    ax.add_patch(ell_sig)
    ax.plot([], [], color="red", lw=2.5, label=f"signal (scale={s_viz})")
    sb_y, sb_x = np.where(sig_box)
    rect_sig = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                         sb_x.max() - sb_x.min() + 1,
                         sb_y.max() - sb_y.min() + 1,
                         fill=False, edgecolor="white", linestyle=":",
                         linewidth=1.0)
    ax.add_patch(rect_sig)
    # null boxes + ellipses
    for (bx_n, by_n, nxc, nyc) in nulls:
        nrect = Rectangle((bx_n - sig_box_w_x // 2 - 0.5,
                           by_n - sig_box_w_y // 2 - 0.5),
                          sig_box_w_x, sig_box_w_y,
                          fill=False, edgecolor="yellow", linestyle=":",
                          linewidth=0.6, alpha=0.5)
        ax.add_patch(nrect)
        ell_n = Ellipse((nxc, nyc), 2 * a * s_viz, 2 * b * s_viz,
                        angle=np.degrees(PA),
                        fill=False, edgecolor="yellow", linewidth=1.0,
                        alpha=0.8)
        ax.add_patch(ell_n)
    ax.plot([], [], color="yellow", lw=1.0,
            label=f"K={len(nulls)} matched nulls (argmax in local box)")
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(f"signal box + {len(nulls)} matched-null boxes")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(f"{cfg.name} {label}  line={cfg.line}  W={cfg.W_kms:.0f}",
                 fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "matched_null_test.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name, "label": label,
        "scale": scale_arr,
        "F_sig": F_sig, "F_null_mean": F_null_mean, "F_null_std": F_null_std,
        "F_clean": F_clean,
        "F_clean_peak": float(F_clean.max()),
        "F_sig_peak": float(F_sig.max()),
        "nsig_at_peak": float(nsig_at_peak),
        "scale_clean_peak": float(scale_arr[F_clean_peak_idx]),
        "n_nulls": len(nulls),
    }


def _save_result_pickle(r, name):
    import pickle
    p = OUT_ROOT / f".matched_null_{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(r, f)


def _load_result_pickle(name):
    import pickle
    p = OUT_ROOT / f".matched_null_{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def main():
    """If --galaxy NAME passed: run that one galaxy + pickle the result.
    Otherwise: re-invoke self per galaxy via subprocess (fresh memory),
    then aggregate the pickled results into the composite figure."""
    import argparse, subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxy", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    det_galaxies = [g for g in GALAXIES if g.name != "NGC3627"]  # OOM skip
    nondet_galaxies = [g for g in NONDETECTION_GALAXIES if g.name != "NGC7793"]
    all_jobs = [(g, "[DETECTION]") for g in det_galaxies] + \
               [(g, "[NON-DET]") for g in nondet_galaxies]

    if args.galaxy is not None:
        # SUB-INVOCATION: run one galaxy, pickle result, exit
        cfg = next((g for (g, _) in all_jobs if g.name == args.galaxy), None)
        if cfg is None:
            print(f"galaxy {args.galaxy} not in job list")
            sys.exit(1)
        r = run_galaxy(cfg, label=args.label)
        if r is not None:
            _save_result_pickle(r, args.galaxy)
        sys.exit(0)

    # TOP-LEVEL: subprocess per galaxy
    script_path = str(Path(__file__).resolve())
    for (cfg, label) in all_jobs:
        print(f"\n{'#'*60}\n# Subprocess for {cfg.name} {label}\n{'#'*60}")
        cmd = ["python", script_path,
               "--galaxy", cfg.name, "--label", label]
        proc = subprocess.run(cmd, capture_output=False)
        if proc.returncode != 0:
            print(f"  {cfg.name} subprocess failed (exit {proc.returncode})")

    # aggregate pickled results
    results_det = []
    results_nondet = []
    for (g, label) in all_jobs:
        r = _load_result_pickle(g.name)
        if r is None:
            continue
        if "NON" in label:
            results_nondet.append(r)
        else:
            results_det.append(r)


    # composite figure: 2 rows = det vs non-det
    n_det = len(results_det)
    n_nd = len(results_nondet)
    ncols = max(n_det, n_nd)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9))
    for ax, r in zip(axes[0], results_det):
        ax.plot(r["scale"], r["F_sig"], "o-", color="C3",
                label=f"F_sig peak {r['F_sig_peak']:+.2f}", markersize=3)
        ax.plot(r["scale"], r["F_null_mean"], "s-", color="C2",
                label="⟨F_null⟩", markersize=3)
        ax.fill_between(r["scale"], r["F_null_mean"] - r["F_null_std"],
                        r["F_null_mean"] + r["F_null_std"],
                        color="C2", alpha=0.15)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"F_clean {r['F_clean_peak']:+.2f} ({r['nsig_at_peak']:.1f}σ)",
                markersize=3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("scale"); ax.set_ylabel("F")
        ax.set_title(f"{r['name']} [DET]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[0][n_det:]:
        ax.axis("off")
    for ax, r in zip(axes[1], results_nondet):
        ax.plot(r["scale"], r["F_sig"], "o-", color="C3",
                label=f"F_sig peak {r['F_sig_peak']:+.2f}", markersize=3)
        ax.plot(r["scale"], r["F_null_mean"], "s-", color="C2",
                label="⟨F_null⟩", markersize=3)
        ax.fill_between(r["scale"], r["F_null_mean"] - r["F_null_std"],
                        r["F_null_mean"] + r["F_null_std"],
                        color="C2", alpha=0.15)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"F_clean {r['F_clean_peak']:+.2f} ({r['nsig_at_peak']:.1f}σ)",
                markersize=3)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("scale"); ax.set_ylabel("F")
        ax.set_title(f"{r['name']} [NON-DET]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[1][n_nd:]:
        ax.axis("off")
    plt.suptitle(
        "Matched-procedure null test — F_null uses argmax in local box (same as signal)\n"
        "Expected: F_clean ≈ truth for DET, F_clean ≈ 0 for NON-DET",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_matched_null_test_composite.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved composite {out}")

    # summary table
    print("\n" + "=" * 100)
    print(f"{'galaxy':<20} {'class':<10} {'F_sig_peak':>12} {'F_clean_peak':>13} "
          f"{'σ':>8} {'F_clean/σ':>10}")
    print("-" * 100)
    for r in results_det + results_nondet:
        cls = "DET" if "DET" in r["label"] and "NON" not in r["label"] else "NON-DET"
        print(f"{r['name']:<20} {cls:<10} {r['F_sig_peak']:>+12.3f} "
              f"{r['F_clean_peak']:>+13.3f} "
              f"{r['F_null_std'][np.argmax(r['F_clean'])]:>8.3f} "
              f"{r['nsig_at_peak']:>10.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
