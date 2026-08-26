"""
Apply ellipse-scale CoG (+ baseline subtraction) to 5 non-detections.

Goal: verify the method's behavior on cubes WITHOUT a real source.
Expectations:
  - seed mask at 3σ should give noise blob OR small false-positive
  - ellipse fit should be poorly constrained (random PA)
  - F(scale) should stay near 0 ± noise (no clear peak, no clear decline)
  - or if there IS a real signal, give a sensible F → useful detection
    threshold calibration

Targets:
  - IC 5063   (S/N 2.2 confirmed non-det)
  - NGC 1386  (S/N 2.5)
  - NGC 7130  (S/N 1.9)
  - NGC 7793  (S/N 2.6, smooth disk caveat case)
  - IRAS F18293-3413 (S/N 3.7, MARGINAL — boundary case)

Output:
  - per-galaxy 3-curve plot (F_obs / F_residual / F_clean ± 1σ)
  - 5-panel composite
  - cross-reference: clean detection peaks vs non-det peaks for threshold
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
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


# ---- non-detection configs ----
NONDETECTION_GALAXIES = [
    GalaxyConfig(
        name="IC5063",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/IC5063/IC5063_H30a_pbcor.fits",
        z=0.011348, line="H30a",
        W_kms=165.0, sf_scan=16,
        signal_ra=313.00982, signal_dec=-57.06876,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=313.00417, noise_dec=-57.06672,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC1386",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC1386/NGC1386_H30a_pbcor.fits",
        z=0.003052, line="H30a",
        W_kms=420.0, sf_scan=7,
        signal_ra=54.19292, signal_dec=-35.99944,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=54.18942, noise_dec=-35.99923,
        noise_size_arcsec=12.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC7130",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7130/NGC7130_H30a_pbcor.fits",
        z=0.016151, line="H30a",
        W_kms=105.0, sf_scan=3,
        signal_ra=327.08137, signal_dec=-34.95125,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=327.07629, noise_dec=-34.95378,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC7793",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC7793/NGC7793_H30a_pbcor.fits",
        z=0.000749, line="H30a",
        W_kms=300.0, sf_scan=3,            # no phase1 → reasonable defaults
        signal_ra=359.457210, signal_dec=-32.590990,
        signal_size_arcsec=45.0, signal_dec_size_arcsec=None,
        noise_ra=359.437428, noise_dec=-32.588212,
        noise_size_arcsec=20.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="IRASF18293-3413",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/IRASF18293-3413/IRASF18293-3413_H30a_pbcor.fits",
        z=0.018176, line="H30a",
        W_kms=480.0, sf_scan=5,
        signal_ra=278.17134, signal_dec=-34.19094,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=278.173855, noise_dec=-34.190947,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=None,
    ),
]


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

    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    eps = compute_eps_map(cube, line_free)
    sigma_eps = float(np.nanstd(eps[np.isfinite(eps)]))

    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    peak_in_sig = float(np.nanmax(mom0_smooth[sig_box]))
    sthresh_peak = peak_in_sig / max(sigma_smooth, 1e-9)
    print(f"  σ_smooth={sigma_smooth:.4f}, peak={peak_in_sig:.4f}, "
          f"peak/σ={sthresh_peak:.1f}σ, ε σ={sigma_eps:.1e}")

    # seed at 3σ with connected component
    raw_mask = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pixels = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][bidx], sig_pixels[1][bidx]
    seed = keep_component_containing(raw_mask, bx, by)
    fallback = None
    if not seed.any():
        for try_st in [2.5, 2.0, 1.5]:
            raw_mask = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw_mask, bx, by)
            if seed.any():
                fallback = try_st
                break
    if not seed.any():
        print(f"  WARN: no seed even at 1.5σ, skipping")
        bundle.data = None  # type: ignore
        return None
    print(f"  seed Npix={int(seed.sum())} = {seed.sum()/ppb:.2f} beams"
          + (f" (fallback {fallback}σ)" if fallback else ""))

    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    fit = fit_ellipse_weighted((ys, xs), weights)
    if fit is None:
        bundle.data = None  # type: ignore
        return None
    xc, yc, a, b, PA = fit
    print(f"  ellipse: center=({xc:.1f},{yc:.1f}), a={a:.2f}, b={b:.2f}, "
          f"PA={np.degrees(PA):.0f}°, b/a={b/max(a,1e-9):.2f}")

    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    masks = [ellipse_pixel_mask((ny, nx), xc, yc, a, b, PA, s)
             for s in scale_arr]
    F_obs = np.array([float(np.nansum(mom0[m]) / ppb) for m in masks])
    F_res, sigma_F_res = predict_F_residual(
        eps_map=eps,
        masks=masks,
        W_kms=cfg.W_kms,
        pix_per_beam=ppb,
    )
    F_clean = F_obs - F_res

    print(f"  F_obs:   peak {F_obs.max():+.3f} at scale "
          f"{scale_arr[np.argmax(F_obs)]:.2f}, "
          f"min {F_obs.min():+.3f}")
    print(f"  F_clean: peak {F_clean.max():+.3f} at scale "
          f"{scale_arr[np.argmax(F_clean)]:.2f}")
    print(f"  F_clean/σ_F_res at peak: "
          f"{F_clean.max()/max(sigma_F_res[np.argmax(F_clean)],1e-9):.2f}")

    # plot per-galaxy: 3 curves + before/after panels
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    ax.plot(scale_arr, F_obs, "o-", color="C3", label="F_obs", markersize=4)
    ax.plot(scale_arr, F_res, "s-", color="C2", label="F_residual",
            markersize=4)
    ax.fill_between(scale_arr, F_res - sigma_F_res, F_res + sigma_F_res,
                    color="C2", alpha=0.20, label="±1σ_F_res")
    ax.plot(scale_arr, F_clean, "^-", color="C0", label="F_clean",
            markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("ellipse scale")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — 3 curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(scale_arr, F_obs, "o-", color="C3",
            label=f"BEFORE  peak {F_obs.max():+.3f}", markersize=4)
    ax.plot(scale_arr, F_clean, "^-", color="C0",
            label=f"AFTER   peak {F_clean.max():+.3f}", markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("ellipse scale")
    ax.set_ylabel("F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — before vs after baseline subtraction")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"{cfg.name} [NON-DETECTION]  line={cfg.line}  W={cfg.W_kms:.0f} km/s  "
        f"sf={cfg.sf_scan}  ε σ={sigma_eps:.1e}",
        fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "ellipse_cog_nondet.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "scale": scale_arr,
        "F_obs": F_obs,
        "F_res": F_res,
        "sigma_F_res": sigma_F_res,
        "F_clean": F_clean,
        "F_obs_peak": float(F_obs.max()),
        "F_clean_peak": float(F_clean.max()),
        "scale_obs_peak": float(scale_arr[np.argmax(F_obs)]),
        "scale_clean_peak": float(scale_arr[np.argmax(F_clean)]),
        "sigma_F_res_at_peak": float(sigma_F_res[np.argmax(F_clean)]),
        "seed_Nbeam": float(seed.sum() / ppb),
        "ellipse_axes": (a, b),
        "ellipse_PA_deg": float(np.degrees(PA)),
        "ellipse_axis_ratio": float(b / max(a, 1e-9)),
    }


def main():
    results = []
    for g in NONDETECTION_GALAXIES:
        try:
            r = run_galaxy(g)
        except Exception as e:
            print(f"  ERROR on {g.name}: {e}")
            r = None
        if r is not None:
            results.append(r)

    # 5-panel composite
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["scale"], r["F_obs"], "o-", color="C3",
                label=f"BEFORE peak {r['F_obs_peak']:+.3f}",
                markersize=3)
        ax.plot(r["scale"], r["F_clean"], "^-", color="C0",
                label=f"AFTER  peak {r['F_clean_peak']:+.3f}",
                markersize=3)
        ax.fill_between(r["scale"],
                        r["F_clean"] - r["sigma_F_res"],
                        r["F_clean"] + r["sigma_F_res"],
                        color="C0", alpha=0.15,
                        label="±1σ_F_res")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(1.0, color="gray", lw=0.5, ls=":")
        ax.set_xlabel("ellipse scale")
        ax.set_ylabel("F (Jy·km/s)")
        nsig = r["F_clean_peak"] / max(r["sigma_F_res_at_peak"], 1e-9)
        ax.set_title(
            f"{r['name']} [NON-DET]\n"
            f"F_clean peak = {r['F_clean_peak']:+.3f} = "
            f"{nsig:.2f}σ_F_res, b/a={r['ellipse_axis_ratio']:.2f}"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Ellipse CoG on 5 NON-DETECTIONS — F_clean should stay near 0\n"
        "(or, for IRAS F18293, a marginal-detection boundary case)",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_ellipse_cog_nondet_5panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 5-panel composite {out}")

    # ---- print summary table ----
    print("\n" + "=" * 90)
    print(f"{'galaxy':<20} {'F_obs_peak':>12} {'F_clean_peak':>14} "
          f"{'σ_F_res':>10} {'F_clean/σ':>10} {'seed Nbeam':>11}")
    print("-" * 90)
    for r in results:
        nsig = r["F_clean_peak"] / max(r["sigma_F_res_at_peak"], 1e-9)
        print(f"{r['name']:<20} {r['F_obs_peak']:>+12.3f} "
              f"{r['F_clean_peak']:>+14.3f} "
              f"{r['sigma_F_res_at_peak']:>10.3f} "
              f"{nsig:>10.2f} {r['seed_Nbeam']:>11.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
