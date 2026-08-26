"""
Audit: re-run ellipse + spatial null pipeline with photutils auto regions
on 6 clean detections, compare F_clean(auto) vs F_clean(manual).

If manual regions were OK → F_clean(auto) ≈ F_clean(manual).
If manual regions were bad (e.g. NGC 4945 noise box on source) → F_clean
will shift, telling us by how much.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import cmasher  # noqa: F401
from photutils.segmentation import detect_threshold, detect_sources
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube
from ellipse_cog_poc import (
    GalaxyConfig, GALAXIES,
    keep_component_containing, fit_ellipse_weighted, ellipse_pixel_mask,
    beam_fwhm_pix, pix_per_beam, smooth_fft, make_box_mask,
    channel_velocities,
    N_SCALES, SCALE_MIN, SCALE_MAX,
)
from ellipse_cog_spatial_null import generate_null_centers, K_NULLS, RNG_SEED
from known_lines import LINES_REST_HZ
from matched_null_test import load_pb_mask
from auto_regions_poc import (
    detect_signal_segments, select_target_segment, find_best_noise_box,
)

OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2/tmp")


def auto_select_regions(mom0, mom0_smooth, pb_mask, beam_fwhm, pix_arcsec,
                         wcs2, ned_ra, ned_dec, manual_noise_size_arcsec,
                         manual_sig_size_arcsec):
    """Auto signal_box + noise_box from HRL mom-0.

    Returns (sig_ra, sig_dec, sig_size_arcsec_xy,
             noise_ra, noise_dec, noise_size_arcsec,
             segm, target_label)
    or None on failure."""
    ny, nx = mom0.shape
    mom0_for_det = mom0_smooth.copy()
    if pb_mask is not None:
        mom0_for_det[~pb_mask] = np.nan

    segm, _ = detect_signal_segments(mom0_for_det, nsigma=3.0, npixels=10)
    if segm is None or segm.nlabels == 0:
        return None
    target_label, dist = select_target_segment(segm, wcs2, ned_ra, ned_dec,
                                                max_dist_arcsec=30)
    if target_label is None:
        return None

    target_mask = (segm.data == target_label)
    ys, xs = np.where(target_mask)
    sig_size_arcsec = (
        (xs.max() - xs.min() + 1) * pix_arcsec,
        (ys.max() - ys.min() + 1) * pix_arcsec,
    )
    sig_xc_pix = float(xs.mean()); sig_yc_pix = float(ys.mean())
    sig_world = wcs2.pixel_to_world_values(sig_xc_pix, sig_yc_pix)

    # auto noise box: exclusion = 3 × beam_FWHM around any detected source
    source_mask = (segm.data > 0)
    noise_box_pix = int(manual_noise_size_arcsec / pix_arcsec)
    exclusion_pix = int(3 * beam_fwhm)
    nbox = find_best_noise_box(mom0, pb_mask, source_mask, noise_box_pix,
                               min_dist_from_source_pix=exclusion_pix)
    if nbox is None:
        return None
    nxc, nyc, _, nscore, _, _ = nbox
    noise_world = wcs2.pixel_to_world_values(nxc, nyc)

    return {
        "sig_ra": float(sig_world[0]),
        "sig_dec": float(sig_world[1]),
        "sig_size_arcsec_x": sig_size_arcsec[0],
        "sig_size_arcsec_y": sig_size_arcsec[1],
        "noise_ra": float(noise_world[0]),
        "noise_dec": float(noise_world[1]),
        "noise_size_arcsec": manual_noise_size_arcsec,
        "noise_score": nscore,
        "n_segments": int(segm.nlabels),
        "target_label": target_label,
        "ned_dist_arcsec": dist * pix_arcsec,
    }


def run_pipeline_with_regions(cfg: GalaxyConfig, sig_ra, sig_dec,
                              sig_size_x, sig_size_y,
                              noise_ra, noise_dec, noise_size,
                              pb_mask=None, label=""):
    """Run ellipse + K=16 spatial null with given regions.
    Returns dict with F_clean stats."""
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

    sig_box, _ = make_box_mask(wcs2, ny, nx, sig_ra, sig_dec,
                               sig_size_x, sig_size_y)
    noise_box, _ = make_box_mask(wcs2, ny, nx, noise_ra, noise_dec,
                                 noise_size, noise_size)

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv

    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))

    # signal ellipse
    raw = (mom0_smooth > 3.0 * sigma_smooth) & sig_box
    sig_pix = np.where(sig_box)
    bidx = int(np.argmax(mom0_smooth[sig_pix]))
    by, bx = sig_pix[0][bidx], sig_pix[1][bidx]
    seed = keep_component_containing(raw, bx, by)
    if not seed.any():
        for try_st in [2.5, 2.0]:
            raw = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw, bx, by)
            if seed.any():
                break
    if not seed.any():
        bundle.data = None  # type: ignore
        return None

    ys, xs = np.where(seed)
    weights = np.maximum(mom0_smooth[ys, xs], 0)
    fit = fit_ellipse_weighted((ys, xs), weights)
    if fit is None:
        bundle.data = None  # type: ignore
        return None
    xc, yc, a, b, PA = fit

    # K=16 null — try several scale_max levels then no-PB as last resort
    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    nulls = None
    for try_scale_max in [SCALE_MAX, 3.0, 2.0, 1.5]:
        max_a_pix = a * try_scale_max
        nulls = generate_null_centers(rng=np.random.default_rng(RNG_SEED),
                                       K=K_NULLS, sig_box=sig_box,
                                       max_a_pix=max_a_pix,
                                       ny=ny, nx=nx, valid_mask=pb_mask)
        if nulls and len(nulls) >= K_NULLS / 2:
            break
    if not nulls:
        # last resort: drop PB constraint
        for try_scale_max in [3.0, 2.0]:
            max_a_pix = a * try_scale_max
            nulls = generate_null_centers(rng=np.random.default_rng(RNG_SEED),
                                           K=K_NULLS, sig_box=sig_box,
                                           max_a_pix=max_a_pix,
                                           ny=ny, nx=nx, valid_mask=None)
            if nulls:
                break
    if not nulls:
        bundle.data = None  # type: ignore
        return None

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

    bundle.data = None  # type: ignore
    return {
        "label": label,
        "F_sig_peak": float(F_sig.max()),
        "F_clean_peak": float(F_clean.max()),
        "scale_sig_peak": float(scale_arr[F_sig_peak_idx]),
        "scale_clean_peak": float(scale_arr[F_clean_peak_idx]),
        "F_null_std_at_peak": float(F_null_std[F_clean_peak_idx]),
        "sigma_smooth": sigma_smooth,
        "n_nulls": len(nulls),
        "ellipse_axes": (a, b),
        "ellipse_PA_deg": float(np.degrees(PA)),
        "seed_Nbeam": float(seed.sum() / ppb),
    }


def audit_galaxy(cfg: GalaxyConfig):
    """Run pipeline twice: with manual regions (from cfg) + with auto regions
    (from photutils on HRL mom-0). Compare F_clean."""
    print(f"\n=== {cfg.name} ===")

    # === MANUAL run ===
    print("  -- MANUAL regions --")
    pb_path = cfg.cube_path.replace("_pbcor.fits", "_pb.fits")
    if not Path(pb_path).exists():
        pb_path = pb_path + ".gz"
    pb_mask = load_pb_mask(pb_path) if Path(pb_path).exists() else None
    sig_dec = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    manual = run_pipeline_with_regions(
        cfg,
        cfg.signal_ra, cfg.signal_dec, cfg.signal_size_arcsec, sig_dec,
        cfg.noise_ra, cfg.noise_dec,
        cfg.noise_size_arcsec, pb_mask=pb_mask, label="manual"
    )
    if manual is None:
        print("  manual run failed")
        return None
    print(f"    σ_smooth={manual['sigma_smooth']:.4f}, "
          f"F_sig={manual['F_sig_peak']:+.3f}, F_clean={manual['F_clean_peak']:+.3f}, "
          f"a={manual['ellipse_axes'][0]:.1f}, b={manual['ellipse_axes'][1]:.1f}, "
          f"seed_Nbeam={manual['seed_Nbeam']:.1f}")

    # === AUTO regions ===
    print("  -- AUTO regions (photutils on HRL mom-0) --")
    bundle = load_cube(cfg.cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINES_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    beam_fwhm = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    mom0_smooth = smooth_fft(mom0, cfg.sf_scan * beam_fwhm)
    bundle.data = None  # type: ignore

    auto = auto_select_regions(
        mom0, mom0_smooth, pb_mask, beam_fwhm, pix_arcsec, wcs2,
        cfg.signal_ra, cfg.signal_dec,  # NED ≈ original signal center
        cfg.noise_size_arcsec, cfg.signal_size_arcsec,
    )
    if auto is None:
        print("  auto region selection failed")
        return {"name": cfg.name, "manual": manual, "auto": None}
    print(f"    detected {auto['n_segments']} segments, "
          f"target dist_NED={auto['ned_dist_arcsec']:.1f}\"")
    print(f"    AUTO sig box: ({auto['sig_ra']:.5f}, {auto['sig_dec']:.5f}) "
          f"size {auto['sig_size_arcsec_x']:.1f}\" × {auto['sig_size_arcsec_y']:.1f}\"")
    print(f"    AUTO noise box: ({auto['noise_ra']:.5f}, {auto['noise_dec']:.5f}) "
          f"score={auto['noise_score']:.4f}")

    auto_run = run_pipeline_with_regions(
        cfg,
        auto["sig_ra"], auto["sig_dec"],
        auto["sig_size_arcsec_x"], auto["sig_size_arcsec_y"],
        auto["noise_ra"], auto["noise_dec"],
        auto["noise_size_arcsec"], pb_mask=pb_mask, label="auto",
    )
    if auto_run is None:
        print("  auto run failed")
        return {"name": cfg.name, "manual": manual, "auto": None}
    print(f"    σ_smooth={auto_run['sigma_smooth']:.4f}, "
          f"F_sig={auto_run['F_sig_peak']:+.3f}, F_clean={auto_run['F_clean_peak']:+.3f}, "
          f"a={auto_run['ellipse_axes'][0]:.1f}, b={auto_run['ellipse_axes'][1]:.1f}, "
          f"seed_Nbeam={auto_run['seed_Nbeam']:.1f}")

    return {"name": cfg.name, "manual": manual, "auto": auto_run,
            "auto_regions": auto}


def main():
    # 6 clean detections (excl. NGC 3627 OOM)
    targets = [g for g in GALAXIES if g.name != "NGC3627"]

    results = []
    for cfg in targets:
        try:
            r = audit_galaxy(cfg)
        except Exception as e:
            print(f"  ERROR on {cfg.name}: {e}")
            import traceback; traceback.print_exc()
            r = None
        if r:
            results.append(r)

    print("\n" + "=" * 110)
    print(f"{'galaxy':<12} {'σ_manual':>10} {'σ_auto':>10} {'F_clean_manual':>17} "
          f"{'F_clean_auto':>15} {'Δ%':>8} {'lit':>10}")
    print("-" * 110)
    LIT = {"NGC5253": "Bendo 0.86", "NGC4945": "Toma 9.27",
           "NGC3628": "Toma ~1.0", "He2-10": "—",
           "Circinus": "Ph2 0.55", "NGC253": "Toma 6.39"}
    for r in results:
        if r["auto"] is None:
            print(f"{r['name']:<12} {r['manual']['sigma_smooth']:>10.4f} "
                  f"{'—':>10} {r['manual']['F_clean_peak']:>+17.3f} "
                  f"{'—':>15} {'—':>8} {LIT.get(r['name'],'?'):>10}")
            continue
        m = r["manual"]; a = r["auto"]
        delta_pct = 100.0 * (a["F_clean_peak"] - m["F_clean_peak"]) / max(abs(m["F_clean_peak"]), 1e-9)
        print(f"{r['name']:<12} {m['sigma_smooth']:>10.4f} {a['sigma_smooth']:>10.4f} "
              f"{m['F_clean_peak']:>+17.3f} {a['F_clean_peak']:>+15.3f} "
              f"{delta_pct:>+7.1f}% {LIT.get(r['name'],'?'):>10}")
    print("=" * 110)


if __name__ == "__main__":
    main()
