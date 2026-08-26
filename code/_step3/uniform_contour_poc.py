"""Uniform contour-prescription PoC v2 (mainline B, decisions.md 2026-08-01 PM).

One fixed prescription for every source:
  mask   = strong-tracer mom-0 2σ contour, union of islands ≥ 1 effective beam
           (smoothing variant s ∈ {0, 1, 3} × beam FWHM — v2 adds this axis
            after v1 no-smoothing fragmented NGC 5253 deep native 0.19″)
  window = strong-tracer line width (W50 vs W20 — PoC decides)
  flux   = HRL pbcor integrated over mask × window,
           minus mean of ≤ K=16 spatial-null translations (may overlap each
           other; only the source ±1 beam is excluded — v1 disjoint placement
           was geometrically impossible for FoV-filling union masks)
  noise  = empirical integrated-spectrum RMS × sqrt(n_chan) × Δv

Extra diagnostic: "skirt" spectrum (mask minus >3σ core) to attribute
low-surface-brightness flux (real line vs baseline residual).

Controls: NGC 4945 / NGC 3628 / NGC 253 / NGC 5253 deep-native.
Gate: flux within literature tolerance (> 25% drift = prescription fails).

Run: conda run -n casa_env --no-capture-output python _step3/uniform_contour_poc.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube, get_beam_area_pix  # noqa: E402
from flux import calc_flux_sn                      # noqa: E402
from known_lines import LINES_REST_HZ, line_free_mask_for_cube  # noqa: E402

C_KMS = 299792.458
WORK = "/Volumes/HouAstro/master/master_thesis/work_dir"
OUT_DIR = Path("/Volumes/HouAstro/master/result_v2/_uniform_poc")
SIGMA_CONTOUR = 2.0
SEED_SIGMA = 5.0        # v3: island survives only if its peak ≥ 5σ (two-level)
W_FLOOR_KMS = 100.0     # v3: window floor — HRL thermal width in dwarfs > cold CO
W20_MEASURE_RES_KMS = 20.0   # v3.1: common velocity resolution for width measurement
SMOOTH_VARIANTS = [0.0, 1.0, 3.0]   # Gaussian kernel FWHM in units of beam FWHM
K_NULLS = 16
RNG_SEED = 42
MASK_FRAC_RED_FLAG = 0.3

REST_HZ = {
    "H30a": LINES_REST_HZ["H30a"], "H40a": LINES_REST_HZ["H40a"],
    "CO21": LINES_REST_HZ["CO2-1"], "CS21": LINES_REST_HZ["CS2-1"],
}

CONFIGS = [
    dict(name="NGC4945", z=0.00188, hrl_line="H30a", strong_line="CO21",
         hrl_path=f"{WORK}/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits",
         strong_path=f"{WORK}/NGC4945/NGC4945_CO_nonpbcor.fits",
         pb_mult=None, strong_win_kms=460.0,
         lit_ref="Toma 2024", lit_flux=9.27, prior_flux=8.839),
    dict(name="NGC3628", z=0.002772, hrl_line="H30a", strong_line="CO21",
         hrl_path=f"{WORK}/NGC3628/NGC3628_H30a_pbcor.fits",
         strong_path=f"{WORK}/NGC3628/NGC3628_CO21_nonpbcor.fits",
         pb_mult=None, strong_win_kms=460.0,
         lit_ref="Toma 2024", lit_flux=1.0, prior_flux=0.982),
    dict(name="NGC253", z=0.00080, hrl_line="H40a", strong_line="CS21",
         hrl_path=f"{WORK}/NGC253/NGC253_H40a_pbcor.fits",
         strong_path=f"{WORK}/NGC253/NGC253_H40a_pbcor.fits",
         pb_mult=f"{WORK}/NGC253/NGC253_spw31_pb_mfs.fits",
         strong_win_kms=300.0,
         lit_ref="pipeline 2026-04", lit_flux=4.695, prior_flux=4.695),
    dict(name="NGC5253deep", z=0.0014, hrl_line="H30a", strong_line="CO21",
         hrl_path=f"{WORK}/NGC5253_deep/NGC5253_H30a_pbcor.fits",
         strong_path=f"{WORK}/NGC5253_deep/NGC5253_CO21_nonpbcor.fits",
         pb_mult=None, strong_win_kms=300.0,
         lit_ref="Bendo 2017", lit_flux=0.86, prior_flux=0.699),
    # survey-grade tier (0.91″, beam_pc 15.6 — inside the sample's validity
    # domain, unlike deep-native 0.19″/3.3 pc); prior = X1c scan result
    dict(name="NGC5253x1c", z=0.0014, hrl_line="H30a", strong_line="CO21",
         hrl_path=f"{WORK}/NGC5253/NGC5253_H30a_pbcor.fits",
         strong_path=f"{WORK}/NGC5253/NGC5253_CO21_nonpbcor.fits",
         pb_mult=None, strong_win_kms=300.0,
         lit_ref="Bendo 2017", lit_flux=0.86, prior_flux=0.344),
]


def channel_velocities(hdr, rest_obs_hz):
    n = hdr["NAXIS3"]
    freqs = hdr["CRVAL3"] + (np.arange(n) - (hdr["CRPIX3"] - 1)) * hdr["CDELT3"]
    return (rest_obs_hz - freqs) / rest_obs_hz * C_KMS, freqs


def multi_mask_spectra(cube, masks, beam_area, ch_lo, ch_hi):
    """One pass over channels [ch_lo, ch_hi]; flat-index gather per mask so
    CPU cost scales with masked pixels, not map size × n_masks."""
    nchan = cube.shape[0]
    idx_list = [np.flatnonzero(m.ravel()) for m in masks]
    specs = np.full((len(masks), nchan), np.nan)
    for ch in range(ch_lo, ch_hi + 1):
        flat = np.asarray(cube[ch], dtype=np.float64).ravel()
        for i, idx in enumerate(idx_list):
            specs[i, ch] = np.nansum(flat[idx]) / beam_area
    return specs


def smooth_spectrum_kms(spec, dv_kms, fwhm_kms=W20_MEASURE_RES_KMS):
    """v3.1: smooth a spectrum to a common ~20 km/s velocity resolution BEFORE
    width measurement. Kills noise-bump threshold crossings on faint tracers
    (He 2-10 CS incident: 20% threshold at noise level → W20 258 vs true ~120)
    and removes the hidden dv-dependence of W20 across heterogeneous cubes
    (sample dv spans 1.3–10 km/s). Outermost-crossing definition unchanged —
    it is required for double-horn profiles whose mid-valley dips below the
    20% threshold (NGC 4945)."""
    if fwhm_kms <= dv_kms:
        return spec
    from astropy.convolution import Gaussian1DKernel, convolve
    sigma_ch = fwhm_kms / dv_kms / (2 * np.sqrt(2 * np.log(2)))
    return convolve(spec, Gaussian1DKernel(sigma_ch), boundary="extend",
                    nan_treatment="interpolate", preserve_nan=True)


def line_width(v, spec, win_kms, frac):
    order = np.argsort(v)
    vs, ss = v[order], spec[order]
    in_win = np.abs(vs) <= win_kms
    if not in_win.any() or not np.isfinite(ss[in_win]).any():
        return np.nan, np.nan
    peak = np.nanmax(ss[in_win])
    thr = frac * peak
    idx = np.where(in_win & (ss >= thr))[0]
    if len(idx) < 2:
        return np.nan, np.nan
    width = vs[idx[-1]] - vs[idx[0]]
    sel = in_win & (ss >= 0.2 * peak)
    v_cen = float(np.nansum(vs[sel] * ss[sel]) / np.nansum(ss[sel]))
    return float(width), v_cen


def grids_equal(h1, h2):
    keys = ["NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
            "CRPIX1", "CRPIX2"]
    return all(np.isclose(h1.get(k, 0), h2.get(k, 1), rtol=1e-9) for k in keys)


def place_nulls(rng, mask, valid, beam_fwhm_pix, K):
    """≤ K integer (dy, dx) shifts. Nulls MAY overlap each other (a compact
    FoV cannot hold K disjoint copies of a union mask); each shifted mask must
    lie fully inside `valid` and avoid the source mask dilated by 1 beam."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ny, nx = mask.shape
    dil = max(3, int(np.ceil(beam_fwhm_pix)))
    src_dil = ndimage.binary_dilation(mask, iterations=dil)
    min_sep = max(2.0, beam_fwhm_pix)
    shifts, tries = [], 0
    while len(shifts) < K and tries < K * 1500:
        tries += 1
        dy = int(rng.integers(-y0, ny - 1 - y1))
        dx = int(rng.integers(-x0, nx - 1 - x1))
        if dy == 0 and dx == 0:
            continue
        sy, sx = ys + dy, xs + dx
        if not valid[sy, sx].all():
            continue
        if src_dil[sy, sx].any():
            continue
        if any((dy - py) ** 2 + (dx - px) ** 2 < min_sep ** 2
               for py, px in shifts):
            continue
        shifts.append((dy, dx))
    return shifts


def beam_fwhm_pix_of(hdr):
    return hdr["BMAJ"] / abs(hdr["CDELT1"])


def compute_mom0(cfg, strong, v_s, dv_s):
    """Strong-tracer mom-0 over the config window, cached to .npy."""
    cache = OUT_DIR / f"{cfg['name']}_strong_mom0.npy"
    in_win = np.abs(v_s) <= cfg["strong_win_kms"] / 2.0
    ch_idx = np.where(in_win)[0]
    if cache.exists():
        mom0 = np.load(cache)
        finite2d = np.isfinite(mom0)
        return mom0, finite2d, ch_idx
    mom0 = np.zeros(strong.data.shape[1:], dtype=np.float64)
    for ch in ch_idx:
        mom0 += np.nan_to_num(np.asarray(strong.data[ch], dtype=np.float64))
    mom0 *= dv_s
    finite2d = np.isfinite(np.asarray(strong.data[ch_idx[0]]))
    if cfg["pb_mult"]:
        pb = np.squeeze(fits.getdata(cfg["pb_mult"]))
        mom0 = mom0 * np.nan_to_num(pb)
        finite2d &= np.isfinite(pb) & (pb > 0.2)
    mom0[~finite2d] = np.nan
    np.save(cache, mom0)
    return mom0, finite2d, ch_idx


def build_mask(mom0, finite2d, beam_fwhm_pix, beam_area, s):
    """2σ contour union mask on mom-0 smoothed by s×beam; island floor
    scales with the effective (smoothed) beam area, 1 + s²."""
    if s > 0:
        sig = s * beam_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
        work = convolve_fft(mom0, Gaussian2DKernel(sig),
                            normalize_kernel=True, nan_treatment="interpolate",
                            boundary="fill", allow_huge=True)
        work[~finite2d] = np.nan
    else:
        work = mom0
    _, med, std = sigma_clipped_stats(work[finite2d], sigma=3.0, maxiters=5)
    raw = np.zeros_like(finite2d)
    raw[finite2d] = work[finite2d] > med + SIGMA_CONTOUR * std
    labels, n_isl = ndimage.label(raw, structure=np.ones((3, 3)))
    floor = beam_area * (1.0 + s * s)
    idx = np.arange(1, n_isl + 1)
    areas = ndimage.sum_labels(np.ones_like(labels), labels, index=idx)
    peaks = ndimage.maximum(np.nan_to_num(work, nan=-np.inf), labels,
                            index=idx) if n_isl else np.array([])
    keep_ids = idx[(areas >= floor) & (peaks >= med + SEED_SIGMA * std)]
    mask = np.isin(labels, keep_ids)
    core = np.zeros_like(finite2d)
    core[finite2d] = work[finite2d] > med + 3.0 * std
    skirt = mask & ~core
    return mask, skirt, n_isl, len(keep_ids), work


def run_galaxy(cfg):
    print(f"\n=== {cfg['name']} ===", flush=True)
    strong = load_cube(cfg["strong_path"])
    s_hdr = strong.header
    rest_s = REST_HZ[cfg["strong_line"]] / (1.0 + cfg["z"])
    v_s, _ = channel_velocities(s_hdr, rest_s)
    dv_s = float(np.abs(np.median(np.diff(v_s))))
    beam_area_s = get_beam_area_pix(strong)
    bfp = beam_fwhm_pix_of(s_hdr)
    win = cfg["strong_win_kms"]

    mom0, finite2d, ch_idx = compute_mom0(cfg, strong, v_s, dv_s)

    variants = {}
    for s in SMOOTH_VARIANTS:
        mask, skirt, n_isl, n_kept, work = build_mask(
            mom0, finite2d, bfp, beam_area_s, s)
        if not mask.any():
            print(f"  [s={s:g}] empty mask — variant skipped", flush=True)
            continue
        frac = mask.sum() / max(finite2d.sum(), 1)
        variants[s] = dict(mask=mask, skirt=skirt, n_isl=n_isl,
                           n_kept=n_kept, frac=frac, work=work)
        print(f"  [s={s:g}×beam] islands {n_isl} → kept {n_kept}; "
              f"{mask.sum()} pix = {mask.sum()/beam_area_s:.1f} beams; "
              f"FoV frac {frac:.3f}"
              f"{'  ⚠ RED FLAG (extended?)' if frac > MASK_FRAC_RED_FLAG else ''}",
              flush=True)

    # --- strong-tracer spectra (all variant masks, one pass) ---
    pad_ch = int(round(400.0 / dv_s))
    lo = max(0, ch_idx.min() - pad_ch)
    hi = min(len(v_s) - 1, ch_idx.max() + pad_ch)
    svar = list(variants.keys())
    s_masks = [variants[s]["mask"] for s in svar]
    s_specs = multi_mask_spectra(strong.data, s_masks, beam_area_s, lo, hi)
    strong.data = None
    for i, s in enumerate(svar):
        spec = s_specs[i]
        base = np.nanmedian(spec[(np.abs(v_s) > win / 2.0) & np.isfinite(spec)])
        sub = spec - base
        w50, v_cen = line_width(v_s, sub, win / 2.0 + 100, 0.5)
        w20, _ = line_width(v_s, sub, win / 2.0 + 100, 0.2)
        variants[s].update(spec_s=sub, w50=w50, w20=w20, v_cen=v_cen)
        print(f"  [s={s:g}] W50={w50:.0f}  W20={w20:.0f}  v_cen={v_cen:+.0f} km/s",
              flush=True)

    # --- HRL cube: nulls + spectra for every variant in ONE pass ---
    hrl = load_cube(cfg["hrl_path"])
    h_hdr = hrl.header
    rest_h = REST_HZ[cfg["hrl_line"]] / (1.0 + cfg["z"])
    v_h, freqs_h = channel_velocities(h_hdr, rest_h)
    dv_h = float(np.abs(np.median(np.diff(v_h))))
    beam_area_h = get_beam_area_pix(hrl)
    if not grids_equal(s_hdr, h_hdr):
        raise RuntimeError("grid mismatch — add reproject step for this pair")
    mid = len(v_h) // 2
    valid = np.isfinite(np.asarray(hrl.data[mid]))
    rng = np.random.default_rng(RNG_SEED)

    all_masks, layout = [], {}
    for s in svar:
        vv = variants[s]
        shifts = place_nulls(rng, vv["mask"], valid, bfp, K_NULLS)
        vv["n_nulls"] = len(shifts)
        ys, xs = np.where(vv["mask"])
        nm = []
        for dy, dx in shifts:
            m = np.zeros_like(vv["mask"])
            m[ys + dy, xs + dx] = True
            nm.append(m)
        layout[s] = (len(all_masks), len(nm))
        all_masks += [vv["mask"], vv["skirt"]] + nm
        print(f"  [s={s:g}] placed {len(shifts)}/{K_NULLS} nulls", flush=True)

    w20max = np.nanmax([variants[s]["w20"] for s in svar])
    v_cen0 = variants[svar[-1]]["v_cen"]
    center_ch = int(np.argmin(np.abs(v_h - v_cen0)))
    span_kms = max(w20max / 2.0, 150.0) + 700.0
    ch_lo = max(0, center_ch - int(round(span_kms / dv_h)))
    ch_hi = min(len(v_h) - 1, center_ch + int(round(span_kms / dv_h)))
    specs = multi_mask_spectra(hrl.data, all_masks, beam_area_h, ch_lo, ch_hi)
    hrl.data = None

    keep = line_free_mask_for_cube(
        v_chan=v_h, primary_line_rest_hz=rest_h,
        cube_freq_range_hz=(float(freqs_h.min()), float(freqs_h.max())),
        z=cfg["z"], primary_line_key=cfg["hrl_line"], fwhm_kms=300.0)
    hrl_zone = np.abs(v_h - v_cen0) <= max(w20max, 300.0) / 2.0
    specs[:, ~(keep | hrl_zone)] = np.nan

    rows = []
    for s in svar:
        vv = variants[s]
        i0, n_null = layout[s]
        spec_sig, spec_skirt = specs[i0], specs[i0 + 1]
        spec_nulls = specs[i0 + 2: i0 + 2 + n_null]
        c_ch = int(np.argmin(np.abs(v_h - vv["v_cen"])))
        for wname, wraw in [("W50", vv["w50"]), ("W20", vv["w20"])]:
            if not np.isfinite(wraw):
                continue
            wkms = max(wraw, W_FLOOR_KMS)   # v3: window floor
            excl = max(300.0, wkms / 2.0 + 60.0)
            F_sig, sn_sig, _, _ = calc_flux_sn(
                spec_sig, c_ch, wkms, dv_h,
                baseline_exclude_half_width_kms=excl)
            F_skirt, sn_skirt, _, _ = calc_flux_sn(
                spec_skirt, c_ch, wkms, dv_h,
                baseline_exclude_half_width_kms=excl)
            f_nulls = np.array([calc_flux_sn(
                sp, c_ch, wkms, dv_h,
                baseline_exclude_half_width_kms=excl)[0]
                for sp in spec_nulls])
            null_mean = float(np.mean(f_nulls)) if len(f_nulls) else np.nan
            null_std = float(np.std(f_nulls)) if len(f_nulls) else np.nan
            # v3: null is a DIAGNOSTIC column; headline flux = baseline-
            # subtracted direct integral (spectral degree-1 baseline)
            sigma_emp = F_sig / sn_sig if sn_sig else np.nan
            dev = (F_sig - cfg["lit_flux"]) / cfg["lit_flux"] * 100
            rows.append(dict(
                galaxy=cfg["name"], smooth=s, variant=wname,
                width_kms=round(wkms, 1), v_cen=round(vv["v_cen"], 1),
                F=round(F_sig, 4), sigma_emp=round(sigma_emp, 4),
                SN=round(sn_sig, 2),
                null_mean=round(null_mean, 4) if np.isfinite(null_mean) else "",
                null_std=round(null_std, 4) if np.isfinite(null_std) else "",
                F_skirt=round(F_skirt, 4), SN_skirt=round(sn_skirt, 2),
                lit=f"{cfg['lit_flux']} ({cfg['lit_ref']})",
                dev_pct=round(dev, 1), prior=cfg["prior_flux"],
                n_islands=vv["n_kept"],
                nbeam=round(vv["mask"].sum() / beam_area_s, 1),
                mask_fov_frac=round(float(vv["frac"]), 3),
                red_flag=bool(vv["frac"] > MASK_FRAC_RED_FLAG),
                n_nulls=vv["n_nulls"],
            ))
            print(f"  [s={s:g} {wname}={wkms:.0f}] F={F_sig:.3f}  σ={sigma_emp:.3f}  "
                  f"S/N={sn_sig:.1f}  null={null_mean:+.3f}±{null_std:.3f}  "
                  f"skirt={F_skirt:.3f}({sn_skirt:.1f}σ)  dev {dev:+.1f}%",
                  flush=True)

        # --- figure per (galaxy, smooth) ---
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        ax = axes[0]
        show = np.where(finite2d, vv["work"], np.nan)
        vmax = np.nanpercentile(show, 99.5)
        ax.imshow(show, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(vv["mask"], levels=[0.5], colors="lime", linewidths=1.2)
        ax.set_title(f"{cfg['name']} s={s:g}×beam: {vv['n_kept']} islands, "
                     f"{vv['mask'].sum()/beam_area_s:.0f} beams, "
                     f"{vv['n_nulls']} nulls")
        ax = axes[1]
        ax.step(v_s, vv["spec_s"], where="mid", color="C0", lw=1)
        for wv, cc in [(vv["w50"], "C3"), (vv["w20"], "C1")]:
            if np.isfinite(wv):
                ax.axvspan(vv["v_cen"] - wv / 2, vv["v_cen"] + wv / 2,
                           alpha=0.12, color=cc)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(-win, win)
        ax.set_title(f"{cfg['strong_line']}: W50={vv['w50']:.0f} "
                     f"W20={vv['w20']:.0f}")
        ax.set_xlabel("v [km/s]"); ax.set_ylabel("S [Jy]")
        ax = axes[2]
        ax.step(v_h, spec_sig, where="mid", color="C0", lw=1, label="signal")
        ax.step(v_h, spec_skirt, where="mid", color="C1", lw=0.8,
                label="skirt (2σ–3σ)")
        for sp in spec_nulls:
            ax.step(v_h, sp, where="mid", color="gray", lw=0.4, alpha=0.35)
        if np.isfinite(vv["w50"]):
            ax.axvspan(vv["v_cen"] - vv["w50"] / 2, vv["v_cen"] + vv["w50"] / 2,
                       alpha=0.12, color="C3")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(v_h[ch_lo], v_h[ch_hi])
        ax.set_title(f"{cfg['hrl_line']} signal / skirt / {vv['n_nulls']} nulls")
        ax.set_xlabel("v [km/s]"); ax.set_ylabel("S [Jy]")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out_png = OUT_DIR / f"{cfg['name']}_s{s:g}_uniform_poc.png"
        plt.savefig(out_png, dpi=130)
        plt.close()
        print(f"  saved {out_png}", flush=True)
    return rows


def main():
    OUT_DIR.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    configs = [c for c in CONFIGS if not only or c["name"] in only]
    all_rows = []
    for cfg in configs:
        try:
            all_rows.extend(run_galaxy(cfg))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR on {cfg['name']}: {e}", flush=True)
    if all_rows:
        out_csv = OUT_DIR / "uniform_poc_results.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\nwrote {out_csv}")
        print(f"\n{'galaxy':<12}{'s':>3}{'var':>5}{'W':>5}{'F':>9}"
              f"{'σ':>8}{'S/N':>7}{'skirt':>8}{'dev%':>7}{'isl':>5}{'nbeam':>7}")
        for r in all_rows:
            print(f"{r['galaxy']:<12}{r['smooth']:>3g}{r['variant']:>5}"
                  f"{r['width_kms']:>5.0f}{r['F']:>9.3f}"
                  f"{r['sigma_emp']:>8.3f}{r['SN']:>7.1f}"
                  f"{r['F_skirt']:>8.3f}{r['dev_pct']:>7.1f}"
                  f"{r['n_islands']:>5}{r['nbeam']:>7.1f}")


if __name__ == "__main__":
    main()
