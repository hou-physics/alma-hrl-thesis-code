"""Mask-level 8um apportioning of the total infrared luminosity — the
survey's ADOPTED x-axis scheme (promoted from the S5 trial, decisions
2026-08-21; the header below records the trial-era finding).

Case-closed finding (README): predecessors' x = MASK-level TIR via the
Spitzer 8μm fraction (Lau eq 2.8); our v1 x was PB-level → mixed apertures
→ points fall below their fit. This trial recomputes x with THEIR recipe:

    x = [f_8μm(CO mask) / f_8μm(galaxy)] × L_TIR(global, SM96 on IRAS)

8μm science mosaics: Spitzer SEIP via IRSA SIA (0.6″/px, MJy/sr — Lau's
own data product family). Fractions are unit-free.

Scope: the 6 detections (offset adjudication); extendable to all i4 sources.
Run: conda run -n casa_env python -u masklevel_x.py   (from this dir)
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

STEP3 = "/Volumes/HouAstro/master/master_thesis/my_code/_step3"
sys.path.insert(0, STEP3)
from uniform_batch_configs import build_table                   # noqa: E402
from uniform_batch_stage1 import REST_HZ, PB_MULT               # noqa: E402
from uniform_batch_stage3_ir import alma_geometry               # noqa: E402
from uniform_contour_poc import build_mask, beam_fwhm_pix_of    # noqa: E402
from mask import reproject_mask                                 # noqa: E402
from uniform_batch_stage4_plot import (load, line_lum_conv, N6,  # noqa: E402
                                       LINE_STYLE, HIST_FIT, HIST_DIGITIZED,
                                       MPC_M, LSUN_W)

OUT = Path(__file__).parent
BATCH = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
DETECTIONS = ["ngc4945", "ngc3628", "ngc253", "he2-10", "eso097-013",
              "ngc4826"]
S_FROZEN = 1.0


def fetch_8um(name, ra, dec):
    """Fetch long-exposure IRAC4 mosaic AND the HDR short-exposure mosaic
    (saturation patch material). Returns (long_path, short_path_or_None)."""
    dest = OUT / f"{name}_irac4.fits"
    dest_s = OUT / f"{name}_irac4_short.fits"
    noshort = OUT / f"{name}_noshort.flag"
    if dest.exists() and (dest_s.exists() or noshort.exists()):
        return dest, (dest_s if dest_s.exists() else None)
    r = requests.get("https://irsa.ipac.caltech.edu/SIA",
                     params=dict(COLLECTION="spitzer_seip",
                                 POS=f"circle {ra} {dec} 0.02",
                                 RESPONSEFORMAT="CSV", MAXREC="200"),
                     timeout=90)
    rows = list(csv.DictReader(io.StringIO(r.text)))
    longs = [x["access_url"] for x in rows
             if x.get("energy_bandpassname") == "IRAC4"
             and x["access_url"].endswith(".IRAC.4.mosaic.fits")
             and "-short" not in x["access_url"]]
    shorts = [x["access_url"] for x in rows
              if x.get("energy_bandpassname") == "IRAC4"
              and "-short.IRAC.4" in x["access_url"]
              and x["access_url"].endswith("mosaic.fits")]
    if not longs:
        return None, None
    if not dest.exists():
        print(f"    downloading {longs[0].rsplit('/',1)[-1]}", flush=True)
        dest.write_bytes(requests.get(longs[0], timeout=600).content)
    if not dest_s.exists() and not noshort.exists():
        if shorts:
            print(f"    downloading {shorts[0].rsplit('/',1)[-1]}", flush=True)
            dest_s.write_bytes(requests.get(shorts[0], timeout=600).content)
        else:
            noshort.touch()
    return dest, (dest_s if dest_s.exists() else None)


def hdr_patch(img_long, h_long, short_path):
    """Replace saturated (NaN, dilated 3 px — near-saturated neighbours are
    nonlinear too) pixels of the long-exposure mosaic with short-exposure
    values. Cross-calibration check: median long/short ratio on bright,
    mutually-valid pixels must be ≈ 1 (both in MJy/sr).
    Returns (patched, n_patched, ratio)."""
    from scipy import ndimage as ndi
    with fits.open(short_path) as hdul:
        img_s = np.asarray(hdul[0].data, dtype=np.float64)
        h_s = hdul[0].header
    if img_s.shape != img_long.shape or \
            not np.isclose(h_s.get("CRVAL1", 0), h_long.get("CRVAL1", 1)):
        from reproject import reproject_interp
        img_s, _ = reproject_interp((img_s, WCS(h_s)), WCS(h_long),
                                    shape_out=img_long.shape)
    bad = ~np.isfinite(img_long)
    bad_d = ndi.binary_dilation(bad, iterations=3)
    ov = np.isfinite(img_long) & np.isfinite(img_s)
    ratio = np.nan
    if ov.sum() > 1000:
        # select on the LONG mosaic's bright valid pixels (real emission);
        # selecting on the noisy short frame picks noise spikes in empty
        # fields and fakes tiny ratios (bug caught 2026-08-03)
        lo, hi = np.nanpercentile(img_long[ov], [99.0, 99.98])
        sel = ov & (img_long > lo) & (img_long < hi)
        if sel.sum() > 50:
            ratio = float(np.nanmedian(img_long[sel] / img_s[sel]))
    patched = img_long.copy()
    fill = bad_d & np.isfinite(img_s)
    patched[fill] = img_s[fill]
    return patched, int(fill.sum()), ratio


def mask_fraction_8um(name, row):
    """Rebuild frozen CO mask, reproject onto the 8μm mosaic, return
    f8(mask)/f8(galaxy) with background subtraction + CoG total."""
    ra, dec, *_ = alma_geometry(row)
    p8, p8s = fetch_8um(name, ra, dec)
    if p8 is None:
        return np.nan, "no-8um"
    mom0 = np.load(BATCH / f"{name}_strong_mom0.npy")
    finite2d = np.isfinite(mom0)
    strong_path = row["strong_path"]
    if name in PB_MULT and (not strong_path or not Path(strong_path).exists()):
        strong_path = row["hrl_path"]
    s_hdr = fits.getheader(strong_path)
    from cube_io import load_cube, get_beam_area_pix
    cube = load_cube(strong_path)
    beam_area = get_beam_area_pix(cube)
    cube.data = None
    mask, _, _, _, _ = build_mask(mom0, finite2d, beam_fwhm_pix_of(s_hdr),
                                  beam_area, S_FROZEN)

    with fits.open(p8) as hdul:
        img8 = np.asarray(hdul[0].data, dtype=np.float64)
        h8 = hdul[0].header
    patch_note = ""
    if p8s is not None and not NO_PATCH:
        img8, n_patch, xratio = hdr_patch(img8, h8, p8s)
        if n_patch:
            patch_note = f" hdrpatch({n_patch}px,r={xratio:.2f})"
            if np.isfinite(xratio) and abs(xratio - 1) > 0.10:
                patch_note += " XCAL!"
    m8 = reproject_mask(mask, WCS(s_hdr), WCS(h8), img8.shape)
    if not m8.any():
        return np.nan, "mask-off-mosaic"

    w8 = WCS(h8)
    xc, yc = w8.world_to_pixel_values(ra, dec)
    pix = abs(float(h8.get("CDELT1") or h8.get("CD1_1"))) * 3600.0
    yy, xx = np.indices(img8.shape)
    r = np.hypot(yy - yc, xx - xc) * pix
    bg_zone = (r > 350) & (r < 550) & np.isfinite(img8)
    if bg_zone.sum() < 500:
        bg_zone = (r > 250) & (r < 400) & np.isfinite(img8)
    _, bg, _ = sigma_clipped_stats(img8[bg_zone], sigma=3.0, maxiters=5)
    work = img8 - bg

    f_mask = float(np.nansum(work[m8]))
    # galaxy total: SB-truncated CoG (annular mean < 2σ of bg-annulus scatter)
    edges = np.arange(0, 340, 6.0)
    ann = np.array([np.nanmean(work[(r >= a) & (r < b)])
                    for a, b in zip(edges[:-1], edges[1:])])
    bg_sb = np.array([np.nanmean(work[(r >= a) & (r < b)])
                      for a, b in zip(np.arange(350, 540, 6),
                                      np.arange(356, 546, 6))])
    thr = 2 * np.nanstd(bg_sb)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    r_edge = 330.0
    for i in range(len(ann) - 1):
        if r_mid[i] > 20 and ann[i] < thr and ann[i + 1] < thr:
            r_edge = r_mid[i]
            break
    f_gal = float(np.nansum(work[(r <= r_edge) & np.isfinite(work)]))
    if f_gal <= 0 or f_mask <= 0:
        return np.nan, "nonpositive-flux"
    flag = "" if f_mask <= f_gal else "mask>gal?"
    n_sat = int(np.sum(~np.isfinite(img8[m8])))
    if n_sat > 0.02 * m8.sum():
        flag += " sat/NaN-in-mask"
    flag += patch_note
    return f_mask / f_gal, flag or "ok"


NO_PATCH = False


def main():
    global NO_PATCH
    run_all = "all" in sys.argv[1:]
    NO_PATCH = "nopatch" in sys.argv[1:]
    tbl = {r["galaxy"]: r for r in build_table()}
    rows = load() if run_all else \
        [r for r in load() if r["galaxy"] in DETECTIONS]
    results = []
    for r in rows:
        g = r["galaxy"]
        print(f"=== {g} ===", flush=True)
        try:
            frac8, flag = mask_fraction_8um(g, tbl[g])
        except Exception as e:
            import traceback; traceback.print_exc()
            frac8, flag = np.nan, f"ERROR {e}"
        d2 = 4 * np.pi * (r["D"] * MPC_M) ** 2 / LSUN_W
        L_gal = (r["fir_central"] / r["frac"]) * d2      # global TIR
        y = r["F"] * line_lum_conv(r) * N6[r["line"]]
        x_pb = r["fir_central"] * d2
        x_mask = L_gal * frac8 if np.isfinite(frac8) else np.nan
        off_pb = np.log10(y) - (HIST_FIT[0] * np.log10(x_pb) + HIST_FIT[1])
        off_mk = np.log10(y) - (HIST_FIT[0] * np.log10(x_mask) + HIST_FIT[1]) \
            if np.isfinite(x_mask) else np.nan
        print(f"  frac8={frac8 if np.isfinite(frac8) else float('nan'):.3f} "
              f"[{flag}]  offset: {off_pb:+.2f} → "
              f"{off_mk if np.isfinite(off_mk) else float('nan'):+.2f} dex",
              flush=True)
        results.append(dict(galaxy=g, line=r["line"], frac8=frac8, flag=flag,
                            x_pb=x_pb, x_mask=x_mask, y=y,
                            det=r["verdict"] == "detection",
                            F=r["F"], sig=r["sig"], conv=line_lum_conv(r),
                            off_pb=round(off_pb, 2),
                            off_mask=round(off_mk, 2) if np.isfinite(off_mk)
                            else ""))

    csv_name = "s5_results_nopatch.csv" if NO_PATCH else "s5_results.csv"
    with open(OUT / csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["galaxy", "line", "frac8", "flag",
                                          "off_pb", "off_mask"])
        w.writeheader()
        w.writerows([{k: r[k] for k in w.fieldnames} for r in results])

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    for lx, ly, ul in HIST_DIGITIZED:
        ax.plot(10 ** lx, 10 ** ly, "x", color="#bbbbbb", ms=6, mew=1.3,
                zorder=1)
    xs = np.array([1e8, 3e12])
    ax.plot(xs, 10 ** (HIST_FIT[0] * np.log10(xs) + HIST_FIT[1]), "-",
            color="#444444", lw=1.2, label="Bittner 2022 fit")
    for r in results:
        st = LINE_STYLE[r["line"]]
        ax.plot(r["x_pb"], r["y"], st["marker"], color=st["color"],
                mfc="none", ms=8, alpha=0.5)
        if np.isfinite(r["x_mask"]):
            ax.plot(r["x_mask"], r["y"], st["marker"], color=st["color"],
                    ms=9)
            ax.annotate("", xy=(r["x_mask"], r["y"]),
                        xytext=(r["x_pb"], r["y"]),
                        arrowprops=dict(arrowstyle="->", lw=0.8,
                                        color="#888888"))
            ax.annotate(r["galaxy"], (r["x_mask"], r["y"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L_TIR [Lsun]  (open = PB-level x, filled = 8um mask-level x)")
    ax.set_ylabel("n6 L_HRL [Lsun]")
    ax.set_title("S5 (EXPLORATORY): x recomputed with predecessors' "
                 "mask-level 8um recipe", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "s5_masklevel_x.png", dpi=150)
    print(f"\nsaved {OUT / 's5_masklevel_x.png'} + s5_results.csv", flush=True)

    if not run_all:
        return
    # ---- FULL corrected overlay: every source at mask-level x where 8um
    # exists; fallbacks (no IRAC4) plotted at PB-level x with gray edge ----
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    for lx, ly, ul in HIST_DIGITIZED:
        ax.plot(10 ** lx, 10 ** ly, "x", color="#bbbbbb", ms=6, mew=1.3,
                zorder=1)
    xs = np.array([1e8, 3e12])
    ax.plot(xs, 10 ** (HIST_FIT[0] * np.log10(xs) + HIST_FIT[1]), "-",
            color="#444444", lw=1.2, label="Bittner 2022 fit")
    n_fb = 0
    for r in results:
        st = LINE_STYLE[r["line"]]
        has8 = np.isfinite(r["x_mask"])
        x = r["x_mask"] if has8 else r["x_pb"]
        ysig = r["sig"] * r["conv"] * N6[r["line"]]
        yF = r["F"] * r["conv"] * N6[r["line"]]
        floor = (not r["det"]) and (yF < 0.5 * ysig)
        yy = max(yF, ysig) if floor else yF
        if yy <= 0 or not np.isfinite(x) or x <= 0:
            n_fb += not has8
            continue
        mec = st["color"] if has8 else "#999999"
        ax.errorbar(x, yy, yerr=ysig if r["det"]
                    else np.array([[0.0], [ysig]]),
                    fmt=st["marker"], color=st["color"], mec=mec,
                    mfc=st["color"] if r["det"] else "none",
                    ms=8 if r["det"] else 6, capsize=2, lw=1.0,
                    zorder=3 if r["det"] else 2)
        if not r["det"]:
            ax.annotate("", xy=(x, yy / 2.2), xytext=(x, yy),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color=st["color"], alpha=0.8))
        if r["det"]:
            ax.annotate(r["galaxy"], (x, yy), textcoords="offset points",
                        xytext=(6, 4), fontsize=7)
        n_fb += not has8
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$L_\mathrm{TIR}$ (mask-level, 8um recipe) [$L_\odot$]")
    ax.set_ylabel(r"$n^6 L_\mathrm{HRL}$ [$L_\odot$]")
    handles = [plt.Line2D([], [], ls="", **LINE_STYLE[k]) for k in LINE_STYLE]
    handles.append(plt.Line2D([], [], ls="", marker="x", color="#bbbbbb",
                              label="previous works (digitized)"))
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0][:1],
              loc="upper left", fontsize=8)
    ax.set_title("FULL corrected overlay (EXPLORATORY) — aperture-consistent "
                 "axes; gray-edged = no IRAC4, PB-level fallback",
                 fontsize=9, color="#666666")
    plt.tight_layout()
    plt.savefig(OUT / "s5_full_corrected.png", dpi=170)
    print(f"saved {OUT / 's5_full_corrected.png'} (fallbacks: {n_fb})",
          flush=True)


if __name__ == "__main__":
    main()
