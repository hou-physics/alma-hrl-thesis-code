"""Frank-meeting figure: the ONE remaining plot decision, side by side.

Two panels, identical y (n^6 L_HRL, v3.2 fluxes, literature distances),
identical previous-works overlay — only the x-axis aperture differs:
  A: ALMA-FoV aperture      x = cov_corr x global TIR   (current pipeline)
  B: mask-level 8um aperture x = frac8    x global TIR   (Lau/Bittner recipe,
     s5 frac8 values — masks unchanged since s5 ran; HDR-patched)
Sources without SEIP 8um (4) fall back to panel-A x in panel B (gray edge).

Output: result_v2/_uniform_batch/xaxis_options_frank.{png,pdf}
Run: conda run -n casa_env python -u _step3/xaxis_options_frank.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_stage4_plot import (load, draw, line_lum_conv, N6,
                                       LINE_STYLE, HIST_FIT, HIST_DIGITIZED,
                                       MPC_M, LSUN_W)
from plot_style import setup_thesis_style

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
S5 = OUT / "offset_investigation" / "s5_masklevel_x"


def overlay(ax):
    for lx, ly, ul in HIST_DIGITIZED:
        ax.plot(10 ** lx, 10 ** ly, "x", color="#bbbbbb", ms=6, mew=1.3,
                zorder=1)
    xs = np.array([1e8, 3e12])
    ax.plot(xs, 10 ** (HIST_FIT[0] * np.log10(xs) + HIST_FIT[1]), "-",
            color="#444444", lw=1.2)


def det_offsets(rows, xkey):
    offs = []
    for r in rows:
        if r["verdict"] != "detection":
            continue
        y = r["F"] * line_lum_conv(r) * N6[r["line"]]
        x = r[xkey]
        if y > 0 and np.isfinite(x) and x > 0:
            offs.append(np.log10(y) - (HIST_FIT[0] * np.log10(x)
                                       + HIST_FIT[1]))
    return np.array(offs)


def main():
    setup_thesis_style()
    frac8 = {r["galaxy"]: float(r["frac8"])
             if r["frac8"] not in ("", "nan") else np.nan
             for r in csv.DictReader(open(S5 / "s5_results.csv"))}
    rows = load()
    lrows = [r for r in rows if np.isfinite(r["D"]) and r["D"] > 0]
    n_fb = 0
    for r in lrows:
        d2 = 4 * np.pi * (r["D"] * MPC_M) ** 2 / LSUN_W
        r["L_tir"] = r["fir_central"] * d2               # panel A
        L_gal = (r["fir_central"] / r["frac"]) * d2      # global TIR
        f8 = frac8.get(r["galaxy"], np.nan)
        if np.isfinite(f8):
            r["L_tir_mask"] = L_gal * f8                 # panel B
        else:
            r["L_tir_mask"] = r["L_tir"]
            r["_fallback"] = True
            n_fb += 1

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), sharey=True)
    for ax, xkey, title in [
            (axes[0], "L_tir",
             "Option A — IR apportioned to the ALMA field of view\n"
             "x = (fraction of IR light inside the ALMA primary beam) "
             r"$\times\ L_{\rm TIR}^{\rm global}$"),
            (axes[1], "L_tir_mask",
             "Option B — IR apportioned to the CO aperture "
             "(Lau/Bittner method)\n"
             "x = (fraction of 8 μm light inside the CO aperture) "
             r"$\times\ L_{\rm TIR}^{\rm global}$")]:
        overlay(ax)
        draw(ax, lrows, xkey, lambda r: line_lum_conv(r) * N6[r["line"]])
        if xkey == "L_tir_mask":        # mark the no-SEIP fallbacks
            for r in lrows:
                if r.get("_fallback"):
                    y = max(r["F"], r["sig"]) * line_lum_conv(r) * N6[r["line"]]
                    if y > 0 and r[xkey] > 0:
                        ax.plot(r[xkey], y, "o", ms=13, mfc="none",
                                mec="#999999", mew=1.4, zorder=4)
        ax.set_ylim(5e7, 2e14)
        off = det_offsets(lrows, xkey)
        ax.set_title(title + f"\ndetections vs Bittner fit: mean "
                     f"{off.mean():+.2f} dex, {sum(off < 0)}/{len(off)} below",
                     fontsize=10)
        ax.set_xlabel(r"$L_\mathrm{TIR}$ (aperture-matched) [$L_\odot$]")
    axes[0].set_ylabel(r"$n^6\,L_\mathrm{HRL}$ [$L_\odot$]")
    handles = [plt.Line2D([], [], ls="", **LINE_STYLE[k]) for k in LINE_STYLE]
    handles += [
        plt.Line2D([], [], ls="", marker="x", color="#999999", mew=1.6,
                   label="previous works (digitized)"),
        plt.Line2D([], [], ls="-", color="#444444",
                   label="linear fit, Bittner 2022"),
        plt.Line2D([], [], ls="", marker="o", ms=10, mfc="none",
                   mec="#999999", mew=1.4,
                   label="no Spitzer 8 μm map → x as in panel A (B only)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)
    fig.suptitle(
        "Open choice: over which area is the IR luminosity apportioned? "
        " ·  both panels: same 38 sources, same HRL measurements, same "
        "distances (literature for nearby, Hubble-flow for distant)  ·  "
        f"panel B: {n_fb} sources without Spitzer 8 μm keep the panel-A x",
        fontsize=10)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"xaxis_options_frank.{ext}", dpi=200)
    print(f"A: det mean offset {det_offsets(lrows,'L_tir').mean():+.2f} dex; "
          f"B: {det_offsets(lrows,'L_tir_mask').mean():+.2f} dex; "
          f"fallbacks {n_fb}")
    print(f"saved {OUT / 'xaxis_options_frank.png'}")


if __name__ == "__main__":
    main()
