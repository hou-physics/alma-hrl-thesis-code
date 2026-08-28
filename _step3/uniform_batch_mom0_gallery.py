"""Bright-tracer moment-0 gallery — one contact sheet for quick visual
inspection of every analyzed galaxy's shape/size (no manual FITS opening).

Per panel: cached stage-1 mom-0 (RdBu_r, zero-centered, project convention),
frozen-prescription mask contour (lime), beam FWHM circle (bottom left),
30″ scale bar with kpc equivalent (cz/70 where valid). Wave-1 targets are
tagged "· w1" with a gold title (double-coded).

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_mom0_gallery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table               # noqa: E402
from uniform_batch_stage1 import PB_MULT                    # noqa: E402
from uniform_contour_poc import build_mask, beam_fwhm_pix_of  # noqa: E402
from plot_style import setup_thesis_style, CMAP_MOMENT0     # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
S_FROZEN = 1.0
C_KMS, H0 = 299792.458, 70.0

# Distances the thesis adopts (literature where one exists, flow otherwise).
# Scale bars must use these: cz/H0 is not usable below ~10 Mpc.
def _adopted_distances():
    import csv as _csv
    f = OUT / "adopted_distances.csv"
    if not f.exists():
        return {}
    with open(f) as fh:
        out = {}
        for r in _csv.DictReader(fh):
            try:
                out[r["galaxy"]] = float(r["D_Mpc"])
            except (KeyError, TypeError, ValueError):
                pass
        return out


ADOPTED_D = _adopted_distances()
MAX_DISP = 700          # downsample display to ≤ this many px per side
WAVE1 = {"irasf05189-2524", "iras19542+1110", "iras09022-3615",
         "irasf14378-3651", "irasf19297-0406", "iras07251-0248",
         "irasf12112+0305", "ngc2369", "irasf14348-1447", "ngc5128",
         "ngc7469", "ngc3256", "ngc1808", "ngc55"}
GOLD = "#b8860b"


def panel(ax, row):
    name = row["galaxy"]
    cache = OUT / f"{name}_strong_mom0.npy"
    if not cache.exists():
        ax.text(0.5, 0.5, f"{name}\n(no mom-0 cache)", ha="center",
                va="center", transform=ax.transAxes, fontsize=8)
        ax.set_axis_off()
        return
    mom0 = np.load(cache)
    finite2d = np.isfinite(mom0)
    # crop to the real-FoV bounding box (product cubes carry padding —
    # NaN in most, exact-zero in some, e.g. NGC 1377 — squeezing the
    # panel into a corner)
    ys, xs = np.where(finite2d & (mom0 != 0.0))
    if len(ys):
        m = 4
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, mom0.shape[0])
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, mom0.shape[1])
        crop = (slice(y0, y1), slice(x0, x1))
    else:
        crop = (slice(None), slice(None))
    strong_path = row["strong_path"]
    if name in PB_MULT and (not strong_path or not Path(strong_path).exists()):
        strong_path = row["hrl_path"]
    hdr = fits.getheader(strong_path)
    pix_as = abs(float(hdr["CDELT1"])) * 3600.0
    beam_as = float(hdr["BMAJ"]) * 3600.0

    note = ""
    try:
        bmaj_pix = beam_fwhm_pix_of(hdr)
        bmin_pix = float(hdr["BMIN"]) / abs(float(hdr["CDELT1"]))
        beam_area = 1.1331 * bmaj_pix * bmin_pix
        mask, _, _, n_kept, _ = build_mask(mom0, finite2d, bmaj_pix,
                                           beam_area, S_FROZEN)
        if not mask.any():
            note, mask = "no-seed", None
    except Exception as e:
        note, mask = f"mask-fail {str(e)[:18]}", None

    mom0c = mom0[crop]
    stride = max(1, int(np.ceil(max(mom0c.shape) / MAX_DISP)))
    disp = mom0c[::stride, ::stride]
    vmax = np.nanpercentile(np.abs(disp[np.isfinite(disp)]), 99.7) \
        if np.isfinite(disp).any() else 1.0
    ax.imshow(disp, origin="lower", cmap=CMAP_MOMENT0, vmin=-vmax, vmax=vmax,
              interpolation="nearest")
    if mask is not None:
        ax.contour(mask[crop][::stride, ::stride], levels=[0.5],
                   colors="lime", linewidths=0.7)

    # beam circle (bottom left) + 30" scale bar (bottom right)
    ny, nx = disp.shape
    b_r = 0.5 * beam_as / (pix_as * stride)
    ax.add_patch(Circle((0.07 * nx, 0.08 * ny), max(b_r, 1.0), fill=False,
                        color="black", lw=0.9))
    # adaptive scale bar: largest ladder step fitting 45% of the panel
    # (a fixed 30″ bar overflows tiny-pixel high-res maps — NGC 1377 —
    # and autoscale then blows up the axes limits)
    z = float(row["z"])
    for bar_as in (60.0, 30.0, 10.0, 5.0, 2.0, 1.0, 0.5):
        bar = bar_as / (pix_as * stride)
        if bar <= 0.45 * nx:
            break
    # Use the distance the thesis adopts, not cz/H0.  Below ~10 Mpc peculiar
    # velocities are comparable to the Hubble flow, and the flow value put
    # NGC 4945 at 8.2 Mpc against its adopted 3.65 Mpc (TRGB), so the bar
    # implied 23 pc for a beam the text reports at 10 pc (2026-08-26).
    d_mpc = ADOPTED_D.get(name)
    if d_mpc is None:
        d_mpc = C_KMS * z / H0 if z >= 0.0008 else None
    amt = f'{bar_as:g}″'
    if d_mpc is None:
        lab = amt
    else:
        kpc = np.radians(bar_as / 3600.0) * d_mpc * 1e3
        lab = f'{amt} ≈ {kpc:.2f} kpc' if kpc < 1 else f'{amt} ≈ {kpc:.1f} kpc'
    ax.plot([0.95 * nx - bar, 0.95 * nx], [0.06 * ny, 0.06 * ny],
            color="black", lw=1.6)
    ax.text(0.95 * nx - bar / 2, 0.09 * ny, lab, ha="center", va="bottom",
            fontsize=16, color="black")
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    # No batch marker: "wave-1" is an internal ingestion label with no meaning
    # to a reader, and colour-coding it invited a caption entry explaining a
    # distinction the thesis never uses (2026-08-26).
    from uniform_batch_stage4_plot import display_name
    LINE_PRETTY = {"CO21": "CO(2-1)", "CS21": "CS(2-1)", "HCO+10": "HCO+(1-0)"}
    l2 = f"{LINE_PRETTY.get(row['strong_line'], row['strong_line'])}  {beam_as:.2f}″"
    if note:
        l2 += f"  [{note}]"
    ax.set_title(f"{display_name(name)}\n{l2}", fontsize=20, color="black")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    setup_thesis_style()
    rows = [r for r in build_table()
            if (OUT / f"{r['galaxy']}_strong_mom0.npy").exists()]
    rows.sort(key=lambda r: r["galaxy"])
    ncol = 7
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.3 * nrow))
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, row in zip(axes.flat, rows):
        ax.set_axis_on()
        print(f"  {row['galaxy']}", flush=True)
        try:
            panel(ax, row)
        except Exception as e:
            import traceback; traceback.print_exc()
            ax.text(0.5, 0.5, f"{row['galaxy']}\nERROR {str(e)[:30]}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8)
    # Plain description only: the old title carried three internal terms
    # ("frozen mask", "lime", "wave-1"), an em dash the thesis style forbids,
    # and a fixed "30″ bar" that four different bar lengths contradict.
    fig.suptitle("Bright-tracer moment-0 maps: integration aperture (green), "
                 "beam (circle), scale bar per panel", fontsize=24, y=1.004)
    plt.tight_layout()
    out = OUT / "co_mom0_gallery.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n{len(rows)} panels → {out}", flush=True)


if __name__ == "__main__":
    main()
