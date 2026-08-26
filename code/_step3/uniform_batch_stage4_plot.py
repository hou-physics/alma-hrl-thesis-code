"""Uniform batch — STAGE 4 (mainline D): the IR-vs-HRL correlation plot, v1.

Panel A (headline, distance-free): F_HRL [Jy km/s] vs aperture-matched
F_TIR [W/m2] — both fluxes scale as D^-2, so distances cancel exactly and
the provisional cz/70 issue does not touch this panel.
Panel B: luminosity version with cz/70 distances (PROVISIONAL watermark;
NEEDS-LIT-DISTANCE sources dropped).

Per Frank's plotting prescription (2026-07-10): EVERY source is plotted —
detections as filled markers with 1σ bars; S/N < 3 as flux + 1σ bar +
downward arrow (F ≤ 0.5σ drawn at the σ floor, hollow). H30α and H40α get
distinct hue+shape (colorblind triple-coding rule). x-axis apportioning:
F_TIR(central) = F_TIR(IRAS global) × min(cov_corr, 1); covered sources → 1.

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_stage4_plot.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup_thesis_style, apply_clean_spines  # noqa: E402

# Global sigma_F calibration (decisions.md 2026-08-22): the signal-free
# population shows the white-noise sigma_F underestimates the windowed-
# integral uncertainty by x1.5; all plotted sigmas/tiers are calibrated.
SIGMA_CAL = 1.5

DISPLAY_NAMES = {"eso097-013": "Circinus", "he2-10": "He 2-10"}


def display_name(g):
    if g in DISPLAY_NAMES:
        return DISPLAY_NAMES[g]
    if g.startswith("ngc"):
        return "NGC " + g[3:]
    if g.startswith("iras"):
        return "IRAS " + g[4:].upper()
    return g

from uniform_batch_configs import build_table                  # noqa: E402
from uniform_batch_stage1 import REST_HZ                       # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
SM96 = dict(w12=13.48, w25=5.16, w60=2.58, w100=1.00)   # Sanders & Mirabel 1996
MPC_M = 3.0857e22
LSUN_W = 3.828e26
C_KMS = 299792.458

LINE_STYLE = {"H30a": dict(color="#1f77b4", marker="o", label="H30α"),
              "H40a": dict(color="#d62728", marker="s", label="H40α")}
# group convention (Toma/Badescu, via Lau Fig. 1.1; Gordon & Sorochenko 2012):
# I_n ∝ n^-6, so lines are multiplied by n^6 to share one scale
N6 = {"H30a": 30.0 ** 6, "H40a": 40.0 ** 6}

# Previous-works points, DIGITIZED BY EYE from Bittner (2022) Fig. 5.2
# (right panel): Lau sample (blue), literature compilation (red), Badescu
# in prep. (gray). Approximate to ~±0.1 dex — REPLACE with table values via
# /paper-query before thesis use. (x, y) in log10; ul = upper limit.
HIST_DIGITIZED = [
    # --- Lau sample ---
    (7.90, 9.62, True), (8.45, 9.50, True), (8.70, 9.95, True),
    (9.00, 10.00, True), (8.95, 10.15, False), (9.10, 10.10, False),
    (9.20, 9.92, True), (9.28, 9.93, True), (9.95, 10.55, False),
    (10.75, 11.75, True), (11.15, 11.97, False), (11.55, 11.73, True),
    (11.72, 12.63, True), (11.90, 12.68, True),
    # --- literature compilation ---
    (9.62, 10.65, True), (9.66, 10.25, True), (10.05, 10.35, True),
    (10.10, 10.85, True), (10.63, 11.50, False), (10.68, 11.55, True),
    (11.10, 11.75, True), (11.40, 12.30, False), (11.43, 12.18, True),
    # --- Badescu (in prep.) ---
    (9.45, 11.35, False), (9.70, 10.80, False), (9.88, 11.00, False),
    (9.97, 11.00, False), (9.95, 11.20, False), (9.93, 11.60, True),
    (10.00, 10.60, False), (10.42, 12.10, True), (10.52, 12.45, True),
    (11.58, 13.45, True), (11.65, 13.50, False), (11.70, 13.40, True),
]
HIST_FIT = (1.234, -1.30)   # Bittner 2022 Fig 5.2 linear fit (log-log)


def draw_previous_works(ax):
    for lx, ly, ul in HIST_DIGITIZED:
        x, y = 10 ** lx, 10 ** ly
        ax.plot(x, y, "x", color="#999999", ms=7, mew=1.6, zorder=1)
        if ul:
            ax.annotate("", xy=(x, y / 1.8), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                        color="#999999", alpha=0.8))
    xs = np.array([2e8, 3e12])
    ax.plot(xs, 10 ** (HIST_FIT[0] * np.log10(xs) + HIST_FIT[1]),
            "-", color="#444444", lw=1.2, zorder=1)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    s2 = {r["galaxy"]: r for r in csv.DictReader(open(OUT / "stage2_flux.csv"))}
    s3 = {r["galaxy"]: r for r in csv.DictReader(open(OUT / "stage3_ir.csv"))}
    # Guard (decisions.md 2026-08-22 (5)): a stage-3 rerun resurrects the
    # PSC/FSC point-source fluxes, which sit up to 0.2 dex below the RBGS
    # total photometry for extended nearby galaxies. Refuse to plot until
    # the RBGS patch has been re-applied.
    if "iras_src" not in next(iter(s3.values())):
        sys.exit("stage3_ir.csv has no iras_src column — stage 3 was rerun "
                 "without re-applying uniform_batch_rbgs_flux_patch.py "
                 "(decisions.md 2026-08-22 (5)). Run the patch, then plot.")
    rows = []
    for g, r2 in s2.items():
        if r2.get("verdict") in ("review", "error"):
            continue
        r3 = s3.get(g)
        if not r3:
            continue
        f60 = fnum(r3["iras_f60"])
        if not np.isfinite(f60):
            continue
        fir = 1.8e-14 * (SM96["w12"] * (fnum(r3["iras_f12"]) or 0)
                         + SM96["w25"] * (fnum(r3["iras_f25"]) or 0)
                         + SM96["w60"] * f60
                         + SM96["w100"] * (fnum(r3["iras_f100"]) or 0))
        cov = r3["coverage"]
        frac = 1.0 if cov.startswith("covered") else \
            min(fnum(r3["cov_corr"]), 1.0)
        if not np.isfinite(frac):
            continue
        rows.append(dict(
            galaxy=g, line=r2["hrl_line"],
            F=fnum(r2["F"]), sig=fnum(r2["sigma_F"]) * SIGMA_CAL,
            sn=fnum(r2["SN"]) / SIGMA_CAL,
            verdict=r2["verdict"], coverage=cov,
            fir_central=fir * frac, frac=frac,
            D=fnum(r3["D_Mpc_cz"]), dflag=r3["dist_flag"],
        ))
    zmap = {r["galaxy"]: fnum(r["z"]) for r in build_table()}
    errp = OUT / "iras_errors.csv"
    emap = {r["galaxy"]: fnum(r["rel_fir"])
            for r in csv.DictReader(open(errp))} if errp.exists() else {}
    # adopted literature distances (2026-08-08): nearby sources override
    # cz/70 (NED-D best-method medians, refcodes in adopted_distances.csv);
    # distant sources absent from the file keep cz/70
    adp = OUT / "adopted_distances.csv"
    dmap = {r["galaxy"]: fnum(r["D_Mpc"])
            for r in csv.DictReader(open(adp))} if adp.exists() else {}
    for r in rows:
        if r["galaxy"] in dmap:
            r["D"] = dmap[r["galaxy"]]
            r["dflag"] = "literature"
        r["z"] = zmap.get(r["galaxy"], np.nan)
        # x-error in dex: IRAS flux propagation (Lau App. D) ⊕ 0.2 dex
        # apportioning systematic for partial-coverage sources (Lau's value;
        # to be updated with the Guliyeva 24μm band scatter). Distance error
        # joins with the literature distance table.
        rel = emap.get(r["galaxy"], np.nan)
        meas_dex = np.log10(1 + rel) if np.isfinite(rel) else 0.0
        # The apportioning systematic applies to every source whose x-axis is
        # scaled by the mask-level 8um fraction — which the thesis panel does
        # for all of them.  The old exemption keyed on the stage-3 coverage
        # flag and wrongly dropped the bar on He 2-10 (2026-08-26, G-2).
        app_dex = 0.2
        r["xdex"] = float(np.hypot(meas_dex, app_dex))
    return rows


def line_lum_conv(r):
    """Jy km/s → line luminosity in L_sun (predecessors' axis convention):
    F[W/m2] = F[Jy km/s] × 1e-26 × ν_obs/c[km/s];  L = 4πD² F / L_sun."""
    nu_obs = REST_HZ[r["line"]] / (1.0 + r["z"])
    fwm2_per_jykms = 1e-26 * nu_obs / C_KMS
    area = 4 * np.pi * (r["D"] * MPC_M) ** 2
    return fwm2_per_jykms * area / LSUN_W


def draw(ax, rows, xkey, ykey_scale):
    for r in rows:
        st = LINE_STYLE[r["line"]]
        x = r[xkey]
        F, sig = r["F"] * ykey_scale(r), r["sig"] * ykey_scale(r)
        # three-tier labeling (user decision 2026-08-03; boundaries are the
        # field convention also used in the §X.4 pre-registration):
        # S/N ≥ 5 detected · 3–5 marginal · < 3 upper limit
        sn = r.get("sn", np.nan)
        tier = "det" if sn >= 5 else ("marg" if sn >= 3 else "ul")
        # Upper limits are drawn AT the 3 sigma_F value the text and Table 5.1
        # quote, with no error bar of their own (final review 2026-08-26, U-1:
        # the old placement at the measured flux made "the plotted 3 sigma
        # limits" in ch6 literally false, and put ten negative-flux sources at
        # unexplained positive y).
        y = 3.0 * sig if tier == "ul" else F
        if y <= 0 or x <= 0:
            continue
        # det/marg: full ±1σ bar; UL: bare 3σ point + downward arrow
        yerr = sig if tier != "ul" else None
        dex = r.get("xdex", 0.0)
        xerr = np.array([[x * (1 - 10 ** -dex)], [x * (10 ** dex - 1)]]) \
            if dex > 0 else None
        mfc = {"det": st["color"], "marg": st["color"], "ul": "none"}[tier]
        alpha = 0.45 if tier == "marg" else 0.95
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=st["marker"],
                    color=st["color"], mfc=mfc,
                    ms=7 if tier == "det" else 6.5, capsize=2, lw=1.0,
                    alpha=alpha, zorder=3 if tier == "det" else 2)
        if tier == "ul":
            ax.annotate("", xy=(x, y / 2.2), xytext=(x, y),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                        color=st["color"], alpha=0.8))
        if tier in ("det", "marg"):
            ax.annotate(display_name(r["galaxy"]), (x, y),
                        textcoords="offset points",
                        xytext=(6, 5), fontsize=7,
                        alpha=1.0 if tier == "det" else 0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    apply_clean_spines(ax)


def main():
    setup_thesis_style()
    rows = load()
    print(f"{len(rows)} sources plotted "
          f"({sum(r['verdict']=='detection' for r in rows)} det)")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    # Panel A — flux vs flux (distance-free; internal diagnostic, not a
    # thesis figure per decisions.md 2026-08-02 (2))
    ax = axes[0]
    draw(ax, rows, "fir_central", lambda r: N6[r["line"]] / N6["H30a"])
    ax.set_xlabel(r"$F_\mathrm{TIR}$ (aperture-matched) [W m$^{-2}$]")
    ax.set_ylabel(r"$n^6$-scaled $F_\mathrm{HRL}$ [Jy km s$^{-1}$, H30α-equiv.]")
    ax.set_title("Flux–flux (distance-free, internal check)")

    # Panel B — luminosity (provisional cz/70 distances)
    ax = axes[1]
    lrows = [r for r in rows if np.isfinite(r["D"]) and r["D"] > 0
             and ("provisional" in r["dflag"] or r["dflag"] == "literature")]
    for r in lrows:
        d2 = 4 * np.pi * (r["D"] * MPC_M) ** 2
        r["L_tir"] = r["fir_central"] * d2 / LSUN_W
    # y = n^6 × L(HRL)/L_sun — predecessors' axis (Lau Fig 1.1 / Bittner Fig 5.2)
    draw(ax, lrows, "L_tir", lambda r: line_lum_conv(r) * N6[r["line"]])
    ax.set_xlabel(r"$L_\mathrm{TIR}$ (aperture-matched) [$L_\odot$]")
    ax.set_ylabel(r"$n^6\,L_\mathrm{HRL}$ [$L_\odot$]")
    n_lit = sum(r["dflag"] == "literature" for r in lrows)
    ax.set_title(f"Luminosity — THESIS PANEL (D: literature n={n_lit} + "
                 f"cz/70 far n={len(lrows)-n_lit})")

    handles = [plt.Line2D([], [], ls="", **LINE_STYLE[k]) for k in LINE_STYLE]
    axes[0].legend(handles=handles, loc="upper left", fontsize=9)
    fig.suptitle("HRL vs aperture-matched TIR — uniform survey v1 "
                 "(filled = detection, arrow = upper limit)", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"correlation_v1.{ext}", dpi=200)
    print(f"saved {OUT / 'correlation_v1.png'} (internal dual-panel)")
    plt.close(fig)

    # thesis figure: luminosity panel ONLY (decisions.md 2026-08-02 (2)).
    # x = MASK-LEVEL 8um apportioning (Option B, the Lau/Bittner aperture;
    # decisions 2026-08-21): sources without an 8um fraction are excluded
    # from the diagram (no mixed apertures), they stay in all tables.
    s5p = OUT / "offset_investigation" / "s5_masklevel_x" / "s5_results.csv"
    frac8 = {r["galaxy"]: fnum(r["frac8"])
             for r in csv.DictReader(open(s5p))} if s5p.exists() else {}
    brows = []
    for r in lrows:
        f8 = frac8.get(r["galaxy"], np.nan)
        if not np.isfinite(f8):
            continue
        d2 = 4 * np.pi * (r["D"] * MPC_M) ** 2
        r = dict(r)
        r["L_tir"] = (r["fir_central"] / r["frac"]) * d2 / LSUN_W * f8
        brows.append(r)
    print(f"thesis panel (mask-level x): {len(brows)} sources "
          f"({len(lrows) - len(brows)} without 8um fraction excluded)")
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    draw_previous_works(ax)
    draw(ax, brows, "L_tir", lambda r: line_lum_conv(r) * N6[r["line"]])
    # Not "CO-aperture": the aperture follows whichever bright tracer the
    # pairing rule selected, and for three of the measured sources that is
    # CS(2-1) rather than CO.
    ax.set_xlabel(r"$L_\mathrm{TIR}$ (aperture-matched, 8 $\mu$m) "
                  r"[$L_\odot$]")
    ax.set_ylabel(r"$n^6\,L_\mathrm{HRL}$ [$L_\odot$]")
    hist_handles = handles + [
        plt.Line2D([], [], ls="", marker="x", color="#999999", mew=1.6,
                   label="previous works (digitized from Bittner 2022)"),
        plt.Line2D([], [], ls="-", color="#444444",
                   label="linear fit, Bittner 2022"),
    ]
    ax.legend(handles=hist_handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"correlation_thesis_v1.{ext}", dpi=200)
    print(f"saved {OUT / 'correlation_thesis_v1.png'} (thesis single-panel)")

    # Vertical offsets of the brightest sources from the Bittner 2022 fit.
    # ch5, ch6 and ch7 all quote the mean, the scatter and the two extremes,
    # and until 2026-08-26 no product recomputed them: the numbers lived only
    # in the prose.  Printed here so a rerun re-derives what the text claims.
    bright = sorted([r for r in brows if r["sn"] >= 2.5],
                    key=lambda r: -r["sn"])[:6]
    if bright:
        offs = []
        for r in bright:
            y = line_lum_conv(r) * N6[r["line"]] * r["F"]
            pred = HIST_FIT[0] * np.log10(r["L_tir"]) + HIST_FIT[1]
            offs.append((np.log10(y) - pred, r["galaxy"]))
        v = np.array([o for o, _ in offs])
        print(f"\nvertical offsets from the Bittner 2022 fit, "
              f"{len(offs)} brightest sources (calibrated S/N >= 2.5):")
        for o, g in offs:
            print(f"   {g:<16}{o:+.2f} dex")
        lo, hi = min(offs), max(offs)
        print(f"   mean {v.mean():+.2f} | scatter (sd) {v.std(ddof=1):.2f} | "
              f"below fit {int((v < 0).sum())}/{len(v)}")
        print(f"   highest {hi[0]:+.2f} ({hi[1]})  lowest {lo[0]:+.2f} ({lo[1]})")


if __name__ == "__main__":
    main()
