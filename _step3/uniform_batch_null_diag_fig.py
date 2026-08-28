"""Sample-level translated-aperture null diagnostics figure (thesis appendix A).

One panel: x = null_std / sigma_F (log), y = null_mean / sigma_F, one point
per measured source with valid nulls (37; IRAS 19542+1110 has none).
Shows (i) the generic off-center inflation of the null scatter even in
signal-free upper-limit fields (pbcor noise rises away from the field
center, so null_std is an upper bound on the at-source uncertainty), and
(ii) the qualitatively different, source-contaminated nulls around the
detections (significant negative means = sidelobe/bowl imprint).

Reads  result_v2/_uniform_batch/stage2_flux.csv
Writes result_v2/_uniform_batch/fig_null_diagnostics.png
       + copy to thesis/figures/fig_null_diagnostics.png
"""
import csv
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import setup_thesis_style, apply_clean_spines

ROOT = Path("/Volumes/HouAstro/master")
CSV = ROOT / "result_v2/_uniform_batch/stage2_flux.csv"

# See uniform_batch_master_table.py for the provenance of this factor.
SIGMA_CAL = 1.5
OUT = ROOT / "result_v2/_uniform_batch/fig_null_diagnostics.png"
THESIS = ROOT / "thesis/figures/fig_null_diagnostics.png"

DISPLAY = {"eso097-013": "Circinus", "he2-10": "He 2-10"}
ANNOTATE = {"ngc4826", "ngc3628", "ngc253", "ngc4945", "he2-10",
            "eso097-013", "ngc2369"}


def disp(name):
    if name in DISPLAY:
        return DISPLAY[name]
    if name.startswith("ngc"):
        return "NGC " + name[3:]
    if name.startswith("iras"):
        return "IRAS " + name[4:]
    return name


rows = []
for r in csv.DictReader(open(CSV)):
    try:
        sig = float(r["sigma_F"])
        ns = float(r["null_std"]) / sig
        nm = float(r["null_mean"]) / sig
        sn = float(r["SN"])
    except (ValueError, TypeError, ZeroDivisionError):
        continue
    # Tier on the CALIBRATED S/N, as the thesis does (sigma_F is scaled by
    # SIGMA_CAL downstream of stage2).  Cutting on the nominal value gave a
    # legend of 5 detections / 1 marginal against the thesis's 2 / 3.
    sn_cal = sn / SIGMA_CAL
    tier = ("detection" if sn_cal >= 5
            else ("marginal" if sn_cal >= 3 else "ul"))
    rows.append((r["galaxy"], tier, ns, nm, int(r["n_nulls"])))

ul = [t for t in rows if t[1] == "ul"]
det = [t for t in rows if t[1] == "detection"]
marg = [t for t in rows if t[1] == "marginal"]
print(f"{len(rows)} sources ({len(ul)} UL, {len(det)} det, {len(marg)} marg)")
ul_ns = np.array([t[2] for t in ul])
ul_nm = np.array([t[3] for t in ul])
print(f"UL null_std/sigF: median {np.median(ul_ns):.2f}, "
      f"range {ul_ns.min():.2f}-{ul_ns.max():.2f}")
print(f"UL null_mean/sigF: median {np.median(ul_nm):+.2f}, "
      f"RMS {np.sqrt(np.mean(ul_nm**2)):.2f}")
for g, v, ns, nm, k in det + marg:
    print(f"  {g:<12} {v:<10} std x{ns:5.2f}  mean {nm:+5.2f} sigF  "
          f"(K={k}, mean = {nm/(ns/np.sqrt(k)):+.1f} x its precision)")

setup_thesis_style()
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.axhline(0, color="#999999", lw=0.8, zorder=1)
ax.axvline(1, color="#999999", lw=0.8, ls="--", zorder=1)
ax.text(1.0, 1.9, "white noise at the\nsource position", fontsize=8,
        color="#777777", ha="center", va="top")

ax.plot([t[2] for t in ul], [t[3] for t in ul], "o", mfc="none",
        color="#2b6ca3", ms=6, mew=1.2,
        label=f"upper limits ({len(ul)})", zorder=3)
ax.plot([t[2] for t in det], [t[3] for t in det], "o", color="#c23b22",
        ms=7, label=f"detections ({len(det)})", zorder=4)
ax.plot([t[2] for t in marg], [t[3] for t in marg], "o", color="#c23b22",
        mfc="#f2b3ad", ms=7, label=f"marginal ({len(marg)})", zorder=4)

OFFSETS = {"ngc253": (6, -14), "ngc2369": (-8, -16), "eso097-013": (6, 9)}
for g, v, ns, nm, k in rows:
    if g not in ANNOTATE:
        continue
    dx, dy = OFFSETS.get(g, (6, 7))
    ax.annotate(disp(g), (ns, nm), textcoords="offset points",
                xytext=(dx, dy), fontsize=8.5,
                ha="right" if dx < 0 else "left")

ax.set_xscale("log")
ax.set_xlim(0.6, 20)
ax.set_ylim(-8.5, 2.5)
ax.set_xlabel(r"translated-aperture scatter / $\sigma_F$")
ax.set_ylabel(r"translated-aperture mean / $\sigma_F$")
ax.legend(loc="lower left", fontsize=9)
apply_clean_spines(ax)
plt.tight_layout()
plt.savefig(OUT, dpi=170)
shutil.copy(OUT, THESIS)
print(f"-> {OUT}\n-> {THESIS}")
