"""Per-source table of the correlation-diagram quantities (thesis appendix).

Reuses stage-4's own loaders so every number is byte-identical to what the
figure plots: x = L_TIR(total) x f8 (mask-level apportioning), y = n^6 L_HRL
(detections/marginals at F, upper limits at 3 sigma_F). Emits
source_table.csv + source_table_rows.tex under result_v2/_uniform_batch/ and
prints the five-source anchor stats (must reproduce the thesis numbers), the
distant-ULIRG offsets below the Bittner 2022 extrapolation, and the axis
spans backing the abstract's dynamic-range claim.

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_source_table.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import uniform_batch_stage4_plot as s4                          # noqa: E402


def latex_name(g):
    d = s4.display_name(g)
    if g.startswith("ic"):
        d = "IC " + g[2:]
    return d.replace(" ", "~", 1)


def latex_line(l):
    return {"H30a": r"H30$\alpha$", "H40a": r"H40$\alpha$"}[l]


def main():
    rows = s4.load()
    lrows = [r for r in rows if np.isfinite(r["D"]) and r["D"] > 0
             and ("provisional" in r["dflag"] or r["dflag"] == "literature")]
    s5p = s4.OUT / "offset_investigation" / "s5_masklevel_x" / "s5_results.csv"
    frac8 = {r["galaxy"]: s4.fnum(r["frac8"]) for r in csv.DictReader(open(s5p))}
    brows = []
    for r in lrows:
        f8 = frac8.get(r["galaxy"], np.nan)
        if not np.isfinite(f8):
            continue
        d2 = 4 * np.pi * (r["D"] * s4.MPC_M) ** 2
        r = dict(r)
        r["f8"] = f8
        r["L_tir"] = (r["fir_central"] / r["frac"]) * d2 / s4.LSUN_W * f8
        conv = s4.line_lum_conv(r) * s4.N6[r["line"]]
        r["logx"] = np.log10(r["L_tir"])
        sn = r["sn"]
        r["tier"] = "det" if sn >= 5 else ("marg" if sn >= 3 else "ul")
        r["logy"] = np.log10((3.0 * r["sig"] if r["tier"] == "ul"
                              else r["F"]) * conv)
        r["off"] = r["logy"] - (s4.HIST_FIT[0] * r["logx"] + s4.HIST_FIT[1])
        brows.append(r)
    print(f"{len(brows)} plotted sources")

    # anchor: five-source stats must reproduce the thesis numbers
    five = [r for r in brows if r["sn"] >= 3]
    v = np.array([r["off"] for r in five])
    print(f"anchor five-source: mean {v.mean():+.2f} sd {v.std(ddof=1):.2f} "
          f"below {(v < 0).sum()}/{len(v)}")
    assert abs(v.mean() + 0.12) < 0.005 and abs(v.std(ddof=1) - 0.80) < 0.005 \
        and (v < 0).sum() == 2, "anchor mismatch - do not use this table"

    # distant ULIRG offsets (the nine at D >= 183 Mpc, all upper limits)
    nine = sorted([r for r in brows if r["D"] >= 183], key=lambda r: r["off"])
    print(f"\ndistant ULIRGs (D >= 183 Mpc): {len(nine)}")
    for r in nine:
        print(f"   {r['galaxy']:<18} D={r['D']:5.1f}  off={r['off']:+.2f} dex"
              f"  ({r['tier']})")
    offs = np.array([r["off"] for r in nine])
    print(f"   below extrapolation: {(offs < 0).sum()}/{len(nine)}; "
          f"range {offs.min():+.2f} to {offs.max():+.2f}; "
          f"deepest {offs.min():+.2f} dex")

    # axis spans over the plotted sample
    lx = np.array([r["logx"] for r in brows])
    ly = np.array([r["logy"] for r in brows])
    print(f"\nspans: x {lx.min():.2f}..{lx.max():.2f} ({lx.max()-lx.min():.2f} dex) | "
          f"y {ly.min():.2f}..{ly.max():.2f} ({ly.max()-ly.min():.2f} dex)")

    brows.sort(key=lambda r: -r["logx"])
    with open(s4.OUT / "source_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["galaxy", "line", "tier", "D_Mpc", "dist", "f8_percent",
                    "log_Ltir_ap", "log_n6Lhrl_or_3sig"])
        for r in brows:
            w.writerow([r["galaxy"], r["line"], r["tier"], f"{r['D']:.1f}",
                        r["dflag"], f"{100*r['f8']:.1f}", f"{r['logx']:.2f}",
                        f"{r['logy']:.2f}"])
    with open(s4.OUT / "source_table_rows.tex", "w") as fh:
        for r in brows:
            lim = "$<$" if r["tier"] == "ul" else ""
            fh.write(f"    {latex_name(r['galaxy'])} & {latex_line(r['line'])} & "
                     f"{r['D']:.1f} & {100*r['f8']:.1f} & {r['logx']:.2f} & "
                     f"{lim}{r['logy']:.2f} \\\\\n")
    print(f"\nwrote {s4.OUT/'source_table.csv'} + source_table_rows.tex")


if __name__ == "__main__":
    main()
