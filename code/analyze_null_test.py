"""Compute effective 3σ threshold from null-test CSV and report per-galaxy stats.

Usage:
    /opt/anaconda3/envs/casa_env/bin/python master_thesis/my_code/analyze_null_test.py

Reads results/_nulltest/nulltest_samples.csv (produced by run_null_test.py).
Splits samples into THRESHOLD_GROUP (used to compute percentile) vs CONTROL_GROUP
(reported separately as health check, NOT mixed into the threshold statistic).

Outputs:
    - stdout: per-galaxy summary (n_offsets, max-S/N min/median/max, control flag)
    - stdout: aggregated 95th percentile across THRESHOLD samples = effective 3σ
    - results/_nulltest/null_distribution.png  (histogram of max-S/N)
    - results/_nulltest/summary.md             (machine-readable + markdown)
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent / "_step3"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import (  # noqa: E402
    FIG_DOUBLE, COLOR_THRESHOLD, COLOR_CONTROL, COLOR_P95, COLOR_NAIVE,
    setup_thesis_style, apply_clean_spines,
)

REPO = Path("/Volumes/HouAstro/master")
CSV = REPO / "results" / "_nulltest" / "nulltest_samples.csv"
OUT_DIR = REPO / "results" / "_nulltest"

THRESHOLD_GALAXIES = {"NGC6810", "NGC7130", "NGC3227",
                      "IRASF18293-3413", "IC5063", "NGC1386"}
CONTROL_GALAXIES = {"NGC3628", "NGC4945"}


def _galaxy_key(name: str) -> str:
    return name.replace(" ", "").upper()


def _group_for(galaxy_key: str) -> str:
    if galaxy_key in CONTROL_GALAXIES:
        return "control"
    if galaxy_key in THRESHOLD_GALAXIES:
        return "threshold"
    return "other"


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"CSV not found: {CSV}; run run_null_test.py first")

    rows = []
    with open(CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["max_sn"] = float(r["max_sn"])
            r["co_contamination"] = int(r["co_contamination"])
            r["galaxy_key"] = _galaxy_key(r["galaxy"])
            r["group"] = _group_for(r["galaxy_key"])
            rows.append(r)

    galaxies_present = sorted({r["galaxy_key"] for r in rows})
    print("=" * 70)
    print(f"Null-test samples loaded: {len(rows)} rows from "
          f"{len(galaxies_present)} galaxies")
    print("=" * 70)

    by_galaxy: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_galaxy[r["galaxy_key"]].append(r)

    print("\nPer-galaxy summary:")
    print(f"{'Galaxy':<22} {'Group':<10} {'N':>4} {'min':>6} {'med':>6} "
          f"{'max':>6} {'p95':>6} {'CO_contam':>9}")
    for gk in galaxies_present:
        sub = by_galaxy[gk]
        sn = np.array([r["max_sn"] for r in sub])
        co = sum(r["co_contamination"] for r in sub)
        g = sub[0]["group"]
        print(f"{gk:<22} {g:<10} {len(sub):>4} "
              f"{sn.min():>6.2f} {np.median(sn):>6.2f} {sn.max():>6.2f} "
              f"{np.percentile(sn, 95):>6.2f} {co:>9}")

    threshold_rows = [r for r in rows if r["group"] == "threshold"]
    control_rows = [r for r in rows if r["group"] == "control"]
    if not threshold_rows:
        print("\nNo threshold-group samples; cannot compute effective 3σ.")
        return

    threshold_sn = np.array([r["max_sn"] for r in threshold_rows])
    control_sn = np.array([r["max_sn"] for r in control_rows])
    p95 = float(np.percentile(threshold_sn, 95))
    p99 = float(np.percentile(threshold_sn, 99))
    threshold_galaxies = {r["galaxy_key"] for r in threshold_rows}
    print("\n" + "=" * 70)
    print(f"THRESHOLD GROUP (n={len(threshold_rows)} samples, "
          f"{len(threshold_galaxies)} galaxies):")
    print(f"  95th percentile max-S/N = {p95:.2f}  ← effective 3σ")
    print(f"  99th percentile max-S/N = {p99:.2f}")
    print(f"  median max-S/N          = {np.median(threshold_sn):.2f}")
    print(f"  max max-S/N             = {threshold_sn.max():.2f}")

    print("\nCONTROL HEALTH CHECK:")
    for gk in sorted(CONTROL_GALAXIES):
        sub = by_galaxy.get(gk, [])
        if sub:
            sn = [r["max_sn"] for r in sub]
            print(f"  {gk}: max max-S/N over null offsets = {max(sn):.2f} "
                  f"(if ≫ {p95:.2f}, pipeline may have issue)")
        else:
            print(f"  {gk}: no null samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_thesis_style()

    # For the diagnostic histogram: drop CO-contaminated control samples (the
    # 6000+ S/N tail squashes the informative range) AND drop degenerate
    # S/N≈0 samples (where the offset pushed the noise window into a NaN
    # region of the NGC 4945 cube — not meaningful null draws).
    control_clean = np.array([
        r["max_sn"] for r in control_rows
        if (not r["co_contamination"]) and r["max_sn"] > 0.3
    ])

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    bin_edges = np.linspace(0, max(5.0, threshold_sn.max() * 1.05), 26)

    ax.hist(
        threshold_sn, bins=bin_edges,
        color=COLOR_THRESHOLD, edgecolor="white", linewidth=0.5,
        alpha=0.85, label=f"threshold group ({len(threshold_sn)} null samples)",
    )
    if len(control_clean):
        ax.hist(
            control_clean, bins=bin_edges,
            color=COLOR_CONTROL, edgecolor="white", linewidth=0.5,
            alpha=0.55,
            label=f"controls, CO-clean ({len(control_clean)}; excluded from $p_{{95}}$)",
        )

    ax.axvline(p95, color=COLOR_P95, ls="--", lw=1.2,
               label=rf"$p_{{95}}={p95:.2f}$ (effective 3$\sigma$)")
    ax.axvline(p99, color=COLOR_P95, ls=(0, (1, 2)), lw=1.0, alpha=0.75,
               label=rf"$p_{{99}}={p99:.2f}$")
    ax.axvline(3.0, color=COLOR_NAIVE, ls=":", lw=1.0,
               label=r"naïve 3$\sigma$ (uncorrected)")

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_xlabel(r"max $S/N$ over 828-combination pipeline scan")
    ax.set_ylabel("number of offsets")
    ax.legend(loc="upper right", frameon=False)
    apply_clean_spines(ax)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"null_distribution.{ext}")
    plt.close(fig)
    print(f"\nWrote {OUT_DIR/'null_distribution.png'} + .pdf")

    summary = OUT_DIR / "summary.md"
    summary.write_text(
        f"# Null-test summary\n\n"
        f"- Samples: {len(rows)} ({len(threshold_rows)} threshold + "
        f"{len(control_rows)} control)\n"
        f"- Threshold galaxies: {sorted(THRESHOLD_GALAXIES)}\n"
        f"- Control galaxies: {sorted(CONTROL_GALAXIES)}\n\n"
        f"## Effective 3σ\n\n"
        f"| Statistic | Value |\n|---|---|\n"
        f"| 95th percentile max-S/N (effective 3σ) | **{p95:.2f}** |\n"
        f"| 99th percentile max-S/N | {p99:.2f} |\n"
        f"| Median max-S/N | {np.median(threshold_sn):.2f} |\n\n"
        f"See `null_distribution.png` and `nulltest_samples.csv`.\n"
    )
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
