"""Clean null-distribution histogram for the group-meeting slide.

Shows ONLY the non-detection ('threshold') galaxies' line-free-offset max-S/N
(pure noise), to make the look-elsewhere point: even pure noise reaches S/N
~3-4 over the 828-combination scan. Deliberately omits the superseded pooled
p95=3.08 / 'effective 3sigma' lines (the current criterion is a per-galaxy
null). Figure generator only; not part of the pipeline.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import setup_thesis_style, apply_clean_spines, COLOR_THRESHOLD, COLOR_NAIVE  # noqa

CSV = Path("/Volumes/HouAstro/master/results/_nulltest/nulltest_samples.csv")
OUT = Path("/Volumes/HouAstro/master/group_meeting_slides/assets/null_distribution.png")
THRESHOLD = {"NGC6810", "NGC7130", "NGC3227", "IRASF18293-3413", "IC5063", "NGC1386"}


def truthy(v):
    return str(v).strip() not in ("", "0", "0.0", "False", "false")


def main():
    setup_thesis_style()
    rows = list(csv.DictReader(open(CSV)))
    sn = np.array([
        float(r["max_sn"]) for r in rows
        if r["galaxy"].replace(" ", "") in THRESHOLD
        and not truthy(r.get("co_contamination", "0"))
        and float(r["max_sn"]) > 0.3
    ])
    print(f"non-detection null samples: {len(sn)}  "
          f"median {np.median(sn):.2f}  95th {np.percentile(sn,95):.2f}  max {sn.max():.2f}")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    edges = np.linspace(0, max(5.0, sn.max() * 1.05), 24)
    ax.hist(sn, bins=edges, color=COLOR_THRESHOLD, edgecolor="white", linewidth=0.5,
            alpha=0.9, label=f"line-free velocity offsets\n(pure noise, {len(sn)} scans)")
    ax.axvline(3.0, color=COLOR_NAIVE, ls=":", lw=1.2, label=r"nominal 3$\sigma$")

    ymax = ax.get_ylim()[1]
    ax.annotate("pure noise alone\nreaches S/N 3–4", xy=(3.15, ymax * 0.55),
                fontsize=10.5, color="#444", ha="left")
    ax.annotate("real detections: S/N 8–24  (far off-scale →)", xy=(0.35, ymax * 0.95),
                fontsize=10.5, color="#7a1f12", ha="left", fontweight="bold")

    ax.set_xlim(0, edges[-1])
    ax.set_xlabel(r"max $S/N$ over the 828-combination scan (line-free velocities)")
    ax.set_ylabel("number of offsets")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    apply_clean_spines(ax)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
