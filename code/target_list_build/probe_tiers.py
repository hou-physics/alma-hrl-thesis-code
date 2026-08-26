"""Probe ALMA tier (FITS / MS / ASDM) for each row in table_d_filtered.csv.

For each (galaxy, best-ranked MOUS) run probe_tiers to find out if FITS is
available (Cube=Y) or only MS (needs re-imaging). Adds a 'tier' column to
the CSV and writes table_d_with_tiers.csv, plus a shortlist of the
FITS-ready rows for the next wave of Phase Y analysis.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).parent
IN_CSV = HERE / "table_d_filtered.csv"
OUT_CSV = HERE / "table_d_with_tiers.csv"
CUBE_READY_MD = HERE / "cube_ready.md"

SKILL = Path("/Volumes/HouAstro/master/.claude/skills/alma-query")
sys.path.insert(0, str(SKILL))
from query import probe_tiers  # noqa: E402


def main() -> None:
    rows = list(csv.DictReader(open(IN_CSV)))
    print(f"Probing {len(rows)} galaxies...")
    for i, r in enumerate(rows, 1):
        try:
            tiers = probe_tiers(r["mousid"])
            r["tier"] = "|".join(sorted(tiers)) if tiers else ""
        except Exception as e:
            r["tier"] = f"ERR:{type(e).__name__}"
        print(f"  [{i:2d}/{len(rows)}] {r['name']:<18} tier={r['tier']}")

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}")

    # Shortlist: cube-ready candidates (FITS available) sorted by predicted S/N
    ready = [r for r in rows if "FITS" in r["tier"]]
    ready.sort(key=lambda r: -float(r["predicted_sn_log"]))
    lines = [
        "## Cube-ready shortlist (FITS tier available — no imaging needed)",
        "",
        f"Probed {len(rows)} Table D candidates; {len(ready)} have FITS cubes "
        "ready for immediate Phase Y analysis.",
        "",
        "| Rank | Galaxy | z | HRL | rms | predicted logS/N | Project | MOUSID |",
        "|-----:|--------|---|-----|-----|------------------|---------|--------|",
    ]
    for i, r in enumerate(ready, 1):
        lines.append(
            f"| {i} | {r['name']} | {float(r['z']):.4f} | {r['hrl_line']} | "
            f"{float(r['sens_10kms_mJy']):.2f} | {float(r['predicted_sn_log']):+.2f} | "
            f"{r['project_id']} | `{r['mousid']}` |"
        )
    lines.append("")
    CUBE_READY_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {CUBE_READY_MD}: {len(ready)} cube-ready candidates")


if __name__ == "__main__":
    main()
