"""Literature distance table — step 2: adopt (user-approved 2026-08-08).

From distance_candidates.csv (full NED-D audit trail) take, per galaxy,
the MEDIAN of the best-ranked method (TRGB > Cepheids > SBF > SNIa >
Tully-Fisher > ...) with the most recent refcode of that method as the
representative citation. Writes adopted_distances.csv, consumed by
uniform_batch_stage4_plot.load(). Distant sources are absent → cz/70.

Run: conda run -n casa_env python -u _step3/distance_adopt.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from distance_table_fetch import METHOD_RANK, rank          # noqa: E402

BATCH = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")


def main():
    rows = list(csv.DictReader(open(BATCH / "distance_candidates.csv")))
    by_gal = {}
    for r in rows:
        by_gal.setdefault(r["galaxy"], {}).setdefault(
            r["method"].strip(), []).append((float(r["D_Mpc"]), r["refcode"]))
    out = []
    for g, methods in sorted(by_gal.items()):
        best = sorted(methods, key=rank)[0]
        vals = methods[best]
        med = float(np.median([v for v, _ in vals]))
        ref = sorted(vals, key=lambda t: t[1])[-1][1]   # most recent bibcode
        out.append(dict(galaxy=g, D_Mpc=round(med, 2), method=best,
                        n_meas=len(vals), refcode=ref,
                        dist_flag="literature"))
        print(f"  {g:<14} {med:7.2f} Mpc  {best:<14} (n={len(vals)})  {ref}")
    with open(BATCH / "adopted_distances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["galaxy", "D_Mpc", "method",
                                          "n_meas", "refcode", "dist_flag"])
        w.writeheader(); w.writerows(out)
    print(f"\n{len(out)} adopted → {BATCH / 'adopted_distances.csv'}")


if __name__ == "__main__":
    main()
