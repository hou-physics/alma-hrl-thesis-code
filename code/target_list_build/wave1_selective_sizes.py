"""Wave-1 refinement: SELECTIVE download sizes (product cubes only).

For each top-30 candidate (wave1_shortlist.csv): expand the archive file
lists and sum ONLY the files our pipeline actually fetches —
`*.cube.I.pbcor.fits` and `*.cube.I.pb.fits.gz` (non-pbcor is reconstructed
locally from these two). Targets whose observations expose NO product cubes
(raw-only, pre-imaging-pipeline cycles) are flagged: they would require
manual CASA imaging and belong to a later wave.

Then repack the 400 GB wave-1 budget with the selective sizes
(depth-first order preserved).

Output: result_v2/_new_sample/wave1_selective.csv
Run: conda run -n casa_env python -u target_list_build/wave1_selective_sizes.py
"""
from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from wave1_shortlist import good_obs_uids, fnum        # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
BUDGET_GB = 400.0
PRODUCT_RE = re.compile(r"\.cube\.I\.pbcor\.fits$|\.cube\.I\.pb\.fits\.gz$")


def selective_size(uids):
    from astroquery.alma import Alma
    tot, nfits = 0.0, 0
    for uid in uids:
        try:
            info = Alma.get_data_info(uid, expand_tarfiles=True)
        except Exception as e:
            print(f"    expand failed {uid}: {str(e)[:40]}", flush=True)
            return np.nan, -1
        for row in info:
            url = str(row["access_url"])
            if PRODUCT_RE.search(url):
                s = fnum(row["content_length"])
                if np.isfinite(s):
                    tot += s
                    nfits += 1
    return tot / 1e9, nfits


def main():
    warnings.filterwarnings("ignore")
    short = list(csv.DictReader(open(OUT / "wave1_shortlist.csv")))
    rows = []
    for r in short:
        name, z = r["name"], fnum(r["z"])
        try:
            uids, best = good_obs_uids(
                *(fnum(x) for x in (
                    next(c["ra"] for c in csv.DictReader(
                        open(OUT / "alma_scout.csv")) if c["name"] == name),
                    next(c["dec"] for c in csv.DictReader(
                        open(OUT / "alma_scout.csv")) if c["name"] == name))),
                z)
            gb, nfits = selective_size(uids)
        except Exception as e:
            print(f"  {name}: ERROR {str(e)[:50]}", flush=True)
            uids, gb, nfits = set(), np.nan, -1
        raw_only = (nfits == 0)
        rows.append(dict(name=name, z=z, sens10=fnum(r["sens10"]),
                         hrl=r["hrl"], tracers=r["tracers"],
                         tarball_GB=fnum(r["size_GB"]),
                         selective_GB=round(gb, 1) if np.isfinite(gb) else "",
                         n_fits=nfits, raw_only=raw_only))
        tag = "RAW-ONLY (needs imaging)" if raw_only else \
            f"{rows[-1]['selective_GB']} GB in {nfits} FITS"
        print(f"  {name:<20} tarball={r['size_GB']:>7} GB → selective: {tag}",
              flush=True)

    wave, tot = [], 0.0
    for r in rows:
        gb = fnum(r["selective_GB"])
        if not r["raw_only"] and np.isfinite(gb) and tot + gb <= BUDGET_GB:
            wave.append(r["name"]); tot += gb
    for r in rows:
        r["wave1"] = r["name"] in wave

    with open(OUT / "wave1_selective.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWAVE 1 (selective sizes): {len(wave)} targets, {tot:.0f} GB "
          f"of {BUDGET_GB:.0f} GB budget", flush=True)
    for n in wave:
        print("  " + n, flush=True)
    nraw = sum(1 for r in rows if r["raw_only"])
    print(f"raw-only targets deferred: {nraw}", flush=True)
    print(f"→ {OUT/'wave1_selective.csv'}", flush=True)


if __name__ == "__main__":
    main()
