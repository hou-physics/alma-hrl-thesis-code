"""New-sample wave-1 shortlist: real download sizes + budget packing.

From alma_scout.csv hits: exclude the existing sample by COORDINATE match,
require best resolution ≤ 8″ (drops TP/coarse-ACA-only targets), rank by
archive 10 km/s sensitivity (deepest first), take top N=30, fetch the TRUE
per-target download size via the archive file-list interface
(Alma.get_data_info on member OUS uids of qualifying observations), then
greedily pack a 400 GB wave-1 budget.

Output: result_v2/_new_sample/wave1_shortlist.csv (+ printed proposal)
Run: conda run -n casa_env python -u target_list_build/wave1_shortlist.py
"""
from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent.parent / "_step3"))
from uniform_batch_configs import build_table          # noqa: E402
from uniform_batch_stage3_ir import alma_geometry      # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
C_KMS = 299792.458
TOP_N = 30
BUDGET_GB = 400.0
RES_MAX = 8.0

HRL_GHZ = {"H30a": 231.900928, "H40a": 99.022952}
TRACER_GHZ = {"CO21": 230.538, "CO10": 115.271202, "CS21": 97.980953,
              "HCN10": 88.631847, "HCO+10": 89.188523, "13CO21": 220.398684}
WIN_RE = re.compile(r"(\d+\.\d+)\.\.(\d+\.\d+)GHz")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def existing_coords():
    coords = []
    for row in build_table():
        try:
            ra, dec, *_ = alma_geometry(row)
            coords.append(SkyCoord(ra * u.deg, dec * u.deg))
        except Exception:
            pass
    return coords


def good_obs_uids(ra, dec, z):
    """Re-run the scout match, return member-OUS uids + finest resolution."""
    from astroquery.alma import Alma
    res = Alma.query_region(SkyCoord(ra * u.deg, dec * u.deg),
                            radius=1.5 * u.arcmin)
    cols = {c.lower(): c for c in res.colnames}

    def get(rr, *names, default=""):
        for n in names:
            if n.lower() in cols:
                try:
                    return rr[cols[n.lower()]]
                except Exception:
                    pass
        return default
    uids, resns = set(), []
    for rr in res:
        if str(get(rr, "data_rights")).lower().startswith("propr"):
            continue
        if str(get(rr, "is_mosaic")).strip().upper() == "T":
            continue
        sr = fnum(get(rr, "spatial_resolution", "ang_res_arcsec",
                      default=np.nan))
        if not np.isfinite(sr) or sr > RES_MAX or sr < 0.05:
            continue
        wins = [(float(a), float(b)) for a, b in
                WIN_RE.findall(str(get(rr, "frequency_support")))]
        h = [k for k, nu in HRL_GHZ.items()
             if any(lo + .05 <= nu / (1 + z) <= hi - .05 for lo, hi in wins)]
        t = [k for k, nu in TRACER_GHZ.items()
             if any(lo + .05 <= nu / (1 + z) <= hi - .05 for lo, hi in wins)]
        if h and t:
            uid = str(get(rr, "member_ous_uid", "member_ous_id"))
            if uid:
                uids.add(uid)
                resns.append(sr)
    return uids, (min(resns) if resns else np.nan)


def true_size_gb(uids):
    from astroquery.alma import Alma
    total = 0.0
    for uid in uids:
        try:
            info = Alma.get_data_info(uid, expand_tarfiles=False)
            sizes = [fnum(x) for x in info["content_length"]]
            total += sum(s for s in sizes if np.isfinite(s))
        except Exception as e:
            print(f"    size query failed {uid}: {str(e)[:40]}", flush=True)
            return np.nan
    return total / 1e9


def main():
    warnings.filterwarnings("ignore")
    hits = [r for r in csv.DictReader(open(OUT / "alma_scout.csv"))
            if r["n_good"] and int(r["n_good"]) > 0]
    ex = existing_coords()
    fresh = []
    for r in hits:
        c = SkyCoord(fnum(r["ra"]) * u.deg, fnum(r["dec"]) * u.deg)
        if any(c.separation(e).arcmin < 2.0 for e in ex):
            continue
        if not np.isfinite(fnum(r["best_res_arcsec"])) or \
                fnum(r["best_res_arcsec"]) > RES_MAX:
            continue
        if not np.isfinite(fnum(r["sens10_mJy"])):
            continue
        fresh.append(r)
    fresh.sort(key=lambda r: fnum(r["sens10_mJy"]))
    short = fresh[:TOP_N]
    print(f"hits={len(hits)}  new+res-ok={len(fresh)}  sizing top {len(short)}",
          flush=True)

    rows = []
    for r in short:
        z = fnum(r["z"])
        try:
            uids, best = good_obs_uids(fnum(r["ra"]), fnum(r["dec"]), z)
            gb = true_size_gb(uids) if uids else np.nan
        except Exception as e:
            print(f"  {r['name']}: ERROR {str(e)[:50]}", flush=True)
            uids, best, gb = set(), np.nan, np.nan
        rows.append(dict(name=r["name"], z=z, sens10=fnum(r["sens10_mJy"]),
                         hrl=r["hrl_lines"], tracers=r["tracers"],
                         res=best, n_uids=len(uids),
                         size_GB=round(gb, 1) if np.isfinite(gb) else np.nan))
        print(f"  {r['name']:<20} sens10={r['sens10_mJy']:<7} "
              f"uids={len(uids)}  size={rows[-1]['size_GB']} GB", flush=True)

    # greedy packing: deepest-first, skip what doesn't fit
    wave, tot = [], 0.0
    for r in rows:
        if np.isfinite(r["size_GB"]) and tot + r["size_GB"] <= BUDGET_GB:
            wave.append(r["name"]); tot += r["size_GB"]
    for r in rows:
        r["wave1"] = r["name"] in wave

    with open(OUT / "wave1_shortlist.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWAVE 1 proposal: {len(wave)} targets, {tot:.0f} GB "
          f"(budget {BUDGET_GB:.0f})", flush=True)
    for n in wave:
        print("  " + n, flush=True)
    print(f"→ {OUT/'wave1_shortlist.csv'}", flush=True)


if __name__ == "__main__":
    main()
