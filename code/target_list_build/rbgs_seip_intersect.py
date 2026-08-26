"""New-sample selection, stage 0: RBGS × DEC pre-filter × Spitzer SEIP.

Criteria stack v2 (decisions.md 2026-08-03 (2)):
  parent   = IRAS RBGS (f60 > 5.24 Jy by catalog definition; Sanders 2003)
  prefilter= DEC ≤ +40° (engineering: culmination elevation ≥ 27°)
  required = SEIP IRAC4 (8μm) coverage — mask-level apportioning resolution
  advisory = (later) WISE central-concentration L1 flag; is_mosaic hard cut
             happens at the alma-query stage (L2); stage-1 guards are L3.
  NOTE: no D cut, no beam_pc hard cut (v2 re-examination).

Output: result_v2/_new_sample/rbgs_seip_candidates.csv
Run: conda run -n casa_env python -u target_list_build/rbgs_seip_intersect.py
"""
from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from astroquery.vizier import Vizier
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
import astropy.units as u

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
RBGS_CAT = "J/AJ/126/1607"          # Sanders et al. 2003, RBGS
DEC_MAX = 40.0


def fetch_rbgs():
    v = Vizier(columns=["**"], row_limit=-1)
    tables = v.get_catalogs(RBGS_CAT)
    t = tables[0]
    print(f"RBGS table: {len(t)} rows; columns: {t.colnames[:12]}", flush=True)
    return t


def main():
    warnings.filterwarnings("ignore")
    OUT.mkdir(exist_ok=True)
    t = fetch_rbgs()
    cols = {c.lower(): c for c in t.colnames}

    def col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_name = col("Name", "IRAS", "SimbadName")
    c_ra = col("RAJ2000", "_RA")
    c_de = col("DEJ2000", "_DE")
    c_f60 = col("F60um", "S60um", "F60")
    c_cz = col("cz", "Vhel", "z")

    rows = []
    for r in t:
        try:
            ra = float(r[c_ra]) if not isinstance(r[c_ra], str) else \
                SkyCoord(f"{r[c_ra]} {r[c_de]}",
                         unit=(u.hourangle, u.deg)).ra.deg
            de = float(r[c_de]) if not isinstance(r[c_de], str) else \
                SkyCoord(f"{r[c_ra]} {r[c_de]}",
                         unit=(u.hourangle, u.deg)).dec.deg
        except Exception:
            continue
        if de > DEC_MAX:
            continue
        rows.append(dict(name=str(r[c_name]).strip(), ra=round(ra, 5),
                         dec=round(de, 5),
                         f60=float(r[c_f60]) if c_f60 else np.nan,
                         cz=float(r[c_cz]) if c_cz else np.nan))
    print(f"after DEC ≤ +{DEC_MAX:.0f}: {len(rows)} candidates", flush=True)

    out_csv = OUT / "rbgs_seip_candidates.csv"
    done = set()
    if out_csv.exists():
        done = {r["name"] for r in csv.DictReader(open(out_csv))}
        print(f"resuming — {len(done)} done", flush=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "ra", "dec", "f60", "cz",
                                          "seip_rows", "n_i4", "n_m1",
                                          "seip_tag"])
        if write_header:
            w.writeheader()
        for i, r in enumerate(rows):
            if r["name"] in done:
                continue
            try:
                q = Irsa.query_region(
                    SkyCoord(r["ra"] * u.deg, r["dec"] * u.deg),
                    catalog="slphotdr4", radius=1.0 * u.arcmin)
                n = len(q)
                i4 = int((q["i4_fluxtype"] > 0).sum()) if n else 0
                m1 = int((q["m1_fluxtype"] > 0).sum()) if n else 0
            except Exception as e:
                n, i4, m1 = -1, 0, 0
                print(f"  {r['name']}: query error {str(e)[:50]}", flush=True)
            tag = ("8um+24um" if i4 and m1 else "8um" if i4 else
                   "24um" if m1 else "imaged-no-phot" if n > 0 else
                   "query-error" if n < 0 else "NO-SPITZER")
            w.writerow(dict(**r, seip_rows=n, n_i4=i4, n_m1=m1, seip_tag=tag))
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  ... {i+1}/{len(rows)}", flush=True)
    print(f"done → {out_csv}", flush=True)


if __name__ == "__main__":
    main()
