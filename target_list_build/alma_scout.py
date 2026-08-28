"""New-sample selection, stage 0-c: ALMA archive scout over the candidate
pool (rbgs_seip_candidates.csv, 8μm-covered subset).

Per candidate (metadata only — nothing downloaded):
  query the ALMA science archive by position; keep PUBLIC, single-pointing
  (is_mosaic != T) observations whose SPW windows cover an HRL
  (H30α/H40α at the source redshift) AND a bright tracer line; record the
  lines hit, best angular resolution, archive-estimated download size, and
  the archive's own 10 km/s line sensitivity where present.

Output: result_v2/_new_sample/alma_scout.csv  (crash-safe, resumable)
Run: conda run -n casa_env python -u target_list_build/alma_scout.py
"""
from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
IN_CSV = OUT / "rbgs_seip_candidates.csv"
OUT_CSV = OUT / "alma_scout.csv"
C_KMS = 299792.458

HRL_GHZ = {"H30a": 231.900928, "H40a": 99.022952}
TRACER_GHZ = {"CO21": 230.538, "CO10": 115.271202, "CS21": 97.980953,
              "HCN10": 88.631847, "HCO+10": 89.188523, "13CO21": 220.398684}
WIN_RE = re.compile(r"(\d+\.\d+)\.\.(\d+\.\d+)GHz")


def parse_windows(fs):
    return [(float(a), float(b)) for a, b in WIN_RE.findall(str(fs))]


def covered(nu, wins, edge=0.05):
    return any(lo + edge <= nu <= hi - edge for lo, hi in wins)


def main():
    warnings.filterwarnings("ignore")
    from astroquery import cache_conf
    cache_conf.cache_active = False      # cache-poisoning guard (memory 2026-08-04)
    from astroquery.alma import Alma
    cands = [r for r in csv.DictReader(open(IN_CSV))
             if "8um" in r["seip_tag"]]
    print(f"scouting {len(cands)} 8μm-covered candidates", flush=True)

    done = set()
    if OUT_CSV.exists():
        done = {r["name"] for r in csv.DictReader(open(OUT_CSV))}
        print(f"resuming — {len(done)} done", flush=True)
    fields = ["name", "ra", "dec", "z", "n_obs", "n_good",
              "hrl_lines", "tracers", "best_res_arcsec",
              "est_size_GB", "sens10_mJy", "note"]
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for i, r in enumerate(cands):
            if r["name"] in done:
                continue
            try:
                cz = float(r["cz"])
                z = cz / C_KMS if np.isfinite(cz) and cz > 0 else np.nan
            except (TypeError, ValueError):
                z = np.nan
            row = dict(name=r["name"], ra=r["ra"], dec=r["dec"],
                       z=round(z, 5) if np.isfinite(z) else "",
                       n_obs=0, n_good=0, hrl_lines="", tracers="",
                       best_res_arcsec="", est_size_GB="", sens10_mJy="",
                       note="")
            try:
                res = Alma.query_region(
                    SkyCoord(float(r["ra"]) * u.deg, float(r["dec"]) * u.deg),
                    radius=1.5 * u.arcmin)
            except Exception as e:
                row["note"] = f"query-error {str(e)[:40]}"
                w.writerow(row); f.flush()
                continue
            row["n_obs"] = len(res)
            if len(res) == 0 or not np.isfinite(z):
                if not np.isfinite(z):
                    row["note"] = "no-z"
                w.writerow(row); f.flush()
                continue
            cols = {c.lower(): c for c in res.colnames}
            def get(rr, *names, default=""):
                for n in names:
                    if n.lower() in cols:
                        try:
                            return rr[cols[n.lower()]]
                        except Exception:
                            pass
                return default
            hrl_hit, tr_hit, good = set(), set(), []
            for rr in res:
                if str(get(rr, "data_rights")).lower().startswith("propr"):
                    continue
                if str(get(rr, "is_mosaic")).strip().upper() == "T":
                    continue
                wins = parse_windows(get(rr, "frequency_support"))
                if not wins:
                    continue
                h = [k for k, nu in HRL_GHZ.items()
                     if covered(nu / (1 + z), wins)]
                t = [k for k, nu in TRACER_GHZ.items()
                     if covered(nu / (1 + z), wins)]
                if h and t:
                    hrl_hit.update(h); tr_hit.update(t)
                    good.append(rr)
            row["n_good"] = len(good)
            if good:
                row["hrl_lines"] = "+".join(sorted(hrl_hit))
                row["tracers"] = "+".join(sorted(tr_hit))
                resns = [float(get(rr, "spatial_resolution",
                                   "ang_res_arcsec", default=np.nan))
                         for rr in good]
                resns = [x for x in resns if np.isfinite(x)]
                if resns:
                    row["best_res_arcsec"] = round(min(resns), 3)
                sizes = [float(get(rr, "access_estsize", default=0)) or 0
                         for rr in good]
                row["est_size_GB"] = round(sum(sizes) / 1e6, 1)  # kB → GB
                sens = [float(get(rr, "sensitivity_10kms",
                                  default=np.nan)) for rr in good]
                sens = [x for x in sens if np.isfinite(x) and x > 0]
                if sens:
                    row["sens10_mJy"] = round(min(sens), 3)
            w.writerow(row); f.flush()
            if row["n_good"]:
                print(f"  HIT {r['name']:<18} good={row['n_good']} "
                      f"{row['hrl_lines']} + {row['tracers']} "
                      f"res={row['best_res_arcsec']}\" "
                      f"size~{row['est_size_GB']}GB", flush=True)
            if (i + 1) % 25 == 0:
                print(f"  ... {i+1}/{len(cands)}", flush=True)
    print(f"done → {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
