"""Literature distance table — step 1: harvest NED-D per nearby source.

For every measured/review source with cz < CZ_NEAR (where cz/70 is >~10%
uncertain), pull NED's distance compilation (NED-D), convert distance
moduli to Mpc, group by method with a quality ranking (TRGB > Cepheid >
SBF > SNIa > Tully-Fisher > others), and write:
  - distance_candidates.csv  (every published row, audit trail)
  - distance_table_draft.md  (per-galaxy method medians + RECOMMENDED value
    — for USER REVIEW; nothing is fed to the pipeline until approved)
Distant sources keep cz/70 (peculiar-velocity error < ~10%).

Run: conda run -n casa_env python -u _step3/distance_table_fetch.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table               # noqa: E402

try:                       # outage-empty responses must never be cached
    from astroquery import cache_conf
    cache_conf.cache_active = False
except Exception:
    pass

BATCH = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
C_KMS = 299792.458
CZ_NEAR = 3500.0           # km/s; below this cz/70 error exceeds ~10%

NED_NAME = {               # canon → NED-resolvable
    "eso097-013": "Circinus Galaxy", "he2-10": "He 2-10",
}
METHOD_RANK = ["TRGB", "Cepheids", "SBF", "SNIa", "Tully-Fisher",
               "Tully est", "IRAS", "Sosies", "Ring Diameter", "FP",
               "GCLF", "PNLF", "Brightest Stars"]


def ned_name(g):
    if g in NED_NAME:
        return NED_NAME[g]
    if g.startswith("ngc"):
        return "NGC " + g[3:]
    if g.startswith("ic"):
        return "IC " + g[2:]
    return g.upper()


def rank(method):
    for i, m in enumerate(METHOD_RANK):
        if m.lower() in method.lower():
            return i
    return len(METHOD_RANK)


def fetch_nedd(nname):
    """NED-D per-object distances via the nDistance CGI (this astroquery
    version's get_table has no 'distances' keyword). Returns list of
    dicts(DM, err, D_Mpc, method, refcode)."""
    import re as _re
    import requests
    txt = None
    for attempt in range(3):
        try:
            r = requests.get(
                "https://ned.ipac.caltech.edu/cgi-bin/nDistance",
                params=dict(name=nname), timeout=90)
            txt = r.text
            break
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {str(e)[:50]}",
                  flush=True)
            time.sleep(10)
    if txt is None or "Individually Referenced" not in txt:
        return []
    seg = txt[txt.find("Individually Referenced"):]
    out = []
    for tr in _re.findall(r"<tr>(.*?)</tr>", seg, _re.S):
        tds = [_re.sub(r"<[^>]+>", "", c).strip()
               for c in _re.findall(r"<td>(.*?)</td>", tr, _re.S)]
        if len(tds) < 5 or not tds[2]:
            continue
        try:
            d = float(tds[2])
        except ValueError:
            continue
        out.append(dict(DM=tds[0], err=tds[1], D_Mpc=d,
                        method=tds[3], refcode=tds[4]))
    return out


def main():
    s2 = list(csv.DictReader(open(BATCH / "stage2_flux.csv")))
    zmap = {r["galaxy"]: float(r["z"]) for r in build_table()}
    targets = [(r["galaxy"], zmap[r["galaxy"]]) for r in s2]
    near = [(g, z) for g, z in sorted(targets) if abs(z) * C_KMS < CZ_NEAR]
    far = [(g, z) for g, z in sorted(targets) if abs(z) * C_KMS >= CZ_NEAR]
    print(f"{len(near)} nearby (NED-D lookup) + {len(far)} distant (keep cz/70)")

    all_rows, summary = [], []
    for g, z in near:
        nname = ned_name(g)
        print(f"=== {g}  ({nname}, cz={z*C_KMS:.0f}) ===", flush=True)
        entries = fetch_nedd(nname)
        if not entries:
            print("    NO NED-D entries", flush=True)
            summary.append(dict(galaxy=g, ned_name=nname,
                                cz=round(z * C_KMS), n_pub=0,
                                best_method="", D_rec_Mpc="",
                                D_cz_Mpc=round(z * C_KMS / 70.0, 1),
                                methods=""))
            continue
        by_method = {}
        for e in entries:
            all_rows.append(dict(galaxy=g, ned_name=nname,
                                 D_Mpc=e["D_Mpc"], DM=e["DM"],
                                 method=e["method"], refcode=e["refcode"]))
            by_method.setdefault(e["method"], []).append(
                (e["D_Mpc"], e["refcode"]))
        tab = entries
        ms = sorted(by_method, key=rank)
        parts, best, best_ref = [], np.nan, ""
        for m in ms[:6]:
            vals = [v for v, _ in by_method[m]]
            med = float(np.median(vals))
            parts.append(f"{m}: {med:.2f} (n={len(vals)})")
            if not np.isfinite(best):
                best = med
                refs = sorted(by_method[m], key=lambda t: t[1])[-1]
                best_ref = refs[1]
        print(f"    {len(tab)} published; " + " | ".join(parts[:3]), flush=True)
        summary.append(dict(galaxy=g, ned_name=nname, cz=round(z * C_KMS),
                            n_pub=len(tab),
                            best_method=ms[0] if ms else "",
                            D_rec_Mpc=round(best, 2) if np.isfinite(best) else "",
                            best_refcode=best_ref,
                            D_cz_Mpc=round(z * C_KMS / 70.0, 1),
                            methods=" | ".join(parts)))

    with open(BATCH / "distance_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["galaxy", "ned_name", "D_Mpc",
                                          "DM", "method", "refcode"])
        w.writeheader(); w.writerows(all_rows)

    with open(BATCH / "distance_table_draft.md", "w") as f:
        f.write("# Literature distance table — DRAFT for user review\n\n")
        f.write("Recommended = median of the best-ranked direct method "
                "(TRGB > Cepheid > SBF > SNIa > TF > ...). Full audit "
                "trail in distance_candidates.csv. Distant sources "
                f"(cz ≥ {CZ_NEAR:.0f}) keep cz/70.\n\n")
        f.write("| galaxy | cz | n_pub | best method | D_rec [Mpc] | "
                "refcode | D_cz/70 | ratio | all methods (medians) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            ratio = (f"{s['D_cz_Mpc']/s['D_rec_Mpc']:.2f}"
                     if s.get("D_rec_Mpc") else "")
            f.write(f"| {s['galaxy']} | {s['cz']} | {s['n_pub']} | "
                    f"{s['best_method']} | {s.get('D_rec_Mpc','')} | "
                    f"{s.get('best_refcode','')} | {s['D_cz_Mpc']} | "
                    f"{ratio} | {s['methods']} |\n")
        f.write("\nDistant sources (cz/70 retained): "
                + ", ".join(g for g, _ in far) + "\n")
    print(f"\n{len(all_rows)} published rows → distance_candidates.csv")
    print("draft → distance_table_draft.md")


if __name__ == "__main__":
    main()
