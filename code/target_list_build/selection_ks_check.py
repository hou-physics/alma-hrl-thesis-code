"""Selection-function indirect checks quoted in thesis ch3 (2026-08-24).

Two-sample Kolmogorov-Smirnov comparisons along the selection chain:
  A. archive-admitted (n_good > 0 in alma_scout.csv, 131) vs not admitted
     (174) within the 305 Spitzer-covered targets — on redshift and on
     RBGS S60. Quantifies WHAT the archival coverage selects on.
  B. declination-kept (delta <= +40, 501) vs removed (128) within the full
     RBGS 629 — on S60. Checks the engineering cut for accidental
     brightness bias (fetches RBGS via VizieR, cache off).

Run: conda run -n casa_env python -u target_list_build/selection_ks_check.py
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

OUT = Path("/Volumes/HouAstro/master/result_v2/_new_sample")


def ks_2samp(a, b):
    a, b = sorted(a), sorted(b)
    import bisect
    d = max(abs(bisect.bisect_right(a, v) / len(a)
                - bisect.bisect_right(b, v) / len(b))
            for v in set(a + b))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
                for j in range(1, 101))
    return d, max(min(p, 1.0), 0.0)


def median(x):
    x = sorted(x)
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


def main():
    # --- A: archival admission within the 305 --------------------------------
    sc = list(csv.DictReader(open(OUT / "alma_scout.csv")))
    sel = [r for r in sc if int(r["n_good"] or 0) > 0]
    non = [r for r in sc if int(r["n_good"] or 0) == 0]
    rb = {re.sub(r"[\s]", "", r["name"]).lower().replace("ngc0", "ngc"):
          float(r["f60"])
          for r in csv.DictReader(open(OUT / "rbgs_seip_candidates.csv"))}

    def zs(rows):
        out = []
        for r in rows:
            try:
                out.append(float(r["z"]))
            except (TypeError, ValueError):
                pass
        return out

    def f60s(rows):
        out = []
        for r in rows:
            k = re.sub(r"[\s]", "", r["name"]).lower().replace("ngc0", "ngc")
            if k in rb:
                out.append(rb[k])
        return out

    za, zb = zs(sel), zs(non)
    d, p = ks_2samp(za, zb)
    print(f"A. admitted {len(sel)} vs not {len(non)} (of 305):")
    print(f"   redshift: medians {median(za):.4f} / {median(zb):.4f}  "
          f"KS D={d:.3f} p={p:.2g}")
    fa, fb = f60s(sel), f60s(non)
    d, p = ks_2samp(fa, fb)
    print(f"   S60 [Jy]: medians {median(fa):.1f} / {median(fb):.1f}  "
          f"KS D={d:.3f} p={p:.2g}")

    # --- B: declination cut within the full RBGS -----------------------------
    from astroquery import cache_conf
    cache_conf.cache_active = False
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = Vizier(columns=["Name", "RAJ2000", "DEJ2000", "F60um"],
               row_limit=-1).get_catalogs("J/AJ/126/1607")[0]
    coords = SkyCoord(t["RAJ2000"], t["DEJ2000"],
                      unit=(u.hourangle, u.deg))
    dec = coords.dec.deg
    f60 = [float(v) for v in t["F60um"]]
    kept = [f for f, de in zip(f60, dec) if de <= 40.0]
    rem = [f for f, de in zip(f60, dec) if de > 40.0]
    d, p = ks_2samp(kept, rem)
    print(f"B. declination cut: kept {len(kept)} vs removed {len(rem)} "
          f"(of {len(f60)}):")
    print(f"   S60 [Jy]: medians {median(kept):.1f} / {median(rem):.1f}  "
          f"KS D={d:.3f} p={p:.2g}")


if __name__ == "__main__":
    main()
