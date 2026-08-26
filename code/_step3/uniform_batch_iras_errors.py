"""Fetch IRAS flux-density uncertainties (PSC/FSC) and propagate to σ_TIR
per Lau Appendix D (Sanders & Mirabel 1996 quadrature). Distance errors are
NOT included here — they join when the literature distance table lands.

2026-08-22 note (decisions (5)): the TIR FLUXES were switched to RBGS
four-band totals, but this error file deliberately stays PSC/FSC-based.
The RBGS e_F* columns are statistical-only (mJy; e.g. 65 mJy on the
967 Jy of NGC 253) and re-propagating them would understate the true
uncertainty by omitting the IRAS absolute-calibration floor. The PSC/FSC
relative errors (3.5-12%) stand in as a conservative statistical+
calibration flux term; either choice is buried by the 0.2 dex
apportioning systematic in stage 4's x-error budget.

Output: result_v2/_uniform_batch/iras_errors.csv
Run: conda run -n casa_env python -u _step3/uniform_batch_iras_errors.py
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table          # noqa: E402
from uniform_batch_stage3_ir import alma_geometry      # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
W = dict(w12=13.48, w25=5.16, w60=2.58, w100=1.00)


def fetch(ra, dec):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"], row_limit=5)
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    for cat in ["II/125/main", "II/156A/fsc"]:
        for attempt in range(3):
            try:
                res = v.query_region(coord, radius=2.0 * u.arcmin, catalog=cat)
                break
            except Exception:
                res = None
        if not res:
            continue
        t = res[0]
        def pair(band):
            try:
                f = float(t[f"Fnu_{band}"][0])
                e_pct = float(t[f"e_Fnu_{band}"][0])
                return f, f * e_pct / 100.0
            except Exception:
                return np.nan, np.nan
        vals = {b: pair(b) for b in ("12", "25", "60", "100")}
        if np.isfinite(vals["60"][0]):
            return vals, cat
    return None, None


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for row in build_table():
        g = row["galaxy"]
        try:
            ra, dec, *_ = alma_geometry(row)
            vals, cat = fetch(ra, dec)
        except Exception as e:
            print(f"{g:<17} ERROR {str(e)[:60]}", flush=True)
            rows.append(dict(galaxy=g, rel_fir=""))
            continue
        if vals is None:
            print(f"{g:<17} no IRAS errors", flush=True)
            rows.append(dict(galaxy=g, rel_fir=""))
            continue
        fir = 1.8e-14 * sum(W[f"w{b}"] * (vals[b][0] if np.isfinite(vals[b][0])
                                          else 0.0)
                            for b in ("12", "25", "60", "100"))
        sig = 1.8e-14 * np.sqrt(sum(
            (W[f"w{b}"] * vals[b][1]) ** 2
            for b in ("12", "25", "60", "100") if np.isfinite(vals[b][1])))
        rel = sig / fir if fir > 0 else np.nan
        print(f"{g:<17} {cat:<14} rel σ_TIR = {rel*100:.1f}%", flush=True)
        rows.append(dict(galaxy=g, rel_fir=round(rel, 4)))
    with open(OUT / "iras_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["galaxy", "rel_fir"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT / 'iras_errors.csv'}", flush=True)


if __name__ == "__main__":
    main()
