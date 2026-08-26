"""Patch stage3_ir.csv IRAS fluxes: PSC/FSC -> RBGS four-band totals.

Motivation (decisions.md 2026-08-22 (5)): stage3 pulled IRAS PSC (II/125) /
FSC (II/156A) point-source fluxes; for extended nearby galaxies these sit
well below the RBGS total-galaxy ADDSCAN photometry (NGC 4945 x0.62,
NGC 253 x0.78), biasing the correlation-diagram x-axis low, one-signed.
The survey parent IS the RBGS, and the Lau/Bittner lineage used RBGS
luminosities, so RBGS four-band fluxes are the correct TIR input.

Behaviour:
  * fetches RBGS (VizieR J/AJ/126/1607) with the astroquery HTTP cache OFF
    (cache-poisoning memory 2026-08-04);
  * for every stage3_ir.csv row whose galaxy matches an RBGS entry, replaces
    iras_f12/f25/f60/f100 with the RBGS values and sets iras_src=RBGS;
    unmatched rows keep their PSC/FSC values with iras_src=PSC/FSC.
    Coordinate-verified non-members (2026-08-22): Circinus/ESO 097-013 is
    excluded from the RBGS by its galactic latitude (|b| = 3.8 < 5);
    He 2-10 is not in the catalog; NGC 5408 is below the 5.24 Jy limit;
    IC 4687 appears only as the blended pair entry "IC 4687/6", and the
    single-source FSC photometry is kept to match the single-source line
    aperture. IRAS 17208-0014 is carried under the RBGS designation
    "IRAS F17207-0014" (alias below);
  * first run backs up the original to stage3_ir_pscfsc_v1.csv;
  * prints the per-source f60 ratio and the Delta log10 TIR.

Run:  conda run -n casa_env python uniform_batch_rbgs_flux_patch.py
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
STAGE3 = OUT / "stage3_ir.csv"
BACKUP = OUT / "stage3_ir_pscfsc_v1.csv"
RBGS_CAT = "J/AJ/126/1607"
SM96 = dict(w12=13.48, w25=5.16, w60=2.58, w100=1.00)
# stage3 key -> normalized RBGS Name (coordinate-verified 2026-08-22)
ALIASES = {"iras17208-0014": "irasf17207-0014"}


def norm(name: str) -> str:
    s = re.sub(r"[\s−]+", "", str(name)).lower()
    s = s.replace("-g", "-")                       # ESO 097-G013 -> eso097-013
    m = re.match(r"^(ngc|ic|ugc)0*(\d.*)$", s)     # NGC 0253 -> ngc253
    if m:
        s = m.group(1) + m.group(2)
    return s


def tir_comb(f12, f25, f60, f100):
    return (SM96["w12"] * f12 + SM96["w25"] * f25
            + SM96["w60"] * f60 + SM96["w100"] * f100)


def main():
    from astroquery import cache_conf
    cache_conf.cache_active = False
    from astroquery.vizier import Vizier

    v = Vizier(columns=["**"], row_limit=-1)
    t = v.get_catalogs(RBGS_CAT)[0]
    print(f"RBGS: {len(t)} rows; columns: {t.colnames}")
    fcols = {}
    for band in ("12", "25", "60", "100"):
        cands = [c for c in t.colnames
                 if re.fullmatch(rf"[FfSs]0*{band}(um)?", c)]
        if not cands:
            raise SystemExit(f"no flux column for {band} um in {t.colnames}")
        fcols[band] = cands[0]
    namecol = "Name" if "Name" in t.colnames else t.colnames[0]
    rbgs = {}
    for row in t:
        try:
            rbgs[norm(row[namecol])] = {b: float(row[fcols[b]])
                                        for b in fcols}
        except (TypeError, ValueError):
            continue

    rows = list(csv.DictReader(open(STAGE3)))
    fields = list(rows[0].keys())
    if "iras_src" not in fields:
        fields.append("iras_src")
    if not BACKUP.exists():
        shutil.copy(STAGE3, BACKUP)
        print(f"backup written: {BACKUP.name}")

    print(f"{'galaxy':16s} {'f60 old':>9s} {'f60 RBGS':>9s} {'ratio':>6s} "
          f"{'dlogTIR':>8s}")
    n_patch = 0
    for r in rows:
        key = norm(r["galaxy"])
        hit = rbgs.get(ALIASES.get(key, key))
        if hit is None:
            r["iras_src"] = "PSC/FSC"
            print(f"{r['galaxy']:16s} {'—':>9s} {'—':>9s} {'—':>6s} {'—':>8s}"
                  f"  (not in RBGS, keeps PSC/FSC)")
            continue
        old = {b: float(r[f"iras_f{b}"] or 0) for b in ("12", "25", "60", "100")}
        import math
        dlog = (math.log10(tir_comb(hit["12"], hit["25"], hit["60"], hit["100"]))
                - math.log10(tir_comb(old["12"], old["25"], old["60"], old["100"]))
                ) if old["60"] > 0 else float("nan")
        ratio = hit["60"] / old["60"] if old["60"] > 0 else float("nan")
        print(f"{r['galaxy']:16s} {old['60']:9.2f} {hit['60']:9.2f} "
              f"{1/ratio if ratio else 0:6.2f} {dlog:+8.3f}"
              .replace("nan", "  —"))
        for b in ("12", "25", "60", "100"):
            r[f"iras_f{b}"] = f"{hit[b]:.3f}"
        r["iras_src"] = "RBGS"
        n_patch += 1

    with open(STAGE3, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"patched {n_patch}/{len(rows)} rows -> {STAGE3.name} "
          f"(original: {BACKUP.name})")


if __name__ == "__main__":
    main()
