"""Merge PHANGS-ALMA and ALMaQUEST galaxy samples into a single candidate pool CSV.

Columns: name, survey, ra_deg, dec_deg, z, logMstar, logSFR, distance_mpc
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "candidate_pool.csv"


def _parse_sexagesimal_ra(s: str) -> float:
    h, m, sec = s.split()
    return 15.0 * (float(h) + float(m) / 60 + float(sec) / 3600)


def _parse_sexagesimal_dec(s: str) -> float:
    parts = s.split()
    sign = -1 if parts[0].startswith("-") else 1
    d = abs(float(parts[0]))
    m = float(parts[1]) if len(parts) > 1 else 0
    sec = float(parts[2]) if len(parts) > 2 else 0
    return sign * (d + m / 60 + sec / 3600)


def parse_phangs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        in_data = False
        for line in f:
            if line.startswith("---"):
                in_data = True
                continue
            if not in_data or line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or not fields[0].strip().isdigit():
                continue
            name = fields[1].strip()
            ra_str = fields[3].strip()
            dec_str = fields[4].strip()
            vlsr_kms = float(fields[5]) if fields[5].strip() else None
            dist_mpc = float(fields[11]) if fields[11].strip() else None
            logm = float(fields[14]) if fields[14].strip() else None
            logsfr = float(fields[19]) if fields[19].strip() else None
            # Redshift from vlsr (approximate, vlsr is km/s)
            z = vlsr_kms / 299792.458 if vlsr_kms is not None else None
            try:
                ra_deg = _parse_sexagesimal_ra(ra_str)
                dec_deg = _parse_sexagesimal_dec(dec_str)
            except Exception:
                continue
            rows.append(dict(
                name=name, survey="PHANGS-ALMA",
                ra_deg=round(ra_deg, 6), dec_deg=round(dec_deg, 6),
                z=round(z, 6) if z is not None else "",
                logMstar=logm if logm is not None else "",
                logSFR=logsfr if logsfr is not None else "",
                distance_mpc=dist_mpc if dist_mpc is not None else "",
            ))
    return rows


def parse_almaquest(path: Path) -> list[dict]:
    # need table1 (plate-ifu, ra, dec, z) and table2 (M*, SFR) joined by ID
    # First pass: table1
    t1 = {}
    with open(path) as f:
        current_table = None
        for line in f:
            if line.startswith("#Table"):
                current_table = line.split("\t", 1)[1].strip().rstrip(":")
                continue
            if line.startswith("#") or not line.strip() or line.startswith("---"):
                continue
            fields = line.rstrip("\n").split("\t")
            if not fields[0].strip().isdigit():
                continue
            if current_table == "J_ApJ_903_145_table1" and len(fields) >= 5:
                ident = fields[1].strip()
                ra = float(fields[2])
                dec = float(fields[3])
                z = float(fields[4])
                t1[ident] = dict(ra=ra, dec=dec, z=z)
            elif current_table == "J_ApJ_903_145_table2" and len(fields) >= 5:
                ident = fields[1].strip()
                # columns (0-indexed after recno=0): 1=ID, 2=Area, 3=Mass, 4=SFR
                if ident in t1:
                    try:
                        t1[ident]["logMstar"] = float(fields[3])
                    except (ValueError, IndexError):
                        pass
                    try:
                        t1[ident]["logSFR"] = float(fields[4])
                    except (ValueError, IndexError):
                        pass
    rows = []
    for ident, data in t1.items():
        rows.append(dict(
            name=f"MaNGA-{ident}", survey="ALMaQUEST",
            ra_deg=round(data["ra"], 6), dec_deg=round(data["dec"], 6),
            z=round(data["z"], 6),
            logMstar=data.get("logMstar", ""),
            logSFR=data.get("logSFR", ""),
            distance_mpc="",  # ALMaQUEST paper uses z-based; leave blank
        ))
    return rows


def main() -> None:
    phangs = parse_phangs(HERE / "phangs_alma.tsv")
    aq = parse_almaquest(HERE / "almaquest.tsv")
    rows = phangs + aq
    fieldnames = ["name", "survey", "ra_deg", "dec_deg", "z",
                  "logMstar", "logSFR", "distance_mpc"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}: {len(rows)} galaxies "
          f"({len(phangs)} PHANGS + {len(aq)} ALMaQUEST)")


if __name__ == "__main__":
    main()
