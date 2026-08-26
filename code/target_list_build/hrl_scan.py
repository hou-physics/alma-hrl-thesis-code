"""Multi-HRL archive scanner for the candidate pool.

For each galaxy in candidate_pool.csv:
  1. Single ALMA archive query (cone search by RA/Dec via Alma.query_region).
  2. For every observation row, check whether ANY Hnα (n∈{27..42}) covered
     in at least one SPW AT THIS GALAXY'S REDSHIFT, and whether a matching
     strong-tracer line (CO10/CO21/CO32) is also covered.
  3. Retain (galaxy, obs, best_line) triples.
  4. Rank globally by predicted S/N ∝ SFR / dist² / rms_10kms.

Output: scan_results.csv with columns:
  name, survey, z, logMstar, logSFR, dist_mpc,
  hrl_line, strong_line, project_id, mousid,
  sens_10kms_mJy, ang_res_arcsec, is_mosaic, tiers_available,
  predicted_sn_log

Doesn't call probe_tiers (expensive, ~s per obs); tier inspection deferred
to a follow-up /alma-query call on the top-ranked rows.
"""
from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.alma import Alma
from astroquery.alma import utils as au

HERE = Path(__file__).parent
POOL = HERE / "candidate_pool.csv"
OUT = HERE / "scan_results.csv"

SKILL_DIR = Path("/Volumes/HouAstro/master/.claude/skills/alma-query")
sys.path.insert(0, str(SKILL_DIR))
from line_catalog import get_rest_freq, observed_freq  # noqa: E402

# Hnα lines in ALMA bands 3/6/7
HRL_LINES = [f"H{n}a" for n in (27, 28, 29, 30, 31, 32, 39, 40, 41, 42)]
STRONG_BY_BAND = {
    # rough band center for each HRL
    "H40a": "CO10", "H41a": "CO10", "H42a": "CO10", "H39a": "CO10",
    "H30a": "CO21", "H29a": "CO21", "H31a": "CO21", "H32a": "CO21",
    "H28a": "CO21",
    "H27a": "CO32",
}
SEARCH_RADIUS = 60 * u.arcsec  # pointing tolerance
SENS_LIMIT_MJY = 30.0


def _fits_within(obs_ghz: float, interval_ghz: Tuple[float, float]) -> bool:
    lo, hi = interval_ghz
    return lo <= obs_ghz <= hi


def scan_galaxy(row: dict) -> List[dict]:
    """Return zero or more (obs × hrl) match dicts for this galaxy."""
    try:
        z = float(row["z"])
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
    except (TypeError, ValueError):
        return []
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    alma = Alma()
    try:
        obs = alma.query_region(coord, radius=SEARCH_RADIUS,
                                sensitivity_10kms=f"<{SENS_LIMIT_MJY}")
    except Exception as e:
        print(f"  {row['name']}: query failed: {e}", file=sys.stderr)
        return []
    if obs is None or len(obs) == 0:
        return []

    hits: List[dict] = []
    seen_mous_line: set = set()  # dedup (mousid, hrl_line)
    for r in obs:
        try:
            intervals = au.parse_frequency_support(r["frequency_support"])
        except Exception:
            continue
        ivs = [(iv[0].to(u.GHz).value, iv[1].to(u.GHz).value) for iv in intervals]

        mousid = str(r["member_ous_uid"])
        for hrl in HRL_LINES:
            obs_ghz = observed_freq(hrl, z)
            if not any(_fits_within(obs_ghz, iv) for iv in ivs):
                continue
            strong = STRONG_BY_BAND[hrl]
            strong_obs = observed_freq(strong, z)
            if not any(_fits_within(strong_obs, iv) for iv in ivs):
                continue
            key = (mousid, hrl)
            if key in seen_mous_line:
                continue
            seen_mous_line.add(key)

            hits.append({
                "name": row["name"],
                "survey": row["survey"],
                "z": z,
                "logMstar": row["logMstar"],
                "logSFR": row["logSFR"],
                "distance_mpc": row["distance_mpc"],
                "hrl_line": hrl,
                "strong_line": strong,
                "project_id": str(r["proposal_id"]),
                "mousid": mousid,
                "sens_10kms_mJy": float(r["sensitivity_10kms"]),
                "ang_res_arcsec": float(r["spatial_resolution"]) if "spatial_resolution" in r.colnames else 0.0,
                "pointing_ra": float(r["s_ra"]),
                "pointing_dec": float(r["s_dec"]),
                "predicted_sn_log": "",  # filled below
            })
    return hits


def predicted_sn_log(row: dict) -> float:
    """log10-scale S/N proxy ∝ SFR / D² / rms.

    Missing SFR → use −1 (faint default). Missing D → derive from z assuming H0=70.
    """
    try:
        logsfr = float(row["logSFR"])
    except (TypeError, ValueError):
        logsfr = -1.0
    d = None
    try:
        d = float(row["distance_mpc"])
    except (TypeError, ValueError):
        pass
    if d is None or d <= 0:
        z = row["z"]
        d = (z * 299792.458) / 70.0  # c*z/H0 in Mpc
    try:
        rms = float(row["sens_10kms_mJy"])
    except (TypeError, ValueError):
        return -99
    if rms <= 0:
        return -99
    return logsfr - 2 * math.log10(max(d, 1.0)) - math.log10(rms)


def main() -> None:
    with open(POOL) as f:
        pool = list(csv.DictReader(f))
    print(f"Pool: {len(pool)} galaxies")

    all_hits: List[dict] = []
    for i, row in enumerate(pool, 1):
        hits = scan_galaxy(row)
        if hits:
            print(f"  [{i:3d}/{len(pool)}] {row['name']:<15} "
                  f"{row['survey']:<12} → {len(hits)} (obs×HRL) hits")
        else:
            print(f"  [{i:3d}/{len(pool)}] {row['name']:<15} "
                  f"{row['survey']:<12} → 0")
        all_hits.extend(hits)
        # Gentle pacing to avoid archive throttling
        time.sleep(0.3)

    for h in all_hits:
        h["predicted_sn_log"] = round(predicted_sn_log(h), 3)
    all_hits.sort(key=lambda h: -h["predicted_sn_log"])

    fieldnames = ["name", "survey", "z", "logMstar", "logSFR", "distance_mpc",
                  "hrl_line", "strong_line", "project_id", "mousid",
                  "sens_10kms_mJy", "ang_res_arcsec",
                  "pointing_ra", "pointing_dec", "predicted_sn_log"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_hits)
    print(f"\nWrote {OUT}: {len(all_hits)} (galaxy × obs × HRL) matches")


if __name__ == "__main__":
    main()
