"""Build Table D for docs/target-list.md from scan_results.csv.

- Deduplicate (galaxy, hrl_line) → keep lowest-sens observation
- Exclude already-analyzed galaxies (Table A/B/C)
- Sort by predicted_sn_log
- Emit a markdown table section appendable to docs/target-list.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
SCAN_CSV = HERE / "scan_results.csv"
OUT_MD = HERE / "table_d_draft.md"
OUT_CSV = HERE / "table_d_filtered.csv"

# Common-name aliases (catalogue_name → exclude_key)
ALIASES = {
    "NGC5236": "M83",
    "NGC5194": "M51",
    "NGC5457": "M101",
}

# Exclude list — normalized (no spaces, uppercase, no leading zeros on NGC/IC)
EXCLUDED = {
    # Table A (reproduction)
    "NGC253", "NGC4945", "M83", "NGC5253", "NGC3256", "NGC1808",
    "ARP220", "NGC3627",
    # Table B (analyzed Phase Y 2026-04-21/22)
    "NGC6810", "NGC7130", "NGC3227", "NGC1386", "IC5063", "IRASF18293-3413",
    # Table C — any galaxy already covered by Lau/Ada/Toma
    "NGC3628", "NGC1097", "NGC4038", "IRAS13120-5453", "IC1623",
    "NGC7582", "NGC5643", "NGC7465", "NGC4501", "NGC6240", "NGC7469",
    "NGC5128", "NGC1068", "NGC55", "NGC1365", "NGC2903",
    "NGC6822", "NGC4418", "NGC1614",
}


def _norm(name: str) -> str:
    """Normalize for exclude-list matching: uppercase, strip spaces/underscores,
    strip leading zeros after NGC/IC prefix, and apply common aliases."""
    import re
    s = name.replace(" ", "").replace("_", "").upper()
    s = re.sub(r"^(NGC|IC)0+(\d)", r"\1\2", s)
    return ALIASES.get(s, s)


def main() -> None:
    rows = list(csv.DictReader(open(SCAN_CSV)))
    # Group by (name, hrl_line), keep min rms entry
    best: dict = {}
    for r in rows:
        key = (r["name"], r["hrl_line"])
        if key not in best or float(r["sens_10kms_mJy"]) < float(best[key]["sens_10kms_mJy"]):
            best[key] = r

    # Further dedupe per galaxy → keep best (highest predicted_sn_log) HRL line
    per_galaxy: dict = {}
    for r in best.values():
        name = r["name"]
        if name not in per_galaxy or float(r["predicted_sn_log"]) > float(per_galaxy[name]["predicted_sn_log"]):
            per_galaxy[name] = r

    # Filter excluded galaxies
    keep = [r for r in per_galaxy.values() if _norm(r["name"]) not in EXCLUDED]
    keep.sort(key=lambda r: -float(r["predicted_sn_log"]))

    # Write filtered CSV for downstream /alma-query invocation
    fieldnames = list(keep[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keep)
    print(f"Wrote {OUT_CSV}: {len(keep)} unique galaxies (after exclude)")

    # Write markdown Table D
    lines = [
        "## Table D — New target pool (PHANGS-ALMA ∪ ALMaQUEST, multi-line HRL, 2026-04-22)",
        "",
        "Built from:",
        "- PHANGS-ALMA (Leroy+2021, 90 galaxies, z<0.01, star-forming MS, Band 6 CO(2-1))",
        "- ALMaQUEST (Lin+2020, 46 galaxies, 0.01<z<0.06, MaNGA IFU, Band 3 CO(1-0))",
        "",
        "Method: per-galaxy ALMA archive cone search (60″) cross-matched against the full",
        "Hnα catalog (H27α–H32α + H39α–H42α), requiring BOTH an HRL AND a CO strong-tracer",
        "line fall in the same SPW. Deduplicated (galaxy × HRL → best rms), then galaxy → best",
        "predicted S/N ∝ log(SFR) − 2log(D/Mpc) − log(rms_10kms). Already-analyzed galaxies",
        "(Table A/B/C) excluded. Top entries below.",
        "",
        "| Rank | Galaxy | Survey | z | logM\\* | logSFR | D (Mpc) | HRL | rms_10kms (mJy) | predicted log S/N | Project |",
        "|-----:|--------|--------|---|--------|--------|---------|-----|-----------------|-------------------|---------|",
    ]
    top_n = min(25, len(keep))
    for i, r in enumerate(keep[:top_n], 1):
        try:
            d = float(r["distance_mpc"])
            d_str = f"{d:.1f}"
        except (ValueError, TypeError):
            d_str = "—"
        lines.append(
            f"| {i} | {r['name']} | {r['survey']} | {float(r['z']):.4f} | "
            f"{r['logMstar']} | {r['logSFR']} | {d_str} | "
            f"{r['hrl_line']} | {float(r['sens_10kms_mJy']):.2f} | "
            f"{float(r['predicted_sn_log']):+.2f} | {r['project_id']} |"
        )
    lines.append("")
    lines.append(f"**Full filtered CSV**: `{OUT_CSV.relative_to(HERE.parents[2])}` "
                 f"({len(keep)} galaxies). Rows not shown have lower predicted S/N.")
    lines.append("")
    lines.append("**Next step**: pick top N (suggested start: N=5), run `/alma-query {GALAXY} "
                 "--weak {HRL} --write-script` for each to generate download scripts, then "
                 "Phase Y pipeline as for Table B.")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_MD}: markdown Table D with top {top_n}")


if __name__ == "__main__":
    main()
