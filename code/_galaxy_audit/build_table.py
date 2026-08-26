"""Build empirical detection-vs-features table across all analyzed galaxies.

For each galaxy under results/ that has a summary.md:
- parse headline S/N + flux + width + Nbeam + status + IoU
- read HRL pbcor FITS header for beam + pixel scale
- query HyperLeda (VII/237) for axis ratio + morphology
- compute inclination from logR25 (thin-disk model q0=0.2)
- estimate MRS ≈ 6 × bmaj (rough rule for 12m-only)

Variants (_adaptive / _all_components / _deep / _aca / etc.) are
de-duplicated: keep the best-S/N row per base galaxy. Output:
- results/_galaxy_audit/galaxies_table.csv (raw)
- results/_galaxy_audit/galaxies_table.md (sorted, classified)
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from astropy.io import fits
from astropy import units as u

warnings.filterwarnings("ignore")

REPO = Path("/Volumes/HouAstro/master")
RESULTS = REPO / "results"
WORK = REPO / "master_thesis/work_dir"
OUT_DIR = RESULTS / "_galaxy_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thesis-verdict overrides: when the summary.md "Status:" string disagrees with
# the joint-diagnostic verdict in docs/progress.md, override here. Mirrors the
# manual map in tag_results.py — keep the two in sync.
# Used to keep methodology-failure variants (NGC 7793 full_field/all/adaptive)
# OUT of the detections regression sample, and to demote summary-only
# "Detections" that fail joint diagnostics (NGC 6810, NGC 5253 X1c).
THESIS_VERDICT_OVERRIDE: dict[str, str] = {
    "NGC7793_full_field": "MethodologyFailure",
    "NGC7793_all_components": "MethodologyFailure",
    "NGC7793_adaptive": "MethodologyFailure",
    "NGC7793": "Non-detection",
    "NGC6810": "Non-detection",
    "NGC5253_x1c": "Non-detection",
    "NGC7582": "Marginal",
    "IRASF18293-3413": "Marginal",
    # NGC 5135 reclassified 2026-05-02 from headline detection (S/N 5.64) →
    # non-detection. Per-galaxy null test (16 offsets) gave galaxy-specific
    # p95 = 6.18 with 2/16 offsets exceeding real-line. Continuum residual +
    # bandpass drift in the H30α SPW produce off-line scan-best S/N 5-6 at
    # multiple offsets, disqualifying the headline as a clean detection.
    "NGC5135": "Non-detection",
}


def base_name(key: str) -> str:
    """Strip variant suffix (e.g. NGC3627_adaptive → NGC3627)."""
    for suf in ("_adaptive", "_all_components", "_deep", "_deep_native",
                "_aca", "_x1c", "_full_field", "_native"):
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def parse_summary(path: Path) -> dict | None:
    """Extract headline S/N, flux, width, Nbeam, status, IoU from summary.md."""
    if not path.exists():
        return None
    text = path.read_text()
    # Status
    m = re.search(r"\*\*Status:\*\*\s*(\S+)", text)
    status = m.group(1) if m else "?"
    # First S/N|Flux|Width|N_beam table is the headline
    m = re.search(r"\| S/N \| ([\-0-9.]+) \|", text)
    sn = float(m.group(1)) if m else float("nan")
    m = re.search(r"\| Flux \| ([\-0-9.]+)\s*Jy", text)
    flux = float(m.group(1)) if m else float("nan")
    m = re.search(r"\| Width \| ([\-0-9.]+)\s*km/s \|", text)
    width = float(m.group(1)) if m else float("nan")
    m = re.search(r"\| N_beam \| ([\-0-9.]+) \|", text)
    nbeam = float(m.group(1)) if m else float("nan")
    # Fixed-envelope (second occurrence of S/N row)
    sn_iter = list(re.finditer(r"\| S/N \| ([\-0-9.]+) \|", text))
    sn_fix = float(sn_iter[1].group(1)) if len(sn_iter) > 1 else float("nan")
    # IoU
    m = re.search(r"IoU.*?\|\s*\*\*([\-0-9.]+)\*\*", text)
    iou = float(m.group(1)) if m else float("nan")
    # Null threshold
    m = re.search(r"null-calibrated threshold = ([\-0-9.]+)", text)
    null_thr = float(m.group(1)) if m else float("nan")
    return dict(status=status, sn=sn, flux=flux, width=width, nbeam=nbeam,
                sn_fix=sn_fix, iou=iou, null_thr=null_thr)


def read_beam_from_fits(galaxy_dir: Path) -> dict:
    """Read bmaj, bmin, bpa, pix_arcsec from HRL pbcor symlink (if present)."""
    candidates = list(galaxy_dir.glob("*_H30a_pbcor.fits")) + \
                 list(galaxy_dir.glob("*_H40a_pbcor.fits")) + \
                 list(galaxy_dir.glob("*_H29a_pbcor.fits"))
    if not candidates:
        return {}
    fpath = candidates[0]
    try:
        with fits.open(fpath, memmap=True) as h:
            hdr = h[0].header
        bmaj = float(hdr.get("BMAJ", 0)) * 3600
        bmin = float(hdr.get("BMIN", 0)) * 3600
        bpa = float(hdr.get("BPA", 0))
        try:
            pix = abs(float(hdr["CDELT1"])) * 3600
        except KeyError:
            pix = abs(float(hdr["CD1_1"])) * 3600
        return dict(bmaj=bmaj, bmin=bmin, bpa=bpa, pix=pix,
                    line=fpath.name.split("_")[1])
    except Exception as e:
        print(f"  WARN: FITS read failed for {fpath.name}: {e}")
        return {}


def query_hyperleda(name: str) -> dict:
    """Get axis ratio + morphology from HyperLeda VII/237."""
    from astroquery.vizier import Vizier
    # Normalize name for query
    name_q = name
    if name.startswith("NGC") and not name.startswith("NGC "):
        name_q = "NGC " + name[3:]
    elif name.startswith("IC") and not name.startswith("IC "):
        name_q = "IC " + name[2:]
    elif name.startswith("IRAS") and not name.startswith("IRAS "):
        name_q = "IRAS " + name[4:]
    try:
        v = Vizier(catalog="VII/237", row_limit=3,
                   columns=["RAJ2000", "DEJ2000", "OType", "MType",
                            "logD25", "logR25", "PA", "ANames"])
        # First try query_object (resolves name → coord then radius search)
        result = v.query_object(name_q, radius=2 * u.arcmin)
        if len(result) == 0 or len(result[0]) == 0:
            return {}
        # Find a row whose ANames mentions our galaxy
        tbl = result[0]
        target_lower = name.replace(" ", "").lower()
        match = None
        for row in tbl:
            anames_lower = str(row.get("ANames", "")).replace(" ", "").lower()
            if target_lower in anames_lower:
                match = row
                break
        if match is None:
            match = tbl[0]  # fall back to first
        return dict(
            otype=str(match.get("OType", "")),
            mtype=str(match.get("MType", "")),
            logr25=float(match.get("logR25", "nan") or "nan"),
            logd25=float(match.get("logD25", "nan") or "nan"),
            pa=float(match.get("PA", "nan") or "nan"),
        )
    except Exception as e:
        print(f"  WARN: HyperLeda query failed for {name}: {e}")
        return {}


def inclination_from_logr25(logr25: float, q0: float = 0.2) -> float:
    """Hubble formula: cos²(i) = (q² - q0²) / (1 - q0²) with q = b/a = 10^-logR25."""
    if not math.isfinite(logr25) or logr25 < 0:
        return float("nan")
    q = 10 ** (-logr25)  # b/a
    if q < q0:
        return 90.0  # edge-on saturated
    cos2 = (q * q - q0 * q0) / (1 - q0 * q0)
    cos2 = max(0.0, min(1.0, cos2))
    return math.degrees(math.acos(math.sqrt(cos2)))


def read_z_from_config(key: str) -> float:
    """Pull z= line from per-galaxy step3_analyze.py."""
    code_dir = REPO / f"master_thesis/my_code/{key.lower()}_analyse_code"
    config = code_dir / "step3_analyze.py"
    if not config.exists():
        # variants like NGC3627_adaptive → look for variant-named script
        config = code_dir / f"step3_analyze_{key.split('_', 1)[1]}.py" \
            if "_" in key else None
    if config is None or not config.exists():
        # fall back to base script
        base = base_name(key)
        config = REPO / f"master_thesis/my_code/{base.lower()}_analyse_code/step3_analyze.py"
    if not config.exists():
        return float("nan")
    text = config.read_text()
    m = re.search(r"\bz\s*=\s*(\d+\.\d+)", text)
    return float(m.group(1)) if m else float("nan")


def collect_one(galaxy_dir: Path) -> dict | None:
    key = galaxy_dir.name
    summary = parse_summary(galaxy_dir / "summary.md")
    if summary is None:
        return None
    base = base_name(key)
    work_dir = WORK / base
    beam = read_beam_from_fits(work_dir)
    z = read_z_from_config(key)
    # Hubble-flow distance (H0 = 70 km/s/Mpc); only meaningful for z > 0.001.
    # For very nearby galaxies (z < 0.002, D < 8 Mpc) Hubble flow underestimates
    # because peculiar velocities dominate; use literature primary distances.
    D_Mpc = (z * 299792.458 / 70.0) if (math.isfinite(z) and z > 0.001) else float("nan")
    # Override with literature primary distances for low-z galaxies
    primary_D = {
        "NGC253": 3.5, "NGC4945": 3.8, "NGC5253": 3.8, "NGC1313": 4.3,
        "NGC1377": 24.0, "NGC300": 1.9, "NGC0300": 1.9,
        "NGC3627": 11.3, "NGC3628": 10.0, "M83": 4.5, "NGC5236": 4.5,
        "NGC4826": 7.4, "NGC7793": 3.6, "IC342": 3.3, "NGC6810": 28.7,
    }
    if base in primary_D:
        D_Mpc = primary_D[base]
    # Physical scale: 1″ at D Mpc = D × π / (180×3600) × 10⁶ pc = 4.848 × D pc
    pc_per_arcsec = 4.848 * D_Mpc if math.isfinite(D_Mpc) else float("nan")
    return dict(key=key, base=base, **summary, **beam,
                z=z, D_Mpc=D_Mpc, pc_per_arcsec=pc_per_arcsec)


def estimate_mrs(bmaj: float) -> float:
    """ALMA 12m-only MRS rough rule (Cycle 7 docs): ~6 × beam major axis."""
    if not math.isfinite(bmaj) or bmaj <= 0:
        return float("nan")
    return 6 * bmaj


def main():
    galaxy_dirs = sorted(d for d in RESULTS.iterdir()
                         if d.is_dir() and not d.name.startswith("_"))
    print(f"Found {len(galaxy_dirs)} galaxy result dirs")

    rows = []
    for gd in galaxy_dirs:
        r = collect_one(gd)
        if r is not None:
            rows.append(r)
        else:
            print(f"  SKIP {gd.name}: no summary.md")
    print(f"Collected {len(rows)} rows from summary.md")

    # Deduplicate by base, keep best-S/N row per base — but skip variants whose
    # thesis verdict is MethodologyFailure (they're spurious-high-S/N variants
    # that would shadow the legitimate canonical row otherwise).
    methfail_keys = {k for k, v in THESIS_VERDICT_OVERRIDE.items()
                     if v == "MethodologyFailure"}
    by_base: dict[str, dict] = {}
    methfail_rows: list[dict] = []
    for r in rows:
        if r["key"] in methfail_keys:
            methfail_rows.append(r)
            continue
        b = r["base"]
        existing = by_base.get(b)
        sn = r["sn"]
        if existing is None or (math.isfinite(sn) and (
                not math.isfinite(existing["sn"]) or sn > existing["sn"])):
            by_base[b] = r
    print(f"After dedup by base: {len(by_base)} unique galaxies "
          f"(plus {len(methfail_rows)} methodology-failure variants kept separate)")

    # Query HyperLeda for each base galaxy
    print("Querying HyperLeda for inclination + morphology ...")
    for b, r in by_base.items():
        info = query_hyperleda(b)
        if info:
            r.update(info)
            r["incl_deg"] = inclination_from_logr25(info.get("logr25", float("nan")))
        else:
            r["incl_deg"] = float("nan")

    # Compute MRS
    for r in by_base.values():
        r["mrs"] = estimate_mrs(r.get("bmaj", float("nan")))

    # Sort by S/N descending
    sorted_rows = sorted(by_base.values(),
                         key=lambda r: (math.isfinite(r["sn"]), r["sn"]),
                         reverse=True)

    # Write CSV
    import csv
    csv_path = OUT_DIR / "galaxies_table.csv"
    fields = ["key", "base", "status", "sn", "flux", "width", "nbeam",
              "sn_fix", "iou", "null_thr", "line", "bmaj", "bmin", "bpa",
              "pix", "mrs", "logr25", "logd25", "pa", "incl_deg",
              "otype", "mtype", "z", "D_Mpc", "pc_per_arcsec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted_rows:
            w.writerow(r)
    print(f"CSV written: {csv_path}")

    # Write Markdown table
    md_path = OUT_DIR / "galaxies_table.md"
    lines = [
        "# Galaxy detection × features table",
        "",
        f"Generated by `master_thesis/my_code/_galaxy_audit/build_table.py` "
        f"on {Path(__file__).stat().st_mtime:.0f}.",
        "",
        f"**N galaxies**: {len(sorted_rows)} unique base sources "
        f"(variants like _adaptive deduplicated, kept best-S/N row).",
        "",
        "**Goal**: identify empirical predictors of HRL detection from "
        "geometry (inclination, axis ratio) + observation parameters "
        "(beam, MRS, sensitivity) — use to refine alma-query selection rubric.",
        "",
        "## Methodology",
        "",
        "- S/N, Flux, W, N_beam, IoU: from `results/{key}/summary.md` (scan-optimized).",
        "- Status: from `summary.md` Status line, overridden by the thesis-verdict map "
        "in `build_table.py::THESIS_VERDICT_OVERRIDE` for galaxies where the joint "
        "diagnostic disagrees with the raw summary string (NGC 6810 / NGC 5253 X1c "
        "demoted to Non-detection; NGC 7793 variants split into Methodology-failure).",
        "- bmaj/bmin/BPA/pix: from HRL pbcor FITS header.",
        "- MRS estimate: 6 × bmaj (12m-only Cycle 7 rule of thumb).",
        "- logR25, MType: from HyperLeda VII/237 (Vizier).",
        "- Inclination i: derived from logR25 via Hubble formula `cos²i = (q² − q0²) / (1 − q0²)`, q0=0.2.",
        "",
        "## Detections (S/N > null threshold, ranked)",
        "",
        "| Galaxy | S/N | Flux (Jy·km/s) | W (km/s) | i (°) | z | D (Mpc) | beam (″) | beam (pc) | MRS (″) | MType |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(v, fmt_str=".2f"):
        if isinstance(v, str): return v
        try:
            if not math.isfinite(v): return "—"
            return f"{v:{fmt_str}}"
        except (TypeError, ValueError):
            return "—"

    def beam_pc(r):
        b = r.get("bmaj", float("nan"))
        s = r.get("pc_per_arcsec", float("nan"))
        if math.isfinite(b) and math.isfinite(s):
            return f"{b * s:.0f}"
        return "—"

    def thesis_verdict(r):
        """Override per THESIS_VERDICT_OVERRIDE; else fall back to summary string."""
        ov = THESIS_VERDICT_OVERRIDE.get(r["key"])
        if ov is not None:
            return ov
        s = r["status"]
        if not isinstance(s, str):
            return "?"
        if "Non" in s:
            return "Non-detection"
        if "Marginal" in s or "marginal" in s:
            return "Marginal"
        if "etection" in s:
            return "Detection"
        return "?"

    detections = [r for r in sorted_rows if thesis_verdict(r) == "Detection"]
    marginals  = [r for r in sorted_rows if thesis_verdict(r) == "Marginal"]
    # Methodology-failure variants are kept separately (didn't go through dedup);
    # surfaced here for the "spurious variants" audit section without
    # contaminating the regression sample.
    methfail   = methfail_rows
    for r in detections:
        lines.append(
            f"| {r['key']} | {fmt(r['sn'], '.2f')} | {fmt(r['flux'], '.3f')} | "
            f"{fmt(r['width'], '.0f')} | {fmt(r['incl_deg'], '.0f')} | "
            f"{fmt(r.get('z', float('nan')), '.4f')} | "
            f"{fmt(r.get('D_Mpc', float('nan')), '.1f')} | "
            f"{fmt(r.get('bmaj', float('nan')), '.2f')} × {fmt(r.get('bmin', float('nan')), '.2f')} | "
            f"{beam_pc(r)} | {fmt(r['mrs'], '.1f')} | "
            f"{r.get('mtype', '—') or '—'} |"
        )

    if marginals:
        lines += ["", "## Marginal (S/N just above null threshold; flagged for caveat)", "",
                  "| Galaxy | S/N | Flux | W | i (°) | D (Mpc) | beam (pc) | IoU | MType |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in marginals:
            lines.append(
                f"| {r['key']} | {fmt(r['sn'], '.2f')} | {fmt(r['flux'], '.3f')} | "
                f"{fmt(r['width'], '.0f')} | {fmt(r['incl_deg'], '.0f')} | "
                f"{fmt(r.get('D_Mpc', float('nan')), '.1f')} | {beam_pc(r)} | "
                f"{fmt(r.get('iou', float('nan')), '.3f')} | "
                f"{r.get('mtype', '—') or '—'} |"
            )

    if methfail:
        lines += ["", "## Methodology-failure variants (excluded from regression)", "",
                  "These rows have summary.md `Status: Detection` but fail joint diagnostics",
                  "(baseline-residual selector pickup, fragmented mask, scan-ceiling width).",
                  "See `docs/paper-notes/caveats-draft.md` § \"Sample selection limit\" for the",
                  "quantitative derivation of why these are spurious.",
                  "",
                  "| Variant | S/N | Flux | W (km/s) | Nbeam | IoU | Reason |",
                  "|---|---|---|---|---|---|---|"]
        reason_map = {
            "NGC7793_full_field": "Nbeam 2017, baseline-residual integral ≫ Murphy+ predicted flux",
            "NGC7793_all_components": "scan-ceiling width 502 km/s, mask fragmented",
            "NGC7793_adaptive": "171 connected components > guard threshold 10",
        }
        for r in methfail:
            lines.append(
                f"| {r['key']} | {fmt(r['sn'], '.2f')} | {fmt(r['flux'], '.3f')} | "
                f"{fmt(r['width'], '.0f')} | {fmt(r['nbeam'], '.0f')} | "
                f"{fmt(r.get('iou', float('nan')), '.3f')} | "
                f"{reason_map.get(r['key'], '—')} |"
            )

    lines += ["", "## Non-detections (ranked by |S/N|)", "",
              "| Galaxy | S/N | i (°) | z | D (Mpc) | beam (″) | beam (pc) | MRS (″) | IoU | MType |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    nondet = [r for r in sorted_rows if thesis_verdict(r) == "Non-detection"]
    for r in sorted(nondet, key=lambda r: abs(r["sn"]) if math.isfinite(r["sn"]) else -1,
                    reverse=True):
        lines.append(
            f"| {r['key']} | {fmt(r['sn'], '.2f')} | "
            f"{fmt(r['incl_deg'], '.0f')} | "
            f"{fmt(r.get('z', float('nan')), '.4f')} | "
            f"{fmt(r.get('D_Mpc', float('nan')), '.1f')} | "
            f"{fmt(r.get('bmaj', float('nan')), '.2f')} × {fmt(r.get('bmin', float('nan')), '.2f')} | "
            f"{beam_pc(r)} | {fmt(r['mrs'], '.1f')} | "
            f"{fmt(r.get('iou', float('nan')), '.3f')} | "
            f"{r.get('mtype', '—') or '—'} |"
        )

    lines += [
        "",
        "## Pattern analysis",
        "",
        f"- Detections: {len(detections)} / {len(sorted_rows)} = "
        f"{100*len(detections)/max(len(sorted_rows),1):.0f}%",
    ]
    if detections:
        det_incls = [r["incl_deg"] for r in detections if math.isfinite(r["incl_deg"])]
        det_bmaj = [r.get("bmaj", float("nan")) for r in detections
                    if math.isfinite(r.get("bmaj", float("nan")))]
        if det_incls:
            lines.append(f"- Detection inclination: median {np.median(det_incls):.0f}°, "
                         f"range {min(det_incls):.0f}–{max(det_incls):.0f}°")
        if det_bmaj:
            lines.append(f"- Detection beam (bmaj): median {np.median(det_bmaj):.2f}″, "
                         f"range {min(det_bmaj):.2f}–{max(det_bmaj):.2f}″")
    if nondet:
        nd_incls = [r["incl_deg"] for r in nondet if math.isfinite(r["incl_deg"])]
        nd_bmaj = [r.get("bmaj", float("nan")) for r in nondet
                   if math.isfinite(r.get("bmaj", float("nan")))]
        if nd_incls:
            lines.append(f"- Non-detection inclination: median {np.median(nd_incls):.0f}°, "
                         f"range {min(nd_incls):.0f}–{max(nd_incls):.0f}°")
        if nd_bmaj:
            lines.append(f"- Non-detection beam (bmaj): median {np.median(nd_bmaj):.2f}″, "
                         f"range {min(nd_bmaj):.2f}–{max(nd_bmaj):.2f}″")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Markdown written: {md_path}")


if __name__ == "__main__":
    main()
