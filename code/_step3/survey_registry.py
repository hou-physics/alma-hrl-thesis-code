"""Survey registry — ONE successor-facing table over every target the
survey has touched or could touch.

Rows = union of
  (a) every processed config (stage-1 class + master-table measurement),
  (b) every RBGS x SEIP candidate with usable archival ALMA data
      (alma_scout n_good > 0) not yet processed.

Columns: identity (name/ra/dec/z/f60) → status → measurement (F, σ, S/N,
W, mask area, RMS, RMS(10 km/s) — measured where done, archive-predicted
sens10 for candidates) → IR leg → data logistics → next_step (what a
successor does with this row).

Derived table: NEVER edit by hand — rerun this script after any batch run.
Run: conda run -n casa_env python -u _step3/survey_registry.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

BATCH = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
NEWS = Path("/Volumes/HouAstro/master/result_v2/_new_sample")
OUT_CSV = Path("/Volumes/HouAstro/master/result_v2/survey_registry.csv")
OUT_MD = Path("/Volumes/HouAstro/master/result_v2/survey_registry.md")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table               # noqa: E402

FIELDS = ["name", "ra", "dec", "z", "f60_Jy", "status", "hrl_line",
          "tracer", "F_Jykms", "sigma_F", "SN", "W_kms",
          "mask_area_beams", "mask_area_arcsec2", "rms_chan_mJy",
          "rms10_mJy", "rms10_pred_mJy", "ir_R90_arcsec",
          "ir_size_source", "coverage", "best_res_arcsec", "est_size_GB",
          "next_step"]

# wave-1 targets dropped at onboarding: spws cover the line at REST freq
# but not at the galaxy's z (scout matched rest-frame; decisions 2026-08-04)
DROPPED_ONBOARD = {"ngc7552": "excluded (line out of band at true z)",
                   "ngc6814": "excluded (line out of band at true z)"}


def norm(name):
    return re.sub(r"[\s\-_]", "", name.lower())


def main():
    s1 = {r["galaxy"]: r for r in csv.DictReader(open(BATCH / "stage1_masks.csv"))}
    s2 = {r["galaxy"]: r for r in csv.DictReader(open(BATCH / "stage2_flux.csv"))}
    s3 = {r["galaxy"]: r for r in csv.DictReader(open(BATCH / "stage3_ir.csv"))}
    mt = {r["galaxy"]: r for r in csv.DictReader(open(BATCH / "master_table.csv"))}
    cfg = {r["galaxy"]: r for r in build_table()}
    rbgs = list(csv.DictReader(open(NEWS / "rbgs_seip_candidates.csv")))
    scout = list(csv.DictReader(open(NEWS / "alma_scout.csv")))

    def rbgs_by_coord(ra, dec):
        for r in rbgs:
            if abs(float(r["ra"]) - ra) < 0.034 and \
               abs(float(r["dec"]) - dec) < 0.034:
                return r
        return None

    rows = []
    proc_coords = []

    # ---- (a) processed configs ----
    for g, c in sorted(cfg.items()):
        r1, r2, r3, rm = s1.get(g, {}), s2.get(g, {}), s3.get(g, {}), mt.get(g, {})
        ra = float(r3["ra"]) if r3.get("ra") else np.nan
        dec = float(r3["dec"]) if r3.get("dec") else np.nan
        if np.isfinite(ra):
            proc_coords.append((ra, dec))
        fc = r1.get("final_class", "")
        tier = rm.get("tier", "")
        # status tiers on CALIBRATED S/N (global sigma_F factor 1.5,
        # decisions.md 2026-08-22); measurement columns stay nominal.
        try:
            sn_cal = float(rm.get("SN", "")) / 1.5
            if np.isfinite(sn_cal):
                tier = ("detected" if sn_cal >= 5 else
                        "marginal" if sn_cal >= 3 else "upper-limit")
        except (TypeError, ValueError):
            pass
        if g in DROPPED_ONBOARD:
            status, nxt = DROPPED_ONBOARD[g], "needs different archival data"
        elif fc.startswith("excluded"):
            status, nxt = "excluded (user)", "closed — see decisions.md"
        elif tier in ("detected", "marginal", "upper-limit"):
            status = f"measured-{tier}"
            nxt = "done" if tier != "marginal" else "adjudicated — done"
        elif r2.get("verdict") == "review":
            status, nxt = "review", f"user review: {r2.get('note', '')[:60]}"
        elif fc.startswith("B"):
            status, nxt = "B-mosaic", "methodology-chapter exemplar only"
        elif fc.startswith("review") or fc.startswith("error"):
            status, nxt = "review", f"stage-1: {fc}"
        else:
            status, nxt = fc or "unknown", "inspect stage CSVs"
        rb = rbgs_by_coord(ra, dec) if np.isfinite(ra) else None
        rows.append(dict(
            name=g, ra=round(ra, 5) if np.isfinite(ra) else "",
            dec=round(dec, 5) if np.isfinite(dec) else "",
            z=c.get("z", ""), f60_Jy=rb["f60"] if rb else "",
            status=status, hrl_line=rm.get("line") or r2.get("hrl_line", ""),
            tracer=r2.get("strong_line") or r1.get("strong_line", ""),
            F_Jykms=rm.get("F_Jykms", ""), sigma_F=rm.get("sigma_F", ""),
            SN=rm.get("SN", ""), W_kms=rm.get("W_kms", ""),
            mask_area_beams=rm.get("mask_area_beams", ""),
            mask_area_arcsec2=rm.get("mask_area_arcsec2", ""),
            rms_chan_mJy=rm.get("rms_chan_mJy", ""),
            rms10_mJy=rm.get("rms10_mJy", ""), rms10_pred_mJy="",
            ir_R90_arcsec=rm.get("ir_R90_arcsec", ""),
            ir_size_source=rm.get("ir_size_source", ""),
            coverage=rm.get("coverage") or r3.get("coverage", ""),
            best_res_arcsec="", est_size_GB="", next_step=nxt))

    # ---- (b) unprocessed scout candidates with usable data ----
    proc_norm = {norm(g) for g in cfg}
    n_cand = 0
    for r in scout:
        if int(r["n_good"] or 0) == 0:
            continue
        ra, dec = float(r["ra"]), float(r["dec"])
        if norm(r["name"]) in proc_norm:
            continue
        if any(abs(ra - pra) < 0.034 and abs(dec - pdec) < 0.034
               for pra, pdec in proc_coords):
            continue
        n_cand += 1
        gb = r.get("est_size_GB", "")
        gb_disp = gb if gb and gb != "nan" else "?"
        dropped = DROPPED_ONBOARD.get(norm(r["name"]))
        rows.append(dict(
            name=r["name"], ra=round(ra, 5), dec=round(dec, 5),
            z=r["z"], f60_Jy=(rbgs_by_coord(ra, dec) or {}).get("f60", ""),
            status=dropped or "candidate", hrl_line=r.get("hrl_lines", ""),
            tracer=r.get("tracers", ""),
            F_Jykms="", sigma_F="", SN="", W_kms="",
            mask_area_beams="", mask_area_arcsec2="", rms_chan_mJy="",
            rms10_mJy="", rms10_pred_mJy=r.get("sens10_mJy", ""),
            ir_R90_arcsec="", ir_size_source="", coverage="",
            best_res_arcsec=r.get("best_res_arcsec", ""),
            est_size_GB=gb,
            next_step=("needs different archival data (wave-1 spws miss "
                       "the line at true z)" if dropped else
                       f"download (~{gb_disp} GB selective-est) → wave "
                       f"onboard → frozen stage 1-4")))

    ORDER = {"measured-detected": 0, "measured-marginal": 1,
             "measured-upper-limit": 2, "review": 3, "B-mosaic": 4,
             "excluded (user)": 5,
             "excluded (line out of band at true z)": 5, "candidate": 6}
    rows.sort(key=lambda r: (ORDER.get(r["status"], 9), r["name"]))

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    with open(OUT_MD, "w") as f:
        f.write("# Survey registry — every target, one table\n\n")
        f.write("Derived table (regenerate with `_step3/survey_registry.py`"
                " after any batch run — never edit by hand).\n\n")
        f.write(
            "## Where the candidates come from (selection chain, "
            "criteria v2 — decisions.md 2026-08-03 (2))\n\n"
            "1. **Parent catalog: IRAS RBGS** (Revised Bright Galaxy "
            "Sample, Sanders et al. 2003) — flux-limited at **IRAS "
            "60 μm**: f60 > 5.24 Jy (the catalog's own definition; "
            "gives a well-defined selection function on the IR axis, "
            "unbiased in HRL). The `f60_Jy` column is this value.\n"
            "2. **DEC ≤ +40°** — engineering pre-filter (ALMA never sees "
            "higher DEC well; archive essentially empty there).\n"
            "3. **Spitzer SEIP IRAC 8 μm coverage required** — the "
            "mask-level TIR apportioning method (Lau/Bittner lineage) "
            "needs an 8 μm map; galaxies without SEIP 8 μm cannot get "
            "an aperture-matched x-axis value. 501 RBGS×DEC targets → "
            "305 with 8 μm.\n"
            "4. **ALMA archive scout** (`target_list_build/alma_scout.py`)"
            " — public, non-mosaic data covering the redshifted HRL "
            "(H30α/H40α) plus a bright tracer line → 131 targets with "
            "usable data. `rms10_pred_mJy` is the archive-reported "
            "sensitivity (RMS per 10 km/s) of the best matching "
            "observation; `best_res_arcsec`/`est_size_GB` likewise from "
            "the archive query.\n"
            "5. Minus targets already processed/downloaded → the "
            "`candidate` rows below.\n\n"
            "Other IR bands appear only at MEASUREMENT time, not "
            "selection: WISE 12/22 μm (IR size + coverage check), "
            "IRAS 12/25/60/100 μm (global TIR via Sanders-Mirabel), "
            "SEIP 8 μm long+short exposure (mask-level fraction + "
            "saturation HDR patch).\n\n"
            "Known caveat (decisions.md 2026-08-04): the scout matched "
            "line coverage at CATALOG redshift with rest-frame windows — "
            "wave-2 re-scout must match REDSHIFTED frequencies and treat "
            "ULIRG catalog z as ±few-hundred km/s uncertain (NGC 7552 / "
            "NGC 6814 lesson).\n\n")
        f.write(
            "## Column key\n\n"
            "| column | meaning |\n|---|---|\n"
            "| name | canonical target name (processed rows: pipeline "
            "lowercase canon; candidates: RBGS name) |\n"
            "| ra, dec | J2000 [deg] (processed: ALMA pointing from the "
            "cube header; candidates: RBGS position) |\n"
            "| z | redshift — catalog value; CO-derived where the wave "
            "manifest says so (z_source column there) |\n"
            "| f60_Jy | IRAS 60 μm flux from RBGS (the selection "
            "quantity; blank = pre-RBGS-era target) |\n"
            "| status | measured-detected / measured-marginal / "
            "measured-upper-limit / review / B-mosaic / excluded (...) "
            "/ candidate |\n"
            "| hrl_line | recombination line (H30a = H30α 231.90 GHz, "
            "H40a = H40α 99.02 GHz) |\n"
            "| tracer | bright line used to build the aperture mask "
            "(CO21, CS21, HCO+10, ...) |\n"
            "| F_Jykms | HRL integrated line flux inside the mask "
            "[Jy km/s] — negative values are noise, normal for upper "
            "limits |\n"
            "| sigma_F | 1σ empirical uncertainty on F (line-free-"
            "channel RMS scaled to the integration window) |\n"
            "| SN | F / sigma_F, NOMINAL; the status tiers are evaluated "
            "on the calibrated S/N = SN/1.5 (global sigma_F calibration, "
            "decisions.md 2026-08-22): ≥5 detected, 3–5 marginal, "
            "<3 upper limit |\n"
            "| W_kms | integration width = tracer W20 line width "
            "(100 km/s floor; v3.2 adaptive bound) |\n"
            "| mask_area_beams, mask_area_arcsec2 | aperture (mask) "
            "area in synthesized beams / arcsec² |\n"
            "| rms_chan_mJy | per-channel RMS of the mask-integrated "
            "HRL spectrum at native channel width [mJy] |\n"
            "| rms10_mJy | the same RMS normalized to 10 km/s channels "
            "[mJy] — the archive-comparable depth number |\n"
            "| rms10_pred_mJy | candidates only: ALMA-archive PREDICTED "
            "sensitivity per 10 km/s — compare with measured rms10_mJy "
            "after processing |\n"
            "| ir_R90_arcsec | radius containing 90% of the IR light "
            "(curve of growth on the cutout) [arcsec] |\n"
            "| ir_size_source | band + method the size came from (e.g. "
            "WISE W4 CoG; 'unresolved' = at the Atlas PSF floor) |\n"
            "| coverage | does the ALMA aperture cover the IR source: "
            "covered / covered-unresolved / partial (needs aperture "
            "correction) |\n"
            "| best_res_arcsec | candidates only: best archival beam "
            "[arcsec] |\n"
            "| est_size_GB | candidates only: estimated download size "
            "(selective products) |\n"
            "| next_step | what a successor does with this row |\n\n")
        f.write("Status counts: " + ", ".join(
            f"{k} {v}" for k, v in sorted(counts.items())) + "\n\n")
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "---|" * len(FIELDS) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in FIELDS) + " |\n")
    print(f"{len(rows)} rows → {OUT_CSV}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<38} {v}")


if __name__ == "__main__":
    main()
