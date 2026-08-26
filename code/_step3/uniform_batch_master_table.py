"""Master results table — Frank's required columns (meeting 2026-07-10):
RMS, flux, area of integration, S/N — plus velocity window, per-channel RMS
normalized to the 10 km/s reference width, detection tier, IR size (R90)
with provenance, and coverage class.

Sources: stage2_flux.csv (flux/σ/SN/nbeam/W), stage1_masks.csv (beam),
stage3_ir.csv (R90/coverage/band), cube headers (Δv, BMAJ/BMIN).

Output: result_v2/_uniform_batch/master_table.{csv,md}
Run: conda run -n casa_env python -u _step3/uniform_batch_master_table.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from uniform_batch_configs import build_table          # noqa: E402
from uniform_batch_stage1 import REST_HZ               # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
C_KMS = 299792.458

# Global uncertainty calibration: sigma_F from the channel RMS underestimates
# the uncertainty of a windowed integral by this factor, measured on the
# signal-free population (thesis ch. 5).  Every sigma_F, S/N and 3-sigma limit
# quoted in the thesis carries it, so tiers must be cut on SN / SIGMA_CAL.
SIGMA_CAL = 1.5


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    s1 = {r["galaxy"]: r for r in csv.DictReader(open(OUT / "stage1_masks.csv"))}
    s2 = {r["galaxy"]: r for r in csv.DictReader(open(OUT / "stage2_flux.csv"))}
    s3 = {r["galaxy"]: r for r in csv.DictReader(open(OUT / "stage3_ir.csv"))}
    cfg = {r["galaxy"]: r for r in build_table()}

    rows = []
    for g, r2 in s2.items():
        if r2.get("verdict") in ("review", "error"):
            tier = "review"
        else:
            # Tier on the CALIBRATED S/N, as the thesis does: stage2 writes the
            # nominal value, and sigma_F is scaled by SIGMA_CAL downstream.
            # Cutting on the nominal value here published 5 detections / 1
            # marginal against the thesis's 2 / 3 until 2026-08-26.
            sn = fnum(r2["SN"]) / SIGMA_CAL
            tier = ("detected" if sn >= 5 else
                    "marginal" if sn >= 3 else "upper-limit")
        # cube header: channel width + beam
        dv = bmaj = bmin = np.nan
        try:
            hdr = fits.getheader(cfg[g]["hrl_path"])
            nu = float(hdr["CRVAL3"])
            dv = C_KMS * abs(float(hdr["CDELT3"])) / nu
            bmaj = float(hdr["BMAJ"]) * 3600
            bmin = float(hdr["BMIN"]) * 3600
        except Exception:
            pass
        beam_area = 1.133 * bmaj * bmin if np.isfinite(bmaj) else np.nan
        nbeam = fnum(r2.get("nbeam"))
        area_asec2 = nbeam * beam_area if np.isfinite(beam_area) else np.nan
        W = fnum(r2.get("W_used"))
        F, sig, sn = fnum(r2.get("F")), fnum(r2.get("sigma_F")), fnum(r2.get("SN"))
        # per-channel RMS of the mask-integrated spectrum (Frank's "RMS at
        # that area"), native Δv and normalized to the 10 km/s reference
        n_chan = W / dv if np.isfinite(W) and np.isfinite(dv) and dv > 0 else np.nan
        rms_chan = sig / (np.sqrt(n_chan) * dv) * 1e3 \
            if np.isfinite(n_chan) and n_chan > 0 else np.nan       # mJy
        rms10 = rms_chan * np.sqrt(dv / 10.0) if np.isfinite(rms_chan) else np.nan
        # IR size + provenance
        r3 = s3.get(g, {})
        r90 = fnum(r3.get("w4_R90"))
        band = r3.get("band_used") or "w4"
        unres = str(r3.get("w4_unresolved")) == "True"
        if not np.isfinite(r90):
            ir_src = "no IR image"
        elif unres:
            ir_src = f"WISE {band.upper()} CoG — unresolved (≤ Atlas PSF)"
        else:
            ir_src = f"WISE {band.upper()} CoG (SB-truncated)"
        rows.append(dict(
            galaxy=g, line=r2.get("hrl_line", ""), tier=tier,
            F_Jykms=F, sigma_F=sig, SN=sn,
            W_kms=W, dv_native_kms=round(dv, 2) if np.isfinite(dv) else "",
            mask_area_beams=nbeam,
            mask_area_arcsec2=round(area_asec2, 1) if np.isfinite(area_asec2) else "",
            rms_chan_mJy=round(rms_chan, 3) if np.isfinite(rms_chan) else "",
            rms10_mJy=round(rms10, 3) if np.isfinite(rms10) else "",
            ir_R90_arcsec=r90 if np.isfinite(r90) else "",
            ir_size_source=ir_src,
            coverage=r3.get("coverage", ""),
        ))

    order = {"detected": 0, "marginal": 1, "upper-limit": 2, "review": 3}
    rows.sort(key=lambda r: (order.get(r["tier"], 9), -(r["SN"] or 0)))

    with open(OUT / "master_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(OUT / "master_table.md", "w") as f:
        f.write("<!-- Master results table — Frank's columns (RMS, flux, "
                "integration area, S/N) + tiers + IR size w/ provenance. "
                "Distances still cz/70-provisional; luminosities not "
                "included until the literature distance table lands. -->\n\n")
        f.write("| Galaxy | Line | Tier | F [Jy km/s] | σ_F | S/N | W [km/s] "
                "| Mask [beams] | Mask [arcsec²] | RMS_chan [mJy] "
                "| RMS(10 km/s) [mJy] | IR R90 [\"] | IR size source | Coverage |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['galaxy']} | {r['line']} | {r['tier']} "
                    f"| {r['F_Jykms']} | {r['sigma_F']} | {r['SN']} "
                    f"| {r['W_kms']} | {r['mask_area_beams']} "
                    f"| {r['mask_area_arcsec2']} | {r['rms_chan_mJy']} "
                    f"| {r['rms10_mJy']} | {r['ir_R90_arcsec']} "
                    f"| {r['ir_size_source']} | {r['coverage']} |\n")
    print(f"wrote {OUT/'master_table.csv'} and .md  ({len(rows)} rows)")
    for r in rows[:8]:
        print(f"  {r['galaxy']:<17}{r['tier']:<12}F={r['F_Jykms']}"
              f"  S/N={r['SN']}  RMS10={r['rms10_mJy']} mJy")


if __name__ == "__main__":
    main()
