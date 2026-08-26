"""Uniform-prescription batch — STAGE 1: mask + classification signals only.

For every canonical galaxy config: strong-tracer mom-0 (cached), frozen mask
rule (s=1×beam smooth, 2σ contour, islands peak ≥ 5σ ∧ area ≥ 1 beam, union),
then per-galaxy classification signals:
  nbeam, n_islands, mask FoV fraction, seed found, red flags.

No HRL flux here — stage 2 runs on class-A sources after user reviews the
classification table. Rows append to CSV as they finish (crash-safe).

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_stage1.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube, get_beam_area_pix          # noqa: E402
from uniform_contour_poc import (build_mask, beam_fwhm_pix_of,  # noqa: E402
                                 channel_velocities, MASK_FRAC_RED_FLAG)
from uniform_batch_configs import build_table              # noqa: E402
from known_lines import LINES_REST_HZ                      # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
S_FROZEN = 1.0

REST_HZ = {
    "H30a": LINES_REST_HZ["H30a"], "H40a": LINES_REST_HZ["H40a"],
    "CO21": LINES_REST_HZ["CO2-1"], "CS21": LINES_REST_HZ["CS2-1"],
    "HCO+10": LINES_REST_HZ["HCO+1-0"], "HCN10": LINES_REST_HZ["HCN1-0"],
    "CO10": LINES_REST_HZ["CO1-0"], "13CO21": LINES_REST_HZ["13CO2-1"],
}
# config-key → known_lines catalog-key (for line_free_mask_for_cube)
CAT_KEY = {"H30a": "H30a", "H40a": "H40a", "CO21": "CO2-1",
           "CS21": "CS2-1", "HCO+10": "HCO+1-0", "HCN10": "HCN1-0",
           "CO10": "CO1-0", "13CO21": "13CO2-1"}

# strong nonpbcor deleted for NGC 253 — flatten pbcor mom-0 with the MFS PB
PB_MULT = {"ngc253": "/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/"
                     "NGC253_spw31_pb_mfs.fits"}

CSV_PATH = OUT / "stage1_masks.csv"
CSV_FIELDS = ["galaxy", "strong_line", "strong_kind", "beam_arcsec",
              "nbeam", "n_islands_raw", "n_islands_kept", "mask_npix",
              "fov_frac", "seed_ok", "flag_extended", "flag_no_seed",
              "class_suggestion", "note"]


def process(row):
    name = row["galaxy"]
    print(f"\n=== {name} ===", flush=True)
    strong_path = row["strong_path"]
    if name in PB_MULT and (not strong_path or not Path(strong_path).exists()):
        strong_path = row["hrl_path"]      # single-cube: pbcor + MFS-PB flatten
        row["strong_kind"] = "pbcor+pbmult"
    if not strong_path or not Path(strong_path).exists():
        return dict(galaxy=name, note="NO STRONG-TRACER FILE — skipped",
                    class_suggestion="no-data")
    cube = load_cube(strong_path)
    hdr = cube.header
    z = float(row["z"])
    rest = REST_HZ[row["strong_line"]] / (1.0 + z)
    v, _ = channel_velocities(hdr, rest)
    dv = float(np.abs(np.median(np.diff(v))))
    win = float(row["window_kms"])
    beam_area = get_beam_area_pix(cube)
    bfp = beam_fwhm_pix_of(hdr)
    beam_arcsec = hdr["BMAJ"] * 3600.0

    cache = OUT / f"{name}_strong_mom0.npy"
    in_win = np.abs(v) <= win / 2.0
    ch_idx = np.where(in_win)[0]
    if len(ch_idx) == 0:
        cube.data = None
        return dict(galaxy=name, note="strong line outside cube band?",
                    class_suggestion="review")
    if cache.exists():
        mom0 = np.load(cache)
        finite2d = np.isfinite(mom0)
    else:
        mom0 = np.zeros(cube.data.shape[1:], dtype=np.float64)
        for ch in ch_idx:
            mom0 += np.nan_to_num(np.asarray(cube.data[ch], dtype=np.float64))
        mom0 *= dv
        finite2d = np.isfinite(np.asarray(cube.data[ch_idx[0]]))
        pbm = row.get("pb_mult") or PB_MULT.get(name)
        if pbm:
            pb = np.squeeze(fits.getdata(pbm))
            mom0 = mom0 * np.nan_to_num(pb)
            finite2d &= np.isfinite(pb) & (pb > 0.2)
        mom0[~finite2d] = np.nan
        np.save(cache, mom0)
    cube.data = None

    mask, skirt, n_isl, n_kept, work = build_mask(
        mom0, finite2d, bfp, beam_area, S_FROZEN)
    seed_ok = bool(mask.any())
    frac = float(mask.sum() / max(finite2d.sum(), 1))
    flag_ext = frac > MASK_FRAC_RED_FLAG
    flag_seed = not seed_ok
    cls = ("B-extended?" if flag_ext else
           "review-no-seed" if flag_seed else "A")
    print(f"  beam {beam_arcsec:.2f}\"  islands {n_isl}→{n_kept}  "
          f"nbeam {mask.sum()/beam_area:.1f}  FoV {frac:.3f}  → {cls}",
          flush=True)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    show = np.where(finite2d, work, np.nan)
    vmax = np.nanpercentile(show, 99.7) if np.isfinite(show).any() else 1
    ax.imshow(show, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    if seed_ok:
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.0)
    ax.set_title(f"{name} {row['strong_line']}  {n_kept} isl  "
                 f"{mask.sum()/beam_area:.0f} bm  FoV {frac:.2f}  {cls}",
                 fontsize=10)
    (OUT / "thumbs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT / "thumbs" / f"{name}.png", dpi=110)
    plt.close()

    return dict(galaxy=name, strong_line=row["strong_line"],
                strong_kind=row["strong_kind"],
                beam_arcsec=round(beam_arcsec, 3),
                nbeam=round(float(mask.sum() / beam_area), 1),
                n_islands_raw=n_isl, n_islands_kept=n_kept,
                mask_npix=int(mask.sum()), fov_frac=round(frac, 4),
                seed_ok=seed_ok, flag_extended=flag_ext,
                flag_no_seed=flag_seed, class_suggestion=cls, note="")


def main():
    OUT.mkdir(exist_ok=True)
    rows = build_table()
    # small cubes first → early table rows while big ones grind
    rows.sort(key=lambda r: Path(r["strong_path"]).stat().st_size
              if r["strong_path"] and Path(r["strong_path"]).exists() else 0)
    done, fieldnames = set(), CSV_FIELDS
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            rd = csv.DictReader(f)
            fieldnames = rd.fieldnames          # honor post-hoc columns
            done = {r["galaxy"] for r in rd}
        print(f"resuming — {len(done)} galaxies already in CSV", flush=True)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            if row["galaxy"] in done:
                continue
            try:
                out = process(row)
            except Exception as e:
                import traceback; traceback.print_exc()
                out = dict(galaxy=row["galaxy"], note=f"ERROR: {e}",
                           class_suggestion="error")
            if "final_class" in fieldnames and "final_class" not in out:
                # new rows: clean A passes straight through; anything
                # flagged awaits user review (conservative overnight rule)
                out["final_class"] = ("A" if out.get("class_suggestion") == "A"
                                      else f"review-{out.get('class_suggestion')}")
            w.writerow({k: out.get(k, "") for k in fieldnames})
            f.flush()
    print(f"\nstage 1 complete → {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
