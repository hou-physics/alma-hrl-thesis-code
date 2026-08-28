"""Uniform-prescription batch — STAGE 2: HRL flux for every class-A source.

Per source (final_class A / A-domain-flagged in stage1_masks.csv):
  1. rebuild frozen mask from the stage-1 cached mom-0 (s=1, 2σ + 5σ seed);
  2. bright-line spectrum → W20 + centroid; coherence gate (W20 unmeasurable
     → data-quality exit, no silent numbers);
  3. HRL pbcor pass (single read, all masks): flux + empirical σ + S/N,
     skirt + ≤16 translation nulls as diagnostics;
  4. verdict: S/N ≥ 3 → detection marker, else upper limit (all sources get
     flux ± 1σ regardless — correlation-plot convention).

Crash-safe: rows append to CSV as they finish; rerun skips completed galaxies.

Run: conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_stage2.py [galaxy ...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube, get_beam_area_pix          # noqa: E402
from flux import calc_flux_sn                              # noqa: E402
from known_lines import LINES_REST_HZ, line_free_mask_for_cube  # noqa: E402
from uniform_contour_poc import (build_mask, beam_fwhm_pix_of,  # noqa: E402
                                 channel_velocities, multi_mask_spectra,
                                 line_width, place_nulls, W_FLOOR_KMS,
                                 smooth_spectrum_kms)
from uniform_batch_configs import build_table              # noqa: E402
from uniform_batch_stage1 import REST_HZ, PB_MULT, CAT_KEY  # noqa: E402

OUT = Path("/Volumes/HouAstro/master/result_v2/_uniform_batch")
S_FROZEN = 1.0
K_NULLS = 16
W20_BOUND_CAP = 600.0      # v3.2 guardrail 1: no sample-class galaxy is
#                            broader; beyond ±600 km/s "width" is not rotation
CSV_PATH = OUT / "stage2_flux.csv"
CSV_FIELDS = ["galaxy", "hrl_line", "strong_line", "final_class",
              "W20_raw", "W_used", "v_cen", "nbeam", "n_islands",
              "F", "sigma_F", "SN", "verdict",
              "F_skirt", "SN_skirt", "null_mean", "null_std", "n_nulls",
              "coherent", "note"]


def stage1_class():
    with open(OUT / "stage1_masks.csv") as f:
        return {r["galaxy"]: r["final_class"] for r in csv.DictReader(f)}


def process(row, final_class):
    name = row["galaxy"]
    print(f"\n=== {name} ({final_class}) ===", flush=True)
    z = float(row["z"])

    mom0 = np.load(OUT / f"{name}_strong_mom0.npy")
    finite2d = np.isfinite(mom0)

    strong_path = row["strong_path"]
    if name in PB_MULT and (not strong_path or not Path(strong_path).exists()):
        strong_path = row["hrl_path"]
    single_cube = strong_path == row["hrl_path"]

    strong = load_cube(strong_path)
    s_hdr = strong.header
    beam_area_s = get_beam_area_pix(strong)
    bfp = beam_fwhm_pix_of(s_hdr)
    rest_s = REST_HZ[row["strong_line"]] / (1.0 + z)
    v_s, _ = channel_velocities(s_hdr, rest_s)
    dv_s = float(np.abs(np.median(np.diff(v_s))))
    win = float(row["window_kms"])

    mask, skirt, n_isl_raw, n_kept, work = build_mask(
        mom0, finite2d, bfp, beam_area_s, S_FROZEN)
    if not mask.any():
        strong.data = None
        return dict(galaxy=name, final_class=final_class, coherent=False,
                    verdict="review", note="no 5σ seed at stage 2")
    nbeam = float(mask.sum() / beam_area_s)

    # bright-line spectrum → W20 / centroid / coherence.
    # v3.2: adaptive search bound — a pinned W20 (extent reaching the
    # bound) extends the bound stepwise up to the cap (guardrail 1) and
    # never past the nearest known-line contaminant (guardrail 2);
    # still pinned at the limit → W20 unmeasurable → review, no number.
    _, freqs_s = channel_velocities(s_hdr, rest_s)
    pad = int(round((W20_BOUND_CAP + 200.0) / dv_s))
    ch_win = np.where(np.abs(v_s) <= win / 2.0)[0]
    lo, hi = max(0, ch_win.min() - pad), min(len(v_s) - 1, ch_win.max() + pad)
    spec_s = multi_mask_spectra(strong.data, [mask], beam_area_s, lo, hi)[0]
    keep_s = line_free_mask_for_cube(
        v_chan=v_s, primary_line_rest_hz=rest_s,
        cube_freq_range_hz=(float(freqs_s.min()), float(freqs_s.max())),
        z=z, primary_line_key=CAT_KEY.get(row["strong_line"],
                                          row["strong_line"]),
        fwhm_kms=300.0)
    contam_v = np.abs(v_s[~keep_s])
    line_stop = float(contam_v.min()) - 50.0 if (~keep_s).any() else np.inf
    bound_max = min(W20_BOUND_CAP, line_stop)

    bound = win / 2.0 + 100.0
    while True:
        b = min(bound, bound_max)
        base_ok = (np.abs(v_s) > b) & keep_s & np.isfinite(spec_s)
        base = np.nanmedian(spec_s[base_ok])
        spec_s_meas = smooth_spectrum_kms(spec_s - base, dv_s)   # v3.1
        w20, v_cen = line_width(v_s, spec_s_meas, b, 0.2)
        pinned = (np.isfinite(w20) and np.isfinite(v_cen)
                  and abs(v_cen) + w20 / 2.0 >= b - max(2 * dv_s, 10.0))
        if not pinned or b >= bound_max:
            break
        bound = min(bound + 150.0, bound_max)
    spec_s_sub = spec_s - base
    coherent = np.isfinite(w20) and np.isfinite(v_cen) and not pinned
    if not single_cube:
        strong.data = None
    if not coherent:
        if single_cube:
            strong.data = None
        why = (f"W20 pinned at ±{b:.0f} bound (cap {W20_BOUND_CAP:.0f}, "
               f"line-stop {line_stop:.0f}) — tracer too weak/broad"
               if pinned else
               "bright-line spectrum incoherent (W20 unmeasurable)")
        return dict(galaxy=name, final_class=final_class, coherent=False,
                    n_islands=n_kept, nbeam=round(nbeam, 1),
                    verdict="review", note=why)
    w_used = max(w20, W_FLOOR_KMS)
    bnote = f" [bound ±{b:.0f}]" if b > win / 2.0 + 100.0 else ""
    print(f"  mask {n_kept} isl, {nbeam:.1f} bm; W20={w20:.0f} → W={w_used:.0f}; "
          f"v_cen={v_cen:+.0f}{bnote}", flush=True)

    # HRL pass
    hrl = strong if single_cube else load_cube(row["hrl_path"])
    h_hdr = hrl.header
    rest_h = REST_HZ[row["weak_line"]] / (1.0 + z)
    v_h, freqs_h = channel_velocities(h_hdr, rest_h)
    dv_h = float(np.abs(np.median(np.diff(v_h))))
    beam_area_h = get_beam_area_pix(hrl)

    mid = len(v_h) // 2
    valid = np.isfinite(np.asarray(hrl.data[mid]))
    rng = np.random.default_rng(42)
    shifts = place_nulls(rng, mask, valid, bfp, K_NULLS)
    ys, xs = np.where(mask)
    null_masks = []
    for dy, dx in shifts:
        m = np.zeros_like(mask)
        m[ys + dy, xs + dx] = True
        null_masks.append(m)

    center_ch = int(np.argmin(np.abs(v_h - v_cen)))
    span = max(w_used / 2.0, 150.0) + 700.0
    ch_lo = max(0, center_ch - int(round(span / dv_h)))
    ch_hi = min(len(v_h) - 1, center_ch + int(round(span / dv_h)))
    specs = multi_mask_spectra(hrl.data, [mask, skirt] + null_masks,
                               beam_area_h, ch_lo, ch_hi)
    hrl.data = None
    if single_cube:
        strong.data = None

    # v3.2: contaminant excision width follows the galaxy's own line width
    # (line width is global — a 500 km/s galaxy has 500 km/s contaminants)
    keep = line_free_mask_for_cube(
        v_chan=v_h, primary_line_rest_hz=rest_h,
        cube_freq_range_hz=(float(freqs_h.min()), float(freqs_h.max())),
        z=z, primary_line_key=row["weak_line"],
        fwhm_kms=max(300.0, w_used))
    hrl_zone = np.abs(v_h - v_cen) <= max(w_used, 300.0) / 2.0
    specs[:, ~(keep | hrl_zone)] = np.nan

    excl = max(300.0, w_used / 2.0 + 60.0)
    F, sn, _, _ = calc_flux_sn(specs[0], center_ch, w_used, dv_h,
                               baseline_exclude_half_width_kms=excl)
    F_sk, sn_sk, _, _ = calc_flux_sn(specs[1], center_ch, w_used, dv_h,
                                     baseline_exclude_half_width_kms=excl)
    f_nulls = np.array([calc_flux_sn(sp, center_ch, w_used, dv_h,
                                     baseline_exclude_half_width_kms=excl)[0]
                        for sp in specs[2:]])
    sigma = F / sn if sn else np.nan
    verdict = "detection" if sn >= 3 else "upper-limit"
    print(f"  F={F:.3f} ± {sigma:.3f}  S/N={sn:.1f}  → {verdict}  "
          f"nulls {len(shifts)}", flush=True)

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    from uniform_batch_stage4_plot import display_name
    PRETTY = {"CO21": "CO(2-1)", "CS21": "CS(2-1)", "HCO+10": "HCO+(1-0)",
              "H30a": "H30α", "H40a": "H40α"}
    ax = axes[0]
    show = np.where(finite2d, work, np.nan)
    vmax = np.nanpercentile(show, 99.7)
    ax.imshow(show, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.0)
    # Panel titles name the source and the line only.  Fluxes, uncertainties
    # and tiers live in the caption and the survey table; printing a third
    # copy here is the one place they can disagree, and it did — these
    # titles carried the nominal sigma and S/N while the text carries the
    # calibrated ones (2026-08-26).
    ax.set_title(f"{display_name(name)}  {PRETTY.get(row['strong_line'], row['strong_line'])} mask",
                 fontsize=15)
    ax = axes[1]
    ax.step(v_s, spec_s_sub, where="mid", color="C0", lw=1)
    ax.axvspan(v_cen - w_used / 2, v_cen + w_used / 2, alpha=0.15, color="C3")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(-win - 100, win + 100)
    ax.set_title(f"{PRETTY.get(row['strong_line'], row['strong_line'])} profile", fontsize=15)
    ax.set_xlabel("v [km/s]"); ax.set_ylabel("S [Jy]")
    ax = axes[2]
    for sp in specs[2:]:
        ax.step(v_h, sp, where="mid", color="gray", lw=0.4, alpha=0.35)
    ax.step(v_h, specs[0], where="mid", color="C0", lw=1, label="signal")
    ax.step(v_h, specs[1], where="mid", color="C1", lw=0.7, label="annulus")
    ax.axvspan(v_cen - w_used / 2, v_cen + w_used / 2, alpha=0.15, color="C3")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(v_h[ch_lo], v_h[ch_hi])
    ax.set_title(f"{PRETTY.get(row['weak_line'], row['weak_line'])} spectrum", fontsize=15)
    ax.set_xlabel("v [km/s]"); ax.set_ylabel("S [Jy]")
    ax.legend(fontsize=11)
    for a in axes:
        a.tick_params(labelsize=12)
        a.xaxis.label.set_size(13); a.yaxis.label.set_size(13)
    plt.tight_layout()
    (OUT / "stage2_figs").mkdir(exist_ok=True)
    plt.savefig(OUT / "stage2_figs" / f"{name}.png", dpi=120)
    plt.close()

    return dict(galaxy=name, hrl_line=row["weak_line"],
                strong_line=row["strong_line"], final_class=final_class,
                W20_raw=round(w20, 1), W_used=round(w_used, 1),
                v_cen=round(v_cen, 1), nbeam=round(nbeam, 1),
                n_islands=n_kept, F=round(F, 4),
                sigma_F=round(sigma, 4), SN=round(sn, 2), verdict=verdict,
                F_skirt=round(F_sk, 4), SN_skirt=round(sn_sk, 2),
                null_mean=round(float(np.mean(f_nulls)), 4) if len(f_nulls) else "",
                null_std=round(float(np.std(f_nulls)), 4) if len(f_nulls) else "",
                n_nulls=len(shifts), coherent=True, note="")


def main():
    classes = stage1_class()
    argv = list(sys.argv[1:])
    # --redraw regenerates the per-source figures for galaxies that are
    # already in the CSV, without writing a row.  The CSV is the source of
    # truth for every number in the thesis; a figure refresh (a changed panel
    # title, a corrected scale) must not be able to move it.
    redraw = "--redraw" in argv
    only = set(a for a in argv if not a.startswith("--"))
    rows = [r for r in build_table()
            if classes.get(r["galaxy"], "").startswith("A")
            and (not only or r["galaxy"] in only)]
    rows.sort(key=lambda r: Path(r["strong_path"]).stat().st_size
              if r["strong_path"] and Path(r["strong_path"]).exists() else 0)
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            done = {r["galaxy"] for r in csv.DictReader(f)}
        print(f"resuming — {len(done)} done", flush=True)

    if redraw:
        prev = {}
        with open(CSV_PATH) as f:
            prev = {r["galaxy"]: r for r in csv.DictReader(f)}
        for row in rows:
            g = row["galaxy"]
            if g not in prev:
                print(f"  {g}: not in CSV, skipped", flush=True)
                continue
            out = process(row, classes[g])
            # The rerun must reproduce the stored numbers exactly; if it does
            # not, the figure and the table would disagree and the run is
            # reported rather than silently accepted.
            for k in ("F", "sigma_F", "SN", "W_used"):
                a, b = prev[g].get(k, ""), out.get(k, "")
                if a and b != "" and abs(float(a) - float(b)) > 5e-4:
                    print(f"  !! {g} {k}: CSV {a} vs rerun {b}", flush=True)
            print(f"  {g}: figure refreshed, CSV untouched", flush=True)
        print("\nredraw complete — stage2_flux.csv not modified", flush=True)
        return

    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            if row["galaxy"] in done:
                continue
            try:
                out = process(row, classes[row["galaxy"]])
            except Exception as e:
                import traceback; traceback.print_exc()
                out = dict(galaxy=row["galaxy"], verdict="error",
                           note=f"ERROR: {e}")
            w.writerow({k: out.get(k, "") for k in CSV_FIELDS})
            f.flush()
    print(f"\nstage 2 complete → {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
