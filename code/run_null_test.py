"""Batch null-test runner for Phase Y pipeline.

For each target galaxy:
  1. Import its science `AnalysisConfig` from `{galaxy}_analyse_code/step3_analyze.py`.
  2. Query HRL cube bandwidth → safe offset range [−(bw/2 − 600), +(bw/2 − 600)] km/s.
  3. Pick N offsets uniformly, excluding |offset| < 300 km/s (real-line zone).
  4. Run step3 pipeline for each offset:
       - First offset: full plots + summary.md in results/{GAL}/nulltest/offset_{+XXXX}/
       - Other offsets: skip_plots=True, output only a CSV row to nulltest_samples.csv

Targets split into:
  - threshold-group (used to build the 95th-percentile effective 3σ line):
      NGC 6810, 7130, 3227, IRAS F18293-3413, IC 5063, NGC 1386
  - control-group (health check only, flagged in CSV, not used for threshold):
      NGC 3628, NGC 4945

Usage (from repo root):
    /opt/anaconda3/envs/casa_env/bin/python master_thesis/my_code/run_null_test.py [GALAXY ...]

With no args, runs all 8 targets. Specific galaxy names skip the others.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

REPO = Path("/Volumes/HouAstro/master")
CODE_ROOT = REPO / "master_thesis" / "my_code"
STEP3 = CODE_ROOT / "_step3"
RESULTS = REPO / "results"
CSV_PATH = RESULTS / "_nulltest" / "nulltest_samples.csv"

sys.path.insert(0, str(STEP3))

THRESHOLD_GROUP = [
    ("NGC6810", "ngc6810_analyse_code"),
    ("NGC7130", "ngc7130_analyse_code"),
    ("NGC3227", "ngc3227_analyse_code"),
    ("IRASF18293-3413", "irasf18293-3413_analyse_code"),
    ("IC5063", "ic5063_analyse_code"),
    ("NGC1386", "ngc1386_analyse_code"),
    ("NGC3627", "ngc3627_analyse_code"),
]
CONTROL_GROUP = [
    ("NGC3628", "ngc3628_analyse_code"),
    ("NGC4945", "ngc4945_analyse_code"),
    ("NGC5135", "ngc5135_analyse_code"),
    ("He2-10", "he2-10_analyse_code"),
    ("ESO097-013", "eso097-013_analyse_code"),
]

N_OFFSETS = 16
SCIENCE_ZONE_KMS = 300.0   # skip |offset| < 300 km/s (where real line lives)
EDGE_BUFFER_KMS = 600.0    # buffer from cube edges: 300 (noise excl) + 250 (int half) + pad
C_KMS = 299792.458

# Known molecular + recombination lines that commonly fall in mm-wave science
# cubes. Used by `_other_line_zones()` to build extra null-test exclusion zones
# for any line (besides the primary HRL + primary strong tracer) whose rest
# frequency falls inside the cube band. Critical for AGN-host / dense-gas-rich
# cubes where multiple lines coexist in one SPW (Circinus +825 spike traced to
# SO 2_3-1_2 at rest 99.300 GHz, verified visually 2026-05-13 — without this
# exclusion, that real SO line is mis-counted as null-distribution noise and
# inflates p95 from ~3 to >10).
# Rydberg-checked 2026-08-26: every H entry below satisfies
# nu = R_H*c * (1/n^2 - 1/(n+dn)^2), R_H*c = 3.288052e15 Hz, to < 0.02%.
# Ten entries did not and were corrected (H25a/H27a/H28a and the beta series
# from H49b up, drifting to +853 MHz by H56b).  None of them lies inside any
# SPW used in this work, so no published number changes.  Same defect class as
# the H39beta error in _step3/known_lines.py; see docs/decisions.md 2026-08-26.
KNOWN_LINES_GHZ = {
    # Hydrogen recombination α series (n+1 → n; Lyman-α conversion factor)
    "H21a": 662.404, "H23a": 507.176, "H25a": 396.901, "H26a": 353.622,
    "H27a": 316.416, "H28a": 284.251, "H29a": 256.302,  # actually H29a = 256.302053
    "H30a": 231.901, "H31a": 210.502, "H32a": 191.657, "H33a": 174.996,
    "H34a": 160.211, "H35a": 147.046, "H36a": 135.286, "H37a": 124.747,
    "H38a": 115.274, "H39a": 106.737, "H40a":  99.023, "H41a":  92.034,
    "H42a":  85.688, "H43a":  79.913, "H44a":  74.644,
    # Hydrogen recombination β series (n+2 → n; intrinsic ~1/8–1/10 of α)
    "H49b": 105.302, "H50b":  99.225, "H51b": 93.607, "H52b": 88.406,
    "H53b": 83.582, "H54b": 79.104, "H55b": 74.940, "H56b": 71.063,
    # CO isotopologues + tracers commonly in same SPW as HRLs
    "CO_1-0":    115.271, "CO_2-1":    230.538, "CO_3-2":    345.796,
    "13CO_1-0":  110.201, "13CO_2-1":  220.399,
    "C18O_1-0":  109.782, "C18O_2-1":  219.560,
    "CN_1-0":     113.491,
    # Dense-gas tracers
    "HCN_1-0":   88.632, "HCN_2-1":  177.261, "HCN_3-2":  265.886, "HCN_4-3":  354.505,
    "HCO+_1-0":  89.189, "HCO+_2-1": 178.375, "HCO+_3-2": 267.558, "HCO+_4-3": 356.734,
    "HNC_1-0":   90.664, "HNC_2-1":  181.325,
    "CS_2-1":    97.981, "CS_3-2":   146.969, "CS_4-3":   195.954, "CS_5-4":   244.936,
    "CS_7-6":    342.883,
    "C34S_2-1":  96.413, "C34S_3-2": 144.617,
    # Sulfur monoxide (key AGN / shock tracer; SO 2_3-1_2 = Circinus +825 spike)
    "SO_2_3-1_2": 99.300, "SO_3_4-2_3":  138.179, "SO_5_4-4_3": 100.029,
    "SO_4_3-3_2": 158.971, "SO_6_5-5_4": 219.949, "SO_5_6-4_5": 251.826,
    # Silicon monoxide (shock tracer)
    "SiO_2-1": 86.847, "SiO_4-3": 173.688, "SiO_5-4": 217.105,
    # HC3N (high-density, AGN-warm-gas tracer)
    "HC3N_10-9": 90.979, "HC3N_11-10": 100.076, "HC3N_12-11": 109.174,
    "HC3N_24-23": 218.325,
    # Methanol common transitions
    "CH3OH_2-1_E": 96.741, "CH3OH_5-4_0E": 241.700,
    # Cyclic C3H2
    "C3H2_2-1": 85.339,
}


def _other_line_zones(hrl_pbcor_path: str, co_pbcor_path: str, z: float,
                       weak_rest_ghz: float, weak_line_name: str, strong_line_name: str,
                       half_window_kms: float = 300.0,
                       margin_kms: float = 50.0) -> List[Tuple[float, float]]:
    """Return list of (offset_min, offset_max) zones for ALL known lines in
    KNOWN_LINES_GHZ that fall inside the cube freq range (single-cube case only).

    Excludes the primary weak (HRL) line and primary strong tracer — those are
    already handled by SCIENCE_ZONE_KMS and the dedicated `_co_contam_zone()`
    function. This function catches secondary lines that the previous design
    didn't anticipate (e.g. SO 2_3-1_2 at +839 km/s from H40α, or H50β at +687
    km/s — both in same Band 3 SPW as H40α + CS21).
    """
    if hrl_pbcor_path != co_pbcor_path:
        return []  # different cubes: no extra contamination
    with fits.open(hrl_pbcor_path) as hdul:
        hdr = hdul[0].header
        nchan = int(hdr["NAXIS3"])
        crval3 = float(hdr["CRVAL3"])
        crpix3 = float(hdr["CRPIX3"])
        cdelt3 = float(hdr["CDELT3"])
    # Observed-frame cube range (Hz)
    f_chan1 = crval3 + (1 - crpix3) * cdelt3
    f_chanN = crval3 + (nchan - crpix3) * cdelt3
    obs_lo, obs_hi = sorted([f_chan1, f_chanN])
    rest_lo = obs_lo * (1.0 + z) / 1e9
    rest_hi = obs_hi * (1.0 + z) / 1e9

    weak_obs_ghz = weak_rest_ghz / (1.0 + z)
    zones: List[Tuple[float, float]] = []
    primary_names_norm = {weak_line_name.lower().replace("_", "").replace("-", ""),
                          strong_line_name.lower().replace("_", "").replace("-", "")}
    for name, rest_ghz in KNOWN_LINES_GHZ.items():
        # Skip primary HRL + primary strong tracer (already excluded elsewhere)
        if name.lower().replace("_", "").replace("-", "") in primary_names_norm:
            continue
        if not (rest_lo <= rest_ghz <= rest_hi):
            continue  # not in cube
        line_obs_ghz = rest_ghz / (1.0 + z)
        offset_kms = (line_obs_ghz - weak_obs_ghz) / weak_obs_ghz * C_KMS
        half_zone = half_window_kms + margin_kms
        zones.append((offset_kms - half_zone, offset_kms + half_zone))
    return zones


def _import_galaxy_config(script_path: Path):
    """Load `config` object from a galaxy's step3_analyze.py."""
    spec = importlib.util.spec_from_file_location("gal_cfg", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def _compute_offset_range_kms(hrl_fits_path: str, z: float,
                               weak_rest_ghz: float) -> Tuple[float, float]:
    """Return (min_offset_kms, max_offset_kms) inside which the shifted HRL
    center stays ≥ EDGE_BUFFER_KMS from either cube edge."""
    with fits.open(hrl_fits_path) as hdul:
        hdr = hdul[0].header
        nchan = int(hdr["NAXIS3"])
        crval3 = float(hdr["CRVAL3"])
        crpix3 = float(hdr["CRPIX3"])
        cdelt3 = float(hdr["CDELT3"])
    obs_ghz = weak_rest_ghz / (1.0 + z)
    obs_hz = obs_ghz * 1e9
    ch_center = crpix3 - 1 + (obs_hz - crval3) / cdelt3
    chan_width_kms = C_KMS * abs(cdelt3) / obs_hz

    buffer_ch = EDGE_BUFFER_KMS / chan_width_kms
    min_center_ch = buffer_ch
    max_center_ch = (nchan - 1) - buffer_ch
    min_offset_kms = (min_center_ch - ch_center) * chan_width_kms
    max_offset_kms = (max_center_ch - ch_center) * chan_width_kms
    if cdelt3 < 0:
        # Frequency decreases with channel → positive velocity = higher channel.
        # But our sign convention in _shift_center_for_null: +offset_kms →
        # +offset_ch = +offset_kms/chan_width_kms (where chan_width is abs).
        # So regardless of cdelt sign, +offset_kms means shifting integration
        # window along the positive-channel direction. Range above is correct.
        pass
    return float(min_offset_kms), float(max_offset_kms)


def _subtract_zone(segments: List[Tuple[float, float]],
                   zone_min: float, zone_max: float) -> List[Tuple[float, float]]:
    """Remove [zone_min, zone_max] from a list of inclusive segments."""
    out: List[Tuple[float, float]] = []
    for a, b in segments:
        if zone_max <= a or zone_min >= b:
            out.append((a, b))
            continue
        if zone_min > a:
            out.append((a, zone_min))
        if zone_max < b:
            out.append((zone_max, b))
    return out


def _pick_offsets(min_kms: float, max_kms: float, n: int,
                  skip_kms: float = SCIENCE_ZONE_KMS,
                  co_contam_zone: Optional[Tuple[float, float]] = None,
                  extra_zones: Optional[List[Tuple[float, float]]] = None) -> List[float]:
    """Pick n offsets uniformly in [min, max] excluding [-skip, +skip], the
    primary CO contamination zone (single-cube case), AND any number of extra
    exclusion zones for other known lines in the same SPW (e.g. SO, H50β).

    extra_zones: list of (offset_min, offset_max) tuples — each marks a velocity
    range where a known non-target line emits and would mis-populate the null
    distribution if sampled. Added 2026-05-13 after Circinus +825 km/s spike
    traced to SO 2_3-1_2 line.
    """
    segments: List[Tuple[float, float]] = []
    if min_kms < -skip_kms:
        segments.append((min_kms, -skip_kms))
    if max_kms > skip_kms:
        segments.append((skip_kms, max_kms))
    if not segments:
        return []
    if co_contam_zone is not None:
        segments = _subtract_zone(segments, co_contam_zone[0], co_contam_zone[1])
    if extra_zones:
        for zlo, zhi in extra_zones:
            segments = _subtract_zone(segments, zlo, zhi)
    if not segments:
        return []

    total_span = sum(b - a for a, b in segments)
    offsets: List[float] = []
    for a, b in segments:
        frac = (b - a) / total_span
        k = max(1, int(round(n * frac)))
        offsets.extend(np.linspace(a, b, k).tolist())
    offsets = [o for o in offsets if abs(o) >= skip_kms - 1e-6]
    return offsets[:n]


def _co_contam_zone(hrl_pbcor_path: str, co_pbcor_path: str, z: float,
                    weak_rest_ghz: float, strong_rest_ghz: float,
                    max_width_kms: float, half_window_kms: float = 300.0,
                    margin_kms: float = 50.0) -> Optional[Tuple[float, float]]:
    """Return the (offset_min, offset_max) CO contamination zone in HRL-center
    velocity frame, or None for dual-cube cases where HRL and CO live in
    different cubes (no offset can land on CO).

    For single-cube setups (HRL pbcor path == CO pbcor path), the integration
    window contaminates CO when it overlaps the CO ±half_window region.
    """
    if hrl_pbcor_path != co_pbcor_path:
        return None
    weak_obs = weak_rest_ghz / (1.0 + z)
    strong_obs = strong_rest_ghz / (1.0 + z)
    co_offset = -(weak_obs - strong_obs) / weak_obs * C_KMS
    half_zone = max_width_kms / 2.0 + half_window_kms + margin_kms
    return (co_offset - half_zone, co_offset + half_zone)


def _look_up_rest_ghz(line: str, fallback_from_cfg: float | None) -> float:
    """Return HRL rest freq; prefer explicit cfg value, else use analyze's table."""
    if fallback_from_cfg is not None:
        return fallback_from_cfg
    from analyze import _look_up_rest_freq  # type: ignore
    return _look_up_rest_freq(line)


def _get_real_line_scan_best(display_name: str) -> Optional[Tuple[float, float, int, float]]:
    """Read results/{galaxy}/tables/results.txt; return (smooth, sthresh, vel_width, sn)
    of the row with maximum scan-best SN. Returns None if file missing/empty.

    Used by --fixed-mask mode: the real-line scan params are locked across all
    null offsets, so each offset uses the SAME mask shape + integration window.
    This removes the selector bias that inflates null p95 when each offset
    re-scans (~1000 trials) and reports max(SN) — that operation is
    'pick the highest SN under noise', which by construction returns 10-16σ
    on pure noise via look-elsewhere. Fixed-mask null measures true Gaussian
    look-elsewhere noise at the single optimal mask configuration.
    """
    path = RESULTS / display_name / "tables" / "results.txt"
    if not path.exists():
        return None
    with open(path) as fh:
        rows = []
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                # cols: smooth sthresh vel_width nbeam mask_width flux sn peak
                rows.append((float(parts[0]), float(parts[1]), int(float(parts[2])),
                             float(parts[6])))
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    best = max(rows, key=lambda r: r[3])
    return best  # (smooth, sthresh, vel_width, sn)


def run_one_galaxy(display_name: str, script_subdir: str, is_control: bool,
                    fixed_mask: bool = False) -> None:
    from analyze import analyze, ConflictReportError  # type: ignore

    script_path = CODE_ROOT / script_subdir / "step3_analyze.py"
    base_cfg = _import_galaxy_config(script_path)
    weak_rest = _look_up_rest_ghz(base_cfg.weak_line, base_cfg.weak_line_rest_ghz)

    # Fixed-mask mode: lock scan params to real-line best (selector-bias-free null).
    fixed_scan_kwargs = {}
    if fixed_mask:
        best = _get_real_line_scan_best(display_name)
        if best is None:
            print(f"\n=== {display_name} === SKIP fixed-mask: no real-line results.txt")
            print(f"   (run the real-line analysis first to populate tables/results.txt)")
            return
        b_smooth, b_sthresh, b_vw, b_sn = best
        fixed_scan_kwargs = dict(
            smoothing_factors=[int(b_smooth)],
            spatial_thresholds=[float(b_sthresh)],
            velocity_widths_kms=[int(b_vw)],
        )
        print(f"   FIXED MASK from real-line best: smooth={b_smooth:g}, "
              f"sthresh={b_sthresh:g}, vel_width={b_vw:g} → real SN={b_sn:.2f}")

    min_kms, max_kms = _compute_offset_range_kms(
        base_cfg.hrl_pbcor_path, base_cfg.z, weak_rest,
    )
    strong_rest = _look_up_rest_ghz(base_cfg.strong_line, base_cfg.strong_line_rest_ghz)
    co_zone = _co_contam_zone(
        base_cfg.hrl_pbcor_path, base_cfg.co_pbcor_path, base_cfg.z,
        weak_rest, strong_rest,
        max_width_kms=max(base_cfg.velocity_widths_kms),
        half_window_kms=base_cfg.strong_line_window_kms,
    )
    extra_line_zones = _other_line_zones(
        base_cfg.hrl_pbcor_path, base_cfg.co_pbcor_path, base_cfg.z,
        weak_rest, base_cfg.weak_line, base_cfg.strong_line,
        half_window_kms=base_cfg.strong_line_window_kms,
    )
    offsets = _pick_offsets(min_kms, max_kms, N_OFFSETS,
                            co_contam_zone=co_zone, extra_zones=extra_line_zones)
    print(f"\n=== {display_name} "
          f"({'CONTROL' if is_control else 'THRESHOLD'}) ===")
    co_msg = f", CO contam excluded {co_zone[0]:+.0f}..{co_zone[1]:+.0f}" if co_zone else ""
    print(f"  Safe offset range: [{min_kms:+.0f}, {max_kms:+.0f}] km/s{co_msg}; "
          f"picked {len(offsets)} offsets.")
    if extra_line_zones:
        # Identify which lines were excluded (for log clarity)
        for (zlo, zhi) in extra_line_zones:
            # Find which line corresponds to this offset zone (re-derive)
            mid = 0.5 * (zlo + zhi)
            weak_obs = weak_rest / (1.0 + base_cfg.z)
            implied_rest_ghz = weak_obs * (1.0 + mid / C_KMS) * (1.0 + base_cfg.z)
            best_match, best_dghz = None, 1e6
            for nm, rg in KNOWN_LINES_GHZ.items():
                if abs(rg - implied_rest_ghz) < best_dghz:
                    best_match, best_dghz = nm, abs(rg - implied_rest_ghz)
            print(f"   line-aware exclude: {best_match} @ rest {KNOWN_LINES_GHZ.get(best_match, 0):.3f} GHz "
                  f"→ offset {mid:+.0f} km/s ± {(zhi-zlo)/2:.0f}")
    if not offsets:
        print("  SKIP: no offsets in safe range (cube too narrow).")
        return

    for i, off in enumerate(offsets):
        offset_tag = f"offset_{int(round(off)):+05d}"
        sub = "nulltest_fixed" if fixed_mask else "nulltest"
        out_dir = RESULTS / display_name / sub / offset_tag
        first = (i == 0)
        csv_path = (RESULTS / "_nulltest" / "nulltest_samples_fixed_mask.csv"
                    if fixed_mask else CSV_PATH)
        cfg = replace(
            base_cfg,
            null_test_offset_kms=float(off),
            skip_plots=(not first),
            results_csv_row=str(csv_path),
            output_dir=str(out_dir),
            **fixed_scan_kwargs,
        )
        try:
            res = analyze(cfg)
            flag = " [CONTROL]" if is_control else ""
            print(f"    [{i+1:2d}/{len(offsets)}] {offset_tag}  "
                  f"max_SN={res.sn:.2f}  nbeam={res.nbeam:.1f}{flag}")
        except ConflictReportError as e:
            print(f"    [{i+1:2d}/{len(offsets)}] {offset_tag}  SKIP: {e}")
        except Exception:
            print(f"    [{i+1:2d}/{len(offsets)}] {offset_tag}  ERROR:")
            traceback.print_exc()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("galaxies", nargs="*", help="Optional subset (display names).")
    ap.add_argument("--fixed-mask", action="store_true",
                    help="Lock scan params (smooth, sthresh, vel_width) to real-line best at every null "
                         "offset (removes selector bias from per-offset re-scan). Requires that the "
                         "real-line analysis has already been run for each target (writes "
                         "results/{galaxy}/tables/results.txt). Writes to nulltest_samples_fixed_mask.csv.")
    args = ap.parse_args()

    all_targets = [(n, d, False) for n, d in THRESHOLD_GROUP] + \
                  [(n, d, True) for n, d in CONTROL_GROUP]
    if args.galaxies:
        wanted = {g.upper().replace(" ", "") for g in args.galaxies}
        all_targets = [t for t in all_targets
                       if t[0].replace(" ", "").upper() in wanted]
        if not all_targets:
            print(f"No matching galaxies in {wanted}")
            sys.exit(1)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixed_csv = RESULTS / "_nulltest" / "nulltest_samples_fixed_mask.csv"
    active_csv = fixed_csv if args.fixed_mask else CSV_PATH
    print(f"Results CSV: {active_csv}")
    print(f"Mode: {'FIXED-MASK (selector-bias-free)' if args.fixed_mask else 'SCAN (legacy; inflated p95 from selector bias)'}")

    for name, subdir, is_ctrl in all_targets:
        try:
            run_one_galaxy(name, subdir, is_ctrl, fixed_mask=args.fixed_mask)
        except Exception:
            print(f"\n!!! Fatal error on {name}:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
