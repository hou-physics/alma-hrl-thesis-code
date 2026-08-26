"""Orchestrator: load → PB cut → common beam → mask → reproject → extract → scan → report."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import AnalysisConfig, AnalysisResult, SkyRegion
from cube_io import (
    CubeBundle, load_cube_bundle, apply_pb_cutoff,
    get_beam_area_pix, channel_width_kms,
)
from flux import make_moment0, extract_spectrum, sigma_clip_noise, calc_flux_sn
from geometry import sky_region_to_pixel_box
from mask import (
    smooth, build_mask,
    compute_common_beam, convolve_to_beam,
    reproject_mask,
)

from reproject import reproject_interp as _reproject_interp


def _reproject_image(image, src_wcs, tgt_wcs, target_shape):
    """Reproject a 2D image (not mask) for visualization. Returns (data, footprint)."""
    return _reproject_interp((image, src_wcs.celestial), tgt_wcs.celestial, shape_out=target_shape)


class ConflictReportError(RuntimeError):
    """Raised when data contradicts a rule (e.g. line outside cube freq range)."""


C_KMS = 299792.458


def _cube_freq_range_hz(cube: CubeBundle) -> Tuple[float, float]:
    n = cube.data.shape[0]
    freqs = cube.crval3_hz + (np.arange(n) + 1 - cube.crpix3) * cube.cdelt3_hz
    return float(min(freqs[0], freqs[-1])), float(max(freqs[0], freqs[-1]))


def _resolve_line_channels(
    cube: CubeBundle, rest_ghz: float, z: float, window_kms: float
) -> Tuple[int, int, int]:
    """Return (ch_min, ch_max, center_chan) for a line at given rest freq, z.

    Raises ConflictReportError if observed freq is outside cube's range ± 10%.
    """
    obs_ghz = rest_ghz / (1.0 + z)
    obs_hz = obs_ghz * 1e9
    f_min, f_max = _cube_freq_range_hz(cube)
    bandwidth = f_max - f_min
    if not (f_min - 0.1 * bandwidth <= obs_hz <= f_max + 0.1 * bandwidth):
        raise ConflictReportError(
            f"Line at rest {rest_ghz} GHz / z={z} (obs {obs_ghz:.3f} GHz) "
            f"is outside cube frequency range ({f_min/1e9:.3f}-{f_max/1e9:.3f} GHz)"
        )
    ch_center = int(round(cube.crpix3 - 1 + (obs_hz - cube.crval3_hz) / cube.cdelt3_hz))
    chan_width_kms_here = C_KMS * abs(cube.cdelt3_hz) / obs_hz
    half_ch = int(round(window_kms / 2.0 / chan_width_kms_here))
    ch_min = max(0, ch_center - half_ch)
    ch_max = min(cube.data.shape[0] - 1, ch_center + half_ch)
    return ch_min, ch_max, ch_center


def _estimate_co_line_half_width_kms(
    co_cube: CubeBundle,
    signal_box: Tuple[int, int, int, int],
    line_center_chan: int,
    chan_width_kms: float,
    default_window_kms: float = 300.0,
    extent_threshold_frac: float = 0.1,
) -> float:
    """Adaptive estimate of CO line half-width-window for moment-0 integration.

    Sums the cube within signal_box per channel to get an integrated CO
    spectrum, finds the line peak near `line_center_chan`, and returns a
    half-window covering the line down to `extent_threshold_frac × peak`
    (approximately FW(10%)/2) plus a 30 km/s buffer. Falls back to
    `default_window_kms / 2` if line is not detected at ≥ 5σ.

    Avoids over-integrating noise channels for narrow-line sources
    (NGC 5253 FWHM 68 km/s with default 600 km/s window: ~5× too wide).
    """
    x1, y1, x2, y2 = signal_box
    data = co_cube.data
    if data.ndim == 4:
        data = data[0]
    sub = data[:, y1:y2 + 1, x1:x2 + 1]
    spec = np.nansum(sub, axis=(1, 2))
    n_chan = len(spec)
    fixed_half = int(round(default_window_kms / chan_width_kms))

    # Estimate noise from outside ±default_window_kms region.
    idx = np.arange(n_chan)
    noise_keep = ((idx < line_center_chan - fixed_half) |
                  (idx > line_center_chan + fixed_half))
    noise_vals = spec[noise_keep]
    noise_vals = noise_vals[np.isfinite(noise_vals)]
    if len(noise_vals) < 10:
        return default_window_kms / 2.0
    sigma = float(np.nanstd(noise_vals))
    if sigma <= 0:
        return default_window_kms / 2.0

    # Find peak in central ±default_window region.
    lo = max(0, line_center_chan - fixed_half)
    hi = min(n_chan - 1, line_center_chan + fixed_half)
    central = spec[lo:hi + 1]
    if not np.any(np.isfinite(central)):
        return default_window_kms / 2.0
    peak = float(np.nanmax(central))
    if peak < 5.0 * sigma:
        # No detection — fall back to default.
        return default_window_kms / 2.0
    threshold = max(extent_threshold_frac * peak, 3.0 * sigma)
    above_local = np.where(central > threshold)[0]
    if len(above_local) == 0:
        return default_window_kms / 2.0
    extent_chan = above_local.max() - above_local.min() + 1
    extent_kms = extent_chan * chan_width_kms
    # Half-window = extent/2 + 30 km/s buffer; cap at default.
    half_window_kms = extent_kms / 2.0 + 30.0
    return min(half_window_kms, default_window_kms / 2.0)


def _shift_center_for_null(
    hrl_center: int, offset_kms: float, chan_width_kms: float, nchan: int,
    buffer_kms: float = 550.0,
) -> int:
    """Shift HRL center channel by offset_kms. Raises if the shifted center
    falls within `buffer_kms` of either cube edge (buffer = max integration
    half-width 250 + noise-exclude half-width 300)."""
    offset_ch = int(round(offset_kms / chan_width_kms))
    new_center = hrl_center + offset_ch
    buffer_ch = int(round(buffer_kms / chan_width_kms))
    if not (buffer_ch <= new_center <= nchan - 1 - buffer_ch):
        raise ConflictReportError(
            f"Null-test offset {offset_kms:+.0f} km/s pushes HRL center to "
            f"ch {new_center} (cube has {nchan} chans, need buffer of {buffer_ch}); "
            f"skip this offset"
        )
    return new_center


def _look_up_rest_freq(line_name: str) -> float:
    """Fallback rest freqs (GHz)."""
    table = {
        "H30a": 231.90092784, "H29a": 256.30203519,
        "H40a": 99.02295247, "H41a": 92.03443415, "H42a": 85.68838984,
        "CO10": 115.271202, "CO21": 230.538000, "CO32": 345.795990,
        "HCN10": 88.631847, "HCN32": 265.886434,
        "HCO+10": 89.188523, "HCO+32": 267.557633,
        "HNC10": 90.663568,
        "CS21": 97.980953,
        "13CO21": 220.398684, "C18O21": 219.560354,
    }
    if line_name not in table:
        raise ConflictReportError(f"Unknown line name: {line_name!r}; specify rest freq in config")
    return table[line_name]


def _format_sky_region(reg: SkyRegion) -> str:
    if reg.dec_size_arcsec is not None and reg.dec_size_arcsec != reg.size_arcsec:
        return f"{reg.size_arcsec:.1f}″ × {reg.dec_size_arcsec:.1f}″ (RA × Dec)"
    return f"{reg.size_arcsec:.1f}″ (square)"


def _render_regions_block(
    config: AnalysisConfig,
    hrl_signal_box: Tuple[int, int, int, int],
    hrl_noise_box: Tuple[int, int, int, int],
    co_signal_box: Tuple[int, int, int, int],
    co_noise_box: Tuple[int, int, int, int],
) -> str:
    """Audit trail of subagent-chosen signal/noise SkyRegions.

    Sky values are deterministic inputs; pixel boxes are WCS-projected
    and clipped to cube shape. Both grids reported because cross-mousid
    cases give different pixel boxes per cube.
    """
    sig = config.signal_region
    noi = config.noise_region
    return (
        "\n## Sky regions (audit trail)\n\n"
        "Selected by `/analyze-galaxy` subagent based on NED + CO mom-0 inspection. "
        "Full rationale in the comment block above each `SkyRegion(...)` call in "
        "the per-galaxy `step3_analyze.py` config script.\n\n"
        "| Region | RA (deg) | Dec (deg) | Size | HRL pixel box (x1,y1,x2,y2) | CO pixel box |\n"
        "|---|---|---|---|---|---|\n"
        f"| signal | {sig.ra_deg:.6f} | {sig.dec_deg:.6f} | "
        f"{_format_sky_region(sig)} | {hrl_signal_box} | {co_signal_box} |\n"
        f"| noise  | {noi.ra_deg:.6f} | {noi.dec_deg:.6f} | "
        f"{_format_sky_region(noi)} | {hrl_noise_box} | {co_noise_box} |\n\n"
        "If a downstream metric looks suspicious (e.g. σ too high, mask cut at edge, "
        "non-detection where one is expected), check whether the signal_region "
        "covers the CO emission peak with margin and the noise_region sits in a "
        "line-free patch at PB > 0.5.\n"
    )


def analyze(config: AnalysisConfig) -> AnalysisResult:
    weak_rest = config.weak_line_rest_ghz or _look_up_rest_freq(config.weak_line)
    strong_rest = config.strong_line_rest_ghz or _look_up_rest_freq(config.strong_line)

    # [1] Load cubes. Dedup by path (single-cube case points multiple roles at
    # the same FITS file; loading 4× wastes 3× memory and OOM-kills on large cubes
    # like NGC 253's 3838-chan Band 3 cube).
    from cube_io import load_cube
    _cache: dict = {}
    def _get(path: str):
        if path not in _cache:
            _cache[path] = load_cube(path)
        return _cache[path]
    bundles = {
        "hrl_pbcor": _get(config.hrl_pbcor_path),
        "hrl_nonpbcor": _get(config.hrl_nonpbcor_path),
        "co_pbcor": _get(config.co_pbcor_path),
        "co_nonpbcor": _get(config.co_nonpbcor_path),
    }
    hrl_pbcor = bundles["hrl_pbcor"]
    hrl_nonpbcor = bundles["hrl_nonpbcor"]
    co_nonpbcor = bundles["co_nonpbcor"]

    # [2] PB cutoff on both non-pbcor cubes, only if PB paths were provided.
    if config.hrl_pb_path:
        from cube_io import load_cube
        hrl_nonpbcor = apply_pb_cutoff(
            hrl_nonpbcor, load_cube(config.hrl_pb_path), config.pb_cutoff)
    if config.co_pb_path:
        from cube_io import load_cube
        co_nonpbcor = apply_pb_cutoff(
            co_nonpbcor, load_cube(config.co_pb_path), config.pb_cutoff)

    # [3] Resolve sky regions
    hrl_signal_box = sky_region_to_pixel_box(
        config.signal_region, hrl_pbcor.wcs, cube_shape=hrl_pbcor.data.shape)
    hrl_noise_box = sky_region_to_pixel_box(
        config.noise_region, hrl_pbcor.wcs, cube_shape=hrl_pbcor.data.shape)
    co_signal_box = sky_region_to_pixel_box(
        config.signal_region, co_nonpbcor.wcs, cube_shape=co_nonpbcor.data.shape)
    co_noise_box = sky_region_to_pixel_box(
        config.noise_region, co_nonpbcor.wcs, cube_shape=co_nonpbcor.data.shape)

    # [4] Resolve channel ranges. CO window is adapted to the actual line
    # FW(10%) width when `strong_line_adaptive_window=True` (default), to
    # avoid over-integrating noise on narrow-line sources (NGC 5253 FWHM
    # 68 km/s with the legacy 600 km/s window dilutes mom-0 by ~5×).
    co_chan_width_kms_provisional = channel_width_kms(co_nonpbcor, strong_rest)
    co_center_provisional = int(round(
        co_nonpbcor.crpix3 - 1
        + ((strong_rest * 1e9 / (1.0 + config.z)) - co_nonpbcor.crval3_hz)
        / co_nonpbcor.cdelt3_hz
    ))
    if config.strong_line_adaptive_window:
        co_half_window_kms = _estimate_co_line_half_width_kms(
            co_nonpbcor, co_signal_box, co_center_provisional,
            co_chan_width_kms_provisional,
            default_window_kms=config.strong_line_window_kms)
        co_window_kms_effective = 2.0 * co_half_window_kms
    else:
        co_window_kms_effective = config.strong_line_window_kms
    co_chmin, co_chmax, co_center = _resolve_line_channels(
        co_nonpbcor, strong_rest, config.z, co_window_kms_effective)
    hrl_chmin, hrl_chmax, hrl_center = _resolve_line_channels(
        hrl_pbcor, weak_rest, config.z, config.weak_line_window_kms)
    try:
        hrl_co_chmin, hrl_co_chmax, _ = _resolve_line_channels(
            hrl_pbcor, strong_rest, config.z, co_window_kms_effective)
    except ConflictReportError:
        hrl_co_chmin = hrl_co_chmax = None

    # [4b] Null-test offset: shift HRL center ONLY (CO center + mask unchanged).
    hrl_chan_width_kms = channel_width_kms(hrl_pbcor, weak_rest)
    co_contamination_flag = False
    if config.null_test_offset_kms != 0.0:
        hrl_center = _shift_center_for_null(
            hrl_center, config.null_test_offset_kms,
            hrl_chan_width_kms, hrl_pbcor.data.shape[0],
        )
        if hrl_co_chmin is not None:
            max_half = int(round(max(config.velocity_widths_kms) / hrl_chan_width_kms / 2.0))
            if (hrl_center - max_half) <= hrl_co_chmax and (hrl_center + max_half) >= hrl_co_chmin:
                co_contamination_flag = True

    # [5] Common beam convolution
    target_beam = compute_common_beam(co_nonpbcor, hrl_pbcor, hrl_nonpbcor)
    co_nonpbcor = convolve_to_beam(co_nonpbcor, target_beam)
    hrl_pbcor = convolve_to_beam(hrl_pbcor, target_beam)
    hrl_nonpbcor = convolve_to_beam(hrl_nonpbcor, target_beam)
    beam_area_pix_common = get_beam_area_pix(hrl_pbcor)
    # Beam area on CO grid (used for the npix-on-CO-grid sanity check below).
    # For same-mousid cases this equals beam_area_pix_common (CO and HRL share
    # pixel scale); for dual-mousid (e.g. X1a HRL 0.019″/pix + X1c CO 0.18″/pix)
    # the two diverge by ~ (CO_pix / HRL_pix)² and using the wrong one makes
    # every candidate mask falsely fail the "≥ 1 beam" gate. Bug found
    # 2026-04-25 in the X1a deep analysis.
    beam_area_pix_co_grid = get_beam_area_pix(co_nonpbcor)

    # [6] CO moment-0
    co_chan_width_kms = channel_width_kms(co_nonpbcor, strong_rest)
    co_mom0 = make_moment0(co_nonpbcor.data, co_chmin, co_chmax, co_chan_width_kms)

    # [7] CO physical footprint (hard-constraint on mask selection, 2026-04-24,
    # 2026-04-25 fragmentation fix). The footprint envelopes the region where CO
    # is detected at config.co_footprint_sigma × σ. Originally computed from the
    # unsmoothed CO mom-0; this fragments at native sub-arcsecond beams (NGC 5253
    # X1a 0.19″: 142 disconnected components, none big enough to anchor a mask)
    # and the connected-component mask filter then collapses the mask to a tiny
    # fragment with no source overlap. We now compute the footprint from a CO
    # mom-0 first smoothed by an absolute-arcsec kernel
    # (`config.co_footprint_smoothing_arcsec`, default 1″) so the noise scale is
    # set by the smoothing rather than the per-pixel beam — the resulting
    # envelope is a single coherent component at any beam size. The smoothing
    # is purely for the footprint definition; the candidate-mask scan still uses
    # CO smoothed by the per-galaxy `smoothing_factors × beam` ladder.
    co_footprint: Optional[np.ndarray] = None
    if config.enable_co_footprint_constraint:
        co_pixel_scale_arcsec = abs(co_nonpbcor.header.get(
            "CDELT1", co_nonpbcor.header.get("CDELT2", 1.0))) * 3600.0
        fp_smooth_pix = (config.co_footprint_smoothing_arcsec /
                         co_pixel_scale_arcsec)
        co_mom0_for_fp = smooth(co_mom0, fp_smooth_pix)
        fp_sigma = sigma_clip_noise(co_mom0_for_fp, co_noise_box)
        co_footprint = co_mom0_for_fp > config.co_footprint_sigma * fp_sigma
        # Restrict footprint to inside the signal_box for the legacy "box" mode.
        # For "dilated_footprint" mode this restriction is skipped — the dilated
        # footprint is computed from `co_footprint` itself and therefore cannot
        # extend beyond it; restricting would be a no-op in that case anyway.
        if config.signal_region_mode == "box":
            x1, y1, x2, y2 = co_signal_box
            fp_box = np.zeros_like(co_footprint, dtype=bool)
            fp_box[y1:y2 + 1, x1:x2 + 1] = True
            co_footprint = co_footprint & fp_box

    # Adaptive signal/noise region computation (S1/N1 from caveats-draft.md).
    # `*_signal_active` and `*_noise_active` are the values actually used by
    # build_mask + sigma_clip_noise downstream; the original *_signal_box and
    # *_noise_box tuples are kept for plotting/audit-table rendering only.
    co_signal_active = co_signal_box
    hrl_signal_active = hrl_signal_box
    co_noise_active = co_noise_box
    hrl_noise_active = hrl_noise_box
    # Center pixel for component selection (used by build_mask "center" mode).
    co_center_pix = (
        (co_signal_box[0] + co_signal_box[2]) // 2,
        (co_signal_box[1] + co_signal_box[3]) // 2,
    )

    if config.signal_region_mode == "dilated_footprint":
        if co_footprint is None:
            raise ConflictReportError(
                "signal_region_mode='dilated_footprint' requires "
                "enable_co_footprint_constraint=True")
        from scipy.ndimage import binary_dilation
        buffer_pix = max(0, int(round(config.signal_buffer_arcsec /
                                      co_pixel_scale_arcsec)))
        if buffer_pix > 0:
            dilated = binary_dilation(co_footprint, iterations=buffer_pix)
        else:
            dilated = co_footprint.copy()
        # Center anchor: circle around NED pixel ensures source center always
        # included even when CO is weak there.
        anchor_pix = config.signal_center_anchor_arcsec / co_pixel_scale_arcsec
        ny, nx = dilated.shape
        cx_co, cy_co = co_center_pix
        yy, xx = np.ogrid[:ny, :nx]
        anchor = ((xx - cx_co) ** 2 + (yy - cy_co) ** 2) <= anchor_pix ** 2
        co_signal_active = dilated | anchor
        # HRL grid: same array for single-cube (HRL pbcor path == CO pbcor path).
        # For dual-cube the HRL signal mask is reprojected from CO grid.
        if config.hrl_pbcor_path == config.co_pbcor_path:
            hrl_signal_active = co_signal_active
        else:
            hrl_signal_active = reproject_mask(
                co_signal_active, co_nonpbcor.wcs, hrl_pbcor.wcs,
                target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]),
            )

    if config.noise_region_mode == "auto":
        # Cube-level σ: every pixel at PB > threshold not inside the CO footprint.
        # Requires PB cube; fall back to legacy box if absent.
        if config.co_pb_path is None:
            raise ConflictReportError(
                "noise_region_mode='auto' requires co_pb_path (PB cube needed "
                "to define line-free region)")
        co_pb_cube = load_cube(config.co_pb_path)
        if co_pb_cube.data.ndim == 3:
            pb2d_co = np.nanmedian(co_pb_cube.data, axis=0)
        else:
            pb2d_co = np.squeeze(co_pb_cube.data)
        co_noise_active = (pb2d_co > config.noise_pb_threshold)
        if co_footprint is not None:
            co_noise_active = co_noise_active & (~co_footprint)
        if config.hrl_pb_path is not None and \
                config.hrl_pbcor_path != config.co_pbcor_path:
            hrl_pb_cube = load_cube(config.hrl_pb_path)
            if hrl_pb_cube.data.ndim == 3:
                pb2d_hrl = np.nanmedian(hrl_pb_cube.data, axis=0)
            else:
                pb2d_hrl = np.squeeze(hrl_pb_cube.data)
            hrl_noise_active = (pb2d_hrl > config.noise_pb_threshold)
            # Reproject footprint to HRL grid for hole-cutting.
            if co_footprint is not None:
                fp_hrl = reproject_mask(
                    co_footprint, co_nonpbcor.wcs, hrl_pbcor.wcs,
                    target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]),
                )
                hrl_noise_active = hrl_noise_active & (~fp_hrl)
        else:
            hrl_noise_active = co_noise_active

    # [8] Scan
    co_pixel_scale = abs(co_nonpbcor.header.get("CDELT1",
                                                 co_nonpbcor.header.get("CDELT2", 1.0)))
    co_bmaj_pix = target_beam.major.to("deg").value / co_pixel_scale

    # Channel-range optimization (2026-04-28, REVERTED 2026-04-28): restricting
    # per-channel masked-sum work to HRL line + σ-buffer was tested as a 2×
    # speedup. Verified on NGC 5248 it changes per-channel RMS estimate
    # (S/N 3.39 → 7.10) because σ_clip on a smaller noise sample gives a
    # different σ — this is a real methodology change, not a wash.
    # We disable the optimization by default to keep all canonical runs
    # (NGC 4945 8.84 / Toma 9.27, etc.) numerically reproducible. The
    # `extract_spectrum(..., ch_range=...)` parameter is preserved for future
    # use if a follow-up re-baselines the entire sample.
    extract_ch_range = None

    results = []
    for sf in config.smoothing_factors:
        smoothed_co = smooth(co_mom0, sf * co_bmaj_pix)
        smoothed_sigma = sigma_clip_noise(smoothed_co, co_noise_active)
        for sthresh in config.spatial_thresholds:
            threshold_val = sthresh * smoothed_sigma
            cx, cy = co_center_pix
            mask_co = build_mask(
                smoothed_co, threshold_val, co_signal_active,
                config.require_connected, (cx, cy),
                co_footprint=co_footprint,
                components_mode=config.mask_components_mode)
            npix = int(np.sum(mask_co))
            if npix / beam_area_pix_co_grid < 1:
                continue
            mask_hrl = reproject_mask(
                mask_co, co_nonpbcor.wcs, hrl_pbcor.wcs,
                target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]))
            spectrum, _, nbeam = extract_spectrum(
                hrl_pbcor.data, mask_hrl, beam_area_pix_common,
                ch_range=extract_ch_range)
            for vw in config.velocity_widths_kms:
                flux, sn, mw, peak = calc_flux_sn(
                    spectrum, hrl_center, vw, hrl_chan_width_kms,
                    exclude_line_chans=(hrl_co_chmin, hrl_co_chmax)
                    if hrl_co_chmin is not None else None,
                    baseline_degree=config.baseline_degree,
                )
                results.append({
                    "smooth": sf, "sthresh": sthresh, "vel_width": vw,
                    "nbeam": nbeam, "mask_width": mw,
                    "flux": flux, "sn": sn, "peak": peak,
                })

    if not results:
        raise ConflictReportError(
            "No valid (smoothing, threshold, width) combination produced nbeam >= 1; "
            "check signal_region and spatial_thresholds in config"
        )
    best = max(results, key=lambda r: r["sn"])

    # Reconstruct best mask for diagnostic plot
    smoothed_co_best = smooth(co_mom0, best["smooth"] * co_bmaj_pix)
    sigma_best = sigma_clip_noise(smoothed_co_best, co_noise_active)
    cx, cy = co_center_pix
    mask_co_best = build_mask(
        smoothed_co_best, best["sthresh"] * sigma_best, co_signal_active,
        config.require_connected, (cx, cy),
        co_footprint=co_footprint,
        components_mode=config.mask_components_mode)

    # Morphology diagnostic: IoU(best_mask, co_footprint) on CO grid. With the
    # hard constraint active, best_mask ⊂ footprint, so this simplifies to the
    # "fill fraction" = |mask| / |footprint|. Reported in summary.md for flagging
    # pathological small-area-within-large-footprint selector picks.
    mask_footprint_iou: Optional[float] = None
    if co_footprint is not None and co_footprint.any():
        inter = np.logical_and(mask_co_best, co_footprint).sum()
        union = np.logical_or(mask_co_best, co_footprint).sum()
        mask_footprint_iou = float(inter) / float(union) if union > 0 else 0.0
    mask_hrl_best = reproject_mask(
        mask_co_best, co_nonpbcor.wcs, hrl_pbcor.wcs,
        target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]))

    if getattr(config, "save_mask_npy", False):
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "best_mask_hrl.npy",
                mask_hrl_best.astype(np.uint8))
    # Best-mask spectrum: NOT truncated — used for plotting and needs full
    # SPW context (so the spectrum panel shows CO line for single-cube,
    # baseline structure, etc.).
    spec_best, _, _ = extract_spectrum(
        hrl_pbcor.data, mask_hrl_best, beam_area_pix_common)

    half_best = int(round(best["vel_width"] / hrl_chan_width_kms / 2.0))
    hrl_mom0_opt = make_moment0(
        hrl_pbcor.data,
        max(0, hrl_center - half_best),
        min(hrl_pbcor.data.shape[0] - 1, hrl_center + half_best),
        hrl_chan_width_kms)

    # Separate DISPLAY moment-0 (narrower window for visual contrast, 2026-04-24).
    # Flux-optimal integration (scan-best width) can be wide enough that mom-0
    # over that window dilutes the per-pixel contrast — the source appears
    # washed out even when S/N > 3.54. For display, use half the scan-best
    # width (still covers the core of the line but reduces noise channels
    # contribution by √2). Spectrum panel continues to highlight the full
    # flux integration window in red so the full vs display relationship is
    # visible to the reader.
    display_vel_width_kms = max(60.0, best["vel_width"] * 0.5)  # min 60 km/s
    half_display = int(round(display_vel_width_kms / hrl_chan_width_kms / 2.0))
    hrl_mom0_display = make_moment0(
        hrl_pbcor.data,
        max(0, hrl_center - half_display),
        min(hrl_pbcor.data.shape[0] - 1, hrl_center + half_display),
        hrl_chan_width_kms)

    # Display PB-cutoff DISABLED (2026-04-29): the previous behavior masked
    # the display mom-0 at PB > display_pb_cutoff (default 0.7), which made
    # HRL render as a small circle while the CO panel rendered across the
    # full HRL grid — visually misleading because the two panels then
    # appear to be at different scales. Now we keep the natural cube
    # footprint (NaN only where the original cube is NaN, i.e. the
    # nonpbcor isnan footprint, ~PB > 0.2). Pbcor edge noise IS visible
    # at low PB, which is the honest representation. The display_pb_cutoff
    # config field is retained for backwards compatibility but no longer
    # applied. FLUX MEASUREMENT IS UNCHANGED.
    pb_valid_2d = np.isfinite(hrl_nonpbcor.data[hrl_center])
    hrl_mom0_display = np.where(pb_valid_2d, hrl_mom0_display, np.nan)

    # Reproject CO mom0 to HRL grid for Panel C visualization
    co_mom0_hrl_grid, _ = _reproject_image(
        co_mom0, co_nonpbcor.wcs, hrl_pbcor.wcs,
        target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]))

    # Ada-style nested-contour inputs: fixed smoothing at the smallest configured
    # factor (typically 3×bmaj, close to Ada 2022's 4×), contours drawn at a fixed
    # σ ladder. Purely diagnostic — does not influence the flux/S/N statistics.
    nested_smooth_factor = config.smoothing_factors[0]
    nested_contour_sigmas = [3.0, 5.0, 8.0, 13.0, 20.0]
    smoothed_co_nested = smooth(co_mom0, nested_smooth_factor * co_bmaj_pix)
    smoothed_co_nested_sigma = sigma_clip_noise(smoothed_co_nested, co_noise_active)
    smoothed_co_nested_hrl_grid, _ = _reproject_image(
        smoothed_co_nested, co_nonpbcor.wcs, hrl_pbcor.wcs,
        target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]))

    # Reproject CO footprint onto HRL grid for optional plot overlay (pale outline
    # showing the physical CO boundary that constrained the scan).
    co_footprint_hrl_grid = None
    if co_footprint is not None:
        fp_reproj, _ = _reproject_image(
            co_footprint.astype(float), co_nonpbcor.wcs, hrl_pbcor.wcs,
            target_shape=(hrl_pbcor.data.shape[1], hrl_pbcor.data.shape[2]))
        co_footprint_hrl_grid = (fp_reproj > 0.5) & np.isfinite(fp_reproj)

    # Find Ada-style fixed-mask flux (smooth=first, thresh=3σ) for secondary reporting.
    ada_match = [r for r in results
                 if r["smooth"] == nested_smooth_factor and r["sthresh"] == 3.0]
    ada_flux = ada_match[0]["flux"] if ada_match else None
    ada_sn = ada_match[0]["sn"] if ada_match else None
    ada_nbeam = ada_match[0]["nbeam"] if ada_match else None
    ada_width = ada_match[0]["mask_width"] if ada_match else None

    # [9, 10] Write outputs
    out = Path(config.output_dir)
    tables_dir = out / "tables"
    plots_dir = out / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Cache everything the plots need so replotting doesn't require a full re-run
    # (scan + cube load dominates wall time; plot generation is ~5 s).
    plot_args = dict(
        hrl_mom0_opt=hrl_mom0_display,   # narrower display window (2026-04-24)
        co_mom0=co_mom0_hrl_grid,
        spatial_mask_hrl=mask_hrl_best,
        spectrum=spec_best,
        hrl_center_chan=hrl_center,
        half_width_chan=half_best,        # spectrum highlight stays on flux window
        half_display_chan=half_display,   # but display uses narrower window
        display_vel_width_kms=display_vel_width_kms,
        co_hrl_chans=(hrl_co_chmin, hrl_co_chmax),
        chan_width_kms=hrl_chan_width_kms,
        vel_start_kms=0.0,
        best_sn=best["sn"], best_flux=best["flux"],
        best_nbeam=best["nbeam"], best_sthresh=best["sthresh"],
        best_smooth=best["smooth"], best_width_kms=best["mask_width"],
        signal_box_hrl=hrl_signal_box,
        gal_name=config.galaxy,
        smoothed_co_hrl_grid=smoothed_co_nested_hrl_grid,
        smoothed_co_sigma=smoothed_co_nested_sigma,
        nested_contour_sigmas=nested_contour_sigmas,
        nested_contour_smooth_factor=nested_smooth_factor,
    )
    heatmap_args = dict(
        results=results,
        smoothing_factors=config.smoothing_factors,
        gal_name=config.galaxy,
    )
    import pickle
    with open(out / "_plot_cache.pkl", "wb") as _f:
        pickle.dump({"plot_args": plot_args, "heatmap_args": heatmap_args}, _f)

    if not config.skip_plots:
        from plots import (plot_optimal_mask_diagnostic, plot_sn_heatmap,
                           plot_individual_panels)
        plot_optimal_mask_diagnostic(
            outpath=str(plots_dir / "optimal_mask.png"),
            **plot_args,
        )
        # Standalone per-panel figures (PNG+PDF) for thesis insertion; same
        # drawing helpers as the combined grid, so panels match it.
        plot_individual_panels(str(plots_dir), **plot_args)
        plot_sn_heatmap(
            outpath=str(plots_dir / "SN_heatmap.png"), **heatmap_args)

    results_file = tables_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("# smooth sthresh vel_width nbeam mask_width flux sn peak\n")
        for r in results:
            f.write(f"{r['smooth']} {r['sthresh']} {r['vel_width']} "
                    f"{r['nbeam']:.1f} {r['mask_width']:.0f} "
                    f"{r['flux']:.4f} {r['sn']:.2f} {r['peak']:.6f}\n")

    is_null_run = config.null_test_offset_kms != 0.0
    summary_path = out / "summary.md"

    # S1+N1 false-positive guard rails (2026-04-28). Skip in null-test runs
    # (we want raw scan results in the null distribution). For science runs,
    # if guards are configured and best mask violates either, override
    # is_detection to False.
    from scipy.ndimage import label as _label
    n_components_best = int(_label(mask_co_best)[1])
    guard_violation: Optional[str] = None
    if not is_null_run:
        if config.nbeam_min is not None and best["nbeam"] < config.nbeam_min:
            guard_violation = (
                f"Nbeam {best['nbeam']:.1f} < guard threshold "
                f"{config.nbeam_min} (likely 1-pixel hotspot)"
            )
        elif (config.n_components_max is not None and
              n_components_best > config.n_components_max):
            guard_violation = (
                f"Mask has {n_components_best} connected components > guard "
                f"threshold {config.n_components_max} (likely fragmented "
                f"selector pickup)"
            )
    is_detection = ((not is_null_run) and best["sn"] > 3 and
                    guard_violation is None)
    if is_null_run:
        status = f"Null test (offset {config.null_test_offset_kms:+.0f} km/s)"
    elif guard_violation is not None:
        status = f"Non-detection (S/N {best['sn']:.2f} above threshold but guard rule failed: {guard_violation})"
    else:
        status = "Detection" if is_detection else "Non-detection"
    ada_block = ""
    if ada_flux is not None:
        ada_block = (
            f"\n## Fixed-envelope mask (diagnostic)\n\n"
            f"Secondary measurement with CO smoothing fixed at "
            f"{nested_smooth_factor}×bmaj and the 3σ envelope as the integration "
            f"mask (no selector optimization):\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| S/N | {ada_sn:.2f} |\n"
            f"| Flux | {ada_flux:.3f} Jy·km/s |\n"
            f"| Width | {ada_width:.0f} km/s |\n"
            f"| N_beam | {ada_nbeam:.1f} |\n\n"
            f"Compare against the headline row above: for a genuine detection "
            f"the two should agree within noise; a large divergence or a sign "
            f"flip (as seen here with non-detections) indicates the scan "
            f"optimizer is picking a noise-sized peak.\n"
        )
    iou_block = ""
    if mask_footprint_iou is not None:
        # Joint (IoU, S/N) interpretation: IoU alone is a fill-fraction under the
        # hard footprint constraint (mask ⊂ footprint), so it reflects "how much
        # of the CO footprint the mask covers" rather than morphological validity.
        # A real compact detection (NGC 3628 style, Nbeam ~1) naturally has low
        # IoU; a real extended detection (NGC 4945 style) has high IoU. What
        # distinguishes selector-bias from either is whether S/N clears the
        # null-calibrated threshold (3.54).
        sn_val = best["sn"]
        iou_val = mask_footprint_iou
        if sn_val >= 3.54:
            if iou_val >= 0.5:
                iou_verdict = (
                    "extended emission fills CO footprint — consistent with real "
                    "source covering the CO-bright region"
                )
            elif iou_val >= 0.05:
                iou_verdict = (
                    "compact-to-moderate mask within CO — consistent with a real "
                    "source concentrated in a subset of the CO footprint"
                )
            else:
                iou_verdict = (
                    "mask is a tiny fraction of CO — may be a real pointsource-"
                    "like detection, or cross-check with Nbeam + CO peak coincidence"
                )
        else:
            if iou_val >= 0.5:
                iou_verdict = (
                    "mask extended but S/N below null threshold — weak-emission "
                    "non-detection"
                )
            elif iou_val >= 0.05:
                iou_verdict = (
                    "mask compact and S/N below null threshold — non-detection, "
                    "scan may have picked a small favored region within CO"
                )
            else:
                iou_verdict = (
                    "mask tiny + S/N below null threshold — classic selector-"
                    "bias pattern (small mask inside CO chosen from noise)"
                )
        iou_block = (
            f"\n## Mask morphology vs CO footprint (IoU diagnostic, 2026-04-24)\n\n"
            f"Intersection-over-Union between the scan-optimal mask and the CO\n"
            f"{config.co_footprint_sigma:.0f}σ physical footprint (computed from CO\n"
            f"smoothed at {config.co_footprint_smoothing_arcsec:.1f}″ reference scale).\n"
            f"With the hard footprint constraint enabled, this is a fill-fraction\n"
            f"(mask ⊂ footprint). Interpret jointly with S/N:\n"
            f"- High IoU + high S/N → real extended detection\n"
            f"- Low IoU + high S/N → real compact detection (e.g. NGC 3628)\n"
            f"- High IoU + low S/N → extended non-detection\n"
            f"- Low IoU + low S/N → classic selector-bias non-detection\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| IoU(mask, CO {config.co_footprint_sigma:.0f}σ footprint) | **{iou_val:.3f}** |\n"
            f"| S/N | {sn_val:.2f} (null-calibrated threshold = 3.54) |\n"
            f"| Interpretation | {iou_verdict} |\n"
        )

    regions_block = _render_regions_block(
        config, hrl_signal_box, hrl_noise_box, co_signal_box, co_noise_box,
    )
    summary_path.write_text(
        f"# {config.galaxy} — {config.weak_line} Analysis Summary\n"
        f"**Date:** (auto-generated)\n"
        f"**Status:** {status}\n\n"
        f"## Result\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| S/N | {best['sn']:.2f} |\n"
        f"| Flux | {best['flux']:.3f} Jy·km/s |\n"
        f"| Width | {best['mask_width']:.0f} km/s |\n"
        f"| N_beam | {best['nbeam']:.1f} |\n"
        f"{ada_block}{iou_block}\n"
        f"## Inputs\n- HRL pbcor: `{config.hrl_pbcor_path}`\n"
        f"- CO pbcor: `{config.co_pbcor_path}`\n"
        f"{regions_block}\n"
        f"## Outputs\n- plots: `{plots_dir}/`\n- tables: `{tables_dir}/`\n"
    )

    if config.results_csv_row is not None:
        csv_path = Path(config.results_csv_row)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        iou_csv = f"{mask_footprint_iou:.4f}" if mask_footprint_iou is not None else ""
        with open(csv_path, "a") as f:
            if write_header:
                f.write("galaxy,offset_kms,max_sn,best_flux,best_mask_width_kms,"
                        "best_nbeam,best_smooth,best_sthresh,best_vel_width,"
                        "n_valid_combos,co_contamination,mask_footprint_iou\n")
            f.write(
                f"{config.galaxy},{config.null_test_offset_kms:.1f},"
                f"{best['sn']:.3f},{best['flux']:.4f},{best['mask_width']:.0f},"
                f"{best['nbeam']:.2f},{best['smooth']},{best['sthresh']},"
                f"{best['vel_width']},{len(results)},"
                f"{int(co_contamination_flag)},{iou_csv}\n"
            )

    return AnalysisResult(
        galaxy=config.galaxy, weak_line=config.weak_line,
        sn=best["sn"], flux_jy_km_s=best["flux"],
        width_kms=best["mask_width"], nbeam=best["nbeam"],
        peak_jy=best["peak"],
        best_smoothing=best["smooth"],
        best_spatial_threshold=best["sthresh"],
        best_velocity_width_kms=best["vel_width"],
        script_path="(set by caller)",
        summary_md_path=str(summary_path),
        plots_dir=str(plots_dir),
        tables_dir=str(tables_dir),
        is_detection=is_detection,
        mask_footprint_iou=mask_footprint_iou,
    )


def replot_from_cache(output_dir: str) -> None:
    """Regenerate plots from the pickled cache written by `analyze()`.

    Use when only the plot style or contour choices change — skips the full
    cube load + scan (which dominate wall time) and just re-renders the
    diagnostic PNG/PDF from saved intermediate state.
    """
    import pickle
    from plots import (plot_optimal_mask_diagnostic, plot_sn_heatmap,
                       plot_individual_panels)
    out = Path(output_dir)
    cache_path = out / "_plot_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No plot cache at {cache_path}. Run the full analysis once "
            f"to produce it before using replot_from_cache."
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_optimal_mask_diagnostic(
        outpath=str(plots_dir / "optimal_mask.png"),
        **cache["plot_args"],
    )
    plot_individual_panels(str(plots_dir), **cache["plot_args"])
    plot_sn_heatmap(
        outpath=str(plots_dir / "SN_heatmap.png"),
        **cache["heatmap_args"],
    )
    print(f"Replotted {cache['plot_args']['gal_name']} from cache "
          f"({cache_path}).")
