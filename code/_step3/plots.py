"""Diagnostic plots: 4-panel moment-0/mask/spectrum overview + S/N heatmap.

All figures use the shared `plot_style` module (serif, CMasher colormaps,
300 dpi, thin inward ticks, publication-quality). Panel titles are kept
short; parametric details (Nbeam, σ thresholds, smoothing factor) go in
the figure caption of the thesis text, not the panel title.

The 4-panel grid (`plot_optimal_mask_diagnostic`) and the standalone
per-panel figures (`plot_individual_panels`) share one set of drawing
helpers, so each standalone panel matches its grid counterpart.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from plot_style import (
    setup_thesis_style, apply_clean_spines,
    CMAP_MOMENT0, COLOR_P95,
)

_MASK_BOUNDARY_SMOOTH_PIX = 1.5
_DISPLAY_SMOOTH_PIX = 2.0  # display-only Gaussian σ on HRL mom-0; suppresses
                            # sub-beam-correlated speckle on native-resolution maps
                            # without affecting source structure (which is
                            # ≫ 1 beam wide for resolved nuclei). Mask + flux are
                            # computed on the unsmoothed cube — purely visual.


def _smooth_for_display(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Soft Gaussian smoothing with NaN-safe boundary handling.

    Uses the weighted-Gaussian trick (smooth filled-array and finite-mask,
    divide) to avoid edge dilution where data borders NaN regions.
    """
    finite_mask = np.isfinite(arr)
    if sigma <= 0 or not finite_mask.any():
        return arr
    arr_filled = np.where(finite_mask, arr, 0.0)
    arr_smooth = gaussian_filter(arr_filled, sigma=sigma)
    weight = finite_mask.astype(np.float64)
    weight_smooth = gaussian_filter(weight, sigma=sigma)
    valid = weight_smooth > 0.5
    safe_weight = np.where(valid, weight_smooth, 1.0)
    out = np.where(valid, arr_smooth / safe_weight, np.nan)
    return np.where(finite_mask, out, np.nan)


# Dark overlay color for nested intensity contours — reads on the white/light
# zero region of the RdBu_r maps and on the CO-disk reds, without clashing with
# the red/blue background (the old warm-wheat tone was for the dark cmr.cosmic /
# cividis era and is invisible on a light background).
_NESTED_COLOR = "#1a1a1a"
_NESTED_LINEWIDTH = 0.5
_NESTED_ALPHA = 0.7

# The single selected mask contour — GREEN so it stands out against both the
# red (positive) and blue (negative) of the RdBu_r maps, instead of blending
# into the red as the old crimson did.
_MASK_CONTOUR_COLOR = "#00a000"
_SELECTED_LINEWIDTH = 1.0
_SELECTED_ALPHA = 0.9
# COLOR_P95 (crimson) is still used for the spectrum integration band, which
# sits on a white axes background where crimson reads fine.
_SELECTED_COLOR = COLOR_P95


def _build_display_state(
    hrl_mom0_opt: np.ndarray,
    co_mom0: np.ndarray,
    spatial_mask_hrl: np.ndarray,
    signal_box_hrl: Tuple[int, int, int, int],
    smoothed_co_hrl_grid: Optional[np.ndarray] = None,
    smoothed_co_sigma: Optional[float] = None,
    nested_contour_sigmas: Optional[List[float]] = None,
    nested_contour_smooth_factor: Optional[int] = None,
) -> dict:
    """Compute every shared display array / color limit / zoom bbox used by the
    diagnostic panels.

    The returned dict is consumed by BOTH the combined 4-panel figure and the
    standalone per-panel figures, guaranteeing the two render identically. No
    measurement happens here — purely display preparation, unchanged from the
    original combined-figure logic.
    """
    x1, y1, x2, y2 = signal_box_hrl
    h, w = hrl_mom0_opt.shape

    # Zoom bbox: tight box around (HRL-finite ∪ CO-bright ∪ spatial_mask) +
    # 10 px. Data is rendered at natural full extent; axes are zoomed via
    # xlim/ylim only, so the data underneath stays intact for any re-zoom.
    finite_for_zoom = np.isfinite(hrl_mom0_opt) & (hrl_mom0_opt != 0)
    if spatial_mask_hrl.any():
        finite_for_zoom = finite_for_zoom | spatial_mask_hrl.astype(bool)
    co_finite = np.isfinite(co_mom0)
    if co_finite.any():
        co_thresh = float(np.nanpercentile(co_mom0[co_finite], 95))
        co_bright = co_finite & (co_mom0 > co_thresh)
        if co_bright.any():
            finite_for_zoom = finite_for_zoom | co_bright
    if finite_for_zoom.any():
        ys, xs = np.where(finite_for_zoom)
        xa = max(0, int(xs.min()) - 10)
        xb = min(w, int(xs.max()) + 10)
        ya = max(0, int(ys.min()) - 10)
        yb = min(h, int(ys.max()) + 10)
    else:
        xa, xb, ya, yb = 0, w, 0, h

    # Display-smoothed HRL map + SYMMETRIC color limit from the signal-box
    # interior (so PB-edge noise can't saturate the cmap). vmin = -vmax keeps
    # zero at the diverging-colormap midpoint, so the negative range is visible.
    sub_full = _smooth_for_display(hrl_mom0_opt, _DISPLAY_SMOOTH_PIX)
    sig_sub = _smooth_for_display(
        hrl_mom0_opt[y1:y2 + 1, x1:x2 + 1], _DISPLAY_SMOOTH_PIX)
    finite_sig = np.isfinite(sig_sub) & (sig_sub != 0)
    vmax_hrl = float(np.nanpercentile(sig_sub[finite_sig], 99)
                     if finite_sig.any()
                     else np.nanpercentile(sub_full[np.isfinite(sub_full)], 99))
    vmin_hrl = -vmax_hrl

    # Nested-contour config (shared by the HRL+CO panel and the CO panel).
    nested_levels_abs = None
    smoothed_full = None
    if (smoothed_co_hrl_grid is not None and smoothed_co_sigma is not None
            and nested_contour_sigmas):
        nested_levels_abs = [s * smoothed_co_sigma for s in nested_contour_sigmas]
        smoothed_full = smoothed_co_hrl_grid

    # Soften the binary mask boundary so contour @ 0.5 traces a smooth curve
    # instead of pixel edges (purely visual; mask itself is unchanged).
    mask_smooth = gaussian_filter(
        spatial_mask_hrl.astype(float), sigma=_MASK_BOUNDARY_SMOOTH_PIX)

    # CO unified onto the same zero-centered RdBu_r diverging map as the HRL
    # panels (user request 2026-06-28): white = 0, the CO disk in reds, any CO
    # negatives in blue. vmax from the zoom window only (PB-edge noise outside
    # the visible region cannot saturate the cmap); vmin = -vmax keeps zero at
    # the colormap midpoint.
    co_zoom = co_mom0[ya:yb, xa:xb]
    finite_co = np.isfinite(co_zoom)
    vmax_co = (float(np.nanpercentile(co_zoom[finite_co], 99.5))
               if finite_co.any() else 1.0)
    vmin_co = -vmax_co

    return {
        "co_mom0": co_mom0,
        "sub_full": sub_full,
        "vmin_hrl": vmin_hrl, "vmax_hrl": vmax_hrl,
        "vmin_co": vmin_co, "vmax_co": vmax_co,
        "xa": xa, "xb": xb, "ya": ya, "yb": yb,
        "full_extent": [0, w, 0, h],
        "nested_levels_abs": nested_levels_abs,
        "smoothed_full": smoothed_full,
        "mask_smooth": mask_smooth,
    }


def _zoom_and_square(ax, st: dict) -> None:
    """Lock axes to the shared zoom bbox + equal aspect (pixel alignment across
    panels). Grid off on image panels."""
    ax.grid(False)
    ax.set_xlim(st["xa"], st["xb"])
    ax.set_ylim(st["ya"], st["yb"])
    ax.set_aspect("equal")


def _draw_hrl_panel(ax, st: dict, *, with_contours: bool, title: str,
                    cbar_label: bool = True) -> None:
    """HRL moment-0 (diverging RdBu_r, zero-centered so the negative range is
    visible). Optionally overlay the nested CO contours + selected-mask contour
    (Panel B). Shared by the combined figure and the standalone panel."""
    im = ax.imshow(st["sub_full"], origin="lower", cmap=CMAP_MOMENT0,
                   vmin=st["vmin_hrl"], vmax=st["vmax_hrl"],
                   extent=st["full_extent"])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cb.set_label(r"Jy$\,\cdot\,$km/s/beam")
    if with_contours:
        if st["nested_levels_abs"] is not None:
            ax.contour(st["smoothed_full"], levels=st["nested_levels_abs"],
                       colors=_NESTED_COLOR, linewidths=_NESTED_LINEWIDTH,
                       alpha=_NESTED_ALPHA, extent=st["full_extent"])
        ax.contour(st["mask_smooth"], levels=[0.5], colors=_MASK_CONTOUR_COLOR,
                   linewidths=_SELECTED_LINEWIDTH, alpha=_SELECTED_ALPHA,
                   extent=st["full_extent"])
    ax.set_title(title)
    _zoom_and_square(ax, st)


def _draw_co_panel(ax, st: dict, *, title: str = "CO moment-0 with contours",
                   cbar_label: bool = True) -> None:
    """CO tracer moment-0, unified onto the same zero-centered RdBu_r diverging
    map as the HRL panels (2026-06-28), with the nested + selected-mask contours
    overlaid.

    Background is the native-resolution CO mom-0 (sharp); the 1″-smoothed
    version used for mask building is overlaid as nested contours so the
    methodology (Ada 2022) stays traceable.
    """
    im = ax.imshow(st["co_mom0"], origin="lower", cmap=CMAP_MOMENT0,
                   vmin=st["vmin_co"], vmax=st["vmax_co"],
                   extent=st["full_extent"])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cb.set_label(r"CO Jy$\,\cdot\,$km/s/beam")
    if st["nested_levels_abs"] is not None:
        ax.contour(st["smoothed_full"], levels=st["nested_levels_abs"],
                   colors=_NESTED_COLOR, linewidths=_NESTED_LINEWIDTH,
                   alpha=_NESTED_ALPHA, extent=st["full_extent"])
    ax.contour(st["mask_smooth"], levels=[0.5], colors=_SELECTED_COLOR,
               linewidths=_SELECTED_LINEWIDTH, alpha=_SELECTED_ALPHA,
               extent=st["full_extent"])
    ax.set_title(title)
    _zoom_and_square(ax, st)


def _draw_spectrum_panel(
    ax, *, spectrum: np.ndarray, hrl_center_chan: int, half_width_chan: int,
    co_hrl_chans: Tuple[Optional[int], Optional[int]],
    chan_width_kms: float, vel_start_kms: float,
) -> None:
    """Extracted HRL spectrum with the integration window highlighted."""
    nchan = len(spectrum)
    vel_axis = vel_start_kms + np.arange(nchan) * chan_width_kms
    ax.plot(vel_axis, spectrum, color="#202020", linewidth=0.7,
            drawstyle="steps-mid")
    ch_left = max(0, hrl_center_chan - half_width_chan)
    ch_right = min(nchan - 1, hrl_center_chan + half_width_chan)
    ax.axvspan(vel_axis[ch_left], vel_axis[ch_right],
               alpha=0.18, color=_SELECTED_COLOR, label="HRL integration")
    if co_hrl_chans and co_hrl_chans[0] is not None:
        c1 = max(0, co_hrl_chans[0])
        c2 = min(nchan - 1, co_hrl_chans[1])
        ax.axvspan(vel_axis[c1], vel_axis[c2], alpha=0.1,
                   color="#808080", label="CO window (excluded)")
    ax.axhline(y=0, color="#606060", linewidth=0.4)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Flux (Jy)")
    ax.set_title("Extracted HRL spectrum")
    ax.legend(loc="best", fontsize=7)
    apply_clean_spines(ax)


def plot_optimal_mask_diagnostic(
    hrl_mom0_opt: np.ndarray,
    co_mom0: np.ndarray,
    spatial_mask_hrl: np.ndarray,
    spectrum: np.ndarray,
    hrl_center_chan: int,
    half_width_chan: int,
    co_hrl_chans: Tuple[Optional[int], Optional[int]],
    chan_width_kms: float,
    vel_start_kms: float,
    best_sn: float,
    best_flux: float,
    best_nbeam: float,
    best_sthresh: float,
    best_smooth: int,
    best_width_kms: float,
    signal_box_hrl: Tuple[int, int, int, int],
    outpath: str,
    gal_name: str,
    smoothed_co_hrl_grid: Optional[np.ndarray] = None,
    smoothed_co_sigma: Optional[float] = None,
    nested_contour_sigmas: Optional[List[float]] = None,
    nested_contour_smooth_factor: Optional[int] = None,
    half_display_chan: Optional[int] = None,
    display_vel_width_kms: Optional[float] = None,
) -> None:
    """Combined 4-panel diagnostic (layout unchanged).

    Panels A/B (HRL moment-0) now use the zero-centered diverging colormap so
    the negative range is visible; Panel C (CO tracer) keeps the sequential
    map. Everything else — zoom, contours, spectrum, titles — is unchanged.
    """
    setup_thesis_style()
    st = _build_display_state(
        hrl_mom0_opt, co_mom0, spatial_mask_hrl, signal_box_hrl,
        smoothed_co_hrl_grid, smoothed_co_sigma,
        nested_contour_sigmas, nested_contour_smooth_factor)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2))
    # Panel A — HRL moment-0
    _draw_hrl_panel(axes[0, 0], st, with_contours=False,
                    title="HRL moment-0", cbar_label=True)
    # Panel B — HRL moment-0 with CO contours (no cbar label: Panel A beside it
    # already carries the scale, matching the original grid layout)
    _draw_hrl_panel(axes[0, 1], st, with_contours=True,
                    title="HRL moment-0 with CO contours", cbar_label=False)
    # Panel C — CO moment-0 with nested contours
    _draw_co_panel(axes[1, 0], st, title="CO moment-0 with contours",
                   cbar_label=True)
    # Panel D — extracted HRL spectrum
    _draw_spectrum_panel(
        axes[1, 1], spectrum=spectrum, hrl_center_chan=hrl_center_chan,
        half_width_chan=half_width_chan, co_hrl_chans=co_hrl_chans,
        chan_width_kms=chan_width_kms, vel_start_kms=vel_start_kms)

    plt.suptitle(gal_name, fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    # Also write PDF alongside (same path, swapped extension) for thesis import
    pdf_path = Path(outpath).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_individual_panels(outdir: str, **plot_args) -> None:
    """Render each diagnostic panel as its OWN standalone figure (PNG + PDF) so
    individual panels drop straight into the thesis without cropping the grid.

    Uses the same drawing helpers as `plot_optimal_mask_diagnostic`, so each
    standalone panel matches its grid counterpart. (The one deliberate
    deviation: standalone Panel B carries the colorbar label that the grid
    omits, since it no longer sits beside Panel A.)

    Call with the same kwargs as the combined figure, e.g.
    `plot_individual_panels(plots_dir, **plot_args)`.
    """
    setup_thesis_style()
    st = _build_display_state(
        plot_args["hrl_mom0_opt"], plot_args["co_mom0"],
        plot_args["spatial_mask_hrl"], plot_args["signal_box_hrl"],
        plot_args.get("smoothed_co_hrl_grid"),
        plot_args.get("smoothed_co_sigma"),
        plot_args.get("nested_contour_sigmas"),
        plot_args.get("nested_contour_smooth_factor"))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(fig, stem: str) -> None:
        fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Panel A — HRL moment-0
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _draw_hrl_panel(ax, st, with_contours=False, title="HRL moment-0",
                    cbar_label=True)
    _save(fig, "panel_a_hrl_mom0")

    # Panel B — HRL moment-0 with CO contours
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _draw_hrl_panel(ax, st, with_contours=True,
                    title="HRL moment-0 with CO contours", cbar_label=True)
    _save(fig, "panel_b_hrl_mom0_co_contours")

    # Panel C — CO moment-0
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    _draw_co_panel(ax, st, title="CO moment-0 with contours", cbar_label=True)
    _save(fig, "panel_c_co_mom0")

    # Panel D — extracted HRL spectrum
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _draw_spectrum_panel(
        ax, spectrum=plot_args["spectrum"],
        hrl_center_chan=plot_args["hrl_center_chan"],
        half_width_chan=plot_args["half_width_chan"],
        co_hrl_chans=plot_args["co_hrl_chans"],
        chan_width_kms=plot_args["chan_width_kms"],
        vel_start_kms=plot_args["vel_start_kms"])
    _save(fig, "panel_d_spectrum")


def plot_sn_heatmap(
    results: List[dict], smoothing_factors: List[int],
    gal_name: str, outpath: str,
) -> None:
    """Best S/N vs spatial threshold, one line per smoothing factor.

    The smoothing factor that contains the overall winning combination
    is drawn as a solid line; the others are dashed. A single panel so
    the curves can be compared directly on the same axis.
    """
    from plot_style import CMAP_QUALITATIVE
    setup_thesis_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    best_overall = max(results, key=lambda r: r["sn"])
    winning_smooth = best_overall["smooth"]

    # Distinct accent colors for each smoothing level (pick every other
    # color from the qualitative palette to maximise separation).
    palette = [CMAP_QUALITATIVE[i] for i in (0, 3, 6, 1, 4, 7)]
    colors = palette[:len(smoothing_factors)]

    for sf, color in zip(smoothing_factors, colors):
        sub = [r for r in results if r["smooth"] == sf]
        if not sub:
            continue
        sthreshs = sorted(set(r["sthresh"] for r in sub))
        best_sn = []
        for st in sthreshs:
            sub_st = [r for r in sub if r["sthresh"] == st]
            best_sn.append(max(sub_st, key=lambda r: r["sn"])["sn"])
        is_winner = (sf == winning_smooth)
        ax.plot(
            sthreshs, best_sn,
            color=color,
            linestyle="-" if is_winner else "--",
            linewidth=1.5 if is_winner else 1.0,
            marker="o", markersize=3.8,
            markerfacecolor=color, markeredgecolor=color,
            label=rf"Gaussian FWHM = {sf}$\times$ beam"
                  + (" (winner)" if is_winner else ""),
            zorder=3 if is_winner else 2,
        )

    ax.axhline(y=0, color="#606060", linewidth=0.4)
    ax.set_xlabel(r"Spatial threshold ($\sigma$)")
    ax.set_ylabel("Best S/N")
    ax.set_title(gal_name)
    ax.legend(frameon=False, fontsize=8, loc="best")
    apply_clean_spines(ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    pdf_path = Path(outpath).with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
