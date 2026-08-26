"""Shared dataclasses for the step3 dual-cube pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SkyRegion:
    """A physical region on the sky, cube-independent.

    Square region when `dec_size_arcsec` is None (side = size_arcsec).
    Rectangular region when dec_size_arcsec is given: width=size_arcsec in RA,
    height=dec_size_arcsec in Dec. Needed for edge-on galaxies (NGC 3628) where
    the disk is thin in Dec.
    """
    ra_deg: float
    dec_deg: float
    size_arcsec: float
    dec_size_arcsec: Optional[float] = None


@dataclass
class AnalysisConfig:
    """All inputs needed to run one step3 analysis."""
    galaxy: str
    z: float
    weak_line: str
    strong_line: str
    hrl_pbcor_path: str
    hrl_nonpbcor_path: str
    co_pbcor_path: str
    co_nonpbcor_path: str
    signal_region: SkyRegion
    noise_region: SkyRegion
    output_dir: str

    hrl_pb_path: Optional[str] = None   # primary beam FITS; if None, skip PB cutoff
    co_pb_path: Optional[str] = None
    weak_line_window_kms: float = 300.0
    strong_line_window_kms: float = 300.0
    # When True (default), CO mom-0 integration window is adapted per-galaxy
    # by measuring CO line FW(10%) within the signal_box and using
    # FW(10%) + 60 km/s buffer (capped at strong_line_window_kms above).
    # Set False for legacy fixed ±strong_line_window_kms behavior.
    # Reason: NGC 5253 FWHM 68 km/s with legacy 600 km/s window dilutes
    # mom-0 by ~5× more noise channels than signal channels.
    strong_line_adaptive_window: bool = True
    pb_cutoff: float = 0.2
    smoothing_factors: List[int] = field(default_factory=lambda: [3, 5, 10])
    spatial_thresholds: List[float] = field(
        default_factory=lambda: [3.0, 5.0, 8.0, 13.0, 14.0, 15.0, 16.0,
                                  17.0, 18.0, 19.0, 20.0, 32.0]
    )
    velocity_widths_kms: List[int] = field(
        default_factory=lambda: list(range(60, 520, 20))
    )
    # Polynomial baseline degree for spectrum baseline subtraction in
    # calc_flux_sn (line-free channels are fit; polynomial is subtracted
    # from the line region). Default 1 (linear) catches per-channel offset
    # AND any across-band slope (e.g., from imperfect continuum subtraction
    # leaving a linear trend). Degree 0 = legacy constant median.
    baseline_degree: int = 1
    # When True, additionally save the best mask + HRL mom-0 as .npy under
    # output_dir for downstream visualisation (mask comparison plots, etc.).
    save_mask_npy: bool = False
    require_connected: bool = True
    # Component-selection mode for the candidate mask after threshold + footprint.
    #   "center"  — keep the connected component containing the galaxy center pixel;
    #               fall back to the largest component if center is empty (legacy
    #               default; tuned for compact-starburst sources like NGC 5253 X1a).
    #   "largest" — keep only the single largest connected component (ignores center).
    #   "all"     — keep all connected components above threshold within the CO
    #               footprint. Use for distributed/disk-dominated sources where CO
    #               sits in bar ends, ring, or multiple arms (NGC 5248, PHANGS face-on
    #               spirals); without this the mask collapses to whichever single
    #               knot happens to be largest, excluding genuine CO regions
    #               (documented limitation, 2026-04-28).
    # Only takes effect when `require_connected=True`. Ignored otherwise.
    mask_components_mode: str = "center"
    weak_line_rest_ghz: Optional[float] = None
    strong_line_rest_ghz: Optional[float] = None

    null_test_offset_kms: float = 0.0
    skip_plots: bool = False
    results_csv_row: Optional[str] = None

    # CO-footprint mask constraint (2026-04-24 selector-bias mitigation).
    # Hard AND: every candidate mask is intersected with `co_mom0 > co_footprint_sigma × σ_unsmoothed`
    # so the scan cannot pick regions outside physically-confirmed CO emission (prevents
    # smoothed-noise peaks outside the CO body from dominating high-threshold + high-kernel
    # combinations). See `docs/paper-notes/mask-baseline.md` for discussion.
    enable_co_footprint_constraint: bool = True
    co_footprint_sigma: float = 3.0
    # Footprint reference smoothing (2026-04-25 fix for high-resolution failure mode).
    # The original definition `unsmoothed_CO > 3σ_unsmoothed` assumed σ ≪ source peak,
    # which holds at typical 0.5-1″ common-beam scales but FAILS at native 0.19″
    # (NGC 5253 deep): σ_unsmoothed becomes comparable to the source peak, the 3σ
    # threshold is satisfied by noise speckles as much as by source emission, and
    # the footprint fragments into hundreds of disconnected components. The
    # connected-component mask filter then collapses to a single tiny fragment
    # containing the geometric center, giving spurious S/N ≈ 0.
    # Fix: compute footprint from CO mom-0 first smoothed by an absolute-arcsec
    # Gaussian kernel (default 1″), so the threshold is applied to a map whose
    # noise scale is set by the smoothing rather than by the per-pixel beam.
    # 1″ is wide enough to bridge fragmentation at 0.19″ beam (verified on NGC
    # 5253 X1a same-mousid: footprint goes from 142 fragments → 5 components,
    # one dominant 1682-pix; recovered S/N 8.79 vs cross-mousid 8.01) and small
    # enough to preserve real galaxy morphology at typical sub-arcsecond beams.
    co_footprint_smoothing_arcsec: float = 1.0

    # Adaptive signal/noise region modes (2026-04-28):
    # The legacy default is hand-picked sky boxes (`signal_region`, `noise_region`).
    # For distributed disk sources (face-on PHANGS spirals) this is too restrictive
    # — the signal_box can clip off most of the CO footprint and the small noise
    # box has a poor σ statistic. The adaptive modes below are the
    # interferometer-standard approach (Sun+ 2018, Leroy+ 2021 PHANGS-ALMA):
    #   signal_region_mode = "box"               (default; uses signal_region)
    #                      = "dilated_footprint" (CO 3σ footprint dilated by buffer
    #                                            ∪ NED-center anchor of given size)
    #   noise_region_mode  = "box"               (default; uses noise_region)
    #                      = "auto"              (PB > pb_threshold ∩ ¬footprint;
    #                                            interferometer-standard cube-level σ)
    # Default kept "box" so existing detection/non-detection results are unchanged.
    signal_region_mode: str = "box"
    signal_buffer_arcsec: float = 5.0
    signal_center_anchor_arcsec: float = 8.0
    noise_region_mode: str = "box"
    noise_pb_threshold: float = 0.5

    # S1+N1 mode false-positive guard rails (2026-04-28).
    # When set (not None), the best-scan mask must satisfy these constraints
    # for the result to be classified as a detection. Override the raw S/N >
    # threshold rule to suppress two empirical S1+N1 failure modes:
    #   nbeam_min        — reject single-pixel hotspots (NGC 5248 S1+N1
    #                      verified case: 1.5-beam mask with S/N 3.39).
    #                      Recommended 5 for adaptive mode.
    #   n_components_max — reject fragmented sprawling masks (NGC 7793 S1+N1
    #                      verified case: 155 components, 1194 beams, S/N 5.61).
    #                      Recommended 10 for adaptive mode.
    # Both default None (no guard) to preserve canonical box-mode behavior;
    # adaptive-mode configs explicitly set them.
    nbeam_min: Optional[float] = None
    n_components_max: Optional[int] = None

    # Display-only PB cutoff (2026-04-25): pbcor noise is amplified by 1/PB,
    # so the PB ~ 0.2-0.4 ring around the mosaic edge looks "bright with noise"
    # when display contrast is stretched. We mask the display moment-0 to
    # PB > display_pb_cutoff (default 0.7) for visual cleanliness. This is
    # purely cosmetic — flux integration uses pb_cutoff (default 0.2). If the
    # PB cube is unavailable, falls back to nonpbcor isnan footprint.
    display_pb_cutoff: float = 0.7


@dataclass
class AnalysisResult:
    galaxy: str
    weak_line: str
    sn: float
    flux_jy_km_s: float
    width_kms: float
    nbeam: float
    peak_jy: float
    best_smoothing: int
    best_spatial_threshold: float
    best_velocity_width_kms: int
    script_path: str
    summary_md_path: str
    plots_dir: str
    tables_dir: str
    is_detection: bool
    # CO-footprint morphology diagnostic (2026-04-24). IoU between best mask and
    # unsmoothed CO 3σ footprint. Populated only if footprint constraint enabled.
    mask_footprint_iou: Optional[float] = None

    def summary(self) -> str:
        status = "Detection ✅" if self.is_detection else "Non-detection"
        return (
            f"Galaxy: {self.galaxy}  Line: {self.weak_line}\n"
            f"S/N: {self.sn:.1f}  Flux: {self.flux_jy_km_s:.3f} Jy·km/s  "
            f"Width: {self.width_kms:.0f} km/s  Nbeam: {self.nbeam:.1f}\n"
            f"Status: {status}"
        )
