"""
PoC: data-driven signal_box + noise_box selection on CO/CS mom-0.

Replaces the subagent's manual RA/Dec/size config picks in
`step3_analyze.py` with:
  - Step 1: photutils.detect_sources on strong-tracer mom-0
  - Step 2: grid-search for best noise position (lowest |⟨mom0⟩|/σ
    outside detected source + inside PB > 0.5)

Compares auto-selected regions to the manual ones for 3 galaxies.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.patches import Rectangle
import cmasher  # noqa: F401

from photutils.segmentation import detect_threshold, detect_sources
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube
from ellipse_cog_poc import (
    channel_velocities, pix_per_beam, beam_fwhm_pix, smooth_fft,
)

C_KMS = 299792.458
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2/tmp")


@dataclass
class GalaxyAutoConfig:
    name: str
    co_cube_path: str       # strong tracer cube (CO/CS)
    pb_path: str | None
    z: float
    co_line_rest_hz: float  # e.g. CO21 = 230.538e9, CS21 = 97.980953e9
    W_kms: float            # velocity window for mom-0
    sf_smooth: float        # CO smoothing factor (in units of beam FWHM)
    ned_ra: float
    ned_dec: float
    # manual regions from step3_analyze.py (for comparison)
    manual_sig_ra: float
    manual_sig_dec: float
    manual_sig_size_arcsec: float
    manual_noise_ra: float
    manual_noise_dec: float
    manual_noise_size_arcsec: float


# ---------- core algorithm ----------
def load_pb_mask(pb_path: str | None, threshold: float = 0.5):
    if pb_path is None or not Path(pb_path).exists():
        return None
    pb_bundle = load_cube(pb_path)
    pb = pb_bundle.data
    if pb.ndim == 3:
        pb = np.asarray(pb[pb.shape[0] // 2, :, :])
    pb_bundle.data = None  # type: ignore
    return (pb > threshold) & np.isfinite(pb)


def detect_signal_segments(mom0_smooth, nsigma=3.0, npixels=10):
    """Run photutils source detection; return SegmentationImage."""
    # detect_threshold computes pixel-by-pixel threshold via sigma-clipping
    threshold = detect_threshold(mom0_smooth, nsigma=nsigma)
    segm = detect_sources(mom0_smooth, threshold, npixels=npixels)
    return segm, threshold


def select_target_segment(segm, wcs2, ned_ra, ned_dec, max_dist_arcsec=30):
    """Pick the segment closest to NED center within max_dist."""
    if segm is None or segm.nlabels == 0:
        return None
    ned_xc, ned_yc = wcs2.world_to_pixel_values(ned_ra, ned_dec)
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600
    max_dist_pix = max_dist_arcsec / pix_arcsec

    # photutils: segment label and centroid
    best_label = None
    best_dist = np.inf
    for label in segm.labels:
        seg_mask = segm.data == label
        ys, xs = np.where(seg_mask)
        # centroid
        cx = float(xs.mean()); cy = float(ys.mean())
        d = np.sqrt((cx - ned_xc) ** 2 + (cy - ned_yc) ** 2)
        if d < best_dist and d < max_dist_pix:
            best_dist = d
            best_label = label
    return best_label, best_dist


def find_best_noise_box(mom0, pb_mask, source_mask, box_size_pix,
                        min_dist_from_source_pix=30, n_candidates=400,
                        rng_seed=42):
    """Grid-search candidate noise box positions; rank by |⟨mom0⟩|/σ.

    Returns (xc, yc, size, score, mean, std) for best candidate."""
    rng = np.random.default_rng(rng_seed)
    ny, nx = mom0.shape
    half = box_size_pix // 2

    # Allowed centers: inside PB, NOT near any source
    excl = binary_dilation(source_mask, iterations=min_dist_from_source_pix)
    valid_centers = np.ones((ny, nx), dtype=bool)
    if pb_mask is not None:
        valid_centers &= pb_mask
    valid_centers &= ~excl
    # also need box to fit inside cube
    valid_centers[:half] = False; valid_centers[-half:] = False
    valid_centers[:, :half] = False; valid_centers[:, -half:] = False

    candidate_idx = np.where(valid_centers)
    if len(candidate_idx[0]) == 0:
        return None
    n_cands = min(n_candidates, len(candidate_idx[0]))
    pick = rng.choice(len(candidate_idx[0]), size=n_cands, replace=False)

    best = None
    best_score = np.inf
    for i in pick:
        yc = int(candidate_idx[0][i])
        xc = int(candidate_idx[1][i])
        box = mom0[yc-half:yc+half+1, xc-half:xc+half+1]
        finite = np.isfinite(box)
        if finite.sum() < 0.9 * box.size:
            continue
        m = float(np.nanmean(box))
        s = float(np.nanstd(box))
        if s <= 0:
            continue
        score = abs(m) / s
        if score < best_score:
            best_score = score
            best = (xc, yc, box_size_pix, score, m, s)
    return best


def run_galaxy(cfg: GalaxyAutoConfig):
    print(f"\n=== {cfg.name} ===")
    bundle = load_cube(cfg.co_cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = cfg.co_line_rest_hz / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)
    wcs2 = WCS(hdr).celestial
    pix_arcsec = abs(wcs2.wcs.cdelt[0]) * 3600
    print(f"  cube {cube.shape}, dv={dv:.1f}, pix/beam={ppb:.1f}, "
          f"pixel scale={pix_arcsec:.3f}\"/pix")

    # mom-0
    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv
    mom0_smooth = smooth_fft(mom0, cfg.sf_smooth * beam_fwhm)
    print(f"  mom-0 range: [{np.nanmin(mom0):+.3f}, {np.nanmax(mom0):+.3f}]")

    # PB mask
    pb_mask = load_pb_mask(cfg.pb_path) if cfg.pb_path else None
    if pb_mask is not None:
        print(f"  PB > 0.5: {pb_mask.sum()} pix ({100*pb_mask.sum()/(ny*nx):.1f}%)")

    # restrict mom-0 to PB > 0.5 region for source detection
    mom0_for_det = mom0_smooth.copy()
    if pb_mask is not None:
        mom0_for_det[~pb_mask] = np.nan

    # Step 1: photutils source detection
    segm, threshold = detect_signal_segments(mom0_for_det, nsigma=3.0, npixels=10)
    if segm is None or segm.nlabels == 0:
        print("  WARN: photutils found 0 sources")
        bundle.data = None  # type: ignore
        return None
    print(f"  photutils detected {segm.nlabels} sources at 3σ above background")

    # Step 1b: pick the one containing/closest to NED center
    target_label, dist = select_target_segment(segm, wcs2,
                                                cfg.ned_ra, cfg.ned_dec,
                                                max_dist_arcsec=30)
    if target_label is None:
        print("  WARN: no segment within 30\" of NED, skipping")
        bundle.data = None  # type: ignore
        return None
    target_mask = (segm.data == target_label)
    seg_ys, seg_xs = np.where(target_mask)
    sig_box_bbox = (seg_xs.min(), seg_ys.min(), seg_xs.max(), seg_ys.max())
    sig_box_size = (sig_box_bbox[2] - sig_box_bbox[0] + 1,
                    sig_box_bbox[3] - sig_box_bbox[1] + 1)
    sig_box_size_arcsec = (sig_box_size[0] * pix_arcsec,
                           sig_box_size[1] * pix_arcsec)
    sig_centroid = (float(seg_xs.mean()), float(seg_ys.mean()))
    sig_centroid_world = wcs2.pixel_to_world_values(*sig_centroid)
    print(f"  target source: label={target_label}, dist_from_NED={dist*pix_arcsec:.1f}\", "
          f"bbox size={sig_box_size_arcsec[0]:.1f}\" × {sig_box_size_arcsec[1]:.1f}\"")
    print(f"  source centroid (RA, Dec): ({sig_centroid_world[0]:.5f}, {sig_centroid_world[1]:.5f})")

    # Step 2: auto noise box (use ALL detected source pixels as exclusion)
    source_mask = (segm.data > 0)
    # match noise box size to manual size (for fair comparison)
    noise_box_pix = int(cfg.manual_noise_size_arcsec / pix_arcsec)
    # exclusion buffer: 3 × beam_FWHM (so noise box is well outside source halo)
    exclusion_pix = int(3 * beam_fwhm)
    nbox = find_best_noise_box(mom0, pb_mask, source_mask, noise_box_pix,
                               min_dist_from_source_pix=exclusion_pix)
    if nbox is None:
        print("  WARN: noise box search failed")
        bundle.data = None  # type: ignore
        return None
    noise_xc, noise_yc, _, noise_score, noise_mean, noise_std = nbox
    noise_world = wcs2.pixel_to_world_values(noise_xc, noise_yc)
    print(f"  auto noise box: ({noise_xc}, {noise_yc}) "
          f"→ (RA, Dec) = ({noise_world[0]:.5f}, {noise_world[1]:.5f})")
    print(f"  |⟨mom0⟩|/σ in noise box = {noise_score:.4f} "
          f"(⟨mom0⟩={noise_mean:+.4f}, σ={noise_std:.4f})")

    # Manual noise box stats for comparison
    manual_nxc, manual_nyc = wcs2.world_to_pixel_values(
        cfg.manual_noise_ra, cfg.manual_noise_dec)
    manual_half = int(cfg.manual_noise_size_arcsec / 2 / pix_arcsec)
    manual_nbox = mom0[int(manual_nyc) - manual_half:int(manual_nyc) + manual_half + 1,
                       int(manual_nxc) - manual_half:int(manual_nxc) + manual_half + 1]
    manual_finite = np.isfinite(manual_nbox)
    if manual_finite.sum() > 0:
        manual_mean = float(np.nanmean(manual_nbox))
        manual_std = float(np.nanstd(manual_nbox))
        manual_score = abs(manual_mean) / max(manual_std, 1e-9)
        print(f"  MANUAL noise box: |⟨mom0⟩|/σ = {manual_score:.4f} "
              f"(⟨mom0⟩={manual_mean:+.4f}, σ={manual_std:.4f})")

    # ---------- visualize ----------
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))

    # left: mom-0 with photutils segments
    ax = axes[0]
    finite = np.isfinite(mom0_smooth)
    vmax = float(np.nanpercentile(mom0_smooth[finite], 99))
    vmin = float(np.nanpercentile(mom0_smooth[finite], 1))
    im = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="CO mom-0 smoothed (Jy/beam · km/s)")

    # all detected sources outlines
    ax.contour((segm.data > 0).astype(int), levels=[0.5],
               colors="white", linewidths=0.5, alpha=0.5)
    # target source (red outline + bbox)
    ax.contour(target_mask.astype(int), levels=[0.5],
               colors="red", linewidths=2.0)
    rect_sig = Rectangle(
        (sig_box_bbox[0] - 0.5, sig_box_bbox[1] - 0.5),
        sig_box_size[0], sig_box_size[1],
        fill=False, edgecolor="red", linestyle="-", linewidth=2)
    ax.add_patch(rect_sig)
    # NED center
    ned_xc, ned_yc = wcs2.world_to_pixel_values(cfg.ned_ra, cfg.ned_dec)
    ax.plot(ned_xc, ned_yc, "+", color="lime", markersize=15,
            markeredgewidth=2, label=f"NED ({cfg.ned_ra:.4f}, {cfg.ned_dec:.4f})")

    # MANUAL sig box (yellow dashed)
    msig_xc, msig_yc = wcs2.world_to_pixel_values(
        cfg.manual_sig_ra, cfg.manual_sig_dec)
    msig_half = int(cfg.manual_sig_size_arcsec / 2 / pix_arcsec)
    rect_msig = Rectangle(
        (msig_xc - msig_half - 0.5, msig_yc - msig_half - 0.5),
        2 * msig_half, 2 * msig_half,
        fill=False, edgecolor="yellow", linestyle="--", linewidth=1.5)
    ax.add_patch(rect_msig)

    # AUTO noise box (cyan)
    rect_noise_auto = Rectangle(
        (noise_xc - noise_box_pix // 2 - 0.5,
         noise_yc - noise_box_pix // 2 - 0.5),
        noise_box_pix, noise_box_pix,
        fill=False, edgecolor="cyan", linewidth=1.5)
    ax.add_patch(rect_noise_auto)

    # MANUAL noise box (orange dashed)
    rect_noise_man = Rectangle(
        (manual_nxc - manual_half - 0.5,
         manual_nyc - manual_half - 0.5),
        2 * manual_half, 2 * manual_half,
        fill=False, edgecolor="orange", linestyle="--", linewidth=1.5)
    ax.add_patch(rect_noise_man)

    # legend stubs
    ax.plot([], [], color="white", lw=0.5, label=f"{segm.nlabels} detected sources (3σ)")
    ax.plot([], [], color="red", lw=2, label="target source (auto signal_box)")
    ax.plot([], [], color="yellow", ls="--", lw=1.5, label="MANUAL signal box")
    ax.plot([], [], color="cyan", lw=1.5,
            label=f"AUTO noise box (|⟨mom0⟩|/σ = {noise_score:.3f})")
    ax.plot([], [], color="orange", ls="--", lw=1.5,
            label="MANUAL noise box")

    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(f"{cfg.name} — photutils + grid-search noise box")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # right: zoom on signal box
    ax = axes[1]
    pad = max(int(0.4 * max(sig_box_size[0], sig_box_size[1])), 20)
    zx_lo = max(0, sig_box_bbox[0] - pad)
    zx_hi = min(nx, sig_box_bbox[2] + pad)
    zy_lo = max(0, sig_box_bbox[1] - pad)
    zy_hi = min(ny, sig_box_bbox[3] + pad)
    im2 = ax.imshow(mom0_smooth, origin="lower", cmap="cmr.cosmic",
                    vmin=vmin, vmax=vmax)
    ax.set_xlim(zx_lo, zx_hi); ax.set_ylim(zy_lo, zy_hi)
    plt.colorbar(im2, ax=ax, label="CO mom-0 smoothed")
    ax.contour(target_mask.astype(int), levels=[0.5],
               colors="red", linewidths=2.0)
    ax.add_patch(Rectangle(
        (sig_box_bbox[0] - 0.5, sig_box_bbox[1] - 0.5),
        sig_box_size[0], sig_box_size[1],
        fill=False, edgecolor="red", linewidth=2,
        label=f"AUTO sig_box {sig_box_size_arcsec[0]:.1f}\"×{sig_box_size_arcsec[1]:.1f}\""))
    ax.add_patch(Rectangle(
        (msig_xc - msig_half - 0.5, msig_yc - msig_half - 0.5),
        2 * msig_half, 2 * msig_half,
        fill=False, edgecolor="yellow", linestyle="--", linewidth=1.5,
        label=f"MANUAL sig_box {cfg.manual_sig_size_arcsec:.0f}\"×{cfg.manual_sig_size_arcsec:.0f}\""))
    ax.plot(ned_xc, ned_yc, "+", color="lime", markersize=15,
            markeredgewidth=2, label="NED")
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("zoom: auto (red) vs manual (yellow) signal box")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    plt.suptitle(
        f"{cfg.name} — auto signal_box (photutils) + noise_box (grid search)\n"
        f"target source bbox: {sig_box_size_arcsec[0]:.1f}\"×{sig_box_size_arcsec[1]:.1f}\", "
        f"auto vs manual noise: |⟨mom0⟩|/σ = {noise_score:.3f} vs "
        f"{manual_score:.3f}",
        fontsize=11)
    plt.tight_layout()
    out_png = OUT_ROOT / f"{cfg.name}__auto_regions.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "n_sources_detected": int(segm.nlabels),
        "auto_sig_size_arcsec": sig_box_size_arcsec,
        "auto_sig_centroid": sig_centroid_world,
        "ned_dist_arcsec": dist * pix_arcsec,
        "auto_noise_xc": noise_xc, "auto_noise_yc": noise_yc,
        "auto_noise_score": noise_score,
        "manual_noise_score": manual_score,
        "manual_sig_size_arcsec": cfg.manual_sig_size_arcsec,
    }


GALAXIES = [
    GalaxyAutoConfig(
        name="NGC5253",
        co_cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_CO21_nonpbcor.fits",
        pb_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_CO21_pb.fits.gz",
        z=0.0014,
        co_line_rest_hz=230.538e9,
        W_kms=300.0, sf_smooth=1.0,
        ned_ra=204.982993, ned_dec=-31.640281,
        manual_sig_ra=204.982993, manual_sig_dec=-31.640281,
        manual_sig_size_arcsec=20.0,
        manual_noise_ra=204.987952, manual_noise_dec=-31.640681,
        manual_noise_size_arcsec=15.0,
    ),
    GalaxyAutoConfig(
        name="NGC4945",
        co_cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_CO_nonpbcor.fits",
        pb_path=None,
        z=0.00188,
        co_line_rest_hz=230.538e9,
        W_kms=400.0, sf_smooth=2.0,
        ned_ra=196.36260, ned_dec=-49.46942,
        manual_sig_ra=196.36260, manual_sig_dec=-49.46942,
        manual_sig_size_arcsec=17.0,
        manual_noise_ra=196.36901, manual_noise_dec=-49.46442,
        manual_noise_size_arcsec=7.0,
    ),
    GalaxyAutoConfig(
        name="NGC3628",
        co_cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_CO21_nonpbcor.fits",
        pb_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_CO21_pb.fits.gz",
        z=0.002772,
        co_line_rest_hz=230.538e9,
        W_kms=400.0, sf_smooth=1.0,
        ned_ra=170.0708, ned_dec=13.5888,
        manual_sig_ra=170.0708, manual_sig_dec=13.5888,
        manual_sig_size_arcsec=40.0,
        manual_noise_ra=170.0707, manual_noise_dec=13.5944,
        manual_noise_size_arcsec=15.0,
    ),
]


def main():
    results = []
    for g in GALAXIES:
        if not Path(g.co_cube_path).exists():
            print(f"SKIP {g.name}: cube missing")
            continue
        try:
            r = run_galaxy(g)
        except Exception as e:
            print(f"  ERROR on {g.name}: {e}")
            import traceback; traceback.print_exc()
            r = None
        if r:
            results.append(r)

    print("\n" + "=" * 95)
    print(f"{'galaxy':<12} {'auto sig (arcsec)':>20} {'manual sig':>13} "
          f"{'ned dist':>10} {'auto noise':>11} {'manual noise':>13}")
    print("-" * 95)
    for r in results:
        asz = f"{r['auto_sig_size_arcsec'][0]:.1f}×{r['auto_sig_size_arcsec'][1]:.1f}"
        print(f"{r['name']:<12} {asz:>20} {r['manual_sig_size_arcsec']:>13.1f} "
              f"{r['ned_dist_arcsec']:>10.2f} {r['auto_noise_score']:>11.4f} "
              f"{r['manual_noise_score']:>13.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
