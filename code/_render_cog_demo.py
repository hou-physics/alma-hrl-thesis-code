"""Prepare data for the interactive curve-of-growth (CoG) demo.

Outputs:
  notes/20-methodology/figs/cog_bg_NGC4945.png   — H30α moment-0 background
  notes/20-methodology/figs/cog_data.json         — per-step CoG arrays

CoG methods compared:
  A) sigma-dilation:    mask = (smoothed_co > sigma * sigma_co), largest
     connected component; sigma sweeps from high (small mask) to low
     (mask spans the whole footprint).
  B) ellipse aperture: an ellipse fitted by intensity-weighted PCA on
     the bright seed is scaled monotonically; mask = pixels inside the
     scaled ellipse.

For each step we record:
  - the threshold parameter (sigma for A, semi-major axis for B)
  - the cumulative flux integrated over the mask (in Jy·km/s)
  - geometry needed to draw the mask outline (polygon for A,
    ellipse parameters for B)

The HTML widget loads this JSON and draws the overlay + CoG curve as a
slider moves through the array.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from skimage import measure

ROOT = Path("/Volumes/HouAstro/master")
OUT  = ROOT / "zhengxu-notes" / "notes" / "20-methodology" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / "master_thesis" / "my_code" / "_step3"))
from cube_io import load_cube  # noqa: E402

GALAXY  = "NGC4945"
LABEL   = "NGC 4945"
FITS_HDR = ROOT / "master_thesis" / "work_dir" / GALAXY / f"{GALAXY}_H30a_nonpbcor.fits"
# pbcor science cube — the moment-0 for the CoG must be integrated over the
# FULL line window (±half_width_chan ≈ best_width), NOT the narrow display
# window baked into pa["hrl_mom0_opt"] (half_display_chan ≈ 160 km/s), or the
# curve of growth tops out at ~half the reported flux. See 2026-07-02 fix.
HRL_CUBE = ROOT / "master_thesis" / "work_dir" / GALAXY / f"{GALAXY}_H30a_spw1_v1_contsub.fits"


def beam_area_pix(bmaj_arcsec: float, bmin_arcsec: float,
                  pix_arcsec: float) -> float:
    """Beam solid angle expressed in pixels²."""
    return (np.pi / (4 * np.log(2))) * (bmaj_arcsec / pix_arcsec) \
        * (bmin_arcsec / pix_arcsec)


def main() -> None:
    # --- load cube cache ---
    pa = pickle.load(open(ROOT / "results" / GALAXY / "_plot_cache.pkl", "rb"))["plot_args"]
    # Full-line-window moment-0 (±half_width_chan ≈ best_width), integrated from
    # the pbcor cube so the CoG reaches the pipeline's reported flux (~8.8 Jy·km/s
    # for NGC 4945). pa["hrl_mom0_opt"] is a narrow 160 km/s DISPLAY mom-0 that
    # only captures ~half the broad line — using it made the CoG plateau at ~4.5.
    cc = int(pa["hrl_center_chan"]); hw = int(pa["half_width_chan"])
    dv = float(pa["chan_width_kms"])
    _cube = load_cube(str(HRL_CUBE))
    hrl = np.nansum(_cube.data[cc - hw:cc + hw + 1], axis=0) * dv   # Jy·km/s/beam
    _cube.data = None  # type: ignore
    print(f"full-window mom-0: channels [{cc-hw}:{cc+hw}] × dv={dv:.2f} km/s")
    co_s  = pa["smoothed_co_hrl_grid"]          # 1″-smoothed CO mom-0
    sigma_co = float(pa["smoothed_co_sigma"])    # σ for footprint threshold
    # Flip vertically so origin='upper' (default) shows the image with
    # north up; mask contour coords then map directly to SVG pixel coords.
    hrl  = np.flipud(hrl)
    co_s = np.flipud(co_s)

    H, W = hrl.shape

    # --- read pixel scale + beam ---
    hdr = fits.open(FITS_HDR)[0].header
    pix_arcsec  = abs(hdr.get("CDELT1", 0)) * 3600
    bmaj_arcsec = hdr.get("BMAJ", 0) * 3600
    bmin_arcsec = hdr.get("BMIN", 0) * 3600
    bpa_deg     = hdr.get("BPA", 0)
    beam_pix    = beam_area_pix(bmaj_arcsec, bmin_arcsec, pix_arcsec)
    print(f"pix={pix_arcsec}″  beam={bmaj_arcsec:.3f}″×{bmin_arcsec:.3f}″ "
          f"PA={bpa_deg:.1f}°  beam_area={beam_pix:.2f}px²")

    # --- noise per pixel from off-source mom-0 region ---
    # The footprint is `co_s > 3*sigma_co`. Pixels with smoothed CO well
    # below σ_CO are line-free; we sigma-clip their mom-0 std to get the
    # per-pixel noise floor σ_pix (in Jy·km/s/beam). This feeds the
    # aperture-integrated σ_F = σ_pix * sqrt(N_pix / beam_area_pix).
    off_source = (co_s < 1.0 * sigma_co) & np.isfinite(hrl)
    from astropy.stats import sigma_clipped_stats
    _, _, sigma_per_pix = sigma_clipped_stats(hrl[off_source], sigma=3.0, maxiters=5)
    sigma_per_pix = float(sigma_per_pix)
    print(f"off-source pixels: {int(off_source.sum())}  σ_per_pix = {sigma_per_pix:.5f} Jy·km/s/beam")

    # --- crop to square around bright source for a tight display ---
    # Use the global pipeline mask as a centroid hint
    src_mask = np.flipud(pa["spatial_mask_hrl"].astype(bool))
    ys, xs = np.where(src_mask)
    if len(xs) == 0:
        cy, cx = H // 2, W // 2
        side = min(H, W)
    else:
        cy = (ys.min() + ys.max()) // 2
        cx = (xs.min() + xs.max()) // 2
        ext = max(ys.max() - ys.min(), xs.max() - xs.min())
        side = int(ext * 2.4) + 20   # generous margin: CoG should reach plateau
        side = min(side, min(H, W))
    half = side // 2
    y0 = max(0, cy - half); y1 = min(H, cy + half)
    x0 = max(0, cx - half); x1 = min(W, cx + half)
    if y0 < 0: y1 += -y0; y0 = 0
    if x0 < 0: x1 += -x0; x0 = 0
    if y1 > H: y0 -= (y1 - H); y1 = H
    if x1 > W: x0 -= (x1 - W); x1 = W
    y0 = max(0, y0); x0 = max(0, x0)

    sub_hrl  = hrl[y0:y1, x0:x1]
    sub_co_s = co_s[y0:y1, x0:x1]
    Hs, Ws = sub_hrl.shape
    half_w_arcsec = Ws * pix_arcsec / 2.0
    half_h_arcsec = Hs * pix_arcsec / 2.0
    print(f"crop {Hs}×{Ws}  span={2*half_w_arcsec:.1f}″×{2*half_h_arcsec:.1f}″")

    # --- background PNG ---
    finite = np.isfinite(sub_hrl)
    vmax = float(np.percentile(np.abs(sub_hrl[finite]), 99)) if finite.any() else 1.0
    fig = plt.figure(figsize=(5.2, 5.6), dpi=130)
    ax  = fig.add_axes([0.10, 0.16, 0.86, 0.78])
    cax = fig.add_axes([0.18, 0.07, 0.68, 0.025])
    im = ax.imshow(
        sub_hrl, cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
        interpolation="nearest", aspect="equal",
        extent=[-half_w_arcsec, +half_w_arcsec, -half_h_arcsec, +half_h_arcsec],
        origin="upper",
    )
    # beam ellipse, lower-left corner
    if bmaj_arcsec > 0:
        bm = Ellipse(
            xy=(-half_w_arcsec + max(bmaj_arcsec, bmin_arcsec) * 1.1,
                -half_h_arcsec + max(bmaj_arcsec, bmin_arcsec) * 1.1),
            width=bmaj_arcsec, height=bmin_arcsec, angle=bpa_deg,
            facecolor="white", edgecolor="black", linewidth=1.0, alpha=0.9,
        )
        ax.add_patch(bm)
    # ticks: pick a nice step
    span = 2 * half_w_arcsec
    for nice in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20):
        if span / nice <= 6:
            step = nice; break
    else:
        step = 50
    nmax = int(half_w_arcsec / step)
    ticks = np.arange(-nmax, nmax + 1) * step
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(direction="in", length=4, labelsize=8)
    ax.set_xlabel("Δα offset (arcsec)", fontsize=9)
    ax.set_ylabel("Δδ offset (arcsec)", fontsize=9)
    ax.set_title(f"{LABEL} H30α moment-0", fontsize=11, pad=8)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("HRL moment-0 (Jy·km/s/beam)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    bg_png = OUT / f"cog_bg_{GALAXY}.png"
    fig.savefig(bg_png, dpi=130, facecolor="white")
    plt.close(fig)
    # also record the PNG's pixel dimensions so the SVG overlay aligns
    from PIL import Image
    bg_w, bg_h = Image.open(bg_png).size
    # the image axes occupy [0.10, 0.16, 0.86, 0.78] of the figure
    ax_left   = 0.10 * bg_w
    ax_bottom = 0.16 * bg_h
    ax_width  = 0.86 * bg_w
    ax_height = 0.78 * bg_h

    # ============================================================
    # Method A — σ-dilation
    # ============================================================
    sigmas = np.linspace(32.0, 2.0, 30)
    method_A = []
    for sigma in sigmas:
        raw = sub_co_s > sigma * sigma_co
        if not raw.any():
            method_A.append({
                "sigma": float(sigma),
                "flux_jykms": 0.0,
                "sn": 0.0,
                "npix": 0,
                "contour": [],
            })
            continue
        # keep largest connected component
        labels, n = ndi.label(raw)
        sizes = ndi.sum(raw, labels, range(1, n + 1))
        mask  = labels == (1 + int(np.argmax(sizes)))
        npix  = int(mask.sum())
        # flux = sum over mask in (Jy·km/s)/beam → Jy·km/s divide by beam pixels
        flux  = float(sub_hrl[mask].sum()) / beam_pix
        # σ_F = σ_per_pix · sqrt(N_beam) where N_beam = N_pix / beam_pix
        sigma_F = sigma_per_pix * np.sqrt(max(npix / beam_pix, 1e-9))
        sn_val  = flux / sigma_F if sigma_F > 0 else 0.0
        # outline: collect all contour segments and downsample
        contours = measure.find_contours(mask.astype(float), 0.5)
        # convert each to arcsec offset from crop centre
        # find_contours returns (row, col) in array coords
        polys = []
        for c in contours:
            if len(c) < 4: continue
            # downsample to keep JSON compact
            stride = max(1, len(c) // 80)
            c = c[::stride]
            xs_a = (c[:, 1] - Ws / 2) * pix_arcsec
            ys_a = (c[:, 0] - Hs / 2) * pix_arcsec
            # SVG y-axis points down; our origin='upper' image has y=0 at top,
            # so an array row that grows downward corresponds to ys_a growing
            # downward too. We flip the sign so positive y means north (up
            # on screen, matching the Δδ axis label).
            ys_a = -ys_a
            polys.append([[float(x), float(y)] for x, y in zip(xs_a, ys_a)])
        method_A.append({
            "sigma": float(sigma),
            "flux_jykms": flux,
            "sn": float(sn_val),
            "npix": npix,
            "contour": polys,
        })

    # ============================================================
    # Method B — ellipse aperture
    # ============================================================
    # Seed for ellipse fit: intensity-weighted PCA on (smoothed CO > 6σ)
    seed = sub_co_s > 6.0 * sigma_co
    if seed.any():
        yy, xx = np.indices(sub_co_s.shape)
        w = sub_co_s[seed]
        cy_seed = float((yy[seed] * w).sum() / w.sum())
        cx_seed = float((xx[seed] * w).sum() / w.sum())
        dx = xx[seed] - cx_seed; dy = yy[seed] - cy_seed
        cov = np.cov(np.vstack([dx, dy]), aweights=w)
        vals, vecs = np.linalg.eigh(cov)
        # vals sorted ascending; major axis = vecs[:, 1]
        a0 = float(np.sqrt(max(vals[1], 0)))
        b0 = float(np.sqrt(max(vals[0], 0)))
        # PA: angle of major-axis eigenvector from +x (east) axis, in degrees
        pa_rad = float(np.arctan2(vecs[1, 1], vecs[0, 1]))
        pa_deg = float(np.degrees(pa_rad))
    else:
        cy_seed = Hs / 2; cx_seed = Ws / 2; a0 = 5.0; b0 = 3.0; pa_deg = 0.0

    cx_arcsec_seed = (cx_seed - Ws / 2) * pix_arcsec
    cy_arcsec_seed = -(cy_seed - Hs / 2) * pix_arcsec  # flip to N-up

    scales = np.linspace(0.4, 4.0, 30)
    method_B = []
    yy, xx = np.indices(sub_hrl.shape)
    for s in scales:
        a = a0 * s; b = b0 * s
        # mask = pixels inside scaled ellipse (rotation by PA)
        dx = xx - cx_seed; dy = yy - cy_seed
        th = np.radians(pa_deg)
        xr =  dx * np.cos(th) + dy * np.sin(th)
        yr = -dx * np.sin(th) + dy * np.cos(th)
        mask = (xr / max(a, 1e-9)) ** 2 + (yr / max(b, 1e-9)) ** 2 <= 1
        npix = int(mask.sum())
        flux = float(sub_hrl[mask].sum()) / beam_pix
        sigma_F = sigma_per_pix * np.sqrt(max(npix / beam_pix, 1e-9))
        sn_val  = flux / sigma_F if sigma_F > 0 else 0.0
        method_B.append({
            "scale":     float(s),
            "a_arcsec":  float(a * pix_arcsec),
            "b_arcsec":  float(b * pix_arcsec),
            "flux_jykms": flux,
            "sn":        float(sn_val),
            "npix":      npix,
        })

    # ============================================================
    # JSON bundle
    # ============================================================
    bundle = {
        "galaxy": LABEL,
        "sigma_per_pix_jykms": sigma_per_pix,
        "background": {
            "png": bg_png.name,
            "png_w_px": int(bg_w),
            "png_h_px": int(bg_h),
            "ax_left":   ax_left,
            "ax_bottom": ax_bottom,
            "ax_width":  ax_width,
            "ax_height": ax_height,
            "data_half_w_arcsec": half_w_arcsec,
            "data_half_h_arcsec": half_h_arcsec,
        },
        "pix_arcsec": pix_arcsec,
        "beam": {"bmaj_arcsec": bmaj_arcsec, "bmin_arcsec": bmin_arcsec, "bpa_deg": bpa_deg},
        "method_A": {
            "name": "σ-dilation",
            "x_label": "σ threshold (× σ_CO)",
            "x_decreasing_grows_mask": True,
            "steps": method_A,
        },
        "method_B": {
            "name": "Ellipse aperture (intensity-weighted PCA fit)",
            "x_label": "semi-major axis a (arcsec)",
            "centre_arcsec":  [cx_arcsec_seed, cy_arcsec_seed],
            "pa_deg": pa_deg,
            "b_over_a": (b0 / a0) if a0 > 0 else 1.0,
            "steps": method_B,
        },
    }
    out_json = OUT / "cog_data.json"
    out_json.write_text(json.dumps(bundle, indent=1))
    print(f"\nWrote {out_json}  ({out_json.stat().st_size // 1024} KB)")
    print(f"      {bg_png}")
    print(f"\n=== Method A (σ-dilation) ===")
    for s in method_A[::5]:
        print(f"  σ={s['sigma']:5.2f}  npix={s['npix']:5d}  F={s['flux_jykms']:.4f}  S/N={s['sn']:5.2f}")
    print(f"\n=== Method B (ellipse) ===")
    for s in method_B[::5]:
        print(f"  a={s['a_arcsec']:5.2f}″  npix={s['npix']:5d}  F={s['flux_jykms']:.4f}  S/N={s['sn']:5.2f}")


if __name__ == "__main__":
    main()
