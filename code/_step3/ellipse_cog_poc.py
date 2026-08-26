"""
PoC: Ellipse-scale CoG (alternative to sthresh-CoG).

Procedure per galaxy:
  1. Compute mom-0 over scan-best W and smooth at scan-best sf.
  2. Build seed mask at high-effective sthresh (HRL self-mask equivalent
     of scan-best), AND with signal box, then keep only the connected
     component containing the source center pixel — single contiguous blob.
  3. Fit ellipse via intensity-weighted second-moment PCA on seed pixels.
     Returns (xc, yc, a, b, PA) — center, 1σ semi-axes, position angle.
  4. Sweep "scale" multiplier in [0.5, 5.0] over 30 cells; at each scale
     generate ellipse mask = {(x_rot/(a·scale))² + (y_rot/(b·scale))² ≤ 1};
     compute F(scale) = sum mom-0 over mask / pix_per_beam.

Output per galaxy:
  - left panel: F vs scale curve with scan-best marker
  - right panel: mom-0 with seed (cyan) + 5 scale ellipses overlaid
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy import ndimage
import cmasher  # noqa: F401
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402

C_KMS = 299792.458
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")
LINE_REST_HZ = {"H30a": 231.900928e9, "H40a": 99.022952e9}

N_SCALES = 30
SCALE_MIN = 0.5
SCALE_MAX = 5.0
# Sample scales for mom-0 overlay (subset of the sweep)
OVERLAY_SCALES = [0.5, 1.0, 2.0, 3.0, 5.0]


@dataclass
class GalaxyConfig:
    name: str
    cube_path: str
    z: float
    line: str
    W_kms: float
    sf_scan: int
    signal_ra: float
    signal_dec: float
    signal_size_arcsec: float
    signal_dec_size_arcsec: float | None
    noise_ra: float
    noise_dec: float
    noise_size_arcsec: float
    noise_dec_size_arcsec: float | None


# ---------- helpers ----------
def channel_velocities(hdr, rest_obs_hz):
    n = hdr["NAXIS3"]
    freqs = hdr["CRVAL3"] + (np.arange(n) - (hdr["CRPIX3"] - 1)) * hdr["CDELT3"]
    return (rest_obs_hz - freqs) / rest_obs_hz * C_KMS


def beam_fwhm_pix(hdr):
    return hdr["BMAJ"] / abs(hdr["CDELT1"])


def pix_per_beam(hdr):
    bmaj = hdr["BMAJ"]; bmin = hdr["BMIN"]; pix = abs(hdr["CDELT1"])
    return (np.pi / (4.0 * np.log(2.0))) * (bmaj / pix) * (bmin / pix)


def smooth_fft(image, kernel_fwhm_pix):
    if kernel_fwhm_pix <= 0:
        return image
    sigma = kernel_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))
    return convolve_fft(image, Gaussian2DKernel(sigma),
                        normalize_kernel=True,
                        nan_treatment="fill", boundary="fill")


def make_box_mask(wcs2d, ny, nx, ra, dec, ra_size, dec_size):
    xc, yc = wcs2d.world_to_pixel_values(ra, dec)
    pix_scale = abs(wcs2d.wcs.cdelt[0]) * 3600.0
    half_x = (ra_size / 2.0) / pix_scale
    half_y = (dec_size / 2.0) / pix_scale
    mask = np.zeros((ny, nx), dtype=bool)
    y0 = max(0, int(np.floor(yc - half_y)))
    y1 = min(ny, int(np.ceil(yc + half_y)))
    x0 = max(0, int(np.floor(xc - half_x)))
    x1 = min(nx, int(np.ceil(xc + half_x)))
    mask[y0:y1, x0:x1] = True
    return mask, (int(xc), int(yc))


def keep_component_containing(mask, cx, cy):
    """Return only the connected component of `mask` that contains pixel (cx, cy).
    If (cx, cy) is not inside any True pixel, return the largest component."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    # Try the named center first
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1] and labels[cy, cx] > 0:
        target_label = labels[cy, cx]
    else:
        # Fallback: largest component
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        target_label = int(np.argmax(sizes)) + 1
    return labels == target_label


def fit_ellipse_weighted(mask_pixels_yx, weights):
    """Intensity-weighted second-moment ellipse fit.
    Returns (xc, yc, a, b, PA_rad) — center, 1σ semi-axes (in pixels),
    position angle (radians, counterclockwise from +x)."""
    ys, xs = mask_pixels_yx
    w = np.asarray(weights, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0 or len(xs) < 3:
        return None
    xc = float((w * xs).sum() / w_sum)
    yc = float((w * ys).sum() / w_sum)
    dx = xs - xc
    dy = ys - yc
    Mxx = float((w * dx * dx).sum() / w_sum)
    Myy = float((w * dy * dy).sum() / w_sum)
    Mxy = float((w * dx * dy).sum() / w_sum)
    trace = Mxx + Myy
    det = Mxx * Myy - Mxy * Mxy
    disc = max(trace * trace / 4 - det, 0.0)
    lam1 = trace / 2 + np.sqrt(disc)
    lam2 = trace / 2 - np.sqrt(disc)
    a = float(np.sqrt(max(lam1, 0.0)))  # 1σ semi-major
    b = float(np.sqrt(max(lam2, 0.0)))  # 1σ semi-minor
    PA = float(0.5 * np.arctan2(2 * Mxy, Mxx - Myy))
    return xc, yc, a, b, PA


def ellipse_pixel_mask(shape, xc, yc, a, b, PA, scale):
    """Return bool mask of pixels inside the scaled ellipse."""
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    cos_pa, sin_pa = np.cos(PA), np.sin(PA)
    dx = xx - xc
    dy = yy - yc
    x_rot = dx * cos_pa + dy * sin_pa
    y_rot = -dx * sin_pa + dy * cos_pa
    return (x_rot / (a * scale)) ** 2 + (y_rot / (b * scale)) ** 2 <= 1.0


# ---------- per-galaxy pipeline ----------
def ellipse_cog_galaxy(cfg: GalaxyConfig):
    print(f"\n=== {cfg.name} ===")
    bundle = load_cube(cfg.cube_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape
    rest_obs_hz = LINE_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv = float(np.abs(np.median(np.diff(v_chan))))
    ppb = pix_per_beam(hdr)
    beam_fwhm = beam_fwhm_pix(hdr)

    wcs2 = WCS(hdr).celestial
    sig_dec = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    noi_dec = cfg.noise_dec_size_arcsec or cfg.noise_size_arcsec
    sig_box, (cx, cy) = make_box_mask(wcs2, ny, nx,
                                      cfg.signal_ra, cfg.signal_dec,
                                      cfg.signal_size_arcsec, sig_dec)
    noise_box, _ = make_box_mask(wcs2, ny, nx,
                                 cfg.noise_ra, cfg.noise_dec,
                                 cfg.noise_size_arcsec, noi_dec)

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv

    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))
    peak_in_sig = float(np.nanmax(mom0_smooth[sig_box]))
    sthresh_peak = peak_in_sig / max(sigma_smooth, 1e-9)
    print(f"  σ_smooth={sigma_smooth:.4f}, peak={peak_in_sig:.4f}, "
          f"peak/σ={sthresh_peak:.1f}σ")

    # ---- seed mask: HRL emission at 3σ, connected component at brightest pixel ----
    # 3σ chosen because:
    #   - high enough to exclude noise blobs (typical false-positive rate)
    #   - low enough to capture the full extent of the source (not just peak)
    #   - matches the lowest sthresh in the production scan ladder
    # Connected-component constraint ensures we get one contiguous source region,
    # not the union of all 3σ blobs across the box (which would re-introduce
    # the noise-accumulation problem we just diagnosed).
    seed_sthresh = 3.0
    raw_mask = (mom0_smooth > seed_sthresh * sigma_smooth) & sig_box
    print(f"  raw mask at {seed_sthresh}σ (before CC): {int(raw_mask.sum())} pix")
    sig_pixels = np.where(sig_box)
    brightest_local_idx = int(np.argmax(mom0_smooth[sig_pixels]))
    by, bx = sig_pixels[0][brightest_local_idx], sig_pixels[1][brightest_local_idx]
    seed = keep_component_containing(raw_mask, bx, by)
    if not seed.any():
        for try_st in [2.5, 2.0]:
            raw_mask = (mom0_smooth > try_st * sigma_smooth) & sig_box
            seed = keep_component_containing(raw_mask, bx, by)
            if seed.any():
                print(f"  fell back to sthresh={try_st:.1f}σ for seed")
                seed_sthresh = try_st
                break
    print(f"  seed (CC at brightest pixel, sthresh={seed_sthresh}σ): "
          f"{int(seed.sum())} pix = {seed.sum()/ppb:.2f} beams")

    # ---- fit ellipse ----
    ys, xs = np.where(seed)
    weights = mom0_smooth[ys, xs]
    weights = np.maximum(weights, 0)  # avoid neg weights
    fit = fit_ellipse_weighted((ys, xs), weights)
    if fit is None:
        print(f"  WARN: ellipse fit failed, skipping")
        bundle.data = None  # type: ignore
        return None
    xc, yc, a_1sig, b_1sig, PA = fit
    # convert 1σ axes to "seed-equivalent" axes — use 1.5× as default
    # (capture ~68% of weighted area; expansion by scale multiplier from here)
    a_seed = a_1sig
    b_seed = b_1sig
    print(f"  ellipse fit: center=({xc:.1f}, {yc:.1f}), "
          f"a={a_seed:.2f} pix, b={b_seed:.2f} pix, PA={np.degrees(PA):.1f}°")
    print(f"             axis ratio b/a = {b_seed/max(a_seed,1e-9):.2f}, "
          f"semi-major (arcsec) = {a_seed*abs(hdr['CDELT1'])*3600:.2f}")

    # ---- scale sweep ----
    scale_arr = np.linspace(SCALE_MIN, SCALE_MAX, N_SCALES)
    F_arr = []
    Nbeam_arr = []
    for s in scale_arr:
        m = ellipse_pixel_mask((ny, nx), xc, yc, a_seed, b_seed, PA, s)
        F = float(np.nansum(mom0[m]) / ppb)
        F_arr.append(F)
        Nbeam_arr.append(int(m.sum()) / ppb)
    F_arr = np.array(F_arr)
    Nbeam_arr = np.array(Nbeam_arr)
    print(f"  F range: [{F_arr.min():+.3f}, {F_arr.max():+.3f}] Jy·km/s")
    print(f"  F at scale=1: {F_arr[np.argmin(np.abs(scale_arr-1.0))]:+.3f}")
    print(f"  F at scale=3: {F_arr[np.argmin(np.abs(scale_arr-3.0))]:+.3f}")

    # ---- plot ----
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # left panel: F vs scale
    ax = axes[0]
    ax.plot(scale_arr, F_arr, "o-", color="C2", markersize=5)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8, label="scale=1 (seed)")
    ax.set_xlabel("Scale factor (×seed ellipse)")
    ax.set_ylabel("F (Jy·km/s)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_title(
        f"{cfg.name} — F vs ellipse scale\n"
        f"seed Nbeam = {seed.sum()/ppb:.1f}, a={a_seed:.1f} pix, "
        f"b={b_seed:.1f} pix, PA={np.degrees(PA):.0f}°, b/a={b_seed/max(a_seed,1e-9):.2f}"
    )
    ax.legend()

    # right panel: mom-0 with multiple scale ellipses
    ax = axes[1]
    # display mom-0 in a tight bbox around source + max ellipse
    max_ext_pix = a_seed * SCALE_MAX
    pad = max(int(max_ext_pix * 1.3), 20)
    x_lo = max(0, int(xc - pad)); x_hi = min(nx, int(xc + pad))
    y_lo = max(0, int(yc - pad)); y_hi = min(ny, int(yc + pad))
    mom0_disp = mom0.copy()
    finite_mask = np.isfinite(mom0_disp) & sig_box
    if finite_mask.any():
        vmax = float(np.nanpercentile(mom0_disp[finite_mask], 99))
        vmin = float(np.nanpercentile(mom0_disp[finite_mask], 1))
    else:
        vmax = float(np.nanmax(mom0_disp)); vmin = -vmax/5
    im = ax.imshow(mom0_disp, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    plt.colorbar(im, ax=ax, label="mom-0 (Jy/beam · km/s)")

    # seed outline (cyan)
    ax.contour(seed.astype(int), levels=[0.5], colors="cyan", linewidths=1.5)
    ax.plot([], [], color="cyan", linewidth=1.5,
            label=f"seed (CC-constrained, {int(seed.sum())} pix)")

    # ellipses at the OVERLAY_SCALES
    colors = ["red", "orange", "yellow", "white", "magenta"]
    for s, c in zip(OVERLAY_SCALES, colors):
        ell = Ellipse((xc, yc), 2*a_seed*s, 2*b_seed*s,
                      angle=np.degrees(PA),
                      fill=False, edgecolor=c, linewidth=1.5)
        ax.add_patch(ell)
        ax.plot([], [], color=c, linewidth=1.5, label=f"scale={s}")

    # signal box outline (dotted white)
    sb_y, sb_x = np.where(sig_box)
    rect = Rectangle((sb_x.min() - 0.5, sb_y.min() - 0.5),
                     sb_x.max() - sb_x.min() + 1,
                     sb_y.max() - sb_y.min() + 1,
                     fill=False, edgecolor="white", linestyle=":",
                     linewidth=1.0)
    ax.add_patch(rect)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(
        f"{cfg.name} — mom-0 + seed (cyan) + ellipse scales"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle(
        f"{cfg.name}  line={cfg.line}  W={cfg.W_kms:.0f} km/s",
        fontsize=12)
    plt.tight_layout()
    out_png = out_dir / "ellipse_cog_poc.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "scale": scale_arr,
        "F": F_arr,
        "Nbeam": Nbeam_arr,
        "seed_Nbeam": seed.sum() / ppb,
        "ellipse_axes": (a_seed, b_seed),
        "ellipse_PA_deg": float(np.degrees(PA)),
    }


# ---------- configs (same as sthresh sweep) ----------
GALAXIES = [
    GalaxyConfig(
        name="NGC5253",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_H30a_pbcor.fits",
        z=0.0014, line="H30a", W_kms=135.0, sf_scan=1,
        signal_ra=204.982993, signal_dec=-31.640281,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=204.987952, noise_dec=-31.640681,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC4945",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits",
        z=0.00188, line="H30a", W_kms=315.0, sf_scan=2,
        signal_ra=196.36260, signal_dec=-49.46942,
        signal_size_arcsec=17.0, signal_dec_size_arcsec=19.0,
        noise_ra=196.36901, noise_dec=-49.46442,
        noise_size_arcsec=7.0, noise_dec_size_arcsec=7.0,
    ),
    GalaxyConfig(
        name="NGC3628",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_H30a_pbcor.fits",
        z=0.002772, line="H30a", W_kms=345.0, sf_scan=1,
        signal_ra=170.0708, signal_dec=13.5888,
        signal_size_arcsec=40.0, signal_dec_size_arcsec=12.0,
        noise_ra=170.0707, noise_dec=13.5944,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=13.0,
    ),
    GalaxyConfig(
        name="He2-10",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pbcor.fits",
        z=0.002912, line="H40a", W_kms=75.0, sf_scan=1,
        signal_ra=129.06329, signal_dec=-26.40935,
        signal_size_arcsec=16.0, signal_dec_size_arcsec=None,
        noise_ra=129.05833, noise_dec=-26.40491,
        noise_size_arcsec=11.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="Circinus",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/ESO097-013/ESO097-013_H40a_pbcor.fits",
        z=0.001448, line="H40a", W_kms=165.0, sf_scan=7,
        signal_ra=213.29152, signal_dec=-65.33909,
        signal_size_arcsec=35.0, signal_dec_size_arcsec=None,
        noise_ra=213.282197, noise_dec=-65.335201,
        noise_size_arcsec=7.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC253",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/NGC253_H40a_pbcor.fits",
        z=0.00080, line="H40a", W_kms=180.0, sf_scan=3,
        signal_ra=11.88808, signal_dec=-25.28838,
        signal_size_arcsec=31.2, signal_dec_size_arcsec=20.0,
        noise_ra=11.87983, noise_dec=-25.28024,
        noise_size_arcsec=12.8, noise_dec_size_arcsec=12.8,
    ),
    GalaxyConfig(
        name="NGC3627",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3627/NGC3627_H30a_pbcor.fits",
        z=0.0024, line="H30a",
        W_kms=300.0, sf_scan=3,
        signal_ra=170.06254, signal_dec=12.99161,
        signal_size_arcsec=40.0, signal_dec_size_arcsec=None,
        noise_ra=170.04688, noise_dec=13.00549,
        noise_size_arcsec=20.0, noise_dec_size_arcsec=None,
    ),
]


def main():
    results = []
    for g in GALAXIES:
        r = ellipse_cog_galaxy(g)
        if r is not None:
            results.append(r)

    # 7-panel composite (F vs scale only)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["scale"], r["F"], "o-", color="C2", markersize=4)
        ax.axvline(1.0, color="gray", ls=":", lw=0.8)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("ellipse scale")
        ax.set_ylabel("F (Jy·km/s)")
        ax.set_title(
            f"{r['name']} ({r['line']})\n"
            f"seed Nbeam={r['seed_Nbeam']:.1f}, "
            f"a/b axes=({r['ellipse_axes'][0]:.1f}, "
            f"{r['ellipse_axes'][1]:.1f}) pix, PA={r['ellipse_PA_deg']:.0f}°"
        )
        ax.grid(alpha=0.3)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Ellipse-scale CoG: F vs scale factor "
        "(seed = best-scan connected component, ellipse fit via weighted 2nd moments)",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_ellipse_cog_7panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 7-panel composite {out}")


if __name__ == "__main__":
    main()
