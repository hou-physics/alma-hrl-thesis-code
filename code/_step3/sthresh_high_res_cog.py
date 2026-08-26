"""
High-resolution sthresh-CoG for clean detections.

For each galaxy, sweep sthresh linearly from scan-best DOWN to 3σ in
30 even steps. Theoretical expectation: F(N) rises monotonically as
sthresh drops (mask grows) and converges to a plateau when mask
encompasses the source's full extent.

Plot F vs Nbeam per galaxy + a 7-panel composite.
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
import cmasher  # noqa: F401  (registers cmr.* colormaps)

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402

C_KMS = 299792.458
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")
LINE_REST_HZ = {"H30a": 231.900928e9, "H40a": 99.022952e9}

N_STEPS = 30
STHRESH_LOW = 2.0  # σ floor (was 3.0; widened per user request to expose plateau)


@dataclass
class GalaxyConfig:
    name: str
    cube_path: str
    z: float
    line: str
    W_kms: float
    sf_scan: int
    sthresh_scan: float
    signal_ra: float
    signal_dec: float
    signal_size_arcsec: float           # box side (or RA side if rectangular)
    signal_dec_size_arcsec: float | None # None means square (use signal_size_arcsec)
    noise_ra: float
    noise_dec: float
    noise_size_arcsec: float
    noise_dec_size_arcsec: float | None


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
    return mask


def high_res_sthresh_cog(cfg: GalaxyConfig):
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
    print(f"  cube {cube.shape}, dv={dv:.2f}, pix/beam={ppb:.2f}, "
          f"beam_FWHM={beam_fwhm:.2f}")

    wcs2 = WCS(hdr).celestial
    sig_dec = cfg.signal_dec_size_arcsec or cfg.signal_size_arcsec
    noi_dec = cfg.noise_dec_size_arcsec or cfg.noise_size_arcsec
    sig_box = make_box_mask(wcs2, ny, nx,
                            cfg.signal_ra, cfg.signal_dec,
                            cfg.signal_size_arcsec, sig_dec)
    noise_box = make_box_mask(wcs2, ny, nx,
                              cfg.noise_ra, cfg.noise_dec,
                              cfg.noise_size_arcsec, noi_dec)

    in_line = np.abs(v_chan) <= cfg.W_kms / 2.0
    mom0 = np.nansum(cube[in_line, :, :], axis=0) * dv

    kernel_fwhm = cfg.sf_scan * beam_fwhm
    mom0_smooth = smooth_fft(mom0, kernel_fwhm)
    sigma_smooth = float(np.nanstd(mom0_smooth[noise_box]))

    # Cap upper sthresh at the actual HRL self-mask peak / σ. The CO-based
    # scan-best sthresh is not directly comparable to HRL-self-σ, so use the
    # in-signal-box peak as the natural HIGH endpoint instead.
    peak_in_sig = float(np.nanmax(mom0_smooth[sig_box]))
    sthresh_peak = peak_in_sig / max(sigma_smooth, 1e-9)
    # Step inward 5% from the peak so the highest cell has ≥ 1 pixel
    sthresh_high_eff = 0.95 * sthresh_peak
    print(f"  scan params: sf={cfg.sf_scan}, sthresh_scan(CO)={cfg.sthresh_scan}; "
          f"σ_HRL_smooth={sigma_smooth:.4f}")
    print(f"  HRL peak in sig_box = {peak_in_sig:.4f} → "
          f"sthresh_peak/σ = {sthresh_peak:.1f}, using high_eff = {sthresh_high_eff:.1f}")

    if sthresh_high_eff <= STHRESH_LOW:
        # source too weak — sweep down to 1σ instead
        sthresh_arr = np.linspace(sthresh_high_eff, 1.0, N_STEPS)
        print(f"  WARN: peak ≤ 3σ, extending down to 1σ")
    else:
        sthresh_arr = np.linspace(sthresh_high_eff, STHRESH_LOW, N_STEPS)

    # apply — keep the actual mask arrays for the smallest / middle / largest
    # so we can overlay them on mom-0
    Nbeam_arr = []
    F_arr = []
    Npix_arr = []
    masks_for_viz: dict[int, np.ndarray] = {}
    viz_indices = {0, N_STEPS // 2, N_STEPS - 1}
    for i, st in enumerate(sthresh_arr):
        mask = (mom0_smooth > st * sigma_smooth) & sig_box
        npix = int(mask.sum())
        if i in viz_indices:
            masks_for_viz[i] = mask.copy()
        if npix == 0:
            Nbeam_arr.append(0.0)
            F_arr.append(0.0)
            Npix_arr.append(0)
            continue
        F = float(np.nansum(mom0[mask]) / ppb)
        Nbeam_arr.append(npix / ppb)
        F_arr.append(F)
        Npix_arr.append(npix)
    Nbeam_arr = np.array(Nbeam_arr)
    F_arr = np.array(F_arr)
    print(f"  Nbeam range: [{Nbeam_arr.min():.2f}, {Nbeam_arr.max():.2f}]")
    print(f"  F range:     [{F_arr.min():+.3f}, {F_arr.max():+.3f}]")
    print(f"  F at scan-best (i=0): {F_arr[0]:+.3f}")
    print(f"  F at 3σ floor (i=29): {F_arr[-1]:+.3f}")
    print(f"  growth ratio F(3σ)/F(scan): {F_arr[-1]/max(F_arr[0],1e-9):.2f}")

    # per-galaxy plot
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: F vs Nbeam
    ax = axes[0]
    ax.plot(Nbeam_arr, F_arr, "o-", color="C0", markersize=5)
    ax.scatter([Nbeam_arr[0]], [F_arr[0]], marker="*", color="red",
               s=200, zorder=10,
               label=f"scan-best sthresh={cfg.sthresh_scan}")
    ax.scatter([Nbeam_arr[-1]], [F_arr[-1]], marker="X", color="black",
               s=120, zorder=10, label=f"3σ floor")
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mask size — Nbeam")
    ax.set_ylabel("Integrated flux F (Jy·km/s)")
    ax.set_title(f"{cfg.name} — F vs Nbeam (30-cell linear sthresh sweep)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # right: F vs sthresh (more informative — shows the sweep axis directly)
    ax = axes[1]
    ax.plot(sthresh_arr, F_arr, "o-", color="C2", markersize=5)
    ax.scatter([sthresh_arr[0]], [F_arr[0]], marker="*", color="red",
               s=200, zorder=10)
    ax.scatter([sthresh_arr[-1]], [F_arr[-1]], marker="X", color="black",
               s=120, zorder=10)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("sthresh (σ)")
    ax.set_ylabel("Integrated flux F (Jy·km/s)")
    ax.invert_xaxis()  # scan-best (high sthresh) on LEFT, 3σ on RIGHT
    ax.set_title(f"{cfg.name} — F vs sthresh (sweep: scan-best → 3σ)")
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"{cfg.name}  line={cfg.line}, W={cfg.W_kms:.0f} km/s, sf={cfg.sf_scan}",
        fontsize=11)
    plt.tight_layout()
    out_png = out_dir / "sthresh_high_res_cog.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"  saved {out_png}")

    # ----------- mom-0 image with smallest / middle / largest mask outlines -----------
    mom0_disp = mom0.copy()
    finite_mask = np.isfinite(mom0_disp) & sig_box
    if finite_mask.any():
        vmax = float(np.nanpercentile(mom0_disp[finite_mask], 99))
        vmin = float(np.nanpercentile(mom0_disp[finite_mask], 1))
    else:
        vmax = float(np.nanmax(mom0_disp)) if np.any(np.isfinite(mom0_disp)) else 1.0
        vmin = -vmax / 5.0

    # crop view to sig_box + small margin for context
    ys, xs = np.where(sig_box)
    pad = max(20, int(0.2 * (xs.max() - xs.min() + 1)))
    x0 = max(0, xs.min() - pad)
    x1 = min(nx, xs.max() + pad + 1)
    y0 = max(0, ys.min() - pad)
    y1 = min(ny, ys.max() + pad + 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mom0_disp, origin="lower", cmap="cmr.cosmic",
                   vmin=vmin, vmax=vmax)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    plt.colorbar(im, ax=ax, label="mom-0 (Jy/beam · km/s)")

    # sig_box outline (dashed white)
    from matplotlib.patches import Rectangle
    sig_y, sig_x = np.where(sig_box)
    rect = Rectangle((sig_x.min() - 0.5, sig_y.min() - 0.5),
                     sig_x.max() - sig_x.min() + 1,
                     sig_y.max() - sig_y.min() + 1,
                     fill=False, edgecolor="white", linestyle=":",
                     linewidth=1.5, label="signal box")
    ax.add_patch(rect)

    # mask outlines
    overlay_specs = [
        (0,             "red",    f"smallest mask (sthresh={sthresh_arr[0]:.1f}σ)"),
        (N_STEPS // 2,  "orange", f"middle mask (sthresh={sthresh_arr[N_STEPS//2]:.1f}σ)"),
        (N_STEPS - 1,   "yellow", f"largest mask (sthresh={sthresh_arr[-1]:.1f}σ)"),
    ]
    for idx, color, label in overlay_specs:
        m = masks_for_viz.get(idx)
        if m is not None and m.any():
            ax.contour(m.astype(int), levels=[0.5],
                       colors=color, linewidths=1.5)
            ax.plot([], [], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(
        f"{cfg.name} — HRL mom-0 over W={cfg.W_kms:.0f} km/s\n"
        f"smallest → largest mask outlines (sthresh sweep "
        f"{sthresh_arr[0]:.1f}σ → {sthresh_arr[-1]:.1f}σ)"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    out_mom0 = out_dir / "sthresh_sweep_mom0_overlay.png"
    plt.savefig(out_mom0, dpi=140)
    plt.close()
    print(f"  saved {out_mom0}")

    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "line": cfg.line,
        "sthresh": sthresh_arr,
        "Nbeam": Nbeam_arr,
        "F": F_arr,
        "Npix": np.array(Npix_arr),
        "F_scan": F_arr[0],
        "F_floor": F_arr[-1],
    }


GALAXIES = [
    # ---- 4 existing in baseline-diag sample ----
    GalaxyConfig(
        name="NGC5253",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_H30a_pbcor.fits",
        z=0.0014, line="H30a", W_kms=135.0,
        sf_scan=1, sthresh_scan=16.2,
        signal_ra=204.982993, signal_dec=-31.640281,
        signal_size_arcsec=20.0, signal_dec_size_arcsec=None,
        noise_ra=204.987952, noise_dec=-31.640681,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC4945",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits",
        z=0.00188, line="H30a", W_kms=315.0,
        sf_scan=2, sthresh_scan=8.6,
        signal_ra=196.36260, signal_dec=-49.46942,
        signal_size_arcsec=17.0, signal_dec_size_arcsec=19.0,
        noise_ra=196.36901, noise_dec=-49.46442,
        noise_size_arcsec=7.0, noise_dec_size_arcsec=7.0,
    ),
    GalaxyConfig(
        name="NGC3628",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_H30a_pbcor.fits",
        z=0.002772, line="H30a", W_kms=345.0,
        sf_scan=1, sthresh_scan=30.3,
        signal_ra=170.0708, signal_dec=13.5888,
        signal_size_arcsec=40.0, signal_dec_size_arcsec=12.0,
        noise_ra=170.0707, noise_dec=13.5944,
        noise_size_arcsec=15.0, noise_dec_size_arcsec=13.0,
    ),
    GalaxyConfig(
        name="He2-10",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pbcor.fits",
        z=0.002912, line="H40a", W_kms=75.0,
        sf_scan=1, sthresh_scan=5.7,
        signal_ra=129.06329, signal_dec=-26.40935,
        signal_size_arcsec=16.0, signal_dec_size_arcsec=None,
        noise_ra=129.05833, noise_dec=-26.40491,
        noise_size_arcsec=11.0, noise_dec_size_arcsec=None,
    ),
    # ---- 3 new (already in cog_clean_detection_poc.py) ----
    GalaxyConfig(
        name="Circinus",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/ESO097-013/ESO097-013_H40a_pbcor.fits",
        z=0.001448, line="H40a", W_kms=165.0,
        sf_scan=7, sthresh_scan=19.9,
        signal_ra=213.29152, signal_dec=-65.33909,
        signal_size_arcsec=35.0, signal_dec_size_arcsec=None,
        noise_ra=213.282197, noise_dec=-65.335201,
        noise_size_arcsec=7.0, noise_dec_size_arcsec=None,
    ),
    GalaxyConfig(
        name="NGC253",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC253/NGC253_H40a_pbcor.fits",
        z=0.00080, line="H40a", W_kms=180.0,
        sf_scan=3, sthresh_scan=32.0,
        signal_ra=11.88808, signal_dec=-25.28838,
        signal_size_arcsec=31.2, signal_dec_size_arcsec=20.0,
        noise_ra=11.87983, noise_dec=-25.28024,
        noise_size_arcsec=12.8, noise_dec_size_arcsec=12.8,
    ),
    GalaxyConfig(
        name="NGC3627",
        cube_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3627/NGC3627_H30a_pbcor.fits",
        z=0.0024, line="H30a",
        # NGC 3627 scan-best is (sf=10, sthresh=20, W=60) but at sf=10 the
        # signal washes below 3σ. Use sf=3 + W=300 as in cog_clean_detection_poc.
        W_kms=300.0, sf_scan=3, sthresh_scan=10.0,
        signal_ra=170.06254, signal_dec=12.99161,
        signal_size_arcsec=40.0, signal_dec_size_arcsec=None,
        noise_ra=170.04688, noise_dec=13.00549,
        noise_size_arcsec=20.0, noise_dec_size_arcsec=None,
    ),
]


def main():
    results = [high_res_sthresh_cog(g) for g in GALAXIES]

    # 7-panel composite (2 rows × 4 cols, last cell empty)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["Nbeam"], r["F"], "o-", color="C0", markersize=4)
        ax.scatter([r["Nbeam"][0]], [r["F"][0]], marker="*", color="red",
                   s=150, zorder=10, label="scan-best")
        ax.scatter([r["Nbeam"][-1]], [r["F"][-1]], marker="X", color="black",
                   s=80, zorder=10, label="3σ floor")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Nbeam")
        ax.set_ylabel("F (Jy·km/s)")
        ax.set_title(
            f"{r['name']} ({r['line']})\n"
            f"F: {r['F_scan']:+.2f} → {r['F_floor']:+.2f} "
            f"(ratio {r['F_floor']/max(r['F_scan'],1e-9):.2f})"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.suptitle(
        "Spatial CoG: 30-cell linear sthresh sweep from scan-best → 3σ",
        fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_ROOT / "_sthresh_high_res_cog_7panel.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nsaved 7-panel composite {out}")


if __name__ == "__main__":
    main()
