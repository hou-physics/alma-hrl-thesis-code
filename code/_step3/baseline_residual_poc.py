"""
Multi-galaxy PoC: baseline-residual fingerprint subtraction along
dilation CoG.

For each target galaxy:
  1. Load HRL pbcor cube.
  2. Compute eps(x, y) = per-pixel mean over line-free channels.
  3. Use phase2 raw seed mask, dilate N_DILATION steps.
  4. At each step compute F_obs(N), F_residual(N) = (sum eps) * W * pix/beam,
     F_clean = F_obs - F_residual.
  5. Save 3-curve plot + eps map per galaxy + a combined comparison figure.

This is run-once exploratory code, not part of the production pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import cmasher  # noqa: F401
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from cube_io import load_cube  # noqa: E402


C_KMS = 299792.458
N_DILATION = 41
OUT_ROOT = Path("/Volumes/HouAstro/master/result_v2")

# H/RRL rest frequencies (Hz)
LINE_REST_HZ = {
    "H30a": 231.900928e9,
    "H40a": 99.022952e9,
}


@dataclass
class GalaxyConfig:
    name: str             # display + output dir name
    hrl_path: str
    seed_mask_path: str
    z: float
    W_kms: float
    line: str             # "H30a" or "H40a"
    n_dilation: int = N_DILATION


def channel_velocities(header, rest_hz_obs):
    n = header["NAXIS3"]
    cdelt = header["CDELT3"]
    crval = header["CRVAL3"]
    crpix = header["CRPIX3"]
    freqs = crval + (np.arange(n) - (crpix - 1)) * cdelt
    return (rest_hz_obs - freqs) / rest_hz_obs * C_KMS


def beam_pixels_per_beam(header):
    bmaj_deg = header["BMAJ"]
    bmin_deg = header["BMIN"]
    pix_deg = abs(header["CDELT1"])
    bmaj_pix = bmaj_deg / pix_deg
    bmin_pix = bmin_deg / pix_deg
    return (np.pi / (4.0 * np.log(2.0))) * bmaj_pix * bmin_pix


def run_galaxy(cfg: GalaxyConfig):
    print(f"\n{'='*60}\n[{cfg.name}] loading cube {Path(cfg.hrl_path).name}")
    bundle = load_cube(cfg.hrl_path)
    cube = bundle.data
    hdr = bundle.header
    nchan, ny, nx = cube.shape

    rest_obs_hz = LINE_REST_HZ[cfg.line] / (1.0 + cfg.z)
    v_chan = channel_velocities(hdr, rest_obs_hz)
    dv_kms = abs(np.median(np.diff(v_chan)))
    pix_per_beam = beam_pixels_per_beam(hdr)
    print(f"  nchan={nchan}, ny={ny}, nx={nx}, dv={dv_kms:.2f} km/s, "
          f"pix/beam={pix_per_beam:.2f}")
    print(f"  v range: [{v_chan.min():+.0f}, {v_chan.max():+.0f}] km/s")

    line_zone = np.abs(v_chan) <= cfg.W_kms / 2.0
    line_free = np.abs(v_chan) > (cfg.W_kms / 2.0 + dv_kms)
    n_line = int(line_zone.sum())
    n_free = int(line_free.sum())
    print(f"  line channels={n_line}, line-free channels={n_free}")

    # eps map (avoid loading whole cube twice — compute in one pass)
    print("  computing eps map ...")
    eps = np.nanmean(cube[line_free, :, :], axis=0)
    eps_finite = eps[np.isfinite(eps)]
    print(f"  eps stats: mean={np.nanmean(eps_finite):+.4e}, "
          f"std={np.nanstd(eps_finite):.4e} Jy/beam")

    # line-zone per-pixel integral (Jy/beam * km/s per pixel-spectrum)
    line_int_per_pix = np.nansum(cube[line_zone, :, :], axis=0) * dv_kms
    eps_x_W = eps * cfg.W_kms

    seed = np.load(cfg.seed_mask_path).astype(bool)
    assert seed.shape == (ny, nx), \
        f"{cfg.name}: seed shape {seed.shape} != cube xy {(ny, nx)}"
    print(f"  seed Npix={int(seed.sum())}, "
          f"Nbeam={seed.sum()/pix_per_beam:.2f}")

    masks = [seed.copy()]
    cur = seed.copy()
    for _ in range(cfg.n_dilation):
        cur = ndimage.binary_dilation(cur, iterations=1)
        masks.append(cur.copy())

    Npix_arr = np.array([int(m.sum()) for m in masks])
    Nbeam_arr = Npix_arr / pix_per_beam
    F_obs = np.array([np.nansum(line_int_per_pix[m]) / pix_per_beam
                      for m in masks])
    F_res = np.array([np.nansum(eps_x_W[m]) / pix_per_beam
                      for m in masks])
    F_clean = F_obs - F_res

    # noise floor of F_residual: ALMA pixels are correlated within a beam,
    # so the number of INDEPENDENT samples in a mask is N_beam, not N_pix.
    # std(eps across pixels) measures the per-beam noise level (since pixels
    # in one beam carry the same convolved value).
    # F_res = sum_beams (eps_beam * W). Each term has sigma = sigma_eps * W.
    # → sigma_F_res(N) = sqrt(N_beam) * sigma_eps * W
    sigma_eps_per_beam = float(np.nanstd(eps_finite))
    sigma_F_res = (sigma_eps_per_beam * np.sqrt(Nbeam_arr) * cfg.W_kms)
    # significance: |F_res| in units of its own noise floor
    F_res_nsigma = np.divide(F_res, sigma_F_res,
                             out=np.zeros_like(F_res),
                             where=sigma_F_res > 0)
    print(f"  max |F_res| nsigma across N: {np.max(np.abs(F_res_nsigma)):.2f}")

    # per-galaxy 3-curve plot
    out_dir = OUT_ROOT / cfg.name
    out_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(Nbeam_arr, F_obs, "o-", color="C3",
            label="F_obs (dilation CoG)", markersize=4)
    # F_residual with ±sigma band + 3σ markers
    ax.plot(Nbeam_arr, F_res, "s-", color="C2",
            label=r"F_residual = ($\Sigma\varepsilon$) $\times$ W",
            markersize=4)
    ax.fill_between(Nbeam_arr, F_res - sigma_F_res, F_res + sigma_F_res,
                    color="C2", alpha=0.20,
                    label="F_residual ±1σ (noise floor)")
    ax.plot(Nbeam_arr, F_clean, "^-", color="C0",
            label="F_clean = F_obs − F_residual", markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mask size — Nbeam")
    ax.set_ylabel("Flux (Jy·km/s)")
    ax.set_title(
        f"{cfg.name} — baseline-residual subtraction along dilation CoG\n"
        f"(line={cfg.line}, W={cfg.W_kms:.0f} km/s, "
        f"seed Nbeam={Nbeam_arr[0]:.2f}, dilation steps={cfg.n_dilation})"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "poc_baseline_residual_dilation.png"
    plt.savefig(plot_path, dpi=140)
    plt.close()
    print(f"  saved {plot_path}")

    # eps map
    fig, ax = plt.subplots(figsize=(7, 6))
    vlim = 3 * np.nanstd(eps_finite)
    im = ax.imshow(eps, origin="lower", cmap="RdBu_r",
                   vmin=-vlim, vmax=+vlim)
    ax.contour(masks[0], levels=[0.5], colors="black", linewidths=1.0)
    for i in [N_DILATION // 4, N_DILATION // 2, N_DILATION]:
        if i < len(masks):
            ax.contour(masks[i], levels=[0.5], colors="black",
                       linewidths=0.5, alpha=0.4)
    plt.colorbar(im, ax=ax, label="ε (Jy/beam)")
    ax.set_title(
        f"{cfg.name} — ε(x,y): per-pixel mean over line-free channels "
        f"(n={n_free})"
    )
    plt.tight_layout()
    eps_path = out_dir / "poc_baseline_eps_map.png"
    plt.savefig(eps_path, dpi=140)
    plt.close()
    print(f"  saved {eps_path}")

    # release the cube data before next galaxy
    del cube, line_int_per_pix
    bundle.data = None  # type: ignore

    return {
        "name": cfg.name,
        "Nbeam": Nbeam_arr,
        "F_obs": F_obs,
        "F_residual": F_res,
        "F_clean": F_clean,
        "sigma_F_res": sigma_F_res,
        "F_res_nsigma_max": float(np.max(np.abs(F_res_nsigma))),
        "eps_mean": float(np.nanmean(eps_finite)),
        "eps_std": float(np.nanstd(eps_finite)),
        "n_free": n_free,
    }


GALAXIES = [
    GalaxyConfig(
        name="NGC5253",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5253/NGC5253_H30a_pbcor.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/NGC5253/phase2_raw_step3/best_mask_hrl.npy",
        z=0.0014,
        W_kms=135.0,
        line="H30a",
    ),
    GalaxyConfig(
        name="NGC6810",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC6810/NGC6810_H30a_pbcor.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/NGC6810/phase2_raw_step3/best_mask_hrl.npy",
        z=0.006775,
        W_kms=435.0,
        line="H30a",
    ),
    GalaxyConfig(
        name="NGC4945",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC4945/NGC4945_H30a_spw1_v1_contsub.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/NGC4945/phase2_raw_step3/best_mask_hrl.npy",
        z=0.00188,
        W_kms=315.0,
        line="H30a",
    ),
    GalaxyConfig(
        name="NGC5135",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC5135/NGC5135_H30a_pbcor.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/NGC5135/phase2_raw_step3/best_mask_hrl.npy",
        z=0.013693,
        W_kms=345.0,
        line="H30a",
    ),
    GalaxyConfig(
        name="He2-10",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/He2-10/He2-10_H40a_pbcor.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/He2-10/phase2_raw_step3/best_mask_hrl.npy",
        z=0.002912,
        W_kms=75.0,
        line="H40a",
    ),
    GalaxyConfig(
        name="NGC3628",
        hrl_path="/Volumes/HouAstro/master/master_thesis/work_dir/NGC3628/NGC3628_H30a_pbcor.fits",
        seed_mask_path="/Volumes/HouAstro/master/result_v2/NGC3628/phase2_raw_step3/best_mask_hrl.npy",
        z=0.002772,
        W_kms=345.0,
        line="H30a",
    ),
]


def main():
    results = []
    for cfg in GALAXIES:
        if not Path(cfg.hrl_path).exists():
            print(f"SKIP {cfg.name}: cube missing")
            continue
        if not Path(cfg.seed_mask_path).exists():
            print(f"SKIP {cfg.name}: seed mask missing")
            continue
        results.append(run_galaxy(cfg))

    # combined comparison figure: grid layout
    n = len(results)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for ax, r in zip(axes_flat, results):
        ax.plot(r["Nbeam"], r["F_obs"], "o-", color="C3",
                label="F_obs", markersize=4)
        ax.plot(r["Nbeam"], r["F_residual"], "s-", color="C2",
                label="F_residual", markersize=4)
        ax.fill_between(r["Nbeam"],
                        r["F_residual"] - r["sigma_F_res"],
                        r["F_residual"] + r["sigma_F_res"],
                        color="C2", alpha=0.20,
                        label="F_res ±1σ")
        ax.plot(r["Nbeam"], r["F_clean"], "^-", color="C0",
                label="F_clean", markersize=4)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Nbeam")
        ax.set_ylabel("Flux (Jy·km/s)")
        fr_min, fr_max = float(np.min(r["F_residual"])), float(np.max(r["F_residual"]))
        fo_amp = float(np.max(r["F_obs"]) - np.min(r["F_obs"]))
        ratio = abs(fr_max - fr_min) / max(fo_amp, 1e-6)
        nsig = r["F_res_nsigma_max"]
        verdict = ("REAL baseline" if nsig > 3 else
                   "baseline consistent w/ 0")
        ax.set_title(
            f"{r['name']}  |F_res|/σ_max = {nsig:.1f}  → {verdict}\n"
            f"F_res ∈ [{fr_min:+.2f}, {fr_max:+.2f}] Jy·km/s, "
            f"|ΔF_res|/|ΔF_obs|={ratio:.2f}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes_flat[len(results):]:
        ax.axis("off")
    plt.tight_layout()
    summary_path = OUT_ROOT / "_poc_baseline_residual_full_sample.png"
    plt.savefig(summary_path, dpi=140)
    plt.close()
    print(f"\nsaved summary {summary_path}")

    # also dump a per-galaxy summary CSV
    csv_path = OUT_ROOT / "_poc_baseline_residual_summary.csv"
    with open(csv_path, "w") as f:
        f.write("galaxy,n_free,eps_mean,eps_std,"
                "F_obs_min,F_obs_max,F_res_min,F_res_max,"
                "F_clean_min,F_clean_max,seed_Nbeam,max_Nbeam,"
                "abs_dFres_over_dFobs,max_F_res_nsigma,verdict\n")
        for r in results:
            fr_min, fr_max = float(np.min(r["F_residual"])), float(np.max(r["F_residual"]))
            fo_min, fo_max = float(np.min(r["F_obs"])), float(np.max(r["F_obs"]))
            fc_min, fc_max = float(np.min(r["F_clean"])), float(np.max(r["F_clean"]))
            fo_amp = fo_max - fo_min
            ratio = abs(fr_max - fr_min) / max(fo_amp, 1e-6)
            nsig = r["F_res_nsigma_max"]
            verdict = "REAL_baseline" if nsig > 3 else "consistent_with_0"
            f.write(f"{r['name']},{r['n_free']},{r['eps_mean']:+.4e},"
                    f"{r['eps_std']:.4e},{fo_min:+.4f},{fo_max:+.4f},"
                    f"{fr_min:+.4f},{fr_max:+.4f},"
                    f"{fc_min:+.4f},{fc_max:+.4f},"
                    f"{r['Nbeam'][0]:.2f},{r['Nbeam'][-1]:.2f},"
                    f"{ratio:.3f},{nsig:.2f},{verdict}\n")
    print(f"saved CSV {csv_path}")


if __name__ == "__main__":
    main()
