"""Per-pixel baseline-residual diagnostic along the dilation-CoG mask sequence.

Implements spec docs/superpowers/specs/2026-05-16-baseline-residual-cog-subtraction-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from astropy.io import fits
from scipy import ndimage

from known_lines import line_free_mask_for_cube


__all__ = [
    "compute_eps_map",
    "dilation_sequence",
    "predict_F_residual",
    "verdict_flag",
    "run_galaxy_baseline_diag",
    "plot_F_curves",
    "plot_eps_map",
    "BaselineDiagResult",
]


@dataclass
class BaselineDiagResult:
    """One galaxy's baseline-residual diagnostic result."""
    name: str
    Nbeam: np.ndarray            # (n_steps + 1,)
    F_obs: np.ndarray            # (n_steps + 1,) Jy·km/s
    F_residual: np.ndarray       # (n_steps + 1,) Jy·km/s
    sigma_F_residual: np.ndarray # (n_steps + 1,) Jy·km/s
    F_clean: np.ndarray          # (n_steps + 1,) Jy·km/s
    F_res_nsigma_max: float
    eps_mean: float              # Jy/beam (global mean of ε map)
    eps_std: float               # Jy/beam (per-beam noise level)
    n_line_free_channels: int
    flag: str                    # "PASS" | "SUSPECT" | "REAL"


def compute_eps_map(
    cube: np.ndarray,
    line_free_channels: np.ndarray,
) -> np.ndarray:
    """Per-pixel mean over the line-free channels of a cube.

    Parameters
    ----------
    cube : (nchan, ny, nx) array
        Pbcor HRL cube in Jy/beam.
    line_free_channels : (nchan,) bool array
        True for channels included in the baseline estimate
        (line zone + KNOWN_LINES windows already excluded).

    Returns
    -------
    eps : (ny, nx) array
        Per-pixel ε(x, y) in Jy/beam.
    """
    assert cube.ndim == 3
    assert line_free_channels.shape == (cube.shape[0],)
    return np.nanmean(cube[line_free_channels, :, :], axis=0)


def dilation_sequence(
    seed: np.ndarray,
    n_steps: int = 41,
) -> List[np.ndarray]:
    """Return [seed, dilate(seed, 1), dilate(seed, 2), ..., dilate(seed, n_steps)].

    Each subsequent mask is the previous one expanded by one pixel ring
    via `scipy.ndimage.binary_dilation(iterations=1)`.
    """
    assert seed.dtype == bool
    masks = [seed.copy()]
    cur = seed.copy()
    for _ in range(n_steps):
        cur = ndimage.binary_dilation(cur, iterations=1)
        masks.append(cur.copy())
    return masks


def predict_F_residual(
    eps_map: np.ndarray,
    masks: Sequence[np.ndarray],
    W_kms: float,
    pix_per_beam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict F_residual(N) and its noise σ_F_residual(N) along a mask sequence.

    Parameters
    ----------
    eps_map : (ny, nx) array
        Per-pixel ε map in Jy/beam (output of compute_eps_map).
    masks : sequence of (ny, nx) bool arrays
        Mask at each CoG step.
    W_kms : float
        Velocity integration window in km/s (scan-best line width).
    pix_per_beam : float
        Number of spatial pixels in one beam area.

    Returns
    -------
    F_residual : (n_masks,) array, Jy·km/s
        ε-predicted residual contribution at each mask size.
    sigma_F_residual : (n_masks,) array, Jy·km/s
        Beam-correlated noise floor: sqrt(N_beam) * σ_ε * W.

    Notes
    -----
    σ_ε is the std of ε across finite pixels, which is the **per-beam**
    noise level (not per-pixel) because pixels within one beam carry
    correlated convolved values. Using sqrt(N_pix) instead of
    sqrt(N_beam) underestimates the noise by sqrt(pix_per_beam) — see
    spec §4.5 for full derivation.
    """
    eps_finite = eps_map[np.isfinite(eps_map)]
    sigma_eps_per_beam = float(np.nanstd(eps_finite)) if eps_finite.size else 0.0
    eps_x_W = eps_map * W_kms

    Npix_arr = np.array([int(m.sum()) for m in masks])
    Nbeam_arr = Npix_arr / pix_per_beam

    F_residual = np.array([
        np.nansum(eps_x_W[m]) / pix_per_beam for m in masks
    ])
    sigma_F_residual = sigma_eps_per_beam * np.sqrt(Nbeam_arr) * W_kms
    return F_residual, sigma_F_residual


def verdict_flag(
    F_residual: np.ndarray,
    sigma_F_residual: np.ndarray,
) -> str:
    """Return one of "PASS" | "SUSPECT" | "REAL".

    PASS if max|F_res|/σ < 2.0
    SUSPECT if 2.0 ≤ max|F_res|/σ < 3.0
    REAL if max|F_res|/σ ≥ 3.0
    """
    nsigma = np.divide(F_residual, sigma_F_residual,
                       out=np.zeros_like(F_residual),
                       where=sigma_F_residual > 0)
    max_nsigma = float(np.max(np.abs(nsigma)))
    if max_nsigma < 2.0:
        return "PASS"
    if max_nsigma < 3.0:
        return "SUSPECT"
    return "REAL"


def run_galaxy_baseline_diag(
    *,
    galaxy_name: str,
    cube: np.ndarray,
    v_chan_kms: np.ndarray,
    primary_line_rest_hz: float,
    cube_freq_range_hz: Tuple[float, float],
    z: float,
    primary_line_key: str,
    W_kms: float,
    pix_per_beam: float,
    seed_mask: np.ndarray,
    out_dir: Path,
    n_dilation_steps: int = 41,
    eps_map_header: fits.Header | None = None,
) -> BaselineDiagResult:
    """Full per-galaxy diagnostic. Writes eps_map.fits, F_curves.csv,
    summary.md into out_dir. Returns the BaselineDiagResult."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. channel selection
    dv = float(np.abs(np.median(np.diff(v_chan_kms))))
    in_line_zone = np.abs(v_chan_kms) <= W_kms / 2.0
    outside_line_buffer = np.abs(v_chan_kms) > W_kms / 2.0 + dv
    keep_known = line_free_mask_for_cube(
        v_chan=v_chan_kms,
        primary_line_rest_hz=primary_line_rest_hz,
        cube_freq_range_hz=cube_freq_range_hz,
        z=z,
        primary_line_key=primary_line_key,
        fwhm_kms=W_kms,  # reuse scan-best width as exclusion fwhm
    )
    line_free = outside_line_buffer & keep_known
    n_free = int(line_free.sum())

    # 2. eps map
    eps = compute_eps_map(cube, line_free)

    # 3. mask sequence
    masks = dilation_sequence(seed_mask, n_steps=n_dilation_steps)

    # 4. F_obs along the same sequence
    line_int_per_pix = np.nansum(cube[in_line_zone, :, :], axis=0) * dv
    F_obs = np.array([
        np.nansum(line_int_per_pix[m]) / pix_per_beam for m in masks
    ])

    # 5. residual + noise model
    F_res, sigma_F_res = predict_F_residual(
        eps_map=eps,
        masks=masks,
        W_kms=W_kms,
        pix_per_beam=pix_per_beam,
    )
    F_clean = F_obs - F_res

    Npix_arr = np.array([int(m.sum()) for m in masks])
    Nbeam_arr = Npix_arr / pix_per_beam

    nsigma = np.divide(F_res, sigma_F_res,
                       out=np.zeros_like(F_res),
                       where=sigma_F_res > 0)
    flag = verdict_flag(F_res, sigma_F_res)

    eps_finite = eps[np.isfinite(eps)]
    result = BaselineDiagResult(
        name=galaxy_name,
        Nbeam=Nbeam_arr,
        F_obs=F_obs,
        F_residual=F_res,
        sigma_F_residual=sigma_F_res,
        F_clean=F_clean,
        F_res_nsigma_max=float(np.max(np.abs(nsigma))),
        eps_mean=float(np.nanmean(eps_finite)),
        eps_std=float(np.nanstd(eps_finite)),
        n_line_free_channels=n_free,
        flag=flag,
    )

    # 6. write outputs
    _write_eps_fits(eps, out_dir / "eps_map.fits", eps_map_header)
    _write_F_curves_csv(result, out_dir / "F_curves.csv")
    _write_summary_md(result, W_kms, pix_per_beam,
                      n_dilation_steps, out_dir / "summary.md")
    plot_F_curves(result, out_dir / "F_curves.png")
    plot_eps_map(eps, seed_mask=seed_mask, n_free=n_free,
                 galaxy_name=galaxy_name,
                 out_path=out_dir / "eps_map.png")
    return result


def _write_eps_fits(eps: np.ndarray, path: Path, header):
    hdu = fits.PrimaryHDU(data=eps.astype(np.float32),
                          header=header)
    hdu.writeto(path, overwrite=True)


def _write_F_curves_csv(result: BaselineDiagResult, path: Path):
    with open(path, "w") as f:
        f.write("step,Nbeam,F_obs,F_residual,sigma_F_residual,F_clean\n")
        for i, n in enumerate(result.Nbeam):
            f.write(
                f"{i},{n:.4f},{result.F_obs[i]:+.6f},"
                f"{result.F_residual[i]:+.6f},"
                f"{result.sigma_F_residual[i]:.6f},"
                f"{result.F_clean[i]:+.6f}\n"
            )


def _write_summary_md(result, W_kms, pix_per_beam, n_dilation, path: Path):
    with open(path, "w") as f:
        f.write(f"# {result.name} — baseline-residual diagnostic\n\n")
        f.write(f"**Flag:** {result.flag}\n\n")
        f.write(f"**Max |F_res| / σ:** {result.F_res_nsigma_max:.2f}\n\n")
        f.write("## Inputs\n\n")
        f.write(f"- W = {W_kms:.0f} km/s\n")
        f.write(f"- pix per beam = {pix_per_beam:.2f}\n")
        f.write(f"- dilation steps = {n_dilation}\n")
        f.write(f"- line-free channels = {result.n_line_free_channels}\n\n")
        f.write("## ε map stats\n\n")
        f.write(f"- mean = {result.eps_mean:+.4e} Jy/beam\n")
        f.write(f"- per-beam σ = {result.eps_std:.4e} Jy/beam\n\n")
        f.write("## F curves\n\n")
        f.write(f"- F_obs range: [{result.F_obs.min():+.4f}, "
                f"{result.F_obs.max():+.4f}] Jy·km/s\n")
        f.write(f"- F_residual range: [{result.F_residual.min():+.4f}, "
                f"{result.F_residual.max():+.4f}] Jy·km/s\n")
        f.write(f"- F_clean range: [{result.F_clean.min():+.4f}, "
                f"{result.F_clean.max():+.4f}] Jy·km/s\n")


def plot_F_curves(result: BaselineDiagResult, out_path: Path) -> None:
    """Three-curve plot with ±1σ band on F_residual."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(result.Nbeam, result.F_obs, "o-", color="C3",
            label="F_obs (dilation CoG)", markersize=4)
    ax.plot(result.Nbeam, result.F_residual, "s-", color="C2",
            label=r"F_residual = ($\Sigma\varepsilon$) $\times$ W",
            markersize=4)
    ax.fill_between(
        result.Nbeam,
        result.F_residual - result.sigma_F_residual,
        result.F_residual + result.sigma_F_residual,
        color="C2", alpha=0.20,
        label="F_res ±1σ (beam-corrected noise floor)",
    )
    ax.plot(result.Nbeam, result.F_clean, "^-", color="C0",
            label="F_clean = F_obs − F_residual", markersize=4)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Mask size — Nbeam")
    ax.set_ylabel("Flux (Jy·km/s)")
    ax.set_title(
        f"{result.name} — baseline-residual diagnostic [{result.flag}, "
        f"max |F_res|/σ = {result.F_res_nsigma_max:.2f}]"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_eps_map(
    eps_map: np.ndarray,
    *,
    seed_mask: np.ndarray,
    n_free: int,
    galaxy_name: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import cmasher  # noqa: F401  (registers cmr.* — kept for consistency
                    # with rest of project; we still use RdBu_r below)
    eps_finite = eps_map[np.isfinite(eps_map)]
    vlim = 3 * float(np.nanstd(eps_finite)) if eps_finite.size else 1e-3
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(eps_map, origin="lower", cmap="RdBu_r",
                   vmin=-vlim, vmax=+vlim)
    ax.contour(seed_mask, levels=[0.5], colors="black", linewidths=1.0)
    plt.colorbar(im, ax=ax, label="ε (Jy/beam)")
    ax.set_title(
        f"{galaxy_name} — ε(x,y), per-pixel mean over "
        f"{n_free} line-free channels"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _ensure_step3_on_syspath():
    """Add _step3/ to sys.path so bare imports of sibling modules
    (known_lines, cube_io) resolve regardless of caller's cwd."""
    import sys
    pkg_dir = str(Path(__file__).parent)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)


def _read_cube_for_cli(cube_path: str):
    """Lightweight wrapper around cube_io.load_cube for the CLI."""
    _ensure_step3_on_syspath()
    from cube_io import load_cube
    return load_cube(cube_path)


def _channel_velocities(header, rest_obs_hz: float) -> np.ndarray:
    n = header["NAXIS3"]
    cdelt = header["CDELT3"]
    crval = header["CRVAL3"]
    crpix = header["CRPIX3"]
    freqs = crval + (np.arange(n) - (crpix - 1)) * cdelt
    return (rest_obs_hz - freqs) / rest_obs_hz * 299792.458


def _pix_per_beam(header) -> float:
    bmaj = header["BMAJ"]; bmin = header["BMIN"]
    pix = abs(header["CDELT1"])
    return (np.pi / (4.0 * np.log(2.0))) * (bmaj / pix) * (bmin / pix)


def _cli_main():
    import argparse
    p = argparse.ArgumentParser(prog="baseline_diag",
                                description="Per-galaxy baseline-residual diagnostic")
    p.add_argument("--galaxy", required=True)
    p.add_argument("--cube", required=True,
                   help="HRL pbcor FITS path")
    p.add_argument("--seed", required=True,
                   help="seed mask .npy path (Phase 2 raw best mask)")
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--line", required=True,
                   help="primary line key, e.g. H30a / H40a")
    p.add_argument("--W-kms", type=float, required=True,
                   dest="W_kms")
    p.add_argument("--out-dir", required=True, dest="out_dir")
    p.add_argument("--n-dilation", type=int, default=41,
                   dest="n_dilation")
    args = p.parse_args()

    from known_lines import LINES_REST_HZ
    rest_hz = LINES_REST_HZ[args.line]
    rest_obs_hz = rest_hz / (1.0 + args.z)

    bundle = _read_cube_for_cli(args.cube)
    cube = bundle.data
    hdr = bundle.header
    v_chan = _channel_velocities(hdr, rest_obs_hz)
    pix_per_beam = _pix_per_beam(hdr)
    nchan = hdr["NAXIS3"]
    cube_freq_lo = hdr["CRVAL3"] + (1 - hdr["CRPIX3"]) * hdr["CDELT3"]
    cube_freq_hi = hdr["CRVAL3"] + (nchan - hdr["CRPIX3"]) * hdr["CDELT3"]
    cube_band = (min(cube_freq_lo, cube_freq_hi),
                 max(cube_freq_lo, cube_freq_hi))

    seed = np.load(args.seed).astype(bool)
    result = run_galaxy_baseline_diag(
        galaxy_name=args.galaxy,
        cube=cube,
        v_chan_kms=v_chan,
        primary_line_rest_hz=rest_obs_hz,
        cube_freq_range_hz=cube_band,
        z=args.z,
        primary_line_key=args.line,
        W_kms=args.W_kms,
        pix_per_beam=pix_per_beam,
        seed_mask=seed,
        out_dir=Path(args.out_dir),
        n_dilation_steps=args.n_dilation,
    )
    print(f"{args.galaxy}: flag={result.flag} "
          f"max|F_res|/σ={result.F_res_nsigma_max:.2f}")


if __name__ == "__main__":
    _cli_main()
