"""Multi-galaxy runner for baseline_diag. Builds master CSV + overview PNG."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from baseline_diag import BaselineDiagResult


@dataclass
class GalaxyJob:
    name: str
    cube_path: str
    seed_path: str
    z: float
    line: str
    W_kms: float
    out_dir: Path


def write_overview_and_master_csv(
    results: List[BaselineDiagResult],
    master_csv_path: Path,
    overview_png_path: Path,
) -> None:
    # master CSV
    with open(master_csv_path, "w") as f:
        f.write(
            "galaxy,n_line_free_channels,eps_mean,eps_std,"
            "F_obs_min,F_obs_max,F_res_min,F_res_max,"
            "F_clean_min,F_clean_max,seed_Nbeam,max_Nbeam,"
            "max_F_res_nsigma,flag\n"
        )
        for r in results:
            f.write(
                f"{r.name},{r.n_line_free_channels},"
                f"{r.eps_mean:+.4e},{r.eps_std:.4e},"
                f"{r.F_obs.min():+.4f},{r.F_obs.max():+.4f},"
                f"{r.F_residual.min():+.4f},{r.F_residual.max():+.4f},"
                f"{r.F_clean.min():+.4f},{r.F_clean.max():+.4f},"
                f"{r.Nbeam[0]:.2f},{r.Nbeam[-1]:.2f},"
                f"{r.F_res_nsigma_max:.2f},{r.flag}\n"
            )

    # overview figure
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    for ax, r in zip(axes.flatten(), results):
        ax.plot(r.Nbeam, r.F_obs, "o-", color="C3",
                label="F_obs", markersize=4)
        ax.plot(r.Nbeam, r.F_residual, "s-", color="C2",
                label="F_res", markersize=4)
        ax.fill_between(r.Nbeam,
                        r.F_residual - r.sigma_F_residual,
                        r.F_residual + r.sigma_F_residual,
                        color="C2", alpha=0.20)
        ax.plot(r.Nbeam, r.F_clean, "^-", color="C0",
                label="F_clean", markersize=4)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Nbeam")
        ax.set_ylabel("Flux (Jy·km/s)")
        ax.set_title(
            f"{r.name}  [{r.flag}]  max|F_res|/σ={r.F_res_nsigma_max:.2f}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(overview_png_path, dpi=140)
    plt.close(fig)


def run_jobs(jobs: List[GalaxyJob]) -> List[BaselineDiagResult]:
    """Run a list of jobs serially; returns results list."""
    from baseline_diag import _read_cube_for_cli, _channel_velocities, \
        _pix_per_beam, run_galaxy_baseline_diag
    from known_lines import LINES_REST_HZ
    results = []
    for job in jobs:
        rest_obs_hz = LINES_REST_HZ[job.line] / (1.0 + job.z)
        bundle = _read_cube_for_cli(job.cube_path)
        cube = bundle.data
        hdr = bundle.header
        v_chan = _channel_velocities(hdr, rest_obs_hz)
        pix_per_beam = _pix_per_beam(hdr)
        nchan = hdr["NAXIS3"]
        cube_freq_lo = hdr["CRVAL3"] + (1 - hdr["CRPIX3"]) * hdr["CDELT3"]
        cube_freq_hi = hdr["CRVAL3"] + (nchan - hdr["CRPIX3"]) * hdr["CDELT3"]
        cube_band = (min(cube_freq_lo, cube_freq_hi),
                     max(cube_freq_lo, cube_freq_hi))
        seed = np.load(job.seed_path).astype(bool)
        r = run_galaxy_baseline_diag(
            galaxy_name=job.name,
            cube=cube,
            v_chan_kms=v_chan,
            primary_line_rest_hz=rest_obs_hz,
            cube_freq_range_hz=cube_band,
            z=job.z,
            primary_line_key=job.line,
            W_kms=job.W_kms,
            pix_per_beam=pix_per_beam,
            seed_mask=seed,
            out_dir=job.out_dir,
        )
        results.append(r)
        # release cube reference
        bundle.data = None  # type: ignore
    return results
