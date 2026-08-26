"""Tests for baseline_diag.py (per-pixel baseline-residual diagnostic)."""
from pathlib import Path
import sys

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import baseline_diag as bd


def test_module_imports():
    """Sanity: module imports and exposes the public functions named in spec §7."""
    for name in [
        "compute_eps_map",
        "dilation_sequence",
        "predict_F_residual",
        "verdict_flag",
        "run_galaxy_baseline_diag",
        "BaselineDiagResult",
    ]:
        assert hasattr(bd, name), f"baseline_diag missing public name {name}"


def test_compute_eps_map_all_zero_cube():
    """Cube of all zeros → ε map of all zeros."""
    cube = np.zeros((10, 4, 4), dtype=np.float32)
    line_free = np.array([True, True, True, True, False,
                          False, True, True, True, True])
    eps = bd.compute_eps_map(cube, line_free)
    assert eps.shape == (4, 4)
    assert np.allclose(eps, 0.0)


def test_compute_eps_map_constant_offset():
    """Cube with constant +0.5 Jy/beam in line-free channels
    → ε map = +0.5 everywhere."""
    cube = np.zeros((10, 4, 4), dtype=np.float32)
    line_free = np.array([True, True, True, True, False,
                          False, True, True, True, True])
    cube[line_free, :, :] = 0.5
    cube[~line_free, :, :] = 999.0  # line zone, must be ignored
    eps = bd.compute_eps_map(cube, line_free)
    assert np.allclose(eps, 0.5)


def test_compute_eps_map_nan_safe():
    """NaN channels in cube → nanmean used, doesn't propagate."""
    cube = np.zeros((10, 4, 4), dtype=np.float32)
    line_free = np.array([True] * 10)
    cube[3, :, :] = np.nan
    cube[~np.array([False, False, False, True, False, False, False, False, False, False]), :, :] = 0.2
    eps = bd.compute_eps_map(cube, line_free)
    assert np.all(np.isfinite(eps))
    assert np.allclose(eps, 0.2)


def test_dilation_sequence_grows_monotonically():
    """Each step has >= pixels of the previous one."""
    seed = np.zeros((20, 20), dtype=bool)
    seed[10, 10] = True   # single pixel seed
    seq = bd.dilation_sequence(seed, n_steps=5)
    assert len(seq) == 6, "expected n_steps + 1 masks (seed included)"
    counts = [int(m.sum()) for m in seq]
    assert counts == sorted(counts), \
        f"Npix should be non-decreasing, got {counts}"
    assert counts[0] == 1
    # 5-step dilation of single pixel → cross-shape with at most ~61 pix
    assert counts[-1] <= 71


def test_dilation_sequence_preserves_seed_dtype_and_shape():
    seed = np.zeros((15, 30), dtype=bool)
    seed[7, 15] = True
    seq = bd.dilation_sequence(seed, n_steps=3)
    for m in seq:
        assert m.dtype == bool
        assert m.shape == seed.shape


def test_predict_F_residual_zero_eps_zero_residual():
    """If ε map is all zero, F_residual is zero everywhere."""
    eps = np.zeros((20, 20))
    masks = [np.zeros((20, 20), dtype=bool)]
    masks[0][10, 10] = True
    F_res, sigma_F_res = bd.predict_F_residual(
        eps_map=eps,
        masks=masks,
        W_kms=100.0,
        pix_per_beam=10.0,
    )
    assert np.allclose(F_res, 0.0)
    # sigma is also 0 because std(eps) = 0
    assert np.allclose(sigma_F_res, 0.0)


def test_predict_F_residual_constant_offset():
    """If ε is uniformly +1e-3 Jy/beam on a mask of 100 pixels,
    W=100, pix_per_beam=10 → F_residual = 100 * 1e-3 * 100 / 10 = 1.0 Jy·km/s."""
    eps = np.full((20, 20), 1e-3)
    mask = np.zeros((20, 20), dtype=bool)
    # set first 100 pixels
    mask.flat[:100] = True
    F_res, _ = bd.predict_F_residual(
        eps_map=eps,
        masks=[mask],
        W_kms=100.0,
        pix_per_beam=10.0,
    )
    assert np.isclose(F_res[0], 1.0)


def test_predict_F_residual_noise_uses_sqrt_Nbeam_not_sqrt_Npix():
    """REGRESSION GUARD against the PoC bug.

    σ_F_res(N) = sqrt(N_beam) * σ_eps * W,
    not sqrt(N_pix) * σ_eps * W / pix_per_beam.

    For pix_per_beam=10, N_pix=100, σ_eps=0.01, W=100:
      correct:   sqrt(100/10) * 0.01 * 100 = sqrt(10) * 1 ≈ 3.162
      wrong:     sqrt(100) * 0.01 * 100 / 10 = 10 * 0.1 = 1.0
    """
    rng = np.random.default_rng(seed=42)
    eps = rng.normal(scale=0.01, size=(50, 50))   # white noise σ=0.01
    mask = np.zeros((50, 50), dtype=bool)
    mask.flat[:100] = True
    _, sigma_F_res = bd.predict_F_residual(
        eps_map=eps,
        masks=[mask],
        W_kms=100.0,
        pix_per_beam=10.0,
    )
    # measured std(eps) ≈ 0.01 (will be very close but use the measured value
    # exactly to avoid sampling noise)
    sigma_eps = float(np.nanstd(eps[np.isfinite(eps)]))
    expected_correct = np.sqrt(100 / 10) * sigma_eps * 100.0
    expected_wrong   = np.sqrt(100) * sigma_eps * 100.0 / 10.0
    assert np.isclose(sigma_F_res[0], expected_correct, rtol=1e-6), \
        f"sigma_F_res should use sqrt(N_beam), got {sigma_F_res[0]}"
    # Sanity: prove the wrong formula would give a different number
    assert not np.isclose(expected_correct, expected_wrong)


def test_predict_F_residual_along_growing_mask():
    """Constant offset eps, dilation sequence → F_residual grows linearly
    with N_beam (= N_pix / pix_per_beam)."""
    eps = np.full((30, 30), 2e-4)
    seed = np.zeros((30, 30), dtype=bool)
    seed[15, 15] = True
    masks = bd.dilation_sequence(seed, n_steps=4)
    F_res, _ = bd.predict_F_residual(
        eps_map=eps,
        masks=masks,
        W_kms=200.0,
        pix_per_beam=5.0,
    )
    Npix = np.array([int(m.sum()) for m in masks])
    expected = Npix * 2e-4 * 200.0 / 5.0
    assert np.allclose(F_res, expected)


def test_verdict_flag_thresholds():
    """PASS < 2σ, SUSPECT in [2, 3), REAL ≥ 3σ (spec §13 decision 1)."""
    # nsigma values, expected flags
    cases = [
        (0.5, "PASS"),
        (1.99, "PASS"),
        (2.0, "SUSPECT"),
        (2.5, "SUSPECT"),
        (2.99, "SUSPECT"),
        (3.0, "REAL"),
        (5.7, "REAL"),
    ]
    for nsigma, expected in cases:
        F_res = np.array([nsigma * 0.1])
        sigma = np.array([0.1])
        assert bd.verdict_flag(F_res, sigma) == expected, \
            f"nsigma={nsigma} should give {expected}"


def test_verdict_flag_takes_max_over_N():
    """Flag is set by the worst |F_res|/σ across all N, not the last value."""
    F_res = np.array([0.10, 0.50, 0.05])  # nsigma = 1, 5, 0.5
    sigma = np.array([0.10, 0.10, 0.10])
    assert bd.verdict_flag(F_res, sigma) == "REAL"


def test_run_galaxy_baseline_diag_returns_result_object(tmp_path):
    """Smoke test: function runs end-to-end on a synthetic minimal cube
    and returns a BaselineDiagResult."""
    # Build a synthetic cube + seed mask
    nchan, ny, nx = 80, 40, 40
    rng = np.random.default_rng(seed=7)
    cube = rng.normal(scale=1e-3, size=(nchan, ny, nx)).astype(np.float32)
    # Inject a faint line near center channels
    line_center = nchan // 2
    yy, xx = np.indices((ny, nx))
    src = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 9).astype(np.float32) * 0.05
    for c in range(line_center - 5, line_center + 5):
        cube[c] += src

    seed = ((yy - 20) ** 2 + (xx - 20) ** 2 <= 9)
    out_dir = tmp_path / "synthetic_galaxy"
    out_dir.mkdir()

    result = bd.run_galaxy_baseline_diag(
        galaxy_name="SYNTH",
        cube=cube,
        v_chan_kms=np.linspace(-400, +400, nchan),
        primary_line_rest_hz=231.9e9,
        cube_freq_range_hz=(230.5e9, 232.5e9),
        z=0.0,
        primary_line_key="H30a",
        W_kms=100.0,
        pix_per_beam=5.0,
        seed_mask=seed,
        out_dir=out_dir,
        n_dilation_steps=10,
    )

    assert isinstance(result, bd.BaselineDiagResult)
    assert result.name == "SYNTH"
    assert result.Nbeam.shape == (11,)
    assert result.F_obs.shape == (11,)
    assert result.F_residual.shape == (11,)
    assert result.flag in ("PASS", "SUSPECT", "REAL")
    # Files written
    assert (out_dir / "eps_map.fits").exists()
    assert (out_dir / "F_curves.csv").exists()
    assert (out_dir / "summary.md").exists()


def test_plot_F_curves_writes_png(tmp_path):
    result = bd.BaselineDiagResult(
        name="TEST",
        Nbeam=np.array([1.0, 2.0, 4.0, 8.0]),
        F_obs=np.array([0.1, 0.15, 0.2, 0.25]),
        F_residual=np.array([0.01, 0.02, 0.03, 0.04]),
        sigma_F_residual=np.array([0.05, 0.07, 0.10, 0.14]),
        F_clean=np.array([0.09, 0.13, 0.17, 0.21]),
        F_res_nsigma_max=0.4,
        eps_mean=0.0,
        eps_std=1e-3,
        n_line_free_channels=200,
        flag="PASS",
    )
    out = tmp_path / "F_curves.png"
    bd.plot_F_curves(result, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_eps_map_writes_png(tmp_path):
    eps = np.random.default_rng(0).normal(0, 1e-3, size=(40, 40))
    seed = np.zeros((40, 40), dtype=bool)
    seed[18:22, 18:22] = True
    out = tmp_path / "eps_map.png"
    bd.plot_eps_map(eps, seed_mask=seed, n_free=200,
                    galaxy_name="TEST", out_path=out)
    assert out.exists()


def test_runner_writes_master_csv_and_overview(tmp_path):
    """Runner that loops over multiple BaselineDiagResults writes a
    master CSV and a multi-panel overview PNG."""
    import sys
    PKG_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PKG_DIR))
    from baseline_diag_runner import write_overview_and_master_csv

    results = [
        bd.BaselineDiagResult(
            name=name,
            Nbeam=np.array([1.0, 2.0, 4.0]),
            F_obs=np.array([0.1, 0.15, 0.2]),
            F_residual=np.array([0.01, 0.02, 0.03]),
            sigma_F_residual=np.array([0.05, 0.07, 0.10]),
            F_clean=np.array([0.09, 0.13, 0.17]),
            F_res_nsigma_max=0.3,
            eps_mean=0.0,
            eps_std=1e-3,
            n_line_free_channels=200,
            flag="PASS",
        )
        for name in ("A", "B", "C")
    ]
    out_csv = tmp_path / "master.csv"
    out_png = tmp_path / "overview.png"
    write_overview_and_master_csv(results, out_csv, out_png)
    assert out_csv.exists()
    assert out_png.exists()

    import csv
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert {r["galaxy"] for r in rows} == {"A", "B", "C"}
    assert all(r["flag"] == "PASS" for r in rows)


@pytest.mark.integration
def test_cli_runs_for_ngc5253(tmp_path):
    """End-to-end on real NGC 5253 data; verify regression against PoC numbers."""
    import subprocess
    cube_path = ("/Volumes/HouAstro/master/master_thesis/work_dir/"
                 "NGC5253/NGC5253_H30a_pbcor.fits")
    seed_path = ("/Volumes/HouAstro/master/result_v2/NGC5253/"
                 "phase2_raw_step3/best_mask_hrl.npy")
    if not Path(cube_path).exists() or not Path(seed_path).exists():
        pytest.skip("NGC 5253 cube or mask not present")
    out = tmp_path / "NGC5253"
    cmd = [
        "python",
        "/Volumes/HouAstro/master/master_thesis/my_code/_step3/baseline_diag.py",
        "--galaxy", "NGC5253",
        "--cube", cube_path,
        "--seed", seed_path,
        "--z", "0.0014",
        "--line", "H30a",
        "--W-kms", "135",
        "--out-dir", str(out),
        "--n-dilation", "41",
    ]
    # Direct script execution: Python auto-adds _step3/ to sys.path so
    # baseline_diag.py's bare imports of sibling modules (known_lines,
    # cube_io) resolve. Matches the convention used by existing
    # per-galaxy step3_analyze.py scripts.
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
    )
    assert (out / "F_curves.csv").exists()
    # Regression against PoC numbers (spec §6)
    import csv
    with open(out / "F_curves.csv") as fh:
        rows = list(csv.DictReader(fh))
    F_obs_min = min(float(r["F_obs"]) for r in rows)
    F_obs_max = max(float(r["F_obs"]) for r in rows)
    # Allow 1% slack for floating-point + minor implementation differences
    assert 0.095 <= F_obs_min <= 0.105, f"F_obs_min off from PoC 0.099"
    assert 0.720 <= F_obs_max <= 0.736, f"F_obs_max off from PoC 0.7282"
