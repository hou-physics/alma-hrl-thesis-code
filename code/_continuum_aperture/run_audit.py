"""Run continuum-aperture audit across all existing scan-optimized analyses.

Discovers each `master_thesis/my_code/{galaxy}_analyse_code/step3_analyze*.py`,
imports the AnalysisConfig, runs the continuum-aperture methodology on the HRL
cube, writes per-galaxy results to `results/_continuum_aperture/{KEY}/`, and
aggregates into `results/_continuum_aperture/audit_summary.md`.

Output is COMPLETELY SEPARATE from the existing scan-optimized results in
`results/{KEY}/`. Cross-comparison happens only in audit_summary.md.

Run: conda run -n casa_env --no-capture-output python run_audit.py
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

# Local imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "_step3"))

from continuum_mask import build_aperture_mask_auto
from aperture_spectrum import extract_aperture_spectrum, measure_line


HRL_REST_GHZ = {
    "H21a": 662.404155, "H22a": 577.896313, "H23a": 507.175528, "H24a": 447.540286,
    "H25a": 396.900713, "H26a": 353.622698, "H27a": 316.415440, "H28a": 285.064064,
    "H29a": 256.302053, "H30a": 231.900928, "H31a": 211.001094, "H32a": 192.585129,
    "H33a": 176.296225, "H34a": 161.829060, "H39a": 106.737358, "H40a": 99.022952,
    "H41a": 92.034434, "H42a": 85.688390,
}

STRONG_REST_GHZ = {
    "CO10": 115.2712018, "CO21": 230.5380000, "CO32": 345.7959899,
    "13CO21": 220.3986842, "C18O21": 219.5603541,
    "HCN10": 88.6316022, "HCN32": 265.8861800, "HCN43": 354.5054779,
    "HCO+10": 89.1885230, "HCO+32": 267.5576260, "HCO+43": 356.7342230,
}

C_KMS = 299792.458

# Best-W scan: extended to 500 km/s (was 300) so edge-on disks (NGC 3628,
# NGC 4945) with rotation envelopes ≥400 km/s are not under-integrated.
# Toma standard W=200 km/s remains the headline column.
BEST_W_SCAN_KMS = (100, 150, 200, 250, 300, 400, 500)


@dataclass
class AuditRow:
    key: str            # results/{key}/ identifier (e.g. NGC253, NGC3627_adaptive)
    galaxy: str
    weak_line: str
    z: float
    cube_path: str
    aperture_radius_arcsec: float
    npix: int
    n_eff_beam: float
    sigma_chan_jy: float
    flux_jy_kms_W200: float
    sigma_int_W200: float
    sn_W200: float
    flux_jy_kms_best: float
    sn_best: float
    width_best: float
    mode_used: str
    error: str | None = None


def import_config_module(script_path: Path):
    """Import a step3_analyze*.py file and return its `config` object."""
    spec = importlib.util.spec_from_file_location(
        f"step3_audit_{script_path.parent.name}_{script_path.stem}", script_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def derive_results_key(script_path: Path) -> str:
    """Map step3_analyze[_variant].py → results/{KEY}/ name."""
    galaxy_dir = script_path.parent.name           # e.g. "ngc3627_analyse_code"
    galaxy = galaxy_dir.replace("_analyse_code", "").upper()
    stem = script_path.stem                        # e.g. "step3_analyze_adaptive"
    if stem == "step3_analyze":
        return galaxy
    suffix = stem.replace("step3_analyze_", "")
    return f"{galaxy}_{suffix}"


def get_aperture_radius(cube_path: str, signal_size_arcsec: float) -> float:
    """Methodology rule: radius = 3 × max(bmaj, bmin), capped at signal/4,
    floored at 3″. Captures ~99% of point-source flux while bounded by box."""
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
    bmaj = float(hdr['BMAJ']) * 3600
    bmin = float(hdr['BMIN']) * 3600
    return max(3.0, min(3 * max(bmaj, bmin), signal_size_arcsec / 4))


def signal_box_pix_from_skyregion(cube_path: str, ra_deg: float, dec_deg: float,
                                  size_arcsec: float, dec_size_arcsec: float | None = None
                                  ) -> tuple[int, int, int, int]:
    """Convert SkyRegion → cube pixel box."""
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u
    with fits.open(cube_path, memmap=True) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr).celestial
        try:
            pix_arcsec = abs(float(hdr['CDELT1'])) * 3600
        except KeyError:
            pix_arcsec = abs(float(hdr['CD1_1'])) * 3600
    sky = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
    cx, cy = wcs.world_to_pixel(sky)
    cx, cy = float(cx), float(cy)
    half_x = int(round(0.5 * size_arcsec / pix_arcsec))
    half_y = int(round(0.5 * (dec_size_arcsec or size_arcsec) / pix_arcsec))
    return (int(cx - half_x), int(cy - half_y),
            int(cx + half_x), int(cy + half_y))


def run_one_galaxy(script_path: Path) -> AuditRow:
    key = derive_results_key(script_path)
    print(f"\n[{key}] script={script_path.name}")
    try:
        config = import_config_module(script_path)
    except Exception as e:
        return AuditRow(key=key, galaxy=key, weak_line="?", z=float('nan'),
                        cube_path="?", aperture_radius_arcsec=float('nan'),
                        npix=0, n_eff_beam=float('nan'),
                        sigma_chan_jy=float('nan'),
                        flux_jy_kms_W200=float('nan'), sigma_int_W200=float('nan'),
                        sn_W200=float('nan'), flux_jy_kms_best=float('nan'),
                        sn_best=float('nan'), width_best=float('nan'),
                        mode_used="error", error=f"config import failed: {e}")

    cube_path = config.hrl_pbcor_path
    if not Path(cube_path).exists():
        return AuditRow(key=key, galaxy=config.galaxy, weak_line=config.weak_line,
                        z=config.z, cube_path=cube_path,
                        aperture_radius_arcsec=float('nan'),
                        npix=0, n_eff_beam=float('nan'), sigma_chan_jy=float('nan'),
                        flux_jy_kms_W200=float('nan'), sigma_int_W200=float('nan'),
                        sn_W200=float('nan'), flux_jy_kms_best=float('nan'),
                        sn_best=float('nan'), width_best=float('nan'),
                        mode_used="error", error="cube not found")

    rest_freq_ghz = HRL_REST_GHZ.get(config.weak_line)
    if rest_freq_ghz is None:
        return AuditRow(key=key, galaxy=config.galaxy, weak_line=config.weak_line,
                        z=config.z, cube_path=cube_path,
                        aperture_radius_arcsec=float('nan'),
                        npix=0, n_eff_beam=float('nan'), sigma_chan_jy=float('nan'),
                        flux_jy_kms_W200=float('nan'), sigma_int_W200=float('nan'),
                        sn_W200=float('nan'), flux_jy_kms_best=float('nan'),
                        sn_best=float('nan'), width_best=float('nan'),
                        mode_used="error", error=f"unknown line {config.weak_line}")
    rest_hz = rest_freq_ghz * 1e9

    sig = config.signal_region
    print(f"  galaxy={config.galaxy} line={config.weak_line} z={config.z}")
    print(f"  cube={Path(cube_path).name}")
    print(f"  signal_region: ra={sig.ra_deg:.4f} dec={sig.dec_deg:.4f} size={sig.size_arcsec}")

    try:
        radius = get_aperture_radius(cube_path, sig.size_arcsec)
        box = signal_box_pix_from_skyregion(
            cube_path, sig.ra_deg, sig.dec_deg, sig.size_arcsec,
            getattr(sig, 'dec_size_arcsec', None))
        cm = build_aperture_mask_auto(
            cube_path,
            ra_deg=sig.ra_deg, dec_deg=sig.dec_deg,
            rest_freq_hz=rest_hz, z=config.z,
            fallback_radius_arcsec=radius,
            line_centers_kms=(0.0,),
            line_window_kms=300.0,
            sigma_thresh=3.0,
            signal_box_pix=box,
            pb_path=getattr(config, 'hrl_pb_path', None),
            pb_cutoff=0.5,
        )
        print(f"  aperture: mode={cm.mode_used}, npix={cm.npix}, radius={radius:.2f}″")

        spec = extract_aperture_spectrum(
            cube_path, cm.mask_2d, rest_freq_hz=rest_hz, z=config.z)

        # Combined-cube case: HRL and strong-line cubes are the same FITS
        # (e.g. NGC 3627 / NGC 3628 spw3 covers both H30α and CO(2-1)).
        # The strong line lives ~1700 km/s away in the H30α-rest velocity
        # frame and must be excluded from the baseline fit, otherwise the
        # linear polynomial gets pulled up at one end and the integration
        # window absorbs ~5-10 mJy/beam of pedestal × Npix as fake flux
        # (the NGC 3627 45-78 Jy·km/s pathology).
        excluded_velocities_kms: tuple[tuple[float, float], ...] = ()
        if Path(config.co_pbcor_path).resolve() == Path(cube_path).resolve():
            strong_rest = STRONG_REST_GHZ.get(config.strong_line)
            if strong_rest is not None:
                # v of strong line in weak-line rest frame (z cancels at low z)
                v_strong = -C_KMS * (strong_rest - rest_freq_ghz) / rest_freq_ghz
                # Exclude the full strong-line envelope. AnalysisConfig
                # defaults strong_line_window_kms to 300 km/s (tuned for the
                # main pipeline's CO mask integration). For CA baseline-fit
                # exclusion we want a generous safety margin, so use
                # max(config-value, 500). Covers nearby-galaxy CO profiles
                # (FWHM ≤ 250 km/s, full envelope ≤ 400 km/s) with cushion.
                vhw_strong = max(float(getattr(config, 'strong_line_window_kms',
                                               300.0) or 300.0), 500.0)
                excluded_velocities_kms = ((v_strong, vhw_strong),)
                print(f"  combined cube: excluding {config.strong_line} at "
                      f"v={v_strong:+.0f} km/s ±{vhw_strong:.0f} from baseline fit")

        # W=200 km/s (Toma standard)
        r200 = measure_line(spec, integration_window_kms=200.0,
                            excluded_velocities_kms=excluded_velocities_kms)

        # Best W from extended scan 100–500 km/s (covers edge-on disks)
        best_sn, best = -np.inf, r200
        best_W = 200.0
        for W in BEST_W_SCAN_KMS:
            try:
                r = measure_line(spec, integration_window_kms=W,
                                 excluded_velocities_kms=excluded_velocities_kms)
                if r['sn'] > best_sn:
                    best_sn = r['sn']
                    best = r
                    best_W = W
            except RuntimeError:
                continue

        print(f"  S/N(W=200)={r200['sn']:.2f}, S/N(best W={best_W})={best['sn']:.2f}, "
              f"flux={r200['flux_jy_kms']:.3f} Jy·km/s")

        return AuditRow(
            key=key, galaxy=config.galaxy, weak_line=config.weak_line,
            z=config.z, cube_path=cube_path,
            aperture_radius_arcsec=radius,
            npix=cm.npix, n_eff_beam=cm.npix / spec.beam_area_pix,
            sigma_chan_jy=spec.sigma_chan_jy,
            flux_jy_kms_W200=r200['flux_jy_kms'], sigma_int_W200=r200['sigma_int_jy_kms'],
            sn_W200=r200['sn'],
            flux_jy_kms_best=best['flux_jy_kms'],
            sn_best=best['sn'], width_best=best_W,
            mode_used=cm.mode_used, error=None)
    except Exception as e:
        traceback.print_exc()
        return AuditRow(key=key, galaxy=config.galaxy, weak_line=config.weak_line,
                        z=config.z, cube_path=cube_path,
                        aperture_radius_arcsec=float('nan'),
                        npix=0, n_eff_beam=float('nan'), sigma_chan_jy=float('nan'),
                        flux_jy_kms_W200=float('nan'), sigma_int_W200=float('nan'),
                        sn_W200=float('nan'), flux_jy_kms_best=float('nan'),
                        sn_best=float('nan'), width_best=float('nan'),
                        mode_used="error", error=str(e))


def write_per_galaxy_summary(out_dir: Path, row: AuditRow):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Persist row as JSON for aggregation across runs
    (out_dir / "row.json").write_text(json.dumps(asdict(row), indent=2, default=str))
    md = out_dir / "summary.md"
    if row.error:
        md.write_text(f"# {row.key} — continuum-aperture\n\n**ERROR**: {row.error}\n")
        return
    md.write_text(f"""# {row.key} — Continuum-Aperture Analysis (Toma-style reproduction)

**Method**: independent reproduction of Toma+ thesis Ch. 2.1.2 methodology.
NOT Toma's code; written by us based on her methodology spec. See
`master_thesis/my_code/_continuum_aperture/` for implementation.

**Galaxy**: {row.galaxy}
**Line**: {row.weak_line} (rest {HRL_REST_GHZ[row.weak_line]:.4f} GHz)
**Redshift**: {row.z}
**Cube**: `{row.cube_path}`

## Aperture

| Field | Value |
|---|---|
| Mode | `{row.mode_used}` |
| Radius | {row.aperture_radius_arcsec:.2f}″ |
| N pixels | {row.npix} |
| N effective beams | {row.n_eff_beam:.1f} |

## Aperture-integrated 1D spectrum

| Field | Value |
|---|---|
| σ_chan (off-line) | {row.sigma_chan_jy*1e3:.3f} mJy |

## Toma-style measurement

### W = 200 km/s (literature standard for upper limits)
| Field | Value |
|---|---|
| Flux | **{row.flux_jy_kms_W200:.4f} Jy·km/s** |
| σ_int | {row.sigma_int_W200:.4f} Jy·km/s |
| **S/N** | **{row.sn_W200:.2f}** |

### Best window (W in 100–300 km/s, no full scan)
| Field | Value |
|---|---|
| W_best | {row.width_best:.0f} km/s |
| Flux | {row.flux_jy_kms_best:.4f} Jy·km/s |
| **S/N** | **{row.sn_best:.2f}** |

## Cross-reference

Compare with `results/{row.key}/summary.md` (scan-optimized headline).
Two methods differ:
- Scan-optimized uses CO-defined mask + 828-combo width/threshold scan.
- Continuum-aperture uses a fixed circular aperture + fixed 200 km/s window.

For a robust detection, both should agree on verdict (detection / non-detection).
Disagreement reveals selector bias in the scan, or methodology-dependent flux drift.
""")


def aggregate_summary(rows: list[AuditRow], out_root: Path):
    """Build the cross-comparison audit_summary.md."""
    from datetime import date
    md = out_root / "audit_summary.md"
    n_total = len(rows)
    n_ok = sum(r.error is None for r in rows)
    lines = [
        f"# Continuum-aperture audit — {n_ok}/{n_total} galaxies",
        f"",
        f"**Date**: {date.today().isoformat()}",
        f"**Method**: independent reproduction of Toma+ thesis Ch. 2.1.2 methodology.",
        f"**Aperture**: auto (continuum > 3σ if available; else circular at NED nucleus).",
        f"**Window**: 200 km/s (Toma standard) + best-of-scan over [100, 150, 200, 250, 300, 400, 500] km/s.",
        f"",
        f"This audit is **independent** of and **does not modify** the existing scan-optimized",
        f"results in `results/{{GAL}}/summary.md`. Cross-comparison only — detection",
        f"verdicts and headline flux numbers in the thesis remain on scan-optimized.",
        f"",
        f"## Results table",
        f"",
        f"| Key | Line | mode | r (″) | Nbeam | σ (mJy) | Flux W=200 (Jy·km/s) | S/N W=200 | Flux best | S/N best | W_best |",
        f"|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x.key):
        if r.error:
            lines.append(f"| {r.key} | {r.weak_line} | error | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r.key} | {r.weak_line} | {r.mode_used} | "
            f"{r.aperture_radius_arcsec:.2f} | {r.n_eff_beam:.1f} | "
            f"{r.sigma_chan_jy*1e3:.2f} | "
            f"{r.flux_jy_kms_W200:.4f} | **{r.sn_W200:.2f}** | "
            f"{r.flux_jy_kms_best:.4f} | **{r.sn_best:.2f}** | {r.width_best:.0f} |"
        )
    lines += [
        f"",
        f"## Errors",
        f"",
    ]
    for r in rows:
        if r.error:
            lines.append(f"- **{r.key}**: {r.error}")
    md.write_text("\n".join(lines) + "\n")
    print(f"\nAggregate written: {md}")


def process_one_subprocess_mode(script_path: Path, out_root: Path):
    """Single-galaxy mode: process one config, write row.json + summary.md, exit.

    Invoked by the orchestrator via subprocess so each galaxy gets a fresh
    Python process (mmap pages from prior cubes released to OS).
    """
    row = run_one_galaxy(script_path)
    write_per_galaxy_summary(out_root / row.key, row)
    print(f"DONE {row.key} sn={row.sn_W200:.2f} flux={row.flux_jy_kms_W200:.4f} mode={row.mode_used}",
          flush=True)
    sys.exit(0 if row.error is None else 2)


def main():
    repo = Path("/Volumes/HouAstro/master")
    out_root = repo / "results/_continuum_aperture"
    out_root.mkdir(parents=True, exist_ok=True)

    if "--process-one" in sys.argv:
        idx = sys.argv.index("--process-one")
        script_path = Path(sys.argv[idx + 1])
        process_one_subprocess_mode(script_path, out_root)
        return  # not reached (exits inside)

    scripts = sorted((repo / "master_thesis/my_code").glob("*_analyse_code/step3_analyze*.py"))
    scripts = [s for s in scripts if "_legacy" not in s.name]
    print(f"Discovered {len(scripts)} step3_analyze configs.", flush=True)

    skip_existing = "--rerun" not in sys.argv

    rows = []
    PYTHON = "/opt/anaconda3/envs/casa_env/bin/python"
    for i, script in enumerate(scripts, 1):
        key = derive_results_key(script)
        cache = out_root / key / "row.json"
        if skip_existing and cache.exists():
            try:
                data = json.loads(cache.read_text())
                rows.append(AuditRow(**data))
                print(f"[{i}/{len(scripts)}] {key} → CACHED (sn={data.get('sn_W200'):.2f})",
                      flush=True)
                continue
            except Exception:
                pass

        print(f"[{i}/{len(scripts)}] {key} → forking subprocess", flush=True)
        proc = subprocess.run(
            [PYTHON, str(Path(__file__).resolve()), "--process-one", str(script)],
            capture_output=False, text=True,
        )
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                rows.append(AuditRow(**data))
            except Exception as e:
                print(f"  WARN: row.json unreadable for {key}: {e}", flush=True)
        else:
            print(f"  ERROR: subprocess for {key} died (rc={proc.returncode}), "
                  f"no row.json written", flush=True)

    aggregate_summary(rows, out_root)
    print(f"\n{sum(r.error is None for r in rows)}/{len(rows)} succeeded", flush=True)


if __name__ == "__main__":
    main()
