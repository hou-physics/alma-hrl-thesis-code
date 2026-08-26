# ALMA HRL survey — thesis code

Analysis pipeline, workflow scripts, and per-galaxy script templates for the
master's thesis "Hydrogen Recombination Lines in Nearby Galaxies: Harvesting
the ALMA Archive" (Zhengxu Hou, AIfA Bonn, 2026).

The survey measures millimeter hydrogen recombination lines (H30a, H40a) in
nearby galaxies from archival ALMA data with one frozen prescription: a
measurement aperture built from a bright molecular tracer line, a
recombination-line flux integrated inside it, a globally calibrated
uncertainty, and an aperture-matched infrared luminosity for the
L(HRL)-L(TIR) correlation.

## Layout

| Path | Content |
|---|---|
| `code/_step3/` | The shared analysis package: uniform-batch stages 1-4 (aperture, spectroscopy, infrared side, correlation plot), mask/baseline/flux modules, survey registry, distance adoption, plot style; `tests/` with runnable fixtures |
| `code/{galaxy}_analyse_code/` | Per-galaxy script templates: `step-1_download.py` (selective ALMA download), `step0_cleanup.sh`, `step1_uvcontsub.py`, `step2_imaging.py` (CASA re-imaging fallback), `step3_analyze.py` |
| `code/target_list_build/` | Sample selection workflow: RBGS x DEC x Spitzer-8um chain, ALMA archive scout, KS selection checks, table builders |
| `code/_continuum_aperture/` | Continuum cross-check tooling |
| `code/_galaxy_audit/` | 36-configuration audit tables |
| `code/_cleanup/` | Working-directory hygiene scripts |
| `code/run_null_test.py`, `code/analyze_null_test.py` | Translated-aperture null diagnostics |
| `code/download_all_table_b.sh` | Batch download driver |

## Environment

All scripts run inside a conda environment (`casa_env`) providing astropy,
astroquery, casatools/casatasks, radio-beam, reproject, matplotlib, and
pypdf. Cube handling is memory-mapped throughout; gzip-compressed FITS are
decompressed to scratch before mapping.

## Related repository

The generic ALMA archive query package used by the selection workflow is
maintained separately at
[hou-physics/alma-archive-tools](https://github.com/hou-physics/alma-archive-tools).
Two scripts under `code/target_list_build/` import it via an absolute local
path; point that path at a checkout of the package to run them elsewhere.

## Data

FITS cubes, CASA intermediates, and per-galaxy products are not part of this
repository. Scripts reference the survey's local directory layout
(`work_dir/{GALAXY}/`, `result_v2/_uniform_batch/`) and serve as templates
for reproducing the processing on downloaded archival data.
