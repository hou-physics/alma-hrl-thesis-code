"""Per-source configuration record — TEMPLATE.

Copy this file to  galaxies/{galaxy}_analyse_code/step3_analyze.py  (and, at
run time, place that directory next to _step3/ so the batch drivers find it:
the config builder globs  *_analyse_code/step3_analyze.py).

The batch pipeline PARSES this file; it does not execute it. The parser
(_step3/uniform_batch_configs.py) reads:
  - the module-level WORKDIR string and any UPPER_CASE string variables,
  - the fields of the AnalysisConfig(...) call listed below.
Everything else (docstring, comments) is a working record for humans:
document the archive project code, the spectral setup, and any pre-flight
checks here, as the shipped per-source files do.
"""

# Where the FITS cubes live. Naming convention:
#   {GALAXY}_{line}_v{N}_{pbcor|nonpbcor|pb}.fits   (line: H30a, H40a, ...)
WORKDIR = "/path/to/work_dir/GALAXY"

HRL_PBCOR = f"{WORKDIR}/GALAXY_H30a_pbcor.fits"
CO_NONPBCOR = f"{WORKDIR}/GALAXY_CO21_nonpbcor.fits"
CO_PBCOR = f"{WORKDIR}/GALAXY_CO21_pbcor.fits"

config = AnalysisConfig(  # noqa: F821 — parsed, not executed
    weak_line="H30a",            # the recombination line measured
    strong_line="CO21",          # the bright tracer that defines the aperture
    hrl_pbcor_path=HRL_PBCOR,
    co_nonpbcor_path=CO_NONPBCOR,
    co_pbcor_path=CO_PBCOR,
    z=0.001234,                  # heliocentric redshift used for both lines
    strong_line_window_kms=300,  # tracer moment-0 collapse window (full width)
)
