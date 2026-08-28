# ALMA archival HRL survey — pipeline and templates

Analysis code of the master's thesis *Hydrogen Recombination Lines in
Nearby Galaxies: Harvesting the ALMA Archive* (Zhengxu Hou, AIfA, University
of Bonn, 2026). The survey measures millimeter recombination-line fluxes, or
upper limits, with one frozen prescription for every source, and places them
on the L_HRL–L_TIR correlation.

This repository is a **template**: the pipeline is global and frozen, so a
new source is added by writing one small configuration file and running the
stages in order — not by tuning the analysis per galaxy. Start with
`template/NEW_SOURCE.md`.

## Layout

| Directory | Contents |
|---|---|
| `_step3/` | The shared analysis package and the survey batch drivers (stages 1–4), figure/table generators, and bookkeeping scripts. Script-by-script inventory: thesis Appendix "Analysis software". |
| `target_list_build/` | Sample selection: `rbgs_seip_intersect.py` (RBGS × declination × Spitzer-coverage criteria), `alma_scout.py` (per-candidate ALMA archive metadata), `selection_ks_check.py` (selection-bias checks). |
| `apportioning/` | `masklevel_x.py` — aperture-matched 8 μm apportioning of the total infrared luminosity (kept next to its data product, `s5_results.csv`). |
| `galaxies/` | One folder per processed source with its configuration record `step3_analyze.py` and, where the source was retrieved by script, its `step-1_download.py`. The 13 wave-1 sources are configured by the single onboarding table `galaxies/wave1_configs.csv` instead of per-source scripts. For NGC 3628, additionally the CASA re-imaging scripts (`step1_uvcontsub.py`, `step2_imaging.py`) used for the one imaging cross-check. |
| `tables/` | The survey tables: `survey_registry.csv` (all 142 targets — processed sources **and the 95 queued candidates** with archive metadata), `master_table.csv` (per-source results), `source_table.csv` (correlation-diagram coordinates per source), `stage1_masks.csv`, `stage2_flux.csv`, `stage3_ir.csv`, `adopted_distances.csv`, `iras_errors.csv`. |
| `template/` | `step3_analyze.py` skeleton + `NEW_SOURCE.md` walkthrough for adding a source. |

## Pipeline order

selection (`target_list_build/`) → retrieval (`galaxies/*/step-1_download.py`)
→ `_step3/uniform_batch_stage1.py` (aperture from the bright tracer)
→ `_step3/uniform_batch_stage2.py` (line width, flux, uncertainty, S/N)
→ `_step3/uniform_batch_stage3_ir.py` + `_step3/uniform_batch_rbgs_flux_patch.py`
  + `apportioning/masklevel_x.py` (infrared axis)
→ `_step3/uniform_batch_stage4_plot.py` (correlation diagram; a rerun prints
  the headline statistics quoted in the thesis)
+ `_step3/uniform_batch_master_table.py`, `_step3/uniform_batch_source_table.py`,
  `_step3/uniform_batch_mom0_gallery.py` (tables and figures), `_step3/survey_registry.py` (bookkeeping).

## Notes for use

- **Run layout.** The batch drivers glob `*_analyse_code/step3_analyze.py`
  next to `_step3/`; the `galaxies/` folder collects these per-source
  directories for readability — place (or symlink) them beside `_step3/`
  when running. The scripts carry absolute data paths from the machine the
  survey ran on: adjust the path constants at the top of the batch drivers
  (and in the configuration records) to your layout.
- **Configuration records are parsed, not executed.** The per-source
  `step3_analyze.py` files are read by `_step3/uniform_batch_configs.py`
  for paths, lines, and redshifts; their docstrings are the working notes
  from ingestion and are kept as provenance.
- **Requirements.** Python ≥ 3.10 with `numpy`, `scipy`, `astropy`,
  `matplotlib`, `astroquery`, `requests`, `reproject`; CASA only for the
  optional re-imaging path.
- Every numerical parameter is frozen; values and justifications are
  registered in the thesis (Appendix A). Do not tune per source.

## Citation

Zhengxu Hou, *Hydrogen Recombination Lines in Nearby Galaxies: Harvesting
the ALMA Archive*, MSc thesis, Argelander-Institut für Astronomie,
University of Bonn, 2026.
