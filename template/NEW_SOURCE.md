# Adding a new source

1. **Candidacy.** Check `tables/survey_registry.csv` — the 95 queued
   candidates already carry archive metadata and a download estimate. For a
   source outside the registry, run the selection chain:
   `target_list_build/rbgs_seip_intersect.py` (sample criteria), then
   `target_list_build/alma_scout.py` (archive metadata per candidate).
2. **Retrieval.** Copy `template/step-1_download.py`, fill in the five TODO
   constants (MOUS, destination, redshift, name, line list), and run it: it
   downloads the archive products with resume, reconstructs the nonpbcor
   cube, and creates the canonically named symlinks. The shipped
   `galaxies/*/step-1_download.py` files are worked examples.
3. **Configuration.** Copy `template/step3_analyze.py` to
   `galaxies/{galaxy}_analyse_code/step3_analyze.py` and fill in paths,
   lines, redshift, window.
4. **Run the frozen pipeline**, in order (no per-source tuning):
   `_step3/uniform_batch_stage1.py` (aperture) →
   `_step3/uniform_batch_stage2.py` (width, flux, S/N) →
   `_step3/uniform_batch_stage3_ir.py` + `_step3/uniform_batch_rbgs_flux_patch.py`
   + `apportioning/masklevel_x.py` (infrared axis) →
   `_step3/uniform_batch_stage4_plot.py` (correlation diagram; prints the
   headline statistics) and `_step3/uniform_batch_master_table.py`.
5. **Bookkeeping.** `_step3/survey_registry.py` regenerates the registry;
   `_step3/distance_table_fetch.py` + `distance_adopt.py` maintain the
   literature-distance table for nearby sources.

Every parameter of the pipeline is frozen and justified in the thesis
(Appendix A); do not tune per source.
