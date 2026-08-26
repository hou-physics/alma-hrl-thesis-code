#!/bin/bash
# Patient IR-fetch retry loop (2026-08-04 overnight).
# SkyView WISE + VizieR IRAS were intermittently down; this loop deletes
# stage3 rows left unusable (no-image / no-iras), re-runs stage 3 (resume =
# only the missing galaxies re-fetch), and repeats every 15 min until clean
# or 10 rounds. Downstream (iras_errors -> master table -> plots) reruns
# once at the end either way.
set -u
cd /Volumes/HouAstro/master/master_thesis/my_code
OUTDIR=/Volumes/HouAstro/master/result_v2/_uniform_batch

# NOTE: heredoc-stdin via `conda run` silently no-ops — keep this a real file
purge_bad () {
    conda run -n casa_env python _step3/wave1_purge_bad_ir.py
}

# Order matters: stage 3 FIRST (fills rows missing after a purge), THEN
# purge-and-check — a purge-first loop breaks immediately on already-purged
# rows while their galaxies are still absent from the CSV.
for round in 1 2 3 4 5 6 7 8 9 10; do
    echo "=== retry round $round  $(date '+%F %T') ==="
    conda run -n casa_env --no-capture-output python -u \
        _step3/uniform_batch_stage3_ir.py 2>&1 \
        | grep -v "WARNING\|FITSFixed\|leading zeroes\|OBSGEO"
    n=$(purge_bad | tee /dev/stderr | grep -o 'BAD=[0-9]*' | cut -d= -f2)
    if [ "$n" = "0" ]; then
        echo "all stage-3 rows clean — done"
        break
    fi
    echo "still $n unusable — sleeping 15 min before round $((round+1))"
    sleep 900
done

echo "=== downstream refresh  $(date '+%F %T') ==="
conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_iras_errors.py 2>&1 | grep -v WARNING
conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_master_table.py 2>&1 | grep -v WARNING
conda run -n casa_env --no-capture-output python -u _step3/uniform_batch_stage4_plot.py 2>&1 | grep -v WARNING
echo "=== retry driver finished  $(date '+%F %T') ==="
