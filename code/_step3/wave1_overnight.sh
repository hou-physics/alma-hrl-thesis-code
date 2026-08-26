#!/bin/bash
# Wave-1 overnight autonomous run (2026-08-04).
# Chain: stage1 masks → stage2 flux → stage3 IR → IRAS errors →
#        master table → stage4 correlation plots.
# Prereq: target_list_build/wave1_onboard.py already run (wave1_configs.csv
# exists) — the driver refuses to start without it.
# Log: work_dir/_wave1/wave1_pipeline.log
set -e
cd /Volumes/HouAstro/master/master_thesis/my_code
CFG=/Volumes/HouAstro/master/master_thesis/work_dir/_wave1/wave1_configs.csv
if [ ! -s "$CFG" ]; then
    echo "FATAL: $CFG missing/empty — run wave1_onboard.py first" >&2
    exit 1
fi
RUN="conda run -n casa_env --no-capture-output python -u"
FILTER="WARNING\|FITSFixed\|VerifyWarning"

stage () {
    echo ""
    echo "############ $(date '+%F %T')  $1 ############"
    $RUN "_step3/$1" 2>&1 | grep -v "$FILTER" || return 1
}

stage uniform_batch_stage1.py
stage uniform_batch_stage2.py
stage uniform_batch_stage3_ir.py
stage uniform_batch_iras_errors.py
stage uniform_batch_master_table.py
stage uniform_batch_stage4_plot.py

echo ""
echo "############ $(date '+%F %T')  ALL STAGES COMPLETE ############"
