#!/usr/bin/env bash
# download_all_table_b.sh — run the 5 Table B top-pick step-1_download.py
# scripts sequentially. Python output goes straight to the terminal so the
# astroquery progress bar renders live. Designed to run inside a `screen` or
# `tmux` session so you can detach and reattach without killing it.
#
# Usage (recommended — screen session, survives disconnect):
#     screen -S alma
#     caffeinate -i bash download_all_table_b.sh
#     # Ctrl-A D to detach; `screen -r alma` to reattach
#
# Usage (simple foreground, terminal must stay open):
#     bash download_all_table_b.sh
#
# To resume after interruption, just re-run — astroquery `continuation=True`
# skips already-downloaded bytes; symlink creation step is idempotent.
set -u

PYTHON="/opt/anaconda3/envs/casa_env/bin/python"
ROOT="/Volumes/HouAstro/master/master_thesis/my_code"
LOG_DIR="$ROOT/_download_logs"
mkdir -p "$LOG_DIR"

GALAXIES=(
  "irasf18293-3413_analyse_code"   # sens 0.55, highest-priority
  "ngc7130_analyse_code"           # sens 0.61
  "ngc3227_analyse_code"           # sens 0.62
  "ic5063_analyse_code"            # sens 1.03, mosaic
  "ngc1386_analyse_code"           # sens 1.07, mosaic
)

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="$LOG_DIR/summary_${STAMP}.log"
ln -sf "summary_${STAMP}.log" "$LOG_DIR/summary_latest.log"

echo "=== ALMA Table B batch download ===" | tee -a "$SUMMARY"
echo "Start : $(date)"                      | tee -a "$SUMMARY"
echo "Python: $PYTHON"                      | tee -a "$SUMMARY"
echo "Order : ${GALAXIES[*]}"               | tee -a "$SUMMARY"
echo ""                                     | tee -a "$SUMMARY"

for dir in "${GALAXIES[@]}"; do
  galaxy="${dir%_analyse_code}"

  echo "-----" | tee -a "$SUMMARY"
  echo "[$(date +%H:%M:%S)] Starting $galaxy" | tee -a "$SUMMARY"

  if [[ ! -d "$ROOT/$dir" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $galaxy (dir $dir not found)" | tee -a "$SUMMARY"
    continue
  fi

  pushd "$ROOT/$dir" >/dev/null || continue
  start=$(date +%s)
  # No redirect: astroquery inherits the screen/tmux tty → progress bar renders.
  if "$PYTHON" step-1_download.py; then
    dur=$(( $(date +%s) - start ))
    echo "[$(date +%H:%M:%S)] OK   $galaxy  (${dur}s)" | tee -a "$SUMMARY"
  else
    dur=$(( $(date +%s) - start ))
    echo "[$(date +%H:%M:%S)] FAIL $galaxy  (${dur}s)" | tee -a "$SUMMARY"
  fi
  popd >/dev/null
done

echo ""                         | tee -a "$SUMMARY"
echo "End: $(date)"              | tee -a "$SUMMARY"
echo "=== Done ==="               | tee -a "$SUMMARY"
