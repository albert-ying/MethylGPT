#!/bin/bash
# Runner for large_gain_J_gs_missing_cpgs_knn_2026-10-28.py (execution only): runs the stages in order inside a
# transient systemd user unit, skipping a stage whose done-marker exists; stops at the first failing stage.
# Start from the repository directory:
#   systemd-run --user --unit=large-gain-J-2026-10-28 --collect --same-dir -E PATH="$PATH" -E TMPDIR=/Entropy/kying/tmp \
#     -E PYTHONUNBUFFERED=1 -E NUMPY_MADVISE_HUGEPAGE=0 bash run_large_gain_J_2026-10-28.sh [first_stage] [last_stage]
set -euo pipefail
cd "$(dirname "$0")"
PY=/usr/bin/python
BL=/home/kying/anaconda3/envs/integrative/bin/python
S=large_gain_J_gs_missing_cpgs_knn_2026-10-28.py
L=artifacts/logs
STAGES=(masks mgpt knn cohort clockin clocks cox checks boot analyse)
FIRST=${1:-masks}
LAST=${2:-analyse}
on=0
for st in "${STAGES[@]}"; do
  [ "$st" = "$FIRST" ] && on=1
  if [ $on = 1 ]; then
    if [ -f "$L/large_gain_J_${st}_2026-10-28.done" ]; then
      echo "[$(date +%T)] $st done earlier, skipped"
    else
      echo "[$(date +%T)] $st start"
      if [ "$st" = clocks ]; then
        $BL -u $S --stage clocks --workers 4 >> "$L/large_gain_J_${st}_2026-10-28.log" 2>&1
      else
        $PY -u $S --stage "$st" >> "$L/large_gain_J_${st}_2026-10-28.log" 2>&1
      fi
      echo "[$(date +%T)] $st finished"
    fi
  fi
  [ "$st" = "$LAST" ] && break
done
