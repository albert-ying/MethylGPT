#!/usr/bin/env bash
# Runner for large-gain analysis I (execution only; the design is in large_gain_I_sequencing_cohorts_2026-10-25.py).
# Runs the remaining stages in order and stops at the first failure: map, commit of the analysis table written by the
# map stage, crosscheck, represent, clocks (biolearn python), fit, analyse. Every stage is resumable per dataset and
# writes artifacts/logs/large_gain_I_<stage>_2026-10-27.done when complete; a stage whose marker exists is skipped.
# Started as a transient user unit:
#   systemd-run --user --unit=large-gain-I-2026-10-27 --collect --same-dir -E PATH="$PATH" -E TMPDIR=/Entropy/kying/tmp \
#       bash run_large_gain_I_2026-10-27.sh
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /Entropy/kying/agent-worktrees/methylgpt-retrain-2026-09-30/retrain
S=large_gain_I_sequencing_cohorts_2026-10-25.py
L=artifacts/logs
D=2026-10-27
PY=/usr/bin/python
BL=/home/kying/anaconda3/bin/python
OI=artifacts/large_gain_2026-10-25/I
mkdir -p "$L"

say() { echo "[$(date '+%F %T')] $*"; }
have() { test -f "$L/large_gain_I_$1_$D.done"; }
run() {
    local stage=$1; shift
    if have "$stage"; then say "skip $stage (done-marker present)"; return 0; fi
    say "start $stage"
    "$@" >> "$L/large_gain_I_${stage}_$D.log" 2>&1
    have "$stage" || { say "FAILED $stage: no done-marker after the stage ended"; exit 1; }
    say "end $stage"
}

run map "$PY" -I "$S" --stage map --workers 4

# the analysis table is frozen by the map stage and committed before representations and fits
say "commit analysis table"
while [ -e .git/index.lock ]; do sleep 5; done
git add -f "$OI/datasets_analysis_2026-10-25.csv" "$OI/mapping_check_2026-10-25.csv" "$OI/coverage_per_sample_2026-10-25.csv"
if ! git diff --cached --quiet -- "$OI"; then
    git commit -q -m "large_gain I: analysis table after mapping (technical exclusions T1-T4, sizes evaluated), mapping checks and per-sample vocabulary coverage; before any representation or fit" \
        -- "$OI/datasets_analysis_2026-10-25.csv" "$OI/mapping_check_2026-10-25.csv" "$OI/coverage_per_sample_2026-10-25.csv"
fi
say "analysis table committed: $(git log -1 --format=%h -- "$OI/datasets_analysis_2026-10-25.csv")"

if "$PY" -I -c "import pandas as pd,sys; t=pd.read_csv('$OI/datasets_analysis_2026-10-25.csv'); sys.exit(0 if bool(t[t.accession=='GSE194154'].included.any()) else 1)"; then
    run crosscheck "$PY" -I "$S" --stage crosscheck
fi
run represent "$PY" -I "$S" --stage represent
run clocks "$BL" -I "$S" --stage clocks
run fit "$PY" -I "$S" --stage fit
run analyse "$PY" -I "$S" --stage analyse
say "all stages done"
