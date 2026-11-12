#!/bin/bash
# Runner for large_gain_K_gs_individual_diseases_2026-10-30.py (execution only): runs the stages after the committed freeze
# inside a transient systemd user unit, in order, in parallel shards where the stage has them. Resumable: finished fold models,
# risks and done-markers are skipped by the script. Stops at the first failing step.
# Start from the repository directory:
#   systemd-run --user --unit=large-gain-K-2026-10-30 --collect --same-dir -p MemoryHigh=110G -E PATH="$PATH" \
#     -E TMPDIR=/Entropy/kying/tmp -E PYTHONUNBUFFERED=1 -E NUMPY_MADVISE_HUGEPAGE=0 bash run_large_gain_K_2026-10-30.sh [first_step] [last_step]
# Steps: features fit_cheap fit_main cohort clockin clocks apply checks boot analyse privacy
set -uo pipefail
cd "$(dirname "$0")"
PY=/usr/bin/python
BL=/home/kying/anaconda3/envs/integrative/bin/python
S=large_gain_K_gs_individual_diseases_2026-10-30.py
L=artifacts/logs
D=2026-10-30
NV12=${NV12:-6}
NGPU=${NGPU:-2}      # GPU fit workers per GPU
FIRST=${1:-features}
LAST=${2:-privacy}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

wait_all() {  # wait for every pid; fail if any failed
  local rc=0
  for p in "$@"; do wait "$p" || rc=1; done
  return $rc
}

step_features() { $PY -u $S --stage features --device cuda:0 >> "$L/large_gain_K_features_$D.log" 2>&1; }

step_fit_cheap() {
  local pids=()
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 $PY -u $S --stage fit --family cheap --shard $i --nshards 4 >> "$L/large_gain_K_fit_cheap_${i}_$D.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}"
}

step_fit_main() {
  local pids=()
  local n=$((NGPU * 2))
  for ((i = 0; i < n; i++)); do
    $PY -u $S --stage fit --family gpu --shard $i --nshards $n --device cuda:$((i % 2)) >> "$L/large_gain_K_fit_gpu_${i}_$D.log" 2>&1 &
    pids+=($!)
  done
  for ((i = 0; i < NV12; i++)); do
    $PY -u $S --stage fit --family v12 --shard $i --nshards $NV12 >> "$L/large_gain_K_fit_v12_${i}_$D.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}"
}

step_cohort() {
  local pids=()
  for i in 0 1; do
    $PY -u $S --stage cohort --shard $i --nshards 2 --device cuda:$i >> "$L/large_gain_K_cohort_${i}_$D.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}"
}

step_clockin() { $PY -u $S --stage clockin >> "$L/large_gain_K_clockin_$D.log" 2>&1; }
step_clocks() { $BL -u $S --stage clocks --workers 4 >> "$L/large_gain_K_clocks_$D.log" 2>&1; }

step_apply() {
  local pids=()
  for i in 0 1 2 3; do
    $PY -u $S --stage apply --shard $i --nshards 4 >> "$L/large_gain_K_apply_${i}_$D.log" 2>&1 &
    pids+=($!)
  done
  wait_all "${pids[@]}"
}

step_checks() { $PY -u $S --stage checks >> "$L/large_gain_K_checks_$D.log" 2>&1; }
step_boot() { OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $PY -u $S --stage boot >> "$L/large_gain_K_boot_$D.log" 2>&1; }
step_analyse() { $PY -u $S --stage analyse >> "$L/large_gain_K_analyse_$D.log" 2>&1; }
step_privacy() { $PY -u $S --stage privacy >> "$L/large_gain_K_privacy_$D.log" 2>&1; }

on=0
for st in features fit_cheap fit_main cohort clockin clocks apply checks boot analyse privacy; do
  [ "$st" = "$FIRST" ] && on=1
  if [ $on = 1 ]; then
    echo "[$(date +%T)] $st start"
    if ! "step_$st"; then
      echo "[$(date +%T)] $st FAILED"
      exit 1
    fi
    echo "[$(date +%T)] $st finished"
  fi
  [ "$st" = "$LAST" ] && break
done
