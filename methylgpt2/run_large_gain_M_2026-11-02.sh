#!/bin/bash
# Runner for large_gain_M_gs_measurement_noise_2026-11-02.py (execution only): runs the stages in order inside a transient
# systemd user unit, skipping a stage whose done-marker exists; the GPU stages mgpt (2 shards), knn and cohort (2 shards)
# run in parallel on cuda:0 and cuda:1, and boot runs in 3 shards; stops at the first failing stage.
# Start from the repository directory:
#   systemd-run --user --unit=large-gain-M-2026-11-02 --collect --same-dir -p MemoryHigh=110G -E PATH="$PATH" \
#     -E TMPDIR=/Entropy/kying/tmp -E PYTHONUNBUFFERED=1 -E NUMPY_MADVISE_HUGEPAGE=0 bash run_large_gain_M_2026-11-02.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=/usr/bin/python
BL=/home/kying/anaconda3/envs/integrative/bin/python
S=large_gain_M_gs_measurement_noise_2026-11-02.py
L=artifacts/logs
D=2026-11-02
done_() { [ -f "$L/large_gain_M_${1}_${D}.done" ]; }
run() {   # run <stage> <logtag> [args...]
  local st=$1 tag=$2; shift 2
  echo "[$(date +%T)] $tag start"
  $PY -u $S --stage "$st" "$@" >> "$L/large_gain_M_${tag}_${D}.log" 2>&1
  echo "[$(date +%T)] $tag finished"
}
for st in inputs wcheck; do
  done_ $st && echo "$st done earlier, skipped" || run $st $st $([ $st = wcheck ] && echo --device cuda:0)
done
pids=()
if ! done_ mgpt; then
  done_ mgpt_shard0of2 || { run mgpt mgpt_shard0 --device cuda:0 --shard 0 --nshards 2 & pids+=($!); }
  done_ mgpt_shard1of2 || { run mgpt mgpt_shard1 --device cuda:1 --shard 1 --nshards 2 & pids+=($!); }
fi
(
  done_ knn || run knn knn --device cuda:1
  if ! done_ cohort; then
    c=()
    done_ cohort_shard0of2 || { run cohort cohort_shard0 --device cuda:0 --shard 0 --nshards 2 & c+=($!); }
    done_ cohort_shard1of2 || { run cohort cohort_shard1 --device cuda:1 --shard 1 --nshards 2 & c+=($!); }
    for p in "${c[@]}"; do wait "$p"; done
  fi
) & pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "a GPU stage failed"; exit 1; }
for st in mgpt knn cohort; do done_ $st || { echo "$st incomplete"; exit 1; }; done
done_ clockin || run clockin clockin
if ! done_ clocks; then
  echo "[$(date +%T)] clocks start"
  $BL -u $S --stage clocks --workers 6 >> "$L/large_gain_M_clocks_${D}.log" 2>&1
  echo "[$(date +%T)] clocks finished"
fi
for st in cox checks; do done_ $st || run $st $st; done
if ! done_ boot; then
  b=()
  for i in 0 1 2; do
    done_ boot_shard${i}of3 || { OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 run boot boot_shard$i --shard $i --nshards 3 & b+=($!); }
  done
  for p in "${b[@]}"; do wait "$p"; done
fi
done_ boot || { echo "boot incomplete"; exit 1; }
for st in analyse privacy; do done_ $st || run $st $st; done
echo "[$(date +%T)] all stages done"
