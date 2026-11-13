#!/usr/bin/env bash
# Revision training runs: seed replicates of the final model (aging heads -> multi-task) and the
# no-pretraining control. Usage: run_revision_training_2026-10-01.sh <device> <seed> <random-init lr>
set -euo pipefail
cd "$(dirname "$0")"
dev=$1; seed=$2; rlr=$3
enc=artifacts/runs/stopped_cf_dense_2026-09-30/best.pt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 train_aging_heads_2026-09-30.py --ckpt $enc --out artifacts/runs/aging_heads_seed${seed} --device $dev --seed $seed \
  > artifacts/logs/train_aging_heads_seed${seed}.log 2>&1
python3 train_multitask_2026-10-01.py --ckpt $enc --out artifacts/runs/multitask_bio_seed${seed} --device $dev --seed $seed \
  --steps 10000 --init-aging artifacts/runs/aging_heads_seed${seed}/best.pt > artifacts/logs/train_multitask_bio_seed${seed}.log 2>&1
python3 train_aging_heads_2026-09-30.py --ckpt $enc --out artifacts/runs/aging_heads_randominit_lr${rlr} --device $dev --seed 20261001 \
  --random-init --lr-enc $rlr > artifacts/logs/train_aging_heads_randominit_lr${rlr}.log 2>&1
