#!/usr/bin/env python3
"""Run the age label-efficiency grid of the pretraining experiment (train_aging_heads_2026-09-30.py --n-labels).

Arms (identical architecture, optimizer and schedule):
  pretrained    factor model fitted to the unlabelled corpus + transformer after masked-value pretraining
  factor_only   the same factor model + transformer from random weights
  none          random orthonormal factor loadings (training means and variances kept) + transformer from random weights
Modes: finetune (encoder and heads trained), frozen (heads only; the encoder keeps its initial weights) and last1 (the last
self-attention block of the encoder and the heads trained; everything else keeps its initial weights).
Label subsets: 100, 300, 1,000, 3,000 and 10,000 training profiles with a clean age, nested for one seed, three seeds;
a fifth of each subset selects the checkpoint. Steps 1,000 / 1,000 / 2,000 / 3,000 / 4,000, evaluation every tenth.
Runs whose output folder exists (finished or in progress) are skipped, so several runners can share the grid;
remove the folder of an interrupted run to repeat it.
Usage: run_label_efficiency_2026-10-12.py [--modes finetune,frozen] [--seeds 0,1,2] [--gpus 0,1] [--workers-per-gpu 2]
"""
import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts/label_efficiency_2026-10-12"
CKPT = "artifacts/runs/stopped_cf_dense_2026-09-30/best.pt"
ARMS = {"pretrained": [], "factor_only": ["--random-init"],
        "none": ["--random-init", "--core", str(HERE / "artifacts/ppca_core_random_2026-10-03.npz")]}
STEPS = {100: 1000, 300: 1000, 1000: 2000, 3000: 3000, 10000: 4000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="finetune,frozen")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    for mode in a.modes.split(","):
        for seed in map(int, a.seeds.split(",")):
            for n, steps in STEPS.items():
                for arm, extra in ARMS.items():
                    name = f"{mode}_{arm}_n{n}_s{seed}"
                    if (OUT / name).exists():   # finished, or started by another runner (the training script creates the folder)
                        continue
                    cmd = [sys.executable, "train_aging_heads_2026-09-30.py", "--ckpt", CKPT, "--out", str(OUT / name), "--n-labels", str(n),
                           "--label-seed", str(seed), "--seed", str(20261001 + seed), "--steps", str(steps), "--eval-every", str(steps // 10)] + extra
                    if mode == "frozen":
                        cmd.append("--freeze-encoder")
                    elif mode == "last1":
                        cmd += ["--train-last-layers", "1"]
                    jobs.put((name, cmd))
    print(f"{jobs.qsize()} runs queued", flush=True)

    def worker(gpu):
        while True:
            try:
                name, cmd = jobs.get_nowait()
            except queue.Empty:
                return
            with open(OUT / "logs" / f"{name}.log", "w") as fh:
                r = subprocess.run(cmd + ["--device", f"cuda:{gpu}"], cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
            print(f"{name}: exit {r.returncode}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus.split(",") for _ in range(a.workers_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all runs finished", flush=True)


if __name__ == "__main__":
    main()
