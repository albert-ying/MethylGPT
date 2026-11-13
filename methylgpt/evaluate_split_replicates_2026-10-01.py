#!/usr/bin/env python3
"""Reconstruction with cores fitted on different training splits.

The full core uses all 145,700 training profiles; the two half cores use the training datasets
with crc32(dataset) mod 2 = 0 or 1, so they share no dataset. All three are scored on the
same 2,000 held-out-dataset profiles and the same 30% mask as
evaluate_heldout_structured_2026-10-01.py, together with the training-split CpG mean. MSE
intervals resample the 107 held-out datasets (2,000 draws).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, Corpus, PPCACore, log, mse_metrics, write_json

CORES = {"all training datasets": "ppca_core_meanfilled_2026-09-30.npz",
         "half A (datasets crc32 mod 2 = 0)": "ppca_core_meanfilled_fold0_2026-09-30.npz",
         "half B (datasets crc32 mod 2 = 1)": "ppca_core_meanfilled_fold1_2026-09-30.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/split_replicates_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    dev = torch.device(a.device)
    corpus = Corpus()
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), 2000, replace=False))
    groups = corpus.profiles.loc[test_rows, "group"].astype(str).to_numpy()
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    obs = torch.isfinite(X)
    masked = obs & torch.from_numpy(np.random.default_rng(20261001).random(tuple(X.shape)) < 0.3).to(dev)
    ctx = obs & ~masked
    res = {"n_test": int(len(X)), "n_datasets": int(len(set(groups))), "cores": {}}
    gidx = pd.Series(np.arange(len(X))).groupby(groups).apply(np.asarray).tolist()
    for name, f in CORES.items():
        core = PPCACore.load(ARTIFACTS / f, rank=1536).to(dev)
        sse, cnt, prs, trs, cps = np.zeros(len(X)), np.zeros(len(X)), [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                s = core.posterior(X[i:i + 32], ctx[i:i + 32].float())
                pred = core.predict(s)
                for b in range(len(s)):
                    m = masked[i + b]
                    p_, t_ = pred[b][m].cpu().numpy(), X[i + b][m].cpu().numpy()
                    sse[i + b] = ((p_ - t_) ** 2).sum(); cnt[i + b] = len(t_)
                    prs.append(p_); trs.append(t_); cps.append(torch.nonzero(m)[:, 0].cpu().numpy())
        draws = []
        for _ in range(2000):
            pick = np.concatenate([gidx[j] for j in rng.integers(0, len(gidx), len(gidx))])
            draws.append(sse[pick].sum() / cnt[pick].sum())
        met = mse_metrics(np.concatenate(prs), np.concatenate(trs), np.concatenate(cps))
        res["cores"][name] = met | {"mse_ci95_dataset_bootstrap": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                                    "n_train_profiles": int(np.load(ARTIFACTS / f)["n_train"])}
        log(f"{name}: MSE {met['mse']:.5f} R2 {met['within_cpg_r2_pooled']:.3f} CI {res['cores'][name]['mse_ci95_dataset_bootstrap']}")
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
