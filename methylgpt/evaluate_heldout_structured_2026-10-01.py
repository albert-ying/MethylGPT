#!/usr/bin/env python3
"""Reconstruction tests of increasing difficulty on held-out datasets.

Same 2,000 test-split profiles (107 datasets never used in training) and methods as
evaluate_heldout_platform_2026-10-01.py: training-split CpG mean, kNN (k = 10) over all
145,700 training profiles, and MethylGPT-2. Three mask types, each a single pass:
  random_30   30% of each profile's observed CpGs (seed 20261001)
  gene_fold   genes split into 5 folds (seeded); profile i loses every CpG annotated to a gene
              of fold i mod 5, so whole genes are unseen
  chromosome  profile i loses every CpG on chromosome (1..22, X)[i mod 23]
Outputs pooled MSE/MAE/within-CpG R2, profile-bootstrap intervals for MethylGPT-2 minus each
baseline, per-chromosome and per-fold errors, MAE by observed-beta bin (random_30), and a
seeded 1,000,000-pair sample of (observed, predicted) values for the density figure.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, Corpus, log, mse_metrics, require, write_json
from methylgpt_model import load_checkpoint

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("plat", HERE / "evaluate_heldout_platform_2026-10-01.py")
plat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plat)
CHROMS = [str(c) for c in range(1, 23)] + ["X"]
BINS = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/heldout_structured_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    corpus = Corpus()
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), a.n_test, replace=False))   # same draw as the platform run
    meta = corpus.profiles.loc[test_rows, ["id", "group", "tissue", "gpl"]].reset_index(drop=True)
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    model, core, ck = load_checkpoint(Path(a.ckpt), None, dev)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    log(f"test profiles {len(X)} from {meta.group.nunique()} datasets; checkpoint step {ck['step']}")

    genes = sorted({g for gs in ann.genes for g in gs.split(";") if g})
    fold_of = dict(zip(genes, np.random.default_rng(20261001).permutation(len(genes)) % 5))
    gene_mask = np.zeros((5, len(ann)), bool)
    for j, gs in enumerate(ann.genes):
        for g in gs.split(";"):
            if g:
                gene_mask[fold_of[g], j] = True
    chrom_mask = np.stack([(ann.chr == c).to_numpy() for c in CHROMS])

    train_rows = np.sort(corpus.rows("train"))
    Rraw = torch.empty((len(train_rows), X.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, len(train_rows), 8192):
        Rraw[i:i + 8192] = torch.from_numpy(corpus.beta[train_rows[i:i + 8192]]).to(dev)
    Rf = torch.where(torch.isfinite(Rraw), Rraw, core.mu.half())
    log("kNN reference loaded")

    obs = torch.isfinite(X)
    g = np.random.default_rng(20261001)
    masks = {
        "random_30": obs & torch.from_numpy(g.random(tuple(X.shape)) < 0.3).to(dev),
        "gene_fold": obs & torch.from_numpy(gene_mask[np.arange(len(X)) % 5]).to(dev),
        "chromosome": obs & torch.from_numpy(chrom_mask[np.arange(len(X)) % len(CHROMS)]).to(dev),
    }
    res = {"n_test": int(len(X)), "n_test_groups": int(meta.group.nunique()), "checkpoint": a.ckpt,
           "n_genes": len(genes), "mask_types": {}}
    for name, masked in masks.items():
        ctx = obs & ~masked
        truth = [X[i][masked[i]].cpu().numpy() for i in range(len(X))]
        cpgs = [torch.nonzero(masked[i])[:, 0].cpu().numpy() for i in range(len(X))]
        preds = {"cpg_mean": [core.mu[torch.from_numpy(c).to(dev)].cpu().numpy() for c in cpgs]}
        has = torch.nonzero(masked.any(1))[:, 0]           # profiles with at least one target
        empty = np.zeros(0, np.float32)
        mg, _ = plat.predict_model(model, core, X[has], ctx[has], masked[has])
        kn = plat.knn_predict(Rf, Rraw, X[has], ctx[has], masked[has])
        preds["methylgpt2"], preds["knn10"] = [empty] * len(X), [empty] * len(X)
        for j, i in enumerate(has.cpu().numpy()):
            preds["methylgpt2"][i], preds["knn10"][i] = mg[j], kn[j]
        block, pp = {}, {}
        tr = np.concatenate(truth); cp = np.concatenate(cpgs)
        for m, p in preds.items():
            pr = np.concatenate(p)
            block[m] = mse_metrics(pr, tr, cp)
            pp[m] = [(float(((p[i] - truth[i]) ** 2).sum()), len(truth[i])) for i in range(len(X)) if len(truth[i])]
        for other in ("knn10", "cpg_mean"):
            block[f"methylgpt2_minus_{other}"] = plat.boot_diff(pp["methylgpt2"], pp[other])
        groups = {"chromosome": [CHROMS[i % len(CHROMS)] for i in range(len(X))],
                  "gene_fold": [f"fold{i % 5}" for i in range(len(X))]}.get(name)
        if groups is not None:
            by = {}
            for key in sorted(set(groups), key=lambda k: (len(k), k)):
                idx = [i for i in range(len(X)) if groups[i] == key and len(truth[i])]
                t_ = np.concatenate([truth[i] for i in idx])
                by[key] = {"n_profiles": len(idx), "n_values": int(len(t_))} | {
                    m: {"mae": float(np.mean(np.abs(np.concatenate([preds[m][i] for i in idx]) - t_))),
                        "mse": float(np.mean((np.concatenate([preds[m][i] for i in idx]) - t_) ** 2))} for m in preds}
            block["by_group"] = by
        if name == "random_30":
            b = np.digitize(tr, BINS) - 1
            block["by_beta_bin"] = {f"[{BINS[k]:.2f},{min(BINS[k + 1], 1):.2f})": {"n": int((b == k).sum())} | {
                m: {"mae": float(np.mean(np.abs(np.concatenate(preds[m])[b == k] - tr[b == k]))),
                    "relative_mae": float(np.mean(np.abs(np.concatenate(preds[m])[b == k] - tr[b == k]) /
                                                  np.maximum(np.minimum(tr[b == k], 1 - tr[b == k]), 0.01)))}
                for m in preds} for k in range(len(BINS) - 1)}
            sel = np.random.default_rng(42).choice(len(tr), min(1_000_000, len(tr)), replace=False)
            np.savez_compressed(Path(a.out).with_suffix(".random30_pairs.npz"), observed=tr[sel].astype(np.float32),
                                **{m: np.concatenate(preds[m])[sel].astype(np.float32) for m in preds})
        res["mask_types"][name] = block
        log(f"{name}: " + ", ".join(f"{m} MSE {block[m]['mse']:.5f} R2 {block[m]['within_cpg_r2_pooled']:.3f}" for m in preds)
            + f"; MGPT2-kNN {block['methylgpt2_minus_knn10']}")
        write_json(Path(a.out), res)
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
