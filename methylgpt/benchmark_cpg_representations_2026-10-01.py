#!/usr/bin/env python3
"""Quantitative benchmark of CpG representations against genomic annotations.

CpG representations (one vector per vocabulary CpG):
  factor_loadings      MethylGPT-2 core loadings V * sqrt(lambda) (1,536), learned from the
                       co-methylation of 145,700 training profiles
  encoder_input        columns of the MethylGPT-2 dense encoder's input weights (2,048)
  raw_profiles         beta values of the CpG across 2,000 random training profiles (training-
                       mean filled)
  raw_pca64            the first 64 principal components of raw_profiles
  random               Gaussian vectors (1,536), a chance-level control
Annotations (EPIC v1 / HM450 manifests): relation to CpG island (island, shore, shelf, open sea),
enhancer (450K or FANTOM5), X chromosome versus autosome. Metrics: kNN (k = 10, cosine)
balanced accuracy in 5-fold stratified CV, silhouette (cosine, 50 PCs, 10,000-CpG sample) and
mean pairwise energy distance between classes (50 PCs scaled to unit total variance,
2,000 CpGs per class).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from methylgpt_lib import ARTIFACTS, Corpus, PPCACore, log, require, write_json


def std(Z):
    Z = np.asarray(Z, np.float64)
    return (Z - Z.mean(0)) / np.where(Z.std(0) > 1e-8, Z.std(0), 1.0)


def energy(P, y, rng, n=2000):
    P = P / np.sqrt((P ** 2).sum(1).mean())
    cats = sorted(set(y))
    idx = {c: rng.choice(np.flatnonzero(y == c), min(n, (y == c).sum()), replace=False) for c in cats}
    T = torch.as_tensor(P)
    w = {c: torch.cdist(T[i], T[i]).mean().item() for c, i in idx.items()}
    out = [2 * torch.cdist(T[idx[a]], T[idx[b]]).mean().item() - w[a] - w[b]
           for k, a in enumerate(cats) for b in cats[k + 1:]]
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/cpg_representation_benchmark_2026-10-01.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261001)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    core = PPCACore.load(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz", rank=1536)
    reps = {"factor_loadings": (core.V * core.lam.sqrt()).numpy()}
    ck = torch.load(require(Path(a.ckpt)), map_location="cpu", weights_only=False)
    reps["encoder_input"] = ck["model"]["dense_w.weight"].float().T.numpy()
    corpus = Corpus()
    rows = np.sort(rng.choice(corpus.rows("train"), 2000, replace=False))
    X = corpus.get(rows)
    X = np.where(np.isfinite(X), X, core.mu.numpy()[None, :])
    reps["raw_profiles"] = X.T
    reps["raw_pca64"] = PCA(64, random_state=0).fit_transform(std(X.T))
    reps["random"] = rng.standard_normal((len(ann), 1536))
    labels = {"island_relation": ann.island_relation.to_numpy(),
              "enhancer": np.where(ann.in_epic_v1, ann.enhancer.astype(str), None),
              "x_chromosome": np.where(ann.chr == "X", "X", "autosome")}
    res = {"n_cpgs": int(len(ann)), "n_raw_profiles": int(len(rows)), "results": {}}
    for name, Z in reps.items():
        Zs = std(Z)
        P = PCA(50, random_state=0).fit_transform(Zs) if Zs.shape[1] > 50 else Zs
        block = {}
        for lab, y in labels.items():
            ok = pd.notna(y)
            yy = np.asarray(y)[ok].astype(str)
            pred = cross_val_predict(KNeighborsClassifier(10, metric="cosine", n_jobs=16), Zs[ok], yy,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=20261001))
            sub = rng.choice(np.flatnonzero(ok), 10000, replace=False)
            block[lab] = {"knn_balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
                          "silhouette": float(silhouette_score(P[sub], np.asarray(y)[sub].astype(str), metric="cosine")),
                          "energy_distance": energy(P[ok], yy, rng)}
        res["results"][name] = block
        log(f"{name}: " + "; ".join(f"{k} kNN {v['knn_balanced_accuracy']:.3f} sil {v['silhouette']:.3f} E {v['energy_distance']:.3f}"
                                    for k, v in block.items()))
        write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
