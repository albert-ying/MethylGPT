#!/usr/bin/env python3
"""Representation metrics with incomplete profiles, and Harmony applied to MethylGPT factor scores.

Uses the profile sets of benchmark_representations_2026-10-01.py (held-out datasets only; rows
from its cached npz files). Natural geometry, same metrics.
  Set A, tissue and sex structure, with each profile reduced to its 27K-array CpGs or with 70%
  of its observed CpGs removed (seed 20261071): raw beta filled with training-split means
  (as a standard pipeline would), its PCA-50, and MethylGPT factor scores computed from the
  remaining CpGs.
  Set B, blood batches: Harmony (batch = dataset) on the PCA-50 of raw beta and of MethylGPT
  factor scores, scored for mixing (batch kNN, iLISI) and retained biology (sex kNN, age R2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, "/Entropy/kying/tmp/pylib")
import harmonypy  # noqa: E402
from harmonypy import compute_lisi  # noqa: E402

from methylgpt_lib import cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, PPCACore, log, require, write_json

PLAT = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")


def center(Z):
    Z = np.asarray(Z, np.float64)
    return Z - Z.mean(0)


def pcs(Z, k=50):
    return PCA(n_components=k, random_state=0).fit_transform(center(Z))


def knn_bacc(Z, y):
    ok = pd.notna(y)
    pred = cross_val_predict(KNeighborsClassifier(10, metric="cosine"), center(Z[ok]), np.asarray(y)[ok],
                             cv=StratifiedKFold(5, shuffle=True, random_state=20261001))
    return float(balanced_accuracy_score(np.asarray(y)[ok], pred))


def main():
    cuda_preinit()
    dev = torch.device("cuda:0")
    corpus = Corpus()
    prof = corpus.profiles
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    core = PPCACore.load(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz", rank=1536).to(dev)
    mu = core.mu.cpu().numpy()
    res = {"set_a": {}, "set_b": {}}

    rows_a = np.load(require(ARTIFACTS / "eval/representations_setA_2026-10-01.npz"))["rows"]
    Xa = corpus.get(np.sort(rows_a))[np.argsort(np.argsort(rows_a))]
    tissue = prof.tissue.to_numpy()[rows_a]
    sex = prof.sex.to_numpy()[rows_a]
    sex = np.where(np.isin(sex, ["M", "F"]), sex, None)
    for scen in ("all", "hm27", "random_70"):
        X = Xa.copy()
        if scen == "hm27":
            X[:, ~PLAT["hm27"]] = np.nan
        elif scen == "random_70":
            X[np.random.default_rng(20261071).random(X.shape) < 0.7] = np.nan
        raw = np.where(np.isfinite(X), X, mu[None, :])
        S = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                x = torch.from_numpy(X[i:i + 64]).to(dev)
                S.append(core.posterior(x, torch.isfinite(x).float()).cpu().numpy())
        reps = {"raw (mean-filled)": raw, "raw PCA-50": pcs(raw), "MethylGPT factor scores": np.concatenate(S)}
        block = {}
        for name, Z in reps.items():
            P = Z if Z.shape[1] == 50 else pcs(Z)
            block[name] = {"tissue_knn_bacc": knn_bacc(Z, tissue), "sex_knn_bacc": knn_bacc(Z, sex),
                           "tissue_silhouette": float(silhouette_score(P, tissue, metric="cosine")),
                           "observed_fraction": float(np.isfinite(X).mean())}
            log(f"A {scen} {name}: {block[name]}")
        res["set_a"][scen] = block

    zb = np.load(require(ARTIFACTS / "eval/representations_setB_2026-10-01.npz"))
    rows_b = zb["rows"]
    batch = prof.group.astype(str).to_numpy()[rows_b]
    sex_b = prof.sex.to_numpy()[rows_b]
    age = lab.age_meta.to_numpy(float)[rows_b]
    meta = pd.DataFrame({"batch": batch})
    for name, Z in (("raw", zb["raw_full"]), ("MethylGPT factor scores", zb["rep_factor_scores"])):
        P = pcs(Z)
        for corr in ("none", "Harmony"):
            Q = P
            if corr == "Harmony":
                ho = harmonypy.run_harmony(P, meta, ["batch"], random_state=0, verbose=False)
                zc = np.asarray(ho.Z_corr)
                Q = zc.T if zc.shape[0] == P.shape[1] else zc
            age_pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Q, age, cv=KFold(5, shuffle=True, random_state=0))
            r = {"batch_knn_bacc": knn_bacc(Q, batch), "batch_ilisi": float(np.median(compute_lisi(Q, meta, ["batch"])[:, 0])),
                 "sex_knn_bacc": knn_bacc(Q, sex_b), "age_r2_ridge": float(r2_score(age, age_pred))}
            res["set_b"][f"{name} | {corr}"] = r
            log(f"B {name} | {corr}: {r}")
    write_json(ARTIFACTS / "eval/representation_benchmark_missing_2026-10-02.json", res)


if __name__ == "__main__":
    main()
