#!/usr/bin/env python3
"""Quantitative benchmark of sample representations on held-out datasets.

All profiles come from test-split datasets that were never used for any training.
  Set A, tissue structure: up to 500 profiles from each of the six largest well-defined test
         tissues (Whole Blood, Brain, Colon, Liver, Muscle, Adipose).
  Set B, batch effects: whole-blood profiles with sex and age from datasets with >= 30 such
         profiles (up to 120 per dataset); batch = dataset.
Representations: raw beta (training-mean filled), PCA-50 of raw, ComBat-corrected raw (R sva,
batch = dataset; set B), Harmony on raw PCA-50 (set B), MethylGPT-2 factor scores, the
self-supervised MethylGPT-2 embedding, the multi-task MethylGPT-2 embedding, and two controls:
the same architecture with an untrained transformer, and a Gaussian random projection of raw
beta to 2,048 dimensions.
Metrics are computed in each representation's own geometry (centred, not rescaled; primary)
and, as a sensitivity analysis, after standardising every feature. kNN (k = 10, cosine, full
representation) balanced accuracy for tissue, sex and batch in 5-fold stratified CV; on the 50
leading principal components: silhouette (cosine), mean pairwise energy distance between
tissues (PCs scaled to unit total variance), iLISI for batch (perplexity 30) and age R2 from
5-fold ridge. Higher is better except batch kNN accuracy (lower = better mixing).
Representations are cached in artifacts/eval/representations_set{A,B}_2026-10-01.npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.insert(0, "/Entropy/kying/tmp/pylib")
import harmonypy  # noqa: E402
from harmonypy import compute_lisi  # noqa: E402

from methylgpt_lib import cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, log, require, write_json
from methylgpt_model import embed_batch, load_checkpoint

TISSUES = ["Whole Blood", "Brain", "Colon", "Liver", "Muscle", "Adipose"]
RLIB = "/Entropy/kying/tmp/rlib"


GEOMETRY = "natural"


def std(Z):
    Z = np.asarray(Z, np.float64)
    Z = Z - Z.mean(0)
    if GEOMETRY == "standardized":
        Z = Z / np.where(Z.std(0) > 1e-8, Z.std(0), 1.0)
    return Z


def pcs(Z, k=50, seed=0):
    return PCA(n_components=min(k, Z.shape[1], Z.shape[0] - 1), random_state=seed).fit_transform(std(Z))


def knn_bacc(Z, y, seed=20261001):
    ok = pd.notna(y)
    Z, y = std(Z[ok]), np.asarray(y)[ok]
    pred = cross_val_predict(KNeighborsClassifier(10, metric="cosine"), Z, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return float(balanced_accuracy_score(y, pred))


def energy_distance(P, labels):
    P = P / np.sqrt((P ** 2).sum(1).mean())
    T = torch.as_tensor(P)
    cats = sorted(set(labels))
    idx = {c: np.flatnonzero(np.asarray(labels) == c) for c in cats}
    within = {c: torch.cdist(T[i], T[i]).mean().item() for c, i in idx.items()}
    out = []
    for a in range(len(cats)):
        for b in range(a + 1, len(cats)):
            ab = torch.cdist(T[idx[cats[a]]], T[idx[cats[b]]]).mean().item()
            out.append(2 * ab - within[cats[a]] - within[cats[b]])
    return float(np.mean(out))


def combat(X, batch):
    """R sva::ComBat on a CpG x sample matrix (training-mean filled beta)."""
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as d:
        d = Path(d)
        # R fills matrix(nrow = CpGs) column by column, so the file must hold one sample after another: X (samples x CpGs) in C order.
        # Corrected 18 Oct 2026: the earlier X.T.tofile() wrote one CpG after another and ComBat ran on a scrambled matrix.
        np.ascontiguousarray(X, dtype=np.float64).tofile(d / "x.bin")
        pd.Series(batch).to_csv(d / "batch.csv", index=False, header=False)
        code = (f'.libPaths(c("{RLIB}", .libPaths())); suppressMessages(library(sva)); '
                f'x <- matrix(readBin("{d}/x.bin", "double", n={X.size}), nrow={X.shape[1]}, ncol={X.shape[0]}); '
                f'b <- read.csv("{d}/batch.csv", header=FALSE)[,1]; keep <- apply(x, 1, sd) > 1e-6; '
                f'y <- x; y[keep,] <- ComBat(x[keep,], batch=b); writeBin(as.vector(y), "{d}/y.bin")')
        subprocess.run(["Rscript", "-e", code], check=True, capture_output=True)
        return np.fromfile(d / "y.bin", dtype=np.float64).reshape(X.shape[0], X.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssl", default=str(ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"))
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only-combat", action="store_true",
                    help="recompute only the set B ComBat entries (corrected data exchange; all CpGs and the CpGs ComBat adjusts, with raw beta values on the same CpGs) and update --out in place")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/representation_benchmark_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    dev = torch.device(a.device)
    corpus = Corpus()
    prof = corpus.profiles
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    rng = np.random.default_rng(20261001)
    test = prof[prof.split == "test"]
    rows_a = np.concatenate([rng.choice(test.index[test.tissue == t], min(500, (test.tissue == t).sum()), replace=False)
                             for t in TISSUES])
    blood = test[(test.tissue == "Whole Blood") & test.sex.isin(["M", "F"]) & np.isfinite(lab.age_meta.to_numpy(float)[test.index])]
    big = blood.group.value_counts()
    big = big[big >= 30].index
    rows_b = np.concatenate([rng.choice(blood.index[blood.group == g], min(120, (blood.group == g).sum()), replace=False)
                             for g in sorted(big)])
    log(f"set A {len(rows_a)} profiles ({prof.loc[rows_a].group.nunique()} datasets); set B {len(rows_b)} blood profiles from {len(big)} datasets")

    ssl_model, core, _ = load_checkpoint(Path(a.ssl), None, dev)
    rnd_model, rnd_core, _ = load_checkpoint(Path(a.ssl), None, dev, random_init=True)
    mt = torch.load(a.multitask, map_location=dev, weights_only=False)
    mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
    mt_model.load_state_dict(mt["model"]); mt_model.eval()
    proj = torch.randn(corpus.beta.shape[1], 2048, generator=torch.Generator().manual_seed(20261001)) / np.sqrt(2048)

    @torch.no_grad()
    def represent(rows):
        X = corpus.get(np.sort(rows))
        order = np.argsort(np.argsort(rows))  # corpus.get sorts rows; restore the requested order
        X = X[order]
        out = {k: [] for k in ("factor_scores", "mgpt2_selfsupervised", "mgpt2_multitask", "untrained_transformer")}
        for i in range(0, len(X), 32):
            x = torch.from_numpy(X[i:i + 32]).to(dev)
            obs = torch.isfinite(x)
            out["factor_scores"].append(core.posterior(x, obs.float()).cpu().numpy())
            out["mgpt2_selfsupervised"].append(embed_batch(ssl_model, core, x, obs).float().cpu().numpy())
            out["mgpt2_multitask"].append(embed_batch(mt_model, mt_core, x, obs).float().cpu().numpy())
            out["untrained_transformer"].append(embed_batch(rnd_model, rnd_core, x, obs).float().cpu().numpy())
        reps = {k: np.concatenate(v) for k, v in out.items()}
        raw = np.where(np.isfinite(X), X, core.mu.cpu().numpy()[None, :])
        reps = {"raw": raw, "raw_pca50": PCA(50, random_state=0).fit_transform(raw - raw.mean(0)), **reps,
                "random_projection": (torch.from_numpy(raw - raw.mean(0)) @ proj).numpy()}
        return reps, raw

    def cached(name, rows):
        f = ARTIFACTS / f"eval/representations_{name}_2026-10-01.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            assert (z["rows"] == rows).all()
            return {k[4:]: z[k] for k in z.files if k.startswith("rep_")}, z["raw_full"]
        reps, raw = represent(rows)
        np.savez(f, rows=rows, raw_full=raw.astype(np.float32), **{f"rep_{k}": v.astype(np.float32) for k, v in reps.items() if k != "raw"})
        reps["raw"] = raw
        return reps, raw

    global GEOMETRY
    res = {"n_set_a": int(len(rows_a)), "n_set_b": int(len(rows_b)), "n_batches_b": int(len(big))}
    reps_a, raw_a = cached("setA", rows_a)
    reps_a["raw"] = raw_a
    tissue = prof.tissue.to_numpy()[rows_a]
    sex = prof.sex.to_numpy()[rows_a]
    sex = np.where(np.isin(sex, ["M", "F"]), sex, None)
    reps_b, raw_b = cached("setB", rows_b)
    reps_b["raw"] = raw_b
    batch = prof.group.astype(str).to_numpy()[rows_b]
    sex_b = prof.sex.to_numpy()[rows_b]
    age = lab.age_meta.to_numpy(float)[rows_b]
    try:
        cb = combat(raw_b, batch)
        reps_b["combat"] = cb
        # sva leaves a CpG unadjusted when it is constant within any batch; CpGs missing from a dataset's array are filled with
        # training means and are constant there. ComBat is also reported on the CpGs it adjusts, next to raw beta values on them.
        # exact criterion, as sva (zero variance within a batch); a float32 SD threshold lets constant CpGs through by rounding
        adj = np.all([np.ptp(raw_b[batch == g], axis=0) > 0 for g in np.unique(batch)], axis=0)
        reps_b["combat_adjusted_cpgs"] = combat(raw_b[:, adj], batch)
        reps_b["raw_adjusted_cpgs"] = raw_b[:, adj]
        res["combat_note"] = {"cpgs_adjusted": int(adj.sum()), "cpgs_constant_within_a_dataset": int((~adj).sum()),
                              "data_exchange": "corrected 18 Oct 2026 (sample-major file for R matrix(nrow = CpGs))"}
    except Exception as e:  # report and continue without ComBat
        log(f"ComBat failed: {e}")
        res["combat_error"] = str(e)
    if a.only_combat:
        old = json.loads(Path(a.out).read_text())
        reps_b = {k: v for k, v in reps_b.items() if k in ("combat", "combat_adjusted_cpgs", "raw_adjusted_cpgs")}
    meta = pd.DataFrame({"batch": batch})
    for GEOMETRY in ("natural", "standardized"):
        ra = {}
        for name, Z in ({} if a.only_combat else reps_a).items():
            P = pcs(Z)
            ra[name] = {"tissue_knn_bacc": knn_bacc(Z, tissue), "sex_knn_bacc": knn_bacc(Z, sex),
                        "tissue_silhouette": float(silhouette_score(P, tissue, metric="cosine")),
                        "tissue_energy_distance": energy_distance(P, tissue)}
            log(f"A {GEOMETRY} {name}: {ra[name]}")
        rb = {}
        hz = pcs(raw_b)
        ho = harmonypy.run_harmony(hz, meta, ["batch"], random_state=0, verbose=False)
        zc = np.asarray(ho.Z_corr)
        reps_b_g = reps_b if a.only_combat else dict(reps_b, harmony_pca50=zc.T if zc.shape[0] == hz.shape[1] and zc.shape[1] == hz.shape[0] else zc)
        for name, Z in reps_b_g.items():
            P = Z if name.endswith("pca50") else pcs(Z)
            ilisi = compute_lisi(P, meta, ["batch"])[:, 0]
            age_pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), P, age, cv=KFold(5, shuffle=True, random_state=0))
            rb[name] = {"batch_knn_bacc": knn_bacc(Z, batch), "batch_ilisi": float(np.median(ilisi)),
                        "sex_knn_bacc": knn_bacc(Z, sex_b), "age_r2_ridge": float(r2_score(age, age_pred)),
                        "batch_silhouette": float(silhouette_score(P, batch, metric="cosine"))}
            log(f"B {GEOMETRY} {name}: {rb[name]}")
        if a.only_combat:
            old[GEOMETRY]["set_b"].update(rb)
            old["combat_note"] = res["combat_note"]
            write_json(Path(a.out), old)
            continue
        res[GEOMETRY] = {"set_a": ra, "set_b": rb}
        write_json(Path(a.out), res)
    if a.only_combat:
        return
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
