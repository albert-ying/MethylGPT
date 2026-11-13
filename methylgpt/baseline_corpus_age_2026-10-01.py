#!/usr/bin/env python3
"""Linear age baselines trained on the same corpus profiles and ages as the aging heads.

Training: training-split profiles with metadata age in [0, 120]. Selection: AltumAge
validation MedAE. Scoring (once): AltumAge test and held-out test-split datasets.
  enet_raw     ElasticNet (l1_ratio 0.5, FISTA on GPU, warm-started path) on all 49,156 CpGs,
               train-mean imputation, per-CpG standardisation, Horvath-transformed age
  ridge_core   ridge on the MethylGPT-2 core scores (1,536 standardised factor scores)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt_lib import (cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, PPCACore, age_scores, load_altumage_split,
                            log, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--core", default=str(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(20261001)
    dev = torch.device(a.device)
    corpus = Corpus()
    prof = corpus.profiles
    import pandas as pd
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    age = lab.age_clean.to_numpy(float)
    age_meta, age_flag = lab.age_meta.to_numpy(float), lab.age_group_flag.to_numpy(bool)
    rows = np.flatnonzero((prof.split == "train").to_numpy() & np.isfinite(age))
    test_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    log(f"training profiles with age {len(rows)}; held-out test-split profiles with age {len(test_rows)}")
    X = torch.empty((len(rows), corpus.beta.shape[1]), dtype=torch.float32, device=dev)
    for i in range(0, len(rows), 8192):
        X[i:i + 8192] = torch.from_numpy(corpus.get(rows[i:i + 8192])).to(dev)
    fin = torch.isfinite(X)
    mu = torch.where(fin, X, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    X = torch.where(fin, X, mu)
    del fin
    sd = X.std(0).clamp_min(1e-4)
    X.sub_(mu).div_(sd)
    y = F(age[np.sort(rows)])
    ym, ys = y.mean(), y.std()
    t = torch.from_numpy(((y - ym) / ys).astype(np.float32)).to(dev)

    def prep(Xn):
        T = torch.from_numpy(Xn).to(dev)
        T = torch.where(torch.isfinite(T), T, mu)
        return (T - mu) / sd

    _, Xva, yva = load_altumage_split("valid")
    _, Xte, yte = load_altumage_split("test")
    Va, Te = prep(Xva), prep(Xte)
    Ho = prep(corpus.get(test_rows))
    y_ho = age_meta[test_rows]
    keep_ho = ~age_flag[test_rows]
    n = X.shape[0]
    v = torch.randn(X.shape[1], device=dev)
    for _ in range(40):
        v = X.T @ (X @ v); v /= v.norm()
    L0 = float((X @ v).norm() ** 2 / n)
    l1 = 0.5
    amax = float((X.T @ t).abs().max() / (n * l1))
    alphas = amax * np.logspace(-1.0, -3.5, 11)
    w = torch.zeros(X.shape[1], device=dev)
    res = {"n_train": int(n), "alphas": alphas.tolist(), "path": [], "models": {}}
    best = None
    for alpha in alphas:
        step = 1.0 / (L0 + alpha * (1 - l1))
        yv, tk = w.clone(), 1.0
        for it in range(a.iters):
            g = X.T @ (X @ yv - t) / n + alpha * (1 - l1) * yv
            u = yv - step * g
            wn = torch.sign(u) * torch.clamp(u.abs() - step * alpha * l1, min=0)
            tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            yv = wn + ((tk - 1) / tk1) * (wn - w)
            w, tk = wn, tk1
        pv = Finv((Va @ w).cpu().numpy() * ys + ym)
        sc = age_scores(pv, yva)
        res["path"].append({"alpha": float(alpha), "nonzero": int((w != 0).sum()), "valid": sc})
        log(f"alpha {alpha:.4g}: nonzero {int((w != 0).sum())} valid MedAE {sc['medae']:.3f}")
        if best is None or sc["medae"] < best[1]["medae"]:
            best = (float(alpha), sc, w.clone())
    alpha, sc, w = best
    res["models"]["enet_raw"] = {"alpha": alpha, "nonzero": int((w != 0).sum()), "valid": sc,
                                 "selected_at_grid_edge": alpha in (alphas[0], alphas[-1]),
                                 "altumage_test": age_scores(Finv((Te @ w).cpu().numpy() * ys + ym), yte),
                                 "heldout_datasets_all": age_scores(Finv((Ho @ w).cpu().numpy() * ys + ym), y_ho),
                                 "heldout_datasets_unflagged": age_scores(Finv((Ho @ w).cpu().numpy() * ys + ym)[keep_ho], y_ho[keep_ho])}
    log(f"enet_raw: {res['models']['enet_raw']}")
    np.savez(Path(a.out).with_suffix(".enet_model.npz"), w=w.cpu().numpy(), mu=mu.cpu().numpy(), sd=sd.cpu().numpy(),
             ym=ym, ys=ys, alpha=alpha)
    del X
    torch.cuda.empty_cache()
    core = PPCACore.load(Path(a.core), rank=1536).to(dev)

    def core_scores(Xn):
        out = []
        for i in range(0, len(Xn), 64):
            x = torch.from_numpy(Xn[i:i + 64]).to(dev)
            out.append((core.posterior(x, torch.isfinite(x).float()) / torch.sqrt(core.lam)).cpu().numpy())
        return np.concatenate(out)

    S = core_scores(corpus.get(rows)); Sva, Ste, Sho = core_scores(Xva), core_scores(Xte), core_scores(corpus.get(test_rows))
    m, s_ = S.mean(0), S.std(0) + 1e-8
    A = (S - m) / s_
    G = A.T @ A
    bestr = None
    for lam in np.logspace(-1, 5, 13):
        wr = np.linalg.solve(G + lam * np.eye(G.shape[0]), A.T @ ((y - ym) / ys))
        scv = age_scores(Finv(((Sva - m) / s_) @ wr * ys + ym), yva)
        if bestr is None or scv["medae"] < bestr[1]["medae"]:
            bestr = (lam, scv, wr)
    lam, scv, wr = bestr
    res["models"]["ridge_core"] = {"lambda": float(lam), "valid": scv,
                                   "altumage_test": age_scores(Finv(((Ste - m) / s_) @ wr * ys + ym), yte),
                                   "heldout_datasets_all": age_scores(Finv(((Sho - m) / s_) @ wr * ys + ym), y_ho),
                                   "heldout_datasets_unflagged": age_scores(Finv(((Sho - m) / s_) @ wr * ys + ym)[keep_ho], y_ho[keep_ho])}
    log(f"ridge_core: {res['models']['ridge_core']}")
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
