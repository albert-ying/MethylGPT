#!/usr/bin/env python3
"""Frozen-representation probes of the pretraining experiment (age prediction, public data).

Question: when the encoder is frozen and only a linear model is fitted, with few or many labelled profiles, does
pretraining on the unlabelled corpus help? One linear model (ridge on the Horvath-transformed age) is fitted on
frozen representations from three versions of the same architecture, none of which has seen a label:
  pretrained    factor model fitted to the 145,700 unlabelled training profiles + transformer after masked-value
                pretraining (artifacts/runs/stopped_cf_dense_2026-09-30/best.pt)
  factor_only   the same factor model + transformer at its random initial weights (seed 20261001)
  none          random orthonormal factor loadings (training means and variances kept;
                artifacts/ppca_core_random_2026-10-03.npz) + transformer at its random initial weights
Feature blocks: the 2,048-dimensional embedding, its 1,536 factor scores and its 512 transformer dimensions, each
standardized per feature; and the factor scores on their natural scale (centred only), which is what the package
function factor_scores() returns and what the representation benchmarks use.
Comparators without any representation: elastic net on the 49,156 raw CpGs (same settings as
baseline_corpus_age_2026-10-01.py: l1 ratio 0.5, FISTA, 11-point path, 1,500 iterations per point, per-CpG
standardization, missing values filled with the fitting means) and ridge on the raw CpGs on their natural scale
(centred only, same filling; subsets up to 10,000 profiles).

Label subsets: 100, 300, 1,000, 3,000, 10,000 and all 41,845 training profiles with a clean age; label_subset()
of methylgpt_lib gives nested subsets for one seed (five seeds; one for all profiles) and holds out every fifth
profile of a subset to select the penalty. No other labels are used: the AltumAge validation set is not touched.
Scoring: AltumAge test set (complete, 90% of CpGs removed, 27K-array CpGs; masks of the age scenarios) and the
5,811 held-out profiles.

Outputs: artifacts/label_efficiency_2026-10-12/frozen_emb_<arm>.npz (cached embeddings),
         artifacts/eval/pretraining_frozen_probe_2026-10-12.{json,csv} and .predictions.npz
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import (ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, age_scores, cuda_preinit, label_subset, load_altumage_split, log,
                            require, write_json)
from methylgpt_model import embed_batch, load_checkpoint

OUT = ARTIFACTS / "label_efficiency_2026-10-12"
CKPT = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
RANDOM_CORE = ARTIFACTS / "ppca_core_random_2026-10-03.npz"
ARMS = {"pretrained": dict(random_init=False, core=None), "factor_only": dict(random_init=True, core=None),
        "none": dict(random_init=True, core=str(RANDOM_CORE))}
SIZES = [100, 300, 1000, 3000, 10000, 0]   # 0 = all profiles with a clean age
SETS = ["altumage_test|all", "altumage_test|random_90", "altumage_test|hm27", "heldout|all"]
RANK = 1536


def degrade(X, scen, plat):
    X = X.copy()
    if scen == "random_90":
        X[np.random.default_rng(20261001 + 90).random(X.shape) < 0.9] = np.nan
    elif scen != "all":
        X[:, ~plat[scen]] = np.nan
    return X


@torch.no_grad()
def embed(model, core, X, dev, batch=128):
    out = np.empty((len(X), core.rank + model.latents.shape[1]), np.float32)
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dev)
        out[i:i + batch] = embed_batch(model, core, x, torch.isfinite(x)).float().cpu().numpy()
    return out


def ridge_probe(Zf, tf, Zs, ys_years, ym, ysd, dev, standardize=True):
    """Ridge path by SVD on the fitting part; the penalty with the lowest MedAE on the selection part is kept.
    Features are centred and, if standardize, scaled per feature; missing values take the fitting mean."""
    A = torch.from_numpy(Zf).to(dev, torch.float64)
    fin = torch.isfinite(A)
    mu = torch.where(fin, A, torch.zeros((), device=dev, dtype=torch.float64)).sum(0) / fin.sum(0).clamp_min(1)
    A = torch.where(fin, A, mu)
    del fin
    sd = A.std(0).clamp_min(1e-6) if standardize else torch.ones_like(mu)
    A = (A - mu) / sd
    U, S, Vt = torch.linalg.svd(A, full_matrices=False)
    del A
    Ut = U.T @ torch.from_numpy(tf).to(dev, torch.float64)
    B = torch.from_numpy(Zs).to(dev, torch.float64)
    B = (torch.where(torch.isfinite(B), B, mu) - mu) / sd
    best = None
    for lam in float((S * S).mean()) * np.logspace(-6, 2, 25):
        w = Vt.T @ (S / (S * S + lam) * Ut)
        med = float(np.median(np.abs(Finv((B @ w).cpu().numpy() * ysd + ym) - ys_years)))
        if best is None or med < best[0]:
            best = (med, float(lam), w)
    return best[2], mu, sd, {"lambda": best[1], "selection_medae": best[0]}


def enet_probe(Xf, tf, Xs, ys_years, ym, ysd, dev, iters=1500):
    X = torch.from_numpy(Xf).to(dev)
    fin = torch.isfinite(X)
    mu = torch.where(fin, X, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    X = torch.where(fin, X, mu)
    del fin
    sd = X.std(0).clamp_min(1e-4)
    X.sub_(mu).div_(sd)
    t = torch.from_numpy(tf.astype(np.float32)).to(dev)
    S = torch.from_numpy(Xs).to(dev)
    S = (torch.where(torch.isfinite(S), S, mu) - mu) / sd
    n = X.shape[0]
    g = torch.Generator(device=dev).manual_seed(20261012)
    v = torch.randn(X.shape[1], device=dev, generator=g)
    for _ in range(40):
        v = X.T @ (X @ v)
        v /= v.norm()
    L0 = float((X @ v).norm() ** 2 / n)
    l1 = 0.5
    amax = float((X.T @ t).abs().max() / (n * l1))
    w = torch.zeros(X.shape[1], device=dev)
    best = None
    for alpha in amax * np.logspace(-1.0, -3.5, 11):
        step = 1.0 / (L0 + alpha * (1 - l1))
        yv, tk = w.clone(), 1.0
        for _ in range(iters):
            gr = X.T @ (X @ yv - t) / n + alpha * (1 - l1) * yv
            u = yv - step * gr
            wn = torch.sign(u) * torch.clamp(u.abs() - step * alpha * l1, min=0)
            tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            yv = wn + ((tk - 1) / tk1) * (wn - w)
            w, tk = wn, tk1
        med = float(np.median(np.abs(Finv((S @ w).cpu().numpy() * ysd + ym) - ys_years)))
        if best is None or med < best[0]:
            best = (med, float(alpha), w.clone())
    return best[2], mu, sd, {"alpha": best[1], "nonzero": int((best[2] != 0).sum()), "selection_medae": best[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--skip-enet", action="store_true")
    a = ap.parse_args()
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        corpus = Corpus()
        lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
        plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
        _, Xte, yte = load_altumage_split("test")
        require(CKPT), require(RANDOM_CORE)
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    prof = corpus.profiles
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    age, age_meta = lab.age_clean.to_numpy(float), lab.age_meta.to_numpy(float)
    pool_mask = (prof.split == "train").to_numpy() & np.isfinite(age)
    pool = np.flatnonzero(pool_mask)
    ho_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    y_pool, y_sets = age[pool], {"altumage_test": yte, "heldout": age_meta[ho_rows]}
    log(f"pool {len(pool)} labelled training profiles; AltumAge test {len(yte)}; held-out {len(ho_rows)}")
    Xsets = {"altumage_test|all": Xte, "altumage_test|random_90": degrade(Xte, "random_90", plat),
             "altumage_test|hm27": degrade(Xte, "hm27", plat), "heldout|all": corpus.get(ho_rows)}

    # ---- frozen embeddings of the three arms (cached) ----
    emb = {}
    for arm, cfg in ARMS.items():
        f = OUT / f"frozen_emb_{arm}.npz"
        if not f.exists():
            torch.manual_seed(20261001)
            model, core, _ = load_checkpoint(CKPT, cfg["core"], dev, random_init=cfg["random_init"])
            model.eval()
            d = {"pool": np.concatenate([embed(model, core, corpus.get(pool[i:i + 4096]), dev) for i in range(0, len(pool), 4096)])}
            for k, X in Xsets.items():
                d[k] = embed(model, core, X, dev)
            d["lam"] = core.lam.float().cpu().numpy()
            np.savez(f, **d)
            del model, core
            torch.cuda.empty_cache()
            log(f"embeddings of arm {arm} written ({time.time() - t0:.0f} s)")
        emb[arm] = dict(np.load(f))
    assert np.allclose(emb["pretrained"]["pool"][:, :RANK], emb["factor_only"]["pool"][:, :RANK], atol=1e-4)

    blocks = {}
    for arm in ARMS:
        blocks[f"embedding|{arm}"] = (arm, slice(0, None))
        blocks[f"transformer_part|{arm}"] = (arm, slice(RANK, None))
    blocks["factor_scores|fitted"] = ("pretrained", slice(0, RANK))
    blocks["factor_scores|random"] = ("none", slice(0, RANK))

    Xpool = None
    if not a.skip_enet:
        Xpool = np.concatenate([corpus.get(pool[i:i + 8192]) for i in range(0, len(pool), 8192)])
    rows, preds = [], {}
    for n in SIZES:
        for seed in range(a.seeds if n else 1):
            fit, sel = label_subset(pool_mask, n or len(pool), seed)
            fi, si = np.searchsorted(pool, fit), np.searchsorted(pool, sel)
            tfy = F(y_pool[fi])
            ym, ysd = float(tfy.mean()), float(tfy.std())
            tf = (tfy - ym) / ysd
            tag = f"n{n or len(pool)}|s{seed}"

            def record(model, info, predict):
                for k in SETS:
                    p = predict(k)
                    preds[f"{model}|{tag}|{k}"] = p.astype(np.float32)
                    sc = age_scores(p, y_sets[k.split("|")[0]])
                    rows.append({"model": model, "n_labels": n or len(pool), "seed": seed, "n_fit": len(fit), "n_selection": len(sel),
                                 "set": k, "medae": sc["medae"], "mae": sc["mae"], "pearson_r": sc["pearson_r"], **info})

            def lin(T, mu, sd, w):
                T = torch.from_numpy(T).to(dev, torch.float64)
                return Finv((((torch.where(torch.isfinite(T), T, mu) - mu) / sd) @ w).cpu().numpy() * ysd + ym)

            for name, (arm, sl) in blocks.items():
                w, mu, sd, info = ridge_probe(emb[arm]["pool"][fi][:, sl], tf, emb[arm]["pool"][si][:, sl], y_pool[si], ym, ysd, dev)
                record(name, info, lambda k: lin(emb[arm][k][:, sl], mu, sd, w))
            for name, arm in (("factor_scores_natural|fitted", "pretrained"), ("factor_scores_natural|random", "none")):
                nat = lambda Z: Z[:, :RANK] * np.sqrt(emb[arm]["lam"])[None, :]
                w, mu, sd, info = ridge_probe(nat(emb[arm]["pool"][fi]), tf, nat(emb[arm]["pool"][si]), y_pool[si], ym, ysd, dev, standardize=False)
                record(name, info, lambda k: lin(nat(emb[arm][k]), mu, sd, w))
            if Xpool is not None and len(fit) <= 8000:
                w, mu, sd, info = ridge_probe(Xpool[fi], tf, Xpool[si], y_pool[si], ym, ysd, dev, standardize=False)
                record("ridge_raw_natural", info, lambda k: lin(Xsets[k], mu, sd, w))
                del w
                torch.cuda.empty_cache()
            if Xpool is not None:
                w, mu, sd, info = enet_probe(Xpool[fi], tf, Xpool[si], y_pool[si], ym, ysd, dev)

                def pen(k):
                    T = torch.from_numpy(Xsets[k]).to(dev)
                    return Finv((((torch.where(torch.isfinite(T), T, mu) - mu) / sd) @ w).cpu().numpy() * ysd + ym)
                record("elastic_net_raw", info, pen)
            r = {x["model"]: x["medae"] for x in rows if x["set"] == "altumage_test|all" and x["n_labels"] == (n or len(pool)) and x["seed"] == seed}
            log(f"{tag}: AltumAge test MedAE " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()) + f" ({time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(ARTIFACTS / "eval/pretraining_frozen_probe_2026-10-12.csv", index=False)

    df = pd.DataFrame(rows)
    summ = (df.groupby(["model", "n_labels", "set"]).agg(medae_mean=("medae", "mean"), medae_sd=("medae", "std"), medae_min=("medae", "min"),
                                                         medae_max=("medae", "max"), mae_mean=("mae", "mean"), subsets=("seed", "count")).reset_index())
    res = {"design": __doc__.split("Outputs:")[0].strip(), "pool": int(len(pool)), "seeds": a.seeds,
           "summary": summ.to_dict(orient="records"), "elapsed_s": time.time() - t0}
    write_json(ARTIFACTS / "eval/pretraining_frozen_probe_2026-10-12.json", res)
    np.savez_compressed(ARTIFACTS / "eval/pretraining_frozen_probe_2026-10-12.predictions.npz", **preds)
    log("done")


if __name__ == "__main__":
    main()
