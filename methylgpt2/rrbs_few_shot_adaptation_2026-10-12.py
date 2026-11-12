#!/usr/bin/env python3
"""Few labelled samples on a new platform: pretrained MethylGPT against models trained on the cohort itself.

Cohort: whole-blood RRBS, GSE194154, 110 samples aged 18-88 (not used for any training). Question: when only n
samples of a sequencing cohort have ages, is it better to adapt the pretrained model or to train a model on the
cohort? For n = 10, 20, 40, 60 and 88 labelled samples (200 random draws each, seed 20261012; the remaining samples
are the test set, identical for all methods):

Pretrained, adapted with the n labelled samples
  MethylGPT recalibrated      released model, age from the measured vocabulary CpGs; a + b * prediction fitted on the n samples
  MethylGPT factor scores     ridge on the frozen 1,536 factor scores (natural scale), penalty by leave-one-out
  array elastic net, imputed  the corpus-trained elastic net with missing CpGs imputed by MethylGPT's factor model,
                              recalibrated in the same way (stored predictions of rrbs_factor_imputed_comparators_2026-10-09)
Trained on the n labelled samples only (no pretraining)
  ridge, genome-wide CpGs     every CpG with >= 10 reads in >= 90% of samples (rrbs_genomewide_matrix_2026-10-12.py)
  ridge, vocabulary CpGs      the vocabulary CpGs with >= 10 reads in >= 90% of samples (the inputs MethylGPT sees)
  elastic net, vocabulary     scikit-learn ElasticNetCV (l1 ratio 0.5, fivefold inner cross-validation), same CpGs
  elastic net, screened       the 10,000 genome-wide CpGs most correlated with age in the n samples, then ElasticNetCV
  mean age                    the mean age of the n samples
Ridge penalties are chosen by closed-form leave-one-out error within the n samples; missing values take the mean of
the n samples; features are centred on the n samples. Reported: MAE, MedAE and Pearson r on the test samples, mean
and central 95% range over the 200 draws, and the paired difference of MethylGPT recalibrated from each method.

Inputs: artifacts/external/GSE194154/{betas_cov10.parquet,samples.csv,betas_genomewide_cov10_2026-10-12.npz},
        artifacts/eval/rrbs_gse194154_2026-10-05.csv, artifacts/eval/rrbs_factor_imputed_comparators_2026-10-09.csv
Outputs: artifacts/eval/rrbs_few_shot_adaptation_2026-10-12.{json,csv}
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
D = ARTIFACTS / "external/GSE194154"
SIZES = [10, 20, 40, 60, 88]
PRETRAINED = ["MethylGPT recalibrated", "MethylGPT factor scores, ridge", "array elastic net with factor-model imputation, recalibrated"]
SCRATCH = ["ridge, genome-wide CpGs", "ridge, vocabulary CpGs", "elastic net, vocabulary CpGs", "elastic net, screened genome-wide CpGs", "mean age"]


def recal(p, y, tr, te):
    b, a = np.polyfit(p[tr], y[tr], 1)
    return a + b * p[te]


def ridge_loo(Ktr, Kte, y):
    """Kernel ridge with the penalty chosen by leave-one-out error. Ktr, Kte: centred linear kernels."""
    ym = y.mean()
    w, U = torch.linalg.eigh(Ktr)
    w = w.clamp_min(0)
    Uy = U.T @ (y - ym)
    best = None
    for lam in float(w.mean()) * np.logspace(-4, 3, 22):
        d = w / (w + lam)
        fit = U @ (d * Uy)
        h = (U * U) @ d
        loo = float((((y - ym) - fit) / (1 - h).clamp_min(1e-8)).abs().mean())
        if best is None or loo < best[0]:
            best = (loo, lam)
    alpha = U @ (Uy / (w + best[1]))
    return (Kte @ alpha + ym).cpu().numpy()


def centred_kernels(X, tr, te, dev):
    """Linear kernels after filling missing values with, and centring on, the training means. X: tensor on dev."""
    A = X[torch.as_tensor(tr, device=dev)].double()
    fin = torch.isfinite(A)
    mu = torch.where(fin, A, torch.zeros((), device=dev, dtype=torch.float64)).sum(0) / fin.sum(0).clamp_min(1)
    A = torch.where(fin, A, mu) - mu
    B = X[torch.as_tensor(te, device=dev)].double()
    B = torch.where(torch.isfinite(B), B, mu) - mu
    return A @ A.T, B @ A.T


def enet(Xtr, ytr, Xte):
    from sklearn.linear_model import ElasticNetCV
    mu = np.nanmean(Xtr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.5)
    A, B = np.where(np.isfinite(Xtr), Xtr, mu) - mu, np.where(np.isfinite(Xte), Xte, mu) - mu
    sd = A.std(0)
    keep = sd > 1e-6
    A, B = A[:, keep] / sd[keep], B[:, keep] / sd[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = ElasticNetCV(l1_ratio=0.5, n_alphas=30, cv=min(5, len(ytr)), max_iter=2000, tol=1e-3).fit(A, ytr)
    return m.predict(B)


def enet_screened(Xg, y, tr, te, top=10000):
    A = Xg[tr]
    mu = np.nanmean(A, 0)
    Ac = np.where(np.isfinite(A), A, mu) - mu
    yc = y[tr] - y[tr].mean()
    r = np.abs(Ac.T @ yc) / (np.sqrt((Ac * Ac).sum(0)) * np.sqrt((yc * yc).sum()) + 1e-12)
    idx = np.argsort(-r)[:top]
    return enet(Xg[tr][:, idx], y[tr], Xg[te][:, idx])


def scores(p, y):
    e = np.abs(p - y)
    return float(e.mean()), float(np.median(e)), float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--scatter-only", action="store_true",
                    help="only write each sample's mean held-out prediction over the draws with 20 labelled samples (same draws), for the figure")
    a = ap.parse_args()
    cuda_preinit()
    dev = torch.device(a.device)
    try:
        betas = pd.read_parquet(require(D / "betas_cov10.parquet"))
        samples = pd.read_csv(require(D / "samples.csv")).set_index("gsm").loc[betas.index]
        st = pd.read_csv(require(ARTIFACTS / "eval/rrbs_gse194154_2026-10-05.csv")).set_index("gsm").loc[betas.index]
        fi = pd.read_csv(require(ARTIFACTS / "eval/rrbs_factor_imputed_comparators_2026-10-09.csv")).set_index("gsm").loc[betas.index]
        gw = np.load(require(D / "betas_genomewide_cov10_2026-10-12.npz"), allow_pickle=False)
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    assert list(gw["samples"]) == list(betas.index)
    y = samples["age"].astype(float).to_numpy()
    n = len(y)
    Xg = gw["beta"]
    Xv = betas.to_numpy(np.float32)
    Xv = Xv[:, np.isfinite(Xv).mean(0) >= 0.9]
    log(f"{n} samples; genome-wide CpGs {Xg.shape[1]}; vocabulary CpGs covered in >= 90% of samples {Xv.shape[1]}")
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(ARTIFACTS / "release/methylgpt2-1.1", device=a.device)
    S = model.factor_scores(betas).to_numpy(np.float32)
    p_mg = st["MethylGPT"].to_numpy(float)
    assert np.allclose(model.predict_age(betas).to_numpy(float).ravel(), p_mg, atol=0.05), "stored MethylGPT predictions differ from the released model"
    p_en = fi["ElasticNet (factor-model imputation)"].to_numpy(float)
    del model
    torch.cuda.empty_cache()

    rng = np.random.default_rng(20261012)
    splits = [(m, d, np.sort(perm[:m]), np.sort(perm[m:])) for m in SIZES for d in range(a.draws) for perm in [rng.permutation(n)]]
    yt = torch.from_numpy(y).to(dev, torch.float64)
    Xt = {"MethylGPT factor scores, ridge": torch.from_numpy(S).to(dev), "ridge, genome-wide CpGs": torch.from_numpy(Xg).to(dev),
          "ridge, vocabulary CpGs": torch.from_numpy(Xv).to(dev)}
    if a.scatter_only:
        acc = {k: np.zeros(n) for k in ("MethylGPT recalibrated", "ridge, genome-wide CpGs")}
        cnt = np.zeros(n)
        for m, d, tr, te in splits:
            if m != 20:
                continue
            acc["MethylGPT recalibrated"][te] += recal(p_mg, y, tr, te)
            Ktr, Kte = centred_kernels(Xt["ridge, genome-wide CpGs"], tr, te, dev)
            acc["ridge, genome-wide CpGs"][te] += ridge_loo(Ktr, Kte, yt[tr])
            cnt[te] += 1
        np.savez(ARTIFACTS / "eval/rrbs_few_shot_adaptation_2026-10-12.heldout_mean_predictions_n20.npz", age=y, times_held_out=cnt,
                 **{k: v / cnt for k, v in acc.items()})
        log(f"mean held-out predictions with 20 labelled samples written; each sample held out {int(cnt.min())}-{int(cnt.max())} times")
        return
    rows = []
    for k, (m, d, tr, te) in enumerate(splits):
        out = {"MethylGPT recalibrated": recal(p_mg, y, tr, te),
               "array elastic net with factor-model imputation, recalibrated": recal(p_en, y, tr, te),
               "mean age": np.full(len(te), y[tr].mean())}
        for name, X in Xt.items():
            Ktr, Kte = centred_kernels(X, tr, te, dev)
            out[name] = ridge_loo(Ktr, Kte, yt[tr])
        for name, p in out.items():
            mae, med, r = scores(p, y[te])
            rows.append({"n_labelled": m, "draw": d, "method": name, "mae": mae, "medae": med, "pearson_r": r})
        if k % 100 == 0:
            log(f"split {k}/{len(splits)}")
    pd.DataFrame(rows).to_csv(ARTIFACTS / "eval/rrbs_few_shot_adaptation_2026-10-12.csv", index=False)
    log("elastic nets trained on the cohort (parallel over draws)")
    for name, jobs in (("elastic net, vocabulary CpGs", [delayed(enet)(Xv[tr], y[tr], Xv[te]) for _, _, tr, te in splits]),
                       ("elastic net, screened genome-wide CpGs", [delayed(enet_screened)(Xg, y, tr, te) for _, _, tr, te in splits])):
        ps = Parallel(n_jobs=a.jobs)(jobs)
        for (m, d, tr, te), p in zip(splits, ps):
            mae, med, r = scores(p, y[te])
            rows.append({"n_labelled": m, "draw": d, "method": name, "mae": mae, "medae": med, "pearson_r": r})
        log(f"{name} done")
    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS / "eval/rrbs_few_shot_adaptation_2026-10-12.csv", index=False)
    e0 = np.abs(p_mg - y)
    res = {"cohort": "GSE194154 whole-blood RRBS", "samples": int(n), "draws": a.draws, "genome_wide_cpgs": int(Xg.shape[1]),
           "vocabulary_cpgs_covered": int(Xv.shape[1]), "pretrained_methods": PRETRAINED, "cohort_trained_methods": SCRATCH,
           "MethylGPT without labels": {"mae": float(e0.mean()), "medae": float(np.median(e0)), "pearson_r": float(np.corrcoef(p_mg, y)[0, 1])},
           "by_n": {}}
    piv = {mtr: df.pivot_table(index=["n_labelled", "draw"], columns="method", values=mtr) for mtr in ("mae", "medae", "pearson_r")}
    for m in SIZES:
        blk = {}
        for name in PRETRAINED + SCRATCH:
            v = {mtr: piv[mtr].loc[m, name].to_numpy() for mtr in piv}
            dd = piv["mae"].loc[m, "MethylGPT recalibrated"].to_numpy() - v["mae"]
            blk[name] = {"mae_mean": float(v["mae"].mean()), "mae_range95": [float(np.percentile(v["mae"], 2.5)), float(np.percentile(v["mae"], 97.5))],
                         "medae_mean": float(v["medae"].mean()), "pearson_r_mean": float(np.nanmean(v["pearson_r"])),
                         "MethylGPT_recalibrated_minus_this_mae": {"mean": float(dd.mean()), "range95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                                                   "draws_MethylGPT_better": int((dd < 0).sum())}}
        res["by_n"][str(m)] = blk
        log(f"n = {m}: " + "; ".join(f"{k} {v['mae_mean']:.2f}" for k, v in blk.items()))
    write_json(ARTIFACTS / "eval/rrbs_few_shot_adaptation_2026-10-12.json", res)


if __name__ == "__main__":
    main()
