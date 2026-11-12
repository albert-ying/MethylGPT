#!/usr/bin/env python3
"""Label efficiency: GS incident disease with fewer labelled subjects (development only).

Outer folds are those of the locked nested protocol. In each outer fold the fitting subjects
are subsampled (stratified by event) to a fraction f; raw-CpG and MethylGPT-2 embedding
elastic-net Cox models (same GPU implementation, settings and inner-CV selection for both)
and an age+sex Cox are fitted on the subsample and scored on the full held-out fold.
f = 1.0 checks the GPU implementation against the locked sksurv results. GS was never used to
train MethylGPT-2, so the embedding carries no outcome information from these subjects.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from methylgpt2_lib import cuda_preinit, ARTIFACTS, log, require, write_json
from gpu_coxnet_2026_10_01 import cindex, select_and_fit  # noqa: E402

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"


def standardise(F_fit, F_all, dev="cuda:1"):
    """Fitting-subject mean imputation and scaling (torch; numpy is very slow on this server)."""
    A = torch.as_tensor(F_fit, device=dev, dtype=torch.float32)
    B = torch.as_tensor(F_all, device=dev, dtype=torch.float32)
    fin = torch.isfinite(A)
    mu = torch.where(fin, A, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    A = torch.where(fin, A, mu); B = torch.where(torch.isfinite(B), B, mu)
    sd = A.std(0); sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
    return ((A - mu) / sd).cpu().numpy(), ((B - mu) / sd).cpu().numpy()


def pooled(ev, du, risk, folds):
    from sksurv.metrics import concordance_index_censored
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], risk[s])
        t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="1.0,0.25,0.1,0.05")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--features", default=str(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mgpt2_embed.embedding.npz"))
    ap.add_argument("--bio-features", default=str(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mt_bio.embedding.npz"))
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/scenario_label_efficiency_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mpath, allow_pickle=False) as z:
        RAW = z["matrix"].astype(np.float32)
    fz = np.load(require(Path(a.features)), allow_pickle=False)
    assert (fz["dnam_id"] == ids).all()
    EMB = fz["embedding"].astype(np.float32)
    bz = np.load(require(Path(a.bio_features)), allow_pickle=False)
    assert (bz["dnam_id"] == ids).all()
    BIO = bz["embedding"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    preds = pd.read_parquet(Path(a.mgpt2_run) / "private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = a.endpoints.split(",") if a.endpoints else sorted(preds.endpoint.unique())
    res = {"settings": vars(a), "results": []}
    for frac in [float(x) for x in a.fractions.split(",")]:
        reps = 1 if frac == 1.0 else a.repeats
        for rep in range(reps):
            for ep in endpoints:
                pe = preds[preds.endpoint == ep].reset_index(drop=True)
                idx = np.array([pos[s] for s in pe.dnam_id.astype(str)])
                ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
                cov = pe[["age", "sex"]].to_numpy(float)
                risks = {k: np.empty(len(pe)) for k in ("age_sex", "raw_cpg", "methylgpt2", "biomarkers", "ensemble")}
                folds = []
                n_fit_used = []
                for f in sorted(pe.outer_fold.unique()):
                    held = np.flatnonzero(pe.outer_fold.to_numpy() == f)
                    fit = np.flatnonzero(pe.outer_fold.to_numpy() != f)
                    folds.append(held)
                    if frac < 1.0:
                        rng = np.random.default_rng(1000 * rep + 17 * f + int(frac * 1000))
                        keep = []
                        for e_val in (True, False):
                            grp = fit[ev[fit] == e_val]
                            keep.append(rng.choice(grp, max(2, int(round(frac * len(grp)))), replace=False))
                        fit = np.sort(np.concatenate(keep))
                    n_fit_used.append(int(len(fit)))
                    c_fit, c_all = standardise(cov[fit], cov, dev)
                    cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit(c_fit, Surv.from_arrays(ev[fit], du[fit]))
                    risks["age_sex"][held] = c_all[held] @ cx.coef_
                    for name, F in (("raw_cpg", RAW), ("methylgpt2", EMB), ("biomarkers", BIO)):
                        F_fit, F_all = standardise(F[idx[fit]], F[idx], dev)
                        Xf = np.hstack([c_fit, F_fit]); Xa = np.hstack([c_all, F_all])
                        w, alpha, k = select_and_fit(Xf, du[fit], ev[fit], seed=20260928 + f, dev=dev, iters=a.iters)
                        risks[name][held] = Xa[held] @ w
                    z_ = lambda v: (v - v.mean()) / v.std()
                    risks["ensemble"][held] = z_(risks["methylgpt2"][held]) + z_(risks["biomarkers"][held])
                row = {"fraction": frac, "repeat": rep, "endpoint": ep, "events_total": int(ev.sum()),
                       "mean_fit_subjects": float(np.mean(n_fit_used))}
                for k_, r_ in risks.items():
                    row[k_] = float(pooled(ev, du, r_, folds))
                res["results"].append(row)
                log(f"f={frac} rep={rep} {ep}: age+sex {row['age_sex']:.4f} raw {row['raw_cpg']:.4f} embed {row['methylgpt2']:.4f} bio {row['biomarkers']:.4f} ens {row['ensemble']:.4f} (fit n~{row['mean_fit_subjects']:.0f})")
                write_json(Path(a.out), res)
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
