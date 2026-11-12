#!/usr/bin/env python3
"""MethylGPT-2 on the fixed GSE308517 imputation protocol (run once, after training).

Uses the stored mask/fold plan and the 136 x 35,676 overlap matrix from
results/prospective_imputation_2026-09-30 and results/geo308517_access_2026-09-29 of the
publication checkout. CpGs of the model vocabulary absent from GSE308517 are unobserved.
Metric: per-sample MAE over masked entries, averaged over the 136 samples and the three
mask seeds, as in the fixed protocol. Variants:
  methylgpt2 / ppca_core            zero-shot (corpus mean and loadings)
  methylgpt2_foldmean / ppca_core_foldmean
                                    the same model with its centring mean replaced by the
                                    fitting-fold CpG mean (the fold-mean baseline's mean)
The stored MLM, fold-mean and donor-kNN predictions are rescored as a check.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, N_CPG, log, require, write_json
from methylgpt2_model import load_checkpoint

DEV_RESULTS = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")


@torch.no_grad()
def predict(model, core, x, ctx, tgt):
    s = core.posterior(x, ctx.float())
    with torch.autocast("cuda", enabled=False):
        xs, c, e = model.inputs(x, ctx, s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode(xs, c, e, ctx, s)
    pred, base = model.predict_targets(z, s, tgt)
    return pred[0].cpu().numpy(), base[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--core", default=None, help="defaults to the core recorded in the checkpoint")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    plan = np.load(require(DEV_RESULTS / "prospective_imputation_2026-09-30/mask_fold_plan.npz"))
    ov = np.load(require(DEV_RESULTS / "geo308517_access_2026-09-29/GSE308517_model_probe_overlap_2026-09-30.npz"))
    beta, vidx, folds = ov["beta_values"], ov["overlap_vocab_indices"], plan["folds"]
    assert (plan["sample_ids"] == ov["sample_ids"]).all() and beta.shape == (136, 35676)
    full = np.full((136, N_CPG), np.nan, np.float32)
    full[:, vidx] = beta
    model, core, ck = load_checkpoint(Path(a.ckpt), a.core, dev)
    corpus_mu = core.mu.cpu().numpy().copy()
    fold_mu = {}
    for f in range(5):
        fit = folds != f
        fin = np.isfinite(beta[fit])
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(fin, beta[fit], 0).sum(0) / fin.sum(0)
        mu_f = corpus_mu.copy()
        mu_f[vidx] = np.where(np.isfinite(m), m, corpus_mu[vidx])
        fold_mu[f] = mu_f
    res = {"checkpoint": a.ckpt, "checkpoint_step": int(ck["step"]), "conditions": {}}
    for rate in (15, 30, 50, 70):
        for seed in (20260930, 20260931, 20260932):
            key = f"seed{seed}_rate{rate}"
            cond = np.load(require(DEV_RESULTS / f"prospective_imputation_2026-09-30/condition_{key}.npz"))
            mask = cond["mask"]
            assert (mask == plan[f"mask_{key}"]).all()
            per = {k: [] for k in ("methylgpt2", "ppca_core", "methylgpt2_foldmean", "ppca_core_foldmean",
                                   "MLM", "fold_mean", "donor_knn_k5")}
            for i in range(136):
                tgt_local = np.flatnonzero(mask[i])
                tgt = vidx[tgt_local]
                y = beta[i, tgt_local]
                x = full[i].copy()
                x[tgt] = np.nan  # masked values are never visible to the model
                xt = torch.from_numpy(x[None]).to(dev)
                ctx = torch.isfinite(xt)
                tt = torch.from_numpy(tgt[None]).to(dev)
                core.mu.copy_(torch.from_numpy(corpus_mu).to(dev))
                pm, pb = predict(model, core, xt, ctx, tt)
                core.mu.copy_(torch.from_numpy(fold_mu[int(folds[i])]).to(dev))
                fm, fb = predict(model, core, xt, ctx, tt)
                for k, p in (("methylgpt2", pm), ("ppca_core", pb), ("methylgpt2_foldmean", fm), ("ppca_core_foldmean", fb)):
                    per[k].append(float(np.mean(np.abs(p.astype(np.float64) - y))))
                for k in ("MLM", "fold_mean", "donor_knn_k5"):
                    per[k].append(float(np.mean(np.abs(cond[f"prediction_{k}"][i, tgt_local].astype(np.float64) - y))))
            core.mu.copy_(torch.from_numpy(corpus_mu).to(dev))
            res["conditions"][key] = {k: float(np.mean(v)) for k, v in per.items()}
            res["conditions"][key + "_per_sample"] = per
            log(f"{key}: " + ", ".join(f"{k} {v:.5f}" for k, v in res["conditions"][key].items()))
    by_rate = {}
    for rate in (15, 30, 50, 70):
        keys = [k for k in res["conditions"] if k.endswith(f"rate{rate}")]
        by_rate[str(rate)] = {m: float(np.mean([res["conditions"][k][m] for k in keys])) for m in res["conditions"][keys[0]]}
    res["seed_averaged_mae_by_rate"] = by_rate
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)
    log(f"wrote {a.out}")
    for r, v in by_rate.items():
        log(f"rate {r}: " + ", ".join(f"{k} {x:.5f}" for k, x in v.items()))


if __name__ == "__main__":
    main()
