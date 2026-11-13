#!/usr/bin/env python3
"""Platform-conversion imputation on held-out datasets (MethylGPT-2, run once).

Each test profile keeps only the vocabulary CpGs present on a platform (27K, EPIC v2 or 450K)
and the other observed CpGs are the targets, i.e. converting data from that array to the
full vocabulary. Otherwise identical to evaluate_heldout_2026-09-30.py.

Test profiles come only from dataset groups assigned to the test split before training.
For each masking rate a fixed seed masks that fraction of each profile's observed CpGs.
Methods: training-split CpG mean, PPCA core, kNN (k = 10) over all training profiles,
the January MethylGPT checkpoint (MLM head, observed-token policy; it may have seen these
datasets during its own pretraining) and MethylGPT-2. Metrics are pooled MSE/MAE,
within-CpG R2 and paired bootstrap intervals over profiles for MSE differences.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, HERE, Corpus, PPCACore, log, mse_metrics, write_json
from methylgpt_model import MethylGPT2, load_checkpoint

RATES = ("keep_hm27", "keep_epic_v2", "keep_hm450")
PLATFORM = np.load(Path(__file__).resolve().parent / "artifacts/platform_vocab_masks_2026-10-01.npz")


def load_model(ckpt_path: Path, core_path, dev):
    return load_checkpoint(ckpt_path, core_path, dev)


@torch.no_grad()
def predict_model(model, core, X, ctx, masked, batch=8, chunk=4096):
    out_m, out_b = [], []
    for i in range(0, len(X), batch):
        x, c, m = X[i:i + batch], ctx[i:i + batch], masked[i:i + batch]
        s = core.posterior(x, c.float())
        with torch.autocast("cuda", enabled=False):
            xs, cc, e = model.inputs(x, c, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(xs, cc, e, c, s)
        for b in range(len(x)):
            idx = torch.nonzero(m[b])[:, 0][None]
            pm, pb = [], []
            for j in range(0, idx.shape[1], chunk):
                pred, base = model.predict_targets(z[b:b + 1], s[b:b + 1], idx[:, j:j + chunk])
                pm.append(pred); pb.append(base)
            out_m.append(torch.cat(pm, 1)[0].cpu().numpy())
            out_b.append(torch.cat(pb, 1)[0].cpu().numpy())
    return out_m, out_b


@torch.no_grad()
def knn_predict(Rf, Rraw, X, ctx, masked, k=10, batch=64, chunk=16384):
    """Squared distance over each query's context CpGs against mean-filled references
    (fp32 per reference chunk, TF32 allowed for this search only); the prediction is the
    mean of the k nearest neighbours' observed values at each masked CpG."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    preds = []
    try:
        for i in range(0, len(X), batch):
            x, c = X[i:i + batch], ctx[i:i + batch].float()
            xin = torch.where(c > 0, x, torch.zeros((), device=x.device))
            q2 = (xin ** 2).sum(1, keepdim=True)
            best_d = torch.full((len(x), k), float("inf"), device=x.device)
            best_i = torch.zeros((len(x), k), dtype=torch.long, device=x.device)
            for r0 in range(0, len(Rf), chunk):
                R = Rf[r0:r0 + chunk].float()
                d = q2 - 2 * xin @ R.T + c @ (R * R).T
                dd = torch.cat([best_d, d], 1)
                ii = torch.cat([best_i, torch.arange(r0, r0 + len(R), device=x.device).expand(len(x), -1)], 1)
                best_d, pos = torch.topk(dd, k, dim=1, largest=False)
                best_i = ii.gather(1, pos)
            for b in range(len(x)):
                cols = torch.nonzero(masked[i + b])[:, 0]
                vals = Rraw[best_i[b]][:, cols].float()
                ok = torch.isfinite(vals)
                num = torch.where(ok, vals, torch.zeros((), device=x.device)).sum(0)
                den = ok.sum(0)
                fallback = Rf[best_i[b]][:, cols].float().mean(0)
                preds.append(torch.where(den > 0, num / den.clamp_min(1), fallback).cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return preds


def january_predict(X_np, masks_np, device):
    spec = importlib.util.spec_from_file_location(
        "diag", HERE / "diagnostics_2026-09-30/diag_pretrain_imputation_baselines_2026-09-30.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    mlm, _, _ = diag.methylgpt_predict(X_np, masks_np, device)
    return mlm


def boot_diff(pp_a, pp_b, n=2000, seed=20260930):
    """Paired bootstrap over profiles of the difference in pooled MSE (a - b)."""
    rng = np.random.default_rng(seed)
    sse_a, sse_b, cnt = (np.array([p[0] for p in pp_a]), np.array([p[0] for p in pp_b]),
                         np.array([p[1] for p in pp_a]))
    point = sse_a.sum() / cnt.sum() - sse_b.sum() / cnt.sum()
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(cnt), len(cnt))
        draws.append(sse_a[i].sum() / cnt[i].sum() - sse_b[i].sum() / cnt[i].sum())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--core", default=None, help="defaults to the core recorded in the checkpoint")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-january", type=int, default=500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-january", action="store_true")
    ap.add_argument("--rates", default=None, help="comma-separated subset of keep_hm27,keep_epic_v2,keep_hm450")
    ap.add_argument("--masks", default=None, help="alternative array-membership masks (npz with hm27, hm450, epic_v2); default: the masks of the reported analysis")
    a = ap.parse_args()
    global PLATFORM, RATES
    if a.masks:
        PLATFORM = np.load(a.masks)
    if a.rates:
        RATES = tuple(a.rates.split(","))
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    corpus = Corpus()
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), a.n_test, replace=False))
    meta = corpus.profiles.loc[test_rows, ["id", "group", "tissue", "gpl"]].reset_index(drop=True)
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    model, core, ck = load_model(Path(a.ckpt), a.core, dev)
    log(f"test profiles {len(X)} from {meta.group.nunique()} dataset groups; checkpoint step {ck['step']}")

    train_rows = corpus.rows("train")
    Rraw = torch.empty((len(train_rows), X.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, len(train_rows), 8192):
        Rraw[i:i + 8192] = torch.from_numpy(corpus.beta[np.sort(train_rows)[i:i + 8192]]).to(dev)
    Rf = torch.where(torch.isfinite(Rraw), Rraw, core.mu.half())
    log("kNN reference loaded")

    results = {"n_test": int(len(X)), "n_test_groups": int(meta.group.nunique()), "checkpoint": a.ckpt,
               "checkpoint_step": int(ck["step"]), "rates": {}}
    per_profile_rows = []
    X_all, meta_all = X, meta
    for rate in RATES:
        keep = torch.from_numpy(PLATFORM[rate.replace("keep_", "")]).to(dev)
        # profiles with nothing left to predict (all observed CpGs are on the array) are skipped
        has = (torch.isfinite(X_all) & ~keep[None, :]).any(1)
        X, meta = X_all[has], meta_all[has.cpu().numpy()].reset_index(drop=True)
        obs = torch.isfinite(X)
        masked = obs & ~keep[None, :]
        ctx = obs & ~masked
        truth = [X[i][masked[i]].cpu().numpy() for i in range(len(X))]
        cpgs = [torch.nonzero(masked[i])[:, 0].cpu().numpy() for i in range(len(X))]
        preds = {"cpg_mean": [core.mu[torch.from_numpy(c).to(dev)].cpu().numpy() for c in cpgs]}
        preds["methylgpt2"], preds["ppca_core"] = predict_model(model, core, X, ctx, masked)
        preds["knn10"] = knn_predict(Rf, Rraw, X, ctx, masked)
        if not a.skip_january:
            n_j = min(a.n_january, len(X))
            m_np = masked[:n_j].cpu().numpy()
            preds["january_mlm"] = january_predict(X[:n_j].cpu().numpy(), list(m_np), a.device) + [None] * (len(X) - n_j)
        block = {}
        pp = {}
        for name, p in preds.items():
            sub = [i for i in range(len(X)) if p[i] is not None]
            pr = np.concatenate([p[i] for i in sub]); tr = np.concatenate([truth[i] for i in sub])
            cp = np.concatenate([cpgs[i] for i in sub])
            block[name] = {**mse_metrics(pr, tr, cp), "n_profiles": len(sub)}
            pp[name] = [(float(((p[i] - truth[i]) ** 2).sum()), len(truth[i])) if p[i] is not None else None
                        for i in range(len(X))]
        for other in ("ppca_core", "knn10", "cpg_mean", "january_mlm"):
            if other in pp:
                sub = [i for i in range(len(X)) if pp[other][i] is not None]
                block[f"methylgpt2_minus_{other}"] = boot_diff([pp["methylgpt2"][i] for i in sub],
                                                               [pp[other][i] for i in sub])
        strata = {}
        for col in ("tissue", "gpl"):
            for val, idx in meta.groupby(col).groups.items():
                if len(idx) < 50:
                    continue
                strata[f"{col}={val}"] = {name: float(sum(pp[name][i][0] for i in idx) / sum(pp[name][i][1] for i in idx))
                                          for name in ("methylgpt2", "ppca_core", "knn10", "cpg_mean")} | {"n": len(idx)}
        block["strata_mse"] = strata
        block["n_profiles_with_targets"] = int(has.sum()); block["cpgs_kept"] = int(keep.sum())
        results["rates"][str(rate)] = block
        for i in range(len(X)):
            per_profile_rows.append({"rate": rate, "id": meta.id[i], **{f"sse_{k}": (v[i][0] if v[i] else None) for k, v in pp.items()},
                                     "n_masked": pp["methylgpt2"][i][1]})
        log(f"rate {rate}: " + ", ".join(f"{k} {v['mse']:.5f}" for k, v in block.items() if isinstance(v, dict) and "mse" in v))
    results["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), results)
    pd.DataFrame(per_profile_rows).to_csv(Path(a.out).with_suffix(".per_profile.csv.gz"), index=False)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
