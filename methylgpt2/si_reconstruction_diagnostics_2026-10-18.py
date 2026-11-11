#!/usr/bin/env python3
"""Supplementary reconstruction diagnostics on held-out datasets (18 Oct 2026; Supplementary Information).

Pre-specified primary metric (fixed before the first run)
  Within-CpG R^2 of MethylGPT minus that of nearest neighbours (k = 10) on the 2,000 held-out profiles
  from 107 test-split datasets with 30% random masks. Within-CpG R^2 is the mse_metrics definition
  (1 - sum_j SSE_j / sum_j SST_j over CpGs with at least 20 masked values, SST_j around the CpG's own
  mean of the masked values). 95% interval: 2,000 dataset-cluster bootstrap replicates (the 107
  datasets, column `group`, resampled with replacement; seed 20261018).

Rows and masks (identical to the source scripts)
  Rows: np.random.default_rng(20261001).choice(test-split rows, 2000), sorted, as in
  evaluate_heldout_2026-09-30.py and evaluate_heldout_structured_2026-10-01.py.
  The two scripts draw different 30% masks of these rows:
    S  np.random.default_rng(20261001).random(X.shape) < 0.3 (structured script; Fig. 1c, 1f and
       heldout_structured_2026-10-01.random30_pairs.npz)
    T  torch CUDA generator seeded int(1000 * 0.3) + 20261001 (09-30 script; Fig. 1e and the
       originally submitted model's 0.00807 on the first 500 profiles)
  (a)-(d) use S. (e) uses T on the first 500 profiles, where the originally submitted model was
  evaluated. As checks, all methods are also recomputed on T for the 2,000 profiles and the originally
  submitted model on S for the first 500.

Methods
  MethylGPT: self-supervised encoder at 2,000 steps (artifacts/runs/stopped_cf_dense_2026-09-30/best.pt;
  its input layer equals the released core.npz). Input layer alone: the core's posterior
  reconstruction from the context CpGs (secondary, tables only). Nearest neighbours: k = 10 over the
  145,700 training profiles (the source scripts' kNN; the mean-filled reference is built per chunk to
  halve GPU memory, numerically identical). CpG means: training-split means. Originally submitted
  model: january_predict of evaluate_heldout_2026-09-30.py (MLM head, observed-token policy).

Secondary metrics (all reported)
  (a) pooled Pearson r, within-CpG R^2 (pooled and median over CpGs), MSE and MAE per method with
      dataset-cluster intervals; paired differences MethylGPT minus each comparator.
  (b) residual (predicted - observed) against predicted and against observed: binned mean and
      quantiles from all masked values, OLS slopes, and a seeded 1,000,000-pair sample for hexbins.
  (c) MAE by observed-beta bin (bins of the structured script) with n per bin and dataset-cluster
      intervals; median within-held-out-dataset CpG SD per bin (SD of the CpG across all test-split
      profiles of the target's dataset, >= 5 observed values); MAE / median SD; sensitivity with the
      SD computed within dataset x tissue; MAE relative to the distance from 0 or 1 (table only).
  (d) MSE and MAE by tissue for tissues with >= 50 of the 2,000 profiles, with profiles and datasets
      per tissue and dataset-cluster intervals within tissue.
  (e) originally submitted model on the first 500 profiles (T masks) with MethylGPT and CpG means on
      the same masked values: MSE, MAE, within-CpG R^2, pooled r, MAE by beta bin, residual summaries.

Outputs (aggregates only)
  artifacts/eval/si_recon_diagnostics_2026-10-18.json
  artifacts/eval/si_recon_diagnostics_2026-10-18.{overall,by_beta_bin,by_tissue,january500}.csv
  artifacts/eval/si_recon_diagnostics_2026-10-18.pairs.npz  (value samples for the hexbins)
Run: python3 si_reconstruction_diagnostics_2026-10-18.py --device cuda:1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, HERE, Corpus, cuda_preinit, log, mse_metrics, require, write_json
from methylgpt2_model import load_checkpoint

spec = importlib.util.spec_from_file_location("plat", HERE / "evaluate_heldout_platform_2026-10-01.py")
plat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plat)

BINS = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001])   # structured script
SD_EDGES = np.linspace(0, 0.5, 1001)        # SD histogram for the bootstrap of the per-bin median SD
B_BOOT = 2000
SEED_BOOT = 20261018
METHODS = ("methylgpt2", "ppca_core", "knn10", "cpg_mean")
OUT = ARTIFACTS / "eval/si_recon_diagnostics_2026-10-18"


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def ci(x):
    lo, hi = np.nanpercentile(np.asarray(x, float), [2.5, 97.5])
    return [float(lo), float(hi)]


def cluster_counts(D, B=B_BOOT, seed=SEED_BOOT):
    """(B, D) multiplicities of each dataset in each cluster-bootstrap replicate."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(B, D))
    return np.stack([np.bincount(r, minlength=D) for r in idx]).astype(np.float64)


@torch.no_grad()
def knn_predict_lowmem(Rraw, mu_h, X, ctx, masked, k=10, batch=64, chunk=16384):
    """plat.knn_predict with the mean-filled reference built per chunk (identical arithmetic)."""
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
            for r0 in range(0, len(Rraw), chunk):
                Rh = Rraw[r0:r0 + chunk]
                R = torch.where(torch.isfinite(Rh), Rh, mu_h).float()
                d = q2 - 2 * xin @ R.T + c @ (R * R).T
                del R
                dd = torch.cat([best_d, d], 1)
                ii = torch.cat([best_i, torch.arange(r0, r0 + len(Rh), device=x.device).expand(len(x), -1)], 1)
                best_d, pos = torch.topk(dd, k, dim=1, largest=False)
                best_i = ii.gather(1, pos)
            for b in range(len(x)):
                cols = torch.nonzero(masked[i + b])[:, 0]
                vals = Rraw[best_i[b]][:, cols].float()
                ok = torch.isfinite(vals)
                num = torch.where(ok, vals, torch.zeros((), device=x.device)).sum(0)
                den = ok.sum(0)
                fallback = torch.where(ok, vals, mu_h[cols].float()).mean(0)
                preds.append(torch.where(den > 0, num / den.clamp_min(1), fallback).cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return preds


def january_predict(X_np, masks_np, device):
    """evaluate_heldout_2026-09-30.january_predict (the originally submitted model)."""
    spec_ = importlib.util.spec_from_file_location(
        "diag", HERE / "diagnostics_2026-09-30/diag_pretrain_imputation_baselines_2026-09-30.py")
    diag = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(diag)
    mlm, _, info = diag.methylgpt_predict(X_np, masks_np, device)
    return mlm


def flatten(masked, X):
    """Per-target truth, CpG index and profile index (row-major over profiles, CpG order within)."""
    nz = torch.nonzero(masked)
    prof = nz[:, 0].cpu().numpy().astype(np.int32)
    cpg = nz[:, 1].cpu().numpy().astype(np.int32)
    truth = X[masked].cpu().numpy().astype(np.float64)
    return truth, cpg, prof


# ---------------------------------------------------------------- aggregate statistics
def ds_sums(ds, D, y, p):
    e = p - y
    return {"n": np.bincount(ds, minlength=D).astype(np.float64), "sse": np.bincount(ds, e * e, D),
            "sae": np.bincount(ds, np.abs(e), D), "sy": np.bincount(ds, y, D), "syy": np.bincount(ds, y * y, D),
            "sp": np.bincount(ds, p, D), "spp": np.bincount(ds, p * p, D), "syp": np.bincount(ds, y * p, D)}


def sums_metrics(a):
    n = a["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (a["syp"] - a["sy"] * a["sp"] / n) / np.sqrt((a["syy"] - a["sy"] ** 2 / n) * (a["spp"] - a["sp"] ** 2 / n))
    return {"mse": a["sse"] / n, "mae": a["sae"] / n, "pooled_r": r}


def within_tables(ds, cpg, D, P, y, preds):
    k = ds.astype(np.int64) * P + cpg
    n = np.bincount(k, minlength=D * P).astype(np.float64).reshape(D, P)
    sy = np.bincount(k, y, D * P).reshape(D, P)
    syy = np.bincount(k, y * y, D * P).reshape(D, P)
    sse = {m: np.bincount(k, (p - y) ** 2, D * P).reshape(D, P) for m, p in preds.items()}
    return n, sy, syy, sse


def within_r2(Cb, n, sy, syy, sse, min_n=20):
    nb, syb, syyb = Cb @ n, Cb @ sy, Cb @ syy
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = syyb - syb ** 2 / np.maximum(nb, 1)
    keep = (nb >= min_n) & (sst > 1e-12)
    out = {}
    for m, s in sse.items():
        sb = Cb @ s
        pooled = 1 - (sb * keep).sum(1) / (sst * keep).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per = np.where(keep, 1 - sb / np.where(keep, sst, 1.0), np.nan)
        out[m] = (pooled, np.nanmedian(per, 1), keep.sum(1))
    return out


def evaluate_set(y, cpg, prof, preds, prof_ds, P, label, boot_seed=SEED_BOOT, min_n=20):
    """Overall metrics for every method with dataset-cluster intervals and paired differences."""
    ds_codes, ds_uni = pd.factorize(prof_ds[prof])
    D = len(ds_uni)
    C = cluster_counts(D, seed=boot_seed)
    ones = np.ones((1, D))
    res = {"n_targets": int(len(y)), "n_profiles": int(len(np.unique(prof))), "n_datasets": int(D), "methods": {}}
    sums = {m: ds_sums(ds_codes, D, y, p) for m, p in preds.items()}
    pt = {m: sums_metrics({k: ones @ v for k, v in s.items()}) for m, s in sums.items()}
    bs = {m: sums_metrics({k: C @ v for k, v in s.items()}) for m, s in sums.items()}
    n, sy, syy, sse = within_tables(ds_codes, cpg, D, P, y, preds)
    w_pt = within_r2(ones, n, sy, syy, sse, min_n)
    w_bs = {m: ([], []) for m in preds}
    for b0 in range(0, len(C), 50):
        wb = within_r2(C[b0:b0 + 50], n, sy, syy, sse, min_n)
        for m in preds:
            w_bs[m][0].append(wb[m][0]); w_bs[m][1].append(wb[m][1])
    w_bs = {m: (np.concatenate(a), np.concatenate(b)) for m, (a, b) in w_bs.items()}
    for m in preds:
        mm = mse_metrics(preds[m], y, cpg, min_n)
        res["methods"][m] = {
            "mse": float(pt[m]["mse"][0]), "mse_ci95": ci(bs[m]["mse"]),
            "mae": float(pt[m]["mae"][0]), "mae_ci95": ci(bs[m]["mae"]),
            "pooled_r": float(pt[m]["pooled_r"][0]), "pooled_r_ci95": ci(bs[m]["pooled_r"]),
            "within_cpg_r2_pooled": float(w_pt[m][0][0]), "within_cpg_r2_pooled_ci95": ci(w_bs[m][0]),
            "within_cpg_r2_median": float(w_pt[m][1][0]), "within_cpg_r2_median_ci95": ci(w_bs[m][1]),
            "n_cpgs_within": int(w_pt[m][2][0]), "mse_metrics_check": mm}
    ref = "methylgpt2" if "methylgpt2" in preds else None
    res["differences"] = {}
    if ref:
        for other in preds:
            if other == ref:
                continue
            d = {}
            for key, a_pt, a_bs, b_pt, b_bs in (
                    ("mse", pt[ref]["mse"][0], bs[ref]["mse"], pt[other]["mse"][0], bs[other]["mse"]),
                    ("mae", pt[ref]["mae"][0], bs[ref]["mae"], pt[other]["mae"][0], bs[other]["mae"]),
                    ("pooled_r", pt[ref]["pooled_r"][0], bs[ref]["pooled_r"], pt[other]["pooled_r"][0], bs[other]["pooled_r"]),
                    ("within_cpg_r2_pooled", w_pt[ref][0][0], w_bs[ref][0], w_pt[other][0][0], w_bs[other][0]),
                    ("within_cpg_r2_median", w_pt[ref][1][0], w_bs[ref][1], w_pt[other][1][0], w_bs[other][1])):
                d[key] = {"diff": float(a_pt - b_pt), "ci95": ci(a_bs - b_bs)}
            res["differences"][f"{ref}_minus_{other}"] = d
    log(f"{label}: " + ", ".join(f"{m} MSE {v['mse']:.5f} r {v['pooled_r']:.3f} R2 {v['within_cpg_r2_pooled']:.3f}"
                                 for m, v in res["methods"].items()))
    return res, ds_codes, D, C


def by_beta_bin(y, preds, ds_codes, D, C, sd=None, sd_alt=None):
    b = np.clip(np.digitize(y, BINS) - 1, 0, len(BINS) - 2)
    NB = len(BINS) - 1
    k = ds_codes.astype(np.int64) * NB + b
    n = np.bincount(k, minlength=D * NB).reshape(D, NB).astype(np.float64)
    rel_den = np.maximum(np.minimum(y, 1 - y), 0.01)
    rows = []
    nb_pt, nb_bs = n.sum(0), C @ n
    sae = {m: np.bincount(k, np.abs(p - y), D * NB).reshape(D, NB) for m, p in preds.items()}
    srel = {m: np.bincount(k, np.abs(p - y) / rel_den, D * NB).reshape(D, NB) for m, p in preds.items()}
    n_prof_bin = None
    sd_block = {}
    for tag, s in (("dataset", sd), ("dataset_tissue", sd_alt)):
        if s is None:
            continue
        ok = np.isfinite(s)
        k2 = k[ok]
        n_sd = np.bincount(k2, minlength=D * NB).reshape(D, NB).astype(np.float64)
        sae_sd = {m: np.bincount(k2, np.abs(p[ok] - y[ok]), D * NB).reshape(D, NB) for m, p in preds.items()}
        sbin = np.clip(np.digitize(s[ok], SD_EDGES) - 1, 0, len(SD_EDGES) - 2)
        hist = np.bincount(k2 * (len(SD_EDGES) - 1) + sbin, minlength=D * NB * (len(SD_EDGES) - 1)).reshape(
            D, NB, len(SD_EDGES) - 1).astype(np.float64)
        med_exact = np.array([np.median(s[ok][b[ok] == j]) if (b[ok] == j).any() else np.nan for j in range(NB)])
        q_exact = np.array([np.percentile(s[ok][b[ok] == j], [25, 75]) if (b[ok] == j).any() else [np.nan, np.nan]
                            for j in range(NB)])
        hb = np.einsum("bd,dkj->bkj", C, hist)
        cum = np.cumsum(hb, 2)
        half = cum[:, :, -1:] / 2
        idx = (cum < half).sum(2)
        centres = (SD_EDGES[:-1] + SD_EDGES[1:]) / 2
        med_bs = centres[np.clip(idx, 0, len(centres) - 1)]
        sd_block[tag] = {"n_sd": n_sd, "sae_sd": sae_sd, "med": med_exact, "q": q_exact, "med_bs": med_bs,
                         "frac_with_sd": float(ok.mean())}
    for j in range(NB):
        lab = f"[{BINS[j]:.2f},{min(BINS[j + 1], 1):.2f})"
        row = {"bin": lab, "lo": float(BINS[j]), "hi": float(min(BINS[j + 1], 1.0)), "n_targets": int(nb_pt[j]),
               "n_datasets": int((n[:, j] > 0).sum())}
        for m in preds:
            mae_bs = (C @ sae[m][:, j]) / nb_bs[:, j]
            row[f"mae_{m}"] = float(sae[m][:, j].sum() / nb_pt[j])
            row[f"mae_{m}_lo"], row[f"mae_{m}_hi"] = ci(mae_bs)
            row[f"relative_mae_{m}"] = float(srel[m][:, j].sum() / nb_pt[j])
        for tag, blk in sd_block.items():
            row[f"median_sd_{tag}"] = float(blk["med"][j])
            row[f"sd_{tag}_q25"], row[f"sd_{tag}_q75"] = float(blk["q"][j][0]), float(blk["q"][j][1])
            row[f"median_sd_{tag}_lo"], row[f"median_sd_{tag}_hi"] = ci(blk["med_bs"][:, j])
            row[f"n_targets_with_sd_{tag}"] = int(blk["n_sd"][:, j].sum())
            for m in preds:
                mae_r = blk["sae_sd"][m][:, j].sum() / blk["n_sd"][:, j].sum()
                ratio_bs = ((C @ blk["sae_sd"][m][:, j]) / (C @ blk["n_sd"][:, j])) / blk["med_bs"][:, j]
                row[f"mae_over_sd_{tag}_{m}"] = float(mae_r / blk["med"][j])
                row[f"mae_over_sd_{tag}_{m}_lo"], row[f"mae_over_sd_{tag}_{m}_hi"] = ci(ratio_bs)
        rows.append(row)
    extra = {tag: blk["frac_with_sd"] for tag, blk in sd_block.items()}
    return rows, extra


def residual_summary(y, preds, edges=np.linspace(0, 1, 51), redges=np.linspace(-1, 1, 2001)):
    """Binned residual (predicted - observed) against predicted and against observed, all targets."""
    out = {"x_edges": edges.tolist(), "methods": {}}
    centres = (redges[:-1] + redges[1:]) / 2
    for m, p in preds.items():
        r = p - y
        blk = {}
        for axis, x in (("predicted", p), ("observed", y)):
            xb = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
            rb = np.clip(np.digitize(r, redges) - 1, 0, len(redges) - 2)
            h = np.bincount(xb * (len(redges) - 1) + rb, minlength=(len(edges) - 1) * (len(redges) - 1)).reshape(
                len(edges) - 1, len(redges) - 1)
            n = h.sum(1)
            mean = np.bincount(xb, r, len(edges) - 1) / np.maximum(n, 1)
            cum = np.cumsum(h, 1) / np.maximum(n[:, None], 1)
            qs = {f"q{int(q * 100):02d}": [float(centres[min(np.searchsorted(c, q), len(centres) - 1)]) if nn > 0 else None
                                          for c, nn in zip(cum, n)] for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
            slope = float(np.cov(x, r, bias=True)[0, 1] / np.var(x))
            blk[axis] = {"n": n.astype(int).tolist(), "mean": [float(v) if nn > 0 else None for v, nn in zip(mean, n)],
                         **qs, "ols_slope": slope}
        blk["mean_residual"] = float(np.mean(r))
        blk["frac_pred_outside_0_1"] = float(np.mean((p < 0) | (p > 1)))
        out["methods"][m] = blk
    return out


def sd_tables(corpus, groups, keys):
    """Within-dataset CpG SD over all test-split profiles of each held-out dataset (>= 5 observed values).
    keys: list of (group, tissue or None). Returns dict key -> (P,) SD array and the profile counts."""
    prof = corpus.profiles
    test = prof[(prof.split == "test") & prof.group.isin(groups)]
    out, counts = {}, {}
    for key in keys:
        g, t = key
        rows = test.index[(test.group == g) & ((test.tissue == t) if t is not None else True)].to_numpy()
        Xg = corpus.get(rows).astype(np.float64)
        cnt = np.isfinite(Xg).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.nanstd(Xg, axis=0, ddof=1)
        sd[cnt < 5] = np.nan
        out[key], counts[key] = sd.astype(np.float32), int(len(rows))
    return out, counts


def by_tissue(y, cpg, prof, preds, meta, P, min_profiles=50):
    rows = []
    tissues = meta.tissue.value_counts()
    for t, n_prof in tissues.items():
        if n_prof < min_profiles:
            continue
        pmask = (meta.tissue.to_numpy() == t)
        sel = pmask[prof]
        res, ds_codes, D, C = evaluate_set(y[sel], cpg[sel], prof[sel], {m: p[sel] for m, p in preds.items()},
                                           meta.group.to_numpy(), P, f"tissue {t}", boot_seed=SEED_BOOT + 1)
        row = {"tissue": t, "n_profiles": int(n_prof), "n_datasets": int(D), "n_targets": res["n_targets"],
               "fewer_than_5_datasets": bool(D < 5)}
        for m, v in res["methods"].items():
            for key in ("mse", "mae", "pooled_r", "within_cpg_r2_pooled"):
                row[f"{key}_{m}"] = v[key]
                row[f"{key}_{m}_lo"], row[f"{key}_{m}_hi"] = v[f"{key}_ci95"]
            row[f"n_cpgs_within_{m}"] = v["n_cpgs_within"]
        for name, d in res["differences"].items():
            for key in ("mse", "within_cpg_r2_pooled"):
                row[f"{key}_{name}"] = d[key]["diff"]
                row[f"{key}_{name}_lo"], row[f"{key}_{name}_hi"] = d[key]["ci95"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"))
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-january", type=int, default=500)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    res = {"script": Path(__file__).name, "primary_metric": "within-CpG R2 (pooled) MethylGPT minus nearest neighbours, "
           "2,000 held-out profiles, structured 30% masks, 2,000 dataset-cluster bootstrap replicates (seed 20261018)"}

    # ---- inputs and checksums
    ck_path = require(Path(a.ckpt))
    rel = ARTIFACTS / "release/methylgpt2-1.1"
    pkg = require(HERE / "package/methylgpt2-1.1.checksums.sha256")
    expected = dict(line.split()[::-1] for line in pkg.read_text().splitlines() if line.strip())
    core_sha = sha256(require(rel / "core.npz"))
    res["inputs"] = {"checkpoint": str(ck_path), "checkpoint_sha256": sha256(ck_path),
                     "release_core_sha256": core_sha, "release_core_matches_package": core_sha == expected.get("core.npz"),
                     "annotation": str(ARTIFACTS / "cpg_annotation_2026-10-01.parquet")}
    log(f"checksums: release core matches package {res['inputs']['release_core_matches_package']}")

    corpus = Corpus()
    rng = np.random.default_rng(20261001)
    test_rows = np.sort(rng.choice(corpus.rows("test"), a.n_test, replace=False))
    meta = corpus.profiles.loc[test_rows, ["id", "group", "tissue", "gpl"]].reset_index(drop=True)
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    P = X.shape[1]
    obs = torch.isfinite(X)
    maskS = obs & torch.from_numpy(np.random.default_rng(20261001).random(tuple(X.shape)) < 0.3).to(dev)
    g = torch.Generator(device=dev).manual_seed(int(1000 * 0.3) + 20261001)
    maskT = obs & (torch.rand(X.shape, device=dev, generator=g) < 0.3)
    res["rows"] = {"n_profiles": int(len(X)), "n_datasets": int(meta.group.nunique()),
                   "n_masked_S": int(maskS.sum()), "n_masked_T": int(maskT.sum()),
                   "first500_n_datasets": int(meta.group[:a.n_january].nunique())}
    log(f"{len(X)} profiles from {meta.group.nunique()} datasets; masked S {int(maskS.sum())}, T {int(maskT.sum())}")

    # ---- originally submitted model first (its loader caps this process's GPU memory fraction)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    import sys
    sys.path.insert(0, "/Entropy/kying/tmp/methylgpt_publication_2026-09-28")
    probes = pd.read_csv(require(Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/processed_dataset/probe_ids_type3.csv")))[
        "illumina_probe_id"].astype(str).to_numpy()
    res["inputs"]["january_probe_order_equals_vocabulary"] = bool(len(probes) == P and (probes == ann.cpg.to_numpy()).all())
    if not res["inputs"]["january_probe_order_equals_vocabulary"]:
        raise SystemExit("Originally submitted model's probe order differs from the vocabulary")
    nj = a.n_january
    Xn = X[:nj].cpu().numpy()
    jan_T = january_predict(Xn, list(maskT[:nj].cpu().numpy()), a.device)
    jan_S = january_predict(Xn, list(maskS[:nj].cpu().numpy()), a.device)
    torch.cuda.set_per_process_memory_fraction(1.0, dev.index)
    torch.cuda.empty_cache()
    log("originally submitted model done")

    # ---- MethylGPT, input layer, CpG means, nearest neighbours on both mask sets
    model, core, ck = load_checkpoint(ck_path, None, dev)
    res["inputs"]["checkpoint_step"] = int(ck["step"])
    train_rows = np.sort(corpus.rows("train"))
    Rraw = torch.empty((len(train_rows), P), dtype=torch.float16, device=dev)
    for i in range(0, len(train_rows), 8192):
        Rraw[i:i + 8192] = torch.from_numpy(corpus.beta[train_rows[i:i + 8192]]).to(dev)
    mu_h = core.mu.half()
    log("kNN reference loaded")
    flat = {}
    for tag, masked in (("S", maskS), ("T", maskT)):
        ctx = obs & ~masked
        y, cpg, prof = flatten(masked, X)
        mg, base = plat.predict_model(model, core, X, ctx, masked)
        kn = knn_predict_lowmem(Rraw, mu_h, X, ctx, masked)
        preds = {"methylgpt2": np.concatenate(mg).astype(np.float64), "ppca_core": np.concatenate(base).astype(np.float64),
                 "knn10": np.concatenate(kn).astype(np.float64),
                 "cpg_mean": core.mu.cpu().numpy().astype(np.float64)[cpg]}
        flat[tag] = (y, cpg, prof, preds)
        log(f"mask {tag}: predictions done ({len(y)} targets)")
    del Rraw
    torch.cuda.empty_cache()

    # ---- (a) overall metrics
    yS, cS, pS, prS = flat["S"]
    yT, cT, pT, prT = flat["T"]
    gr = meta.group.to_numpy()
    res["overall_S"], dsS, DS, CS = evaluate_set(yS, cS, pS, prS, gr, P, "S (primary)")
    res["overall_T"], _, _, _ = evaluate_set(yT, cT, pT, prT, gr, P, "T")
    d = res["overall_S"]["differences"]["methylgpt2_minus_knn10"]["within_cpg_r2_pooled"]
    res["primary"] = {"metric": res["primary_metric"], "methylgpt2": res["overall_S"]["methods"]["methylgpt2"]["within_cpg_r2_pooled"],
                      "knn10": res["overall_S"]["methods"]["knn10"]["within_cpg_r2_pooled"], "diff": d["diff"], "ci95": d["ci95"],
                      "n_profiles": res["overall_S"]["n_profiles"], "n_datasets": res["overall_S"]["n_datasets"],
                      "n_targets": res["overall_S"]["n_targets"], "bootstrap_replicates": B_BOOT,
                      "resampled": "datasets (column group), with replacement"}
    log(f"PRIMARY within-CpG R2 MethylGPT - kNN = {d['diff']:.4f} {d['ci95']}")

    # ---- checks against the stored results
    chk = {}
    st = json.loads((ARTIFACTS / "eval/heldout_structured_2026-10-01.json").read_text())["mask_types"]["random_30"]
    hr = json.loads((ARTIFACTS / "eval/heldout_cf_dense_step2000_2026-09-30.json").read_text())["rates"]["0.3"]
    for tag, ref, mine in (("S_vs_structured_json", st, res["overall_S"]), ("T_vs_0930_json", hr, res["overall_T"])):
        chk[tag] = {m: {"stored_mse": ref[m]["mse"], "recomputed_mse": mine["methods"][m]["mse"],
                        "stored_n_targets": ref[m]["n_targets"], "recomputed_n_targets": mine["n_targets"],
                        "stored_within_r2": ref[m]["within_cpg_r2_pooled"],
                        "recomputed_within_r2": mine["methods"][m]["within_cpg_r2_pooled"]}
                    for m in ("methylgpt2", "knn10", "cpg_mean")}
    sel = np.random.default_rng(42).choice(len(yS), min(1_000_000, len(yS)), replace=False)
    old = np.load(ARTIFACTS / "eval/heldout_structured_2026-10-01.random30_pairs.npz")
    chk["pairs_npz_max_abs_diff"] = {k: float(np.max(np.abs(old[k].astype(np.float64) - (yS if k == "observed" else prS[k])[sel].astype(np.float32))))
                                     for k in ("observed", "methylgpt2", "knn10", "cpg_mean")}
    res["checks"] = chk
    log(f"checks: {json.dumps(chk)[:600]}")

    # ---- (b) residuals
    res["residuals_S"] = residual_summary(yS, {m: prS[m] for m in ("methylgpt2", "knn10", "cpg_mean")})

    # ---- (c) MAE by beta bin with the within-dataset CpG SD
    groups = sorted(meta.group.unique())
    combos = sorted(set(zip(meta.group, meta.tissue)))
    sd_ds, n_ds = sd_tables(corpus, groups, [(g_, None) for g_ in groups])
    sd_dt, n_dt = sd_tables(corpus, groups, combos)
    gi = {g_: i for i, g_ in enumerate(groups)}
    ci_ = {c_: i for i, c_ in enumerate(combos)}
    SDm = np.stack([sd_ds[(g_, None)] for g_ in groups])
    SDt = np.stack([sd_dt[c_] for c_ in combos])
    prof_g = np.array([gi[g_] for g_ in meta.group])
    prof_c = np.array([ci_[c_] for c_ in zip(meta.group, meta.tissue)])
    sdS = SDm[prof_g[pS], cS]
    sdS_t = SDt[prof_c[pS], cS]
    res["sd_reference"] = {"profiles_per_dataset_used": {g_: n_ds[(g_, None)] for g_ in groups},
                           "n_dataset_tissue_strata": len(combos), "min_observed_values": 5}
    bins, frac = by_beta_bin(yS, {m: prS[m] for m in METHODS}, dsS, DS, CS, sdS, sdS_t)
    res["by_beta_bin_S"] = bins
    res["by_beta_bin_S_frac_targets_with_sd"] = frac
    log("beta bins done")

    # ---- (d) by tissue
    res["by_tissue_S"] = by_tissue(yS, cS, pS, {m: prS[m] for m in METHODS}, meta, P)

    # ---- (e) originally submitted model, first 500 profiles, T masks (and S as a check)
    for tag, jan, (y_, c_, p_, pr_) in (("T", jan_T, flat["T"]), ("S", jan_S, flat["S"])):
        sel = p_ < nj
        jj = np.concatenate(jan).astype(np.float64)
        if len(jj) != sel.sum():
            raise SystemExit(f"originally submitted model returned {len(jj)} values for {sel.sum()} targets")
        pr = {"january_mlm": jj, "methylgpt2": pr_["methylgpt2"][sel], "cpg_mean": pr_["cpg_mean"][sel]}
        r_, ds_, D_, C_ = evaluate_set(y_[sel], c_[sel], p_[sel], pr, gr, P, f"first {nj}, {tag}", boot_seed=SEED_BOOT + 2)
        # differences relative to the originally submitted model
        r_["by_beta_bin"], _ = by_beta_bin(y_[sel], pr, ds_, D_, C_)
        r_["residuals"] = residual_summary(y_[sel], pr)
        res[f"january500_{tag}"] = r_
        if tag == "T":
            selT = np.random.default_rng(43).choice(sel.sum(), min(500_000, int(sel.sum())), replace=False)
            jan_pairs = {f"j500_observed": y_[sel][selT].astype(np.float32),
                         **{f"j500_{m}": v[selT].astype(np.float32) for m, v in pr.items()}}
    res["checks"]["january_T_mse_vs_reported_0.00807"] = {
        "recomputed": res["january500_T"]["methods"]["january_mlm"]["mse"], "stored_json": hr["january_mlm"]["mse"],
        "stored_methylgpt2_minus_january": hr["methylgpt2_minus_january_mlm"]["diff"],
        "recomputed_methylgpt2_minus_january": res["january500_T"]["differences"]["methylgpt2_minus_january_mlm"]["mse"]["diff"]}

    # ---- outputs
    np.savez_compressed(OUT.with_suffix(".pairs.npz"), observed=yS[sel_ := np.random.default_rng(42).choice(
        len(yS), min(1_000_000, len(yS)), replace=False)].astype(np.float32),
        **{m: prS[m][sel_].astype(np.float32) for m in METHODS}, **jan_pairs)
    res["elapsed_s"] = time.time() - t0
    write_json(OUT.with_suffix(".json"), res)
    rows = []
    for tag in ("overall_S", "overall_T", "january500_T", "january500_S"):
        for m, v in res[tag]["methods"].items():
            rows.append({"set": tag, "method": m, "n_targets": res[tag]["n_targets"], "n_profiles": res[tag]["n_profiles"],
                         "n_datasets": res[tag]["n_datasets"],
                         **{k: (v[k] if not isinstance(v[k], list) else None) for k in v if not k.endswith("ci95") and k != "mse_metrics_check"},
                         **{f"{k[:-5]}_lo": v[k][0] for k in v if k.endswith("ci95")},
                         **{f"{k[:-5]}_hi": v[k][1] for k in v if k.endswith("ci95")}})
    pd.DataFrame(rows).to_csv(OUT.with_suffix(".overall.csv"), index=False)
    pd.DataFrame(res["by_beta_bin_S"]).to_csv(OUT.with_suffix(".by_beta_bin.csv"), index=False)
    pd.DataFrame(res["by_tissue_S"]).to_csv(OUT.with_suffix(".by_tissue.csv"), index=False)
    pd.DataFrame(res["january500_T"]["by_beta_bin"]).to_csv(OUT.with_suffix(".january500.csv"), index=False)
    log(f"wrote {OUT}.*; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
