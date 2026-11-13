#!/usr/bin/env python3
"""MethylGPT-2 factor + residual representation for GS incident disease (development subjects).

The PPCA core splits each profile into shared structure (posterior factor scores, which the
encoder turns into the 2,048-d embedding) and a per-CpG residual x - x_hat at observed CpGs.
Diagnostics showed that the endocrine/metabolic signal of raw-CpG Cox sits in CpGs the core
reconstructs poorly in GS (ABCG1, TXNIP, CPT1A; core R2 <= 0.32), so the factor embedding
alone cannot carry it. This model gives the elastic-net Cox the whole decomposition:
[embedding (2,048), distilled full-array biomarkers (25), core residuals (49,156)].

Raw-CpG Cox is refitted with the same GPU elastic net, outer folds, inner 3-fold alpha
selection, seeds and iterations, so the two differ only in their inputs. Fold models are
fitted at full coverage and then applied, unchanged, to held-out subjects whose CpGs are
removed (random_<p>, platform subsets), as in scenario_gs_missingness_v2. Under masking
MethylGPT-2 recomputes embedding, biomarkers and factor scores from the remaining CpGs and
missing residuals take the fitting mean; raw CpGs take the fitting mean. The masks are a
methodological stress test applied to measured data.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from methylgpt_lib import cuda_preinit, ARTIFACTS, log, require, write_json
from methylgpt_model import BioHeads, embed_batch, load_checkpoint
from gpu_coxnet_2026_10_01 import select_and_fit  # noqa: E402

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"


def counts(ev, du, r):
    _, c, d, t, _ = concordance_index_censored(ev, du, r)
    return np.array([c, d, t], float)


def pooled(ev, du, r, folds):
    t = sum(counts(ev[s], du[s], r[s]) for s in folds)
    return (t[0] + 0.5 * t[2]) / t.sum()


def fit_scaler(F, dev):
    """Fitting-subject mean and scale (NaN-aware, torch)."""
    A = torch.as_tensor(F, device=dev, dtype=torch.float32)
    fin = torch.isfinite(A)
    mu = torch.where(fin, A, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    A = torch.where(fin, A, mu)
    sd = A.std(0)
    sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
    return mu.cpu().numpy(), sd.cpu().numpy()


def apply_scaler(F, mu, sd, dev):
    A = torch.as_tensor(F, device=dev, dtype=torch.float32)
    m = torch.as_tensor(mu, device=dev); s = torch.as_tensor(sd, device=dev)
    return ((torch.where(torch.isfinite(A), A, m) - m) / s).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--scenarios", default="full,random_30,random_50,random_70,random_90,epic_v2,hm450,hm27")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "gs_cox/factor_residual_2026-10-01"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    out = Path(a.out_dir); (out / "models").mkdir(parents=True, exist_ok=True)

    mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mpath, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))

    ag = torch.load(a.aging, map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    model.load_state_dict(ag["model"]); model.eval()
    mt = torch.load(a.multitask, map_location=dev, weights_only=False)
    mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
    mt_model.load_state_dict(mt["model"]); mt_model.eval()
    mt_heads = BioHeads(mt_core.rank + mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
    mt_heads.load_state_dict(mt["heads"]); mt_heads.eval()

    def features(Xs):
        """[embedding, biomarkers, core residual] for every subject; residual NaN where unobserved."""
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                obs = torch.isfinite(x)
                E.append(embed_batch(model, core, x, obs).float().cpu().numpy())
                B.append(mt_heads(embed_batch(mt_model, mt_core, x, obs))[3].float().cpu().numpy())
                s = core.posterior(x, obs.float())
                R.append(torch.where(obs, x - core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        return np.concatenate(E), np.concatenate(B), np.concatenate(R)

    preds = pd.read_parquet(Path(a.mgpt2_run) / "private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = a.endpoints.split(",") if a.endpoints else sorted(preds.endpoint.unique())

    log("computing full-coverage features")
    E, B, R = features(X)
    ref = np.load(require(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mgpt2_embed.embedding.npz"), allow_pickle=False)
    assert (ref["dnam_id"] == ids).all()
    log(f"embedding vs stored features: max abs diff {np.abs(E - ref['embedding']).max():.2e}")
    FR = np.hstack([E, B, R]).astype(np.float32)
    del E, B, R
    log(f"factor+residual features {FR.shape}")

    # ---- fit fold models at full coverage ----
    fold_models = {}
    pred_rows = {"factor_residual": [], "raw_cpg_gpu": []}
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        idx = np.array([pos[s] for s in pe.dnam_id.astype(str)])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(np.float32)
        risks = {k: np.empty(len(pe)) for k in pred_rows}
        for f in sorted(pe.outer_fold.unique()):
            held = np.flatnonzero(pe.outer_fold.to_numpy() == f)
            fit = np.flatnonzero(pe.outer_fold.to_numpy() != f)
            cm, cs = fit_scaler(cov[fit], dev)
            for name, F in (("factor_residual", FR), ("raw_cpg_gpu", X)):
                fm, fs = fit_scaler(F[idx[fit]], dev)
                Xf = np.hstack([apply_scaler(cov[fit], cm, cs, dev), apply_scaler(F[idx[fit]], fm, fs, dev)])
                w, alpha, k = select_and_fit(Xf, du[fit], ev[fit], seed=20260928 + int(f), dev=dev, iters=a.iters)
                del Xf
                mdl = {"cov_mean": cm, "cov_scale": cs, "mean": fm, "scale": fs, "coefficient": w.astype(np.float32),
                       "alpha": alpha, "alpha_index": k}
                np.savez(out / "models" / f"{name}_{ep}_outer{f}.npz", **mdl)
                fold_models[(name, ep, int(f))] = mdl
                Xh = np.hstack([apply_scaler(cov[held], cm, cs, dev), apply_scaler(F[idx[held]], fm, fs, dev)])
                risks[name][held] = Xh @ w
                log(f"{ep} fold {f} {name}: alpha index {k}, nonzero {int((w[2:] != 0).sum())}")
        folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
        for name in pred_rows:
            q = pe[["dnam_id", "duration", "event", "age", "sex", "endpoint", "age_adjustment", "outer_fold", "baseline_risk"]].copy()
            q["risk"] = risks[name]
            pred_rows[name].append(q)
        log(f"{ep}: factor+residual {pooled(ev, du, risks['factor_residual'], folds):.4f} raw (GPU) {pooled(ev, du, risks['raw_cpg_gpu'], folds):.4f}")
    for name, rows in pred_rows.items():
        (out / name).mkdir(exist_ok=True)
        pd.concat(rows).to_parquet(out / name / "private_nested_predictions.parquet", index=False)
    del FR

    # ---- apply the fold models under missing-CpG scenarios ----
    res = {"settings": vars(a), "scenarios": {}, "observed_fraction": {}}
    for scen in a.scenarios.split(","):
        Xs = X.copy()
        if scen.startswith("random_"):
            p = int(scen.split("_")[1]) / 100
            r = np.random.default_rng(20261001 + int(p * 100))
            Xs[r.random(Xs.shape) < p] = np.nan
        elif scen != "full":
            Xs[:, ~plat[scen]] = np.nan
        res["observed_fraction"][scen] = float(np.isfinite(Xs).mean())
        E, B, R = features(Xs)
        FRs = np.hstack([E, B, R]).astype(np.float32)
        del E, B, R
        block = {}
        for ep in endpoints:
            pe = preds[preds.endpoint == ep].reset_index(drop=True)
            idx = np.array([pos[s] for s in pe.dnam_id.astype(str)])
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            cov = pe[["age", "sex"]].to_numpy(np.float32)
            folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
            row = {}
            for name, F in (("factor_residual", FRs), ("raw_cpg_gpu", Xs)):
                r_ = np.empty(len(pe))
                for f, held in zip(sorted(pe.outer_fold.unique()), folds):
                    m = fold_models[(name, ep, int(f))]
                    Xh = np.hstack([apply_scaler(cov[held], m["cov_mean"], m["cov_scale"], dev),
                                    apply_scaler(F[idx[held]], m["mean"], m["scale"], dev)])
                    r_[held] = Xh @ m["coefficient"]
                row[name] = float(pooled(ev, du, r_, folds))
                if scen == "full":
                    stored = pd.concat(pred_rows[name]).query("endpoint == @ep").risk.to_numpy()
                    if np.max(np.abs(stored - r_)) > 1e-4 * (1 + np.abs(stored).max()):
                        raise ValueError(f"full-scenario risks differ from fitted risks for {ep} {name}")
            block[ep] = row
        res["scenarios"][scen] = block
        log(f"{scen} (observed {res['observed_fraction'][scen]:.3f}): " +
            ", ".join(f"{ep[:10]} FR {v['factor_residual']:.3f} raw {v['raw_cpg_gpu']:.3f}" for ep, v in block.items()))
        write_json(out / "scenarios.json", res)
    res["elapsed_s"] = time.time() - t0
    write_json(out / "scenarios.json", res)


if __name__ == "__main__":
    main()
