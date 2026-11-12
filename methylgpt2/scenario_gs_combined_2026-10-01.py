#!/usr/bin/env python3
"""Held-out risks of every fitted GS model under the missing-CpG scenarios (development subjects).

Fold models are reused unchanged: the locked-protocol MethylGPT-2 embedding and biomarker runs,
the v12 raw-CpG runs, and the GPU factor+residual and raw-CpG runs of
gs_factor_residual_cox_2026-10-01.py. Only held-out inputs are degraded (random_<p> removes p% of
each subject's observed CpGs with the seeds of scenario_gs_missingness_v2; platform scenarios
keep the CpGs on that array). MethylGPT-2 recomputes embedding, biomarkers, factor scores and
residuals from the remaining CpGs; raw-CpG models fill missing CpGs with fitting means.

The pre-specified combined MethylGPT-2 model (fixed before its full-coverage results were seen)
is the equal-weight sum of the within-fold standardised risks of the embedding, biomarker and
factor+residual models. Risks are written to a parquet file (controlled-data derived, stays in
artifacts/) so any comparison can be bootstrapped later. The masks are a methodological stress
test applied to measured data.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import cuda_preinit, ARTIFACTS, log, require, write_json
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_DIRS = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]
COMBOS = {"mgpt2_combined": ["embedding", "biomarkers", "factor_residual"],
          "ensemble_embed_bio": ["embedding", "biomarkers"]}


def counts(ev, du, r):
    _, c, d, t, _ = concordance_index_censored(ev, du, r)
    return np.array([c, d, t], float)


def pooled(ev, du, r, folds):
    t = sum(counts(ev[s], du[s], r[s]) for s in folds)
    return (t[0] + 0.5 * t[2]) / t.sum()


def locked_risk(model, feats, cov):
    z = np.where(np.isfinite(feats), feats, model["mean"])
    z = (z - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return c @ model["coefficient"][:2] + z @ model["coefficient"][2:]


def gpu_risk(model, feats, cov):
    z = np.where(np.isfinite(feats), feats, model["mean"])
    z = (z - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return np.hstack([c, z]) @ model["coefficient"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--bio-run", default=str(ARTIFACTS / "gs_cox/mt_bio_linear_2026-10-01"))
    ap.add_argument("--fr-dir", default=str(ARTIFACTS / "gs_cox/factor_residual_2026-10-01"))
    ap.add_argument("--scenarios", default="full,random_30,random_50,random_70,random_90,epic_v2,hm450,hm27")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/scenario_combined_2026-10-01"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fr_dir = Path(a.fr_dir)

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
    endpoints = sorted(preds.endpoint.unique())
    stored = {name: pd.read_parquet(require(fr_dir / name / "private_nested_predictions.parquet")) for name in ("factor_residual", "raw_cpg_gpu")}
    bio_stored = pd.read_parquet(Path(a.bio_run) / "private_nested_predictions.parquet")
    bio_stored = bio_stored[bio_stored.age_adjustment == "linear"]

    res = {"settings": vars(a), "combos": COMBOS, "scenarios": {}, "observed_fraction": {}}
    risk_frames = []
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
        FR = np.hstack([E, B, R]).astype(np.float32)
        del R
        block = {}
        for ep in endpoints:
            pe = preds[preds.endpoint == ep].reset_index(drop=True)
            idx = np.array([pos[s] for s in pe.dnam_id.astype(str)])
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            cov = pe[["age", "sex"]].to_numpy(float)
            fold_ids = sorted(pe.outer_fold.unique())
            folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in fold_ids]
            risk = {k: np.empty(len(pe)) for k in ("embedding", "biomarkers", "factor_residual", "raw_cpg_v12", "raw_cpg_gpu")}
            for f, held in zip(fold_ids, folds):
                mm = dict(np.load(Path(a.mgpt2_run) / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                bm = dict(np.load(Path(a.bio_run) / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                rdir = [d for d in RAW_DIRS if (d / f"{ep}_linear_outer{f}_model.npz").exists()][0]
                rm = dict(np.load(rdir / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                risk["embedding"][held] = locked_risk(mm, E[idx[held]].astype(np.float64), cov[held])
                risk["biomarkers"][held] = locked_risk(bm, B[idx[held]].astype(np.float64), cov[held])
                risk["raw_cpg_v12"][held] = locked_risk(rm, Xs[idx[held]].astype(np.float64), cov[held])
                for name, F in (("factor_residual", FR), ("raw_cpg_gpu", Xs)):
                    gm = dict(np.load(require(fr_dir / "models" / f"{name}_{ep}_outer{f}.npz"), allow_pickle=False))
                    risk[name][held] = gpu_risk(gm, F[idx[held]], cov[held].astype(np.float32))
            if scen == "full":  # the reused fold models must reproduce their stored out-of-fold risks
                checks = {"embedding": pe.risk.to_numpy(float),
                          "biomarkers": bio_stored[bio_stored.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)}
                for name in ("factor_residual", "raw_cpg_gpu"):
                    s_ = stored[name]; checks[name] = s_[s_.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)
                for name, ref in checks.items():
                    if np.max(np.abs(ref - risk[name])) > 1e-3 * (1 + np.abs(ref).max()):
                        raise ValueError(f"recomputed {name} risks differ from stored ones for {ep}")
            for cname, parts in COMBOS.items():
                v = np.zeros(len(pe))
                for held in folds:
                    for k in parts:
                        x_ = risk[k][held]; v[held] += (x_ - x_.mean()) / x_.std()
                risk[cname] = v
            block[ep] = {k: float(pooled(ev, du, v, folds)) for k, v in risk.items()}
            fr_ = pe[["dnam_id", "endpoint", "outer_fold", "event", "duration", "age", "sex", "baseline_risk"]].copy()
            fr_["scenario"] = scen
            for k, v in risk.items():
                fr_[k] = v
            risk_frames.append(fr_)
        res["scenarios"][scen] = block
        log(f"{scen} (observed {res['observed_fraction'][scen]:.3f}): " + ", ".join(
            f"{ep[:10]} comb {v['mgpt2_combined']:.3f} FR {v['factor_residual']:.3f} ens {v['ensemble_embed_bio']:.3f} raw {v['raw_cpg_v12']:.3f}/{v['raw_cpg_gpu']:.3f}"
            for ep, v in block.items()))
        write_json(out / "scenario_cindex.json", res)
        pd.concat(risk_frames).to_parquet(out / "private_scenario_risks.parquet", index=False)
    res["elapsed_s"] = time.time() - t0
    write_json(out / "scenario_cindex.json", res)


if __name__ == "__main__":
    main()
