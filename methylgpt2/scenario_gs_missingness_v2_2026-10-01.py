#!/usr/bin/env python3
"""Deployment with missing CpGs: Generation Scotland incident disease (development subjects).

The fold-specific Cox models of the locked nested protocol are reused unchanged (raw-CpG runs
from the v12 publication checkout; MethylGPT-2 embedding run from this branch). Only the
held-out subjects' methylation inputs are degraded, as happens when a fitted model is applied
to data from another array or with failed probes:
  full            no change (must reproduce the stored C-indices)
  random_<p>      p% of each subject's observed vocabulary CpGs removed at random (seeded)
  epic_v2, hm450, hm27   only CpGs present on that platform remain
Raw-CpG models fill missing CpGs with the fitting-fold mean, as their preprocessing does.
MethylGPT-2 recomputes its embedding from the remaining CpGs. Combined C-index pools
within-fold comparable pairs; paired bootstrap (500 draws within folds) for MethylGPT-2 minus
raw CpG. The artificial masks are a methodological stress test of real data.
v2 adds the distilled-biomarker model (25 predicted full-array biomarkers from the multi-task
checkpoint) and the fixed equal-weight ensemble of embedding and biomarker risks; masks and
seeds are identical to v1.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import cuda_preinit, ARTIFACTS, log, require, write_json
from methylgpt2_model import BioHeads, Heads, embed_batch, load_checkpoint

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_DIRS = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]


def counts(ev, du, r):
    _, c, d, t, _ = concordance_index_censored(ev, du, r)
    return np.array([c, d, t], float)


def cidx(c):
    return (c[0] + 0.5 * c[2]) / c.sum()


def risk(model, feats, cov):
    z = np.where(np.isfinite(feats), feats, model["mean"])
    z = (z - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    coef = model["coefficient"]
    return c @ coef[:2] + z @ coef[2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--bio-run", default=str(ARTIFACTS / "gs_cox/mt_bio_linear_2026-10-01"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/scenario_missingness_v2_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
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

    def bio(Xs):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                out.append(mt_heads(embed_batch(mt_model, mt_core, x, torch.isfinite(x)))[3].float().cpu().numpy())
        return np.concatenate(out)

    def embed(Xs):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                out.append(embed_batch(model, core, x, torch.isfinite(x)).float().cpu().numpy())
        return np.concatenate(out)

    scenarios = {"full": None}
    for p in (30, 50, 70, 90):
        scenarios[f"random_{p}"] = ("random", p / 100)
    for k in ("epic_v2", "hm450", "hm27"):
        scenarios[k] = ("platform", plat[k])
    preds = pd.read_parquet(Path(a.mgpt2_run) / "private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = sorted(preds.endpoint.unique())
    res = {"scenarios": {}, "observed_fraction": {}}
    rng_b = np.random.default_rng(20260928)
    for name, spec in scenarios.items():
        Xs = X.copy()
        if spec is not None and spec[0] == "random":
            r = np.random.default_rng(20261001 + int(spec[1] * 100))
            Xs[r.random(Xs.shape) < spec[1]] = np.nan
        elif spec is not None:
            Xs[:, ~spec[1]] = np.nan
        res["observed_fraction"][name] = float(np.isfinite(Xs).mean())
        E = embed(Xs)
        BI = bio(Xs)
        block = {}
        for ep in endpoints:
            pe = preds[preds.endpoint == ep]
            idx = np.array([pos[s] for s in pe.dnam_id.astype(str)])
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            cov = pe[["age", "sex"]].to_numpy(float)
            r_m, r_r, r_b, r_e = (np.empty(len(pe)) for _ in range(4))
            folds = []
            for f in sorted(pe.outer_fold.unique()):
                sel = np.flatnonzero(pe.outer_fold.to_numpy() == f)
                folds.append(sel)
                mm = dict(np.load(Path(a.mgpt2_run) / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                rdir = [d for d in RAW_DIRS if (d / f"{ep}_linear_outer{f}_model.npz").exists()][0]
                rm = dict(np.load(rdir / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                r_m[sel] = risk(mm, E[idx[sel]], cov[sel])
                bm = dict(np.load(Path(a.bio_run) / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                r_b[sel] = risk(bm, BI[idx[sel]], cov[sel])
                z_ = lambda v: (v - v.mean()) / v.std()
                r_e[sel] = z_(r_m[sel]) + z_(r_b[sel])
                r_r[sel] = risk(rm, Xs[idx[sel]].astype(np.float64), cov[sel])
            if name == "full":
                stored = pe.risk.to_numpy(float)
                if np.max(np.abs(stored - r_m)) > 1e-3 * (1 + np.abs(stored).max()):
                    raise ValueError(f"Recomputed MethylGPT-2 risks differ from stored ones for {ep}")
            C = {k: cidx(sum(counts(ev[s_], du[s_], r[s_]) for s_ in folds))
                 for k, r in (("embedding", r_m), ("biomarkers", r_b), ("ensemble", r_e), ("raw_cpg", r_r))}
            diffs = []
            for _ in range(a.draws):
                tm, tr = np.zeros(3), np.zeros(3)
                for s_ in folds:
                    b = rng_b.choice(s_, len(s_), replace=True)
                    tm += counts(ev[b], du[b], r_e[b]); tr += counts(ev[b], du[b], r_r[b])
                diffs.append(cidx(tm) - cidx(tr))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            block[ep] = C | {"ensemble_minus_raw": C["ensemble"] - C["raw_cpg"], "ci95": [float(lo), float(hi)]}
        res["scenarios"][name] = block
        log(f"{name} (observed {res['observed_fraction'][name]:.3f}): " +
            ", ".join(f"{ep[:10]} ens {v['ensemble']:.3f} bio {v['biomarkers']:.3f} raw {v['raw_cpg']:.3f}" for ep, v in block.items()))
        write_json(Path(a.out), res)
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
