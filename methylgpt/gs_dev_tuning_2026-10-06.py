#!/usr/bin/env python3
"""Development-only tuning of the MethylGPT disease models (Generation Scotland, 9,866 training-subset participants).

The test subset is not read. Same five outer folds, inner folds, GPU estimator and seeds as the development
analyses (gs_stacked_2026-10-04.py). Candidate components (age and sex always unpenalized):
  E        embedding (2,048), penalized
  B1       25 distilled outputs of run 1, penalized                      (= development "B")
  B3       25 distilled outputs averaged over the three runs, penalized
  R        [E, B1, factor-model residuals], all penalized                (= development "R")
  B3u_res  B3 unpenalized + factor-model residuals penalized
  B3u_E_res B3 unpenalized + [E, residuals] penalized
Stacks: weights on a 0.1 simplex grid chosen per outer fold from inner out-of-fold risks of the fitting rows
(as gs_stacked_2026-10-04), for several component sets. Comparators: the stored out-of-fold risks of the dense
raw-CpG GPU fit and of GrimAge2 + DunedinPACE (same folds). Paired bootstrap of the mean C-index over endpoints
(1,000 draws, participants resampled within outer folds per endpoint, seed 20261006).
Outputs (aggregate): artifacts/gs_tuning_2026-10-06/tuning.json; private OOF risks in the same folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from methylgpt_model import embed_batch, load_checkpoint

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stk", HERE / "gs_stacked_2026-10-04.py")
stk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stk)

SPLIT = stk.SPLIT
CACHE = stk.CACHE
OUT = ARTIFACTS / "gs_tuning_2026-10-06"
FEAT1 = ARTIFACTS / "gs_features_2026-10-01"
FEATS = ARTIFACTS / "gs_features_seeds_2026-10-04"
PREDS = ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet"
COMP_STORED = ARTIFACTS / "gs_cox/stacked_2026-10-04/private_component_risks.parquet"
RAW_GPU = ARTIFACTS / "gs_cox/factor_residual_2026-10-01/raw_cpg_gpu/private_nested_predictions.parquet"
GD = ARTIFACTS / "gs_cox/clock_comparators_2026-10-01/grimage2_dunedinpace/private_nested_predictions.parquet"
AGING = ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"
STACKS = {"S_EB1R": ("E", "B1", "R"), "S_EB3R": ("E", "B3", "R"), "S_EB3_B3ures": ("E", "B3", "B3u_res"),
          "S_E_B3_R_B3ures": ("E", "B3", "R", "B3u_res"), "S_B3_B3ures": ("B3", "B3u_res")}


def npz(folder, split, tag):
    z = np.load(require(folder / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{tag}.embedding.npz"), allow_pickle=False)
    return z["dnam_id"].astype(str), z["embedding"].astype(np.float32)


def residuals(split, dev):
    mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    ag = torch.load(AGING, map_location=dev, weights_only=False)
    _, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            x = torch.from_numpy(X[i:i + 64]).to(dev)
            obs = torch.isfinite(x)
            s = core.posterior(x, obs.float())
            out.append(torch.where(obs, x - core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
    return ids, np.concatenate(out)


def simplex(k, step=0.1):
    n = int(round(1 / step))
    return np.array([c for c in itertools.product(range(n + 1), repeat=k) if sum(c) == n], float) / n


def learn(oof, comps, assign, ev, du, dev):
    Z = np.stack([stk.zfold(oof[c], assign) for c in comps])
    G = simplex(len(comps))
    tot = np.zeros((len(G), 3))
    for j in np.unique(assign):
        s = assign == j
        tot += stk.PairCounter(ev[s], du[s], dev).counts(G @ Z[:, s])
    c = stk.cidx(tot)
    cand = np.flatnonzero(c >= c.max() - 1e-12)
    pick = cand[np.argmin(((G[cand] - 1 / len(comps)) ** 2).sum(1))]
    return G[pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--components", default="E,B1,B3,R,B3u_res,B3u_E_res")
    a = ap.parse_args()
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    dev = a.device
    OUT.mkdir(parents=True, exist_ok=True)
    comps_run = a.components.split(",")

    ids, E = npz(FEAT1, "train", "mgpt2_embed")
    i1, B1 = npz(FEAT1, "train", "mt_bio")
    i2, B2 = npz(FEATS, "train", "mtseed2_bio")
    i3, B3s = npz(FEATS, "train", "mtseed3_bio")
    assert (i1 == ids).all() and (i2 == ids).all() and (i3 == ids).all()
    B3 = (B1 + B2 + B3s) / 3
    rid, RES = residuals("train", dev)
    assert (rid == ids).all()
    pos = {s: i for i, s in enumerate(ids)}
    # name -> (unpenalized feature block or None, penalized feature block)
    spec = {"E": (None, E), "B1": (None, B1), "B3": (None, B3), "R": (None, np.hstack([E, B1, RES])),
            "B3u_res": (B3, RES), "B3u_E_res": (B3, np.hstack([E, RES]))}

    preds = pd.read_parquet(require(PREDS))
    preds = preds[preds.age_adjustment == "linear"].copy(); preds["dnam_id"] = preds.dnam_id.astype(str)
    endpoints = sorted(preds.endpoint.unique())
    ref = {}
    for nm, p in (("raw_cpg_dense", RAW_GPU), ("grimage2_dunedinpace", GD)):
        d = pd.read_parquet(require(p)); d = d[d.age_adjustment == "linear"].copy(); d["dnam_id"] = d.dnam_id.astype(str)
        ref[nm] = d.set_index(["endpoint", "dnam_id"]).risk
    stored = pd.read_parquet(require(COMP_STORED)); stored["dnam_id"] = stored.dnam_id.astype(str)
    stored = stored.set_index(["endpoint", "dnam_id"])

    frames, info = [], {}
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        idx = np.array([pos[s] for s in pe.dnam_id])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(np.float32)
        of = pe.outer_fold.to_numpy()
        risk = {c: np.empty(len(pe)) for c in comps_run}
        stacks = {s: np.empty(len(pe)) for s in STACKS if set(STACKS[s]) <= set(comps_run)}
        info[ep] = {}
        for f in sorted(np.unique(of)):
            held, fit = np.flatnonzero(of == f), np.flatnonzero(of != f)
            oof, assign0 = {}, None
            for c in comps_run:
                unp, pen = spec[c]
                blocks_fit = [cov[fit]] + ([unp[idx[fit]]] if unp is not None else []) + [pen[idx[fit]]]
                blocks_held = [cov[held]] + ([unp[idx[held]]] if unp is not None else []) + [pen[idx[held]]]
                Xf_parts, Xh_parts = [], []
                for bf, bh in zip(blocks_fit, blocks_held):
                    m, s = stk.fit_scaler(bf, dev)
                    Xf_parts.append(stk.apply_scaler(bf, m, s, dev)); Xh_parts.append(stk.apply_scaler(bh, m, s, dev))
                Xf, Xh = np.hstack(Xf_parts), np.hstack(Xh_parts)
                n_unpen = 2 + (unp.shape[1] if unp is not None else 0)
                w, alpha, k, oo, assign, sc = stk.select_and_fit_oof(Xf, du[fit], ev[fit], seed=20260928 + int(f), n_unpen=n_unpen, dev=dev, iters=a.iters)
                if assign0 is None:
                    assign0 = assign
                elif not (assign == assign0).all():
                    raise ValueError("inner folds differ between components")
                oof[c] = oo
                risk[c][held] = Xh @ w
                info[ep].setdefault(c, []).append({"fold": int(f), "alpha_index": k, "inner_mean_c": float(sc[k]), "nonzero_penalized": int((w[n_unpen:] != 0).sum())})
                del Xf, Xh
            for sname in stacks:
                cs = STACKS[sname]
                wv = learn(oof, cs, assign0, ev[fit], du[fit], dev)
                stacks[sname][held] = sum(wv[i] * stk.zfold(risk[c][held], np.zeros(len(held))) for i, c in enumerate(cs))
                info[ep].setdefault(sname, []).append({"fold": int(f), "weights": dict(zip(cs, map(float, wv)))})
            log(f"{ep} fold {f} done ({time.time() - t0:.0f}s)")
        q = pe[["dnam_id", "endpoint", "outer_fold", "event", "duration"]].copy()
        for c in comps_run:
            q[c] = risk[c]
        for sname, v in stacks.items():
            q[sname] = v
        key = list(zip([ep] * len(pe), pe.dnam_id))
        for nm in ref:
            q[nm] = ref[nm].loc[key].to_numpy(float)
        q["stored_R"] = stored.risk_R.loc[key].to_numpy(float)
        q["stored_stacked"] = stored.risk_stacked.loc[key].to_numpy(float)
        frames.append(q)
        folds = [np.flatnonzero(of == f) for f in sorted(np.unique(of))]
        log(f"{ep}: " + ", ".join(f"{c} {stk.cidx(sum(stk.PairCounter(ev[s], du[s], dev).counts(q[c].to_numpy()[s][None])[0] for s in folds)):.4f}"
                                  for c in list(risk) + list(stacks) + list(ref)))
        pd.concat(frames).to_parquet(OUT / "private_dev_oof_risks.parquet", index=False)

    R = pd.concat(frames, ignore_index=True)
    models = [c for c in R.columns if c not in ("dnam_id", "endpoint", "outer_fold", "event", "duration")]
    rng = np.random.default_rng(20261006)
    boot, point = {}, {}
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        ev, du, of = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float), g.outer_fold.to_numpy()
        tot = np.zeros((len(models), a.draws, 3)); tot0 = np.zeros((len(models), 3))
        for f in sorted(np.unique(of)):
            s = np.flatnonzero(of == f)
            pc = stk.PairCounter(ev[s], du[s], dev)
            ii = rng.integers(0, len(s), size=(a.draws, len(s)))
            W = np.zeros((a.draws, len(s)), np.float32); np.add.at(W, (np.repeat(np.arange(a.draws), len(s)), ii.ravel()), 1)
            M = np.stack([g[m].to_numpy(float)[s] for m in models])
            tot += pc.counts(M, W); tot0 += pc.counts(M)
        boot[ep] = {m: stk.cidx(tot[j]) for j, m in enumerate(models)}
        point[ep] = {m: float(stk.cidx(tot0[j])) for j, m in enumerate(models)}
    mean_pt = {m: float(np.mean([point[e][m] for e in endpoints])) for m in models}
    contr = {}
    for m in models:
        for c in ("raw_cpg_dense", "grimage2_dunedinpace"):
            d = np.mean([boot[e][m] - boot[e][c] for e in endpoints], 0)
            contr[f"{m} - {c}"] = {"difference": mean_pt[m] - mean_pt[c], "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    write_json(OUT / "tuning.json", {"per_endpoint_c": point, "mean_c": mean_pt, "contrasts_mean": contr, "fold_info": info,
                                     "test_subset_read": False, "elapsed_s": time.time() - t0})
    for m in sorted(models, key=lambda m: -mean_pt[m]):
        cr, cg = contr[f"{m} - raw_cpg_dense"], contr[f"{m} - grimage2_dunedinpace"]
        log(f"{m:22s} mean C {mean_pt[m]:.4f}  vs raw {cr['difference']:+.4f} [{cr['ci95'][0]:+.4f},{cr['ci95'][1]:+.4f}]  "
            f"vs GD {cg['difference']:+.4f} [{cg['ci95'][0]:+.4f},{cg['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
