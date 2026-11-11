#!/usr/bin/env python3
"""Learned-weight (stacked) MethylGPT-2 disease model in Generation Scotland (development subjects).

Question: do combination weights learned by inner cross-validation remove the deficit of the
locked equal-weight combined model (mgpt2_combined_2026-10-01) against the dense raw-CpG GPU Cox
fit, in particular for Endocrine_and_Metabolic_Diseases?

Inputs
  - GS train-split methylation matrix (_frame_cache, 9,866 participants x 49,156 CpGs) and the
    aging-heads / multitask checkpoints; features are computed with the code of
    gs_factor_residual_cox_2026-10-01.py / scenario_gs_combined_2026-10-01.py at full coverage:
      E = MethylGPT-2 embedding (2,048), B = 25 distilled-clock outputs,
      R = [E, B, core residuals] (51,229; residual NaN where unobserved -> fitting mean).
  - Endpoints, durations, events, age, sex, baseline_risk and the locked outer folds from
    artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet (linear).
  - Comparator out-of-fold risks: raw_cpg_v12 (v12 nested raw-CpG runs), raw_cpg_gpu
    (factor_residual_2026-10-01/raw_cpg_gpu), clock comparators (grimage2, dunedinpace,
    grimage2_dunedinpace) and the locked equal-weight combined model.

Method
  1. For each endpoint and outer fold k (fitting rows = the other four folds) and each component
     (E, B, R): GPU elastic-net Cox with the settings of gpu_coxnet_2026_10_01.select_and_fit
     (l1 0.5, 12 alphas, ratio 0.05, age+sex unpenalised, standardisation from fitting rows,
     inner StratifiedKFold(3, shuffle, random_state=20260928+k), iters 200). The variant here
     also returns every fitting row's inner out-of-fold risk at the selected alpha.
  2. Weights (wE, wB, wR) on the 0.1 simplex grid maximise the C-index pooled within inner folds
     of the inner-fold-standardised inner out-of-fold risks (fitting rows only); ties go to the
     grid point closest (Euclidean) to equal weights, then to the first such point in grid order.
  3. Held-out fold: each component's outer risk (refit on all fitting rows) is standardised
     within the held-out fold and combined with the learned weights (stacked); the equal-weight
     sum of the same GPU components is formed for reference.
  4. Pooled C-index (counts pooled within outer folds).
  5. Paired bootstrap (1,000 draws, subjects resampled within outer folds, seed 20261004; the
     same draws are shared by all comparators of an endpoint) of stacked minus each comparator,
     BH across all endpoint x comparator contrasts. C-index counts use an exact pairwise GPU
     implementation of sksurv's concordance_index_censored (checked against sksurv).
  6. Reproduction checks: Pearson correlation of GPU R / E / B outer risks with the saved
     factor_residual and locked sksurv embedding / biomarker risks.

Outputs (participant-level files stay under artifacts/, which is git-ignored)
  artifacts/gs_cox/stacked_2026-10-04/private_nested_predictions.parquet (stacked risks)
  artifacts/gs_cox/stacked_2026-10-04/private_component_risks.parquet (E, B, R, equal-weight)
  artifacts/gs_cox/stacked_2026-10-04.json, artifacts/gs_cox/paired_bootstrap_stacked_2026-10-04.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import cuda_preinit, ARTIFACTS, log, require, write_json
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint
from gpu_coxnet_2026_10_01 import fit_path, cindex

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_V12 = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]
COMPONENTS = ("E", "B", "R")
GRID = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
TIED_TOL = 1e-8  # sksurv default


# ---------------------------------------------------------------- feature scaling (as in FR script)
def fit_scaler(F, dev):
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


# ---------------------------------------------------------------- select_and_fit with inner OOF risk
def select_and_fit_oof(X, time_, event, seed, n_unpen=2, folds=3, dev="cuda", **kw):
    """As gpu_coxnet_2026_10_01.select_and_fit, plus the inner out-of-fold risk at the selected
    alpha for every fitting row and the inner fold assignment."""
    inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = None
    oof = None
    assign = np.full(len(X), -1)
    for j, (fit_i, ev_i) in enumerate(inner.split(X, event)):
        alphas, path = fit_path(X[fit_i], time_[fit_i], event[fit_i], n_unpen, dev=dev, **kw)
        Xe = torch.as_tensor(X[ev_i], device=dev, dtype=torch.float32)
        rk = np.stack([(Xe @ w).cpu().numpy() for w in path])
        if oof is None:
            oof = np.empty((len(path), len(X)))
        oof[:, ev_i] = rk
        assign[ev_i] = j
        s = np.array([cindex(event[ev_i], time_[ev_i], r) for r in rk])
        scores = s if scores is None else scores + s
    alphas, path = fit_path(X, time_, event, n_unpen, dev=dev, **kw)
    k = int(np.argmax(scores))
    return path[k].cpu().numpy(), float(alphas[k]), k, oof[k], assign, scores / folds


# ---------------------------------------------------------------- exact pairwise concordance on GPU
class PairCounter:
    """Concordant / discordant / tied-risk counts of sksurv.concordance_index_censored for one
    subject set, optionally with bootstrap multiplicity weights (duplicates of a subject have equal
    times, so self pairs are never comparable, matching sksurv on the resampled data)."""

    def __init__(self, ev, du, dev):
        e = torch.as_tensor(ev.astype(bool), device=dev)
        t = torch.as_tensor(du, device=dev, dtype=torch.float64)
        self.comp = e[:, None] & ((t[None, :] > t[:, None]) | ((t[None, :] == t[:, None]) & ~e[None, :]))
        self.dev = dev

    def counts(self, risks, W=None):
        """risks: (m, n) array. W: (b, n) weights or None. Returns (m, 3) or (m, b, 3) float64."""
        R = torch.as_tensor(np.atleast_2d(risks), device=self.dev, dtype=torch.float64)
        out = []
        Wt = None if W is None else torch.as_tensor(W, device=self.dev, dtype=torch.float64)
        for r in R:
            d = r[None, :] - r[:, None]  # r_j - r_i
            tie = d.abs() <= TIED_TOL
            con = self.comp & (d < 0) & ~tie
            tied = self.comp & tie
            dis = self.comp & ~tie & (d > 0)
            mats = [con, dis, tied]
            if Wt is None:
                out.append(torch.stack([m.sum().double() for m in mats]))
            else:
                out.append(torch.stack([((Wt @ m.double()) * Wt).sum(1) for m in mats], 1))
        return torch.stack(out).cpu().numpy()


def cidx(cnt):
    cnt = np.asarray(cnt, float)
    return (cnt[..., 0] + 0.5 * cnt[..., 2]) / cnt.sum(-1)


def zfold(r, groups):
    out = np.empty(len(r))
    for g in np.unique(groups):
        s = groups == g
        out[s] = (r[s] - r[s].mean()) / r[s].std(ddof=1)
    return out


def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def learn_weights(oof, assign, ev, du, dev):
    """oof: dict comp -> inner OOF risk. Returns best weights, best pooled inner C and grid table."""
    Z = np.stack([zfold(oof[c], assign) for c in COMPONENTS])  # (3, n)
    G = np.array(GRID)  # (66, 3)
    tot = np.zeros((len(G), 3))
    for j in np.unique(assign):
        s = assign == j
        pc = PairCounter(ev[s], du[s], dev)
        tot += pc.counts(G @ Z[:, s])
    c = cidx(tot)
    best = c.max()
    cand = np.flatnonzero(c >= best - 1e-12)
    dist = ((G[cand] - 1 / 3) ** 2).sum(1)
    pick = cand[np.argmin(dist)]
    return G[pick], float(c[pick]), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--bio-run", default=str(ARTIFACTS / "gs_cox/mt_bio_linear_2026-10-01"))
    ap.add_argument("--fr-dir", default=str(ARTIFACTS / "gs_cox/factor_residual_2026-10-01"))
    ap.add_argument("--clock-dir", default=str(ARTIFACTS / "gs_cox/clock_comparators_2026-10-01"))
    ap.add_argument("--combined-run", default=str(ARTIFACTS / "gs_cox/mgpt2_combined_2026-10-01"))
    ap.add_argument("--endpoints", default="")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--skip-fit", action="store_true", help="reuse saved component risks")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "gs_cox/stacked_2026-10-04"))
    ap.add_argument("--out-json", default=str(ARTIFACTS / "gs_cox/stacked_2026-10-04.json"))
    ap.add_argument("--out-csv", default=str(ARTIFACTS / "gs_cox/paired_bootstrap_stacked_2026-10-04.csv"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    comp_path = out / "private_component_risks.parquet"
    wpath = out / "learned_weights.json"

    preds = pd.read_parquet(require(Path(a.mgpt2_run) / "private_nested_predictions.parquet"))
    preds = preds[preds.age_adjustment == "linear"].copy()
    preds["dnam_id"] = preds.dnam_id.astype(str)
    endpoints = a.endpoints.split(",") if a.endpoints else sorted(preds.endpoint.unique())

    # ------------------------------------------------------------ fitting
    if not a.skip_fit:
        mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
        ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
        with np.load(mpath, allow_pickle=False) as z:
            X = z["matrix"].astype(np.float32)
        pos = {s: i for i, s in enumerate(ids)}
        ag = torch.load(a.aging, map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
        model.load_state_dict(ag["model"]); model.eval()
        mt = torch.load(a.multitask, map_location=dev, weights_only=False)
        mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
        mt_model.load_state_dict(mt["model"]); mt_model.eval()
        mt_heads = BioHeads(mt_core.rank + mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
        mt_heads.load_state_dict(mt["heads"]); mt_heads.eval()
        log("computing full-coverage features")
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                x = torch.from_numpy(X[i:i + 32]).to(dev)
                obs = torch.isfinite(x)
                E.append(embed_batch(model, core, x, obs).float().cpu().numpy())
                B.append(mt_heads(embed_batch(mt_model, mt_core, x, obs))[3].float().cpu().numpy())
                s = core.posterior(x, obs.float())
                R.append(torch.where(obs, x - core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        E, B, R = np.concatenate(E), np.concatenate(B), np.concatenate(R)
        del X
        feat_check = {}
        for nm, F, key in (("E", E, "mgpt2_embed"), ("B", B, "mt_bio")):
            ref = np.load(require(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_{key}.embedding.npz"), allow_pickle=False)
            assert (ref["dnam_id"].astype(str) == ids).all()
            feat_check[nm] = float(np.abs(F - ref["embedding"]).max())
            log(f"{nm} vs stored {key} features: max abs diff {feat_check[nm]:.2e}")
        FEATS = {"E": E, "B": B, "R": np.hstack([E, B, R]).astype(np.float32)}
        del R
        log("feature shapes " + ", ".join(f"{k} {v.shape}" for k, v in FEATS.items()))

        frames, weights = [], {}
        for ep in endpoints:
            pe = preds[preds.endpoint == ep].reset_index(drop=True)
            idx = np.array([pos[s] for s in pe.dnam_id])
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            cov = pe[["age", "sex"]].to_numpy(np.float32)
            of = pe.outer_fold.to_numpy()
            risk = {c: np.empty(len(pe)) for c in COMPONENTS}
            stacked, equal = np.empty(len(pe)), np.empty(len(pe))
            weights[ep] = {}
            for f in sorted(np.unique(of)):
                held, fit = np.flatnonzero(of == f), np.flatnonzero(of != f)
                cm, cs = fit_scaler(cov[fit], dev)
                oof, assign0, info = {}, None, {}
                for c in COMPONENTS:
                    F = FEATS[c]
                    fm, fs = fit_scaler(F[idx[fit]], dev)
                    Xf = np.hstack([apply_scaler(cov[fit], cm, cs, dev), apply_scaler(F[idx[fit]], fm, fs, dev)])
                    w, alpha, k, oo, assign, sc = select_and_fit_oof(Xf, du[fit], ev[fit], seed=20260928 + int(f), dev=dev, iters=a.iters)
                    del Xf
                    if assign0 is None:
                        assign0 = assign
                    elif not (assign == assign0).all():
                        raise ValueError("inner folds differ between components")
                    oof[c] = oo
                    Xh = np.hstack([apply_scaler(cov[held], cm, cs, dev), apply_scaler(F[idx[held]], fm, fs, dev)])
                    risk[c][held] = Xh @ w
                    info[c] = {"alpha": alpha, "alpha_index": k, "inner_mean_c": float(sc[k]),
                               "nonzero": int((w[2:] != 0).sum())}
                wbest, cbest, grid_c = learn_weights(oof, assign0, ev[fit], du[fit], dev)
                zc = {c: zfold(risk[c][held], np.zeros(len(held))) for c in COMPONENTS}
                stacked[held] = sum(wbest[i] * zc[c] for i, c in enumerate(COMPONENTS))
                equal[held] = sum(zc[c] for c in COMPONENTS)
                weights[ep][int(f)] = {"weights": dict(zip(COMPONENTS, map(float, wbest))), "inner_pooled_c": cbest,
                                       "inner_pooled_c_single": {c: float(grid_c[GRID.index(tuple(float(x) for x in np.eye(3)[i]))])
                                                                 for i, c in enumerate(COMPONENTS)},
                                       "components": info}
                log(f"{ep} fold {f}: w(E,B,R)={tuple(map(float, wbest))} inner C {cbest:.4f}; " +
                    ", ".join(f"{c} a{info[c]['alpha_index']} nz{info[c]['nonzero']}" for c in COMPONENTS))
            q = pe[["dnam_id", "duration", "event", "age", "sex", "endpoint", "age_adjustment", "outer_fold", "baseline_risk"]].copy()
            for c in COMPONENTS:
                q[f"risk_{c}"] = risk[c]
            q["risk_equal"] = equal
            q["risk_stacked"] = stacked
            frames.append(q)
            pd.concat(frames).to_parquet(comp_path, index=False)
            write_json(wpath, {"weights": weights, "feature_check_max_abs_diff": feat_check})
        del FEATS

    comp = pd.read_parquet(require(comp_path))
    wj = json.loads(Path(wpath).read_text())
    sp = comp[["dnam_id", "duration", "event", "age", "sex", "endpoint", "age_adjustment", "outer_fold", "baseline_risk"]].copy()
    sp["risk"] = comp.risk_stacked
    sp = sp[["dnam_id", "duration", "event", "age", "sex", "endpoint", "age_adjustment", "outer_fold", "risk", "baseline_risk"]]
    sp.to_parquet(out / "private_nested_predictions.parquet", index=False)

    # ------------------------------------------------------------ comparators
    def load(paths):
        d = pd.concat([pd.read_parquet(require(Path(p) / "private_nested_predictions.parquet")) for p in paths])
        d = d[d.age_adjustment == "linear"].copy(); d["dnam_id"] = d.dnam_id.astype(str)
        return d
    comps = {"age_sex": None, "raw_cpg_v12": load(RAW_V12), "raw_cpg_gpu": load([Path(a.fr_dir) / "raw_cpg_gpu"])}
    for nm in ("grimage2", "dunedinpace", "grimage2_dunedinpace"):
        comps[nm] = load([Path(a.clock_dir) / nm])
    comps["mgpt2_combined_locked"] = load([a.combined_run])
    refs = {"R_vs_factor_residual": load([Path(a.fr_dir) / "factor_residual"]),
            "E_vs_locked_embedding": load([a.mgpt2_run]), "B_vs_locked_biomarkers": load([a.bio_run])}

    res = {"settings": vars(a), "grid_points": len(GRID), "endpoints": {}, "learned_weights": wj["weights"],
           "feature_check_max_abs_diff": wj.get("feature_check_max_abs_diff"), "reproduction": {}}
    rows = []
    rng = np.random.default_rng(a.seed)
    sk_checked = False
    for ep in endpoints:
        m = comp[comp.endpoint == ep].set_index("dnam_id")
        ev, du = m.event.astype(bool).to_numpy(), m.duration.to_numpy(float)
        of = m.outer_fold.to_numpy()
        fids = sorted(np.unique(of))
        folds = [np.flatnonzero(of == f) for f in fids]
        series = {"stacked": m.risk_stacked, "equal_gpu": m.risk_equal, "E_gpu": m.risk_E, "B_gpu": m.risk_B, "R_gpu": m.risk_R}
        for name, o in comps.items():
            if o is None:
                series[name] = m.baseline_risk
                continue
            o = o[o.endpoint == ep].set_index("dnam_id")
            if set(o.index) != set(m.index) or not (o.loc[m.index, "outer_fold"].to_numpy() == of).all():
                raise ValueError(f"subject or fold mismatch for {ep} vs {name}")
            series[name] = o.loc[m.index, "risk"]
        names = list(series)
        Rm = np.stack([series[n].to_numpy(float) for n in names])
        pcs = [PairCounter(ev[s], du[s], dev) for s in folds]
        tot = sum(pc.counts(Rm[:, s]) for pc, s in zip(pcs, folds))
        cs = dict(zip(names, map(float, cidx(tot))))
        if not sk_checked:  # exactness of the GPU counter against sksurv, unweighted and resampled
            for j, s in enumerate(folds[:2]):
                ref = np.array(concordance_index_censored(ev[s], du[s], Rm[0, s])[1:4], float)
                got = pcs[j].counts(Rm[0, s])[0]
                b = rng.choice(len(s), len(s), replace=True)
                refb = np.array(concordance_index_censored(ev[s][b], du[s][b], Rm[0, s][b])[1:4], float)
                gotb = pcs[j].counts(Rm[0, s], np.bincount(b, minlength=len(s))[None, :])[0, 0]
                if not (np.allclose(ref, got) and np.allclose(refb, gotb)):
                    raise ValueError(f"GPU concordance counts differ from sksurv: {ref} {got} {refb} {gotb}")
            rng = np.random.default_rng(a.seed)  # reset so the check does not shift the bootstrap stream
            sk_checked = True
            res["gpu_concordance_matches_sksurv"] = True
        # reproduction checks
        rep = {}
        for key, ref in refs.items():
            r = ref[ref.endpoint == ep].set_index("dnam_id").loc[m.index, "risk"].to_numpy(float)
            mine = series[key.split("_")[0] + "_gpu"].to_numpy(float)
            per_fold = [float(np.corrcoef(mine[s], r[s])[0, 1]) for s in folds]
            tr = sum(pc.counts(r[s]) for pc, s in zip(pcs, folds))
            rep[key] = {"pearson_pooled": float(np.corrcoef(mine, r)[0, 1]), "pearson_within_fold_min": min(per_fold),
                        "pearson_within_fold_mean": float(np.mean(per_fold)),
                        "c_gpu": cs[key.split("_")[0] + "_gpu"], "c_reference": float(cidx(tr[0]))}
        res["reproduction"][ep] = rep
        res["endpoints"][ep] = {"n": int(len(m)), "events": int(ev.sum()), "c_index": cs}
        log(f"{ep}: " + ", ".join(f"{k} {v:.4f}" for k, v in cs.items()))
        # bootstrap: same multiplicity weights for every model
        W = [np.zeros((a.draws, len(s))) for s in folds]
        for d in range(a.draws):
            for j, s in enumerate(folds):
                b = rng.choice(len(s), len(s), replace=True)
                W[j][d] = np.bincount(b, minlength=len(s))
        cmp_names = list(comps)
        use = ["stacked"] + cmp_names
        Ru = np.stack([series[n].to_numpy(float) for n in use])
        bt = sum(pc.counts(Ru[:, s], Wj) for pc, s, Wj in zip(pcs, folds, W))  # (m, draws, 3)
        cb = cidx(bt)  # (m, draws)
        for i, name in enumerate(cmp_names, 1):
            diffs = cb[0] - cb[i]
            p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
            rows.append({"endpoint": ep, "comparator": name, "c_stacked": cs["stacked"], "c_comparator": cs[name],
                         "difference": cs["stacked"] - cs[name], "ci95_low": float(np.percentile(diffs, 2.5)),
                         "ci95_high": float(np.percentile(diffs, 97.5)), "p": p})
            log(f"  vs {name}: {rows[-1]['difference']:+.4f} [{rows[-1]['ci95_low']:+.4f}, {rows[-1]['ci95_high']:+.4f}] p {p:.3f}")
    df = pd.DataFrame(rows)
    df["q_bh"] = bh(df.p.to_numpy())
    df.to_csv(a.out_csv, index=False)
    keys = list(next(iter(res["endpoints"].values()))["c_index"])
    res["mean_c_index"] = {k: float(np.mean([v["c_index"][k] for v in res["endpoints"].values()])) for k in keys}
    res["bootstrap"] = {"draws": a.draws, "seed": a.seed, "contrasts": df.to_dict("records")}
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out_json), res)
    log("mean C: " + ", ".join(f"{k} {v:.4f}" for k, v in res["mean_c_index"].items()))
    log(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
