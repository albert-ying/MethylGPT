#!/usr/bin/env python3
"""Paired significance of the label-efficiency difference: MethylGPT distilled-clock Cox vs GrimAge2+DunedinPACE (GS).

Inputs
  artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet  (endpoints, outer folds, event,
      duration, age, sex; age_adjustment == "linear")
  artifacts/gs_features_2026-10-01/..._train_mt_bio.embedding.npz   (25 distilled MethylGPT outputs, "biomarkers")
  task_a_gs_clock_scores_fixed_grimage_20260629.csv                 (BioLearn GrimAgeV2 and DunedinPACE)
  artifacts/gs_cox/scenario_label_efficiency_summary_2026-10-01.csv and the per-repeat JSONs (reproduction check)

Method
  1. Held-out risks of both models are recomputed with the subsamples of the original scripts
     (scenario_gs_label_efficiency_2026-10-01.py, scenario_clocks_label_efficiency_2026-10-01.py): in each outer
     fold the fitting subjects are subsampled, stratified by event, with
     rng = default_rng(1000*rep + 17*f + int(frac*1000)), for fractions 0.05/0.10/0.25 (repeats 0, 1) and 1.0 (one
     repeat, all fitting subjects).
       biomarkers: standardise() (torch, fitting-subject mean/SD) of age, sex and the 25 distilled outputs, GPU
         elastic-net Cox select_and_fit(seed=20260928+f, iters=200); age and sex unpenalised.
       grimage2_dunedinpace: unpenalised Breslow Cox (sksurv) of age, sex, GrimAgeV2, DunedinPACE standardised with
         fitting-subject mean and SD (ddof 0).
     Participant-level risks are written only to artifacts/gs_cox/label_efficiency_significance_risks_2026-10-04.parquet.
  2. Pooled within-fold C-indices (sksurv concordance_index_censored, pairs pooled over outer folds) are compared with
     the stored summary (mean over repeats) and with the stored per-repeat values; agreement criterion |diff| < 5e-4.
  3. Paired bootstrap (1,000 draws, numpy default_rng(20261004)): participants are resampled with replacement within
     each outer fold of an endpoint; the same resample is applied to every fraction, repeat and model of that
     endpoint. Per draw: pooled C per repeat and model, averaged over repeats, difference = biomarkers minus
     GrimAge2+DunedinPACE. 95% percentile CI; two-sided p = min(1, 2*min(P(d<=0), P(d>=0))). Benjamini-Hochberg over
     the 24 endpoint x fraction tests at fractions 0.05/0.10/0.25; fraction 1.0 is reported separately (own BH over 8).
     Bootstrap concordance counts use the weighted pair form sum_ij w_i w_j K_ij (w = resample multiplicities), which
     equals sksurv on the resampled arrays (checked on the observed data and on several draws).
  4. Mean-over-endpoints difference per fraction: outer-fold assignment differs between endpoints (checked), so the
     primary CI resamples each endpoint independently (the draws of step 3) and averages draw-wise. Sensitivity
     analysis: one resample of all participants per draw (not stratified by fold; separate generator
     default_rng([20261004, 1])), shared by all endpoints, each endpoint pooling pairs within its own folds.

Outputs (aggregate only)
  artifacts/gs_cox/label_efficiency_significance_2026-10-04.json and .csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from gpu_coxnet_2026_10_01 import select_and_fit

SCORES = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
BIO = ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mt_bio.embedding.npz"
PREDS = ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet"
GS = ARTIFACTS / "gs_cox"
CONFIGS = [(0.05, 0), (0.05, 1), (0.1, 0), (0.1, 1), (0.25, 0), (0.25, 1), (1.0, 0)]
MODELS = ("biomarkers", "grimage2_dunedinpace")
TIED_TOL = 1e-8  # sksurv default


def standardise(F_fit, F_all, dev):
    """Copy of scenario_gs_label_efficiency_2026-10-01.standardise (fitting-subject mean imputation and scaling)."""
    A = torch.as_tensor(F_fit, device=dev, dtype=torch.float32)
    B = torch.as_tensor(F_all, device=dev, dtype=torch.float32)
    fin = torch.isfinite(A)
    mu = torch.where(fin, A, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    A = torch.where(fin, A, mu); B = torch.where(torch.isfinite(B), B, mu)
    sd = A.std(0); sd = torch.where(sd > 1e-8, sd, torch.ones_like(sd))
    return ((A - mu) / sd).cpu().numpy(), ((B - mu) / sd).cpu().numpy()


def pooled(ev, du, r, folds):
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], r[s]); t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


def subsample(fit, ev, frac, rep, f):
    """Identical to the original scripts' stratified draw."""
    if frac >= 1.0:
        return fit
    rng = np.random.default_rng(1000 * rep + 17 * f + int(frac * 1000))
    keep = []
    for e_val in (True, False):
        grp = fit[ev[fit] == e_val]
        keep.append(rng.choice(grp, max(2, int(round(frac * len(grp)))), replace=False))
    return np.sort(np.concatenate(keep))


def compute_risks(preds, endpoints, dev, iters):
    bz = np.load(require(BIO), allow_pickle=False)
    BIOF = bz["embedding"].astype(np.float32)
    pos = {s: i for i, s in enumerate(bz["dnam_id"].astype(str))}
    log(f"distilled outputs: {BIOF.shape[1]} columns")
    sc = pd.read_csv(require(SCORES))
    wide = sc.pivot_table(index="sample_id", columns="model", values="score")
    wide.index = wide.index.astype(str)
    out = []
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        ids = pe.dnam_id.astype(str).to_numpy()
        idx = np.array([pos[s] for s in ids])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(float)
        M = np.column_stack([cov, wide.loc[ids, "GrimAgeV2"].to_numpy(float), wide.loc[ids, "DunedinPACE"].to_numpy(float)])
        fold = pe.outer_fold.to_numpy()
        frame = pd.DataFrame({"endpoint": ep, "dnam_id": ids, "outer_fold": fold, "event": ev, "duration": du})
        for frac, rep in CONFIGS:
            rb, rc = np.empty(len(pe)), np.empty(len(pe))
            for f in sorted(np.unique(fold)):
                held = np.flatnonzero(fold == f)
                fit = subsample(np.flatnonzero(fold != f), ev, frac, rep, f)
                # biomarkers (as scenario_gs_label_efficiency_2026-10-01.py)
                c_fit, c_all = standardise(cov[fit], cov, dev)
                F_fit, F_all = standardise(BIOF[idx[fit]], BIOF[idx], dev)
                Xf = np.hstack([c_fit, F_fit]); Xa = np.hstack([c_all, F_all])
                w, _, _ = select_and_fit(Xf, du[fit], ev[fit], seed=20260928 + f, dev=dev, iters=iters)
                rb[held] = Xa[held] @ w
                # GrimAge2 + DunedinPACE (as scenario_clocks_label_efficiency_2026-10-01.py)
                m, s = M[fit].mean(0), M[fit].std(0)
                cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M[fit] - m) / s, Surv.from_arrays(ev[fit], du[fit]))
                rc[held] = ((M[held] - m) / s) @ cx.coef_
            frame[f"biomarkers|{frac}|{rep}"] = rb
            frame[f"grimage2_dunedinpace|{frac}|{rep}"] = rc
            folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
            log(f"{ep} f={frac} rep={rep}: biomarkers {pooled(ev, du, rb, folds):.4f}  grimage2+dunedinpace {pooled(ev, du, rc, folds):.4f}")
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def reproduction_check(cidx):
    """cidx: DataFrame endpoint, fraction, repeat, model, c."""
    summ = pd.read_csv(require(GS / "scenario_label_efficiency_summary_2026-10-01.csv"))
    rows = []
    mean = cidx.groupby(["fraction", "endpoint", "model"]).c.mean().unstack("model").reset_index()
    for _, r in mean[mean.fraction < 1.0].iterrows():
        s = summ[(np.isclose(summ.fraction, r.fraction)) & (summ.endpoint == r.endpoint)].iloc[0]
        for mdl in MODELS:
            rows.append({"level": "summary_mean_over_repeats", "fraction": r.fraction, "repeat": None, "endpoint": r.endpoint,
                         "model": mdl, "recomputed": float(r[mdl]), "stored": float(s[mdl])})
    stored_b = []
    for f in sorted(GS.glob("scenario_label_efficiency*_2026-10-01.json")):
        stored_b += json.load(open(f))["results"]
    sb = pd.DataFrame(stored_b).drop_duplicates(["fraction", "repeat", "endpoint"])
    sc = pd.DataFrame(json.load(open(GS / "scenario_clocks_label_efficiency_2026-10-01.json"))["results"])
    for _, r in cidx[cidx.fraction < 1.0].iterrows():
        src = sb if r.model == "biomarkers" else sc
        hit = src[np.isclose(src.fraction, r.fraction) & (src["repeat"] == r["repeat"]) & (src.endpoint == r.endpoint)]
        rows.append({"level": "per_repeat", "fraction": r.fraction, "repeat": int(r["repeat"]), "endpoint": r.endpoint,
                     "model": r.model, "recomputed": float(r.c), "stored": float(hit[r.model].iloc[0])})
    # fraction 1.0: no stored GPU run; reference values from other runs on all fitting subjects
    dc = pd.DataFrame(json.load(open(GS / "scenario_distilled_clocks_label_efficiency_2026-10-04.json"))["results"])
    locked = pd.read_csv(GS / "mt_bio_linear_2026-10-01/nested_development_results.csv")
    locked = locked[locked.age_adjustment == "linear"].set_index("endpoint")
    for _, r in cidx[cidx.fraction == 1.0].iterrows():
        if r.model == "grimage2_dunedinpace":
            hit = dc[np.isclose(dc.fraction, 1.0) & (dc.endpoint == r.endpoint)]
            ref, lvl = float(hit.grimage2_dunedinpace.iloc[0]), "f1_vs_scenario_distilled_clocks_2026-10-04"
        else:
            ref, lvl = float(locked.loc[r.endpoint, "within_fold_pairs_c_index"]), "f1_vs_locked_sksurv_mt_bio_linear (different solver)"
        rows.append({"level": lvl, "fraction": 1.0, "repeat": 0, "endpoint": r.endpoint, "model": r.model,
                     "recomputed": float(r.c), "stored": ref})
    d = pd.DataFrame(rows)
    d["abs_diff"] = (d.recomputed - d.stored).abs()
    d["match_3dp"] = d.abs_diff < 5e-4
    return d


def bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def pair_mats(ev, du, risks, dev):
    """Comparable-pair matrix A (sksurv definition) and, per risk vector, concordant and tied-risk matrices."""
    t = torch.as_tensor(du, device=dev, dtype=torch.float64)
    e = torch.as_tensor(ev, device=dev)
    A = e[:, None] & ((t[None, :] > t[:, None]) | ((t[None, :] == t[:, None]) & ~e[None, :]))
    con, tie = [], []
    for r in risks:
        rr = torch.as_tensor(r, device=dev, dtype=torch.float64)
        dif = rr[:, None] - rr[None, :]  # risk_i - risk_j
        ti = dif.abs() <= TIED_TOL
        con.append((A & (dif > 0) & ~ti).float()); tie.append((A & ti).float())
    return A.float(), con, tie


def wcount(W, K):
    """Per-row sum_ij w_i w_j K_ij (exact: integer-valued float32 matmul without TF32, float64 reduction)."""
    return ((W @ K).double() * W.double()).sum(1)


def weights_within(rng, n, draws):
    idx = rng.integers(0, n, size=(draws, n))
    W = np.zeros((draws, n), np.float32)
    np.add.at(W, (np.repeat(np.arange(draws), n), idx.ravel()), 1)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reuse-risks", action="store_true", help="skip refitting; read the stored risk parquet")
    ap.add_argument("--risks", default=str(GS / "label_efficiency_significance_risks_2026-10-04.parquet"))
    ap.add_argument("--out", default=str(GS / "label_efficiency_significance_2026-10-04.json"))
    a = ap.parse_args()
    t0 = time.time()
    cuda_preinit()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(a.seed)  # seeds the power-iteration start vector inside fit_path (unseeded in the original run)
    preds = pd.read_parquet(require(PREDS))
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = sorted(preds.endpoint.unique())

    if a.reuse_risks:
        R = pd.read_parquet(require(Path(a.risks)))
    else:
        R = compute_risks(preds, endpoints, dev, a.iters)
        R.to_parquet(a.risks, index=False)
        log(f"wrote participant-level risks to {a.risks} (artifacts/, git-ignored)")
    rcols = [f"{m}|{fr}|{rp}" for fr, rp in CONFIGS for m in MODELS]

    # ---- point estimates (sksurv) and reproduction check
    crow = []
    per_ep = {}
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        ev, du, fold = g.event.to_numpy(bool), g.duration.to_numpy(float), g.outer_fold.to_numpy()
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        per_ep[ep] = (g, ev, du, folds)
        for col in rcols:
            m, fr, rp = col.split("|")
            crow.append({"endpoint": ep, "fraction": float(fr), "repeat": int(rp), "model": m, "c": pooled(ev, du, g[col].to_numpy(float), folds)})
    cidx = pd.DataFrame(crow)
    rep = reproduction_check(cidx)
    rep.to_csv(Path(a.out).with_name("label_efficiency_significance_reproduction_2026-10-04.csv"), index=False)
    for lvl, gq in rep.groupby("level"):
        log(f"reproduction [{lvl}]: {int(gq.match_3dp.sum())}/{len(gq)} within 5e-4; max |diff| {gq.abs_diff.max():.5f}")

    # ---- participant alignment across endpoints (step 4 check)
    ref_ids = per_ep[endpoints[0]][0].dnam_id.to_numpy()
    same_order = all((per_ep[ep][0].dnam_id.to_numpy() == ref_ids).all() for ep in endpoints)
    same_set = all(set(per_ep[ep][0].dnam_id) == set(ref_ids) for ep in endpoints)
    ref_fold = pd.Series(per_ep[endpoints[0]][0].outer_fold.to_numpy(), index=ref_ids)
    same_folds = all((per_ep[ep][0].set_index("dnam_id").loc[ref_ids, "outer_fold"].to_numpy() == ref_fold.to_numpy()).all() for ep in endpoints)
    shared = same_order and same_folds
    log(f"participant set identical: {same_set}; row order identical: {same_order}; outer folds identical: {same_folds}")

    # ---- bootstrap
    rng = np.random.default_rng(a.seed)
    rng_glob = np.random.default_rng([a.seed, 1])
    D = a.draws
    G = None
    if not shared:
        idxg = rng_glob.integers(0, len(ref_ids), size=(D, len(ref_ids)))
        G = np.zeros((D, len(ref_ids)), np.float32)
        np.add.at(G, (np.repeat(np.arange(D), len(ref_ids)), idxg.ravel()), 1)
    gpos = {s: i for i, s in enumerate(ref_ids)}
    boot = {}       # ep -> {col: (D,) pooled C} primary (within-fold)
    boot_g = {}     # ep -> {col: (D,) pooled C} sensitivity (shared global resample)
    shared_W = None
    max_check = 0.0
    for ep in endpoints:
        g, ev, du, folds = per_ep[ep]
        risks = [g[c].to_numpy(float) for c in rcols]
        comp = torch.zeros(D, dtype=torch.float64, device=dev); compg = comp.clone()
        con = [comp.clone() for _ in rcols]; tie = [comp.clone() for _ in rcols]
        cong = [comp.clone() for _ in rcols]; tieg = [comp.clone() for _ in rcols]
        gidx = np.array([gpos[s] for s in g.dnam_id])
        for k, s in enumerate(folds):
            A, Cs, Ts = pair_mats(ev[s], du[s], [r[s] for r in risks], dev)
            if shared:
                if shared_W is None:
                    shared_W = [weights_within(rng, len(ss), D) for ss in folds]
                Wn = shared_W[k]
            else:
                Wn = weights_within(rng, len(s), D)
            # row 0 of the check block: observed data (w = 1); rows 1..3: first three draws, verified against sksurv below
            W = torch.as_tensor(np.vstack([Wn, np.ones((1, len(s)), np.float32)] + ([G[:, gidx[s]]] if G is not None else [])), device=dev)
            cA = wcount(W, A)
            comp += cA[:D]
            if G is not None:
                compg += cA[D + 1:]
            for j in range(len(rcols)):
                cc, tt = wcount(W, Cs[j]), wcount(W, Ts[j])
                con[j] += cc[:D]; tie[j] += tt[:D]
                if G is not None:
                    cong[j] += cc[D + 1:]; tieg[j] += tt[D + 1:]
                if j == 0:  # exactness check against sksurv on the observed fold and on draws 0-2
                    for d_ in (D, 0, 1, 2):
                        if d_ == D:
                            b = s
                        else:
                            b = s[np.repeat(np.arange(len(s)), Wn[d_].astype(int))]
                        _, c1, d1, t1, _ = concordance_index_censored(ev[b], du[b], risks[0][b])
                        got = np.array([cc[d_].item(), cA[d_].item() - cc[d_].item() - tt[d_].item(), tt[d_].item()])
                        max_check = max(max_check, float(np.abs(got - np.array([c1, d1, t1])).max()))
            del A, Cs, Ts
        boot[ep] = {c: ((con[j] + 0.5 * tie[j]) / comp).cpu().numpy() for j, c in enumerate(rcols)}
        if G is not None:
            boot_g[ep] = {c: ((cong[j] + 0.5 * tieg[j]) / compg).cpu().numpy() for j, c in enumerate(rcols)}
        log(f"bootstrap {ep} done ({time.time() - t0:.0f}s)")
    log(f"weighted-count vs sksurv max abs pair-count difference: {max_check}")

    def diff_draws(bd, ep, frac):
        reps = [rp for fr, rp in CONFIGS if fr == frac]
        b = np.mean([bd[ep][f"biomarkers|{frac}|{rp}"] for rp in reps], 0)
        c = np.mean([bd[ep][f"grimage2_dunedinpace|{frac}|{rp}"] for rp in reps], 0)
        return b, c, b - c

    rows = []
    for frac in (0.05, 0.1, 0.25, 1.0):
        for ep in endpoints:
            pt = cidx[(np.isclose(cidx.fraction, frac)) & (cidx.endpoint == ep)].groupby("model").c.mean()
            _, _, d = diff_draws(boot, ep, frac)
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            rows.append({"fraction": frac, "endpoint": ep, "c_biomarkers": float(pt["biomarkers"]),
                         "c_grimage2_dunedinpace": float(pt["grimage2_dunedinpace"]),
                         "difference": float(pt["biomarkers"] - pt["grimage2_dunedinpace"]),
                         "boot_mean_difference": float(d.mean()),
                         "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)), "p": p})
    df = pd.DataFrame(rows)
    sub = df.fraction < 1.0
    df["q_bh"] = np.nan
    df.loc[sub, "q_bh"] = bh(df.loc[sub, "p"])
    df.loc[~sub, "q_bh"] = bh(df.loc[~sub, "p"])
    df["bh_family"] = np.where(sub, "fractions_0.05_0.10_0.25 (24 tests)", "fraction_1.0 (8 tests)")
    df["direction"] = np.where(df.q_bh < 0.05, np.where(df.difference > 0, "biomarkers_better", "grimage2_dunedinpace_better"), "ns")

    means = []
    for frac in (0.05, 0.1, 0.25, 1.0):
        pt = float(df[np.isclose(df.fraction, frac)].difference.mean())
        for label, bd in (("independent_per_endpoint_within_fold", boot), ("shared_global_participant_resample", boot_g)):
            if not bd:
                continue
            dm = np.mean([diff_draws(bd, ep, frac)[2] for ep in endpoints], 0)
            g_ = df[np.isclose(df.fraction, frac)]
            means.append({"fraction": frac, "resampling": label, "mean_difference": pt,
                          "ci95_low": float(np.percentile(dm, 2.5)), "ci95_high": float(np.percentile(dm, 97.5)),
                          "p": float(min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))),
                          "n_q05_biomarkers_better": int((g_.direction == "biomarkers_better").sum()),
                          "n_q05_grimage2_dunedinpace_better": int((g_.direction == "grimage2_dunedinpace_better").sum()),
                          "significant_endpoints": g_[g_.q_bh < 0.05].endpoint.tolist()})
    mf = pd.DataFrame(means)
    for _, r in mf.iterrows():
        log(f"f={r.fraction} [{r.resampling}] mean diff {r.mean_difference:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] p {r.p:.3f}; "
            f"q<0.05: bio better {r.n_q05_biomarkers_better}, clocks better {r.n_q05_grimage2_dunedinpace_better}")
    for _, r in df.iterrows():
        log(f"f={r.fraction} {r.endpoint}: {r.c_biomarkers:.4f} vs {r.c_grimage2_dunedinpace:.4f} diff {r.difference:+.4f} "
            f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] p {r.p:.3f} q {r.q_bh:.3f}")

    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    mf.to_csv(Path(a.out).with_name("label_efficiency_significance_means_2026-10-04.csv"), index=False)
    write_json(Path(a.out), {
        "settings": vars(a), "configs": CONFIGS,
        "participant_alignment": {"same_set": bool(same_set), "same_row_order": bool(same_order), "same_outer_folds": bool(same_folds),
                                  "mean_ci_resampling": "shared within-fold resample" if shared else
                                  "independent per-endpoint within-fold resamples (primary) + shared global participant resample (sensitivity)"},
        "weighted_count_vs_sksurv_max_abs_diff": max_check,
        "reproduction": {"n_checked": int(len(rep)), "n_within_5e-4": int(rep.match_3dp.sum()),
                         "by_level": rep.groupby("level").agg(n=("match_3dp", "size"), n_match=("match_3dp", "sum"), max_abs_diff=("abs_diff", "max")).reset_index().to_dict("records"),
                         "mismatches": rep[~rep.match_3dp].to_dict("records")},
        "per_endpoint": df.to_dict("records"), "mean_over_endpoints": mf.to_dict("records"),
        "elapsed_s": time.time() - t0})
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
