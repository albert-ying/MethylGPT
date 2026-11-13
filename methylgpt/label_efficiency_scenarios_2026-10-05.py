#!/usr/bin/env python3
"""Label efficiency when the participants to be scored lack CpGs (GS development subjects, locked outer folds).

Question: with 5%, 10% or 25% of the fitting participants, GrimAge2 + DunedinPACE computed from the full array
predicted disease slightly better than MethylGPT's 25 distilled outputs (label_efficiency_significance_2026-10-04).
Does this hold when the held-out participants have only 27K-array CpGs or 70% / 90% of CpGs removed, as in the
missing-CpG scenarios of Fig. 6c,d?

Design (identical to label_efficiency_significance_2026-10-04.py except for the held-out inputs)
  - Subsamples, model forms, solver settings and seeds are imported from that script: fitting subjects are
    subsampled within each outer fold (stratified by event; fractions 0.05/0.10/0.25 with repeats 0, 1 and 1.0);
    the distilled-output model is the GPU elastic-net Cox of age, sex and the 25 outputs (select_and_fit), and the
    clock model an unpenalised Breslow Cox of age, sex, GrimAgeV2 and DunedinPACE. Both are fitted on complete
    profiles, as before.
  - Held-out subjects are then scored from degraded profiles, exactly as in scenario_gs_combined_2026-10-01.py:
    random_<p> removes p% of each subject's CpGs (numpy default_rng(20261001 + p) over the GS matrix), hm27 keeps
    the vocabulary CpGs on the 27K array. MethylGPT recomputes its distilled outputs from the remaining CpGs (released
    multitask model); GrimAge2 and DunedinPACE are the BioLearn scores of the same scenarios
    (artifacts/gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv, missing clock CpGs filled with
    development-set means). Fitting-subject standardisation is applied unchanged.
  - Checks: distilled outputs recomputed from complete profiles must equal the stored features used by the label-
    efficiency study, and complete-profile risks must reproduce the stored risks of that study.
  - Paired bootstrap (1,000 draws, participants resampled within outer folds, numpy default_rng(20261005)), pooled
    C per repeat averaged over repeats, difference = distilled outputs minus GrimAge2 + DunedinPACE; Benjamini-Hochberg
    over the 24 endpoint x subsampled-fraction tests within each scenario, separately over the 8 full-data tests;
    mean over endpoints resampled independently per endpoint.

Outputs (aggregate only; participant-level risks stay in artifacts/, which is git-ignored)
  artifacts/gs_cox/label_efficiency_scenarios_2026-10-05.json, .csv and _means.csv
  artifacts/gs_cox/label_efficiency_scenarios_risks_2026-10-05.parquet (private)
Run: python3 label_efficiency_scenarios_2026-10-05.py --device cuda:0
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from methylgpt_model import BioHeads, embed_batch, load_checkpoint
from gpu_coxnet_2026_10_01 import select_and_fit

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("les", HERE / "label_efficiency_significance_2026-10-04.py")
les = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(les)

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = les.SPLIT
GS = ARTIFACTS / "gs_cox"
CLOCK_SCEN = ARTIFACTS / "gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv"
STORED_RISKS = GS / "label_efficiency_significance_risks_2026-10-04.parquet"
CONFIGS = les.CONFIGS
MODELS = les.MODELS


def distilled_outputs(scenarios, dev, multitask):
    """Distilled outputs (standardised head units, as stored in gs_features) for every GS training-subset subject."""
    mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mpath, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    mt = torch.load(multitask, map_location=dev, weights_only=False)
    mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
    mt_model.load_state_dict(mt["model"]); mt_model.eval()
    heads = BioHeads(mt_core.rank + mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
    heads.load_state_dict(mt["heads"]); heads.eval()
    out, observed = {}, {}
    for scen in scenarios:
        Xs = X.copy()
        if scen.startswith("random_"):
            p = int(scen.split("_")[1]) / 100
            r = np.random.default_rng(20261001 + int(p * 100))
            Xs[r.random(Xs.shape) < p] = np.nan
        elif scen != "full":
            Xs[:, ~plat[scen]] = np.nan
        observed[scen] = float(np.isfinite(Xs).mean())
        B = []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                B.append(heads(embed_batch(mt_model, mt_core, x, torch.isfinite(x)))[3].float().cpu().numpy())
        out[scen] = np.concatenate(B)
        log(f"distilled outputs, {scen}: observed fraction {observed[scen]:.3f}")
        del Xs
    return ids, out, observed


def compute_risks(preds, endpoints, dev, iters, ids_x, BIO, clock):
    """BIO: scenario -> (n_subjects, 25) distilled outputs; clock: scenario -> DataFrame (GrimAgeV2, DunedinPACE)."""
    pos = {s: i for i, s in enumerate(ids_x)}
    scen = [s for s in BIO if s != "full"]
    frames = []
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        sid = pe.dnam_id.astype(str).to_numpy()
        idx = np.array([pos[s] for s in sid])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(float)
        M = {s: np.column_stack([cov, clock[s].loc[sid, "GrimAgeV2"].to_numpy(float), clock[s].loc[sid, "DunedinPACE"].to_numpy(float)])
             for s in ["full"] + scen}
        fold = pe.outer_fold.to_numpy()
        frame = pd.DataFrame({"endpoint": ep, "dnam_id": sid, "outer_fold": fold, "event": ev, "duration": du})
        for frac, rep in CONFIGS:
            rb = {s: np.empty(len(pe)) for s in ["full"] + scen}
            rc = {s: np.empty(len(pe)) for s in ["full"] + scen}
            for f in sorted(np.unique(fold)):
                held = np.flatnonzero(fold == f)
                fit = les.subsample(np.flatnonzero(fold != f), ev, frac, rep, f)
                c_fit, c_all = les.standardise(cov[fit], cov, dev)
                F_fit, _ = les.standardise(BIO["full"][idx[fit]], BIO["full"][idx], dev)
                w, _, _ = select_and_fit(np.hstack([c_fit, F_fit]), du[fit], ev[fit], seed=20260928 + f, dev=dev, iters=iters)
                for s in ["full"] + scen:
                    _, F_s = les.standardise(BIO["full"][idx[fit]], BIO[s][idx], dev)  # fitting-subject statistics
                    rb[s][held] = np.hstack([c_all, F_s])[held] @ w
                m, sd = M["full"][fit].mean(0), M["full"][fit].std(0)
                cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M["full"][fit] - m) / sd, Surv.from_arrays(ev[fit], du[fit]))
                for s in ["full"] + scen:
                    rc[s][held] = ((M[s][held] - m) / sd) @ cx.coef_
            folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
            for s in ["full"] + scen:
                frame[f"{s}|biomarkers|{frac}|{rep}"] = rb[s]
                frame[f"{s}|grimage2_dunedinpace|{frac}|{rep}"] = rc[s]
            log(f"{ep} f={frac} rep={rep}: " + "; ".join(
                f"{s} {les.pooled(ev, du, rb[s], folds):.4f} vs {les.pooled(ev, du, rc[s], folds):.4f}" for s in ["full"] + scen))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap(R, endpoints, cols, dev, draws, seed):
    """Pooled within-fold C for every risk column under the same within-fold participant resample (per endpoint)."""
    rng = np.random.default_rng(seed)
    boot = {}
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        ev, du, fold = g.event.to_numpy(bool), g.duration.to_numpy(float), g.outer_fold.to_numpy()
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        comp = torch.zeros(draws, dtype=torch.float64, device=dev)
        con = {c: comp.clone() for c in cols}; tie = {c: comp.clone() for c in cols}
        for s in folds:
            W = torch.as_tensor(les.weights_within(rng, len(s), draws), device=dev)
            for k in range(0, len(cols), 14):  # bounded GPU memory: 14 risk vectors per pass
                chunk = cols[k:k + 14]
                A, Cs, Ts = les.pair_mats(ev[s], du[s], [g[c].to_numpy(float)[s] for c in chunk], dev)
                if k == 0:
                    comp += les.wcount(W, A)
                for j, c in enumerate(chunk):
                    con[c] += les.wcount(W, Cs[j]); tie[c] += les.wcount(W, Ts[j])
                del A, Cs, Ts
        boot[ep] = {c: ((con[c] + 0.5 * tie[c]) / comp).cpu().numpy() for c in cols}
        log(f"bootstrap {ep} done")
    return boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scenarios", default="hm27,random_70,random_90")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--multitask", default=str(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt"))
    ap.add_argument("--reuse-risks", action="store_true")
    ap.add_argument("--risks", default=str(GS / "label_efficiency_scenarios_risks_2026-10-05.parquet"))
    ap.add_argument("--out", default=str(GS / "label_efficiency_scenarios_2026-10-05.json"))
    a = ap.parse_args()
    t0 = time.time()
    cuda_preinit()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    scen = [s for s in a.scenarios.split(",") if s]
    preds = pd.read_parquet(require(les.PREDS))
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = sorted(preds.endpoint.unique())
    checks = {}

    if a.reuse_risks:
        R = pd.read_parquet(require(Path(a.risks)))
        observed = None
    else:
        torch.backends.cuda.matmul.allow_tf32 = True  # as in extract_gs_features_2026-10-01 and scenario_gs_combined_2026-10-01
        ids_x, BIO, observed = distilled_outputs(["full"] + scen, dev, a.multitask)
        torch.backends.cuda.matmul.allow_tf32 = False  # as in label_efficiency_significance_2026-10-04 for the fits
        bz = np.load(require(les.BIO), allow_pickle=False)
        stored = pd.DataFrame(bz["embedding"].astype(np.float32), index=bz["dnam_id"].astype(str))
        diff = float(np.abs(stored.loc[ids_x].to_numpy() - BIO["full"]).max())
        checks["distilled_full_vs_stored_features_max_abs_diff"] = diff
        log(f"check: recomputed complete-profile distilled outputs vs stored features, max |diff| {diff:.2e}")
        if diff > 1e-3:
            raise SystemExit("recomputed distilled outputs do not match the stored features")
        BIO["full"] = stored.loc[ids_x].to_numpy()  # use exactly the inputs of the label-efficiency study
        cs = pd.read_csv(require(CLOCK_SCEN)); cs["sample_id"] = cs.sample_id.astype(str)
        clock = {s: cs[cs.scenario == s].set_index("sample_id") for s in ["full"] + scen}
        torch.manual_seed(20261004)  # same power-iteration start as label_efficiency_significance_2026-10-04
        R = compute_risks(preds, endpoints, dev, a.iters, ids_x, BIO, clock)
        R.to_parquet(a.risks, index=False)
        log(f"wrote participant-level risks to {a.risks} (artifacts/, git-ignored)")

    # reproduction of the complete-profile risks of the label-efficiency study (aggregate statistics only)
    old = pd.read_parquet(require(STORED_RISKS))
    key = ["endpoint", "dnam_id"]
    mg = R[key + [f"full|{m}|{fr}|{rp}" for fr, rp in CONFIGS for m in MODELS]].merge(
        old[key + [f"{m}|{fr}|{rp}" for fr, rp in CONFIGS for m in MODELS]], on=key)
    rep_rows = []
    for fr, rp in CONFIGS:
        for m in MODELS:
            x, y = mg[f"full|{m}|{fr}|{rp}"].to_numpy(float), mg[f"{m}|{fr}|{rp}"].to_numpy(float)
            rep_rows.append({"model": m, "fraction": fr, "repeat": rp, "pearson": float(np.corrcoef(x, y)[0, 1]),
                             "max_abs_diff_rel": float(np.abs(x - y).max() / (np.abs(y).max() + 1e-12))})
    rep = pd.DataFrame(rep_rows)
    checks["complete_risks_vs_label_efficiency_study"] = {"n_rows_matched": int(len(mg)), "min_pearson": float(rep.pearson.min()),
                                                          "max_rel_abs_diff": float(rep.max_abs_diff_rel.max())}
    log(f"check: complete-profile risks vs stored study risks, min r {rep.pearson.min():.6f}, max rel diff {rep.max_abs_diff_rel.max():.2e}")

    allscen = ["full"] + scen
    cols = [f"{s}|{m}|{fr}|{rp}" for s in allscen for fr, rp in CONFIGS for m in MODELS]
    # point estimates
    cidx = []
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        ev, du, fold = g.event.to_numpy(bool), g.duration.to_numpy(float), g.outer_fold.to_numpy()
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        for c in cols:
            s, m, fr, rp = c.split("|")
            cidx.append({"scenario": s, "endpoint": ep, "fraction": float(fr), "repeat": int(rp), "model": m,
                         "c": les.pooled(ev, du, g[c].to_numpy(float), folds)})
    cidx = pd.DataFrame(cidx)
    boot = bootstrap(R, endpoints, cols, dev, a.draws, a.seed)

    def diff_draws(s, ep, frac):
        reps = [rp for fr, rp in CONFIGS if fr == frac]
        b = np.mean([boot[ep][f"{s}|biomarkers|{frac}|{rp}"] for rp in reps], 0)
        c = np.mean([boot[ep][f"{s}|grimage2_dunedinpace|{frac}|{rp}"] for rp in reps], 0)
        return b - c

    rows = []
    for s in allscen:
        for frac in (0.05, 0.1, 0.25, 1.0):
            for ep in endpoints:
                pt = cidx[(cidx.scenario == s) & np.isclose(cidx.fraction, frac) & (cidx.endpoint == ep)].groupby("model").c.mean()
                d = diff_draws(s, ep, frac)
                rows.append({"scenario": s, "fraction": frac, "endpoint": ep, "c_biomarkers": float(pt["biomarkers"]),
                             "c_grimage2_dunedinpace": float(pt["grimage2_dunedinpace"]),
                             "difference": float(pt["biomarkers"] - pt["grimage2_dunedinpace"]),
                             "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                             "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
    df = pd.DataFrame(rows)
    df["q_bh"] = np.nan
    for s in allscen:
        for sub in (df.fraction < 1.0, df.fraction >= 1.0):
            m_ = (df.scenario == s) & sub
            df.loc[m_, "q_bh"] = les.bh(df.loc[m_, "p"])
    df["direction"] = np.where(df.q_bh < 0.05, np.where(df.difference > 0, "biomarkers_better", "grimage2_dunedinpace_better"), "ns")

    means = []
    for s in allscen:
        for frac in (0.05, 0.1, 0.25, 1.0):
            g_ = df[(df.scenario == s) & np.isclose(df.fraction, frac)]
            dm = np.mean([diff_draws(s, ep, frac) for ep in endpoints], 0)
            means.append({"scenario": s, "fraction": frac, "mean_c_biomarkers": float(g_.c_biomarkers.mean()),
                          "mean_c_grimage2_dunedinpace": float(g_.c_grimage2_dunedinpace.mean()),
                          "mean_difference": float(g_.difference.mean()),
                          "ci95_low": float(np.percentile(dm, 2.5)), "ci95_high": float(np.percentile(dm, 97.5)),
                          "p": float(min(1.0, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))),
                          "n_q05_biomarkers_better": int((g_.direction == "biomarkers_better").sum()),
                          "n_q05_grimage2_dunedinpace_better": int((g_.direction == "grimage2_dunedinpace_better").sum())})
    mf = pd.DataFrame(means)
    for _, r in mf.iterrows():
        log(f"{r.scenario} f={r.fraction}: {r.mean_c_biomarkers:.4f} vs {r.mean_c_grimage2_dunedinpace:.4f}, diff {r.mean_difference:+.4f} "
            f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] p {r.p:.3f}; q<0.05 bio {r.n_q05_biomarkers_better}, clocks {r.n_q05_grimage2_dunedinpace_better}")
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    mf.to_csv(Path(a.out).with_name("label_efficiency_scenarios_means_2026-10-05.csv"), index=False)
    write_json(Path(a.out), {"settings": vars(a), "configs": CONFIGS, "observed_fraction": observed, "checks": checks,
                             "reproduction_by_config": rep.to_dict("records"),
                             "per_endpoint": df.to_dict("records"), "mean_over_endpoints": mf.to_dict("records"),
                             "elapsed_s": time.time() - t0})
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
