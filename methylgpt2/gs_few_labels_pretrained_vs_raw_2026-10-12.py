#!/usr/bin/env python3
"""Few labelled participants: frozen MethylGPT outputs against raw-CpG models fitted from scratch (Generation Scotland).

Development participants only (9,866; the two held-out subsets are not read). Question: when a cohort has few
participants with outcomes, does a Cox model on MethylGPT's pretrained outputs predict disease better than a Cox
model fitted to the raw CpGs of the same participants? MethylGPT was never trained on Generation Scotland, so its
25 distilled clock outputs are frozen features here.

Design: identical to scenario_gs_label_efficiency_2026-10-01.py and label_efficiency_significance_2026-10-04.py.
In each of the five outer folds the fitting participants are subsampled, stratified by event, to 5%, 10% or 25%
(two seeded repeats each) or all are used; every model is fitted on the subsample and scored on the whole held-out
fold; C-indices pool pairs within folds.
  raw CpG              age, sex and the 49,156 CpGs, GPU elastic-net Cox (select_and_fit, 200 iterations; the early-stopped
                       fit of the paper), refitted here so that held-out risks are available for paired tests
  distilled outputs    age, sex and MethylGPT's 25 distilled clock outputs, same estimator (stored risks of 4 October)
  GrimAge2+DunedinPACE age, sex and the two BioLearn clocks, unpenalised Cox (stored risks of 4 October)
  age and sex          unpenalised Cox
Paired bootstrap: 1,000 draws, participants resampled within outer folds (default_rng(20261012)); the pooled C of
each model is averaged over repeats per draw; P = min(1, 2 (k + 1) / (B + 1)); Benjamini-Hochberg over the 24
endpoint x fraction tests of the subsampled fractions and separately over the 8 tests with all participants; the
mean over endpoints averages independent per-endpoint draws.

Outputs (aggregate only): artifacts/gs_cox/few_labels_pretrained_vs_raw_2026-10-12.{json,csv}
Participant-level risks stay in artifacts/gs_cox/few_labels_raw_risks_2026-10-12.parquet (git-ignored, never printed).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from gpu_coxnet_2026_10_01 import select_and_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
L = importlib.import_module("label_efficiency_significance_2026-10-04")
GS = ARTIFACTS / "gs_cox"
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
MODELS = ("raw_cpg", "biomarkers", "grimage2_dunedinpace", "age_sex")
NAMES = {"raw_cpg": "raw CpG", "biomarkers": "MethylGPT distilled outputs", "grimage2_dunedinpace": "GrimAge2 + DunedinPACE", "age_sex": "age and sex"}
CONTRASTS = [("biomarkers", "raw_cpg"), ("biomarkers", "age_sex"), ("raw_cpg", "age_sex"), ("biomarkers", "grimage2_dunedinpace"),
             ("grimage2_dunedinpace", "raw_cpg")]


def compute_raw_risks(preds, endpoints, dev, iters, out_path):
    mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{L.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mpath, allow_pickle=False) as z:
        RAW = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    out = []
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        pid = pe.dnam_id.astype(str).to_numpy()
        idx = np.array([pos[s] for s in pid])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(float)
        fold = pe.outer_fold.to_numpy()
        frame = pd.DataFrame({"endpoint": ep, "dnam_id": pid, "outer_fold": fold, "event": ev, "duration": du})
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        for frac, rep in L.CONFIGS:
            rr, ra = np.empty(len(pe)), np.empty(len(pe))
            for f in sorted(np.unique(fold)):
                held = np.flatnonzero(fold == f)
                fit = L.subsample(np.flatnonzero(fold != f), ev, frac, rep, f)
                c_fit, c_all = L.standardise(cov[fit], cov, dev)
                cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit(c_fit, Surv.from_arrays(ev[fit], du[fit]))
                ra[held] = c_all[held] @ cx.coef_
                F_fit, F_all = L.standardise(RAW[idx[fit]], RAW[idx[held]], dev)
                w, _, _ = select_and_fit(np.hstack([c_fit, F_fit]), du[fit], ev[fit], seed=20260928 + f, dev=dev, iters=iters)
                rr[held] = np.hstack([c_all[held], F_all]) @ w
            frame[f"raw_cpg|{frac}|{rep}"], frame[f"age_sex|{frac}|{rep}"] = rr, ra
            log(f"{ep} f={frac} rep={rep}: raw CpG {L.pooled(ev, du, rr, folds):.4f}  age and sex {L.pooled(ev, du, ra, folds):.4f}")
            pd.concat(out + [frame], ignore_index=True).to_parquet(out_path, index=False)
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reuse-risks", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cuda_preinit()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20261012)
    risks_path = GS / "few_labels_raw_risks_2026-10-12.parquet"
    try:
        preds = pd.read_parquet(require(L.PREDS))
        prev = pd.read_parquet(require(GS / "label_efficiency_significance_risks_2026-10-04.parquet"))
        stored = pd.concat([pd.DataFrame(json.loads(require(GS / f).read_text())["results"])
                            for f in ("scenario_label_efficiency_2026-10-01.json", "scenario_label_efficiency_f005_2026-10-01.json")], ignore_index=True)
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = sorted(preds.endpoint.unique())
    R = pd.read_parquet(require(risks_path)) if a.reuse_risks else compute_raw_risks(preds, endpoints, dev, a.iters, risks_path)
    R = R.merge(prev.drop(columns=["outer_fold", "event", "duration"]), on=["endpoint", "dnam_id"], how="left", validate="one_to_one")
    rcols = [f"{m}|{fr}|{rp}" for fr, rp in L.CONFIGS for m in MODELS]
    assert R[rcols].notna().all().all()

    # point estimates and agreement of the refitted raw-CpG model with the stored run of 1 October
    crow, per_ep = [], {}
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        ev, du, fold = g.event.to_numpy(bool), g.duration.to_numpy(float), g.outer_fold.to_numpy()
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        per_ep[ep] = (g, ev, du, folds)
        for col in rcols:
            m, fr, rp = col.split("|")
            crow.append({"endpoint": ep, "fraction": float(fr), "repeat": int(rp), "model": m, "c": L.pooled(ev, du, g[col].to_numpy(float), folds)})
    cidx = pd.DataFrame(crow)
    chk = cidx[cidx.model == "raw_cpg"].merge(stored[["fraction", "repeat", "endpoint", "raw_cpg"]], on=["fraction", "repeat", "endpoint"])
    agree = {"configurations_compared": int(len(chk)), "max_abs_difference": float((chk.c - chk.raw_cpg).abs().max()) if len(chk) else None,
             "mean_abs_difference": float((chk.c - chk.raw_cpg).abs().mean()) if len(chk) else None}
    log(f"refitted raw-CpG C-index against the stored run: {agree}")

    rng = np.random.default_rng(20261012)
    D = a.draws
    boot = {}
    for ep in endpoints:
        g, ev, du, folds = per_ep[ep]
        risks = [g[c].to_numpy(float) for c in rcols]
        comp = torch.zeros(D, dtype=torch.float64, device=dev)
        con = [comp.clone() for _ in rcols]
        tie = [comp.clone() for _ in rcols]
        for s in folds:
            A, Cs, Ts = L.pair_mats(ev[s], du[s], [r[s] for r in risks], dev)
            W = torch.as_tensor(L.weights_within(rng, len(s), D), device=dev)
            comp += L.wcount(W, A)
            for j in range(len(rcols)):
                con[j] += L.wcount(W, Cs[j])
                tie[j] += L.wcount(W, Ts[j])
            del A, Cs, Ts
        boot[ep] = {c: ((con[j] + 0.5 * tie[j]) / comp).cpu().numpy() for j, c in enumerate(rcols)}
        log(f"bootstrap {ep} done ({time.time() - t0:.0f} s)")

    fracs = sorted({fr for fr, _ in L.CONFIGS})

    def draws(ep, m, frac):
        return np.mean([boot[ep][f"{m}|{frac}|{rp}"] for fr, rp in L.CONFIGS if fr == frac], 0)

    def pval(d):
        return float(min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1)))

    rows, means = [], []
    for frac in fracs:
        pt = cidx[np.isclose(cidx.fraction, frac)].groupby(["endpoint", "model"]).c.mean().unstack()
        n_fit = float(stored[np.isclose(stored.fraction, frac)].mean_fit_subjects.mean()) if (np.isclose(stored.fraction, frac)).any() else None
        for a_, b_ in CONTRASTS:
            dm = np.mean([draws(ep, a_, frac) - draws(ep, b_, frac) for ep in endpoints], 0)
            means.append({"fraction": frac, "contrast": f"{NAMES[a_]} minus {NAMES[b_]}", "mean_c_first": float(pt[a_].mean()), "mean_c_second": float(pt[b_].mean()),
                          "difference": float(pt[a_].mean() - pt[b_].mean()), "ci95_low": float(np.percentile(dm, 2.5)), "ci95_high": float(np.percentile(dm, 97.5)),
                          "p": pval(dm), "endpoints_first_higher": int((pt[a_] > pt[b_]).sum()), "mean_fitting_participants_per_fold": n_fit})
            for ep in endpoints:
                d = draws(ep, a_, frac) - draws(ep, b_, frac)
                rows.append({"fraction": frac, "contrast": f"{NAMES[a_]} minus {NAMES[b_]}", "endpoint": ep, "c_first": float(pt.loc[ep, a_]), "c_second": float(pt.loc[ep, b_]),
                             "difference": float(pt.loc[ep, a_] - pt.loc[ep, b_]), "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)), "p": pval(d)})
    df = pd.DataFrame(rows)
    df["q_bh"] = np.nan
    for con_name in df.contrast.unique():
        for full in (False, True):
            m = (df.contrast == con_name) & (np.isclose(df.fraction, 1.0) == full)
            if m.any():
                df.loc[m, "q_bh"] = L.bh(df.loc[m, "p"].to_numpy())
    df.to_csv(GS / "few_labels_pretrained_vs_raw_2026-10-12.csv", index=False)
    mdf = pd.DataFrame(means)
    for _, r in mdf.iterrows():
        log(f"fraction {r.fraction}: {r.contrast}: {r.mean_c_first:.4f} vs {r.mean_c_second:.4f}, difference {r.difference:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] P {r.p:.3f}; "
            f"{r.endpoints_first_higher} of 8 endpoints")
    write_json(GS / "few_labels_pretrained_vs_raw_2026-10-12.json",
               {"design": __doc__.split("Outputs")[0].strip(), "draws": D, "raw_cpg_agreement_with_stored_run": agree,
                "mean_c_by_fraction": cidx.groupby(["fraction", "model"]).c.mean().unstack().rename(columns=NAMES).reset_index().to_dict(orient="records"),
                "means": means, "endpoint_tests_significant_q05": df[df.q_bh < 0.05].groupby(["fraction", "contrast"]).apply(
                    lambda x: {"first_higher": int((x.difference > 0).sum()), "second_higher": int((x.difference < 0).sum())}).reset_index(name="n").to_dict(orient="records"),
                "elapsed_s": time.time() - t0})


if __name__ == "__main__":
    main()
