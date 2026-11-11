#!/usr/bin/env python3
"""Few labelled participants: does the transformer transfer to disease beyond the factor model? (Generation Scotland)

Development participants only (9,866; the two held-out subsets are not read). MethylGPT was never trained on
Generation Scotland or on any disease outcome, so every representation below is frozen. The question is whether
the parts of MethylGPT that the transformer learned from public labels (age, sex, tissue, 25 clock outputs)
predict incident disease in a new cohort better than the factor model's scores alone, with the same estimator
and the same labelled participants.

Design: identical to gs_few_labels_pretrained_vs_raw_2026-10-12.py (five outer folds; fitting participants
subsampled, stratified by event, to 5%, 10% or 25% with two seeded repeats, or all; models scored on the whole
held-out fold; C-indices pool pairs within folds). Each model is the GPU elastic-net Cox (select_and_fit, 200
iterations) of age, sex (unpenalised) and one frozen representation:
  factor scores                 1,536 factor scores (factor model only)
  transformer, age stage        the 512 transformer dimensions of the embedding after age, sex and tissue training
  embedding, age stage          factor scores + those 512 dimensions (2,048; the embedding of the disease models)
  transformer, clock stage      the 512 transformer dimensions after the 25 clock outputs were added as targets
  embedding, clock stage        factor scores + those 512 dimensions (2,048)
Stored risks are added for comparison: the 25 distilled outputs and GrimAge2 + DunedinPACE (4 October), raw CpGs
and age and sex (12 October).
Paired bootstrap as before (1,000 draws within outer folds, default_rng(20261013); add-one P values;
Benjamini-Hochberg over the 24 endpoint x fraction tests of the subsampled fractions and over the 8 tests with all
participants). Contrasts: every model minus the factor scores.

Outputs (aggregate only): artifacts/gs_cox/few_labels_representations_2026-10-13.{json,csv}
Participant-level risks stay in artifacts/gs_cox/few_labels_representation_risks_2026-10-13.parquet (git-ignored).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from gpu_coxnet_2026_10_01 import select_and_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
L = importlib.import_module("label_efficiency_significance_2026-10-04")
GS = ARTIFACTS / "gs_cox"
FEAT = ARTIFACTS / "gs_features_2026-10-01"
NEW = ("factor_scores", "transformer_age_stage", "embedding_age_stage", "transformer_clock_stage", "embedding_clock_stage")
STORED = ("biomarkers", "grimage2_dunedinpace", "raw_cpg", "age_sex")
NAMES = {"factor_scores": "factor scores (1,536)", "transformer_age_stage": "transformer dimensions, age stage (512)",
         "embedding_age_stage": "embedding, age stage (2,048)", "transformer_clock_stage": "transformer dimensions, clock stage (512)",
         "embedding_clock_stage": "embedding, clock stage (2,048)", "biomarkers": "25 distilled outputs", "grimage2_dunedinpace": "GrimAge2 + DunedinPACE",
         "raw_cpg": "raw CpG", "age_sex": "age and sex"}


def features():
    def load(tag):
        z = np.load(require(FEAT / f"COMMON_SAMPLE_SPLIT_{L.SPLIT}_train_{tag}.embedding.npz"), allow_pickle=False)
        return z["dnam_id"].astype(str), z["embedding"].astype(np.float32)
    ids, core = load("mgpt2_coreonly")
    i2, emb_age = load("mgpt2_embed")
    i3, emb_clock = load("mt_embed")
    i4, lat_age = load("mgpt2_latentonly")
    assert (ids == i2).all() and (ids == i3).all() and (ids == i4).all()
    assert np.allclose(emb_age[:, 1536:], lat_age, atol=1e-4), "age-stage transformer dimensions differ between feature files"
    same_scores = float(np.abs(emb_age[:, :1536] - emb_clock[:, :1536]).max())
    log(f"factor-score block identical in the two embeddings to {same_scores:.1e}; "
        f"largest difference from the separately stored factor scores {float(np.abs(emb_age[:, :1536] - core).max()):.1e}")
    core = emb_age[:, :1536]   # the factor-score block the embedding contains, so the arms differ only by the transformer dimensions
    return ids, {"factor_scores": core, "transformer_age_stage": lat_age, "embedding_age_stage": emb_age,
                 "transformer_clock_stage": emb_clock[:, 1536:], "embedding_clock_stage": emb_clock}


def compute_risks(preds, endpoints, dev, iters, out_path):
    ids, feats = features()
    pos = {s: i for i, s in enumerate(ids)}
    out = []
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        pid = pe.dnam_id.astype(str).to_numpy()
        idx = np.array([pos[s] for s in pid])
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(float)
        fold = pe.outer_fold.to_numpy()
        frame = pd.DataFrame({"endpoint": ep, "dnam_id": pid})
        folds = [np.flatnonzero(fold == f) for f in sorted(np.unique(fold))]
        for frac, rep in L.CONFIGS:
            risks = {m: np.empty(len(pe)) for m in NEW}
            for f in sorted(np.unique(fold)):
                held = np.flatnonzero(fold == f)
                fit = L.subsample(np.flatnonzero(fold != f), ev, frac, rep, f)
                c_fit, c_all = L.standardise(cov[fit], cov, dev)
                for m in NEW:
                    F_fit, F_all = L.standardise(feats[m][idx[fit]], feats[m][idx[held]], dev)
                    w, _, _ = select_and_fit(np.hstack([c_fit, F_fit]), du[fit], ev[fit], seed=20260928 + f, dev=dev, iters=iters)
                    risks[m][held] = np.hstack([c_all[held], F_all]) @ w
            for m in NEW:
                frame[f"{m}|{frac}|{rep}"] = risks[m]
            log(f"{ep} f={frac} rep={rep}: " + "  ".join(f"{m} {L.pooled(ev, du, risks[m], folds):.4f}" for m in NEW))
        out.append(frame)
        pd.concat(out, ignore_index=True).to_parquet(out_path, index=False)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reuse-risks", action="store_true")
    ap.add_argument("--endpoint", default="", help="fit the models for this endpoint only and write its risks (run one process per endpoint, "
                                                    "then once more with --reuse-risks for the tests)")
    a = ap.parse_args()
    t0 = time.time()
    cuda_preinit()
    dev = a.device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20261013)
    risks_path = GS / "few_labels_representation_risks_2026-10-13.parquet"
    try:
        preds = pd.read_parquet(require(L.PREDS))
        prev = pd.read_parquet(require(GS / "label_efficiency_significance_risks_2026-10-04.parquet"))
        raw = pd.read_parquet(require(GS / "few_labels_raw_risks_2026-10-12.parquet"))
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    preds = preds[preds.age_adjustment == "linear"]
    endpoints = sorted(preds.endpoint.unique())
    part = lambda ep: risks_path.with_name(risks_path.stem + f".{ep}.parquet")
    if a.endpoint:
        compute_risks(preds, [a.endpoint], dev, a.iters, part(a.endpoint))
        return
    if a.reuse_risks:
        R = pd.concat([pd.read_parquet(require(part(ep))) for ep in endpoints], ignore_index=True)
    else:
        R = compute_risks(preds, endpoints, dev, a.iters, risks_path)
    R = raw.merge(R, on=["endpoint", "dnam_id"], validate="one_to_one").merge(
        prev.drop(columns=["outer_fold", "event", "duration"]), on=["endpoint", "dnam_id"], validate="one_to_one")
    models = NEW + STORED
    rcols = [f"{m}|{fr}|{rp}" for fr, rp in L.CONFIGS for m in models]
    assert R[rcols].notna().all().all()

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

    rng = np.random.default_rng(20261013)
    D = a.draws
    boot = {}
    for ep in endpoints:
        g, ev, du, folds = per_ep[ep]
        comp = torch.zeros(D, dtype=torch.float64, device=dev)
        con = {c: comp.clone() for c in rcols}
        tie = {c: comp.clone() for c in rcols}
        for s in folds:
            W = torch.as_tensor(L.weights_within(rng, len(s), D), device=dev)
            first = True
            for k in range(0, len(rcols), 9):   # nine risk vectors at a time keeps the pair matrices in memory
                cols = rcols[k:k + 9]
                A, Cs, Ts = L.pair_mats(ev[s], du[s], [g[c].to_numpy(float)[s] for c in cols], dev)
                if first:
                    comp += L.wcount(W, A)
                    first = False
                for c, Cm, Tm in zip(cols, Cs, Ts):
                    con[c] += L.wcount(W, Cm)
                    tie[c] += L.wcount(W, Tm)
                del A, Cs, Ts
        boot[ep] = {c: ((con[c] + 0.5 * tie[c]) / comp).cpu().numpy() for c in rcols}
        log(f"bootstrap {ep} done ({time.time() - t0:.0f} s)")

    fracs = sorted({fr for fr, _ in L.CONFIGS})

    def draws(ep, m, frac):
        return np.mean([boot[ep][f"{m}|{frac}|{rp}"] for fr, rp in L.CONFIGS if fr == frac], 0)

    def pval(d):
        return float(min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (len(d) + 1)))

    rows, means = [], []
    for frac in fracs:
        pt = cidx[np.isclose(cidx.fraction, frac)].groupby(["endpoint", "model"]).c.mean().unstack()
        for m in models:
            if m == "factor_scores":
                continue
            dm = np.mean([draws(ep, m, frac) - draws(ep, "factor_scores", frac) for ep in endpoints], 0)
            means.append({"fraction": frac, "contrast": f"{NAMES[m]} minus {NAMES['factor_scores']}", "mean_c_first": float(pt[m].mean()),
                          "mean_c_second": float(pt["factor_scores"].mean()), "difference": float(pt[m].mean() - pt["factor_scores"].mean()),
                          "ci95_low": float(np.percentile(dm, 2.5)), "ci95_high": float(np.percentile(dm, 97.5)), "p": pval(dm),
                          "endpoints_first_higher": int((pt[m] > pt["factor_scores"]).sum())})
            for ep in endpoints:
                d = draws(ep, m, frac) - draws(ep, "factor_scores", frac)
                rows.append({"fraction": frac, "contrast": f"{NAMES[m]} minus {NAMES['factor_scores']}", "endpoint": ep, "c_first": float(pt.loc[ep, m]),
                             "c_second": float(pt.loc[ep, "factor_scores"]), "difference": float(pt.loc[ep, m] - pt.loc[ep, "factor_scores"]),
                             "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)), "p": pval(d)})
    df = pd.DataFrame(rows)
    df["q_bh"] = np.nan
    for con_name in df.contrast.unique():
        for full in (False, True):
            msk = (df.contrast == con_name) & (np.isclose(df.fraction, 1.0) == full)
            df.loc[msk, "q_bh"] = L.bh(df.loc[msk, "p"].to_numpy())
    df.to_csv(GS / "few_labels_representations_2026-10-13.csv", index=False)
    tab = cidx.groupby(["fraction", "model"]).c.mean().unstack()[list(models)].rename(columns=NAMES)
    log("mean C-index over eight endpoints by fraction of fitting participants:\n" + tab.round(4).to_string())
    for r in means:
        log(f"fraction {r['fraction']}: {r['contrast']}: {r['difference']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] P {r['p']:.3f}; {r['endpoints_first_higher']} of 8")
    write_json(GS / "few_labels_representations_2026-10-13.json",
               {"design": __doc__.split("Outputs")[0].strip(), "draws": D, "mean_c_by_fraction": tab.reset_index().to_dict(orient="records"),
                "means": means, "elapsed_s": time.time() - t0})


if __name__ == "__main__":
    main()
