#!/usr/bin/env python3
"""Supplementary analyses: further tests in the settings MethylGPT is built for (missing CpGs, unseen datasets).

Written on 19 Oct 2026, before running, at the authors' request for more comparisons in these settings. Every result
of every experiment below is reported, whichever model it favours.

Experiments
  X1  Age on the 34 held-out test datasets with missing CpGs. MethylGPT (mean of the three released training runs,
      runs/multitask_bio_*/final.pt) against the elastic net trained on the same labels with missing CpGs filled with
      training means (baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz). Scenarios: 30, 50, 70 and 90% of CpGs
      removed at random and the vocabulary CpGs of the 27K, 450K or EPIC v2 array (degrade() of
      evaluate_age_revision_2026-10-01.py, same seeds). The AltumAge test set has these results in Fig. 4e; the held-out
      datasets have them only for complete profiles. Per tissue (at least 50 profiles) with 27K-array CpGs and with 90%
      of CpGs removed.
  X2  Pretraining on the same held-out profiles and scenarios: the age, sex and tissue stage with pretraining
      (runs/aging_heads_clean_2026-10-01/best.pt) against the same stage without pretraining (random input-layer
      loadings and transformer weights; runs/aging_heads_nopretrain_lr5e-5/best.pt), the comparison of Supplementary
      Fig. 3, which reports these scenarios for the AltumAge test set only.
  X3  The linear age probe of si_embedding_probe_2026-10-18.py (fitted on all training profiles, embeddings cached in
      artifacts/si_embedding_probe_2026-10-18/) applied to held-out and AltumAge test profiles with 30, 50 or 70% of CpGs
      removed and with 450K-array or EPIC v2-array CpGs; age-stage encoder (three seeds) against no pretraining.
  X4  Reconstruction by dataset: the number of the 107 held-out datasets in which MethylGPT has a lower masked-value MSE
      than nearest-neighbour imputation, for random masks of 15-70% (heldout_cf_dense_step2000_2026-09-30.per_profile)
      and the array scenarios (heldout_platform_2026-10-01.per_profile); two-sided sign test.
  X5  Sample representations with missing CpGs and each dataset held out: set A (2,371 profiles, six tissues, 43
      datasets) reduced to 27K-array CpGs or with 70% of CpGs removed (protocol of
      benchmark_representations_missing_2026-10-02.py); tissue and sex 10-nearest-neighbour balanced accuracy with
      leave-one-dataset-out CV, latent scores from the remaining CpGs against raw beta values filled with training means.

PRE-SPECIFIED PRIMARY: X1 with 27K-array CpGs, MedAE of MethylGPT minus the elastic net on the 5,811 held-out profiles,
paired 95% interval from resampling the 34 datasets (2,000 draws, seed 20261019).
Secondary (all reported): every other scenario, tissue, experiment and metric above (MedAE and mean absolute error for
X1 and X2), each with a paired 95% interval resampling datasets where datasets are the unit.
Checks: complete-profile predictions must reproduce the saved held-out predictions (age_ensemble_2026-10-04,
age_stage_sweep_2026-10-03, age_nopretrain_2026-10-03).

Stages: --stage predict (GPU) writes artifacts/si_favourable_extensions_2026-10-19/; --stage analyse writes
artifacts/eval/si_favourable_extensions_2026-10-19.json and .csv.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
OUT = ARTIFACTS / "si_favourable_extensions_2026-10-19"
PROBE = ARTIFACTS / "si_embedding_probe_2026-10-18"
RUNS = {"run 1": "runs/multitask_bio_2026-10-01/final.pt", "seed 2": "runs/multitask_bio_seed20261002/final.pt",
        "seed 3": "runs/multitask_bio_seed20261003/final.pt"}
STAGE = {"pretrained": "runs/aging_heads_clean_2026-10-01/best.pt", "no_pretraining": "runs/aging_heads_nopretrain_lr5e-5/best.pt"}
ENET = EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"
SCEN = ["all", "random_30", "random_50", "random_70", "random_90", "hm27", "hm450", "epic_v2"]
PROBE_SCEN = ["random_30", "random_50", "random_70", "hm450", "epic_v2"]
PROBE_ARMS = ["age_stage", "age_stage_s2", "age_stage_s3", "nopretrain"]
DRAWS, SEED = 2000, 20261019


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def heldout():
    from methylgpt_lib import Corpus
    corpus = Corpus()
    prof = corpus.profiles
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    return corpus, prof, rows, lab.age_meta.to_numpy(float)[rows]


def predict(device):
    import torch
    from methylgpt_lib import cuda_preinit, load_altumage_split
    ev = _module("evaluate_age_revision", "evaluate_age_revision_2026-10-01.py")
    fp = _module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    cuda_preinit()
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus, prof, rows, _ = heldout()
    Xho = corpus.get(rows)
    t0 = time.time()
    # X1, X2: age predictions on degraded held-out profiles
    out = {"rows": rows}
    z = np.load(require(ENET))
    for s in SCEN:
        X = ev.degrade(Xho, s)
        T = np.where(np.isfinite(X), X, z["mu"])
        out[f"enet|{s}"] = ev.Finv(((T - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))
    for name, rel in {**RUNS, **STAGE}.items():
        f = ev.load_model(require(ARTIFACTS / rel), dev)
        for s in SCEN:
            out[f"{name}|{s}"] = f(ev.degrade(Xho, s))
        log(f"{name}: predictions written ({time.time() - t0:.0f} s)")
        del f
        torch.cuda.empty_cache()
    np.savez(OUT / "age_predictions.npz", **out)
    # X3: embeddings of degraded profiles for the probe arms
    from methylgpt_model import load_checkpoint
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    _, Xte, _ = load_altumage_split("test")

    def deg(X, s):
        X = X.copy()
        if s.startswith("random_"):
            p = int(s.split("_")[1]) / 100
            X[np.random.default_rng(20261001 + int(p * 100)).random(X.shape) < p] = np.nan
        else:
            X[:, ~plat[s]] = np.nan
        return X
    sets = {f"{dom}|{s}": deg(X, s) for dom, X in (("heldout", Xho), ("altumage_test", Xte)) for s in PROBE_SCEN}
    se = _module("si_embedding_probe", "si_embedding_probe_2026-10-18.py")
    for arm in PROBE_ARMS:
        ck = torch.load(require(ARTIFACTS / se.ARMS[arm]), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(HERE / ck["ckpt"], ck["config"].get("core"), dev, random_init=True)
        model.load_state_dict(ck["model"]); model.eval()
        del ck
        np.savez(OUT / f"probe_emb_{arm}.npz", **{k: fp.embed(model, core, X, dev) for k, X in sets.items()})
        log(f"{arm}: probe embeddings written ({time.time() - t0:.0f} s)")
        del model, core
        torch.cuda.empty_cache()


def boot(stat, groups, *arrays, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in uniq]
    vals = []
    for _ in range(draws):
        take = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])   # one resample shared by all arrays (paired)
        vals.append(stat(*(a[take] for a in arrays)))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def paired(pa, pb, y, g):
    """MedAE and MAE of a minus b with paired dataset-resampled intervals."""
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    res = {}
    for name, f in (("medae", np.median), ("mae", np.mean)):
        res[name] = {"a": float(f(ea)), "b": float(f(eb)), "difference": float(f(ea) - f(eb)),
                     "ci95": boot(lambda u, v: f(u) - f(v), g, ea, eb)}
    return res


def analyse(device):
    import torch
    from scipy.stats import binomtest
    from sklearn.metrics import balanced_accuracy_score
    from methylgpt_lib import F, Finv, PPCACore, label_subset, load_altumage_split
    corpus, prof, rows, y = heldout()
    g = prof.group.to_numpy()[rows].astype(str)
    tissue = prof.tissue.to_numpy()[rows]
    P = dict(np.load(require(OUT / "age_predictions.npz")))
    assert np.array_equal(P["rows"], rows)
    res = {"checks": {}, "x1_methylgpt_vs_enet": {}, "x1_by_tissue": {}, "x2_pretraining": {}, "x3_probe": {}, "x4_reconstruction_by_dataset": {},
           "x5_representations_lodo": {}}
    ens = {s: np.mean([P[f"{r}|{s}"] for r in RUNS], axis=0) for s in SCEN}
    za = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    zs = np.load(require(EVAL / "age_stage_sweep_2026-10-03.predictions.npz"))
    zn = np.load(require(EVAL / "age_nopretrain_2026-10-03.predictions.npz"))
    res["checks"] = {"ensemble_heldout_max_abs_diff": float(np.max(np.abs(ens["all"] - za["ensemble|heldout"]))),
                     "pretrained_stage_heldout_max_abs_diff": float(np.max(np.abs(P["pretrained|all"] - zs["Age-sex-tissue stage|heldout"]))),
                     "no_pretraining_heldout_max_abs_diff": float(np.max(np.abs(P["no_pretraining|all"] - zn["No pretraining lr 5e-5|heldout"]))),
                     "enet_heldout_max_abs_diff": float(np.max(np.abs(P["enet|all"] - zs["ElasticNet (same labels)|heldout"])))}
    log(f"checks {res['checks']}")
    for s in SCEN:
        res["x1_methylgpt_vs_enet"][s] = paired(ens[s], P[f"enet|{s}"], y, g)
        res["x2_pretraining"][s] = paired(P[f"pretrained|{s}"], P[f"no_pretraining|{s}"], y, g)
    res["primary"] = {"definition": "X1, 27K-array CpGs: MedAE of MethylGPT (mean of three runs) minus the elastic net on 5,811 held-out profiles, 34 datasets",
                      **res["x1_methylgpt_vs_enet"]["hm27"]["medae"]}
    tis = pd.Series(tissue).fillna("no tissue label")
    for t, n in tis.value_counts().items():
        if n < 50 or t in ("no tissue label", "Other"):
            continue
        m = (tis == t).to_numpy()
        res["x1_by_tissue"][t] = {"profiles": int(m.sum()), "datasets": int(len(np.unique(g[m]))),
                                  **{s: paired(ens[s][m], P[f"enet|{s}"][m], y[m], g[m])["medae"] for s in ("all", "hm27", "random_90")}}

    # X3: probe of si_embedding_probe_2026-10-18.py, all training labels
    fp = _module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    dev = torch.device(device)
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    age = lab.age_clean.to_numpy(float)
    pool_mask = (prof.split == "train").to_numpy() & np.isfinite(age)
    pool = np.flatnonzero(pool_mask)
    fit, sel = label_subset(pool_mask, len(pool), 0)
    fi, si = np.searchsorted(pool, fit), np.searchsorted(pool, sel)
    tfy = F(age[fit]); ym, ysd = float(tfy.mean()), float(tfy.std())
    ids_te, _, yte = load_altumage_split("test")
    gte = prof.set_index("id").reindex(ids_te.astype(str)).group.astype(str).to_numpy()
    yd, gd = {"heldout": y, "altumage_test": yte}, {"heldout": g, "altumage_test": gte}
    pr = {}
    for arm in PROBE_ARMS:
        Zp = np.load(require(PROBE / f"emb_{arm}.npz"))["pool"]
        w, mu, sd, _ = fp.ridge_probe(Zp[fi], (tfy - ym) / ysd, Zp[si], age[sel], ym, ysd, dev)
        E = np.load(require(OUT / f"probe_emb_{arm}.npz"))
        for k in E.files:
            B = torch.from_numpy(E[k]).to(dev, torch.float64)
            pr[(arm, k)] = Finv((((torch.where(torch.isfinite(B), B, mu) - mu) / sd) @ w).cpu().numpy() * ysd + ym)
    for dom in ("heldout", "altumage_test"):
        for s in PROBE_SCEN:
            k = f"{dom}|{s}"
            r = paired(pr[("age_stage", k)], pr[("nopretrain", k)], yd[dom], gd[dom])["medae"]
            r["age_stage_seed_range"] = [float(min(np.median(np.abs(pr[(a_, k)] - yd[dom])) for a_ in PROBE_ARMS[:3])),
                                         float(max(np.median(np.abs(pr[(a_, k)] - yd[dom])) for a_ in PROBE_ARMS[:3]))]
            res["x3_probe"][k] = r

    # X4: reconstruction by dataset
    for name, f in (("random masks", "heldout_cf_dense_step2000_2026-09-30.per_profile.csv.gz"), ("array scenarios", "heldout_platform_2026-10-01.per_profile.csv.gz")):
        d = pd.read_csv(require(EVAL / f))
        d["group"] = d.id.map(dict(zip(prof.id, prof.group.astype(str))))
        for rate, b in d.groupby("rate"):
            agg = b.groupby("group")[["sse_methylgpt2", "sse_knn10", "n_masked"]].sum()
            wins = int((agg.sse_methylgpt2 < agg.sse_knn10).sum()); n = int(len(agg))
            rec = {"datasets_methylgpt_lower": wins, "datasets": n, "sign_test_p": float(binomtest(wins, n).pvalue)}
            if "sse_january_mlm" in b and b.sse_january_mlm.notna().any():
                bj = b[b.sse_january_mlm.notna()].groupby("group")[["sse_methylgpt2", "sse_january_mlm"]].sum()
                wj = int((bj.sse_methylgpt2 < bj.sse_january_mlm).sum())
                rec.update({"datasets_lower_than_originally_submitted_model": wj, "datasets_with_original_model": int(len(bj)),
                            "sign_test_p_original_model": float(binomtest(wj, len(bj)).pvalue)})
            res["x4_reconstruction_by_dataset"][f"{name}|{rate}"] = rec

    # X5: set A with missing CpGs, leave one dataset out
    sr = _module("si_sample_representations", "si_sample_representations_2026-10-18.py")
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    core = PPCACore.load(require(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"), rank=1536).to(dev)
    mu = core.mu.cpu().numpy()
    rows_a = np.load(require(EVAL / "representations_setA_2026-10-01.npz"))["rows"]
    Xa = corpus.get(np.sort(rows_a))[np.argsort(np.argsort(rows_a))]
    ga = prof.group.to_numpy()[rows_a].astype(str)
    labels = {"tissue": prof.tissue.to_numpy()[rows_a].astype(str), "sex": prof.sex.to_numpy()[rows_a]}
    for scen in ("hm27", "random_70"):
        X = Xa.copy()
        if scen == "hm27":
            X[:, ~plat["hm27"]] = np.nan
        else:
            X[np.random.default_rng(20261071).random(X.shape) < 0.7] = np.nan
        raw = np.where(np.isfinite(X), X, mu[None, :])
        with torch.no_grad():
            S = np.concatenate([core.posterior(torch.from_numpy(X[i:i + 64]).to(dev), torch.from_numpy(np.isfinite(X[i:i + 64])).to(dev).float()).cpu().numpy()
                                for i in range(0, len(X), 64)])
        reps = {"latent scores": S, "raw beta values (mean-filled)": raw}
        for task, lab_ in labels.items():
            keep = np.isin(lab_, ["M", "F"]) if task == "sex" else np.ones(len(lab_), bool)
            yk, gk = np.asarray(lab_)[keep].astype(str), ga[keep]
            sp = sr.splits("lodo", yk, gk)
            pred = {k: sr.knn_oof(Z[keep] - Z[keep].mean(0), yk, sp).astype(str) for k, Z in reps.items()}
            a_, b_ = pred["latent scores"], pred["raw beta values (mean-filled)"]
            ba = lambda u, v, t: balanced_accuracy_score(t, u) - balanced_accuracy_score(t, v)
            res["x5_representations_lodo"][f"{scen}|{task}"] = {
                "latent_scores": float(balanced_accuracy_score(yk, a_)), "raw_mean_filled": float(balanced_accuracy_score(yk, b_)),
                "difference": float(ba(a_, b_, yk)), "ci95": boot(ba, gk, a_, b_, yk), "profiles": int(keep.sum()), "datasets": int(len(np.unique(gk)))}
    rows_out = []
    for sec in ("x1_methylgpt_vs_enet", "x2_pretraining"):
        for s, r in res[sec].items():
            for met, v in r.items():
                rows_out.append({"experiment": sec, "scenario": s, "metric": met, "a": v["a"], "b": v["b"], "difference": v["difference"],
                                 "ci95_low": v["ci95"][0], "ci95_high": v["ci95"][1], "profiles": len(y), "datasets": len(np.unique(g))})
    pd.DataFrame(rows_out).to_csv(EVAL / "si_favourable_extensions_2026-10-19.csv", index=False)
    write_json(EVAL / "si_favourable_extensions_2026-10-19.json", res)
    log(f"primary {res['primary']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "analyse"], required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    predict(a.device) if a.stage == "predict" else analyse(a.device)


if __name__ == "__main__":
    main()
