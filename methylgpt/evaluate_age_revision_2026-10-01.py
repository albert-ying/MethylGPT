#!/usr/bin/env python3
"""Age-prediction analyses of the revision, for one or more MethylGPT-2 checkpoints.

For each checkpoint (aging-heads or multi-task) and for the ElasticNet trained on the same
labels (baseline_corpus_age_2026-10-01.py, saved model):
  * AltumAge test (4,626 curated ages, excluded from all training): MedAE/MAE/r with all CpGs,
    with 30/50/70/90% of each profile's CpGs removed at random (seeded), and with only the
    vocabulary CpGs of the 27K, 450K or EPIC v2 array. ElasticNet fills missing CpGs with
    training means.
  * Held-out test-split datasets with metadata ages (datasets never used in training), all and
    unflagged, and by tissue (>= 50 profiles).
  * AltumAge test by tissue.
  * Published clocks on the same AltumAge test profiles, computed from each sample's full array
    (ClockBase: Horvath 2013, Hannum, Lin, Horvath skin & blood; pyaging: AltumAge and PC clocks;
    Zhang 2019 elastic net from the v12 full-matrix rerun), each on the profiles it scores.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import (cuda_preinit, ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, age_scores, load_altumage_split, log,
                            require, write_json)
from methylgpt_model import BioHeads, Heads, embed_batch, load_checkpoint

CLOCKBASE = Path("/Entropy/kying/GEO_data/GEO_hs_dnam")
ZHANG = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_round3_20260703/revision_analyses/"
             "46_mean_imputed_public_clock_benchmark/results/zhang2019_full_matrix_altumage_predictions.csv")
PLAT = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")


def cluster_ci(pred, y, groups, n=2000, seed=20261001):
    """95% interval of MedAE from resampling datasets (groups) with replacement."""
    rng = np.random.default_rng(seed)
    e = np.abs(np.asarray(pred) - np.asarray(y))
    g = pd.Series(e).groupby(np.asarray(groups)).apply(np.asarray).tolist()
    draws = [np.median(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(n)]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def paired_ci(pa, pb, y, groups, n=2000, seed=20261002):
    """Difference in MedAE (a - b) with a 95% interval from resampling datasets."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(np.asarray(pa) - y), np.abs(np.asarray(pb) - y)
    idx = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(np.asarray).tolist()
    d = []
    for _ in range(n):
        k = np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))])
        d.append(np.median(ea[k]) - np.median(eb[k]))
    return {"difference": float(np.median(ea) - np.median(eb)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(Path(ck["ckpt"]), ck.get("config", {}).get("core"), dev)
    model.load_state_dict(ck["model"]); model.eval()
    core_only = bool(ck.get("config", {}).get("core_only", False))
    d_in = core.rank if core_only else core.rank + model.latents.shape[1]
    heads = (BioHeads(d_in, len(ck["tissues"]) + 1, len(ck["bio_names"])) if "bio_names" in ck
             else Heads(d_in, len(ck["tissues"]) + 1)).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()

    @torch.no_grad()
    def predict(X, batch=32):
        out = []
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch]).to(dev)
            obs = torch.isfinite(x)
            e = core.posterior(x, obs.float()) / torch.sqrt(core.lam) if core_only else embed_batch(model, core, x, obs)
            out.append(heads(e)[0].float().cpu().numpy())
        return Finv(np.concatenate(out) * ck["fs"] + ck["fm"])
    return predict


def degrade(X, scen):
    X = X.copy()
    if scen.startswith("random_"):
        p = int(scen.split("_")[1]) / 100
        X[np.random.default_rng(20261001 + int(p * 100)).random(X.shape) < p] = np.nan
    elif scen != "all":
        X[:, ~PLAT[scen]] = np.nan
    return X


def clock_table(ids):
    """Full-array clock predictions for AltumAge test GSMs from ClockBase and the Zhang rerun."""
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group"]).set_index("id")
    gse = prof.reindex(ids).group.astype(str).str.split(" ").str[0]
    rows = []
    for g in sorted(set(gse.dropna())):
        frames = []
        for sub, cols in (("clock_result", {"Horvathv1": "Horvath 2013", "Hannum": "Hannum", "Lin": "Lin",
                                            "Horvathv2": "Horvath skin & blood"}),
                          ("clock_result_pyaging", {"altumage": "AltumAge", "pchorvath2013": "PC-Horvath 2013",
                                                    "pchannum": "PC-Hannum", "pcskinandblood": "PC-skin & blood"})):
            f = CLOCKBASE / sub / f"{g}.csv"
            if f.exists():
                d = pd.read_csv(f, index_col=0)
                d = d.loc[:, ~d.columns.duplicated()]
                frames.append(d[[c for c in cols if c in d.columns]].rename(columns=cols))
        if frames:
            rows.append(pd.concat(frames, axis=1))
    t = pd.concat(rows) if rows else pd.DataFrame()
    t = t[~t.index.duplicated()].reindex(ids)
    z = pd.read_csv(ZHANG, usecols=["GSM_ID", "Zhang2019_EN_full_matrix_all_samples", "zhang_observed_fraction"]).set_index("GSM_ID")
    t["Zhang 2019"] = z.reindex(ids).Zhang2019_EN_full_matrix_all_samples.to_numpy()
    t["zhang_observed_fraction"] = z.reindex(ids).zhang_observed_fraction.to_numpy()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=f"MethylGPT-2={ARTIFACTS / 'runs/multitask_bio_2026-10-01/final.pt'}",
                    help="comma-separated name=checkpoint pairs")
    ap.add_argument("--enet", default=str(ARTIFACTS / "eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scenarios", default="all,random_30,random_50,random_70,random_90,hm27,hm450,epic_v2")
    ap.add_argument("--clocks", action="store_true", help="also score the published full-array clocks")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/age_revision_2026-10-01.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    dev = torch.device(a.device)
    corpus = Corpus()
    prof = corpus.profiles
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    ids, Xte, yte = load_altumage_split("test")
    tis_te = prof.set_index("id").reindex(ids).tissue.fillna("unlabelled").to_numpy()
    grp_te = prof.set_index("id").reindex(ids).group.astype(str).to_numpy()
    ho_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    Xho, yho = corpus.get(ho_rows), lab.age_meta.to_numpy(float)[ho_rows]
    flag_ho = lab.age_group_flag.to_numpy(bool)[ho_rows]
    tis_ho = prof.tissue.to_numpy()[ho_rows]
    grp_ho = prof.group.astype(str).to_numpy()[ho_rows]

    preds = {}
    for k, spec in enumerate(a.models.split(",")):
        name, path = spec.split("=", 1)
        f = load_model(require(Path(path)), dev)
        scen = a.scenarios.split(",") if k == 0 else ["all"]   # missingness sweep for the primary model only
        preds[name] = {"test": {s: f(degrade(Xte, s)) for s in scen}, "heldout": f(Xho)}
        log(f"{name}: AltumAge test MedAE {age_scores(preds[name]['test']['all'], yte)['medae']:.3f}")
    if Path(a.enet).exists():
        z = np.load(a.enet)
        def enet(X):
            T = np.where(np.isfinite(X), X, z["mu"])
            return Finv(((T - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))
        preds["ElasticNet (same labels)"] = {"test": {s: enet(degrade(Xte, s)) for s in a.scenarios.split(",")},
                                             "heldout": enet(Xho)}
    else:
        log(f"ElasticNet model not found ({a.enet}); skipped")

    res = {"models": list(preds), "n_altumage_test": int(len(yte)), "n_heldout": int(len(yho)),
           "n_heldout_unflagged": int((~flag_ho).sum()), "altumage_test": {}, "heldout": {}, "by_tissue": {}}
    for name, p in preds.items():
        res["altumage_test"][name] = {s: age_scores(v, yte) for s, v in p["test"].items()}
        res["heldout"][name] = {"all": age_scores(p["heldout"], yho), "unflagged": age_scores(p["heldout"][~flag_ho], yho[~flag_ho])}
        res["altumage_test"][name]["all"]["medae_ci95_dataset_bootstrap"] = cluster_ci(p["test"]["all"], yte, grp_te)
        res["heldout"][name]["all"]["medae_ci95_dataset_bootstrap"] = cluster_ci(p["heldout"], yho, grp_ho)
        res["heldout"][name]["unflagged"]["medae_ci95_dataset_bootstrap"] = cluster_ci(p["heldout"][~flag_ho], yho[~flag_ho], grp_ho[~flag_ho])
        bt = {}
        for t in sorted(set(tis_te)):
            m = tis_te == t
            if m.sum() >= 50:
                bt[f"altumage_test:{t}"] = age_scores(p["test"]["all"][m], yte[m])
        for t in sorted(set(pd.Series(tis_ho).dropna())):
            m = (tis_ho == t) & ~flag_ho
            if m.sum() >= 50:
                bt[f"heldout_unflagged:{t}"] = age_scores(p["heldout"][m], yho[m])
        res["by_tissue"][name] = bt
    if a.clocks:
        ct = clock_table(ids)
        cres = {}
        first = next(iter(preds))
        for c in [c for c in ct.columns if c != "zhang_observed_fraction"]:
            v = ct[c].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            cres[c] = {"clock": age_scores(v[ok], yte[ok]),
                       **{name: age_scores(p["test"]["all"][ok], yte[ok]) for name, p in preds.items()}}
        zf = ct.zhang_observed_fraction.to_numpy(float)
        ok = np.isfinite(ct["Zhang 2019"].to_numpy(float)) & (zf >= 0.9)
        cres["Zhang 2019 (>= 90% of its CpGs observed)"] = {"clock": age_scores(ct["Zhang 2019"].to_numpy(float)[ok], yte[ok]),
                                                            **{name: age_scores(p["test"]["all"][ok], yte[ok]) for name, p in preds.items()}}
        res["clocks_on_altumage_test"] = cres
        for c, v in cres.items():
            log(f"{c}: clock MedAE {v['clock']['medae']:.2f} (n {v['clock']['n']}) vs {first} {v[first]['medae']:.2f}")
    for name in preds:
        log(f"{name}: " + ", ".join(f"{s} {v['medae']:.2f}" for s, v in res["altumage_test"][name].items())
            + f"; held-out {res['heldout'][name]['all']['medae']:.2f} / unflagged {res['heldout'][name]['unflagged']['medae']:.2f}")
    first = next(iter(preds))
    if "ElasticNet (same labels)" in preds:
        en = preds["ElasticNet (same labels)"]
        res["paired_vs_elasticnet"] = {
            name: {"altumage_test": paired_ci(p["test"]["all"], en["test"]["all"], yte, grp_te),
                   "heldout_unflagged": paired_ci(p["heldout"][~flag_ho], en["heldout"][~flag_ho], yho[~flag_ho], grp_ho[~flag_ho])}
            for name, p in preds.items() if name != "ElasticNet (same labels)"}
    np.savez_compressed(Path(a.out).with_suffix(".predictions.npz"), altumage_ids=np.asarray(ids, str), altumage_age=yte,
                        altumage_tissue=np.asarray(tis_te, str), heldout_age=yho, heldout_flag=flag_ho,
                        heldout_tissue=np.asarray(pd.Series(tis_ho).fillna("unlabelled"), str),
                        **{f"{name}|{s_}": v for name, p in preds.items() for s_, v in p["test"].items()},
                        **{f"{name}|heldout": p["heldout"] for name, p in preds.items()})
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
