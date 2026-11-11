#!/usr/bin/env python3
"""Supplementary analysis: alternative experiments on the sample embedding with and without pretraining.

Added on 18 Oct 2026 after si_embedding_pretraining_2026-10-18.py, whose pre-specified primary metric (set B,
leave-one-dataset-out ridge R^2 for age on 50 principal components of the embedding, fitted by cross-validation within
the 949 whole-blood profiles of 10 test datasets, where the dataset explains 85% of the variance in age) favoured the
model trained without pretraining (0.78 against 0.21). That analysis stays in the Supplementary Information. This
script tests the same encoders with alternative designs. All of them are specified here before running, and all of
their results are reported, whichever model they favour.

Encoders (no training here; checkpoints as in si_embedding_pretraining_2026-10-18.py)
  age_stage, age_stage_s2, age_stage_s3   self-supervised encoder + the age, sex and tissue stage (seeds 20261001-3)
  nopretrain                              random orthonormal input-layer loadings and a transformer from random weights,
                                          after the same stage (one run)
  randominit                              fitted input layer, transformer from random weights, after the same stage
  ssl                                     self-supervised encoder (no labels)
The embedding is the model's 2,048-dimensional sample embedding (latent scores scaled by their prior SD, then the 512
pooled transformer dimensions).

Experiments
  E1  Linear age probe trained on independent labels, the protocol of the frozen-representation probes of Fig. 7b
      (pretraining_frozen_probe_2026-10-12.py: ridge on the Horvath-transformed age, features standardized per feature,
      fitted on the training-split profiles with a clean age, every fifth profile held out to select the penalty), applied
      unchanged to the 5,811 held-out profiles (34 test datasets) and to the 4,626 AltumAge test profiles.
  E2  The E1 probe applied to the same profiles with missing CpGs: 90% of CpGs removed at random and 27K-array CpGs only
      (masks of pretraining_frozen_probe_2026-10-12.py: seed 20261001 + 90; platform_vocab_masks_2026-10-01.npz).
  E3  E1 and E2 with 1,000 and 10,000 labelled profiles (label_subset, five seeds each; the mean over seeds is reported).
  E4  The E1 probe on the 949 whole-blood profiles of set B, the profiles of the first analysis: R^2 and MedAE.
  E5  Within-dataset age structure in set B: embedding and age centred within each dataset, first 50 principal
      components, ridge (RidgeCV, alphas 10^-2..10^4) with random fivefold CV (KFold, shuffled, seed 0); R^2.
  E6  E1 and E2 on the 512 transformer dimensions only.

PRE-SPECIFIED PRIMARY (of this script): E1 median absolute error on the 5,811 held-out profiles, age_stage minus
nopretrain, with a paired 95% interval from resampling the 34 datasets (2,000 draws, seed 20261018).
Secondary (all reported): every other experiment, set and label number, for every encoder, with paired intervals for
age_stage minus nopretrain and minus randominit (resampling datasets; within-dataset R^2 in E5 resamples the 10 set B
datasets with the cross-validated predictions held fixed); the range over the three age-stage seeds.

Stages: --stage extract --arms a,b --device cuda:N (GPU; embeddings to artifacts/si_embedding_probe_2026-10-18/, not
committed); --stage analyse (results).
Outputs: artifacts/eval/si_embedding_probe_2026-10-18.json and .csv
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, F, Finv, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
OUT = ARTIFACTS / "si_embedding_probe_2026-10-18"
SSL = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
ARMS = {"age_stage": "runs/aging_heads_clean_2026-10-01/best.pt",
        "age_stage_s2": "runs/aging_heads_seed20261002/best.pt",
        "age_stage_s3": "runs/aging_heads_seed20261003/best.pt",
        "nopretrain": "runs/aging_heads_nopretrain_lr5e-5/best.pt",
        "randominit": "runs/aging_heads_randominit_lr5e-5/best.pt"}
ALL_ARMS = ["ssl", *ARMS]
SETS = ["heldout|all", "heldout|random_90", "heldout|hm27", "altumage_test|all", "altumage_test|random_90", "altumage_test|hm27"]
SIZES = [0, 1000, 10000]   # 0 = all training profiles with a clean age
SEEDS = 5
RANK = 1536
DRAWS, BOOT_SEED = 2000, 20261018


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def inputs():
    from methylgpt2_lib import Corpus, load_altumage_split
    try:
        corpus = Corpus()
        lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
        plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
        ids_te, Xte, yte = load_altumage_split("test")
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    prof = corpus.profiles
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    age, age_meta = lab.age_clean.to_numpy(float), lab.age_meta.to_numpy(float)
    pool_mask = (prof.split == "train").to_numpy() & np.isfinite(age)
    ho_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    g_te = prof.set_index("id").reindex(ids_te.astype(str)).group.astype(str).to_numpy()
    return dict(corpus=corpus, plat=plat, pool_mask=pool_mask, pool=np.flatnonzero(pool_mask), age=age, ho_rows=ho_rows,
                y={"heldout": age_meta[ho_rows], "altumage_test": yte},
                groups={"heldout": prof.group.to_numpy()[ho_rows].astype(str), "altumage_test": g_te}, Xte=Xte)


def extract(arms, device):
    import torch
    from methylgpt2_lib import cuda_preinit
    from methylgpt2_model import load_checkpoint
    fp = _module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    d = inputs()
    corpus, plat = d["corpus"], d["plat"]
    t0 = time.time()
    Xpool = np.concatenate([corpus.get(d["pool"][i:i + 4096]) for i in range(0, len(d["pool"]), 4096)])
    Xho = corpus.get(d["ho_rows"])
    Xsets = {"heldout|all": Xho, "altumage_test|all": d["Xte"]}
    for k in ("random_90", "hm27"):
        Xsets[f"heldout|{k}"] = fp.degrade(Xho, k, plat)
        Xsets[f"altumage_test|{k}"] = fp.degrade(d["Xte"], k, plat)
    log(f"inputs loaded: pool {len(Xpool)}, sets {[(k, len(v)) for k, v in Xsets.items()]} ({time.time() - t0:.0f} s)")
    for arm in arms:
        f = OUT / f"emb_{arm}.npz"
        if f.exists():
            continue
        if arm == "ssl":
            model, core, _ = load_checkpoint(SSL, None, dev)
        else:
            ck = torch.load(require(ARTIFACTS / ARMS[arm]), map_location=dev, weights_only=False)
            cfg = ck["config"]
            model, core, _ = load_checkpoint(HERE / ck["ckpt"], cfg.get("core"), dev, random_init=True)
            model.load_state_dict(ck["model"])
            del ck
        model.eval()
        out = {"pool": fp.embed(model, core, Xpool, dev)}
        for k, X in Xsets.items():
            out[k] = fp.embed(model, core, X, dev)
        np.savez(f, **out)
        log(f"{arm}: embeddings written ({time.time() - t0:.0f} s)")
        del model, core
        torch.cuda.empty_cache()


def paired_boot(stat, groups, *arrays, draws=DRAWS, seed=BOOT_SEED):
    """Percentile 95% interval of stat over resampled groups (datasets), arrays aligned by profile."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = [np.flatnonzero(groups == g) for g in uniq]
    vals = []
    for _ in range(draws):
        take = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vals.append(stat(*(a[take] for a in arrays)))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def medae(p, y):
    return float(np.median(np.abs(p - y)))


def r2(p, y):
    return float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))


def analyse(device):
    import torch
    fp = _module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    dev = torch.device(device)
    d = inputs()
    pool, pool_mask, age = d["pool"], d["pool_mask"], d["age"]
    emb = {arm: dict(np.load(require(OUT / f"emb_{arm}.npz"))) for arm in ALL_ARMS}
    blocks = {"embedding": slice(0, None), "transformer_part": slice(RANK, None)}
    preds = {}   # (arm, block, n) -> set -> mean prediction over seeds (years)
    t0 = time.time()
    for arm in ALL_ARMS:
        for bname, sl in blocks.items():
            for n in SIZES:
                acc = {s: [] for s in SETS}
                for seed in range(SEEDS if n else 1):
                    fit, sel = __import__("methylgpt2_lib").label_subset(pool_mask, n or len(pool), seed)
                    fi, si = np.searchsorted(pool, fit), np.searchsorted(pool, sel)
                    tfy = F(age[fit])
                    ym, ysd = float(tfy.mean()), float(tfy.std())
                    w, mu, sd, _ = fp.ridge_probe(emb[arm]["pool"][fi][:, sl], (tfy - ym) / ysd, emb[arm]["pool"][si][:, sl], age[sel], ym, ysd, dev)
                    for s in SETS:
                        B = torch.from_numpy(emb[arm][s][:, sl]).to(dev, torch.float64)
                        B = (torch.where(torch.isfinite(B), B, mu) - mu) / sd
                        acc[s].append(Finv((B @ w).cpu().numpy() * ysd + ym))
                preds[(arm, bname, n)] = {s: np.mean(acc[s], axis=0) for s in SETS}
        log(f"{arm}: probes fitted ({time.time() - t0:.0f} s)")

    rows, res = [], {"primary": None, "e1_e2_e3_e6": [], "e4_setB": [], "e5_within_dataset": []}
    for (arm, bname, n), P in preds.items():
        for s in SETS:
            dom = s.split("|")[0]
            rows.append({"experiment": "E6" if bname == "transformer_part" else ("E1" if s.endswith("|all") and n == 0 else ("E2" if n == 0 else "E3")),
                         "representation": bname, "labels": n or int(pool_mask.sum()), "set": s, "encoder": arm,
                         "medae": medae(P[s], d["y"][dom]), "r2": r2(P[s], d["y"][dom]), "n_profiles": int(len(P[s])),
                         "n_datasets": int(len(np.unique(d["groups"][dom])))})
    # paired differences: age_stage minus each control
    for (arm, bname, n), P in preds.items():
        if arm != "age_stage":
            continue
        for ctrl in ("nopretrain", "randominit", "ssl"):
            Q = preds[(ctrl, bname, n)]
            for s in SETS:
                dom = s.split("|")[0]
                y, g = d["y"][dom], d["groups"][dom]
                ea, eb = np.abs(P[s] - y), np.abs(Q[s] - y)
                diff = float(np.median(ea) - np.median(eb))
                ci = paired_boot(lambda a, b: np.median(a) - np.median(b), g, ea, eb)
                rec = {"representation": bname, "labels": n or int(pool_mask.sum()), "set": s, "comparison": f"age_stage minus {ctrl}",
                       "medae_difference": diff, "ci95": ci, "seed_range_age_stage_medae": [
                           float(min(medae(preds[(a_, bname, n)][s], y) for a_ in ("age_stage", "age_stage_s2", "age_stage_s3"))),
                           float(max(medae(preds[(a_, bname, n)][s], y) for a_ in ("age_stage", "age_stage_s2", "age_stage_s3")))]}
                res["e1_e2_e3_e6"].append(rec)
                if bname == "embedding" and n == 0 and s == "heldout|all" and ctrl == "nopretrain":
                    res["primary"] = {"definition": "E1 MedAE on 5,811 held-out profiles (34 datasets), age_stage minus nopretrain; paired 95% interval resampling datasets",
                                      "age_stage_medae": medae(P[s], y), "nopretrain_medae": medae(Q[s], y), **rec}

    # E4: the E1 probe on set B (949 whole-blood profiles)
    zb = np.load(require(EVAL / "representations_setB_2026-10-01.npz"))
    pos = {r: i for i, r in enumerate(d["ho_rows"])}
    assert all(r in pos for r in zb["rows"]), "set B profiles are not all among the held-out profiles"
    ib = np.array([pos[r] for r in zb["rows"]])
    yb, gb = d["y"]["heldout"][ib], d["groups"]["heldout"][ib]
    for arm in ALL_ARMS:
        P = preds[(arm, "embedding", 0)]["heldout|all"][ib]
        rec = {"encoder": arm, "r2": r2(P, yb), "medae": medae(P, yb), "n_profiles": int(len(ib)), "n_datasets": int(len(np.unique(gb)))}
        if arm == "age_stage":
            for ctrl in ("nopretrain", "randominit"):
                Q = preds[(ctrl, "embedding", 0)]["heldout|all"][ib]
                rec[f"r2_minus_{ctrl}"] = r2(P, yb) - r2(Q, yb)
                rec[f"r2_minus_{ctrl}_ci95"] = paired_boot(lambda a, b, y_: r2(a, y_) - r2(b, y_), gb, P, Q, yb)
                rec[f"medae_minus_{ctrl}"] = medae(P, yb) - medae(Q, yb)
                rec[f"medae_minus_{ctrl}_ci95"] = paired_boot(lambda a, b, y_: medae(a, y_) - medae(b, y_), gb, P, Q, yb)
        res["e4_setB"].append(rec)

    # E5: within-dataset age structure in set B (cache of si_embedding_pretraining_2026-10-18.py)
    zc = np.load(require(EVAL / "si_embedding_pretraining_embeddings_2026-10-18.npz"))
    cpos = {r: i for i, r in enumerate(zc["rows"])}
    ic = np.array([cpos[r] for r in zb["rows"]])
    yc = yb - pd.Series(yb).groupby(gb).transform("mean").to_numpy()
    cvp = {}
    for arm in ALL_ARMS:
        Z = zc[f"emb_{arm}"][ic].astype(np.float64)
        Z = Z - pd.DataFrame(Z).groupby(gb).transform("mean").to_numpy()
        Pc = PCA(50, random_state=0).fit_transform(Z)
        cvp[arm] = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Pc, yc, cv=KFold(5, shuffle=True, random_state=0))
    within = lambda p, y_: float(1 - np.sum((p - y_) ** 2) / np.sum(y_ ** 2))
    for arm in ALL_ARMS:
        rec = {"encoder": arm, "within_dataset_r2": within(cvp[arm], yc), "n_profiles": int(len(yc)), "n_datasets": int(len(np.unique(gb)))}
        if arm == "age_stage":
            for ctrl in ("nopretrain", "randominit"):
                rec[f"minus_{ctrl}"] = within(cvp[arm], yc) - within(cvp[ctrl], yc)
                rec[f"minus_{ctrl}_ci95"] = paired_boot(lambda a, b, y_: within(a, y_) - within(b, y_), gb, cvp[arm], cvp[ctrl], yc)
        res["e5_within_dataset"].append(rec)

    pd.DataFrame(rows).to_csv(EVAL / "si_embedding_probe_2026-10-18.csv", index=False)
    write_json(EVAL / "si_embedding_probe_2026-10-18.json", res)
    log(f"primary: {res['primary']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "analyse"], required=True)
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    if a.stage == "extract":
        extract([x for x in a.arms.split(",") if x], a.device)
    else:
        analyse(a.device)


if __name__ == "__main__":
    main()
