#!/usr/bin/env python3
"""Supplementary analysis: sample embedding structure with and without pretraining.

Question: does the age and tissue structure of the MethylGPT embedding need pretraining, or does the
supervised age, sex and tissue stage produce it from randomly initialized weights as well?

Encoders (no training in this script; only checkpoints that exist are used)
  ssl            self-supervised encoder after masked-value pretraining (runs/stopped_cf_dense_2026-09-30/best.pt)
  age_stage      age-stage encoder: ssl + the age, sex and tissue stage, seed 20261001
                 (runs/aging_heads_clean_2026-10-01/best.pt; source of release weights_age_stage.pt)
  age_stage_s2   the same stage, seed 20261002 (runs/aging_heads_seed20261002/best.pt)
  age_stage_s3   the same stage, seed 20261003 (runs/aging_heads_seed20261003/best.pt)
  nopretrain     control (iii): random orthonormal input-layer loadings (ppca_core_random_2026-10-03.npz) and a
                 transformer from random weights, after the same age, sex and tissue stage
                 (runs/aging_heads_nopretrain_lr5e-5/best.pt; one run)
  randominit     control (iv): fitted input layer, transformer from random weights, after the same stage
                 (runs/aging_heads_randominit_lr5e-5/best.pt; one run)
  references from the cached set A/B representations (representations_set{A,B}_2026-10-01.npz): raw beta values,
  latent scores (input layer posterior means) and the untrained transformer (fitted input layer, initial weights).
The embedding is the model's 2,048-dimensional sample embedding (latent scores scaled by their prior SD, then the
512 pooled transformer dimensions); the 512 transformer dimensions alone, and the 1,536 input-layer dimensions alone
(latent scores divided by their prior SD; random-loading scores for the control), are secondary representations.

Profiles: test-split datasets only (never used for any training). Set A: 2,371 profiles, six tissues, 43 datasets;
set B: 949 whole-blood profiles with sex and metadata age, 10 datasets (rows of the representation caches).
Held-out PCA (illustration only): the 5,811 test-split profiles with metadata age (34 datasets), PCA fitted on each
encoder's embeddings of these profiles.

PRE-SPECIFIED PRIMARY METRIC (written before running): in set B, the leave-one-dataset-out (GroupKFold with one fold
per dataset) ridge R^2 for metadata age (years) on the first 50 principal components of the embedding, age-stage
encoder (seed 20261001) minus the no-pretraining control (iii); 95% interval from resampling the 10 datasets
(2,000 draws, seed 20261018) with the cross-validated predictions held fixed.

Secondary metrics (all reported): for every encoder and reference, set A tissue silhouette (cosine, 50 PCs) and
tissue 10-NN balanced accuracy (stratified 5-fold and leave-one-dataset-out); age ridge R^2 on 50 PCs and 10-NN
(cosine) median absolute error with random 5-fold CV (KFold, shuffled, seed 0, the protocol of the 2026-10-01
benchmark) and with leave-one-dataset-out CV, in set B and within each set A tissue (profiles with metadata age in
tissues whose age-labelled profiles come from at least two datasets; per tissue and pooled over tissues); GroupKFold with 5 folds as a sensitivity analysis; the same for the 512
transformer dimensions; the range over the three age-stage seeds. Intervals resample datasets (2,000 draws).
Composition of the age-labelled data (figure panel): 41,845 training profiles with a clean age, 1,366 AltumAge
validation and 4,626 AltumAge test profiles (4,016 in split excluded_downstream, i.e. from training-split datasets,
and 610 others) and 5,811 held-out profiles, by tissue, with ages.

Stages: --stage extract (GPU, cuda:1) writes the embedding cache; --stage analyse (CPU) writes the results.
Outputs
  artifacts/eval/si_embedding_pretraining_embeddings_2026-10-18.npz   embeddings (cache)
  artifacts/eval/si_embedding_pretraining_2026-10-18.json             all metrics, checks, composition counts
  artifacts/eval/si_embedding_pretraining_2026-10-18.csv              table rows
  artifacts/eval/si_embedding_pretraining_pca_2026-10-18.npz          held-out PCA coordinates (figure)
  artifacts/eval/si_age_label_composition_2026-10-18.csv              composition by split and tissue (figure)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
CACHE = EVAL / "si_embedding_pretraining_embeddings_2026-10-18.npz"
SSL = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
ARMS = {"age_stage": "runs/aging_heads_clean_2026-10-01/best.pt",
        "age_stage_s2": "runs/aging_heads_seed20261002/best.pt",
        "age_stage_s3": "runs/aging_heads_seed20261003/best.pt",
        "nopretrain": "runs/aging_heads_nopretrain_lr5e-5/best.pt",
        "randominit": "runs/aging_heads_randominit_lr5e-5/best.pt"}
RANK = 1536


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract(device):
    import torch
    from methylgpt_lib import Corpus, cuda_preinit
    from methylgpt_model import embed_batch, load_checkpoint
    cuda_preinit()
    dev = torch.device(device)
    corpus = Corpus()
    prof = corpus.profiles
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    za = np.load(require(EVAL / "representations_setA_2026-10-01.npz"))
    zb = np.load(require(EVAL / "representations_setB_2026-10-01.npz"))
    ho = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    rows = np.unique(np.concatenate([za["rows"], zb["rows"], ho]))
    log(f"embedding {len(rows)} profiles (set A {len(za['rows'])}, set B {len(zb['rows'])}, held-out {len(ho)})")
    checks = {}
    # release bundle: checksum of weights_age_stage.pt and identity with the age-stage checkpoint
    rel = ARTIFACTS / "release/methylgpt2-1.1"
    sums = dict(line.split()[::-1] for line in (HERE / "package/methylgpt-1.1.checksums.sha256").read_text().split("\n") if line.strip())
    checks["weights_age_stage_sha256_matches_package"] = sha256(rel / "weights_age_stage.pt") == sums["weights_age_stage.pt"]
    w = torch.load(rel / "weights_age_stage.pt", map_location="cpu", mmap=True, weights_only=False)["encoder"]
    ck = torch.load(ARTIFACTS / ARMS["age_stage"], map_location="cpu", mmap=True, weights_only=False)["model"]
    checks["weights_age_stage_equals_age_stage_checkpoint_max_abs_diff"] = float(max((w[k].float() - ck[k].float()).abs().max() for k in w))
    del w, ck
    log(f"release checks {checks}")
    X = corpus.get(rows)   # rows sorted

    @torch.no_grad()
    def embed(model, core):
        out = np.empty((len(X), RANK + model.latents.shape[1]), np.float32)
        for i in range(0, len(X), 32):
            x = torch.from_numpy(X[i:i + 32]).to(dev)
            out[i:i + 32] = embed_batch(model, core, x, torch.isfinite(x)).float().cpu().numpy()
        return out

    emb, info = {}, {}
    model, core, _ = load_checkpoint(SSL, None, dev)
    emb["ssl"] = embed(model, core)
    info["ssl"] = {"checkpoint": str(SSL.relative_to(HERE))}
    del model, core
    torch.cuda.empty_cache()
    for arm, rel_path in ARMS.items():
        ck = torch.load(require(ARTIFACTS / rel_path), map_location=dev, weights_only=False)
        cfg = ck["config"]
        model, core, _ = load_checkpoint(HERE / ck["ckpt"], cfg.get("core"), dev, random_init=True)
        model.load_state_dict(ck["model"])
        model.eval()
        emb[arm] = embed(model, core)
        info[arm] = {"checkpoint": f"artifacts/{rel_path}", "step": int(ck["step"]), "seed": cfg.get("seed"), "core": cfg.get("core"),
                     "random_init": cfg.get("random_init"), "lr_enc": cfg.get("lr_enc"), "valid_medae": ck["valid"]["medae"]}
        log(f"{arm}: step {ck['step']} core {cfg.get('core')} random_init {cfg.get('random_init')}")
        del model, core, ck
        torch.cuda.empty_cache()
    # checks: row order against the 2026-10-01 caches (self-supervised embedding recomputed)
    pos = {r: i for i, r in enumerate(rows)}
    for name, z in (("setA", za), ("setB", zb)):
        idx = np.array([pos[r] for r in z["rows"]])
        d = np.abs(emb["ssl"][idx] - z["rep_mgpt2_selfsupervised"])
        checks[f"ssl_vs_cache_{name}_max_abs_diff"] = float(d.max())
        checks[f"ssl_vs_cache_{name}_latent_part_max_abs_diff"] = float(d[:, :RANK].max())
    log(f"checks {checks}")
    np.savez(CACHE, rows=rows, **{f"emb_{k}": v for k, v in emb.items()})
    write_json(EVAL / "si_embedding_pretraining_extract_record_2026-10-18.json", {"checks": checks, "arms": info, "n_rows": int(len(rows))})


def _helpers():
    spec = importlib.util.spec_from_file_location("si_sample_representations", HERE / "si_sample_representations_2026-10-18.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def boot_r2_within(y, pred, codes, unit_stratum, W):
    """Pooled within-stratum R^2 = 1 - sum SSE / sum SST, SST about each stratum's own mean; exact per resampled set."""
    G = W.shape[1]
    st = np.asarray(unit_stratum)
    out_sse, out_sst = 0.0, 0.0
    pt_sse, pt_sst = 0.0, 0.0
    for s in np.unique(st):
        cols = np.flatnonzero(st == s)
        m = np.isin(codes, cols)
        sums = [np.bincount(codes[m], weights=v, minlength=G)[cols] for v in (np.ones(m.sum()), y[m], y[m] ** 2, (y[m] - pred[m]) ** 2)]
        n, sy, syy, sse = (W[:, cols] @ v for v in sums)
        with np.errstate(invalid="ignore", divide="ignore"):
            out_sse = out_sse + sse
            out_sst = out_sst + np.where(n > 0, syy - sy ** 2 / np.where(n > 0, n, 1), 0.0)
        pt_sse += ((y[m] - pred[m]) ** 2).sum()
        pt_sst += ((y[m] - y[m].mean()) ** 2).sum()
    return float(1 - pt_sse / pt_sst), 1 - out_sse / out_sst


def composition():
    """Age-labelled profiles by split group and tissue, with ages (figure panel a)."""
    from methylgpt_lib import load_altumage_split
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    age_clean, age_meta = lab.age_clean.to_numpy(float), lab.age_meta.to_numpy(float)
    parts = []
    tr = (prof.split == "train").to_numpy() & np.isfinite(age_clean)
    parts.append(pd.DataFrame({"group": "training (metadata ages)", "tissue": prof.tissue[tr].to_numpy(), "age": age_clean[tr],
                               "dataset": prof.group[tr].astype(str).to_numpy()}))
    pidx = prof.set_index("id")
    for name, split in (("valid", "AltumAge validation"), ("test", "AltumAge test")):
        ids, _, y = load_altumage_split(name)
        p = pidx.reindex(ids.astype(str))
        assert p.split.notna().all()
        g = np.where(p.split.to_numpy() == "excluded_downstream", f"{split}, training-split datasets", f"{split}, other datasets") \
            if name == "test" else np.full(len(ids), split)
        parts.append(pd.DataFrame({"group": g, "tissue": p.tissue.to_numpy(), "age": y, "dataset": p.group.astype(str).to_numpy(),
                                   "corpus_split": p.split.to_numpy()}))
    ho = (prof.split == "test").to_numpy() & np.isfinite(age_meta)
    parts.append(pd.DataFrame({"group": "held-out test-split datasets", "tissue": prof.tissue[ho].to_numpy(), "age": age_meta[ho],
                               "dataset": prof.group[ho].astype(str).to_numpy()}))
    comp = pd.concat(parts, ignore_index=True)
    comp["tissue"] = comp.tissue.fillna("no tissue label")
    counts = {g: {"profiles": int((comp.group == g).sum()), "datasets": int(comp[comp.group == g].dataset.nunique())} for g in comp.group.unique()}
    st1 = prof.split.value_counts().to_dict()
    checks = {"training_41845": counts["training (metadata ages)"]["profiles"] == 41845,
              "altumage_validation_1366": counts["AltumAge validation"]["profiles"] == 1366,
              "altumage_test_4626": counts["AltumAge test, training-split datasets"]["profiles"] + counts["AltumAge test, other datasets"]["profiles"] == 4626,
              "altumage_test_4016_training_split_datasets": counts["AltumAge test, training-split datasets"]["profiles"] == 4016,
              "altumage_test_610_other": counts["AltumAge test, other datasets"]["profiles"] == 610,
              "heldout_5811": counts["held-out test-split datasets"]["profiles"] == 5811,
              "heldout_34_datasets": counts["held-out test-split datasets"]["datasets"] == 34,
              "st1_excluded_downstream_equals_valid_plus_test_in_excluded_downstream":
                  int(st1.get("excluded_downstream", 0)) == int((comp.corpus_split == "excluded_downstream").sum()),
              "st1_split_counts": {k: int(v) for k, v in st1.items()}}
    ens = __import__("json").loads(require(EVAL / "age_ensemble_2026-10-04.json").read_text())
    checks["age_ensemble_altumage_test_n"] = ens["a_altumage_test"]["ensemble"]["n"]
    checks["age_ensemble_outside_training_datasets_n"] = ens["f_altumage_test_outside_training_datasets"]["n_profiles"]
    checks["age_ensemble_heldout_n"] = ens["c_heldout"]["all_34_datasets"]["n_profiles"]
    checks["matches_age_ensemble"] = (checks["age_ensemble_altumage_test_n"] == 4626 and checks["age_ensemble_outside_training_datasets_n"] == 610
                                      and checks["age_ensemble_heldout_n"] == 5811)
    tab = comp.groupby(["group", "tissue"]).agg(profiles=("age", "size"), datasets=("dataset", "nunique"), age_median=("age", "median"),
                                               age_q1=("age", lambda a: float(np.percentile(a, 25))), age_q3=("age", lambda a: float(np.percentile(a, 75))),
                                               age_min=("age", "min"), age_max=("age", "max")).reset_index()
    tab.to_csv(EVAL / "si_age_label_composition_2026-10-18.csv", index=False)
    np.savez(EVAL / "si_age_label_composition_2026-10-18.npz", group=comp.group.to_numpy(str), tissue=comp.tissue.to_numpy(str), age=comp.age.to_numpy(float))
    return counts, checks


def analyse():
    sr = _helpers()
    t0 = time.time()
    z = np.load(require(CACHE))
    rows = z["rows"]
    emb = {k[4:]: z[k] for k in z.files if k.startswith("emb_")}
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    rec = __import__("json").loads(require(EVAL / "si_embedding_pretraining_extract_record_2026-10-18.json").read_text())
    sets = sr.load_sets()
    pos = {r: i for i, r in enumerate(rows)}
    res = {"primary_metric": "set B leave-one-dataset-out ridge R^2 for age (50 PCs of the embedding), age-stage encoder (seed 20261001) "
                             "minus the no-pretraining control; 95% interval from resampling the 10 datasets, 2,000 draws",
           "bootstrap": {"draws": sr.NBOOT, "seed": sr.SEED}, "extract_record": rec}
    arms = ["ssl", "age_stage", "age_stage_s2", "age_stage_s3", "randominit", "nopretrain"]
    rows_csv = []

    def add(set_, rep, metric, cv, e, n, nd, note=""):
        rows_csv.append({"set": set_, "representation": rep, "metric": metric, "cv": cv, "value": e["value"], "ci95_low": e["ci95"][0],
                         "ci95_high": e["ci95"][1], "n_profiles": n, "n_datasets": nd, "note": note})

    def representations(setname):
        reps_c, meta = sets[setname]
        idx = np.array([pos[r] for r in meta.row.to_numpy()])
        reps = {"raw": reps_c["raw"], "latent_scores": reps_c["factor_scores"], "untrained_transformer": reps_c["untrained_transformer"]}
        for a in arms:
            reps[a] = emb[a][idx]
        for a in ["untrained_transformer"] + arms:
            reps[f"{a}|transformer_dims"] = reps[a][:, RANK:]
        # the input-layer part of the embedding: latent scores divided by their prior SD (identical for every arm with the
        # fitted input layer), and the same for the random orthonormal loadings of the no-pretraining control
        reps["latent_scores_scaled"] = emb["ssl"][idx][:, :RANK]
        reps["nopretrain|latent_part"] = emb["nopretrain"][idx][:, :RANK]
        return reps, meta

    # ---------------- set A: tissue structure, and age within tissue ----------------
    reps, meta = representations("A")
    tissue, ds = meta.tissue.to_numpy(), meta.dataset.to_numpy()
    nA, ndA = len(meta), meta.dataset.nunique()
    cA, WA = sr.boot_weights((meta.dataset + "|" + meta.tissue).to_numpy(), strata=tissue)
    ra = {}
    for name, Z in reps.items():
        P = sr.pcs(Z)
        Zg = sr.geom(Z)
        d = {"tissue_silhouette": sr.entry(*sr.boot_silhouette(P, tissue, cA, WA))}
        for scheme in ("stratified5", "lodo"):
            pr = sr.knn_oof(Zg, tissue, sr.splits(scheme, tissue, ds))
            d[f"tissue_knn_bacc|{scheme}"] = sr.entry(*sr.boot_bacc(tissue, pr, cA, WA))
        ra[name] = d
        for k, v in d.items():
            m, _, cv = k.partition("|")
            add("A", name, m, cv or "none", v, nA, ndA)
        log(f"A {name}: sil {d['tissue_silhouette']['value']:.3f} knn {d['tissue_knn_bacc|stratified5']['value']:.3f} / {d['tissue_knn_bacc|lodo']['value']:.3f}")
    res["set_a_tissue"] = ra

    # age within tissue (profiles with metadata age)
    # tissues whose age-labelled profiles come from at least two datasets (leave-one-dataset-out needs two)
    agedat = meta[np.isfinite(meta.age.to_numpy())].groupby("tissue").dataset.nunique()
    age_tissues = sorted(agedat[agedat >= 2].index)
    hasage = np.isfinite(meta.age.to_numpy()) & meta.tissue.isin(age_tissues).to_numpy()
    res["set_a_age_tissues_excluded_single_dataset"] = sorted(agedat[agedat < 2].index)
    am = meta[hasage].reset_index(drop=True)
    unit = (am.dataset + "|" + am.tissue).to_numpy()
    cAa, WAa = sr.boot_weights(unit, strata=am.tissue.to_numpy())
    u_lv = np.unique(unit)
    u_stratum = np.array([u.split("|", 1)[1] for u in u_lv])
    y = am.age.to_numpy()
    per_tissue_n = am.groupby("tissue").agg(n=("age", "size"), datasets=("dataset", "nunique")).to_dict("index")
    res["set_a_age_profiles"] = {"n": int(len(am)), "per_tissue": per_tissue_n}
    rwa = {}
    for name, Z in reps.items():
        Z = Z[hasage]
        pr = {s: np.empty(len(am)) for s in ("ridge|kfold5", "ridge|lodo", "knn|kfold5", "knn|lodo")}
        per = {}
        for t in sorted(am.tissue.unique()):
            m = (am.tissue == t).to_numpy()
            P = sr.pcs(Z[m])
            Zg = sr.geom(Z[m])
            for scheme in ("kfold5", "lodo"):
                sp = sr.splits(scheme, y[m], am.dataset.to_numpy()[m])
                pr[f"ridge|{scheme}"][m] = sr.ridge_oof(P, y[m], sp)
                pr[f"knn|{scheme}"][m] = sr.knn_reg_oof(Zg, y[m], sp)
            per[t] = {f"{k}": (float(sr.r2_score(y[m], v[m])) if k.startswith("ridge") else float(np.median(np.abs(v[m] - y[m]))))
                      for k, v in pr.items()}
        d = {"per_tissue": per}
        for scheme in ("kfold5", "lodo"):
            d[f"age_ridge_r2_within_tissue|{scheme}"] = sr.entry(*boot_r2_within(y, pr[f"ridge|{scheme}"], cAa, u_stratum, WAa))
            d[f"age_knn_medae_within_tissue|{scheme}"] = sr.entry(*sr.boot_median(np.abs(pr[f"knn|{scheme}"] - y), cAa, WAa))
        rwa[name] = d
        for k, v in d.items():
            if k != "per_tissue":
                m_, _, cv = k.partition("|")
                add("A (profiles with age, within tissue)", name, m_, cv, v, int(len(am)), int(am.dataset.nunique()))
        log(f"A age {name}: R2 {d['age_ridge_r2_within_tissue|kfold5']['value']:.3f} / {d['age_ridge_r2_within_tissue|lodo']['value']:.3f}")
    res["set_a_age_within_tissue"] = rwa

    # ---------------- set B: age with random and leave-dataset-out CV ----------------
    reps, meta = representations("B")
    batch, age = meta.dataset.to_numpy(), meta.age.to_numpy()
    nB, ndB = len(meta), meta.dataset.nunique()
    cB, WB = sr.boot_weights(batch)
    rb, oof = {}, {}
    for name, Z in reps.items():
        P = sr.pcs(Z)
        Zg = sr.geom(Z)
        d = {}
        for scheme in ("kfold5", "lodo", "groupkfold5"):
            sp = sr.splits(scheme, batch, batch)
            oof[(name, scheme)] = sr.ridge_oof(P, age, sp)
            d[f"age_ridge_r2|{scheme}"] = sr.entry(*sr.boot_r2(age, oof[(name, scheme)], cB, WB))
            d[f"age_knn_medae|{scheme}"] = sr.entry(*sr.boot_median(np.abs(sr.knn_reg_oof(Zg, age, sp) - age), cB, WB))
        d["dataset_knn_bacc|stratified5"] = sr.entry(*sr.boot_bacc(batch, sr.knn_oof(Zg, batch, sr.splits("stratified5", batch, batch)), cB, WB))
        rb[name] = d
        for k, v in d.items():
            m_, _, cv = k.partition("|")
            add("B", name, m_, cv, v, nB, ndB)
        log(f"B {name}: age R2 kfold {d['age_ridge_r2|kfold5']['value']:.3f} lodo {d['age_ridge_r2|lodo']['value']:.3f} "
            f"MedAE {d['age_knn_medae|kfold5']['value']:.2f} / {d['age_knn_medae|lodo']['value']:.2f}")
    res["set_b_age"] = rb

    def paired(a, b, scheme="lodo"):
        pa, da = sr.boot_r2(age, oof[(a, scheme)], cB, WB)
        pb, db = sr.boot_r2(age, oof[(b, scheme)], cB, WB)
        return {"a": a, "b": b, "a_value": pa, "b_value": pb, "difference": pa - pb, "ci95": sr.ci(da - db)}
    res["primary"] = {**paired("age_stage", "nopretrain"), "n_profiles": int(nB), "n_datasets": int(ndB)}
    log(f"PRIMARY {res['primary']}")
    res["paired_set_b_age_r2"] = {f"{a} minus {b} | {s}": paired(a, b, s) for s in ("lodo", "kfold5")
                                  for a, b in (("age_stage", "nopretrain"), ("age_stage", "randominit"), ("ssl", "nopretrain"),
                                               ("age_stage", "ssl"), ("age_stage", "latent_scores"), ("age_stage", "raw"),
                                               ("age_stage|transformer_dims", "nopretrain|transformer_dims"),
                                               ("latent_scores_scaled", "nopretrain|latent_part"))}
    # range over the three age-stage seeds
    seeds = ["age_stage", "age_stage_s2", "age_stage_s3"]
    rng_ = {}
    for blockname, block in (("set_a_tissue", ra), ("set_b_age", rb)):
        for k in block["age_stage"]:
            vals = [block[s][k]["value"] for s in seeds]
            rng_[f"{blockname}|{k}"] = [float(min(vals)), float(max(vals))]
    for k in rwa["age_stage"]:
        if k != "per_tissue":
            vals = [rwa[s][k]["value"] for s in seeds]
            rng_[f"set_a_age_within_tissue|{k}"] = [float(min(vals)), float(max(vals))]
    res["age_stage_three_seed_range"] = rng_

    # ---------------- held-out PCA (illustration) ----------------
    ho = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    hidx = np.array([pos[r] for r in ho])
    pca_out = {"rows": ho, "age": lab.age_meta.to_numpy(float)[ho], "tissue": prof.tissue.fillna("no tissue label").to_numpy(str)[ho],
               "dataset": prof.group.astype(str).to_numpy()[ho]}
    evr = {}
    for a in ("ssl", "age_stage", "randominit", "nopretrain"):
        E = emb[a][hidx].astype(np.float64)
        E = E - E.mean(0)
        pca_out[a] = sr.pca_transform(E, sr.pca_fit(E, 2))
        evr[a] = [float(v) for v in pca_out[a].var(0) / E.var(0).sum()]
        pca_out[f"{a}_pc_age_spearman"] = np.array([sr.spearmanr(pca_out[a][:, j], pca_out["age"]).statistic for j in range(2)])
    np.savez(EVAL / "si_embedding_pretraining_pca_2026-10-18.npz", **pca_out)
    res["heldout_pca"] = {"n_profiles": int(len(ho)), "n_datasets": int(pd.Series(pca_out["dataset"]).nunique()), "explained_variance_ratio": evr,
                          "spearman_pc_vs_age": {a: [float(v) for v in pca_out[f"{a}_pc_age_spearman"]] for a in evr}}

    counts, checks = composition()
    res["age_label_composition"] = {"counts": counts, "checks": checks}
    log(f"composition {counts} checks {checks}")
    res["elapsed_s"] = time.time() - t0
    write_json(EVAL / "si_embedding_pretraining_2026-10-18.json", res)
    pd.DataFrame(rows_csv).to_csv(EVAL / "si_embedding_pretraining_2026-10-18.csv", index=False)
    log(f"done in {res['elapsed_s']:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "analyse"], required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a.device)
    else:
        analyse()


if __name__ == "__main__":
    main()
