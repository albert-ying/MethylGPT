#!/usr/bin/env python3
"""Tables behind the redrawn Fig. 3 and Fig. 4a-e of the MethylGPT revision.

Tables only: no figure, legend
or manuscript text; the figures are built on the Mac. Released weights (artifacts/release/methylgpt2-1.1, loaded with
package/src/methylgpt2) and the two named checkpoints; nothing is trained. Profile-level tables hold only public GEO
profiles (GSM sample in a GSE series; the script stops otherwise); no Generation Scotland file is read.
Design written on 10 Nov 2026 before any computation below was run. The export is descriptive: no representation or
set is chosen, and every row is written.

FIG. 3 (sample representations with every tissue)
  Sets, each profile with the CpGs it was deposited with (no masking, no subsampling by tissue):
    set C  every test-split GEO profile: 13,191 profiles, 113 datasets. The specified 14,185 profiles (116 datasets)
           include 994 test-split TCGA and TARGET profiles (TCGA-LIHC 408, TCGA-LUSC 391, TARGET-AML 195) that are not
           GEO profiles and carry no tissue, sex or platform label; they are left out so that every profile in the
           embeddings and metrics is in the exported table (deviation).
    set D  set C plus 30,000 training-split GEO profiles drawn uniformly at random without replacement
           (numpy default_rng(20261110).choice over the 136,028 training-split GEO rows in corpus order); split column.
  Representations (released bundle; inputs clipped to [0, 1] as the package does):
    raw        beta values of the 49,156 vocabulary CpGs, unmeasured CpGs filled with the training means of the input
               layer (core mu)
    latent     latent scores (posterior means of the input layer, 1,536; MethylGPT.factor_scores)
    embedding  2,048-dimensional embedding of released run 1 (weights.pt, after the age, sex, tissue and clock stage;
               MethylGPT.embed, stage 'final')
    encoder    dimensions 1,537 to 2,048 of that embedding (the transformer part, after the 1,536 scaled latent scores)
  For each set and representation separately: centre (no rescaling), exact PCA (float64 eigendecomposition of the
  covariance or Gram matrix on the GPU), first 50 principal components; UMAP of the 50 components (umap-learn,
  30 neighbours, min_dist 0.3, Euclidean, random_state 0; the parameters of plot_fig3_2026-10-02.py).
  Array of each profile: from the GEO platform of its series when the corpus table lists one platform (GPL8490 27K;
  GPL13534, GPL16304 450K; GPL21145, GPL23976 EPIC; GPL33022 EPIC v2; any other platform 'other'); when it lists none
  or several, from the observed vocabulary CpGs (artifacts/platform_vocab_masks_2026-10-01.npz): 27K if fewer than 10%
  of the 26,925 CpGs outside the 27K mask are observed; otherwise EPIC v2 if at least 50% of the 918 CpGs in the EPIC v2
  mask but not the EPIC v1 mask are observed and fewer than 30% of the 3,212 CpGs in the EPIC v1 mask but not the EPIC
  v2 mask; otherwise 450K if at least 50% of the 918 are observed; otherwise EPIC. (The 918 CpGs are 450K probes absent
  from EPIC v1 and restored on EPIC v2.) Column array_source records the rule used for each profile.
  Metrics, for each set and representation, on the 50 principal components (cosine), every row with n profiles,
  datasets and classes and a 95% interval from 2,000 draws resampling the datasets of the set with replacement
  (seed 20261110; cross-validated predictions and per-profile iLISI held fixed; balanced accuracy and silhouette
  recomputed exactly on each resampled set from per-dataset sums as in si_sample_representations_2026-10-18.py, the
  zero distances between copies of a profile excluded; median iLISI of the resampled profiles; percentile interval):
    tissue 10-NN balanced accuracy, stratified fivefold (StratifiedKFold, shuffled, seed 20261001) and with each dataset
      held out (neighbours from other datasets only), and tissue silhouette (cosine). Profiles of the tissues "with at
      least 30 profiles in at least 2 datasets", read in both ways and both reported: rule 1, at least 30 profiles in
      each of at least 2 datasets; rule 2, at least 30 profiles in total, from at least 2 datasets.
    whole blood, on the whole-blood profiles of the datasets with at least 30 whole-blood profiles in the set:
      array 10-NN balanced accuracy, stratified fivefold and with each dataset held out (arrays with at least 30 of
      these profiles from at least 2 datasets);
      dataset 10-NN balanced accuracy, stratified fivefold;
      dataset iLISI (harmonypy compute_lisi, perplexity 30), median over profiles.
    sex 10-NN balanced accuracy, stratified fivefold and with each dataset held out (every profile labelled F or M).
  10-NN: majority vote of the 10 most cosine-similar eligible profiles (stable order on similarity ties; vote ties go
  to the first class in sorted order, as scikit-learn).

FIG. 4a  Training-split profiles with a clean age (age_clean; 41,845): profiles per one-year age bin [k, k + 1), overall
  and for Whole Blood, Brain, Cells, Skin, Muscle and Other (every other tissue label and no label). Counts only.

FIG. 4b-d  The 5,811 test-split profiles with metadata ages (34 datasets); encoders ssl (runs/stopped_cf_dense_2026-09-30/
  best.pt) and age_stage (runs/aging_heads_clean_2026-10-01/best.pt), loaded and embedded as in
  si_embedding_pretraining_2026-10-18.py. Per encoder: the transformer part (dimensions 1,537 to 2,048), centred, first
  five principal components (exact PCA fitted per encoder) and UMAP of its first 50 components; the whole embedding,
  centred, UMAP of its first 50 components (UMAP parameters as Fig. 3). Second table: explained variance ratio of
  components 1 to 5 of the transformer part and of the whole embedding, and their Spearman correlation with age over
  all profiles and within whole blood. Sign of each component: its loading of largest magnitude is positive.

FIG. 4e  MedAE by dataset (datasets with at least 10 scored profiles; dataset = corpus series) and over all scored
  profiles, per method.
  AltumAge test set (4,626): stored predictions. MethylGPT, mean of three runs (age_ensemble_2026-10-04.predictions.npz);
    elastic net and AltumAge-style MLP, mean of three seeds (missing_cpgs_comparators_2026-10-18.predictions.npz);
    Horvath multi-tissue, Hannum, Horvath skin and blood, Lin and AltumAge in pyaging from ClockBase, and Zhang 2019 on
    the profiles with >= 90% of its CpGs observed, through clock_table() of evaluate_age_revision_2026-10-01.py with the
    masks of age_ensemble_2026-10-04.py.
  AltumAge validation set (1,366): no per-profile predictions are stored, so they are computed here without training:
    MethylGPT (released bundle, mean of three runs), elastic net (stored model), MLP (stored weights,
    runs/mlp_altumage_style_2026-10-18/weights.pt, mean of three seeds), clocks (the same lookup and masks; a clock with
    fewer than 50 scored validation profiles is not reported). The validation set chose the settings of the MethylGPT
    age head, the elastic net and the MLP (column validation_used_for_selection).

CHECKS (abort on failure; values in the README)
  1. every exported profile is a GEO profile (GSM in GSE); set C has 13,191 test-split profiles and 113 datasets, set D
     43,191 profiles; Fig. 4b-d 5,811 profiles and 34 datasets; Fig. 4a 41,845 profiles.
  2. set A rows of artifacts/eval/representations_setA_2026-10-01.npz (2,371 profiles, all in set C): raw values equal
     the cache (max abs diff <= 1e-6); latent scores max abs diff <= 1e-3; cosine similarity of each embedding with the
     cached multitask embedding >= 0.999.
  3. Fig. 4b-d embeddings against si_embedding_pretraining_embeddings_2026-10-18.npz: max abs diff <= 1e-2; whole-
     embedding PC1 and PC2 explained variance ratios and absolute Spearman correlations with age equal
     artifacts/figure_tables_2026-10-23/table4_fig4bcd_pca_variance.csv.gz within 1e-4.
  4. n_observed_vocab_cpgs equals the corpus n_obs column.
  5. array: the observed-CpG rule agrees with the single-platform assignment for >= 98% of the single-platform profiles
     of set D.
  6. 10-NN against scikit-learn (KNeighborsClassifier(10, metric='cosine'), same folds; set C, latent, sex, stratified
     fivefold): >= 99.9% of predictions agree; silhouette point estimates from the bootstrap code with unit weights
     equal sklearn silhouette_score within 1e-6; balanced accuracy point estimates equal balanced_accuracy_score.
  7. Fig. 4e: overall test-set MedAE equals age_ensemble_2026-10-04.json (MethylGPT, elastic net, clocks, with the same
     n) within 1e-9 and missing_cpgs_comparators_2026-10-18_table.csv (MLP, 4 decimals) within 5e-5; recomputed test
     predictions (bundle, elastic net, MLP) equal the stored ones within 1e-3 years; validation MedAE of the elastic net
     equals baseline_corpus_age_clean_rerun_2026-10-01.json within 1e-4 and that of each MLP seed (half-precision inputs,
     as in training) its best_valid_medae within 1e-3.

AMENDMENT 1 (10 Nov 2026; the first run stopped at the PCA of set D raw values, before any metric or table was computed):
  cuSOLVER's eigendecomposition refuses the 43,191 x 43,191 Gram matrix of set D raw values (CUSOLVER_STATUS_INVALID_VALUE
  at the workspace query). For a Gram or covariance matrix larger than 40,000 x 40,000 the first 50 eigenpairs come from
  block subspace iteration in float64 (block 200, random start seed 0, QR every step, Rayleigh-Ritz every 5 steps, until
  the relative residual |G v - l v| / l of each of the 50 pairs is below 1e-9). Check 8 (added): on set C raw values the
  iteration gives the components of the full eigendecomposition (max abs difference of the scores <= 1e-6 of the largest
  score). The bundle pass is cached in the work folder so a rerun does not repeat it.

STAGES  --stage extract (GPU: embeddings, PCA, Fig. 4b-d and 4e predictions), umap (CPU), metrics (GPU), write (tables,
  README), or all. Run with CUDA_VISIBLE_DEVICES set to one GPU.
OUTPUTS  artifacts/figure_tables_2026-11-10/ (tables, README.md, checks); work files in
  artifacts/figure_tables_2026-11-10_work/ (not exported).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, Finv, log, require, write_json  # noqa: E402

DATE = "2026-11-10"
OUT = ARTIFACTS / f"figure_tables_{DATE}"
WORK = ARTIFACTS / f"figure_tables_{DATE}_work"
EVAL = ARTIFACTS / "eval"
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
SSL = ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"
AGE_STAGE = ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"
MLP_W = ARTIFACTS / "runs/mlp_altumage_style_2026-10-18/weights.pt"
SEED_D = SEED_BOOT = 20261110
NBOOT = 2000
RANK = 1536
REPS = ["raw", "latent", "embedding", "encoder"]
SETS = ["C", "D"]
UMAP_KW = dict(n_neighbors=30, min_dist=0.3, random_state=0)
GEO_ID, GEO_DS = r"^GSM\d+$", r"^GSE\d+$"
GPL_ARRAY = {"GPL8490": "27K", "GPL13534": "450K", "GPL16304": "450K", "GPL21145": "EPIC", "GPL23976": "EPIC",
             "GPL33022": "EPIC v2"}
PIE = ["Whole Blood", "Brain", "Cells", "Skin", "Muscle"]
ENCODERS = ["ssl", "age_stage"]
CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    log(f"check {name}: {'pass' if ok else 'FAIL'} {detail}")


def save_checks(stage):
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / f"checks_{stage}.json", CHECKS)
    bad = [c for c in CHECKS if not c["passed"]]
    if bad:
        sys.exit(f"{stage}: {len(bad)} check(s) failed: {[c['check'] for c in bad]}")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def is_geo(ids, groups):
    return (pd.Series(np.asarray(ids, str)).str.match(GEO_ID).to_numpy()
            & pd.Series(np.asarray(groups, str)).str.match(GEO_DS).to_numpy())


def tables():
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    return prof, lab


def set_rows(prof):
    geo = is_geo(prof.id, prof.group)
    C = np.flatnonzero((prof.split == "test").to_numpy() & geo)
    tr = np.flatnonzero((prof.split == "train").to_numpy() & geo)
    add = np.random.default_rng(SEED_D).choice(tr, 30000, replace=False)
    return C, np.sort(np.concatenate([C, add])), len(tr)


# ----------------------------------------------------------------------------------------------------------------------
# GPU helpers
# ----------------------------------------------------------------------------------------------------------------------
DEV = None


def dev():
    global DEV
    if DEV is None:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        DEV = torch.device("cuda:0")
    return DEV


def top_eigh(G, k, block=200, tol=1e-9, max_iter=2000, seed=0):
    """First k eigenpairs of a symmetric PSD matrix by block subspace iteration (float64), Rayleigh-Ritz every 5 steps
    until every relative residual is below tol (amendment 1). Returns eigenvalues (descending), vectors, max residual."""
    import torch
    n = G.shape[0]
    gen = torch.Generator(device=G.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, block, generator=gen, dtype=torch.float64, device=G.device))
    res = np.inf
    for it in range(max_iter):
        Q, _ = torch.linalg.qr(G @ Q)
        if it % 5 == 4:
            w, Y = torch.linalg.eigh(Q.T @ (G @ Q))
            o = torch.argsort(w, descending=True)
            w, Vk = w[o][:k], Q @ Y[:, o][:, :k]
            res = float(((G @ Vk - Vk * w).norm(dim=0) / w.abs()).max())
            if res < tol:
                log(f"subspace iteration converged after {it + 1} steps (max relative residual {res:.2e})")
                return w, Vk, res
    sys.exit(f"subspace iteration did not converge (max relative residual {res:.2e})")


def pca(T, k=50, force_iter=False):
    """Exact PCA of the centred rows of T (torch, n x p, on the GPU). Returns scores (n x k, float64), the variance of
    each component, the total variance and the loadings (p x k). Sign: the loading of largest magnitude is positive.
    Matrices larger than 40,000 x 40,000 (or force_iter) use top_eigh (amendment 1)."""
    import torch
    n, p = T.shape
    d = T.device
    mu = torch.zeros(p, dtype=torch.float64, device=d)
    for i in range(0, n, 4096):
        mu += T[i:i + 4096].double().sum(0)
    mu /= n
    k = min(k, p, n - 1)
    if p <= n:
        Cm = torch.zeros(p, p, dtype=torch.float64, device=d)
        for i in range(0, n, 4096):
            c = T[i:i + 4096].double() - mu
            Cm.addmm_(c.T, c)
        total = float(torch.trace(Cm))
        if p > 40000 or force_iter:
            L, V, _ = top_eigh(Cm, k)
        else:
            L, V = torch.linalg.eigh(Cm)
            o = torch.argsort(L, descending=True)[:k]
            L, V = L[o], V[:, o]
        del Cm
        S = torch.cat([(T[i:i + 4096].double() - mu) @ V for i in range(0, n, 4096)])
    else:
        G = torch.zeros(n, n, dtype=torch.float64, device=d)
        for j in range(0, p, 2048):
            c = T[:, j:j + 2048].double() - mu[j:j + 2048]
            G.addmm_(c, c.T)
            del c
        total = float(torch.trace(G))
        if n > 40000 or force_iter:
            L, U, _ = top_eigh(G, k)
        else:
            L, U = torch.linalg.eigh(G)
            o = torch.argsort(L, descending=True)[:k]
            L, U = L[o], U[:, o].clone()
        del G
        torch.cuda.empty_cache()
        sq = torch.sqrt(L.clamp_min(1e-300))
        S = U * sq
        V = torch.empty(p, k, dtype=torch.float64, device=d)
        for j in range(0, p, 2048):
            V[j:j + 2048] = ((T[:, j:j + 2048].double() - mu[j:j + 2048]).T @ U) / sq
        del U
    sign = torch.sign(V[torch.argmax(V.abs(), 0), torch.arange(k, device=d)])
    sign[sign == 0] = 1
    S, V = S * sign, V * sign
    return S.cpu().numpy(), (L / (n - 1)).cpu().numpy(), total / (n - 1), V.cpu().numpy()


# ----------------------------------------------------------------------------------------------------------------------
# stage extract
# ----------------------------------------------------------------------------------------------------------------------
def assign_array(gpl, O):
    pm = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    not27 = ~pm["hm27"]
    v2only = pm["epic_v2"] & ~pm["epic_v1"]
    v1only = pm["epic_v1"] & ~pm["epic_v2"]
    assert (int(not27.sum()), int(v2only.sum()), int(v1only.sum())) == (26925, 918, 3212)
    f27, f2, f1 = O[:, not27].mean(1), O[:, v2only].mean(1), O[:, v1only].mean(1)
    pat = np.where(f27 < 0.10, "27K", np.where((f2 >= 0.5) & (f1 < 0.3), "EPIC v2", np.where(f2 >= 0.5, "450K", "EPIC")))
    single = np.array([isinstance(g, str) and len(g.split()) == 1 for g in gpl])
    from_gpl = np.array([GPL_ARRAY.get(g, "other") if s else "" for g, s in zip(gpl, single)], dtype=object)
    arr = np.where(single, from_gpl, pat).astype(str)
    src = np.where(single, "series platform", "observed CpGs")
    known = single & np.isin(from_gpl.astype(str), list(GPL_ARRAY.values()))
    agree = float((pat[known] == from_gpl[known].astype(str)).mean())
    conf = pd.crosstab(pd.Series(from_gpl[known].astype(str), name="platform"), pd.Series(pat[known], name="observed_rule"))
    return arr, src, agree, conf, {"f_outside_27k": f27, "f_v2_not_v1": f2, "f_v1_not_v2": f1}


def stage_extract():
    import torch
    from methylgpt2_lib import Corpus, cuda_preinit
    cuda_preinit()
    d = dev()
    t0 = time.time()
    prof, lab = tables()
    C, D, n_train_geo = set_rows(prof)
    chk("1 set C size", len(C) == 13191 and prof.group.iloc[C].nunique() == 113, f"{len(C)} profiles, {prof.group.iloc[C].nunique()} datasets")
    chk("1 set D size", len(D) == 43191 and np.isin(C, D).all() and (prof.split.iloc[D] == "train").sum() == 30000,
        f"{len(D)} profiles, {prof.group.iloc[D].nunique()} datasets; training-split GEO pool {n_train_geo}")
    chk("1 GEO only (sets C, D)", is_geo(prof.id.iloc[D], prof.group.iloc[D]).all())
    corpus = Corpus()
    X = corpus.get(D)                        # rows sorted
    np.clip(X, 0, 1, out=X)
    O = np.isfinite(X)
    nobs = O.sum(1)
    chk("4 n_observed equals corpus n_obs", (nobs == prof.n_obs.to_numpy()[D]).all())
    arr, src, agree, conf, frac = assign_array(prof.gpl.astype(object).to_numpy()[D], O)
    chk("5 array rule agreement (set D single-platform profiles)", agree >= 0.98, f"{agree:.5f}; {conf.to_dict()}")
    del O

    # released bundle: latent scores and run-1 embedding
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device=str(d))
    assert model.vocab == list(np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str))
    cache = WORK / "bundle_pass_D.npz"
    if cache.exists() and np.array_equal(np.load(cache)["rows"], D):
        zb = np.load(cache)
        lat, emb = zb["lat"], zb["emb"]
        log("bundle pass loaded from the work cache")
    else:
        lat = np.empty((len(D), RANK), np.float32)
        emb = np.empty((len(D), RANK + 512), np.float32)
        t1 = time.time()
        with torch.no_grad():
            for i in range(0, len(D), 32):
                x = torch.from_numpy(X[i:i + 32]).to(d)
                _, s, e = model._encode(x)
                lat[i:i + 32] = s.float().cpu().numpy()
                emb[i:i + 32] = e.float().cpu().numpy()
                if i % 6400 == 0:
                    log(f"bundle {i}/{len(D)} ({time.time() - t1:.0f} s)")
        log(f"bundle pass done in {time.time() - t1:.0f} s")
        np.savez(cache, rows=D, lat=lat, emb=emb)
    mu = model.core.mu.detach().clone()

    # raw: training-mean filled, on the GPU
    T = torch.empty((len(D), X.shape[1]), dtype=torch.float32, device=d)
    for i in range(0, len(D), 4096):
        t = torch.from_numpy(X[i:i + 4096]).to(d)
        T[i:i + 4096] = torch.where(torch.isfinite(t), t, mu)
    del X

    # check 2: set A rows against the 2026-10-01 cache
    za = np.load(require(EVAL / "representations_setA_2026-10-01.npz"))
    pos = np.searchsorted(D, za["rows"])
    inC = np.isin(za["rows"], C)
    chk("2 set A rows in set C", inC.all() and (D[pos] == za["rows"]).all(), f"{inC.sum()} of {len(inC)}")
    draw = float((T[torch.as_tensor(pos, device=d)].cpu().numpy() - za["raw_full"]).__abs__().max())
    dlat = float(np.abs(lat[pos] - za["rep_factor_scores"]).max())
    a, b = emb[pos].astype(np.float64), za["rep_mgpt2_multitask"].astype(np.float64)
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    chk("2 raw equals set A cache", draw <= 1e-6, f"max abs diff {draw:.3g}")
    chk("2 latent equals set A cache", dlat <= 1e-3, f"max abs diff {dlat:.3g}")
    chk("2 embedding matches set A multitask cache", cos.min() >= 0.999,
        f"cosine min {cos.min():.6f}, median {np.median(cos):.6f}; max abs diff {np.abs(a - b).max():.3g}")

    # PCA per set and representation
    out = {"rows_C": C, "rows_D": D, "array": arr, "array_source": src, "n_observed": nobs,
           **{k: v.astype(np.float32) for k, v in frac.items()}}
    pcinfo = {}
    posC = np.searchsorted(D, C)
    for s_ in SETS:
        idx = None if s_ == "D" else torch.as_tensor(posC, device=d)
        for rep in REPS:
            t2 = time.time()
            if rep == "raw":
                M = T if idx is None else T[idx]
            elif rep == "latent":
                M = torch.from_numpy(lat if idx is None else lat[posC]).to(d)
            elif rep == "embedding":
                M = torch.from_numpy(emb if idx is None else emb[posC]).to(d)
            else:
                M = torch.from_numpy(np.ascontiguousarray((emb if idx is None else emb[posC])[:, RANK:])).to(d)
            S, var, tot, _ = pca(M)
            if s_ == "C" and rep == "raw":                # check 8 (amendment 1)
                S2, var2, _, _ = pca(M, force_iter=True)
                dS = float(np.abs(S2 - S).max() / np.abs(S).max())
                chk("8 subspace iteration equals full eigendecomposition (set C raw)", dS <= 1e-6,
                    f"max abs score difference / max abs score {dS:.2e}; max relative variance difference {float(np.abs(var2 / var - 1).max()):.2e}")
                del S2
            del M
            torch.cuda.empty_cache()
            out[f"pcs_{s_}_{rep}"] = S
            pcinfo[f"{s_}|{rep}"] = {"n": int(S.shape[0]), "explained_variance_ratio_pc1_5": (var[:5] / tot).tolist(),
                                     "explained_variance_ratio_50": float(var.sum() / tot), "seconds": round(time.time() - t2, 1)}
            log(f"PCA set {s_} {rep}: {S.shape}, 50 PCs explain {var.sum() / tot:.3f} ({time.time() - t2:.0f} s)")
    del T, lat, emb, model
    torch.cuda.empty_cache()

    # Fig. 4b-d: ssl and age-stage encoders on the 5,811 held-out profiles with metadata ages
    from methylgpt2_model import embed_batch, load_checkpoint
    ho = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    chk("1 Fig. 4b-d profiles", len(ho) == 5811 and prof.group.iloc[ho].nunique() == 34 and is_geo(prof.id.iloc[ho], prof.group.iloc[ho]).all(),
        f"{len(ho)} profiles, {prof.group.iloc[ho].nunique()} datasets")
    Xh = corpus.get(ho)
    E = {}
    for enc in ENCODERS:
        if enc == "ssl":
            m, core, _ = load_checkpoint(SSL, None, d)
        else:
            ck = torch.load(AGE_STAGE, map_location=d, weights_only=False)
            cfg = ck["config"]
            m, core, _ = load_checkpoint(HERE / ck["ckpt"], cfg.get("core"), d, random_init=True)
            m.load_state_dict(ck["model"])
            m.eval()
            del ck
        e = np.empty((len(ho), RANK + 512), np.float32)
        with torch.no_grad():
            for i in range(0, len(ho), 32):
                x = torch.from_numpy(Xh[i:i + 32]).to(d)
                e[i:i + 32] = embed_batch(m, core, x, torch.isfinite(x)).float().cpu().numpy()
        E[enc] = e
        del m, core
        torch.cuda.empty_cache()
    zc = np.load(require(EVAL / "si_embedding_pretraining_embeddings_2026-10-18.npz"))
    pc_ = np.searchsorted(zc["rows"], ho)
    assert (zc["rows"][pc_] == ho).all()
    for enc in ENCODERS:
        dd = float(np.abs(E[enc] - zc[f"emb_{enc}"][pc_]).max())
        chk(f"3 {enc} embedding equals 2026-10-18 cache", dd <= 1e-2, f"max abs diff {dd:.3g}")
    old = pd.read_csv(require(ARTIFACTS / "figure_tables_2026-10-23/table4_fig4bcd_pca_variance.csv.gz"))
    from scipy.stats import spearmanr
    age = lab.age_meta.to_numpy(float)[ho]
    wb = (prof.tissue.to_numpy()[ho] == "Whole Blood")
    out["rows_ho"] = ho
    var_rows = []
    for enc in ENCODERS:
        for part, M in (("transformer", E[enc][:, RANK:]), ("whole", E[enc])):
            S, var, tot, _ = pca(torch.from_numpy(np.ascontiguousarray(M)).to(d))
            out[f"ho_{enc}_{part}"] = S
            for j in range(5):
                var_rows.append({"encoder": enc, "part": part, "pc": j + 1, "explained_variance_ratio": float(var[j] / tot),
                                 "spearman_age_all": float(spearmanr(S[:, j], age).statistic), "n_all": int(len(age)),
                                 "spearman_age_whole_blood": float(spearmanr(S[wb, j], age[wb]).statistic),
                                 "n_whole_blood": int(wb.sum())})
            if part == "whole":
                for j in (1, 2):
                    o = old[(old.encoder == ("selfsupervised" if enc == "ssl" else enc)) & (old.pc == j)].iloc[0]
                    r = [v for v in var_rows if v["encoder"] == enc and v["part"] == "whole" and v["pc"] == j][0]
                    ok = (abs(r["explained_variance_ratio"] - o.explained_variance_ratio) <= 1e-4
                          and abs(abs(r["spearman_age_all"]) - abs(o.spearman_with_age)) <= 1e-4)
                    chk(f"3 {enc} whole-embedding PC{j} equals table4", ok,
                        f"EVR {r['explained_variance_ratio']:.6f} vs {o.explained_variance_ratio:.6f}; "
                        f"rho {r['spearman_age_all']:.6f} vs {o.spearman_with_age:.6f}")
    pd.DataFrame(var_rows).to_csv(WORK / "fig4bcd_variance.csv", index=False)
    del E, Xh

    # Fig. 4e: validation predictions (and test reproduction checks)
    fig4e_predict(model_cls=MethylGPT, d=d, out=out)

    np.savez(WORK / "extract.npz", **out)
    write_json(WORK / "extract_info.json", {"pca": pcinfo, "seconds": time.time() - t0,
                                            "array_confusion_single_platform_D": conf.to_dict()})
    save_checks("extract")
    log(f"extract done in {time.time() - t0:.0f} s")


def fig4e_predict(model_cls, d, out):
    import torch
    from methylgpt2_lib import load_altumage_split
    mc = _load("missing_cpgs_comparators", "missing_cpgs_comparators_2026-10-18.py")
    zm = np.load(require(EVAL / "missing_cpgs_comparators_2026-10-18.predictions.npz"), allow_pickle=True)
    ze = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    ids_t, Xt, yt = load_altumage_split("test")
    ids_v, Xv, yv = load_altumage_split("valid")
    assert (ze["altumage_ids"].astype(str) == ids_t).all() and (zm["altumage_ids"].astype(str) == ids_t).all()
    model = model_cls.from_pretrained(str(BUNDLE), device=str(d))
    pt = model.predict_age_runs(pd.DataFrame(Xt, columns=model.vocab), batch_size=32).mean(1).to_numpy(float)
    pv = model.predict_age_runs(pd.DataFrame(Xv, columns=model.vocab), batch_size=32).mean(1).to_numpy(float)
    dm = float(np.abs(pt - ze["ensemble|all"]).max())
    chk("7 bundle test predictions equal stored ensemble", dm <= 1e-3, f"max abs diff {dm:.3g} years")
    del model
    torch.cuda.empty_cache()
    zen = np.load(require(EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))

    def enet(X):
        T = np.where(np.isfinite(X), X, zen["mu"])
        return Finv(((T - zen["mu"]) / zen["sd"]) @ zen["w"] * float(zen["ys"]) + float(zen["ym"]))
    et, ev = enet(Xt), enet(Xv)
    de = float(np.abs(et - zm["Elastic net|all"]).max())
    chk("7 elastic-net test predictions equal stored", de <= 1e-3, f"max abs diff {de:.3g} years")
    ej = json.loads(require(EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.json").read_text())["models"]["enet_raw"]["valid"]
    mv = float(np.median(np.abs(ev - yv)))
    chk("7 elastic-net validation MedAE equals stored", abs(mv - ej["medae"]) <= 1e-4 and ej["n"] == len(yv), f"{mv:.6f} vs {ej['medae']:.6f}")
    W = torch.load(MLP_W, map_location=d, weights_only=False)["altumage_train"]
    std = mc.Standardiser(W["mu"], W["sd"], W["used"], d)
    tm, ts = float(W["target_center"]), float(W["target_scale"])
    tj = json.loads(require(EVAL / "missing_cpgs_comparators_2026-10-18_training.json").read_text())["mlp_altumage_train"]["seeds"]
    best = {int(s["seed"]): float(s["best_valid_medae"]) for s in tj}
    mt, mvv = [], []
    for k in (1, 2, 3):
        net = mc.make_mlp(Xt.shape[1]).to(d)
        net.load_state_dict(W["seeds"][k])
        net.eval()
        a = Finv(mc.predict_mlp(net, std(Xt)) * ts + tm)
        dk = float(np.abs(a - zm[f"MLP seed {k}|all"]).max())
        chk(f"7 MLP seed {k} test predictions equal stored", dk <= 1e-3, f"max abs diff {dk:.3g} years")
        mt.append(a)
        mvv.append(Finv(mc.predict_mlp(net, std(Xv)) * ts + tm))
        vh = float(np.median(np.abs(Finv(mc.predict_mlp(net, std(Xv).half()) * ts + tm) - yv)))
        chk(f"7 MLP seed {k} validation MedAE equals best_valid_medae", abs(vh - best[k]) <= 1e-3, f"{vh:.6f} vs {best[k]:.6f}")
    out.update({"e_test_ids": ids_t, "e_test_age": yt, "e_valid_ids": ids_v, "e_valid_age": yv,
                "e_valid|MethylGPT": pv, "e_valid|Elastic net": ev, "e_valid|MLP": np.mean(mvv, 0),
                **{f"e_valid|MLP seed {k}": v for k, v in zip((1, 2, 3), mvv)}})


# ----------------------------------------------------------------------------------------------------------------------
# stage umap
# ----------------------------------------------------------------------------------------------------------------------
def _umap(P):
    import umap
    return umap.UMAP(**UMAP_KW).fit_transform(np.asarray(P, np.float32)).astype(np.float32)


def stage_umap(workers=8):
    from concurrent.futures import ProcessPoolExecutor
    z = np.load(require(WORK / "extract.npz"), allow_pickle=True)
    jobs = {f"umap_{s}_{r}": f"pcs_{s}_{r}" for s in SETS for r in REPS}
    jobs.update({f"umap_ho_{e}_{p}": f"ho_{e}_{p}" for e in ENCODERS for p in ("transformer", "whole")})
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        fut = {k: ex.submit(_umap, z[v][:, :50]) for k, v in jobs.items()}
        res = {k: f.result() for k, f in fut.items()}
    np.savez(WORK / "umap.npz", **res)
    log(f"{len(res)} UMAPs in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------
# stage metrics
# ----------------------------------------------------------------------------------------------------------------------
def knn_pred(P, y, excl, k=10):
    """10-NN class prediction by cosine similarity; profile i may use profile j only if excl[i] != excl[j]."""
    import torch
    d = dev()
    T = torch.as_tensor(P, dtype=torch.float64, device=d)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-300)
    cl, yi = np.unique(np.asarray(y, str), return_inverse=True)
    yt = torch.as_tensor(yi, device=d)
    ex = torch.as_tensor(np.asarray(excl), device=d)
    tie = torch.arange(len(cl), device=d, dtype=torch.float64) * 1e-3
    out = np.empty(len(yi), np.int64)
    for i in range(0, len(yi), 1024):
        S = T[i:i + 1024] @ T.T
        S.masked_fill_(ex[i:i + 1024, None] == ex[None, :], -np.inf)
        nb = torch.sort(S, dim=1, descending=True, stable=True).indices[:, :k]
        cnt = torch.zeros((len(nb), len(cl)), dtype=torch.float64, device=d).scatter_add_(1, yt[nb], torch.ones(nb.shape, dtype=torch.float64, device=d))
        out[i:i + 1024] = (cnt - tie).argmax(1).cpu().numpy()
    return cl[out]


def folds(y, scheme, codes):
    from sklearn.model_selection import StratifiedKFold
    if scheme == "lodo":
        return np.asarray(codes)
    f = np.empty(len(y), np.int64)
    for j, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=20261001).split(np.zeros(len(y)), y)):
        f[te] = j
    return f


def boot_bacc(y, pred, codes, W):
    import torch
    from sklearn.metrics import balanced_accuracy_score
    y, pred = np.asarray(y, str), np.asarray(pred, str)
    cl = np.unique(np.concatenate([y, pred]))
    yi, pi = np.searchsorted(cl, y), np.searchsorted(cl, pred)
    units, uc = np.unique(codes, return_inverse=True)
    C = len(cl)
    M = np.zeros((len(units), C, C))
    np.add.at(M, (uc, yi, pi), 1)
    point = float(balanced_accuracy_score(y, pred))

    def bacc(Cm):
        rows = Cm.sum(-1)
        diag = torch.diagonal(Cm, dim1=-2, dim2=-1)
        per = torch.where(rows > 0, diag / rows.clamp_min(1e-300), torch.full_like(rows, float("nan")))
        return torch.nanmean(per, -1)
    d = dev()
    Mt = torch.as_tensor(M.reshape(len(units), -1), device=d)
    chk_point = float(bacc(Mt.sum(0).reshape(C, C)))
    Wt = torch.as_tensor(W[:, units], dtype=torch.float64, device=d)
    draws = torch.cat([bacc((Wt[i:i + 200] @ Mt).reshape(-1, C, C)) for i in range(0, len(Wt), 200)]).cpu().numpy()
    return point, draws, abs(point - chk_point)


def boot_silhouette(P, labels, codes, W, chunk=50):
    """Cosine silhouette on P, recomputed exactly on each resampled set (datasets resampled; copies of a profile excluded
    from its own-class partners)."""
    import torch
    d = dev()
    _, li = np.unique(np.asarray(labels, str), return_inverse=True)
    C, m = li.max() + 1, len(li)
    keys, blk = np.unique(np.asarray(codes) * C + li, return_inverse=True)
    K = len(keys)
    bunit, bcls = keys // C, keys % C
    T = torch.as_tensor(P, dtype=torch.float64, device=d)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-300)
    B1 = torch.zeros(m, K, dtype=torch.float64, device=d)
    B1[torch.arange(m, device=d), torch.as_tensor(blk, device=d)] = 1
    Ds = torch.empty(m, K, dtype=torch.float64, device=d)
    for i in range(0, m, 2048):
        Dc = (1 - T[i:i + 2048] @ T.T).clamp_(0, 2)
        r = torch.arange(Dc.shape[0], device=d)
        Dc[r, r + i] = 0
        Ds[i:i + 2048] = Dc @ B1
        del Dc
    nblk = B1.sum(0)
    del B1
    A = torch.zeros(K, C, dtype=torch.float64, device=d)
    A[torch.arange(K, device=d), torch.as_tensor(bcls, device=d)] = 1
    li_t = torch.as_tensor(li, device=d)
    cod_t = torch.as_tensor(np.asarray(codes), device=d)
    bu_t = torch.as_tensor(bunit, device=d)

    def sil(Wb):
        wk = Wb[:, bu_t]                                         # (b, K)
        S = torch.matmul(Ds, wk[:, :, None] * A[None])           # (b, m, C)
        Nc = (wk * nblk) @ A                                     # (b, C)
        own = li_t[None, :, None].expand(len(Wb), m, 1)
        wi = Wb[:, cod_t]                                        # (b, m) copies of each profile
        a_n = Nc.gather(1, li_t[None].expand(len(Wb), m)) - wi
        a = S.gather(2, own)[..., 0] / torch.where(a_n > 0, a_n, torch.ones_like(a_n))
        means = S / torch.where(Nc > 0, Nc, torch.ones_like(Nc))[:, None, :]
        means = torch.where((Nc > 0)[:, None, :].expand_as(means), means, torch.full_like(means, float("inf")))
        means.scatter_(2, own, float("inf"))
        b = means.min(2).values
        s = (b - a) / torch.maximum(a, b)
        s = torch.where((a_n > 0) & torch.isfinite(s), s, torch.zeros_like(s))
        return (s * wi).sum(1) / wi.sum(1)
    G = W.shape[1]
    point = float(sil(torch.ones((1, G), dtype=torch.float64, device=d))[0])
    Wt = torch.as_tensor(W, dtype=torch.float64, device=d)
    draws = torch.cat([sil(Wt[i:i + chunk]) for i in range(0, len(Wt), chunk)]).cpu().numpy()
    return point, draws


def stage_metrics():
    import torch
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    sys.path.insert(0, "/Entropy/kying/tmp/pylib")
    from harmonypy import compute_lisi
    sr = _load("si_sample_representations", "si_sample_representations_2026-10-18.py")
    t0 = time.time()
    prof, lab = tables()
    z = np.load(require(WORK / "extract.npz"), allow_pickle=True)
    D = z["rows_D"]
    meta_all = pd.DataFrame({"row": D, "dataset": prof.group.astype(str).to_numpy()[D], "tissue": prof.tissue.to_numpy()[D],
                             "sex": prof.sex.to_numpy()[D], "array": z["array"]})
    rows_out = []
    for s_ in SETS:
        rows = z[f"rows_{s_}"]
        meta = meta_all.iloc[np.searchsorted(D, rows)].reset_index(drop=True)
        ds = meta.dataset.to_numpy()
        _, codes = np.unique(ds, return_inverse=True)
        _, W = sr.boot_weights(ds, n=NBOOT, seed=SEED_BOOT)
        G = W.shape[1]
        # profile subsets
        tis = meta.tissue
        ct = meta[tis.notna()].groupby(["tissue", "dataset"]).size()
        r1 = sorted(((ct >= 30).groupby("tissue").sum() >= 2).pipe(lambda s: s[s].index))
        tot, nd = meta[tis.notna()].groupby("tissue").size(), meta[tis.notna()].groupby("tissue").dataset.nunique()
        r2 = sorted(tot[(tot >= 30) & (nd >= 2)].index)
        wbn = meta[tis == "Whole Blood"].dataset.value_counts()
        wbm = ((tis == "Whole Blood") & meta.dataset.isin(wbn[wbn >= 30].index)).to_numpy()
        aw = meta[wbm].groupby("array").agg(n=("row", "size"), nd=("dataset", "nunique"))
        arrays = sorted(aw[(aw.n >= 30) & (aw.nd >= 2)].index)
        subsets = {"tissue rule 1": (tis.isin(r1).to_numpy(), "tissue"),
                   "tissue rule 2": (tis.isin(r2).to_numpy(), "tissue"),
                   "whole blood, arrays": (wbm & meta.array.isin(arrays).to_numpy(), "array"),
                   "whole blood, datasets": (wbm, "dataset"),
                   "sex labelled": (meta.sex.isin(["F", "M"]).to_numpy(), "sex")}
        np.save(WORK / f"subset_flags_{s_}.npy", np.stack([v[0] for v in subsets.values()], 1))
        log(f"set {s_}: rule 1 {len(r1)} tissues, rule 2 {len(r2)}, whole blood {wbm.sum()} ({len(wbn[wbn >= 30])} datasets), arrays {arrays}")

        def add(rep, metric, cv, sub, point, draws, m, note=""):
            y = meta.loc[m, subsets[sub][1]].astype(str)
            lo, hi = sr.ci(draws)
            rows_out.append({"set": s_, "representation": rep, "metric": metric, "cv": cv, "profiles": sub, "value": point,
                             "ci95_low": lo, "ci95_high": hi, "n_profiles": int(m.sum()), "n_datasets": int(meta.dataset[m].nunique()),
                             "n_classes": int(y.nunique()), "classes": ";".join(sorted(y.unique())) if subsets[sub][1] != "dataset" else "",
                             "note": note})
            log(f"{s_} {rep} {metric} {cv} [{sub}]: {point:.4f} [{lo:.4f}, {hi:.4f}]")

        for rep in REPS:
            P = z[f"pcs_{s_}_{rep}"][:, :50]
            for sub, (m, col) in subsets.items():
                y = meta.loc[m, col].astype(str).to_numpy()
                if len(set(y)) < 2:
                    rows_out.append({"set": s_, "representation": rep, "metric": f"{col} 10-NN balanced accuracy", "cv": "not computed",
                                     "profiles": sub, "value": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n_profiles": int(m.sum()),
                                     "n_datasets": int(meta.dataset[m].nunique()), "n_classes": len(set(y)), "classes": ";".join(sorted(set(y))),
                                     "note": "fewer than 2 classes meet the rule"})
                    continue
                cvs = ["stratified5"] if sub == "whole blood, datasets" else ["stratified5", "lodo"]
                for cv in cvs:
                    pr = knn_pred(P[m], y, folds(y, cv, codes[m]))
                    pt, dr, dev_ = boot_bacc(y, pr, codes[m], W)
                    chk(f"6 balanced accuracy point estimate ({s_} {rep} {sub} {cv})", dev_ < 1e-9, f"{dev_:.2e}")
                    add(rep, f"{col} 10-NN balanced accuracy", cv, sub, pt, dr, m)
                    if s_ == "C" and rep == "latent" and sub == "sex labelled" and cv == "stratified5":
                        f = folds(y, cv, codes[m])
                        sk = cross_val_predict(KNeighborsClassifier(10, metric="cosine"), P[m], y,
                                               cv=[(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(5)])
                        ag = float((sk == pr).mean())
                        chk("6 10-NN agrees with scikit-learn (set C, latent, sex, stratified5)", ag >= 0.999, f"{ag:.5f}")
                if col == "tissue":
                    pt, dr = boot_silhouette(P[m], y, codes[m], W)
                    sk = float(silhouette_score(P[m], y, metric="cosine"))
                    chk(f"6 silhouette point estimate ({s_} {rep} {sub})", abs(pt - sk) <= 1e-6, f"{pt:.8f} vs sklearn {sk:.8f}")
                    add(rep, "tissue silhouette (cosine)", "none", sub, pt, dr, m)
                if sub == "whole blood, datasets":
                    li = compute_lisi(P[m], pd.DataFrame({"dataset": y}), ["dataset"], perplexity=30)[:, 0]
                    pt, dr = sr.boot_median(li, codes[m], W)
                    add(rep, "dataset iLISI (median)", "none", sub, pt, dr, m, note=f"iLISI ranges from 1 to {len(set(y))} (datasets)")
            torch.cuda.empty_cache()
        log(f"set {s_} done ({time.time() - t0:.0f} s)")
    pd.DataFrame(rows_out).to_csv(WORK / "fig3_metrics.csv", index=False)
    save_checks("metrics")


# ----------------------------------------------------------------------------------------------------------------------
# stage write
# ----------------------------------------------------------------------------------------------------------------------
def fig4a_counts(prof, lab):
    a = lab.age_clean.to_numpy(float)
    m = (prof.split == "train").to_numpy() & np.isfinite(a)
    chk("1 Fig. 4a profiles", m.sum() == 41845, f"{m.sum()}")
    t = prof.tissue.to_numpy()[m]
    grp = np.where(np.isin(t, PIE), t, "Other")
    b = np.floor(a[m]).astype(int)
    lo, hi = b.min(), b.max()
    rows = []
    for g in ["all"] + PIE + ["Other"]:
        sel = np.ones(len(b), bool) if g == "all" else grp == g
        cnt = np.bincount(b[sel] - lo, minlength=hi - lo + 1)
        rows += [{"group": g, "age_bin_start": int(lo + i), "age_bin_end": int(lo + i + 1), "n_profiles": int(c)} for i, c in enumerate(cnt)]
    df = pd.DataFrame(rows)
    s = df.groupby("group").n_profiles.sum()
    chk("1 Fig. 4a groups sum to all", s["all"] == 41845 and s.drop("all").sum() == 41845, s.to_dict())
    return df


def fig4e_table(prof, z):
    ar = _load("age_revision", "evaluate_age_revision_2026-10-01.py")
    ens = json.loads(require(EVAL / "age_ensemble_2026-10-04.json").read_text())
    zm = np.load(require(EVAL / "missing_cpgs_comparators_2026-10-18.predictions.npz"), allow_pickle=True)
    ze = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    tb = pd.read_csv(require(EVAL / "missing_cpgs_comparators_2026-10-18_table.csv"))
    mlp_ref = float(tb[(tb.part == "overall") & (tb.scenario == "all") & (tb.model_or_contrast == "MLP")].estimate.iloc[0])
    pid = prof.set_index("id")
    CL = {"Horvath multi-tissue": "Horvath 2013", "Hannum": "Hannum", "Horvath skin and blood": "Horvath skin & blood",
          "Lin": "Lin", "Zhang 2019": "Zhang 2019", "AltumAge (pyaging)": "AltumAge"}
    rows = []
    for split in ("AltumAge test", "AltumAge validation"):
        key = "test" if split == "AltumAge test" else "valid"
        ids, y = z[f"e_{key}_ids"].astype(str), z[f"e_{key}_age"]
        grp = pid.reindex(ids).group.astype(str).to_numpy()
        if key == "test":
            P = {"MethylGPT": ze["ensemble|all"], "Elastic net": zm["Elastic net|all"], "MLP (AltumAge-style)": zm["MLP|all"]}
        else:
            P = {"MethylGPT": z["e_valid|MethylGPT"], "Elastic net": z["e_valid|Elastic net"], "MLP (AltumAge-style)": z["e_valid|MLP"]}
        masks = {k: np.ones(len(y), bool) for k in P}
        ct = ar.clock_table(list(ids))
        zf = ct.zhang_observed_fraction.to_numpy(float)
        for name, col in CL.items():
            if col not in ct.columns:
                log(f"{split}: {name} has no predictions; not reported")
                continue
            v = ct[col].to_numpy(float)
            ok = np.isfinite(v) & ((zf >= 0.9) if col == "Zhang 2019" else True)
            if ok.sum() >= 50:
                P[name], masks[name] = v, ok
            else:
                log(f"{split}: {name} scored on {ok.sum()} profiles; not reported")
        for name, p in P.items():
            m = masks[name]
            e = np.abs(p[m] - y[m])
            sel = name in ("MethylGPT", "Elastic net", "MLP (AltumAge-style)")
            base = {"set": split, "method": name, "validation_used_for_selection": bool(sel and key == "valid")}
            rows.append({**base, "dataset": "all", "n_scored": int(m.sum()), "medae": float(np.median(e)),
                         "n_datasets": int(pd.Series(grp[m]).nunique())})
            per = pd.DataFrame({"g": grp[m], "e": e}).groupby("g").e.agg(["size", "median"])
            for g, r in per[per["size"] >= 10].iterrows():
                rows.append({**base, "dataset": g, "n_scored": int(r["size"]), "medae": float(r["median"]), "n_datasets": 1})
            if key == "test":
                ov = float(np.median(e))
                if name == "MethylGPT":
                    ref, nref = ens["a_altumage_test"]["ensemble"]["medae"], 4626
                elif name == "Elastic net":
                    ref, nref = ens["a_altumage_test"]["elastic net"]["medae"], 4626
                elif name == "MLP (AltumAge-style)":
                    ref, nref = mlp_ref, 4626
                else:
                    k_ = "Zhang 2019 (>= 90% of its CpGs observed)" if CL[name] == "Zhang 2019" else CL[name]
                    ref, nref = ens["e_clocks_altumage_test"][k_]["clock"]["medae"], ens["e_clocks_altumage_test"][k_]["n"]
                tol = 5e-5 if name == "MLP (AltumAge-style)" else 1e-9
                chk(f"7 test MedAE equals stored ({name})", abs(ov - ref) <= tol and int(m.sum()) == nref, f"{ov:.6f} vs {ref:.6f}; n {m.sum()} vs {nref}")
    df = pd.DataFrame(rows)
    chk("1 Fig. 4e dataset names are GEO series", df[df.dataset != "all"].dataset.str.match(GEO_DS).all(),
        sorted(set(df[(df.dataset != "all") & ~df.dataset.str.match(GEO_DS)].dataset)))
    return df


def stage_write():
    prof, lab = tables()
    z = np.load(require(WORK / "extract.npz"), allow_pickle=True)
    u = np.load(require(WORK / "umap.npz"))
    OUT.mkdir(parents=True, exist_ok=True)
    D = z["rows_D"]
    files = {}
    # Fig. 3 profile tables
    for s_ in SETS:
        rows = z[f"rows_{s_}"]
        p = np.searchsorted(D, rows)
        flags = np.load(WORK / f"subset_flags_{s_}.npy")
        t = pd.DataFrame({"corpus_row": rows, "profile_id": prof.id.to_numpy()[rows], "dataset": prof.group.astype(str).to_numpy()[rows],
                          "split": prof.split.to_numpy()[rows], "tissue": prof.tissue.fillna("no tissue label").to_numpy()[rows],
                          "sex": prof.sex.where(prof.sex.isin(["F", "M"]), "").to_numpy()[rows],
                          "age": lab.age_meta.to_numpy(float)[rows], "array": z["array"][p], "array_source": z["array_source"][p],
                          "n_observed_vocab_cpgs": z["n_observed"][p],
                          "in_tissue_rule1": flags[:, 0], "in_tissue_rule2": flags[:, 1], "in_whole_blood_datasets": flags[:, 3]})
        for rep in REPS:
            t[f"umap1_{rep}"], t[f"umap2_{rep}"] = u[f"umap_{s_}_{rep}"][:, 0], u[f"umap_{s_}_{rep}"][:, 1]
        chk(f"1 GEO only (Fig. 3 set {s_} table)", is_geo(t.profile_id, t.dataset).all())
        f = OUT / f"fig3_set{s_}_umap_{DATE}.csv.gz"
        t.to_csv(f, index=False, float_format="%.6g")
        files[f.name] = len(t)
    met = pd.read_csv(WORK / "fig3_metrics.csv")
    f = OUT / f"fig3_metrics_{DATE}.csv"
    met.to_csv(f, index=False)
    files[f.name] = len(met)
    # Fig. 4a
    a = fig4a_counts(prof, lab)
    f = OUT / f"fig4a_training_age_bins_{DATE}.csv"
    a.to_csv(f, index=False)
    files[f.name] = len(a)
    # Fig. 4b-d
    ho = z["rows_ho"]
    t = pd.DataFrame({"corpus_row": ho, "profile_id": prof.id.to_numpy()[ho], "dataset": prof.group.astype(str).to_numpy()[ho],
                      "age": lab.age_meta.to_numpy(float)[ho], "tissue": prof.tissue.fillna("no tissue label").to_numpy()[ho]})
    for e in ENCODERS:
        for j in range(5):
            t[f"pc{j + 1}_transformer_{e}"] = z[f"ho_{e}_transformer"][:, j]
        for part in ("transformer", "whole"):
            t[f"umap1_{part}_{e}"], t[f"umap2_{part}_{e}"] = u[f"umap_ho_{e}_{part}"][:, 0], u[f"umap_ho_{e}_{part}"][:, 1]
    chk("1 GEO only (Fig. 4b-d table)", is_geo(t.profile_id, t.dataset).all())
    f = OUT / f"fig4bcd_heldout_embedding_{DATE}.csv.gz"
    t.to_csv(f, index=False, float_format="%.6g")
    files[f.name] = len(t)
    v = pd.read_csv(WORK / "fig4bcd_variance.csv")
    f = OUT / f"fig4bcd_pc_variance_age_{DATE}.csv"
    v.to_csv(f, index=False)
    files[f.name] = len(v)
    # Fig. 4e
    e = fig4e_table(prof, z)
    f = OUT / f"fig4e_medae_by_dataset_{DATE}.csv"
    e.to_csv(f, index=False)
    files[f.name] = len(e)
    allchecks = []
    for st in ("extract", "metrics"):
        allchecks += json.loads(require(WORK / f"checks_{st}.json").read_text())
    allchecks += CHECKS
    pd.DataFrame(allchecks).to_csv(OUT / f"checks_{DATE}.csv", index=False)
    write_json(WORK / "files.json", {k: {"rows": n, "bytes": (OUT / k).stat().st_size} for k, n in files.items()})
    save_checks("write")
    write_readme(files, met, pd.DataFrame(allchecks), e)
    log(f"written: {files}")


COLS_SET = {"corpus_row": "row in the corpus profile table", "profile_id": "GEO sample (GSM) ID", "dataset": "GEO series (GSE)",
            "split": "corpus split (test, or train for the 30,000 added training-split profiles of set D)",
            "tissue": "tissue label of the corpus ('no tissue label' if missing)", "sex": "sex label (F, M; empty if unlabelled)",
            "age": "metadata age in years (age_meta; empty if missing)", "array": "27K, 450K, EPIC, EPIC v2 or other (see array_source)",
            "array_source": "'series platform': the one GEO platform the corpus lists for the series; 'observed CpGs': no or several platforms listed, assigned from the observed vocabulary CpGs (rule in the script docstring)",
            "n_observed_vocab_cpgs": "vocabulary CpGs with a value (of 49,156)",
            "in_tissue_rule1": "profile in the tissue metrics, rule 1 (tissues with >= 30 profiles in each of >= 2 datasets)",
            "in_tissue_rule2": "profile in the tissue metrics, rule 2 (tissues with >= 30 profiles in total, from >= 2 datasets)",
            "in_whole_blood_datasets": "whole-blood profile of a dataset with >= 30 whole-blood profiles in the set (whole-blood metrics)"}
REP_TXT = {"raw": "raw beta values (training-mean filled)", "latent": "latent scores (1,536)", "embedding": "embedding (2,048, released run 1)",
           "encoder": "transformer part of the embedding (512)"}
COLS_MET = {"set": "C or D", "representation": "raw, latent, embedding or encoder", "metric": "metric", "cv": "stratified5 (stratified fivefold, seed 20261001), lodo (each dataset held out), none (no cross-validation), or 'not computed'",
            "profiles": "profile subset (tissue rule 1 or 2; whole blood, arrays; whole blood, datasets; sex labelled)", "value": "metric value",
            "ci95_low": "2.5th percentile, 2,000 dataset-resampling draws", "ci95_high": "97.5th percentile", "n_profiles": "profiles in the subset",
            "n_datasets": "datasets in the subset", "n_classes": "classes (tissues, arrays, datasets or sexes)", "classes": "class names (empty for datasets)", "note": "note"}


def write_readme(files, met, checks, e4):
    size = lambda f: (OUT / f).stat().st_size
    info = json.loads(require(WORK / "extract_info.json").read_text())
    src_n = pd.read_csv(OUT / f"fig3_setD_umap_{DATE}.csv.gz", usecols=["array_source"]).array_source.value_counts().to_dict()
    L = ["# Tables behind the redrawn Fig. 3 and Fig. 4a-e (MethylGPT revision)", "",
         f"Written by `export_figure_tables_{DATE}.py` on 10 Nov 2026 for the Mac figure builds. "
         "Tables only: no figure, legend or manuscript text, no model weights. The design is in the script docstring, committed before the script ran. "
         "Profile-level tables hold only public GEO profiles (GSM samples in GSE series); the script checks this before writing. "
         "No Generation Scotland file is read. Released weights (methylgpt2-1.1) and the named checkpoints; nothing was trained.", "", "## Files", ""]

    def block(f, desc, src, inputs, cols):
        L.extend([f"### {f}", "", desc, "", f"- Rows: {files[f]:,}; size: {size(f):,} bytes", f"- Source script: {src}",
                  "- Inputs: " + "; ".join(f"`{i}`" for i in inputs), "- Columns:"])
        L.extend(f"  - `{k}`: {v}" for k, v in cols.items())
        L.append("")
    for s_, n, desc in (("C", "13,191", "every test-split GEO profile (113 datasets)"),
                        ("D", "43,191", "set C plus 30,000 training-split GEO profiles drawn at random (seed 20261110)")):
        cols = dict(COLS_SET)
        for rep in REPS:
            cols[f"umap1_{rep}"] = f"UMAP 1, {REP_TXT[rep]}"
            cols[f"umap2_{rep}"] = f"UMAP 2, {REP_TXT[rep]}"
        block(f"fig3_set{s_}_umap_{DATE}.csv.gz",
              f"Fig. 3. Set {s_}: {desc}, {n} profiles, each with the CpGs it was deposited with. For each representation separately: centred, exact PCA, "
              "first 50 principal components, UMAP (umap-learn, 30 neighbours, min_dist 0.3, Euclidean, seed 0).",
              f"export_figure_tables_{DATE}.py (stages extract, umap)",
              ["artifacts/release/methylgpt2-1.1 (weights.pt, core.npz)", "artifacts/corpus_2026-09-30/beta_f16.npy, profiles.parquet, profiles_age_clean_2026-10-01.parquet",
               "artifacts/platform_vocab_masks_2026-10-01.npz"], cols)
    block(f"fig3_metrics_{DATE}.csv",
          "Fig. 3. Per set and representation, on the 50 principal components of that set and representation (cosine): tissue 10-NN balanced accuracy "
          "(stratified fivefold and each dataset held out) and tissue silhouette, for two readings of the tissue rule; within whole blood, array 10-NN "
          "balanced accuracy (both schemes), dataset 10-NN balanced accuracy (stratified fivefold) and median dataset iLISI; sex 10-NN balanced accuracy "
          "(both schemes). 95% intervals from 2,000 draws resampling the datasets of the set (seed 20261110).",
          f"export_figure_tables_{DATE}.py (stage metrics)", [f"artifacts/figure_tables_{DATE}_work/extract.npz (principal components; not exported)"], COLS_MET)
    block(f"fig4a_training_age_bins_{DATE}.csv",
          "Fig. 4a. The 41,845 training-split profiles with a clean age (age_clean): profiles per one-year age bin, overall and per tissue group of the pie "
          "(Whole Blood, Brain, Cells, Skin, Muscle; Other = every other tissue label and no label). Counts only; bins without profiles are listed with 0.",
          f"export_figure_tables_{DATE}.py (stage write)", ["artifacts/corpus_2026-09-30/profiles.parquet, profiles_age_clean_2026-10-01.parquet"],
          {"group": "all, Whole Blood, Brain, Cells, Skin, Muscle or Other", "age_bin_start": "bin start (years, inclusive)",
           "age_bin_end": "bin end (years, exclusive)", "n_profiles": "profiles"})
    cols = {"corpus_row": "row in the corpus profile table", "profile_id": "GEO sample (GSM) ID", "dataset": "GEO series (GSE)",
            "age": "metadata age (years)", "tissue": "tissue label ('no tissue label' if missing)"}
    for enc, txt in (("ssl", "self-supervised encoder (runs/stopped_cf_dense_2026-09-30/best.pt)"),
                     ("age_stage", "age-stage encoder (runs/aging_heads_clean_2026-10-01/best.pt)")):
        for j in range(5):
            cols[f"pc{j + 1}_transformer_{enc}"] = f"PC{j + 1} of the transformer part of the embedding (dimensions 1,537 to 2,048), {txt}"
        for part in ("transformer", "whole"):
            ptxt = "transformer part" if part == "transformer" else "whole 2,048-dimensional embedding"
            cols[f"umap1_{part}_{enc}"] = f"UMAP 1 of the 50 PCs of the {ptxt}, {txt}"
            cols[f"umap2_{part}_{enc}"] = f"UMAP 2, as above"
    block(f"fig4bcd_heldout_embedding_{DATE}.csv.gz",
          "Fig. 4b-d. The 5,811 test-split profiles with metadata ages (34 datasets), embedded by the self-supervised and the age-stage encoders. "
          "PCA fitted per encoder and part on these profiles (centred; sign: loading of largest magnitude positive); UMAP parameters as Fig. 3.",
          f"export_figure_tables_{DATE}.py (stages extract, umap)",
          ["artifacts/runs/stopped_cf_dense_2026-09-30/best.pt", "artifacts/runs/aging_heads_clean_2026-10-01/best.pt",
           "artifacts/corpus_2026-09-30/beta_f16.npy, profiles.parquet, profiles_age_clean_2026-10-01.parquet"], cols)
    block(f"fig4bcd_pc_variance_age_{DATE}.csv",
          "Fig. 4b-d. Principal components 1 to 5 of the transformer part and of the whole embedding, per encoder: fraction of the part's total variance and "
          "Spearman correlation with metadata age, over all 5,811 profiles and within whole blood.",
          f"export_figure_tables_{DATE}.py (stage extract)", ["as fig4bcd_heldout_embedding"],
          {"encoder": "ssl or age_stage", "part": "transformer (512 dimensions) or whole (2,048)", "pc": "principal component (1 to 5)",
           "explained_variance_ratio": "fraction of the part's total variance", "spearman_age_all": "Spearman correlation with age, all profiles",
           "n_all": "profiles", "spearman_age_whole_blood": "Spearman correlation with age, whole-blood profiles", "n_whole_blood": "whole-blood profiles"})
    block(f"fig4e_medae_by_dataset_{DATE}.csv",
          "Fig. 4e. Median absolute error (years) per method over all scored profiles (dataset 'all') and for every dataset with at least 10 scored profiles, "
          "on the AltumAge test set (4,626 profiles; stored predictions) and the AltumAge validation set (1,366 profiles; predictions computed here without training). "
          "Clocks are scored where ClockBase has a prediction; Zhang 2019 on profiles with >= 90% of its CpGs observed.",
          f"export_figure_tables_{DATE}.py (stages extract, write)",
          ["artifacts/eval/age_ensemble_2026-10-04.predictions.npz", "artifacts/eval/missing_cpgs_comparators_2026-10-18.predictions.npz",
           "ClockBase and Zhang 2019 predictions (clock_table() of evaluate_age_revision_2026-10-01.py)", "artifacts/release/methylgpt2-1.1",
           "artifacts/eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz", "artifacts/runs/mlp_altumage_style_2026-10-18/weights.pt",
           "AltumAge validation and test splits (load_altumage_split)"],
          {"set": "AltumAge test or AltumAge validation", "method": "MethylGPT (mean of three runs), Elastic net, MLP (AltumAge-style, mean of three seeds), Horvath multi-tissue, Hannum, Horvath skin and blood, Lin, Zhang 2019, AltumAge (pyaging)",
           "validation_used_for_selection": "True where the validation set chose the method's settings (MethylGPT age head, elastic net, MLP)",
           "dataset": "corpus series of the profiles, or 'all'", "n_scored": "scored profiles", "medae": "median absolute error (years)",
           "n_datasets": "datasets among the scored profiles (1 for a dataset row)"})
    L.extend([f"### checks_{DATE}.csv", "", "Every check of the script (all passed).", ""])
    # metric summary
    L.extend(["## Fig. 3 metrics (value and 95% interval; every row of fig3_metrics)", ""])
    for s_ in SETS:
        x = met[met.set == s_]
        L.extend([f"### Set {s_}", "", "| metric | cv | profiles | n profiles / datasets / classes | " + " | ".join(REPS) + " |",
                  "|---|---|---|---|" + "---|" * len(REPS)])
        for (mt, cv, sub), g in x.groupby(["metric", "cv", "profiles"], sort=False):
            g = g.set_index("representation")
            r0 = g.iloc[0]
            cells = [("not computed" if not np.isfinite(g.loc[r, "value"]) else f"{g.loc[r, 'value']:.3f} [{g.loc[r, 'ci95_low']:.3f}, {g.loc[r, 'ci95_high']:.3f}]")
                     if r in g.index else "" for r in REPS]
            L.append(f"| {mt} | {cv} | {sub} | {int(r0.n_profiles):,} / {int(r0.n_datasets)} / {int(r0.n_classes)} | " + " | ".join(cells) + " |")
        L.append("")
    L.extend(["Classes per subset are listed in fig3_metrics (column classes).", "",
              "PCA: fraction of variance in the first 50 components: " + "; ".join(f"set {k.replace('|', ' ')} {v['explained_variance_ratio_50']:.3f}" for k, v in info["pca"].items()), ""])
    L.extend(["## Fig. 4e overall MedAE", "", "| set | method | n scored | datasets | MedAE |", "|---|---|---|---|---|"])
    for _, r in e4[e4.dataset == "all"].iterrows():
        L.append(f"| {r.set} | {r.method} | {r.n_scored:,} | {r.n_datasets} | {r.medae:.3f} |")
    L.extend(["", "## Checks", ""])
    L.extend(f"- {r.check}: {'pass' if r.passed else 'FAIL'}{'; ' + r.detail if r.detail else ''}" for _, r in checks.iterrows()
             if not r.check.startswith("6 balanced accuracy point estimate"))
    nb = checks.check.str.startswith("6 balanced accuracy point estimate")
    L.append(f"- 6 balanced accuracy point estimates equal balanced_accuracy_score: {int(checks[nb].passed.sum())} of {int(nb.sum())} rows")
    L.extend(["", "## Deviations and notes", "",
              "- Set C holds the 13,191 test-split GEO profiles (113 datasets). The specified 14,185 profiles (116 datasets) include 994 test-split TCGA and TARGET "
              "profiles (TCGA-LIHC 408, TCGA-LUSC 391, TARGET-AML 195) that are not GEO profiles and carry no tissue, sex or platform label; they were left out "
              "so that every profile in the embeddings and metrics is in the exported table. Set D adds training-split GEO profiles only.",
              "- 'Tissues with at least 30 profiles in at least 2 datasets' can be read two ways; both are reported (rule 1: >= 30 in each of >= 2 datasets; "
              "rule 2: >= 30 in total from >= 2 datasets).",
              "- Whole-blood metrics use the whole-blood profiles of datasets with >= 30 whole-blood profiles in the set (the dataset rule of set B in "
              "the 2026-10-01 benchmark); array classes need >= 30 of these profiles from >= 2 datasets. Array 10-NN is also reported with each dataset held "
              "out, because with stratified folds a profile's neighbours can come from its own dataset, whose array is the same. Not in the specification; fixed before running.",
              "- All metrics use the principal components of the whole set; within whole blood they are not refitted.",
              "- Array: the corpus table lists the GEO platform(s) of each series; where one platform is listed it gives the array, otherwise the observed CpGs do "
              "(see the agreement check). The 450K mask leaves out vocabulary CpGs that 450K profiles observe, so the rule uses the CpG sets whose observed "
              "fractions separate the platforms (outside 27K; EPIC v2 not EPIC v1; EPIC v1 not EPIC v2). In set D the platform gives the array for "
              f"{src_n.get('series platform', 0):,} profiles and the observed CpGs for {src_n.get('observed CpGs', 0):,}.",
              "- Fig. 4e validation: no per-profile validation predictions were stored, so MethylGPT (released bundle), the elastic net (stored model) and the "
              "MLP (stored weights) were run on the 1,366 validation profiles without training, and the clocks looked up; the checks compare the recomputed test "
              "predictions with the stored ones and the validation MedAE with the values stored at selection. A clock with fewer than 50 scored validation "
              "profiles is not reported.",
              "- Fig. 4b-d: the variance table also lists the whole embedding's first five components; its PC1 and PC2 reproduce table4_fig4bcd_pca_variance of "
              "the 2026-10-23 export.",
              "- Age: the set tables give metadata ages (age_meta); Fig. 4a counts the cleaned training ages (age_clean), the labels of the age stage.",
              "- Signs of principal components are arbitrary; here the loading of largest magnitude is positive. PC2 of the whole embedding is the negative of "
              "PC2 in table4_fig4bcd_pca_variance for both encoders (same variance, Spearman correlation of opposite sign); PC1 has the same sign.",
              "- umap-learn warned for 7 of the 12 UMAPs that the 30-neighbour graph is not fully connected (groups of profiles with no neighbours outside the "
              "group); UMAP lays such components out separately. The coordinates are written as computed.",
              "- Array 'other': 54 training-split profiles of set D from series listed with platform GPL18809 only (5 datasets); none is a whole-blood "
              "profile, so 'other' is not an array class of the whole-blood metrics.",
              "- Zhang 2019 is not reported on the validation set: the Zhang 2019 rerun file has no prediction for any validation profile.",
              "- set D raw values: PCA by subspace iteration (amendment 1, check 8); every other PCA is a full eigendecomposition.", ""])
    (OUT / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "umap", "metrics", "write", "all"], required=True)
    a = ap.parse_args()
    stages = ["extract", "umap", "metrics", "write"] if a.stage == "all" else [a.stage]
    for s in stages:
        CHECKS.clear()
        {"extract": stage_extract, "umap": stage_umap, "metrics": stage_metrics, "write": stage_write}[s]()


if __name__ == "__main__":
    main()
