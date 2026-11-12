#!/usr/bin/env python3
"""Supplementary analysis: sample representations by dataset, age and sex.
Companion to benchmark_representations_2026-10-01.py: reads its cached representations and adds the
columns of Supplementary Table 3 listed below. CPU only.

Inputs
  artifacts/eval/representations_setA_2026-10-01.npz   set A: 2,371 profiles, six tissues, 43 test-split datasets
  artifacts/eval/representations_setB_2026-10-01.npz   set B: 949 whole-blood profiles with sex and age, 10 datasets
  artifacts/eval/representation_benchmark_2026-10-01.json   existing metrics (reproduction check)
  artifacts/corpus_2026-09-30/profiles.parquet, profiles_age_clean_2026-10-01.parquet   tissue, sex, dataset, age
Representations: raw beta values (training-mean filled), raw PCA-50, latent scores (input-layer posterior means),
self-supervised and multitask MethylGPT embeddings, untrained transformer (fitted input layer, initial weights), and
a Gaussian random projection of raw beta values (2,048 dimensions); in set B also ComBat (R sva) and Harmony.
Geometry as in the 2026-10-01 benchmark: natural (centred) is primary, per-feature standardized is a sensitivity.

PRE-SPECIFIED PRIMARY METRIC (written before running): set A, leave-dataset-out tissue 10-NN balanced accuracy
(cosine, full representation, natural geometry; GroupKFold with one fold per dataset, i.e. leave one dataset out) of
latent scores minus that of raw beta values. 95% interval from resampling datasets within tissue (dataset-tissue
blocks, so every draw holds all six tissues; 2,000 draws, seed 20261018) with the cross-validated predictions held
fixed; the unstratified dataset bootstrap is reported as a sensitivity.

Secondary (all reported)
  (i)   tissue and sex 10-NN balanced accuracy with leave-one-dataset-out and GroupKFold(5) CV next to the stratified
        5-fold values (set A; sex also in set B), both geometries.
  (ii)  Leiden clustering (python-igraph community_leiden, modularity, unweighted symmetric 15-NN cosine graph on
        the first 50 principal components, resolutions 0.02-2.0, seed 0): number of clusters, ARI and NMI against
        tissue and against dataset (set A) and against dataset (set B); k-means (k = 6 in set A, 10 in set B, on
        L2-normalised 50 PCs, 10 starts, seed 0) as the reference with the number of classes given; ARI/NMI of the
        tissue labels themselves against dataset (set A nesting reference). Every representation, including the
        untrained transformer and the random projection.
  (iii) 95% intervals from resampling datasets for every metric (set A within tissue, set B the 10 datasets; models,
        clusterings and per-sample values held fixed; silhouette, energy distance, ARI and NMI recomputed exactly on
        each resampled set from per-dataset block sums, without the zero distances between copies of a profile, and
        with the copies of a dataset drawn twice treated as different datasets where dataset is the label).
  (iv)  per-profile fraction of the 10 nearest neighbours (cosine, natural geometry) from another tissue, raw beta
        values and latent scores; Spearman correlation between the two; profiles with at least 5 of 10 neighbours
        from another tissue in one representation and not the other ("discordant"), counted by dataset (descriptive).
  (v)   ComBat with protected covariates in set B (on all CpGs, and on the CpGs ComBat can adjust, i.e. those with
        nonzero variance within every dataset): age and sex tabulated by dataset and design estimability checked
        (rank of [dataset indicators, covariates]); ComBat without covariates, with sex, and with sex and age.
        In-sample fits (R sva 3.56, all profiles; the matrix is passed to R in the CpG x profile order sva expects, which the
        2026-10-01 wrapper did not do, so its ComBat values are recomputed here) give an upper bound for any metric of a
        protected label; within-fold
        fits (Python port of the parametric sva ComBat, validated against R on the in-sample fits; fitted on the
        training folds of a 5-fold split stratified by dataset, applied to the held-out fold without its labels) give
        dataset 10-NN, sex 10-NN, age ridge R^2 (50 PCs fitted on the training fold), silhouette and iLISI. The same
        within-fold protocol is run for raw beta values, latent scores and the embedding.
  (vi)  dataset 10-NN balanced accuracy after regressing age and sex out of every feature, and among profiles aged
        25 to 45 years in datasets with at least 20 such profiles.
  Set B also gets age ridge R^2 and age 10-NN MedAE with leave-one-dataset-out CV next to random 5-fold CV.
  UMAP coordinates of set B (latent scores and raw beta values; PCA-50 then UMAP, 30 neighbours, min_dist 0.3,
  seed 0, as Fig. 3) for the supplementary figure.
Outputs
  artifacts/eval/si_sample_representations_2026-10-18.json
  artifacts/eval/si_sample_representations_2026-10-18.csv          (Supplementary Table 3 rows)
  artifacts/eval/si_sample_representations_design_setB_2026-10-18.csv   age and sex by dataset (set B)
  artifacts/eval/si_sample_representations_cross_tissue_2026-10-18.csv  per-profile cross-tissue fractions (set A)
  artifacts/eval/si_sample_representations_umap_setB_2026-10-18.npz
"""
from __future__ import annotations

import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

sys.path.insert(0, "/Entropy/kying/tmp/pylib")
import harmonypy  # noqa: E402
from harmonypy import compute_lisi  # noqa: E402

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json  # noqa: E402

EVAL = ARTIFACTS / "eval"
RLIB = "/Entropy/kying/tmp/rlib"
SEED = 20261018
NBOOT = 2000
RESOLUTIONS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
REPS_A = ["raw", "raw_pca50", "factor_scores", "mgpt2_selfsupervised", "mgpt2_multitask", "untrained_transformer", "random_projection"]


# ----------------------------------------------------------------------------------------------------------------
# geometry and cross-validation helpers
# ----------------------------------------------------------------------------------------------------------------
def geom(Z, g="natural"):
    Z = np.asarray(Z, np.float64)
    Z = Z - Z.mean(0)
    if g == "standardized":
        s = Z.std(0)
        Z = Z / np.where(s > 1e-8, s, 1.0)
    return Z


_DEV = None


def _t(X):
    """float64 tensor on the GPU when one is visible (run with CUDA_VISIBLE_DEVICES=1), else CPU. NumPy on this server
    links the reference BLAS, so every large product goes through torch."""
    global _DEV
    if _DEV is None:
        _DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float64, device=_DEV)


def pca_fit(Zc, k=50):
    """Exact PCA of a centred matrix (eigendecomposition of the smaller Gram matrix). Returns loadings (p x k)."""
    T = _t(Zc)
    n, p = T.shape
    k = min(k, p, n - 1)
    if p <= n:
        L, V = torch.linalg.eigh(T.T @ T)
        V = V[:, torch.argsort(L, descending=True)[:k]]
    else:
        L, U = torch.linalg.eigh(T @ T.T)
        o = torch.argsort(L, descending=True)[:k]
        V = (T.T @ U[:, o]) / torch.sqrt(L[o].clamp_min(1e-300))
    return V


def pca_transform(Zc, V):
    return (_t(Zc) @ V).cpu().numpy()


def pcs(Z, g="natural", k=50):
    Zg = geom(Z, g)
    return pca_transform(Zg, pca_fit(Zg, k))


def cos_sim(A, B=None):
    TA = _t(A)
    TA = TA / TA.norm(dim=1, keepdim=True).clamp_min(1e-300)
    if B is None:
        TB = TA
    else:
        TB = _t(B)
        TB = TB / TB.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return (TA @ TB.T).cpu().numpy()


def _knn_vote(S_te_tr, ytr, k=10):
    """10-NN by cosine similarity, majority vote; ties go to the first class in sorted order (as scikit-learn)."""
    nb = np.argsort(-S_te_tr, axis=1, kind="stable")[:, :k]
    cl, yi = np.unique(ytr, return_inverse=True)
    cnt = np.zeros((len(S_te_tr), len(cl)))
    np.add.at(cnt, (np.repeat(np.arange(len(S_te_tr)), k), yi[nb].ravel()), 1)
    return cl[cnt.argmax(1)]


def splits(scheme, y, groups):
    n = len(y)
    if scheme == "stratified5":
        return list(StratifiedKFold(5, shuffle=True, random_state=20261001).split(np.zeros(n), y))
    if scheme == "kfold5":
        return list(KFold(5, shuffle=True, random_state=0).split(np.zeros(n)))
    if scheme == "lodo":
        return list(LeaveOneGroupOut().split(np.zeros(n), y, groups))
    if scheme == "groupkfold5":
        return list(GroupKFold(5).split(np.zeros(n), y, groups))
    raise ValueError(scheme)


def knn_oof(Zg, y, sp, k=10, S=None):
    """Out-of-fold 10-NN (cosine) class predictions; Zg already in its geometry (S: its cosine similarity matrix)."""
    S = cos_sim(Zg) if S is None else S
    y = np.asarray(y)
    pred = np.empty(len(y), dtype=object)
    for tr, te in sp:
        pred[te] = _knn_vote(S[np.ix_(te, tr)], y[tr], k)
    return pred


def knn_reg_oof(Zg, y, sp, k=10, S=None):
    S = cos_sim(Zg) if S is None else S
    pred = np.empty(len(y))
    for tr, te in sp:
        nb = np.argsort(-S[np.ix_(te, tr)], axis=1, kind="stable")[:, :k]
        pred[te] = np.asarray(y)[tr][nb].mean(1)
    return pred


def ridge_oof(P, y, sp):
    pred = np.empty(len(y))
    for tr, te in sp:
        pred[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(P[tr], y[tr]).predict(P[te])
    return pred


# ----------------------------------------------------------------------------------------------------------------
# dataset bootstrap: weights W (draws x units); every statistic is recomputed exactly from per-unit sums
# ----------------------------------------------------------------------------------------------------------------
def boot_weights(units, strata=None, n=NBOOT, seed=SEED):
    """units: per-profile unit label (dataset, or dataset|tissue). strata: per-profile stratum (resample units within
    strata) or None. Returns (unit codes per profile, W of shape (n, n_units))."""
    u, codes = np.unique(np.asarray(units, str), return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.zeros((n, len(u)))
    if strata is None:
        groups = [np.arange(len(u))]
    else:
        st = pd.Series(np.asarray(strata, str)).groupby(codes).first()
        groups = [np.flatnonzero(st.to_numpy() == s) for s in sorted(set(st))]
    for gi in groups:
        draws = rng.integers(0, len(gi), (n, len(gi)))
        for b in range(n):
            np.add.at(W[b], gi[draws[b]], 1)
    return codes, W


def ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _bacc_from_conf(C):
    rows = C.sum(-1)
    diag = np.diagonal(C, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = np.where(rows > 0, diag / rows, np.nan)
    return np.nanmean(per, -1)


def boot_bacc(y, pred, codes, W):
    ok = pd.notna(y)
    y, pred, codes = np.asarray(y)[ok].astype(str), np.asarray(pred)[ok].astype(str), codes[ok]
    cl = np.unique(np.concatenate([y, pred]))
    yi, pi = np.searchsorted(cl, y), np.searchsorted(cl, pred)
    M = np.zeros((W.shape[1], len(cl), len(cl)))
    np.add.at(M, (codes, yi, pi), 1)
    point = float(balanced_accuracy_score(y, pred))
    assert abs(point - _bacc_from_conf(M.sum(0))) < 1e-9
    return point, _bacc_from_conf(np.einsum("bg,gij->bij", W, M))


def boot_r2(y, pred, codes, W):
    G = W.shape[1]
    s = [np.bincount(codes, weights=v, minlength=G) for v in (np.ones_like(y), y, y * y, (y - pred) ** 2)]
    n, sy, syy, sse = (W @ v for v in s)
    point = float(r2_score(y, pred))
    return point, 1 - sse / (syy - sy ** 2 / n)


def boot_median(vals, codes, W):
    order = np.argsort(codes, kind="stable")
    by = np.split(np.asarray(vals, float)[order], np.cumsum(np.bincount(codes, minlength=W.shape[1]))[:-1])
    out = np.empty(len(W))
    for b in range(len(W)):
        out[b] = np.median(np.concatenate([np.repeat(by[g], int(w)) for g, w in enumerate(W[b]) if w > 0] or [np.array([np.nan])]))
    return float(np.median(vals)), out


def _ari_nmi(C):
    """C: (..., I, J) contingency tables. Returns ARI, NMI (arithmetic normalization), as sklearn."""
    n = C.sum((-1, -2))
    a, b = C.sum(-1), C.sum(-2)
    comb = lambda x: x * (x - 1) / 2
    idx = comb(C).sum((-1, -2))
    sa, sb = comb(a).sum(-1), comb(b).sum(-1)
    exp = sa * sb / comb(n)
    mx = (sa + sb) / 2
    ari = (idx - exp) / (mx - exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        pij = C / n[..., None, None]
        pa, pb = a / n[..., None], b / n[..., None]
        mi = np.nansum(np.where(C > 0, pij * np.log(pij / (pa[..., :, None] * pb[..., None, :])), 0.0), (-1, -2))
        ha = -np.nansum(np.where(pa > 0, pa * np.log(pa), 0.0), -1)
        hb = -np.nansum(np.where(pb > 0, pb * np.log(pb), 0.0), -1)
    nmi = mi / np.maximum((ha + hb) / 2, np.finfo(float).eps)
    return ari, nmi


_RESAMPLE_CACHE = {}


def _resampled(codes, W):
    """Per draw: profile indices of the resampled set and the copy number of each (0 for the first copy of a unit)."""
    key = (id(W), id(codes))
    if key not in _RESAMPLE_CACHE:
        members = [np.flatnonzero(codes == g) for g in range(W.shape[1])]
        out = []
        for wb in W.astype(int):
            idx = [np.tile(members[g], w) for g, w in enumerate(wb) if w > 0]
            cp = [np.repeat(np.arange(w), len(members[g])) for g, w in enumerate(wb) if w > 0]
            out.append((np.concatenate(idx), np.concatenate(cp)))
        _RESAMPLE_CACHE[key] = (W, codes, out)
    return _RESAMPLE_CACHE[key][2]


def boot_ari_nmi(lab_a, lab_b, codes, W, distinct_copies_b=False):
    """ARI and NMI between labelings a (clusters) and b. With distinct_copies_b (b = dataset labels), the copies of a
    dataset drawn more than once are treated as different datasets, as independently drawn datasets would be."""
    _, ai = np.unique(np.asarray(lab_a, str), return_inverse=True)
    _, bi = np.unique(np.asarray(lab_b, str), return_inverse=True)
    M = np.zeros((W.shape[1], ai.max() + 1, bi.max() + 1))
    np.add.at(M, (codes, ai, bi), 1)
    pa, pn = _ari_nmi(M.sum(0))
    assert abs(pa - adjusted_rand_score(ai, bi)) < 1e-9 and abs(pn - normalized_mutual_info_score(ai, bi)) < 1e-9
    if not distinct_copies_b:
        ba, bn = [], []
        for i in range(0, len(W), 250):
            x, y = _ari_nmi(np.einsum("bg,gij->bij", W[i:i + 250], M))
            ba.append(x); bn.append(y)
        return (float(pa), np.concatenate(ba)), (float(pn), np.concatenate(bn))
    na, nb = ai.max() + 1, bi.max() + 1
    ba, bn = np.empty(len(W)), np.empty(len(W))
    for b, (idx, cp) in enumerate(_resampled(codes, W)):
        col = bi[idx] + nb * cp
        C = np.bincount(ai[idx] * (nb * (cp.max() + 1)) + col, minlength=na * nb * (cp.max() + 1)).reshape(na, -1).astype(float)
        C = C[:, C.sum(0) > 0]
        ba[b], bn[b] = _ari_nmi(C)
    return (float(pa), ba), (float(pn), bn)


def _cosine_dist(P):
    Q = P / np.linalg.norm(P, axis=1, keepdims=True)
    return np.clip(1 - Q @ Q.T, 0, 2)


def boot_silhouette(P, labels, codes, W, chunk=100):
    """Cosine silhouette on P, recomputed exactly on each resampled set from per-unit distance sums."""
    _, li = np.unique(np.asarray(labels, str), return_inverse=True)
    C, G, n = li.max() + 1, W.shape[1], len(li)
    D = _cosine_dist(P)
    np.fill_diagonal(D, 0.0)
    blk = codes * C + li
    Ds = np.zeros((n, G * C))
    for j in range(G * C):
        m = blk == j
        if m.any():
            Ds[:, j] = D[:, m].sum(1)
    Ds = Ds.reshape(n, G, C)
    Ng = np.zeros((G, C))
    np.add.at(Ng, (codes, li), 1)
    point = float(silhouette_score(P, labels, metric="cosine"))

    def sil(Wb):  # Wb: (B, G)
        S = np.einsum("bg,igc->bic", Wb, Ds)
        Nc = Wb @ Ng
        own = li[None, :]
        # same-cluster partners exclude every copy of the profile itself (a duplicated dataset adds no zero distances)
        a_n = np.take_along_axis(Nc, np.broadcast_to(own, S.shape[:2]), 1) - Wb[:, codes]
        a = np.take_along_axis(S, own[..., None].repeat(len(Wb), 0), 2)[..., 0] / np.where(a_n > 0, a_n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = S / Nc[:, None, :]
        means[np.broadcast_to((Nc == 0)[:, None, :], means.shape)] = np.inf
        np.put_along_axis(means, own[..., None].repeat(len(Wb), 0), np.inf, 2)
        b = means.min(2)
        s = (b - a) / np.maximum(a, b)
        s = np.where(a_n > 0, s, 0.0)
        wi = Wb[:, codes]
        return (np.nan_to_num(s) * wi).sum(1) / wi.sum(1)

    chk = float(sil(np.ones((1, G)))[0])
    assert abs(chk - point) < 1e-6, (chk, point)
    return point, np.concatenate([sil(W[i:i + chunk]) for i in range(0, len(W), chunk)])


def boot_energy(P, labels, codes, W):
    """Mean pairwise energy distance between label classes (Euclidean, PCs scaled to unit mean squared norm; the
    2026-10-01 definition), recomputed exactly per resampled set."""
    _, li = np.unique(np.asarray(labels, str), return_inverse=True)
    C, G = li.max() + 1, W.shape[1]
    sq = (P ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * P @ P.T, 0))
    blk = codes * C + li
    E = np.zeros((G * C, G * C))
    for j in range(G * C):
        mj = blk == j
        if mj.any():
            E[j] = np.bincount(blk, weights=D[mj].sum(0), minlength=G * C)
    E = E.reshape(G, C, G, C)
    Ng = np.zeros((G, C)); np.add.at(Ng, (codes, li), 1)
    Sq = np.bincount(codes, weights=sq, minlength=G)

    def ed(Wb):
        T = np.einsum("bh,bchd->bcd", Wb, np.einsum("bg,gchd->bchd", Wb, E))
        Nc = Wb @ Ng
        scale = np.sqrt((Wb @ Sq) / Nc.sum(1))
        pairs = Nc[:, :, None] * Nc[:, None, :]
        # within-class pairs: drop the zero-distance pairs between copies of the same profile (self pairs kept, as the
        # 2026-10-01 definition includes them)
        dup = (Wb ** 2 - Wb) @ Ng
        idx_ = np.arange(C)
        pairs[:, idx_, idx_] -= dup
        with np.errstate(invalid="ignore", divide="ignore"):
            m = T / pairs
        out = []
        for a in range(C):
            for b in range(a + 1, C):
                out.append(2 * m[:, a, b] - m[:, a, a] - m[:, b, b])
        return np.nanmean(np.stack(out, 1), 1) / scale

    point = float(ed(np.ones((1, G)))[0])
    return point, np.concatenate([ed(W[i:i + 200]) for i in range(0, len(W), 200)])


def entry(point, draws, **extra):
    return {"value": float(point), "ci95": ci(draws), **extra}


# ----------------------------------------------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------------------------------------------
def leiden_grid(P, k=15, seed=0):
    import igraph as ig
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(P)
    _, ind = nn.kneighbors(P)
    edges = {(min(i, j), max(i, j)) for i in range(len(P)) for j in ind[i, 1:] if i != j}
    g = ig.Graph(n=len(P), edges=sorted(edges))
    out = {}
    for r in RESOLUTIONS:
        random.seed(seed)
        ig.set_random_number_generator(random)
        part = g.community_leiden(objective_function="modularity", resolution=r, n_iterations=-1)
        out[r] = np.asarray(part.membership)
    return out


def kmeans_labels(P, k):
    Q = P / np.linalg.norm(P, axis=1, keepdims=True)
    return KMeans(k, n_init=10, random_state=0).fit_predict(Q)


# ----------------------------------------------------------------------------------------------------------------
# ComBat: R sva (in-sample) and a Python port of the parametric sva 3.56 algorithm (fit / apply)
# ----------------------------------------------------------------------------------------------------------------
def combat_r(X, batch, covars=None, legacy_transposed_write=False):
    """R sva::ComBat on a sample x CpG matrix; covars: DataFrame of protected covariates (mod = model.matrix(~ .)).

    The sample x CpG matrix is written in C order, which R reads column-major as the CpG x sample matrix it expects.
    The 2026-10-01 wrapper wrote X.T in C order instead, so R received a scrambled matrix (each R column mixed values of
    different profiles and CpGs); legacy_transposed_write=True reproduces that write for the check in main()."""
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as d:
        d = Path(d)
        (X.astype(np.float64).T if legacy_transposed_write else np.ascontiguousarray(X, dtype=np.float64)).tofile(d / "x.bin")
        pd.Series(batch).to_csv(d / "batch.csv", index=False, header=False)
        mod = "NULL"
        if covars is not None:
            covars.to_csv(d / "cov.csv", index=False)
            mod = f'model.matrix(~ ., data=read.csv("{d}/cov.csv"))'
        code = (f'.libPaths(c("{RLIB}", .libPaths())); suppressMessages(library(sva)); '
                f'x <- matrix(readBin("{d}/x.bin", "double", n={X.size}), nrow={X.shape[1]}, ncol={X.shape[0]}); '
                f'b <- read.csv("{d}/batch.csv", header=FALSE)[,1]; keep <- apply(x, 1, sd) > 1e-6; '
                f'y <- x; y[keep,] <- ComBat(x[keep,], batch=b, mod={mod}); writeBin(as.vector(y), "{d}/y.bin")')
        subprocess.run(["Rscript", "-e", code], check=True, capture_output=True)
        return np.fromfile(d / "y.bin", dtype=np.float64).reshape(X.shape[0], X.shape[1])


def combat_fit(X, batch, C=None, conv=1e-4):
    """Parametric ComBat (sva 3.56 algorithm, mean.only = FALSE). X: samples x features (no missing values);
    batch: labels; C: samples x p protected covariates (no intercept) or None. Features with SD <= 1e-6 overall or
    zero variance within a batch are left unadjusted (as sva and the 2026-10-01 wrapper). Returns (adjusted, params)."""
    X = np.asarray(X, np.float64)
    lv, bi = np.unique(np.asarray(batch, str), return_inverse=True)
    nb, n = len(lv), len(X)
    keep = X.std(0, ddof=1) > 1e-6
    for j in range(nb):
        m = bi == j
        if m.sum() > 1:
            keep &= X[m].max(0) != X[m].min(0)
    Xk = X[:, keep]
    Bd = np.eye(nb)[bi]
    D = Bd if C is None else np.hstack([Bd, np.asarray(C, np.float64)])
    if np.linalg.matrix_rank(D) < D.shape[1]:
        raise ValueError("design not estimable")
    Bh = np.linalg.solve(D.T @ D, D.T @ Xk)
    nbat = Bd.sum(0)
    grand = (nbat / n) @ Bh[:nb]
    var_pooled = ((Xk - D @ Bh) ** 2).mean(0)
    stand = grand[None, :] + (0 if C is None else np.asarray(C, np.float64) @ Bh[nb:])
    s = (Xk - stand) / np.sqrt(var_pooled)
    gamma_hat = np.stack([s[bi == j].mean(0) for j in range(nb)])
    delta_hat = np.stack([s[bi == j].var(0, ddof=1) for j in range(nb)])
    gamma_bar = gamma_hat.mean(1)
    t2 = gamma_hat.var(1, ddof=1)
    m_, s2 = delta_hat.mean(1), delta_hat.var(1, ddof=1)
    a_pr, b_pr = (2 * s2 + m_ ** 2) / s2, (m_ * s2 + m_ ** 3) / s2
    gstar, dstar = np.empty_like(gamma_hat), np.empty_like(delta_hat)
    for j in range(nb):
        sd = s[bi == j]
        nj = len(sd)
        g_old, d_old = gamma_hat[j].copy(), delta_hat[j].copy()
        change = 1.0
        while change > conv:
            g_new = (t2[j] * nj * gamma_hat[j] + d_old * gamma_bar[j]) / (t2[j] * nj + d_old)
            sum2 = ((sd - g_new) ** 2).sum(0)
            d_new = (0.5 * sum2 + b_pr[j]) / (nj / 2 + a_pr[j] - 1)
            change = max(np.max(np.abs(g_new - g_old) / g_old), np.max(np.abs(d_new - d_old) / d_old))
            g_old, d_old = g_new, d_new
        gstar[j], dstar[j] = g_new, d_new
    cov_mean = None if C is None else np.stack([np.asarray(C, np.float64)[bi == j].mean(0) for j in range(nb)])
    params = dict(levels=lv, keep=keep, grand=grand, beta=None if C is None else Bh[nb:], var_pooled=var_pooled,
                  gstar=gstar, dstar=dstar, cov_mean=cov_mean)
    out = X.copy()
    out[:, keep] = (s - gstar[bi]) / np.sqrt(dstar[bi]) * np.sqrt(var_pooled) + stand
    return out, params


def combat_apply(X, batch, p):
    """Apply fitted ComBat to new profiles of the fitted batches without their covariates: the protected-covariate
    term uses the training mean of the covariates in the profile's batch."""
    X = np.asarray(X, np.float64)
    bi = np.searchsorted(p["levels"], np.asarray(batch, str))
    assert (p["levels"][bi] == np.asarray(batch, str)).all()
    stand = p["grand"][None, :] + (0 if p["beta"] is None else p["cov_mean"][bi] @ p["beta"])
    k = p["keep"]
    s = (X[:, k] - stand) / np.sqrt(p["var_pooled"])
    out = X.copy()
    out[:, k] = (s - p["gstar"][bi]) / np.sqrt(p["dstar"][bi]) * np.sqrt(p["var_pooled"]) + stand
    return out


# ----------------------------------------------------------------------------------------------------------------
def load_sets():
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    out = {}
    for s in ("A", "B"):
        z = np.load(require(EVAL / f"representations_set{s}_2026-10-01.npz"))
        rows = z["rows"]
        reps = {k[4:]: z[k] for k in z.files if k.startswith("rep_")}
        reps["raw"] = z["raw_full"]
        meta = pd.DataFrame({"row": rows, "dataset": prof.group.astype(str).to_numpy()[rows], "tissue": prof.tissue.to_numpy()[rows],
                             "sex": prof.sex.to_numpy()[rows], "age": lab.age_meta.to_numpy(float)[rows], "gpl": prof.gpl.to_numpy()[rows]})
        meta["sex"] = meta.sex.where(meta.sex.isin(["M", "F"]))
        out[s] = (reps, meta)
    return out


def main():
    t0 = time.time()
    sets = load_sets()
    old = __import__("json").loads(require(EVAL / "representation_benchmark_2026-10-01.json").read_text())
    res = {"primary_metric": "set A leave-one-dataset-out tissue 10-NN balanced accuracy, latent scores minus raw beta values "
                             "(natural geometry); 95% interval from resampling datasets within tissue, 2,000 draws",
           "bootstrap": {"draws": NBOOT, "seed": SEED}, "checks": {}}
    rows_csv = []

    def add(set_, geometry, rep, metric, cv, e, n, ndat, note=""):
        rows_csv.append({"set": set_, "geometry": geometry, "representation": rep, "metric": metric, "cv": cv, "value": e["value"],
                         "ci95_low": e["ci95"][0], "ci95_high": e["ci95"][1], "n_profiles": n, "n_datasets": ndat, "note": note})

    # ======================================== set A ========================================
    reps, meta = sets["A"]
    tissue, sex, ds = meta.tissue.to_numpy(), meta.sex.to_numpy(), meta.dataset.to_numpy()
    nA, ndA = len(meta), meta.dataset.nunique()
    unitA = (meta.dataset + "|" + meta.tissue).to_numpy()
    cA, WA = boot_weights(unitA, strata=tissue)
    cA_u, WA_u = boot_weights(ds)                  # unstratified (sensitivity for the primary metric)
    res["set_a_composition"] = {"n_profiles": int(nA), "n_datasets": int(ndA),
                                "datasets_per_tissue": meta.groupby("tissue").dataset.nunique().to_dict(),
                                "profiles_per_tissue": meta.tissue.value_counts().to_dict(),
                                "datasets_in_more_than_one_tissue": sorted(meta.groupby("dataset").tissue.nunique().loc[lambda s: s > 1].index)}
    sa = {}
    oof = {}
    for g in ("natural", "standardized"):
        sa[g] = {}
        for name in REPS_A:
            P = pcs(reps[name], g)
            d = {}
            for target, y in (("tissue", tissue), ("sex", sex)):
                ok = pd.notna(y)
                Sg = cos_sim(geom(reps[name][ok], g))    # centred (and scaled) on the profiles with the label, as 2026-10-01
                for scheme in ("stratified5", "lodo", "groupkfold5"):
                    sp = splits(scheme, y[ok].astype(str), ds[ok])
                    pr = np.full(len(y), None, dtype=object)
                    pr[ok] = knn_oof(None, y[ok].astype(str), sp, S=Sg)
                    oof[(g, name, target, scheme)] = pr
                    pt, dr = boot_bacc(y[ok], pr[ok], cA[ok], WA)
                    d[f"{target}_knn_bacc|{scheme}"] = entry(pt, dr)
                    add("A", g, name, f"{target} 10-NN balanced accuracy", scheme, d[f"{target}_knn_bacc|{scheme}"], int(ok.sum()), int(pd.Series(ds[ok]).nunique()))
            pt, dr = boot_silhouette(P, tissue, cA, WA)
            d["tissue_silhouette"] = entry(pt, dr)
            add("A", g, name, "tissue silhouette (cosine, 50 PCs)", "none", d["tissue_silhouette"], nA, ndA)
            pt, dr = boot_energy(P, tissue, cA, WA)
            d["tissue_energy_distance"] = entry(pt, dr)
            add("A", g, name, "tissue energy distance (50 PCs)", "none", d["tissue_energy_distance"], nA, ndA)
            for k_old, k_new in (("tissue_knn_bacc", "tissue_knn_bacc|stratified5"), ("sex_knn_bacc", "sex_knn_bacc|stratified5"),
                                 ("tissue_silhouette", "tissue_silhouette"), ("tissue_energy_distance", "tissue_energy_distance")):
                res["checks"][f"A|{g}|{name}|{k_old}_minus_2026-10-01"] = float(d[k_new]["value"] - old[g]["set_a"][name][k_old])
            sa[g][name] = d
            log(f"A {g} {name}: tissue strat {d['tissue_knn_bacc|stratified5']['value']:.4f} lodo {d['tissue_knn_bacc|lodo']['value']:.4f} "
                f"sex strat {d['sex_knn_bacc|stratified5']['value']:.4f} lodo {d['sex_knn_bacc|lodo']['value']:.4f} sil {d['tissue_silhouette']['value']:.4f}")
    res["set_a"] = sa

    # primary metric
    def paired_bacc(a, b, g, target, scheme, codes, W):
        y = tissue if target == "tissue" else sex
        ok = pd.notna(y)
        pa, da = boot_bacc(y[ok], oof[(g, a, target, scheme)][ok], codes[ok], W)
        pb, db = boot_bacc(y[ok], oof[(g, b, target, scheme)][ok], codes[ok], W)
        return {"latent_scores": pa, "raw": pb, "difference": pa - pb, "ci95": ci(da - db)}
    prim = paired_bacc("factor_scores", "raw", "natural", "tissue", "lodo", cA, WA)
    prim["ci95_unstratified_dataset_bootstrap"] = paired_bacc("factor_scores", "raw", "natural", "tissue", "lodo", cA_u, WA_u)["ci95"]
    prim.update(n_profiles=int(nA), n_datasets=int(ndA))
    res["primary"] = prim
    log(f"PRIMARY {prim}")
    res["paired_differences_set_a"] = {f"{g}|{t}|{s}|{a} minus raw": paired_bacc(a, "raw", g, t, s, cA, WA)
                                       for g in ("natural", "standardized") for t in ("tissue", "sex") for s in ("stratified5", "lodo")
                                       for a in ("factor_scores", "mgpt2_multitask")}
    # per-tissue recall under leave-one-dataset-out (where the misclassified profiles go)
    res["set_a_lodo_tissue_recall"] = {name: {t: float(np.mean(oof[("natural", name, "tissue", "lodo")][tissue == t] == t)) for t in sorted(set(tissue))}
                                       for name in REPS_A}

    # (ii) clustering, natural geometry
    cl = {}
    ref_ari, ref_nmi = boot_ari_nmi(tissue, ds, cA, WA, distinct_copies_b=True)
    res["set_a_tissue_labels_vs_dataset"] = {"ari": entry(*ref_ari), "nmi": entry(*ref_nmi)}
    for name in REPS_A:
        P = pcs(reps[name], "natural")
        labs = {f"leiden_{r}": v for r, v in leiden_grid(P).items()}
        labs["kmeans_6"] = kmeans_labels(P, 6)
        d = {}
        for key, lv in labs.items():
            e = {"n_clusters": int(len(set(lv)))}
            for tgt, y in (("tissue", tissue), ("dataset", ds)):
                (pa, da), (pn, dn) = boot_ari_nmi(lv, y, cA, WA, distinct_copies_b=(tgt == "dataset"))
                e[f"ari_{tgt}"], e[f"nmi_{tgt}"] = entry(pa, da), entry(pn, dn)
                add("A", "natural", name, f"{key} ARI vs {tgt}", "none", e[f"ari_{tgt}"], nA, ndA, f"{e['n_clusters']} clusters")
                add("A", "natural", name, f"{key} NMI vs {tgt}", "none", e[f"nmi_{tgt}"], nA, ndA, f"{e['n_clusters']} clusters")
            d[key] = e
        cl[name] = d
        log(f"A clustering {name}: " + ", ".join(f"{k} n={v['n_clusters']} ARI-t {v['ari_tissue']['value']:.3f} ARI-d {v['ari_dataset']['value']:.3f}" for k, v in d.items()))
    res["set_a_clustering"] = cl

    # (iv) cross-tissue neighbours
    frac = {}
    for name in ("raw", "factor_scores", "mgpt2_multitask"):
        S = cos_sim(geom(reps[name], "natural"))
        np.fill_diagonal(S, -np.inf)                       # self excluded
        nb = np.argsort(-S, axis=1, kind="stable")[:, :10]
        frac[name] = (tissue[nb] != tissue[:, None]).mean(1)
    ct = meta[["row", "dataset", "tissue"]].copy()
    ct["cross_tissue_fraction_raw"], ct["cross_tissue_fraction_latent"], ct["cross_tissue_fraction_embedding"] = frac["raw"], frac["factor_scores"], frac["mgpt2_multitask"]
    mixed_raw, mixed_lat = frac["raw"] >= 0.5, frac["factor_scores"] >= 0.5
    disc = mixed_raw != mixed_lat
    ct["discordant"] = disc
    ct.to_csv(EVAL / "si_sample_representations_cross_tissue_2026-10-18.csv", index=False)
    rho = spearmanr(frac["raw"], frac["factor_scores"])
    res["set_a_cross_tissue_neighbours"] = {
        "definition": "fraction of the 10 nearest neighbours (cosine, natural geometry, self excluded) from another tissue; "
                      "mixed = at least 5 of 10; discordant = mixed in exactly one of raw beta values and latent scores",
        "spearman_raw_vs_latent": {"rho": float(rho.statistic), "p": float(rho.pvalue), "n": int(nA)},
        "spearman_raw_vs_embedding": float(spearmanr(frac["raw"], frac["mgpt2_multitask"]).statistic),
        "mean_fraction": {k: float(v.mean()) for k, v in frac.items()},
        "profiles_with_any_cross_tissue_neighbour": {k: int((v > 0).sum()) for k, v in frac.items()},
        "profiles_mixed": {"raw": int(mixed_raw.sum()), "latent_scores": int(mixed_lat.sum()), "embedding": int((frac["mgpt2_multitask"] >= 0.5).sum())},
        "discordant_total": int(disc.sum()),
        "discordant_mixed_in_raw_only": int((mixed_raw & ~mixed_lat).sum()),
        "discordant_mixed_in_latent_only": int((mixed_lat & ~mixed_raw).sum()),
        "discordant_by_dataset": ct[disc].groupby(["dataset", "tissue"]).size().rename("n").reset_index().to_dict("records"),
        "mixed_by_dataset_raw": ct[mixed_raw].groupby(["dataset", "tissue"]).size().rename("n").reset_index().to_dict("records"),
        "mixed_by_dataset_latent": ct[mixed_lat].groupby(["dataset", "tissue"]).size().rename("n").reset_index().to_dict("records")}
    log(f"cross-tissue {res['set_a_cross_tissue_neighbours']['spearman_raw_vs_latent']} discordant {disc.sum()}")

    # ======================================== set B ========================================
    reps_b, mb = sets["B"]
    batch, sex_b, age = mb.dataset.to_numpy(), mb.sex.to_numpy(), mb.age.to_numpy()
    nB, ndB = len(mb), mb.dataset.nunique()
    cB, WB = boot_weights(batch)
    raw_b = reps_b["raw"]
    # design table and estimability
    des = mb.groupby("dataset").agg(n=("sex", "size"), male=("sex", lambda s: int((s == "M").sum())), female=("sex", lambda s: int((s == "F").sum())),
                                    age_min=("age", "min"), age_q1=("age", lambda a: float(np.percentile(a, 25))), age_median=("age", "median"),
                                    age_q3=("age", lambda a: float(np.percentile(a, 75))), age_max=("age", "max"), platform=("gpl", "first")).reset_index()
    des.to_csv(EVAL / "si_sample_representations_design_setB_2026-10-18.csv", index=False)
    lv = np.unique(batch)
    Bd = (batch[:, None] == lv[None, :]).astype(float)
    sexm = (sex_b == "M").astype(float)
    res["set_b_design"] = {"table": des.to_dict("records"),
                           "datasets_with_both_sexes": int(((des.male > 0) & (des.female > 0)).sum()),
                           "datasets_with_age_variation": int((des.age_max > des.age_min).sum()),
                           "rank_batch_plus_sex": [int(np.linalg.matrix_rank(np.hstack([Bd, sexm[:, None]]))), int(Bd.shape[1] + 1)],
                           "rank_batch_plus_sex_age": [int(np.linalg.matrix_rank(np.hstack([Bd, sexm[:, None], age[:, None]]))), int(Bd.shape[1] + 2)],
                           "age_r2_explained_by_dataset": float(1 - ((age - pd.Series(age).groupby(batch).transform("mean").to_numpy()) ** 2).sum() / ((age - age.mean()) ** 2).sum()),
                           "sex_male_fraction_range": [float((des.male / des.n).min()), float((des.male / des.n).max())]}
    res["set_b_design"]["datasets_with_constant_filled_cpgs"] = sorted(lv[[int((raw_b[batch == b_].max(0) == raw_b[batch == b_].min(0)).sum()) > 1000 for b_ in lv]].tolist())
    res["set_b_design"]["sex_estimable"] = res["set_b_design"]["rank_batch_plus_sex"][0] == res["set_b_design"]["rank_batch_plus_sex"][1]
    res["set_b_design"]["sex_age_estimable"] = res["set_b_design"]["rank_batch_plus_sex_age"][0] == res["set_b_design"]["rank_batch_plus_sex_age"][1]
    log(f"set B design: {res['set_b_design']['rank_batch_plus_sex']} {res['set_b_design']['rank_batch_plus_sex_age']} age R2 by dataset {res['set_b_design']['age_r2_explained_by_dataset']:.3f}")

    # ComBat in-sample (R sva) and port check
    cov_sex = pd.DataFrame({"male": sexm})
    cov_sex_age = pd.DataFrame({"male": sexm, "age": age})
    combat_in = {}
    port_check = {}
    for key, cv_ in (("combat", None), ("combat_sex", cov_sex), ("combat_sex_age", cov_sex_age)):
        Xr = combat_r(raw_b, batch, cv_)
        Xp, prm_ = combat_fit(raw_b, batch, None if cv_ is None else cv_.to_numpy())
        port_check[key] = {"max_abs_diff_port_vs_R": float(np.abs(Xr - Xp).max()), "mean_abs_diff": float(np.abs(Xr - Xp).mean()),
                           "cpgs_adjusted": int(prm_["keep"].sum()), "cpgs_total": int(raw_b.shape[1])}
        if key == "combat":
            adjustable = prm_["keep"]
        combat_in[key] = Xr
        log(f"ComBat {key}: port vs R max abs diff {port_check[key]['max_abs_diff_port_vs_R']:.2e}")
    res["checks"]["combat_port_vs_R"] = port_check
    # R binary round trip of the wrapper, and the 2026-10-01 ComBat value reproduced with the transposed write
    Xt = np.arange(12, dtype=np.float64).reshape(3, 4)
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as d_:
        np.ascontiguousarray(Xt).tofile(Path(d_) / "x.bin")
        subprocess.run(["Rscript", "-e", f'x <- matrix(readBin("{d_}/x.bin", "double", n=12), nrow=4, ncol=3); '
                        f'stopifnot(all(x[, 2] == c(4, 5, 6, 7))); writeBin(as.vector(x), "{d_}/y.bin")'], check=True, capture_output=True)
        res["checks"]["r_round_trip_identity"] = bool(np.array_equal(np.fromfile(Path(d_) / "y.bin").reshape(3, 4), Xt))
    legacy = combat_r(raw_b, batch, None, legacy_transposed_write=True)
    lz = geom(legacy)
    res["checks"]["legacy_2026-10-01_combat_reproduced"] = {
        "dataset_knn_bacc_with_transposed_write": float(balanced_accuracy_score(batch.astype(str), knn_oof(lz, batch.astype(str), splits("stratified5", batch, batch)).astype(str))),
        "value_in_representation_benchmark_2026-10-01": old["natural"]["set_b"]["combat"]["batch_knn_bacc"],
        "dataset_knn_bacc_correct_write": None}
    del legacy, lz
    def harmony_on(g):   # Harmony on the first 50 PCs of raw beta values in the geometry, as 2026-10-01
        hz = pcs(raw_b, g)
        ho = harmonypy.run_harmony(hz, pd.DataFrame({"batch": batch}), ["batch"], random_state=0, verbose=False)
        zc = np.asarray(ho.Z_corr)
        return zc.T if zc.shape[0] == hz.shape[1] and zc.shape[1] == hz.shape[0] else zc

    sb = {}
    ages_oof = {}
    for g in ("natural", "standardized"):
        reps_b_all = dict(reps_b, combat=combat_in["combat"], combat_sex=combat_in["combat_sex"], combat_sex_age=combat_in["combat_sex_age"],
                          harmony_pca50=harmony_on(g))
        # sva leaves CpGs with no variance within a dataset unadjusted; in set B these are the CpGs not measured in the
        # 27K-array dataset (filled with training means). Sensitivity: raw and ComBat restricted to the adjustable CpGs.
        reps_b_all["raw|combat_adjustable_cpgs"] = raw_b[:, adjustable]
        for k_ in ("combat", "combat_sex", "combat_sex_age"):
            reps_b_all[f"{k_}|combat_adjustable_cpgs"] = combat_in[k_][:, adjustable]
        sb[g] = {}
        for name, Z in reps_b_all.items():
            Sg = cos_sim(geom(Z, g))
            P = Z if name.endswith("pca50") else pcs(Z, g)
            d = {}
            note = ("in-sample ComBat with sex protected: sex metrics are an upper bound" if name == "combat_sex" else
                    "in-sample ComBat with sex and age protected: sex and age metrics are upper bounds" if name == "combat_sex_age" else "")
            for target, y in (("dataset", batch), ("sex", sex_b)):
                for scheme in (("stratified5",) if target == "dataset" else ("stratified5", "lodo")):
                    pr = knn_oof(None, y.astype(str), splits(scheme, y.astype(str), batch), S=Sg)
                    pt, dr = boot_bacc(y, pr, cB, WB)
                    d[f"{target}_knn_bacc|{scheme}"] = entry(pt, dr)
                    add("B", g, name, f"{target} 10-NN balanced accuracy", scheme, d[f"{target}_knn_bacc|{scheme}"], nB, ndB, note)
            for scheme in ("kfold5", "lodo"):
                sp = splits(scheme, batch, batch)
                pr = ridge_oof(P, age, sp)
                ages_oof[(g, name, "ridge", scheme)] = pr
                pt, dr = boot_r2(age, pr, cB, WB)
                d[f"age_ridge_r2|{scheme}"] = entry(pt, dr)
                add("B", g, name, "age ridge R2 (50 PCs)", scheme, d[f"age_ridge_r2|{scheme}"], nB, ndB, note)
                pk = knn_reg_oof(None, age, sp, S=Sg)
                pt, dr = boot_median(np.abs(pk - age), cB, WB)
                d[f"age_knn_medae|{scheme}"] = entry(pt, dr)
                add("B", g, name, "age 10-NN MedAE (years)", scheme, d[f"age_knn_medae|{scheme}"], nB, ndB, note)
            il = compute_lisi(P, pd.DataFrame({"batch": batch}), ["batch"])[:, 0]
            pt, dr = boot_median(il, cB, WB)
            d["batch_ilisi"] = entry(pt, dr)
            add("B", g, name, "dataset iLISI (median)", "none", d["batch_ilisi"], nB, ndB, note)
            pt, dr = boot_silhouette(P, batch, cB, WB)
            d["batch_silhouette"] = entry(pt, dr)
            add("B", g, name, "dataset silhouette (cosine, 50 PCs)", "none", d["batch_silhouette"], nB, ndB, note)
            if name in old[g]["set_b"]:
                for k_old, k_new in (("batch_knn_bacc", "dataset_knn_bacc|stratified5"), ("sex_knn_bacc", "sex_knn_bacc|stratified5"),
                                     ("age_r2_ridge", "age_ridge_r2|kfold5"), ("batch_ilisi", "batch_ilisi"), ("batch_silhouette", "batch_silhouette")):
                    res["checks"][f"B|{g}|{name}|{k_old}_minus_2026-10-01"] = float(d[k_new]["value"] - old[g]["set_b"][name][k_old])
            sb[g][name] = d
            log(f"B {g} {name}: dataset {d['dataset_knn_bacc|stratified5']['value']:.3f} sex {d['sex_knn_bacc|stratified5']['value']:.3f}/"
                f"{d['sex_knn_bacc|lodo']['value']:.3f} age R2 {d['age_ridge_r2|kfold5']['value']:.3f}/{d['age_ridge_r2|lodo']['value']:.3f}")
    res["set_b"] = sb
    res["checks"]["legacy_2026-10-01_combat_reproduced"]["dataset_knn_bacc_correct_write"] = sb["natural"]["combat"]["dataset_knn_bacc|stratified5"]["value"]

    # ---- corrected ComBat points for Fig. 3d (same metrics and protocol as Fig. 3d) ----
    def pick(name):
        v = sb["natural"][name]
        return {"dataset_knn_bacc": v["dataset_knn_bacc|stratified5"], "age_ridge_r2": v["age_ridge_r2|kfold5"],
                "sex_knn_bacc": v["sex_knn_bacc|stratified5"]}
    const_per_ds = {b_: int((raw_b[batch == b_].max(0) == raw_b[batch == b_].min(0)).sum()) for b_ in lv}
    sd_low = raw_b.std(0, ddof=1) <= 1e-6
    res["combat_corrected_for_fig3d"] = {
        "set": "B (949 whole-blood profiles from 10 test datasets)", "n_profiles": int(nB), "n_datasets": int(ndB),
        "points": {"ComBat, all 49,156 CpGs (correct orientation)": pick("combat"),
                   "ComBat, the CpGs it adjusts": pick("combat|combat_adjustable_cpgs"),
                   "Raw beta values, the same CpGs": pick("raw|combat_adjustable_cpgs"),
                   "Raw beta values, all CpGs (recomputed)": pick("raw"),
                   "Latent scores (recomputed)": pick("factor_scores"),
                   "Embedding, multitask (recomputed)": pick("mgpt2_multitask"),
                   "Harmony on raw PCs (recomputed)": pick("harmony_pca50")},
        "v13_values_representation_benchmark_2026-10-01": {
            k: {"dataset_knn_bacc": old["natural"]["set_b"][k]["batch_knn_bacc"], "age_ridge_r2": old["natural"]["set_b"][k]["age_r2_ridge"],
                "sex_knn_bacc": old["natural"]["set_b"][k]["sex_knn_bacc"]}
            for k in ("raw", "factor_scores", "mgpt2_multitask", "combat", "harmony_pca50")},
        "v13_combat_values_invalid_reason": "benchmark_representations_2026-10-01.py wrote X.T in C order while R read it column-major as a "
                                            "CpG x profile matrix, so sva received a scrambled matrix; the old value (dataset 10-NN 0.0776) is "
                                            "reproduced exactly with that write (checks.legacy_2026-10-01_combat_reproduced)",
        "cpg_counts": {"total": int(raw_b.shape[1]), "adjusted_by_combat": int(adjustable.sum()),
                       "left_unadjusted": int((~adjustable).sum()),
                       "sd_at_most_1e-6_over_all_profiles (wrapper pre-filter)": int(sd_low.sum()),
                       "constant_within_dataset_by_dataset": const_per_ds,
                       "constant_within_at_least_one_dataset": int(np.any([raw_b[batch == b_].max(0) == raw_b[batch == b_].min(0) for b_ in lv], 0).sum())},
        "reason_unadjusted": ("sva::ComBat does not adjust a CpG whose values have zero variance within any batch. Missing CpGs are filled with "
                              "the training mean before ComBat, so every CpG not measured on a dataset's array is constant within that "
                              "dataset: the 27K-array dataset GSE20067 and the four EPIC datasets (GSE125367, GSE137594, GSE158063, "
                              "GSE201638) carry such CpGs, and their union is left unadjusted in every dataset."),
        "protocol": ("As Fig. 3d (representation_benchmark_2026-10-01.json, natural geometry): each representation centred, not rescaled. "
                     "Dataset and sex: 10-nearest-neighbour balanced accuracy, cosine distance on all dimensions (Harmony: its 50 corrected "
                     "components), stratified 5-fold CV (seed 20261001). Age: ridge regression (RidgeCV, alphas 10^-2 to 10^4) on the first 50 "
                     "principal components (fitted on all 949 profiles; Harmony: its corrected components), 5-fold KFold shuffled with seed 0, "
                     "R^2 of the pooled out-of-fold predictions against metadata age. ComBat: R sva 3.56.0 ComBat, batch = dataset, no covariates "
                     "(mod = NULL), fitted on all 949 profiles on training-mean-filled beta values, CpGs with SD <= 1e-6 not passed to ComBat. "
                     "Harmony: harmonypy 2.0.2 on the first 50 PCs of raw beta values, batch = dataset, seed 0. Principal components here are "
                     "exact (eigendecomposition); the 2026-10-01 script used randomized PCA, which moves raw and Harmony values by at most "
                     "0.012 (checks). The Fig. 3d point 'Harmony on latent-score PCs' comes from representation_benchmark_missing_2026-10-02.json "
                     "and does not involve ComBat."),
        "intervals": "95% percentile intervals from resampling the 10 datasets (2,000 draws, seed 20261018), cross-validated predictions held fixed",
    }

    # set B clustering against dataset
    clb = {}
    reps_b_all["harmony_pca50"] = harmony_on("natural")
    for name, Z in reps_b_all.items():
        P = Z if name.endswith("pca50") else pcs(Z)
        labs = {f"leiden_{r}": v for r, v in leiden_grid(P).items()}
        labs["kmeans_10"] = kmeans_labels(P, 10)
        d = {}
        for key, lv_ in labs.items():
            (pa, da), (pn, dn) = boot_ari_nmi(lv_, batch, cB, WB, distinct_copies_b=True)
            d[key] = {"n_clusters": int(len(set(lv_))), "ari_dataset": entry(pa, da), "nmi_dataset": entry(pn, dn)}
            add("B", "natural", name, f"{key} ARI vs dataset", "none", d[key]["ari_dataset"], nB, ndB, f"{d[key]['n_clusters']} clusters")
            add("B", "natural", name, f"{key} NMI vs dataset", "none", d[key]["nmi_dataset"], nB, ndB, f"{d[key]['n_clusters']} clusters")
        clb[name] = d
    res["set_b_clustering"] = clb

    # (v) within-fold ComBat
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(nB), batch))
    wf = {}
    raw_adj = raw_b[:, adjustable]
    within_reps = {"raw": raw_b, "factor_scores": reps_b["factor_scores"], "mgpt2_multitask": reps_b["mgpt2_multitask"],
                   "raw|combat_adjustable_cpgs": raw_adj}
    variants = {"raw": None, "factor_scores": None, "mgpt2_multitask": None, "raw|combat_adjustable_cpgs": None,
                "combat": "none", "combat_sex": "sex", "combat_sex_age": "sex_age",
                "combat|combat_adjustable_cpgs": "none", "combat_sex|combat_adjustable_cpgs": "sex", "combat_sex_age|combat_adjustable_cpgs": "sex_age"}
    for name, var in variants.items():
        src = raw_adj if name.endswith("|combat_adjustable_cpgs") else raw_b
        pred_age = np.empty(nB); pred_sex = np.empty(nB, dtype=object); pred_ds = np.empty(nB, dtype=object)
        assembled = np.empty((nB, 50))
        for tr, te in folds:
            if var is None:
                Xtr, Xte = np.asarray(within_reps[name], np.float64)[tr], np.asarray(within_reps[name], np.float64)[te]
            else:
                C = None if var == "none" else (np.c_[sexm] if var == "sex" else np.c_[sexm, age])
                Xtr, prm = combat_fit(src[tr], batch[tr], None if C is None else C[tr])
                Xte = combat_apply(src[te], batch[te], prm)
            mu = Xtr.mean(0)
            V = pca_fit(Xtr - mu, 50)
            Ptr, Pte = pca_transform(Xtr - mu, V), pca_transform(Xte - mu, V)
            assembled[te] = Pte
            pred_age[te] = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Ptr, age[tr]).predict(Pte)
            S_te_tr = cos_sim(Xte - mu, Xtr - mu)
            pred_sex[te] = _knn_vote(S_te_tr, sex_b[tr])
            pred_ds[te] = _knn_vote(S_te_tr, batch[tr])
        d = {}
        d["dataset_knn_bacc"] = entry(*boot_bacc(batch, pred_ds, cB, WB))
        d["sex_knn_bacc"] = entry(*boot_bacc(sex_b, pred_sex, cB, WB))
        d["age_ridge_r2"] = entry(*boot_r2(age, pred_age, cB, WB))
        d["batch_silhouette_heldout_folds"] = entry(*boot_silhouette(assembled, batch, cB, WB))
        il = compute_lisi(assembled, pd.DataFrame({"batch": batch}), ["batch"])[:, 0]
        d["batch_ilisi_heldout_folds"] = entry(*boot_median(il, cB, WB))
        wf[name] = d
        for k, v in d.items():
            add("B", "natural", name, f"within-fold {k}", "stratified5 by dataset (ComBat and PCA fitted on training folds)", v, nB, ndB,
                "ComBat fitted within folds" if var else "")
        log(f"within-fold {name}: dataset {d['dataset_knn_bacc']['value']:.3f} sex {d['sex_knn_bacc']['value']:.3f} age R2 {d['age_ridge_r2']['value']:.3f}")
    res["set_b_combat_within_fold"] = wf

    # (vi) dataset identifiability after removing age and sex, and in an age-overlap window
    resid = {}
    Xcov = np.c_[np.ones(nB), sexm, age]
    win = (age >= 25) & (age <= 45)
    keep_ds = pd.Series(batch[win]).value_counts().loc[lambda s: s >= 20].index
    wmask = win & np.isin(batch, keep_ds)
    cW, WW = boot_weights(batch[wmask])
    for name in ("raw", "factor_scores", "mgpt2_multitask"):
        Zg = geom(reps_b[name])
        Tz, Tc = _t(Zg), _t(Xcov)
        R = (Tz - Tc @ torch.linalg.lstsq(Tc, Tz).solution).cpu().numpy()
        pr = knn_oof(R, batch.astype(str), splits("stratified5", batch.astype(str), batch))
        e1 = entry(*boot_bacc(batch, pr, cB, WB))
        Zw = geom(reps_b[name][wmask])
        prw = knn_oof(Zw, batch[wmask].astype(str), splits("stratified5", batch[wmask].astype(str), batch[wmask]))
        e2 = entry(*boot_bacc(batch[wmask], prw, cW, WW))
        resid[name] = {"dataset_knn_bacc_age_sex_residualised": e1, "dataset_knn_bacc_age_25_45": e2}
        add("B", "natural", name, "dataset 10-NN balanced accuracy, age and sex regressed out", "stratified5", e1, nB, ndB)
        add("B", "natural", name, "dataset 10-NN balanced accuracy, ages 25-45 only", "stratified5", e2, int(wmask.sum()), int(len(keep_ds)))
    res["set_b_dataset_identifiability_without_age_sex"] = {"age_window": [25, 45], "datasets_in_window": sorted(keep_ds.tolist()),
                                                            "n_profiles_in_window": int(wmask.sum()), **resid}

    # UMAP coordinates of set B for the figure
    import umap
    um = {}
    for k, X in (("latent", reps_b["factor_scores"]), ("raw", raw_b)):
        P = pcs(X)
        um[k] = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(P)
    np.savez(EVAL / "si_sample_representations_umap_setB_2026-10-18.npz", rows=mb.row.to_numpy(), dataset=batch.astype(str), age=age, sex=sex_b.astype(str), **um)

    res["elapsed_s"] = time.time() - t0
    write_json(EVAL / "si_sample_representations_2026-10-18.json", res)
    pd.DataFrame(rows_csv).to_csv(EVAL / "si_sample_representations_2026-10-18.csv", index=False)
    log(f"done in {res['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
