#!/usr/bin/env python3
"""Neighbouring CpGs (18 Oct 2026; Supplementary Information).

MethylGPT uses no genomic position or sequence: each CpG enters through its own learned vectors and
input-layer loadings, indexed by vocabulary order (alphabetical probe ID). This script measures how
close vocabulary CpGs are to one another, how strongly neighbouring CpGs co-vary and whether the input
layer reproduces that co-variation, and how much reconstruction relies on a target's nearest neighbour.

Pre-specified primary metric (fixed before the first run)
  For target CpGs whose nearest vocabulary CpG on the same chromosome lies within 1 kb (<= 1,000 bp):
  masked MSE of MethylGPT when each target is masked together with its nearest neighbour, minus the
  masked MSE when the same targets are masked together with the same number of randomly chosen other
  observed CpGs (paired: same profiles, same targets, same mask size per profile). Profiles: the 2,000
  held-out profiles (107 test-split datasets) of the reconstruction benchmark. 95% interval: 2,000
  dataset-cluster bootstrap replicates (datasets resampled with replacement; seed 20261018).

Design
  (i) Distance from each vocabulary CpG to its nearest vocabulary CpG on the same chromosome
      (hg19 positions from artifacts/cpg_annotation_2026-10-01.parquet).
  (ii) Pairs of autosomal vocabulary CpGs in distance bins (<50 bp, 50-200 bp, 200 bp-1 kb, 1-10 kb,
      10-100 kb, 100 kb-1 Mb; all pairs in a bin, or 5,000 drawn at random) plus 5,000 pairs on
      different autosomes. For each pair: Pearson correlation across profiles of beta values centred
      within tissue (pairwise-complete profiles, >= 100 required), and the correlation of the
      input-layer reconstructions (posterior reconstruction from each profile's observed CpGs, centred
      within tissue) over the same profiles. Two profile sets: 20,000 training-split profiles drawn at
      random from the corpus, and 20,000 training-split whole-blood profiles. Sensitivity: centring
      within dataset. Medians per bin with 1,000-replicate dataset-cluster bootstrap intervals.
      Sex chromosomes are left out of (ii) because sex differences correlate X CpGs at any distance.
  (iii) Neighbour masking on the 2,000 held-out profiles. Targets: 10% of observed CpGs whose nearest
      vocabulary neighbour is observed (seeded); targets whose nearest neighbour is itself a target
      are dropped so that each target's neighbour is visible in the random condition. Neighbour
      condition: targets plus their nearest neighbours masked. Random condition: targets plus the same
      number of other observed CpGs, drawn from CpGs that are neither targets nor target neighbours.
      Error is scored on the targets only, for MethylGPT and the input layer alone.
      Secondary design: targets (5% of observed CpGs with at least one observed vocabulary CpG within
      1 kb; targets with another target within 1 kb dropped) masked with every observed vocabulary CpG
      within 1 kb, against a size-matched random mask.
Secondary metrics: distance quantiles and fractions; per-bin median correlations, IQR, Spearman
  correlation between observed and reconstruction correlations; neighbour-minus-random MSE for all
  targets and by nearest-neighbour distance bin, relative change, input layer, window design.

Outputs (aggregates only)
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.json
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.{nn_distance,pair_correlation,neighbour_mask}.csv
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.pairs.npz   (per-pair correlations)
Run: CUDA_VISIBLE_DEVICES=1 python3 si_neighbouring_cpgs_2026-10-18.py --device cuda:0
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

from methylgpt2_lib import ARTIFACTS, HERE, Corpus, cuda_preinit, log, require, write_json
from methylgpt2_model import load_checkpoint

spec = importlib.util.spec_from_file_location("plat", HERE / "evaluate_heldout_platform_2026-10-01.py")
plat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plat)

OUT = ARTIFACTS / "eval/si_neighbouring_cpgs_2026-10-18"
SEED = 20261018
DIST_BINS = [(1, 50, "<50 bp"), (50, 200, "50-200 bp"), (200, 1000, "200 bp-1 kb"), (1000, 10_000, "1-10 kb"),
             (10_000, 100_000, "10-100 kb"), (100_000, 1_000_000, "100 kb-1 Mb")]
NN_BINS = [(0, 50, "<50 bp"), (50, 200, "50-200 bp"), (200, 1000, "200 bp-1 kb"), (1000, 10_000, "1-10 kb"),
           (10_000, 100_000, "10-100 kb"), (100_000, np.inf, ">=100 kb")]
AUTOSOMES = [str(c) for c in range(1, 23)]
PAIRS_PER_BIN = 5000
MIN_PAIR_N = 100


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def cluster_counts(D, B, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(B, D))
    return np.stack([np.bincount(r, minlength=D) for r in idx]).astype(np.float64)


# ---------------------------------------------------------------- (i) distances
def nearest_neighbours(chr_, pos):
    P = len(pos)
    nn = np.full(P, -1, np.int64)
    dist = np.full(P, np.inf)
    for c in np.unique(chr_):
        idx = np.flatnonzero(chr_ == c)
        if len(idx) < 2:
            continue
        o = idx[np.argsort(pos[idx], kind="stable")]
        p = pos[o]
        dl = np.r_[np.inf, np.diff(p)]
        dr = np.r_[np.diff(p), np.inf]
        left = dl <= dr
        nn[o] = np.where(left, np.r_[-1, o[:-1]], np.r_[o[1:], -1])
        dist[o] = np.minimum(dl, dr)
    return nn, dist


def window_adjacency(chr_, pos, w=1000):
    """Sparse symmetric adjacency of vocabulary CpGs on the same chromosome within w bp."""
    rows, cols = [], []
    for c in np.unique(chr_):
        idx = np.flatnonzero(chr_ == c)
        o = idx[np.argsort(pos[idx], kind="stable")]
        p = pos[o]
        right = np.searchsorted(p, p + w, "right")
        cnt = right - np.arange(len(p)) - 1
        ii = np.repeat(np.arange(len(p)), cnt)
        jj = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(np.arange(len(p)) + 1, cnt)
        rows += [o[ii], o[jj]]
        cols += [o[jj], o[ii]]
    r, c_ = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c_)), shape=(len(pos), len(pos)))


# ---------------------------------------------------------------- (ii) pair correlations
def enumerate_pairs(chr_, pos, lo, hi):
    I, J = [], []
    for c in AUTOSOMES:
        idx = np.flatnonzero(chr_ == c)
        o = idx[np.argsort(pos[idx], kind="stable")]
        p = pos[o]
        left = np.searchsorted(p, p + lo, "left")
        right = np.searchsorted(p, p + hi, "left")
        cnt = right - left
        ii = np.repeat(np.arange(len(p)), cnt)
        jj = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(left, cnt)
        I.append(o[ii]); J.append(o[jj])
    return np.concatenate(I), np.concatenate(J)


def build_pairs(chr_, pos, rng):
    out = []
    auto = np.flatnonzero(np.isin(chr_, AUTOSOMES))
    for k, (lo, hi, lab) in enumerate(DIST_BINS):
        I, J = enumerate_pairs(chr_, pos, lo, hi)
        n_all = len(I)
        if n_all > PAIRS_PER_BIN:
            s = np.sort(rng.choice(n_all, PAIRS_PER_BIN, replace=False))
            I, J = I[s], J[s]
        out.append(pd.DataFrame({"i": I, "j": J, "bin": k, "label": lab, "n_pairs_in_bin": n_all,
                                 "distance": np.abs(pos[I] - pos[J])}))
    I, J = [], []
    while len(I) < PAIRS_PER_BIN:
        a, b = rng.choice(auto, 4 * PAIRS_PER_BIN), rng.choice(auto, 4 * PAIRS_PER_BIN)
        keep = chr_[a] != chr_[b]
        I += list(a[keep]); J += list(b[keep])
    out.append(pd.DataFrame({"i": I[:PAIRS_PER_BIN], "j": J[:PAIRS_PER_BIN], "bin": len(DIST_BINS),
                             "label": "different chromosomes", "n_pairs_in_bin": -1, "distance": np.nan}))
    return pd.concat(out, ignore_index=True)


@torch.no_grad()
def reconstruct(core, X, batch=32):
    R = torch.empty_like(X)
    obs = torch.isfinite(X)
    for i in range(0, len(X), batch):
        s = core.posterior(X[i:i + batch], obs[i:i + batch].float())
        R[i:i + batch] = core.predict(s)
    return R


@torch.no_grad()
def group_means(X, gidx, G):
    ok = torch.isfinite(X)
    s = torch.zeros(G, X.shape[1], device=X.device, dtype=torch.float64)
    n = torch.zeros(G, X.shape[1], device=X.device, dtype=torch.float64)
    for i in range(0, len(X), 2048):
        s.index_add_(0, gidx[i:i + 2048], torch.where(ok[i:i + 2048], X[i:i + 2048], 0).double())
        n.index_add_(0, gidx[i:i + 2048], ok[i:i + 2048].double())
    return (s / n.clamp_min(1)).float()


@torch.no_grad()
def pair_stats(X, R, cidx, Mx, Mr, ds_idx, D, I, J, batch=1024):
    """Per-dataset sufficient statistics (11, D, n_pairs) of centred values over pairwise-complete profiles."""
    out = np.zeros((11, D, len(I)))
    for b0 in range(0, len(I), batch):
        i = torch.as_tensor(I[b0:b0 + batch], device=X.device)
        j = torch.as_tensor(J[b0:b0 + batch], device=X.device)
        A = X[:, i] - Mx[:, i][cidx]
        Bv = X[:, j] - Mx[:, j][cidx]
        ok = torch.isfinite(A) & torch.isfinite(Bv)
        A = torch.where(ok, A, 0); Bv = torch.where(ok, Bv, 0)
        Ra = torch.where(ok, R[:, i] - Mr[:, i][cidx], 0)
        Rb = torch.where(ok, R[:, j] - Mr[:, j][cidx], 0)
        for k, s in enumerate((ok.float(), A, Bv, A * A, Bv * Bv, A * Bv, Ra, Rb, Ra * Ra, Rb * Rb, Ra * Rb)):
            acc = torch.zeros(D, s.shape[1], device=X.device, dtype=torch.float64)
            acc.index_add_(0, ds_idx, s.double())
            out[k, :, b0:b0 + len(i)] = acc.cpu().numpy()
    return out


def corr_from(st):
    n, sa, sb, saa, sbb, sab, ra, rb, raa, rbb, rab = st
    with np.errstate(invalid="ignore", divide="ignore"):
        r_obs = (sab - sa * sb / n) / np.sqrt((saa - sa ** 2 / n) * (sbb - sb ** 2 / n))
        r_rec = (rab - ra * rb / n) / np.sqrt((raa - ra ** 2 / n) * (rbb - rb ** 2 / n))
    return n, r_obs, r_rec


def correlation_set(name, corpus, rows, core, pairs, dev, B=1000):
    prof = corpus.profiles.loc[rows]
    X = torch.from_numpy(corpus.get(rows)).to(dev)
    t0 = time.time()
    R = reconstruct(core, X)
    log(f"{name}: {len(rows)} profiles reconstructed in {time.time() - t0:.0f} s")
    ds_codes, ds_uni = pd.factorize(prof.group)
    D = len(ds_uni)
    ds_idx = torch.as_tensor(ds_codes, device=dev)
    tis_codes, tis_uni = pd.factorize(prof.tissue.fillna("NA"))
    info = {"n_profiles": int(len(rows)), "n_datasets": int(D),
            "tissues": prof.tissue.fillna("NA").value_counts().to_dict()}
    I, J = pairs.i.to_numpy(), pairs.j.to_numpy()
    C = cluster_counts(D, B)
    results = {}
    per_pair = {}
    for centring, codes, G in (("tissue", tis_codes, len(tis_uni)), ("dataset", ds_codes, D)):
        cidx = torch.as_tensor(codes, device=dev)
        Mx, Mr = group_means(X, cidx, G), group_means(R, cidx, G)
        st = pair_stats(X, R, cidx, Mx, Mr, ds_idx, D, I, J)
        n, r_obs, r_rec = corr_from(st.sum(1))
        valid = (n >= MIN_PAIR_N) & np.isfinite(r_obs) & np.isfinite(r_rec)
        per_pair[centring] = (n, r_obs, r_rec, valid)
        # bootstrap medians per bin
        bins = pairs.bin.to_numpy()
        nbins = bins.max() + 1
        med_bs = {k: np.zeros((B, nbins)) for k in ("obs", "rec", "diff")}
        for b0 in range(0, B, 50):
            Cb = C[b0:b0 + 50]
            agg = np.stack([Cb @ st[k] for k in range(11)])
            _, ro, rr = corr_from(agg)
            ro = np.where(valid, ro, np.nan); rr = np.where(valid, rr, np.nan)
            for kb in range(nbins):
                m = bins == kb
                med_bs["obs"][b0:b0 + 50, kb] = np.nanmedian(ro[:, m], 1)
                med_bs["rec"][b0:b0 + 50, kb] = np.nanmedian(rr[:, m], 1)
                med_bs["diff"][b0:b0 + 50, kb] = np.nanmedian(rr[:, m] - ro[:, m], 1)
        rows_out = []
        for kb in range(nbins):
            m = (bins == kb) & valid
            lab = pairs.label[bins == kb].iloc[0]
            ro, rr = r_obs[m], r_rec[m]
            rows_out.append({"set": name, "centring": centring, "bin": int(kb), "label": lab,
                             "n_pairs_in_bin": int(pairs.n_pairs_in_bin[bins == kb].iloc[0]),
                             "n_pairs_used": int(m.sum()), "n_pairs_dropped": int(((bins == kb) & ~valid).sum()),
                             "median_profiles_per_pair": float(np.median(n[m])),
                             "median_r_observed": float(np.median(ro)), "median_r_observed_lo": ci(med_bs["obs"][:, kb])[0],
                             "median_r_observed_hi": ci(med_bs["obs"][:, kb])[1],
                             "q25_r_observed": float(np.percentile(ro, 25)), "q75_r_observed": float(np.percentile(ro, 75)),
                             "median_r_reconstruction": float(np.median(rr)), "median_r_reconstruction_lo": ci(med_bs["rec"][:, kb])[0],
                             "median_r_reconstruction_hi": ci(med_bs["rec"][:, kb])[1],
                             "q25_r_reconstruction": float(np.percentile(rr, 25)), "q75_r_reconstruction": float(np.percentile(rr, 75)),
                             "median_diff_rec_minus_obs": float(np.median(rr - ro)), "median_diff_lo": ci(med_bs["diff"][:, kb])[0],
                             "median_diff_hi": ci(med_bs["diff"][:, kb])[1],
                             "frac_r_observed_gt_0.5": float(np.mean(ro > 0.5)),
                             "spearman_obs_vs_rec": float(spearmanr(ro, rr).correlation)})
        results[centring] = rows_out
        log(f"{name} / {centring}: " + "; ".join(f"{r['label']} obs {r['median_r_observed']:.3f} rec {r['median_r_reconstruction']:.3f}"
                                                for r in rows_out))
    del X, R
    torch.cuda.empty_cache()
    return info, results, per_pair


# ---------------------------------------------------------------- (iii) neighbour masking
@torch.no_grad()
def predict_targets(model, core, X, ctx, tgt):
    mg, base = plat.predict_model(model, core, X, ctx, tgt)
    return np.concatenate(mg).astype(np.float64), np.concatenate(base).astype(np.float64)


def random_partner(obs, excl, sizes, rng):
    R = np.zeros_like(obs)
    for i in range(len(obs)):
        cand = np.flatnonzero(obs[i] & ~excl[i])
        R[i, rng.choice(cand, sizes[i], replace=False)] = True
    return R


def masking_experiment(model, core, X, meta, nn, dist, A1k, rng, dev):
    obs_t = torch.isfinite(X)
    obs = obs_t.cpu().numpy()
    n_prof, P = obs.shape
    has_nn = nn >= 0
    nn_safe = np.where(has_nn, nn, np.arange(P))
    out = {}
    designs = {}
    # nearest-neighbour design
    cand = obs & has_nn[None] & obs[:, nn_safe]
    T = cand & (rng.random(obs.shape) < 0.10)
    n_drawn = int(T.sum())
    T = T & ~T[:, nn_safe]
    Nb = np.zeros_like(T)
    pi, pj = np.nonzero(T)
    Nb[pi, nn_safe[pj]] = True
    assert not (Nb & T).any()
    Rm = random_partner(obs, T | Nb, Nb.sum(1), rng)
    designs["nearest_neighbour"] = (T, Nb, Rm, {"n_targets_drawn": n_drawn, "n_targets_after_dropping_target_neighbours": int(T.sum()),
                                                "n_candidate_cpg_values": int(cand.sum()), "n_observed_values": int(obs.sum())})
    # 1 kb window design
    n_win_obs = np.asarray(A1k @ obs.T.astype(np.float32)).T          # A1k is symmetric
    cand2 = obs & (n_win_obs > 0)
    T2 = cand2 & (rng.random(obs.shape) < 0.05)
    n_drawn2 = int(T2.sum())
    near_t = np.asarray(A1k @ T2.T.astype(np.float32)).T > 0
    T2 = T2 & ~near_t
    W = (np.asarray(A1k @ T2.T.astype(np.float32)).T > 0) & obs & ~T2
    R2 = random_partner(obs, T2 | W, W.sum(1), rng)
    designs["window_1kb"] = (T2, W, R2, {"n_targets_drawn": n_drawn2, "n_targets_after_dropping_close_targets": int(T2.sum()),
                                         "n_candidate_cpg_values": int(cand2.sum()), "n_observed_values": int(obs.sum())})
    ds_codes, ds_uni = pd.factorize(meta.group)
    D = len(ds_uni)
    C = cluster_counts(D, 2000)
    for name, (T_, Nb_, R_, info) in designs.items():
        assert (Nb_.sum(1) == R_.sum(1)).all() and not (R_ & (T_ | Nb_)).any()
        tgt = torch.from_numpy(T_).to(dev)
        res = {}
        for cond, M in (("neighbour", Nb_), ("random", R_)):
            ctx = obs_t & ~tgt & ~torch.from_numpy(M).to(dev)
            res[cond] = predict_targets(model, core, X, ctx, tgt)
            log(f"{name} / {cond}: predicted {int(T_.sum())} targets")
        nz = np.nonzero(T_)
        y = X[tgt].cpu().numpy().astype(np.float64)
        prof, cpg = nz[0], nz[1]
        d_t = dist[cpg]
        ds_t = ds_codes[prof]
        mask_size = (T_ | Nb_).sum(1)
        info.update({"mask_fraction_of_observed_median": float(np.median(mask_size / obs.sum(1))),
                     "extra_masked_per_profile_median": float(np.median(Nb_.sum(1))),
                     "frac_targets_nn_within_1kb": float(np.mean(d_t <= 1000)),
                     "n_targets": int(len(y)), "n_profiles": int(n_prof), "n_datasets": int(D)})
        subsets = {"nn_within_1kb": d_t <= 1000, "all_targets": np.ones(len(y), bool)}
        for lo, hi, lab in NN_BINS:
            subsets[f"nn {lab}"] = (d_t >= lo) & (d_t < hi)
        rows = []
        for sname, sel in subsets.items():
            if sel.sum() == 0:
                continue
            row = {"design": name, "subset": sname, "n_targets": int(sel.sum()),
                   "n_profiles": int(len(np.unique(prof[sel]))), "n_datasets": int(len(np.unique(ds_t[sel])))}
            for model_name, k in (("methylgpt2", 0), ("ppca_core", 1)):
                se_n = (res["neighbour"][k][sel] - y[sel]) ** 2
                se_r = (res["random"][k][sel] - y[sel]) ** 2
                n_d = np.bincount(ds_t[sel], minlength=D).astype(float)
                sn_d = np.bincount(ds_t[sel], se_n, D)
                sr_d = np.bincount(ds_t[sel], se_r, D)
                mse_n, mse_r = sn_d.sum() / n_d.sum(), sr_d.sum() / n_d.sum()
                bn, br, bc = C @ sn_d, C @ sr_d, C @ n_d
                row[f"mse_neighbour_{model_name}"] = float(mse_n)
                row[f"mse_random_{model_name}"] = float(mse_r)
                row[f"diff_{model_name}"] = float(mse_n - mse_r)
                row[f"diff_{model_name}_lo"], row[f"diff_{model_name}_hi"] = ci((bn - br) / bc)
                row[f"rel_change_{model_name}"] = float(mse_n / mse_r - 1)
                row[f"rel_change_{model_name}_lo"], row[f"rel_change_{model_name}_hi"] = ci(bn / br - 1)
                row[f"frac_targets_worse_{model_name}"] = float(np.mean(se_n > se_r))
                if sname == "nn_within_1kb":
                    row[f"per_dataset_diff_{model_name}"] = [float(a) if c > 0 else None for a, c in
                                                             zip((sn_d - sr_d) / np.maximum(n_d, 1), n_d)]
            rows.append(row)
        info["dataset_order"] = list(map(str, ds_uni))
        out[name] = {"info": info, "rows": rows}
        log(f"{name}: " + "; ".join(f"{r['subset']} n={r['n_targets']} dMSE {r['diff_methylgpt2']:.6f} "
                                   f"[{r['diff_methylgpt2_lo']:.6f},{r['diff_methylgpt2_hi']:.6f}]" for r in rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ARTIFACTS / "runs/stopped_cf_dense_2026-09-30/best.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-profiles", type=int, default=20000)
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    rng = np.random.default_rng(SEED)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    chr_ = ann.chr.astype(str).to_numpy()
    pos = ann.pos.to_numpy().astype(np.int64)
    P = len(ann)
    res = {"script": Path(__file__).name,
           "primary_metric": "MethylGPT masked MSE, neighbour-masked minus size-matched random-masked, targets with nearest "
                             "vocabulary CpG within 1 kb, 2,000 held-out profiles, 2,000 dataset-cluster bootstrap replicates",
           "position_used_by_model": False,
           "vocabulary_order_alphabetical": bool((ann.cpg.to_numpy()[:-1] <= ann.cpg.to_numpy()[1:]).all())}

    # (i)
    nn, dist = nearest_neighbours(chr_, pos)
    fin = np.isfinite(dist)
    auto = np.isin(chr_, AUTOSOMES)
    thr = [50, 200, 1000, 10_000, 100_000, 1_000_000]
    res["nn_distance"] = {
        "n_cpgs": int(P), "n_with_neighbour": int(fin.sum()), "n_duplicate_positions": int((dist == 0).sum()),
        "quantiles_bp": {f"q{q}": float(np.percentile(dist[fin], q)) for q in (5, 10, 25, 50, 75, 90, 95)},
        "frac_within": {f"<={t}": float(np.mean(dist[fin] <= t)) for t in thr},
        "frac_within_autosomes": {f"<={t}": float(np.mean(dist[fin & auto] <= t)) for t in thr},
        "frac_within_X": {f"<={t}": float(np.mean(dist[fin & (chr_ == "X")] <= t)) for t in thr},
    }
    edges = np.logspace(0, 8, 81)
    h = np.histogram(dist[fin], bins=edges)[0]
    pd.DataFrame({"lo_bp": edges[:-1], "hi_bp": edges[1:], "n_cpgs": h, "cum_frac": np.cumsum(h) / fin.sum()}).to_csv(
        OUT.with_suffix(".nn_distance.csv"), index=False)
    np.savez_compressed(OUT.with_suffix(".nn.npz"), nn_distance=dist.astype(np.float64), chr=chr_)
    A1k = window_adjacency(chr_, pos, 1000)
    n_win = np.asarray(A1k.sum(1)).ravel()
    res["nn_distance"]["vocabulary_cpgs_within_1kb"] = {"frac_with_any": float(np.mean(n_win > 0)),
                                                        "mean_count": float(n_win.mean()), "max_count": int(n_win.max())}
    log(f"(i) median NN distance {res['nn_distance']['quantiles_bp']['q50']:.0f} bp; within 1 kb "
        f"{res['nn_distance']['frac_within']['<=1000']:.3f}")

    # (ii)
    corpus = Corpus()
    model, core, ck = load_checkpoint(require(Path(a.ckpt)), None, dev)
    pairs = build_pairs(chr_, pos, rng)
    res["pairs"] = pairs.groupby("label", sort=False).agg(n_pairs_sampled=("i", "size"), n_pairs_in_bin=("n_pairs_in_bin", "first")).reset_index().to_dict("records")
    prof = corpus.profiles
    train = prof.index[prof.split == "train"].to_numpy()
    wb = prof.index[(prof.split == "train") & (prof.tissue == "Whole Blood")].to_numpy()
    sets = {"corpus": np.sort(rng.choice(train, a.n_profiles, replace=False)),
            "whole_blood": np.sort(rng.choice(wb, min(a.n_profiles, len(wb)), replace=False))}
    res["correlation"] = {}
    pair_out = {"i": pairs.i.to_numpy(), "j": pairs.j.to_numpy(), "bin": pairs.bin.to_numpy(),
                "distance": pairs.distance.to_numpy()}
    corr_rows = []
    for name, rows in sets.items():
        info, results, per_pair = correlation_set(name, corpus, rows, core, pairs, dev)
        res["correlation"][name] = {"info": info, **results}
        for centring, rr in results.items():
            corr_rows += rr
            n, ro, rc, valid = per_pair[centring]
            pair_out[f"{name}_{centring}_n"] = n.astype(np.float32)
            pair_out[f"{name}_{centring}_r_observed"] = np.where(valid, ro, np.nan).astype(np.float32)
            pair_out[f"{name}_{centring}_r_reconstruction"] = np.where(valid, rc, np.nan).astype(np.float32)
    pd.DataFrame(corr_rows).to_csv(OUT.with_suffix(".pair_correlation.csv"), index=False)
    np.savez_compressed(OUT.with_suffix(".pairs.npz"), **pair_out)

    # (iii)
    trng = np.random.default_rng(20261001)
    test_rows = np.sort(trng.choice(corpus.rows("test"), 2000, replace=False))
    meta = prof.loc[test_rows, ["id", "group", "tissue"]].reset_index(drop=True)
    X = torch.from_numpy(corpus.get(test_rows)).to(dev)
    res["masking"] = masking_experiment(model, core, X, meta, nn, dist, A1k, np.random.default_rng(SEED + 1), dev)
    prim = [r for r in res["masking"]["nearest_neighbour"]["rows"] if r["subset"] == "nn_within_1kb"][0]
    res["primary"] = {"metric": res["primary_metric"], "diff": prim["diff_methylgpt2"],
                      "ci95": [prim["diff_methylgpt2_lo"], prim["diff_methylgpt2_hi"]],
                      "mse_neighbour": prim["mse_neighbour_methylgpt2"], "mse_random": prim["mse_random_methylgpt2"],
                      "n_targets": prim["n_targets"], "n_profiles": prim["n_profiles"], "n_datasets": prim["n_datasets"],
                      "frac_targets_nn_within_1kb": res["masking"]["nearest_neighbour"]["info"]["frac_targets_nn_within_1kb"],
                      "resampled": "datasets (column group), with replacement", "bootstrap_replicates": 2000}
    log(f"PRIMARY neighbour minus random MSE (<=1 kb) {prim['diff_methylgpt2']:.6f} "
        f"[{prim['diff_methylgpt2_lo']:.6f}, {prim['diff_methylgpt2_hi']:.6f}]")
    mrows = [{k: v for k, v in r.items() if not k.startswith("per_dataset")} for d in res["masking"].values() for r in d["rows"]]
    pd.DataFrame(mrows).to_csv(OUT.with_suffix(".neighbour_mask.csv"), index=False)
    res["elapsed_s"] = time.time() - t0
    write_json(OUT.with_suffix(".json"), res)
    log(f"wrote {OUT}.*; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
