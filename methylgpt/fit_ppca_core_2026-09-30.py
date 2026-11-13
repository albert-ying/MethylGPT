#!/usr/bin/env python3
"""Fit the PPCA core of the retrained MethylGPT on all training-split profiles.

Covariance is pairwise-complete (each entry averages over profiles observing both CpGs),
accumulated on the GPU in float64; the top eigenpairs give the loadings V (orthonormal
over all 49,156 CpGs) and variances lam. sigma2 for rank K is (trace - sum lam_1..K)/(P-K).
Then ranks are compared on dev-split profiles (masked-value MSE at 30% masking) so the core
rank is chosen without touching the test split.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, Corpus, PPCACore, log, masked_view, mse_metrics, write_json


@torch.no_grad()
def top_eigs(C, k, oversample=128, iters=20, seed=0):
    """Randomized subspace iteration with Rayleigh-Ritz (cuSOLVER syevd fails at n = 49,156)."""
    g = torch.Generator(device=C.device).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(C.shape[0], k + oversample, device=C.device, generator=g))
    for _ in range(iters):
        Q, _ = torch.linalg.qr(C @ Q)
    CQ = C @ Q
    B = (Q.T @ CQ).double()
    w, U = torch.linalg.eigh(0.5 * (B + B.T))
    order = torch.argsort(w, descending=True)[:k]
    w, U = w[order], U[:, order]
    V = (Q.double() @ U).float()
    m = min(1024, k)
    R = (CQ.double() @ U[:, :m]) - V[:, :m].double() * w[:m]
    rel = float((R.norm(dim=0) / w[:m].abs()).max())
    return w, V, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=1536)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--min-pair", type=int, default=200, help="minimum co-observations for a covariance entry")
    ap.add_argument("--dev-profiles", type=int, default=1000)
    ap.add_argument("--ranks", default="128,256,512,768,1024,1536")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ARTIFACTS / "ppca_core_2026-09-30.npz"))
    ap.add_argument("--group-fold", type=int, default=-1,
                    help="if 0 or 1, fit on training groups with crc32(group) %% 2 == fold (cross-fitting)")
    ap.add_argument("--cov", choices=["pairwise", "mean_filled"], default="pairwise",
                    help="pairwise-complete covariance, or covariance of mean-filled profiles (PSD)")
    ap.add_argument("--tissue", default="", help="fit (and select the rank) on profiles of this corpus tissue only")
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    corpus = Corpus()
    train = corpus.rows("train")
    if a.group_fold in (0, 1):
        import zlib
        fold = corpus.profiles.group.map(lambda g: zlib.crc32(str(g).encode()) % 2).to_numpy()
        train = train[fold[train] == a.group_fold]
    tissue = corpus.profiles.tissue.to_numpy()
    if a.tissue:
        train = train[tissue[train] == a.tissue]
    P = corpus.beta.shape[1]
    log(f"train profiles {len(train)}")

    tot = torch.zeros(P, dtype=torch.float64, device=dev)
    cnt = torch.zeros(P, dtype=torch.float64, device=dev)
    for i in range(0, len(train), a.chunk):
        x = torch.from_numpy(corpus.get(train[i:i + a.chunk])).to(dev)
        w = torch.isfinite(x)
        tot += torch.where(w, x, torch.zeros((), device=dev)).sum(0, dtype=torch.float64)
        cnt += w.sum(0, dtype=torch.float64)
    mu = (tot / cnt.clamp_min(1)).float()
    mu = torch.where(cnt > 0, mu, torch.full_like(mu, 0.5))
    log("mean done")

    num = torch.zeros(P, P, dtype=torch.float64, device=dev)
    den = torch.zeros(P, P, dtype=torch.float32, device=dev)
    for i in range(0, len(train), a.chunk):
        x = torch.from_numpy(corpus.get(train[i:i + a.chunk])).to(dev)
        w = torch.isfinite(x)
        xc = torch.where(w, x - mu, torch.zeros((), device=dev)).double()
        num += xc.T @ xc
        wf = w.float()
        den += wf.T @ wf
        if (i // a.chunk) % 5 == 0:
            log(f"covariance rows {i + len(x)}/{len(train)}")
    del xc, wf
    for r0 in range(0, P, 4096):  # divide in place, block by block, to bound GPU memory
        d = den[r0:r0 + 4096].double()
        if a.cov == "pairwise":
            blk = num[r0:r0 + 4096] / d.clamp_min(1)
            blk[d < a.min_pair] = 0
        else:
            blk = num[r0:r0 + 4096] / len(train)
        num[r0:r0 + 4096] = blk
    del d, blk
    pair_cov = float((den >= a.min_pair).float().mean())
    del den
    C = num.float()
    del num
    C = 0.5 * (C + C.T)
    trace = float(torch.diagonal(C).double().sum())
    log(f"covariance assembled; entries with >= {a.min_pair} co-observations: {pair_cov:.4f}; trace {trace:.3f}")
    np.save(Path(a.out).with_suffix(".covariance_f32.npy"), C.cpu().numpy())  # temporary, for re-solves
    lam_t, V_t, rel_res = top_eigs(C, a.kmax)
    del C
    lam = lam_t.cpu().numpy()
    V = V_t.contiguous().cpu().numpy()
    log(f"eigendecomposition done; top lam {lam[:3]}, lam[{a.kmax - 1}] {lam[-1]:.3e}, "
        f"negative in top: {(lam <= 0).sum()}; max relative residual (first 1024) {rel_res:.2e}; "
        f"variance explained by top {a.kmax}: {lam.sum() / trace:.4f}")
    np.savez(a.out, mu=mu.cpu().numpy(), V=V.astype(np.float32), lam=lam, trace=trace,
             count=cnt.cpu().numpy(), min_pair=a.min_pair, n_train=len(train))
    log(f"saved {a.out}")

    # Rank selection on dev-split profiles only.
    rng = np.random.default_rng(20260930)
    dev_pool = corpus.rows("dev")
    if a.tissue:
        dev_pool = dev_pool[tissue[dev_pool] == a.tissue]
    dev_rows = np.sort(rng.choice(dev_pool, min(a.dev_profiles, len(dev_pool)), replace=False))
    X = torch.from_numpy(corpus.get(dev_rows)).to(dev)
    g = torch.Generator(device=dev).manual_seed(20260930)
    obs, ctx, masked = masked_view(X, torch.full((len(X),), 0.3, device=dev), g)
    res = {"n_dev_profiles": int(len(X)), "mask_ratio": 0.3, "ranks": {}}
    truth = X[masked].cpu().numpy()
    cpg = masked.nonzero()[:, 1].cpu().numpy()
    mean_pred = mu.expand_as(X)[masked].cpu().numpy()
    res["cpg_mean"] = mse_metrics(mean_pred, truth, cpg)
    n_pos = int((lam > 0).sum()) if (lam > 0).all() else int(np.argmax(lam <= 0))
    for r in [int(x) for x in a.ranks.split(",")]:
        if r > n_pos:
            log(f"rank {r} skipped: only the first {n_pos} eigenvalues are positive")
            continue
        core = PPCACore.load(a.out, rank=r).to(dev)
        preds = []
        for j in range(0, len(X), 64):
            s = core.posterior(X[j:j + 64], ctx[j:j + 64].float())
            preds.append(core.predict(s)[masked[j:j + 64]])
        p = torch.cat(preds).cpu().numpy()
        res["ranks"][r] = {**mse_metrics(p, truth, cpg), "sigma2": core.sigma2}
        log(f"rank {r}: dev MSE {res['ranks'][r]['mse']:.5f} MAE {res['ranks'][r]['mae']:.4f} "
            f"R2 {res['ranks'][r]['within_cpg_r2_pooled']:.3f}")
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out).with_suffix(".rank_selection_dev.json"), res)


if __name__ == "__main__":
    main()
