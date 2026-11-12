"""Fused CUDA kernels (numba) for the FISTA iterations of the batched logistic elastic net of
large_gain_A_disease_few_labels_2026-10-24.py. Same algebra as the torch loop in ENet.fit_path, with 7 kernel launches
per iteration instead of about 40, which matters on a GPU shared with other jobs. Written 24 Oct 2026."""
from __future__ import annotations

import math

import numpy as np
from numba import cuda, float32

TPB = 256


@cuda.jit
def _eta_k(eta, yc, part2, y, W):
    N, P = eta.shape
    nblk = part2.shape[1]
    idx = cuda.grid(1)
    if idx >= N * P:
        return
    i = idx // P
    q = idx - i * P
    mu = float32(0.0)
    for b in range(nblk):
        mu += part2[q, b]
    e = eta[i, q] + (yc[q] - mu)
    eta[i, q] = (float32(1.0) / (float32(1.0) + math.exp(-e)) - y[i]) * W[q, i]


@cuda.jit
def _prox_k(G, m, inv_nq, yb, beta, rs, ridge, stp, thr, nb, d1, part):
    P, p = G.shape
    q = cuda.blockIdx.y
    bx = cuda.blockIdx.x
    nblk = cuda.gridDim.x
    tx = cuda.threadIdx.x
    sh = cuda.shared.array(TPB, float32)
    rsq = rs[q]
    rq = ridge[q]
    sq = stp[q]
    tq = thr[q]
    acc = float32(0.0)
    j = bx * TPB + tx
    while j < p:
        y_ = yb[q, j]
        g = (G[q, j] - m[q, j] * rsq) * inv_nq[q, j] + y_ * rq
        z = y_ - sq * g
        c = z
        if c > tq:
            c = tq
        elif c < -tq:
            c = -tq
        n_ = z - c
        d = n_ - beta[q, j]
        nb[q, j] = n_
        d1[q, j] = d
        acc += (y_ - n_) * d
        j += nblk * TPB
    sh[tx] = acc
    cuda.syncthreads()
    s = TPB // 2
    while s > 0:
        if tx < s:
            sh[tx] += sh[tx + s]
        cuda.syncthreads()
        s //= 2
    if tx == 0:
        part[q, bx] = sh[0]


@cuda.jit
def _scal_k(part, rs, nq, step, yc, b0, tk, mom):
    P = rs.shape[0]
    nblk = part.shape[1]
    q = cuda.grid(1)
    if q >= P:
        return
    dot = float32(0.0)
    for b in range(nblk):
        dot += part[q, b]
    nc = yc[q] - step[q] * (rs[q] / nq[q])
    t = tk[q]
    if dot + (yc[q] - nc) * (nc - b0[q]) > float32(0.0):
        t = float32(1.0)
    t1 = (float32(1.0) + math.sqrt(float32(1.0) + float32(4.0) * t * t)) / float32(2.0)
    mo = (t - float32(1.0)) / t1
    yc[q] = nc + mo * (nc - b0[q])
    b0[q] = nc
    tk[q] = t1
    mom[q] = mo


@cuda.jit
def _upd_k(nb, d1, mom, inv, m, yb, u, part2):
    P, p = nb.shape
    q = cuda.blockIdx.y
    bx = cuda.blockIdx.x
    nblk = cuda.gridDim.x
    tx = cuda.threadIdx.x
    sh = cuda.shared.array(TPB, float32)
    mo = mom[q]
    acc = float32(0.0)
    j = bx * TPB + tx
    while j < p:
        v = nb[q, j] + mo * d1[q, j]
        yb[q, j] = v
        uu = v * inv[q, j]
        u[q, j] = uu
        acc += m[q, j] * uu
        j += nblk * TPB
    sh[tx] = acc
    cuda.syncthreads()
    s = TPB // 2
    while s > 0:
        if tx < s:
            sh[tx] += sh[tx + s]
        cuda.syncthreads()
        s //= 2
    if tx == 0:
        part2[q, bx] = sh[0]


def fista_lambda(X, y, W, m, inv, inv_nq, nq, beta, b0, lam, a, Lsm, iters, nblk_cap=48):
    """One lambda of the FISTA loop for one batch element: X (N, p), y (N,), W (P, N), per-problem stats (P, p).
    Returns the new (beta, b0). Same arithmetic as the torch loop of ENet.fit_path."""
    import torch
    P, p = m.shape
    N = X.shape[0]
    L = Lsm + lam * (1 - a)
    step = (1.0 / L).contiguous()
    thr = (step * lam * a).contiguous()
    ridge = (lam * (1 - a)).contiguous()
    nblk = int(min(nblk_cap, max(1, (p + TPB * 4 - 1) // (TPB * 4))))
    yb = beta.clone()
    yc = b0.clone().contiguous()
    b0c = b0.clone().contiguous()
    tk = torch.ones(P, device=X.device, dtype=X.dtype)
    mom = torch.empty_like(tk)
    u = (yb * inv).contiguous()
    part = torch.zeros(P, nblk, device=X.device, dtype=X.dtype)
    part2 = torch.zeros(P, nblk, device=X.device, dtype=X.dtype)
    part2[:, 0] = torch.linalg.vecdot(m, u)
    bufA = beta.clone()
    bufB = torch.empty_like(yb)
    d1 = torch.empty_like(yb)
    eta = torch.empty(N, P, device=X.device, dtype=X.dtype)
    G = torch.empty(P, p, device=X.device, dtype=X.dtype)
    rs = torch.empty(P, device=X.device, dtype=X.dtype)
    stream = cuda.external_stream(torch.cuda.current_stream(X.device).cuda_stream)
    A = cuda.as_cuda_array
    c = {k: A(v) for k, v in dict(eta=eta, yc=yc, part2=part2, y=y, W=W, G=G, m=m, inv_nq=inv_nq, yb=yb, rs=rs,
                                    ridge=ridge, stp=step, thr=thr, d1=d1, part=part, nq=nq, b0=b0c, tk=tk, mom=mom,
                                    inv=inv, u=u, bufA=bufA, bufB=bufB).items()}
    g_eta = ((N * P + TPB - 1) // TPB,)
    g_row = (nblk, P)
    g_scal = ((P + TPB - 1) // TPB,)
    cur, nxt = "bufA", "bufB"                     # cur holds beta (x_k); nxt receives x_{k+1}
    for _ in range(iters):
        torch.mm(X, u.t(), out=eta)
        _eta_k[g_eta, TPB, stream](c["eta"], c["yc"], c["part2"], c["y"], c["W"])
        torch.sum(eta, 0, out=rs)
        torch.mm(eta.t(), X, out=G)
        _prox_k[g_row, TPB, stream](c["G"], c["m"], c["inv_nq"], c["yb"], c[cur], c["rs"], c["ridge"], c["stp"], c["thr"],
                                    c[nxt], c["d1"], c["part"])
        _scal_k[g_scal, TPB, stream](c["part"], c["rs"], c["nq"], c["stp"], c["yc"], c["b0"], c["tk"], c["mom"])
        _upd_k[g_row, TPB, stream](c[nxt], c["d1"], c["mom"], c["inv"], c["m"], c["yb"], c["u"], c["part2"])
        cur, nxt = nxt, cur
    out = bufA if cur == "bufA" else bufB
    return out.clone(), b0c.clone()
