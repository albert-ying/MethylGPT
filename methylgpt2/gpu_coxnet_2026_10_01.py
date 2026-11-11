"""GPU elastic-net Cox (Breslow ties) with the locked protocol's settings, for scenario runs.

Objective: (1/n) * negative Breslow log partial likelihood + alpha * (l1 * |w|_1 + (1-l1)/2 * |w|^2)
on standardised features, with the first `n_unpen` columns (age, sex) unpenalised. FISTA with
a fixed step from a Lipschitz bound, warm-started along a path of `n_alphas` values from
alpha_max down to alpha_min_ratio * alpha_max (sksurv's Coxnet convention). Model selection:
inner StratifiedKFold(3) C-index, as in compare_joint_cox_2026-09-28.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored


def breslow_nll(eta, order, tie_last, event_sorted):
    """eta: (n,) linear predictor. order sorts subjects by time descending; tie_last maps each
    sorted position to the last position of its tied-time group (so ties share a risk set)."""
    e = eta[order]
    lse = torch.logcumsumexp(e, 0)[tie_last]
    return -((e - lse) * event_sorted).sum() / eta.shape[0]


def prepare_times(time_, event, dev):
    order = np.argsort(-time_, kind="stable")
    ts = time_[order]
    tie_last = np.empty(len(ts), dtype=np.int64)
    i = 0
    while i < len(ts):
        j = i
        while j + 1 < len(ts) and ts[j + 1] == ts[i]:
            j += 1
        tie_last[i:j + 1] = j
        i = j + 1
    return (torch.as_tensor(order, device=dev), torch.as_tensor(tie_last, device=dev),
            torch.as_tensor(event[order].astype(np.float32), device=dev))


def fit_path(X, time_, event, n_unpen=2, l1=0.5, n_alphas=12, alpha_min_ratio=0.05, iters=300, dev="cuda"):
    """X: (n, p) standardised float32 numpy (covariates first). Returns list of weight vectors."""
    Xt = torch.as_tensor(X, device=dev, dtype=torch.float32)
    n, p = Xt.shape
    order, tie_last, ev = prepare_times(time_, event, dev)
    pen = torch.ones(p, device=dev); pen[:n_unpen] = 0
    # Lipschitz bound of the Cox gradient: 0.25 * ||X||_2^2 / n is not valid in general; use ||X||_2^2 / n.
    v = torch.randn(p, device=dev)
    for _ in range(30):
        v = Xt.T @ (Xt @ v); v /= v.norm()
    L = float((Xt @ v).norm() ** 2 / n)
    w = torch.zeros(p, device=dev)
    # alpha_max from the gradient at the unpenalised-covariates-only solution (approximate: at w = 0)
    w0 = w.clone().requires_grad_(True)
    g0 = torch.autograd.grad(breslow_nll(Xt @ w0, order, tie_last, ev), w0)[0]
    amax = float(g0[n_unpen:].abs().max() / l1)
    alphas = amax * np.logspace(0, math.log10(alpha_min_ratio), n_alphas)
    path = []
    for a in alphas:
        step = 1.0 / (L + a * (1 - l1))
        y, tk = w.clone(), 1.0
        for _ in range(iters):
            yv = y.clone().requires_grad_(True)
            g = torch.autograd.grad(breslow_nll(Xt @ yv, order, tie_last, ev), yv)[0] + a * (1 - l1) * pen * y
            u = y - step * g
            thr = step * a * l1 * pen
            wn = torch.sign(u) * torch.clamp(u.abs() - thr, min=0)
            tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            y = wn + ((tk - 1) / tk1) * (wn - w)
            w, tk = wn, tk1
        path.append(w.clone())
    return alphas, path


def cindex(event, time_, risk):
    return concordance_index_censored(event.astype(bool), time_, risk)[0]


def select_and_fit(X, time_, event, seed, n_unpen=2, folds=3, dev="cuda", **kw):
    """Inner CV over the alpha path (mean inner C-index), then refit on all fitting rows."""
    inner = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = None
    for fit_i, ev_i in inner.split(X, event):
        alphas, path = fit_path(X[fit_i], time_[fit_i], event[fit_i], n_unpen, dev=dev, **kw)
        Xe = torch.as_tensor(X[ev_i], device=dev, dtype=torch.float32)
        s = np.array([cindex(event[ev_i], time_[ev_i], (Xe @ w).cpu().numpy()) for w in path])
        scores = s if scores is None else scores + s
    alphas, path = fit_path(X, time_, event, n_unpen, dev=dev, **kw)
    k = int(np.argmax(scores))
    return path[k].cpu().numpy(), float(alphas[k]), k
