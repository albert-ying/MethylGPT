#!/usr/bin/env python3
"""Large-gain analysis H: changed training methods for few-label disease classification, selected under analysis G's
locked sparse-input setting. SIMULATED MISSINGNESS ON EMPIRICAL PROFILES: the primary setting keeps, per profile, a
random fraction of its observed CpGs (G's masks, seed 20261025); the training augmentation below also masks empirical
profiles. No beta values are simulated.

Written on 25 Oct 2026 after G's lock (G/locked_setting_2026-10-25.json, committed) and before any H model is fitted.
Reuses A's protocol, frozen held-out primary set and draws (large_gain_A_disease_few_labels_2026-10-24.py), F's dev-val
datasets, training strata, duplicate removal and trainer components (large_gain_F_disease_trained_representation_
2026-10-24.py) and G's masks, masked representations and comparators (large_gain_G_sparse_inputs_2026-10-25.py), all
imported. Every result is reported, whichever method it favours.

SETTINGS
  Primary: G's locked keep fraction f* (sparse, simulated missingness). Secondary: complete profiles.
  Comparators at f*: G's raw_mean, raw_knn and clocks (G's held-out per-draw table, same draws); on complete profiles:
  A's raw and clocks (A's per-draw table). The released-MethylGPT baseline: G's locked representation at f* and A's
  embedding on complete profiles.

CANDIDATE METHODS (pre-specified set; candidates may be added by dated amendment during development)
  H1  Disease training with masking augmentation and a consistency loss. F's design D1 with F's best D1 settings:
      frozen age-stage encoder, identity-initialized full-rank linear adapter (2,048 -> 2,048) on the standardized
      embedding, episodic objective (F.episodic_loss), pool S15 (F.training_strata; development: dev-val groups and
      F's duplicates removed; final: all train-split strata, primary duplicates removed), lr 1e-4, wd 0.1, dropout
      0.1, 8 episodes per step, 4,000 steps with 300-step warm-up and cosine decay, seed 20261024; checkpoints
      evaluated at 250, 500, 1,000, 2,000 and 4,000.
      Training views: the complete profile and V = 4 masked views per training profile, drawn once: view v of profile
      p keeps a fraction exp(U(log 0.03, log 0.5)) of its observed CpGs (U from default_rng([20261025, h(p), 7, v])),
      the kept CpGs chosen by the smallest of default_rng([20261025, h(p), 8, v]).random(49156) (h as in G; a separate
      stream from G's evaluation masks); embeddings of every view computed once with the frozen encoder (bank).
      Each episode uses the complete views with probability 0.25, otherwise one masked view index for all its
      profiles. Consistency loss (weight lambda): for 256 random training profiles and one masked view each,
      mean ||A(e_masked) - sg(A(e_complete))||^2 / mean ||sg(A(e_complete)) - its batch mean||^2 (sg = stop
      gradient; the ratio keeps the loss scale-free). lambda in {0, 0.3, 1, 3} (4 runs).
      Output representation: A(e) of the target profile's embedding (masked at f*, or complete).
  H2  Label-free cohort whitening before A's classifier: the representation of all contrast profiles of the target
      dataset (labels unused) is centred by its cohort mean and projected on its principal components (SVD), each
      component scaled to unit variance; components with singular value <= 1e-6 x the largest are dropped. Base
      representations: the released embedding, G's locked representation if different, and the best H1 checkpoint.
      (Centring or per-feature scaling alone are not candidates: A's classifier standardizes every feature with the
      training draw's mean and SD, so they cannot change the result.)
  H3  Few-label adapter fine-tuning in the target cohort instead of the elastic net: on the base representation
      standardized with the training draw's mean and SD, logit = w . (z + U V^T z) + b with a rank-16 update (U, V
      2,048 x 16, U ~ N(0, 0.01^2), V = 0, w = 0, b = logit of the case fraction), full-batch Adam (lr 1e-3, weight
      decay 1e-4) on the training draw, number of epochs chosen from {10, 25, 50, 100, 200, 400} by the mean held-out
      binomial deviance over A's inner 5 folds of that draw (StratifiedKFold, shuffle, random_state = draw index),
      then refitted on the whole draw for that number of epochs; AUC on the remaining contrast profiles (A's draws and
      scoring). Base representations: the released embedding and the best H1 checkpoint.
  Selection rule: the candidate (H1 run and checkpoint, H2 base, H3 base) with the highest dev-val metric under f*:
  mean over F's 24 dev-val datasets of (mean AUC - 0.5) at n = 50 (A's draws; A's classifier per dataset as in A's fit
  stage, FP32). Ties (1e-4) go to the earlier candidate in the order H2, H3, H1. The complete-profile dev-val metric is
  computed for every candidate and reported, not used to select. Development never reads held-out labels.

LOCK, RETRAINING AND HELD-OUT EVALUATION
  Stage lock writes H/locked_design_2026-10-25.json (candidate, configuration, base, checkpoint step, protocol). If
  the locked design contains H1, H1 is retrained once on the final strata with the development schedule stopped at
  the selected step (stage final, checkpoint H/checkpoints/final_<k>.pt, not committed). The script, lock file and
  development log are committed before stage heldout runs. Held-out: the locked design on A's primary datasets under
  f* (G's held-out masked embeddings) and on complete profiles (A's embedding files), every size A evaluates.
  PRIMARY: M(H) at n = 50 under f* over the primary datasets evaluable at n = 50 (24), best comparator = the best of
  raw_mean, raw_knn and clocks at f* on the same datasets, relative gain, 95% interval of M(H) - M(best) by A.compare
  (paired dataset bootstrap, 2,000 resamples, seed 20261024). PASS iff relative gain >= 10% and lower bound > 0.
  SECONDARY: n = 20 and 100; complete profiles against A's raw and clocks (A's 21-dataset decision set and all);
  gain over G's locked released representation at f* and over A's embedding on complete profiles; external against
  corpus datasets. If the locked evaluation fails, development may resume on dev-val only and a second design may be
  locked and evaluated once (numbered); every locked evaluation is reported.
AMENDMENT 25 Oct 2026, during development (after the H2 probes and H3 on the embedding, before any H1 run): H3 is
  also run on G's locked representation (base g_locked:imputed, the MethylGPT-imputed CpGs at f*), because H3 raised the
  embedding's dev-val metric (0.2152 against 0.2070 for A's classifier on the same embedding at f* = 0.05) and G's
  representation is the strongest released-model representation at f* on dev-val (0.2249). Same H3 settings.

Stages: bank (--phase dev|final), train (--config), probe (--candidate H2|H3 --base ...), lock, final, heldout, analyse.
Outputs: artifacts/large_gain_2026-10-25/H/ (README.md, dev log, lock, per-draw and per-dataset AUC, summary, rule
outcome); working files in artifacts/large_gain_2026-10-25/work_H/; checkpoints in H/checkpoints/ (not committed).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = ART / "large_gain_2026-10-25"
OUTH = OUT / "H"
WORKH = OUT / "work_H"
CKPT = OUTH / "checkpoints"
G_SCRIPT = HERE / "large_gain_G_sparse_inputs_2026-10-25.py"
DATE = "2026-10-25"
SEED = 20261024
SEED_H = 20261025
N_VIEWS = 4
KEEP_LO, KEEP_HI = 0.03, 0.5
P_COMPLETE = 0.25
RANK3 = 16
EPOCH_GRID = [10, 25, 50, 100, 200, 400]
DEV_LOG = OUTH / f"dev_log_{DATE}.csv"
H1_DEFAULT = dict(design="H1", pool="S15", objective="episodic", adapter="linear_id", K=2048, hidden=1024, dropout=0.1,
                  aug27k=False, lr=1e-4, wd=0.1, episodes_per_step=8, steps=4000, eval_steps=[250, 500, 1000, 2000, 4000],
                  seed=SEED, lam=0.0, n_views=N_VIEWS, p_complete=P_COMPLETE)
ORDER = {"H2": 0, "H3": 1, "H1": 2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def load_g():
    spec = importlib.util.spec_from_file_location("large_gain_G", G_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["large_gain_G"] = m
    spec.loader.exec_module(m)
    return m


G = load_g()
F = G.load_f()
A = F.A


def g_lock():
    p = G.OUTG / f"locked_setting_{G.DATE}.json"
    if not (p.exists() and G.git_committed(p)):
        sys.exit("Cannot proceed - G's locked setting is not committed")
    return json.loads(p.read_text())


def pid_hash(pid):
    return int.from_bytes(hashlib.sha256(str(pid).encode()).digest()[:8], "little")


# ------------------------------------------------------------------ masked-view bank for H1
def view_masks(pid, x):
    """Kept CpGs of the N_VIEWS training views of one profile (bool, N_VIEWS x 49,156)."""
    h = pid_hash(pid)
    obs = np.flatnonzero(np.isfinite(x))
    out = np.zeros((N_VIEWS, len(x)), bool)
    for v in range(N_VIEWS):
        frac = math.exp(np.random.default_rng([SEED_H, h, 7, v]).uniform(math.log(KEEP_LO), math.log(KEEP_HI)))
        u = np.random.default_rng([SEED_H, h, 8, v]).random(len(x))
        k = max(1, int(round(frac * len(obs))))
        out[v, obs[np.argsort(u[obs], kind="stable")[:k]]] = True
    return out


def bank_rows(phase):
    st = F.training_strata(dict(pool="S15"), phase)
    return np.array(sorted({x for s in st for x in np.r_[s["case_rows"], s["ctrl_rows"]]}))


def stage_bank(phase, device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True           # as A's and F's embeddings
    dev = torch.device(device)
    WORKH.mkdir(parents=True, exist_ok=True)
    rows = bank_rows(phase)
    fb, fr = WORKH / f"bank_{phase}.npy", WORKH / f"bank_{phase}_rows.npy"
    if fb.exists() and fr.exists() and np.array_equal(np.load(fr), rows):
        log("bank up to date")
        return
    prof, xc = F.profiles(), F.XCache()
    ids = prof.loc[rows, "id"].astype(str).to_numpy()
    model, core, _, _ = F.load_encoder("age_stage", dev)
    out = np.lib.format.open_memmap(fb, mode="w+", dtype=np.float32, shape=(N_VIEWS, len(rows), 2048))
    fracs = np.zeros((N_VIEWS, len(rows)), np.float32)
    t0 = time.time()
    for i in range(0, len(rows), 1024):
        X = xc.get(rows[i:i + 1024])
        M = np.stack([view_masks(p, x) for p, x in zip(ids[i:i + 1024], X)], 1)          # (V, b, P)
        for v in range(N_VIEWS):
            Xm = np.where(M[v], X, np.nan).astype(np.float32)
            out[v, i:i + len(X)] = F.embed_profiles(model, core, Xm, dev)[0]
            fracs[v, i:i + len(X)] = M[v].sum(1) / np.isfinite(X).sum(1)
        log(f"bank {phase}: {min(i + 1024, len(rows))}/{len(rows)} profiles ({time.time() - t0:.0f} s)")
    out.flush()
    np.save(fr, rows)
    np.save(WORKH / f"bank_{phase}_fractions.npy", fracs)
    write_json(OUTH / f"bank_{phase}_{DATE}.json", {"phase": phase, "profiles": int(len(rows)), "views": N_VIEWS,
               "keep_fraction_quantiles": np.quantile(fracs, [0, 0.25, 0.5, 0.75, 1]).round(4).tolist(),
               "rows_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(), "label": G.LABEL})


# ------------------------------------------------------------------ H1 trainer
class H1:
    def __init__(self, cfg, phase, dev):
        import torch
        import torch.nn as nn
        self.cfg, self.phase, self.dev = cfg, phase, dev
        torch.manual_seed(cfg["seed"])
        xc, E = F.emb_cache()
        strata = F.training_strata(cfg, phase)
        rows = np.load(need(WORKH / f"bank_{phase}_rows.npy"))
        if not np.array_equal(rows, bank_rows(phase)):
            sys.exit("bank rows differ from the training strata")
        pos = pd.Series(np.arange(len(rows)), index=rows)
        self.strata = [dict(s, case=pos.loc[s["case_rows"]].to_numpy(), ctrl=pos.loc[s["ctrl_rows"]].to_numpy()) for s in strata]
        self.sampler = F.StrataSampler(self.strata, cfg["seed"])
        Ec = E[xc.pos.loc[rows].to_numpy()]
        self.Ec = torch.from_numpy(Ec).to(dev)
        self.Ev = torch.from_numpy(np.load(need(WORKH / f"bank_{phase}.npy"))[: cfg["n_views"]]).to(dev)     # (V, N, 2048)
        self.adapter = F.make_adapter(cfg, 2048, Ec.mean(0), Ec.std(0) + 1e-6).to(dev)
        self.log_lam = nn.Parameter(torch.zeros((), device=dev))
        self.log_tau = nn.Parameter(torch.zeros((), device=dev))
        self.opt = torch.optim.AdamW([{"params": list(self.adapter.parameters()) + [self.log_lam, self.log_tau],
                                       "lr": cfg["lr"], "weight_decay": cfg["wd"]}], eps=1e-10)
        self.base_lr = cfg["lr"]
        self.rng = np.random.default_rng(cfg["seed"] + 101)
        self.n_strata, self.n_groups, self.n_items = len(self.strata), len(self.sampler.groups), len(rows)
        log(f"H1 trainer {phase}: {self.n_strata} strata, {self.n_groups} groups, {self.n_items} profiles, lambda {cfg['lam']}")

    def step(self, k):
        import torch
        cfg = self.cfg
        warm = min(1.0, k / 300)
        cos = 0.5 * (1 + math.cos(math.pi * k / cfg["steps"]))
        for g in self.opt.param_groups:
            g["lr"] = self.base_lr * warm * (0.05 + 0.95 * cos)
        self.adapter.train(True)
        idx, ys, sup, qry = self.sampler.episodes(cfg["episodes_per_step"])
        view = np.where(self.rng.random(idx.shape[0]) < cfg["p_complete"], -1, self.rng.integers(0, cfg["n_views"], idx.shape[0]))
        feats = []
        for e in range(idx.shape[0]):
            it = torch.as_tensor(idx[e], device=self.dev)
            feats.append(self.Ec[it] if view[e] < 0 else self.Ev[view[e], it])
        r = self.adapter(torch.stack(feats).view(-1, 2048)).view(idx.shape[0], idx.shape[1], -1)
        loss = F.episodic_loss(r, *(torch.as_tensor(v, device=self.dev) for v in (ys, sup, qry)), self.log_lam, self.log_tau)
        parts = {"episodic": float(loss)}
        if cfg["lam"] > 0:
            b = torch.as_tensor(self.rng.integers(0, self.n_items, 256), device=self.dev)
            v = torch.as_tensor(self.rng.integers(0, cfg["n_views"], 256), device=self.dev)
            rm = self.adapter(self.Ev[v, b])
            with torch.no_grad():
                self.adapter.eval()
                rc = self.adapter(self.Ec[b])
                self.adapter.train(True)
            lc = ((rm - rc) ** 2).sum(1).mean() / ((rc - rc.mean(0)) ** 2).sum(1).mean().clamp_min(1e-8)
            loss = loss + cfg["lam"] * lc
            parts["consistency"] = float(lc)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.adapter.parameters()) + [self.log_lam, self.log_tau], 1.0)
        self.opt.step()
        return parts

    def represent(self, E):
        import torch
        self.adapter.eval()
        with torch.no_grad():
            Et = torch.from_numpy(np.asarray(E, np.float32)).to(self.dev)
            return torch.cat([self.adapter(Et[i:i + 4096]) for i in range(0, len(Et), 4096)]).cpu().numpy()

    def state(self):
        return {"cfg": self.cfg, "phase": self.phase, "adapter": {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()},
                "log_lam": float(self.log_lam), "log_tau": float(self.log_tau), "strata": [s["stratum"] for s in self.strata]}


def adapter_from_state(st, dev):
    a = st["adapter"]
    ad = F.make_adapter(st["cfg"], 2048, a["m"].numpy(), a["s"].numpy()).to(dev)
    ad.load_state_dict(a)
    ad.eval()
    return ad


def apply_adapter(ad, E, dev):
    import torch
    with torch.no_grad():
        Et = torch.from_numpy(np.asarray(E, np.float32)).to(dev)
        return torch.cat([ad(Et[i:i + 4096]) for i in range(0, len(Et), 4096)]).cpu().numpy()


# ------------------------------------------------------------------ dev-val data
def dev_inputs(fstar):
    """{acc: (y, E_sparse, E_complete, ids)} on dev-val: G's masked embeddings at f* and F's complete embeddings."""
    xc, E = F.emb_cache()
    out = {}
    for a, (sub, _) in sorted(F.devval_sets(with_x=False).items()):
        Es = np.load(need(G.WORKG / "dev" / f"{a}_{G.ftag(fstar)}_embedding.npy"))
        Ec = E[xc.pos.loc[sub.corpus_row.to_numpy()].to_numpy()]
        out[a] = ((sub.role == "case").to_numpy(np.float32), Es, Ec, sub.id.astype(str).to_numpy())
    return out


def dev_g_locked_rep(fstar, rep):
    """G's locked representation on dev-val at f* (for H2 when G's representation is not the embedding)."""
    out = {}
    for a in sorted(F.devval_sets(with_x=False)):
        if rep == "imputed":
            out[a] = np.load(need(G.WORKG / "dev" / f"{a}_{G.ftag(fstar)}_imputed.npy"))
        else:
            E = np.load(need(G.WORKG / "dev" / f"{a}_{G.ftag(fstar)}_embedding.npy"))
            out[a] = E if rep == "embedding" else E[:, :1536]
    return out


def dev_metric(Z, ys, dev, classifier="enet"):
    accs = sorted(Z)
    if classifier == "enet":
        d = G.protocol_per_dataset(A, Z, ys, 50, dev, accs)
    else:
        d = h3_protocol(Z, ys, 50, dev, accs)
    per = d.groupby("accession").auc.mean() - 0.5
    return float(per.mean()), per, d


def log_dev(row):
    old = pd.read_csv(DEV_LOG) if DEV_LOG.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(DEV_LOG, index=False)


# ------------------------------------------------------------------ H2: cohort whitening
def whiten(Z):
    Z = np.asarray(Z, np.float64)
    Zc = Z - Z.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(Zc, full_matrices=False)
    keep = S > 1e-6 * S[0]
    return (U[:, keep] * math.sqrt(len(Z) - 1)).astype(np.float32)


# ------------------------------------------------------------------ H3: few-label adapter fine-tuning
def h3_protocol(Z, ys, n, dev, accs):
    """A's draws and inner folds; rank-16 adapter + logistic head trained by full-batch Adam; epochs chosen by inner-CV
    deviance; refit; AUC on the remaining profiles."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    rows = []
    for acc in accs:
        Zall, y = np.asarray(Z[acc], np.float32), ys[acc]
        trs = A.draws_for(int(A.accession_number(acc)), n, y)
        # problems: per draw, 5 inner folds + the whole draw
        Xs, Ys, Ws = [], [], []
        for r, tr in enumerate(trs):
            skf = StratifiedKFold(A.FOLDS, shuffle=True, random_state=r)
            folds = np.zeros(len(tr), int)
            for k, (_, te) in enumerate(skf.split(np.zeros(len(tr)), y[tr])):
                folds[te] = k
            Xs.append(Zall[tr])
            Ys.append(y[tr])
            W = np.ones((A.FOLDS + 1, len(tr)), np.float32)
            for k in range(A.FOLDS):
                W[k, folds == k] = 0
            Ws.append(W)
        D, S, m = len(trs), A.FOLDS + 1, n
        X = torch.from_numpy(np.stack(Xs)).to(dev)                          # (D, m, p)
        Yt = torch.from_numpy(np.stack(Ys)).to(dev)                          # (D, m)
        Wt = torch.from_numpy(np.stack(Ws)).to(dev)                          # (D, S, m) fitting weights
        X = X[:, None].expand(D, S, m, -1)
        wsum = Wt.sum(-1, keepdim=True)
        mean = (X * Wt[..., None]).sum(2, keepdim=True) / wsum[..., None]
        sd = ((((X - mean) ** 2) * Wt[..., None]).sum(2, keepdim=True) / wsum[..., None]).sqrt()
        good = (sd > 1e-6).float()
        Xz = (X - mean) / sd.clamp_min(1e-6) * good
        p = Xz.shape[-1]
        gen = torch.Generator(device=dev).manual_seed(SEED)
        U = (torch.randn(D, S, p, RANK3, device=dev, generator=gen) * 0.01).requires_grad_(True)
        V = torch.zeros(D, S, p, RANK3, device=dev, requires_grad=True)
        w = torch.zeros(D, S, p, device=dev, requires_grad=True)
        ybar = ((Yt[:, None] * Wt).sum(-1) / Wt.sum(-1)).clamp(1e-4, 1 - 1e-4)
        b = torch.log(ybar / (1 - ybar)).clone().requires_grad_(True)
        opt = torch.optim.Adam([U, V, w, b], lr=1e-3, weight_decay=1e-4)

        def logits():
            h = Xz + torch.einsum("dsmr,dspr->dsmp", torch.einsum("dsmp,dspr->dsmr", Xz, V), U)
            return torch.einsum("dsmp,dsp->dsm", h, w) + b[..., None]

        dev_hist = {}
        snap = {}
        for ep in range(1, max(EPOCH_GRID) + 1):
            opt.zero_grad(set_to_none=True)
            lg = logits()
            ll = torch.nn.functional.binary_cross_entropy_with_logits(lg, Yt[:, None].expand_as(lg), reduction="none")
            loss = ((ll * Wt).sum(-1) / Wt.sum(-1)).sum()
            loss.backward()
            opt.step()
            if ep in EPOCH_GRID:
                with torch.no_grad():
                    lg = logits()
                    ll = torch.nn.functional.binary_cross_entropy_with_logits(lg, Yt[:, None].expand_as(lg), reduction="none")
                    held = 1 - Wt[:, :A.FOLDS]
                    dev_hist[ep] = (2 * (ll[:, :A.FOLDS] * held).sum((1, 2)) / held.sum((1, 2))).cpu().numpy()   # (D,)
                    snap[ep] = (U.detach()[:, A.FOLDS].clone(), V.detach()[:, A.FOLDS].clone(), w.detach()[:, A.FOLDS].clone(),
                                b.detach()[:, A.FOLDS].clone(), mean[:, A.FOLDS].clone(), sd[:, A.FOLDS].clone(), good[:, A.FOLDS].clone())
        grid = np.array(EPOCH_GRID)
        devs = np.stack([dev_hist[e] for e in EPOCH_GRID], 1)                # (D, G)
        sel = grid[np.argmin(devs, 1)]
        for r, tr in enumerate(trs):
            Uf, Vf, wf, bf, mf, sf, gf = (t[r] for t in snap[int(sel[r])])
            mask = np.ones(len(y), bool)
            mask[tr] = False
            with torch.no_grad():
                Xr = torch.from_numpy(Zall[mask]).to(dev)
                zr = (Xr - mf) / sf.clamp_min(1e-6) * gf
                h = zr + (zr @ Vf) @ Uf.T
                sc = (h @ wf + bf).cpu().numpy()
            ysc = y[mask]
            rows.append({"accession": acc, "n_train": n, "draw": r, "auc": A.auc(sc, ysc), "epochs": int(sel[r]),
                         "cv_deviance": float(devs[r].min()), "n_score_case": int(ysc.sum()), "n_score_ctrl": int((1 - ysc).sum()),
                         "features": int(Zall.shape[1])})
        del X, Xz, U, V, w, opt
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stages train and probe
def stage_train(device, cfg_json, tag):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False          # classifier precision (A's fit stage); adapter training is light
    dev = torch.device(device)
    gl = g_lock()
    fstar = gl["f"]
    di = dev_inputs(fstar)
    ys = {a: v[0] for a, v in di.items()}
    over = json.loads(cfg_json)
    for o in (over if isinstance(over, list) else [over]):
        cfg = dict(H1_DEFAULT, **o)
        rid = "H1_" + hashlib.sha256(json.dumps({k: cfg[k] for k in sorted(cfg) if k != "eval_steps"}, sort_keys=True).encode()).hexdigest()[:10]
        if DEV_LOG.exists() and set(cfg["eval_steps"]) <= set(pd.read_csv(DEV_LOG).query("run_id == @rid").step):
            log(f"{rid} already logged")
            continue
        tr = H1(cfg, "dev", dev)
        CKPT.mkdir(parents=True, exist_ok=True)
        t0, hist = time.time(), []
        for k in range(1, cfg["steps"] + 1):
            hist.append(tr.step(k))
            if k % 500 == 0:
                log(f"{rid} step {k}: " + " ".join(f"{a} {b:.4f}" for a, b in pd.DataFrame(hist[-100:]).mean().items()))
            if k in cfg["eval_steps"]:
                Ms, pers, _ = dev_metric({a: tr.represent(v[1]) for a, v in di.items()}, ys, dev)
                Mc, perc, _ = dev_metric({a: tr.represent(v[2]) for a, v in di.items()}, ys, dev)
                ck = CKPT / f"{rid}_step{k}.pt"
                torch.save(tr.state(), ck)
                log_dev({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "candidate": "H1", "run_id": rid, "step": k,
                         "base": "age-stage embedding", "devval_metric_sparse": Ms, "devval_metric_complete": Mc, "f_star": fstar,
                         "per_dataset_sparse": json.dumps(pers.round(5).to_dict()), "per_dataset_complete": json.dumps(perc.round(5).to_dict()),
                         "config": json.dumps(cfg, sort_keys=True), "checkpoint": str(ck.relative_to(HERE)),
                         "train_strata": tr.n_strata, "train_profiles": tr.n_items, "seconds": round(time.time() - t0, 1),
                         "masking": G.LABEL})
                log(f"DEV {rid} step {k}: sparse (f = {fstar}) {Ms:.4f}, complete {Mc:.4f}")
        del tr
        torch.cuda.empty_cache()


def best_h1():
    d = pd.read_csv(need(DEV_LOG))
    d = d[d.candidate == "H1"]
    return d.sort_values("devval_metric_sparse", ascending=False).iloc[0]


def base_inputs(base, fstar, dev):
    """(sparse dict, complete dict or None) of a base representation on dev-val."""
    di = dev_inputs(fstar)
    if base == "embedding":
        return {a: v[1] for a, v in di.items()}, {a: v[2] for a, v in di.items()}
    if base.startswith("g_locked:"):
        return dev_g_locked_rep(fstar, base.split(":")[1]), None
    import torch
    st = torch.load(need(HERE / base), map_location="cpu", weights_only=False)
    ad = adapter_from_state(st, dev)
    return {a: apply_adapter(ad, v[1], dev) for a, v in di.items()}, {a: apply_adapter(ad, v[2], dev) for a, v in di.items()}


def stage_probe(device, candidate, base, tag):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    gl = g_lock()
    fstar = gl["f"]
    if base == "best_h1":
        base = best_h1().checkpoint
    ys = {a: v[0] for a, v in dev_inputs(fstar).items()}
    Zs, Zc = base_inputs(base, fstar, dev)
    t0 = time.time()
    if candidate == "H2":
        Ms, pers, _ = dev_metric({a: whiten(z) for a, z in Zs.items()}, ys, dev)
        Mc, perc = (dev_metric({a: whiten(z) for a, z in Zc.items()}, ys, dev)[:2] if Zc is not None else (np.nan, pd.Series(dtype=float)))
    else:
        Ms, pers, _ = dev_metric(Zs, ys, dev, classifier="h3")
        Mc, perc = (dev_metric(Zc, ys, dev, classifier="h3")[:2] if Zc is not None else (np.nan, pd.Series(dtype=float)))
    rid = f"{candidate}_" + hashlib.sha256(base.encode()).hexdigest()[:10]
    log_dev({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "candidate": candidate, "run_id": rid, "step": np.nan,
             "base": base, "devval_metric_sparse": Ms, "devval_metric_complete": Mc, "f_star": fstar,
             "per_dataset_sparse": json.dumps(pers.round(5).to_dict()), "per_dataset_complete": json.dumps(perc.round(5).to_dict()),
             "config": json.dumps({"candidate": candidate, "base": base}), "checkpoint": base if base.endswith(".pt") else "",
             "seconds": round(time.time() - t0, 1), "masking": G.LABEL})
    log(f"DEV {candidate} on {base}: sparse (f = {fstar}) {Ms:.4f}, complete {Mc:.4f}")


# ------------------------------------------------------------------ lock, final, held-out
def stage_lock(k):
    d = pd.read_csv(need(DEV_LOG))
    d = d.assign(rank=d.candidate.map(ORDER), m=d.devval_metric_sparse.round(4)).sort_values(["m", "rank"], ascending=[False, True])
    best = d.iloc[0]
    gl = g_lock()
    lock = {"lock_number": k, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "candidate": best.candidate, "run_id": best.run_id,
            "step": None if pd.isna(best.step) else int(best.step), "base": best.base, "config": json.loads(best.config),
            "checkpoint": best.checkpoint if isinstance(best.checkpoint, str) else "",
            "devval_metric_sparse": float(best.devval_metric_sparse), "devval_metric_complete": float(best.devval_metric_complete),
            "f_star": gl["f"], "g_lock": gl, "development_rows": int(len(d)),
            "selection_rule": "highest dev-val mean (AUC - 0.5) at n = 50 under G's f*; ties to H2, H3, H1",
            "a_protocol_fingerprint": F.protocol_fingerprint(), "script_sha256": sha256_file(Path(__file__)),
            "masking": G.LABEL}
    write_json(OUTH / (f"locked_design_{DATE}.json" if k == 1 else f"locked_design_{DATE}_{k}.json"), lock)
    log(f"locked {k}: {best.candidate} {best.run_id} step {best.step} base {best.base}: dev-val sparse {best.devval_metric_sparse:.4f}")


def read_lock(k):
    return json.loads(need(OUTH / (f"locked_design_{DATE}.json" if k == 1 else f"locked_design_{DATE}_{k}.json")).read_text())


def h1_config_of(lock):
    """H1 configuration (and selected step) inside the locked design, or None."""
    if lock["candidate"] == "H1":
        return lock["config"], lock["step"]
    b = lock["base"]
    if isinstance(b, str) and b.endswith(".pt"):
        import torch
        st = torch.load(need(HERE / b), map_location="cpu", weights_only=False)
        return st["cfg"], int(b.split("_step")[1].split(".")[0])
    return None


def stage_final(device, k):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    dev = torch.device(device)
    lock = read_lock(k)
    h = h1_config_of(lock)
    if h is None:
        log("locked design has no trained component; nothing to retrain")
        return
    cfg, stop = h
    tr = H1(dict(cfg, eval_steps=[]), "final", dev)
    for s in range(1, stop + 1):
        tr.step(s)
    CKPT.mkdir(parents=True, exist_ok=True)
    st = tr.state()
    torch.save(st, CKPT / f"final_{k}.pt")
    write_json(OUTH / f"final_training_{k}_{DATE}.json", {"lock": k, "strata": tr.n_strata, "groups": tr.n_groups,
                                                         "profiles": tr.n_items, "steps": stop, "schedule_steps": cfg["steps"]})
    log(f"final H1 model saved ({tr.n_strata} strata)")


def heldout_inputs(lock, k, fstar, dev):
    """Representation of the locked design's base on held-out datasets: (sparse dict, complete dict, ys, sizes)."""
    fr = G.heldout_table(F)
    ys, sizes, Es, Ec = {}, {}, {}, {}
    for d in fr.itertuples():
        pt, keep = A.load_contrast(d.accession)
        ys[d.accession] = (pt.role == "case").to_numpy(np.float32)
        sizes[d.accession] = [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
        Es[d.accession] = np.load(need(G.WORKG / "heldout" / f"{d.accession}_{G.ftag(fstar)}_embedding.npy"))
        Ec[d.accession] = A.features(d.accession, "embedding")[0]
    base = lock["base"] if lock["candidate"] != "H1" else "h1"
    if lock["candidate"] == "H1" or (isinstance(base, str) and base.endswith(".pt")):
        import torch
        st = torch.load(need(CKPT / f"final_{k}.pt"), map_location="cpu", weights_only=False)
        ad = adapter_from_state(st, dev)
        Es = {a: apply_adapter(ad, z, dev) for a, z in Es.items()}
        Ec = {a: apply_adapter(ad, z, dev) for a, z in Ec.items()}
    elif isinstance(base, str) and base.startswith("g_locked:"):
        rep = base.split(":")[1]
        if rep == "imputed":
            Es = {a: np.load(need(G.WORKG / "heldout" / f"{a}_{G.ftag(fstar)}_imputed.npy")) for a in Es}
        elif rep == "latent":
            Es = {a: z[:, :1536] for a, z in Es.items()}
        Ec = None
    return Es, Ec, ys, sizes


def stage_heldout(device, k):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    lp = OUTH / (f"locked_design_{DATE}.json" if k == 1 else f"locked_design_{DATE}_{k}.json")
    for p_ in (Path(__file__), lp, DEV_LOG):
        if not G.git_committed(p_):
            sys.exit(f"Cannot proceed - {p_} must be committed before the locked evaluation")
    lock = read_lock(k)
    if F.protocol_fingerprint() != lock["a_protocol_fingerprint"]:
        sys.exit("Cannot proceed - A's protocol changed after the lock")
    out = OUTH / f"per_draw_auc_H_eval{k}_{DATE}.csv.gz"
    if out.exists():
        sys.exit(f"locked evaluation {k} already run")
    fstar = lock["f_star"]
    Es, Ec, ys, sizes = heldout_inputs(lock, k, fstar, dev)
    tf = whiten if lock["candidate"] == "H2" else (lambda z: z)
    clf = "h3" if lock["candidate"] == "H3" else "enet"
    parts = []
    for setting, Z in (("sparse", Es), ("complete", Ec)):
        if Z is None:
            continue
        Z = {a: tf(z) for a, z in Z.items()}
        for n in A.SIZES:
            accs = [a for a in sorted(Z) if n in sizes[a]]
            d = G.protocol_per_dataset(A, Z, ys, n, dev, accs) if clf == "enet" else h3_protocol(Z, ys, n, dev, accs)
            parts.append(d.assign(setting=setting, f=fstar if setting == "sparse" else 1.0, representation="H"))
        log(f"{setting}: fitted")
    pd.concat(parts, ignore_index=True).assign(masking=lambda x: np.where(x.setting == "sparse", G.LABEL, "complete profiles")).to_csv(out, index=False)
    log(f"locked evaluation {k} written")


def stage_analyse(k):
    lock = read_lock(k)
    fstar = lock["f_star"]
    h = pd.read_csv(need(OUTH / f"per_draw_auc_H_eval{k}_{DATE}.csv.gz"))
    g = pd.read_csv(need(G.OUTG / f"per_draw_auc_heldout_{G.DATE}.csv.gz"))
    a = pd.read_csv(need(F.OUTA / f"per_draw_auc_{F.DATE}.csv.gz"))
    fr = G.heldout_table(F).set_index("accession")
    gl = g_lock()
    rows, res = [], {"lock": lock, "primary": None, "secondary": {}}
    for setting in ("sparse", "complete"):
        hh = h[h.setting == setting]
        if not len(hh):
            continue
        for n in A.SIZES:
            wh = hh[hh.n_train == n].groupby("accession").auc.mean().rename("H") - 0.5
            if setting == "sparse":
                gg = g[(g.f == fstar) & (g.n_train == n)].groupby(["accession", "representation"]).auc.mean().unstack() - 0.5
                comps = G.COMP_REPS
                ref = gl["representation"]
                w = gg.join(wh, how="inner")
                w = w.rename(columns={ref: f"G_{ref}"}) if ref in w else w
                reps = ["H", f"G_{ref}"] + comps
            else:
                aa = a[(a.n_train == n) & a.accession.isin(fr.index)].groupby(["accession", "representation"]).auc.mean().unstack() - 0.5
                comps = ["raw", "clocks"]
                w = aa.join(wh, how="inner")
                reps = ["H", "embedding"] + comps
            sets = {"primary": list(w.index), "external": [i for i in w.index if fr.loc[i, "group"] == "external"],
                    "corpus (test, val)": [i for i in w.index if fr.loc[i, "group"] in ("test", "val")]}
            if setting == "complete":
                sets["A's 21 datasets"] = list(w.dropna(subset=["clocks"]).index)
            for name, ids in sets.items():
                ww = w.reindex(ids)
                r = A.compare(ww, [x for x in reps if x in ww], "H", comps, f"{setting}|{name}|n={n}")
                if r is None:
                    continue
                decision = setting == "sparse" and name == "primary" and n == A.PRIMARY_SIZE
                if decision:
                    res["primary"] = r
                else:
                    res["secondary"][f"{setting}|{name}|n={n}"] = r
                rows.append({"setting": setting, "f": fstar if setting == "sparse" else 1.0, "set": name, "n_train": n,
                             "datasets": r["datasets"], "M_H": r["methylgpt_M"], **{f"M_{x}": r["M"].get(x) for x in reps},
                             "best_comparator": r["best_comparator"], "best_comparator_M": r["best_comparator_M"],
                             "difference": r["difference"], "difference_ci95_low": r["difference_ci95"][0],
                             "difference_ci95_high": r["difference_ci95"][1], "relative_gain": r["relative_gain"],
                             "meets_rule": r["pass"], **{f"H_minus_{x}": v["difference"] for x, v in r["vs_each"].items()},
                             **{f"H_minus_{x}_ci95_low": v["ci95"][0] for x, v in r["vs_each"].items()},
                             **{f"H_minus_{x}_ci95_high": v["ci95"][1] for x, v in r["vs_each"].items()},
                             "decision_set": decision, "masking": G.LABEL if setting == "sparse" else "complete profiles"})
    pd.DataFrame(rows).to_csv(OUTH / f"summary_eval{k}_{DATE}.csv", index=False)
    p = res["primary"]
    pd.DataFrame([{"analysis": f"H (locked evaluation {k}): {lock['candidate']} under G's sparse setting f = {fstar} ({G.LABEL}), n = 50",
                   "datasets": p["datasets"], "methylgpt_value": p["methylgpt_M"], "best_comparator": p["best_comparator"],
                   "best_comparator_value": p["best_comparator_M"], "relative_gain": p["relative_gain"],
                   "difference": p["difference"], "difference_ci95_low": p["difference_ci95"][0],
                   "difference_ci95_high": p["difference_ci95"][1],
                   "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
                   "outcome": "PASS" if p["pass"] else "FAIL", "masking": G.LABEL}]).to_csv(OUTH / f"rule_outcome_eval{k}_{DATE}.csv", index=False)
    write_json(OUTH / f"results_eval{k}_{DATE}.json", res)
    log(f"PRIMARY (H lock {k}): {p['methylgpt_M']:.4f} vs {p['best_comparator']} {p['best_comparator_M']:.4f}, "
        f"gain {p['relative_gain']:.3f}, CI {p['difference_ci95']}, {'PASS' if p['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["bank", "train", "probe", "lock", "final", "heldout", "analyse"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--phase", default="dev", choices=["dev", "final"])
    ap.add_argument("--config", default="{}")
    ap.add_argument("--candidate", default="H2", choices=["H2", "H3"])
    ap.add_argument("--base", default="embedding")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lock", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "bank":
        stage_bank(a.phase, a.device)
    elif a.stage == "train":
        stage_train(a.device, a.config, a.tag)
    elif a.stage == "probe":
        stage_probe(a.device, a.candidate, a.base, a.tag)
    elif a.stage == "lock":
        stage_lock(a.lock)
    elif a.stage == "final":
        stage_final(a.device, a.lock)
    elif a.stage == "heldout":
        stage_heldout(a.device, a.lock)
    else:
        stage_analyse(a.lock)


if __name__ == "__main__":
    main()
