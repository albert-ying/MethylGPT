#!/usr/bin/env python3
"""Clean corpus metadata ages before age supervision (rule fixed before any retraining).

Problem found: 11.8% of AltumAge profiles carry corpus metadata ages >5 years from their
curated AltumAge ages, mostly placenta/cord-blood datasets annotated with maternal age.

Rule
  1. AltumAge profiles: use the curated AltumAge age.
  2. Every other dataset group with metadata ages: compare metadata ages with a reference
     ElasticNet clock trained only on AltumAge training profiles (curated ages; alpha chosen
     on AltumAge validation). Groups whose median |metadata - reference| exceeds 10 years
     are flagged and their metadata ages are not used.
  3. Ages outside [0, 120] are dropped.
The same group flag is applied to held-out test-split datasets for evaluation; results are
reported with and without the flag.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import (cuda_preinit, CORPUS_DIR, Corpus, F, Finv, age_scores, load_altumage_split, log,
                            write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max-group-mad", type=float, default=10.0)
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    corpus = Corpus()
    prof = corpus.profiles.copy()
    ids_tr, Xtr, ytr = load_altumage_split("train")
    ids_va, Xva, yva = load_altumage_split("valid")
    ids_te, _, yte = load_altumage_split("test")
    curated = pd.Series(np.concatenate([ytr, yva, yte]), index=np.concatenate([ids_tr, ids_va, ids_te]))

    T = torch.from_numpy(Xtr).to(dev)
    fin = torch.isfinite(T)
    mu = torch.where(fin, T, torch.zeros((), device=dev)).sum(0) / fin.sum(0).clamp_min(1)
    T = torch.where(fin, T, mu)
    sd = T.std(0).clamp_min(1e-4)
    T = (T - mu) / sd
    y = F(ytr); ym, ys = y.mean(), y.std()
    t = torch.from_numpy(((y - ym) / ys).astype(np.float32)).to(dev)

    def prep(X):
        Z = torch.from_numpy(X).to(dev)
        return (torch.where(torch.isfinite(Z), Z, mu) - mu) / sd

    Va = prep(Xva)
    n = T.shape[0]
    v = torch.randn(T.shape[1], device=dev)
    for _ in range(40):
        v = T.T @ (T @ v); v /= v.norm()
    L0 = float((T @ v).norm() ** 2 / n)
    l1 = 0.5
    amax = float((T.T @ t).abs().max() / (n * l1))
    w = torch.zeros(T.shape[1], device=dev)
    best = None
    for alpha in amax * np.logspace(-1.5, -3.5, 9):
        step = 1.0 / (L0 + alpha * (1 - l1))
        yv, tk = w.clone(), 1.0
        for _ in range(1500):
            g = T.T @ (T @ yv - t) / n + alpha * (1 - l1) * yv
            u = yv - step * g
            wn = torch.sign(u) * torch.clamp(u.abs() - step * alpha * l1, min=0)
            tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
            yv = wn + ((tk - 1) / tk1) * (wn - w)
            w, tk = wn, tk1
        sc = age_scores(Finv((Va @ w).cpu().numpy() * ys + ym), yva)
        if best is None or sc["medae"] < best[1]["medae"]:
            best = (float(alpha), sc, w.clone())
    alpha, sc_ref, w = best
    log(f"reference clock (AltumAge train only): alpha {alpha:.4g}, valid {sc_ref}")

    meta = prof.age.where((prof.age >= 0) & (prof.age <= 120))
    rows = np.flatnonzero(meta.notna().to_numpy())
    pred = np.empty(len(rows))
    for i in range(0, len(rows), 4096):
        Z = prep(corpus.get(rows[i:i + 4096]))
        pred[i:i + 4096] = Finv((Z @ w).cpu().numpy() * ys + ym)
    ref = pd.Series(np.nan, index=prof.index)
    ref.iloc[np.sort(rows)] = pred
    prof["age_meta"] = meta
    prof["age_reference_clock"] = ref
    prof["age_curated"] = prof.id.map(curated)
    dev_abs = (prof.age_meta - prof.age_reference_clock).abs()
    gstat = dev_abs.groupby(prof.group).median()
    flagged = set(gstat[gstat > a.max_group_mad].index)
    prof["age_group_flag"] = prof.group.isin(flagged)
    prof["age_clean"] = np.where(prof.age_curated.notna(), prof.age_curated,
                                 np.where(prof.age_group_flag, np.nan, prof.age_meta))
    out = CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"
    prof[["id", "group", "split", "age_meta", "age_curated", "age_reference_clock", "age_group_flag", "age_clean"]].to_parquet(out, index=False)
    summ = {"rule": __doc__.split("Rule")[1].strip(), "reference_clock_valid": sc_ref, "reference_alpha": alpha,
            "groups_with_meta_age": int(gstat.size), "groups_flagged": len(flagged),
            "profiles_flagged_by_split": prof[prof.age_group_flag & prof.age_meta.notna()].groupby("split").size().to_dict(),
            "clean_ages_by_split": prof[prof.age_clean.notna()].groupby("split").size().to_dict(),
            "flagged_groups": sorted(flagged), "elapsed_s": time.time() - t0}
    write_json(CORPUS_DIR / "age_cleaning_2026-10-01.json", summ)
    log(f"groups flagged {len(flagged)}/{gstat.size}; clean ages by split {summ['clean_ages_by_split']}; "
        f"flagged profiles by split {summ['profiles_flagged_by_split']}")


if __name__ == "__main__":
    main()
