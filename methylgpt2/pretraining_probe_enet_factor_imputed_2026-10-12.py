#!/usr/bin/env python3
"""Limited-label elastic nets on raw CpGs, scored with missing CpGs filled by MethylGPT's factor model.

Companion to pretraining_frozen_probe_2026-10-12.py. That script fills the elastic net's missing test values with
the fitting means, which is the usual practice but denies the comparator the factor model. Here the same elastic
nets (same label subsets, settings and seeds) are also scored after each missing test value is replaced by the
factor model's reconstruction from the profile's remaining CpGs (ppca_core_meanfilled_2026-09-30, rank 1,536;
clipped to [0, 1]; observed values kept), as in evaluate_age_enet_factor_imputed_2026-10-02.py. Training profiles
are mean-filled as before. The mean-filled scores must reproduce the stored probe results.
Output: artifacts/eval/pretraining_probe_enet_factor_imputed_2026-10-12.csv
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, Corpus, F, Finv, PPCACore, age_scores, cuda_preinit, label_subset, load_altumage_split, log, require

sys.path.insert(0, str(Path(__file__).resolve().parent))
P = importlib.import_module("pretraining_frozen_probe_2026-10-12")


@torch.no_grad()
def factor_impute(core, X, dev, batch=256):
    out = X.copy()
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(dev)
        obs = torch.isfinite(x)
        rec = core.predict(core.posterior(x, obs.float())).float().clamp(0, 1)
        out[i:i + batch] = torch.where(obs, x, rec).cpu().numpy()
    return out


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    try:
        corpus = Corpus()
        lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
        plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
        _, Xte, yte = load_altumage_split("test")
        stored = pd.read_csv(require(ARTIFACTS / "eval/pretraining_frozen_probe_2026-10-12.csv"))
        core = PPCACore.load(require(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz"), rank=1536).to(dev)
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    prof = corpus.profiles
    age, age_meta = lab.age_clean.to_numpy(float), lab.age_meta.to_numpy(float)
    pool_mask = (prof.split == "train").to_numpy() & np.isfinite(age)
    pool = np.flatnonzero(pool_mask)
    ho_rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age_meta))
    y_pool, y_sets = age[pool], {"altumage_test": yte, "heldout": age_meta[ho_rows]}
    Xsets = {"altumage_test|all": Xte, "altumage_test|random_90": P.degrade(Xte, "random_90", plat),
             "altumage_test|hm27": P.degrade(Xte, "hm27", plat), "heldout|all": corpus.get(ho_rows)}
    Ximp = {k: factor_impute(core, X, dev) for k, X in Xsets.items()}
    log(f"factor-model imputation of the test sets done ({time.time() - t0:.0f} s)")
    Xpool = np.concatenate([corpus.get(pool[i:i + 8192]) for i in range(0, len(pool), 8192)])
    rows = []
    for n in P.SIZES:
        for seed in range(5 if n else 1):
            fit, sel = label_subset(pool_mask, n or len(pool), seed)
            fi, si = np.searchsorted(pool, fit), np.searchsorted(pool, sel)
            tfy = F(y_pool[fi])
            ym, ysd = float(tfy.mean()), float(tfy.std())
            w, mu, sd, info = P.enet_probe(Xpool[fi], (tfy - ym) / ysd, Xpool[si], y_pool[si], ym, ysd, dev)
            for k in P.SETS:
                for fill, src in (("fitting means", Xsets[k]), ("factor model", Ximp[k])):
                    T = torch.from_numpy(src).to(dev)
                    p = Finv((((torch.where(torch.isfinite(T), T, mu) - mu) / sd) @ w).cpu().numpy() * ysd + ym)
                    sc = age_scores(p, y_sets[k.split("|")[0]])
                    rows.append({"model": "elastic_net_raw", "missing test values filled by": fill, "n_labels": n or len(pool), "seed": seed, "set": k,
                                 "medae": sc["medae"], "mae": sc["mae"], "alpha": info["alpha"], "nonzero": info["nonzero"]})
            r = {(x["set"], x["missing test values filled by"]): x["medae"] for x in rows[-8:]}
            log(f"n{n or len(pool)}|s{seed}: 27K mean-filled {r[('altumage_test|hm27', 'fitting means')]:.2f}, factor-imputed {r[('altumage_test|hm27', 'factor model')]:.2f}; "
                f"90% removed {r[('altumage_test|random_90', 'fitting means')]:.2f}, {r[('altumage_test|random_90', 'factor model')]:.2f} ({time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(ARTIFACTS / "eval/pretraining_probe_enet_factor_imputed_2026-10-12.csv", index=False)
    df = pd.DataFrame(rows)
    chk = df[df["missing test values filled by"] == "fitting means"].merge(stored[stored.model == "elastic_net_raw"], on=["n_labels", "seed", "set"], suffixes=("", "_stored"))
    log(f"mean-filled scores against the stored probe: {len(chk)} values, largest difference {float((chk.medae - chk.medae_stored).abs().max()):.4f} years")


if __name__ == "__main__":
    main()
