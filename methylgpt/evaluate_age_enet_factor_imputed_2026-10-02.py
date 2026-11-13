#!/usr/bin/env python3
"""Elastic net on AltumAge test profiles with missing CpGs filled by the MethylGPT factor model.

The elastic net is the one trained on the same labels as MethylGPT (baseline_corpus_age_2026-10-01.py).
Profiles, scenarios and masks are those of evaluate_age_revision_2026-10-01.py (its degrade() is
imported). For each profile, posterior factor scores are computed from the CpGs that remain
(ppca_core_meanfilled_2026-09-30, rank 1,536) and each missing value is replaced by its
reconstruction mu + V s, clipped to [0, 1]; observed values are kept. The elastic net with
training-mean filling is recomputed alongside as a check against the main evaluation.
Output: eval/age_enet_factor_imputed_2026-10-02.json
Run: python3 evaluate_age_enet_factor_imputed_2026-10-02.py --device cuda:1
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, PPCACore, Finv, age_scores, load_altumage_split, log, require, write_json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("age_revision", HERE / "evaluate_age_revision_2026-10-01.py")
age_revision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(age_revision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--scenarios", default="all,random_30,random_50,random_70,random_90,hm27,hm450,epic_v2")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/age_enet_factor_imputed_2026-10-02.json"))
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    dev = torch.device(a.device)
    core = PPCACore.load(ARTIFACTS / "ppca_core_meanfilled_2026-09-30.npz", rank=1536).to(dev)
    z = np.load(require(ARTIFACTS / "eval/baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))

    def enet(T):
        return Finv(((T - z["mu"]) / z["sd"]) @ z["w"] * float(z["ys"]) + float(z["ym"]))

    _, Xte, yte = load_altumage_split("test")
    res = {"n": int(len(yte)), "core": "ppca_core_meanfilled_2026-09-30.npz (rank 1536)", "scenarios": {}}
    for s in a.scenarios.split(","):
        X = age_revision.degrade(Xte, s)
        filled = np.empty_like(X)
        for i in range(0, len(X), 64):
            x = torch.from_numpy(X[i:i + 64]).to(dev)
            obs = torch.isfinite(x)
            rec = core.predict(core.posterior(x, obs.float())).clamp(0, 1)
            filled[i:i + 64] = torch.where(obs, x, rec).cpu().numpy()
        mean_filled = np.where(np.isfinite(X), X, z["mu"])
        res["scenarios"][s] = {"observed_fraction": float(np.isfinite(X).mean()),
                               "enet_mean_filled": age_scores(enet(mean_filled), yte),
                               "enet_factor_filled": age_scores(enet(filled), yte)}
        log(f"{s}: observed {res['scenarios'][s]['observed_fraction']:.3f}; elastic net mean-filled "
            f"{res['scenarios'][s]['enet_mean_filled']['medae']:.2f}, factor-filled {res['scenarios'][s]['enet_factor_filled']['medae']:.2f}")
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
