#!/usr/bin/env python3
"""Paired comparison of MethylGPT and donor nearest neighbours on the GSE308517 benchmark.

Input: eval/gse308517_cf_dense_step2000_2026-09-30.json (per-profile mean absolute error for each
mask seed and masking rate; 136 profiles). For each rate, per-profile MAE is averaged over the
three mask seeds; the difference MethylGPT minus donor kNN (k = 5) is summarised by its mean, the
number of profiles where MethylGPT is lower, and a 95% percentile interval from resampling profiles
(2,000 draws, seed 20261002).
Output: eval/gse308517_paired_bootstrap_2026-10-02.json
"""
import json

import numpy as np

from methylgpt_lib import ARTIFACTS, write_json


def main():
    g = json.loads((ARTIFACTS / "eval/gse308517_cf_dense_step2000_2026-09-30.json").read_text())["conditions"]
    seeds = sorted({k.split("_")[0] for k in g})
    rng = np.random.default_rng(20261002)
    out = {"comparison": "methylgpt2 minus donor_knn_k5, per-profile MAE averaged over mask seeds", "seeds": seeds, "rates": {}}
    for rate in ("15", "30", "50", "70"):
        per = {m: np.mean([g[f"{s}_rate{rate}_per_sample"][m] for s in seeds], axis=0) for m in ("methylgpt2", "donor_knn_k5")}
        d = per["methylgpt2"] - per["donor_knn_k5"]
        draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out["rates"][rate] = {"n_profiles": int(len(d)), "mean_difference": float(d.mean()),
                              "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                              "profiles_methylgpt_lower": int((d < 0).sum())}
        print(rate, out["rates"][rate])
    write_json(ARTIFACTS / "eval/gse308517_paired_bootstrap_2026-10-02.json", out)


if __name__ == "__main__":
    main()
