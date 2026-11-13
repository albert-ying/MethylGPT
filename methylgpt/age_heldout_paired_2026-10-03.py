#!/usr/bin/env python3
"""Paired dataset-bootstrap comparison of MethylGPT and the elastic net on held-out test datasets.

Input: eval/age_revision_2026-10-01.predictions.npz (held-out predictions, same row order as the
evaluation: test-split profiles with metadata ages). For all 34 datasets and for the 21 with reliable
metadata ages, the difference in median and in mean absolute error (MethylGPT minus elastic net) is
summarised with a 95% percentile interval from resampling datasets (2,000 draws, seed 20261002), and
the number of datasets in which MethylGPT has the lower median absolute error.
Output: eval/age_heldout_paired_2026-10-03.json
"""
import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, write_json


def paired(ea, eb, groups, stat, n=2000, seed=20261002):
    rng = np.random.default_rng(seed)
    idx = pd.Series(np.arange(len(ea))).groupby(groups).apply(np.asarray).tolist()
    d = [stat(ea[k]) - stat(eb[k]) for k in (np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(n))]
    return {"methylgpt": float(stat(ea)), "elastic_net": float(stat(eb)), "difference": float(stat(ea) - stat(eb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    z = np.load(ARTIFACTS / "eval/age_revision_2026-10-01.predictions.npz")
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["split", "group"])
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    y, flag = z["heldout_age"], z["heldout_flag"]
    assert len(rows) == len(y)
    grp = prof.group.astype(str).to_numpy()[rows]
    ea, eb = np.abs(z["MethylGPT|heldout"] - y), np.abs(z["ElasticNet (same labels)|heldout"] - y)
    out = {}
    for name, m in (("all_34_datasets", np.ones(len(y), bool)), ("reliable_ages_21_datasets", ~flag)):
        per = pd.DataFrame({"g": grp[m], "a": ea[m], "b": eb[m]}).groupby("g").median()
        out[name] = {"n_profiles": int(m.sum()), "n_datasets": int(per.shape[0]),
                     "datasets_methylgpt_lower_medae": int((per.a < per.b).sum()),
                     "medae": paired(ea[m], eb[m], grp[m], np.median), "mae": paired(ea[m], eb[m], grp[m], np.mean)}
    write_json(ARTIFACTS / "eval/age_heldout_paired_2026-10-03.json", out)
    print(out)


if __name__ == "__main__":
    main()
