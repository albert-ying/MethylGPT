#!/usr/bin/env python3
"""Published clocks on the held-out test datasets (datasets never used in training).

Profiles: test-split profiles with metadata ages (34 datasets), as in evaluate_age_revision_2026-10-01.py.
Clock predictions come from ClockBase (full-array Horvath multi-tissue, Hannum, Lin, Horvath skin and
blood; pyaging AltumAge and principal-component clocks) via clock_table() of that script. Each clock is
compared with MethylGPT and the elastic net on the profiles it scores, for all datasets and for the
datasets with reliable metadata ages; MedAE differences use 2,000 dataset bootstrap draws (seed 20261004).
Datasets in which ClockBase returned the same value for every sample (failed scoring) are excluded for that clock.
Output: eval/age_heldout_clocks_2026-10-04.json
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, write_json

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("age_revision", HERE / "evaluate_age_revision_2026-10-01.py")
ar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ar)


def paired_medae(a, b, y, g, n=2000, seed=20261004):
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(a - y), np.abs(b - y)
    idx = pd.Series(np.arange(len(y))).groupby(g).apply(np.asarray).tolist()
    d = [np.median(ea[k]) - np.median(eb[k]) for k in (np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(n))]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    z = np.load(ARTIFACTS / "eval/age_revision_2026-10-01.predictions.npz")
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "split", "group"])
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    ids = prof.id.to_numpy()[rows].astype(str)
    grp = prof.group.astype(str).to_numpy()[rows]
    y, flag = z["heldout_age"], z["heldout_flag"]
    assert len(ids) == len(y)
    mg, en = z["MethylGPT|heldout"], z["ElasticNet (same labels)|heldout"]
    ct = ar.clock_table(list(ids))
    out = {}
    for c in [c for c in ct.columns if c not in ("zhang_observed_fraction", "Zhang 2019")]:
        v = ct[c].to_numpy(float).copy()
        # ClockBase returns one constant value for every sample of some datasets (failed scoring, e.g. 61.70 years for
        # Horvath 2013); those datasets are excluded for that clock.
        const = pd.Series(v).groupby(grp).transform(lambda s: s.dropna().nunique() <= 1 and s.notna().sum() > 1).to_numpy(bool)
        out.setdefault("excluded_constant_output_datasets", {})[c] = sorted(set(grp[const & np.isfinite(v)]))
        v[const] = np.nan
        for name, m0 in (("all", np.ones(len(y), bool)), ("reliable_ages", ~flag)):
            m = m0 & np.isfinite(v)
            if m.sum() < 50:
                continue
            out.setdefault(c, {})[name] = {"n_profiles": int(m.sum()), "n_datasets": int(pd.Series(grp[m]).nunique()),
                                           "clock_medae": float(np.median(np.abs(v[m] - y[m]))),
                                           "methylgpt_medae": float(np.median(np.abs(mg[m] - y[m]))),
                                           "elastic_net_medae": float(np.median(np.abs(en[m] - y[m]))),
                                           "methylgpt_minus_clock_ci95": paired_medae(mg[m], v[m], y[m], grp[m])}
    write_json(ARTIFACTS / "eval/age_heldout_clocks_2026-10-04.json", out)
    print("excluded:", out["excluded_constant_output_datasets"])
    for c, d in out.items():
        if c == "excluded_constant_output_datasets":
            continue
        for k, s in d.items():
            print(f"{c} [{k}] n={s['n_profiles']} ({s['n_datasets']} datasets): clock {s['clock_medae']:.2f} MethylGPT {s['methylgpt_medae']:.2f} "
                  f"EN {s['elastic_net_medae']:.2f} CI {[round(x, 2) for x in s['methylgpt_minus_clock_ci95']]}")


if __name__ == "__main__":
    main()
