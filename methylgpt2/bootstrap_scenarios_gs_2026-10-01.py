#!/usr/bin/env python3
"""Paired bootstrap of the combined MethylGPT-2 model against baselines in every missing-CpG scenario.

Risks come from scenario_gs_combined_2026-10-01.py (MethylGPT-2 models, raw CpG v12 and GPU) and
scenario_clocks_cox_2026-10-01.py (GrimAge2, DunedinPACE, both). For each scenario, endpoint and
comparator, subjects are resampled within outer folds (draws as set, seed 20260928 + task index)
and the pooled within-fold C-index difference is recorded. BH adjustment over all contrasts.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

from methylgpt2_lib import ARTIFACTS, log, require, write_json

MODEL = "mgpt2_combined"
COMPARATORS = ["raw_cpg_v12", "raw_cpg_gpu", "grimage2", "dunedinpace", "grimage2_dunedinpace", "age_sex"]


def counts(ev, du, r):
    _, c, d, t, _ = concordance_index_censored(ev, du, r)
    return np.array([c, d, t], float)


def cidx(c):
    return (c[0] + 0.5 * c[2]) / c.sum()


def bh(p):
    p = np.asarray(p); n = len(p); order = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def task(args):
    k, scen, ep, d, draws = args
    rng = np.random.default_rng(20260928 + k)
    ev, du = d.event.astype(bool).to_numpy(), d.duration.to_numpy(float)
    folds = [np.flatnonzero(d.outer_fold.to_numpy() == f) for f in sorted(d.outer_fold.unique())]
    r1 = d[MODEL].to_numpy(float)
    out = []
    samples = [[rng.choice(f, len(f), replace=True) for f in folds] for _ in range(draws)]
    c1 = cidx(sum(counts(ev[f], du[f], r1[f]) for f in folds))
    boot1 = [sum(counts(ev[b], du[b], r1[b]) for b in s) for s in samples]
    for comp in COMPARATORS:
        r0 = d[comp].to_numpy(float)
        c0 = cidx(sum(counts(ev[f], du[f], r0[f]) for f in folds))
        diffs = np.array([cidx(b1) - cidx(sum(counts(ev[b], du[b], r0[b]) for b in s)) for b1, s in zip(boot1, samples)])
        p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
        out.append({"scenario": scen, "endpoint": ep, "comparator": comp, "c_mgpt2_combined": c1, "c_comparator": c0,
                    "difference": c1 - c0, "ci95_low": float(np.percentile(diffs, 2.5)),
                    "ci95_high": float(np.percentile(diffs, 97.5)), "p": p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/bootstrap_scenarios_mgpt2_combined_2026-10-01.csv"))
    a = ap.parse_args()
    risks = pd.read_parquet(require(ARTIFACTS / "gs_cox/scenario_combined_2026-10-01/private_scenario_risks.parquet"))
    risks["dnam_id"] = risks.dnam_id.astype(str)
    risks["age_sex"] = risks.baseline_risk
    ck = pd.read_parquet(require(ARTIFACTS / "gs_cox/private_clock_scenario_risks_2026-10-01.parquet"))
    ck["dnam_id"] = ck.dnam_id.astype(str)
    ck = ck.pivot_table(index=["scenario", "endpoint", "dnam_id", "outer_fold"], columns="model", values="risk").reset_index()
    m = risks.merge(ck, on=["scenario", "endpoint", "dnam_id", "outer_fold"], how="inner", validate="one_to_one")
    if len(m) != len(risks):
        raise ValueError(f"clock risks missing for {len(risks) - len(m)} subject-scenario rows")
    jobs = [(k, scen, ep, g.reset_index(drop=True), a.draws)
            for k, ((scen, ep), g) in enumerate(m.groupby(["scenario", "endpoint"], sort=True))]
    log(f"{len(jobs)} scenario x endpoint tasks, {len(COMPARATORS)} comparators each, {a.draws} draws")
    with Pool(a.workers) as pool:
        rows = [r for block in pool.map(task, jobs) for r in block]
    df = pd.DataFrame(rows)
    df["q_bh"] = bh(df.p.to_numpy())
    df.to_csv(a.out, index=False)
    summ = {}
    for (scen, comp), g in df.groupby(["scenario", "comparator"]):
        summ.setdefault(scen, {})[comp] = {"better": int((g.ci95_low > 0).sum()), "worse": int((g.ci95_high < 0).sum()),
                                           "point_higher": int((g.difference > 0).sum()), "mean_difference": float(g.difference.mean())}
    write_json(Path(a.out).with_suffix(".json"), {"draws": a.draws, "summary": summ})
    for scen, v in summ.items():
        log(f"{scen}: " + "; ".join(f"{c} +{s['better']}/-{s['worse']} (higher {s['point_higher']}/8, mean {s['mean_difference']:+.4f})" for c, s in v.items()))


if __name__ == "__main__":
    main()
