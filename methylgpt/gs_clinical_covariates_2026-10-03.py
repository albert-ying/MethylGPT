#!/usr/bin/env python3
"""Incident disease in Generation Scotland: improvement over a clinical covariate model.

Participants, endpoints and outer folds are those of the locked nested protocol (9,866 training-subset
participants, 8 endpoints, 5 outer folds). Clinical covariates measured at the methylation baseline
(meta.csv of the GS release used throughout): age, sex, mean systolic and diastolic blood pressure,
total and HDL cholesterol, glucose and creatinine.
In each outer fold, an unpenalized Cox model of the clinical covariates is fitted to the four fitting
folds (covariates standardized and missing values filled with fitting-fold medians) and scored on the
held-out fold. Methylation predictors are added with the rule that defined the combined MethylGPT model:
the equal-weight sum of within-fold standardized risks. Their out-of-fold risks come from the locked
runs (MethylGPT combined model; GrimAge2 with DunedinPACE). C-indices pool comparable pairs within
outer folds; differences use 1,000 paired bootstrap draws of participants within folds (seed 20261003)
with Benjamini-Hochberg adjustment.
Output (aggregate only): artifacts/gs_cox/clinical_covariates_2026-10-03.{json,csv}
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, log, require, write_json

META = "/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/meta.csv"
CLIN = ["avg_sys", "avg_dia", "Total_cholesterol", "HDL_cholesterol", "Glucose", "Creatinine"]
G = ARTIFACTS / "gs_cox"
RUNS = {"MethylGPT combined": G / "mgpt2_combined_2026-10-01", "GrimAge2 + DunedinPACE": G / "clock_comparators_2026-10-01/grimage2_dunedinpace"}


def load(path):
    d = pd.read_parquet(require(path / "private_nested_predictions.parquet"))
    return d[d.age_adjustment == "linear"]


def counts(ev, du, r):
    _, c, dd, tr, _ = concordance_index_censored(ev.astype(bool), du, r)
    return np.array([c, dd, tr], float)


def cidx(cnt):
    return (cnt[0] + 0.5 * cnt[2]) / cnt.sum()


def z(v):
    return (v - v.mean()) / v.std()


def main():
    t0 = time.time()
    meta = pd.read_csv(META, usecols=["dnam_id"] + CLIN).assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id")
    base = load(RUNS["MethylGPT combined"]).rename(columns={"risk": "MethylGPT combined"})
    gd = load(RUNS["GrimAge2 + DunedinPACE"])[["dnam_id", "endpoint", "risk"]].rename(columns={"risk": "GrimAge2 + DunedinPACE"})
    d = base.merge(gd, on=["dnam_id", "endpoint"], how="inner").join(meta, on="dnam_id")
    log(f"rows {len(d)}; participants {d.dnam_id.nunique()}; clinical missing fraction {d[CLIN].isna().mean().round(3).to_dict()}")
    rng = np.random.default_rng(20261003)
    rows, boots = [], []
    for ep, de in d.groupby("endpoint"):
        risks = {k: np.empty(len(de)) for k in ("Clinical", "Clinical + MethylGPT", "Clinical + GrimAge2 + DunedinPACE", "MethylGPT combined")}
        de = de.reset_index(drop=True)
        folds = []
        for f in sorted(de.outer_fold.unique()):
            held = np.flatnonzero(de.outer_fold.to_numpy() == f); fit = np.flatnonzero(de.outer_fold.to_numpy() != f)
            X = de[["age", "sex"] + CLIN].astype(float)
            med, mu, sd = X.iloc[fit].median(), None, None
            Xf = X.fillna(med); mu, sd = Xf.iloc[fit].mean(), Xf.iloc[fit].std()
            Xs = ((Xf - mu) / sd).to_numpy()
            cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit(Xs[fit], Surv.from_arrays(de.event.to_numpy()[fit].astype(bool), de.duration.to_numpy()[fit]))
            lp = Xs[held] @ cx.coef_
            risks["Clinical"][held] = lp
            risks["Clinical + MethylGPT"][held] = z(lp) + z(de["MethylGPT combined"].to_numpy()[held])
            risks["Clinical + GrimAge2 + DunedinPACE"][held] = z(lp) + z(de["GrimAge2 + DunedinPACE"].to_numpy()[held])
            risks["MethylGPT combined"][held] = de["MethylGPT combined"].to_numpy()[held]
            folds.append(held)
        ev, du = de.event.to_numpy(), de.duration.to_numpy()
        cnt = {k: sum(counts(ev[h], du[h], r[h]) for h in folds) for k, r in risks.items()}
        row = {"endpoint": ep, "events": int(ev.sum()), **{k: float(cidx(c)) for k, c in cnt.items()}}
        rows.append(row)
        for comp in ("Clinical + MethylGPT", "Clinical + GrimAge2 + DunedinPACE"):
            diffs = []
            for _ in range(1000):
                tot = {k: np.zeros(3) for k in (comp, "Clinical")}
                for h in folds:
                    s = h[rng.integers(0, len(h), len(h))]
                    for k in tot:
                        tot[k] += counts(ev[s], du[s], risks[k][s])
                diffs.append(cidx(tot[comp]) - cidx(tot["Clinical"]))
            diffs = np.asarray(diffs)
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            boots.append({"endpoint": ep, "comparison": f"{comp} minus Clinical", "difference": row[comp] - row["Clinical"],
                          "ci95_low": float(np.percentile(diffs, 2.5)), "ci95_high": float(np.percentile(diffs, 97.5)), "p": float(max(p, 1 / 1000))})
        log(f"{ep}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items() if isinstance(v, float)))
    b = pd.DataFrame(boots)
    order = np.argsort(b.p.to_numpy()); q = np.empty(len(b)); m = len(b)
    q[order] = np.minimum.accumulate((b.p.to_numpy()[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    b["q_bh"] = np.minimum(q, 1)
    t = pd.DataFrame(rows)
    t.to_csv(G / "clinical_covariates_2026-10-03.csv", index=False)
    b.to_csv(G / "clinical_covariates_bootstrap_2026-10-03.csv", index=False)
    write_json(G / "clinical_covariates_2026-10-03.json", {"covariates": ["age", "sex"] + CLIN, "means": t.drop(columns=["endpoint", "events"]).mean().to_dict(),
                                                          "per_endpoint": rows, "bootstrap": b.to_dict("records"), "elapsed_s": time.time() - t0})
    log("mean C: " + ", ".join(f"{k} {v:.4f}" for k, v in t.drop(columns=["endpoint", "events"]).mean().items()))
    for comp, g in b.groupby("comparison"):
        log(f"{comp}: significant gain (q<0.05) for {int(((g.q_bh < 0.05) & (g.difference > 0)).sum())} of 8 endpoints, loss for {int(((g.q_bh < 0.05) & (g.difference < 0)).sum())}")


if __name__ == "__main__":
    main()
