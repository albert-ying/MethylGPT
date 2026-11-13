#!/usr/bin/env python3
"""Large-gain analysis W: MethylGPT against named standard baselines, beside the strongest baseline, for the Fig. 6
analyses. Tables only. W selects rows the source analyses already computed with their own pre-specified intervals
(J, K, L, M, O summaries; G's per-comparator contrasts) and computes one missing interval from stored draws (N: MethylGPT
against the noisy values used directly, from N's bootstrap draws). No model is fitted and no participant-level file is read.

Written on 10 Nov 2026 after the results of A to V were known, so that each figure can name the baselines a reader expects
next to the strongest one. The decisions of the source analyses and of R are unchanged; W adds named contrasts only.
Every row is reported.

NAMED BASELINES
  elastic_net_cpg: elastic-net Cox on CpGs, sksurv Coxnet with nested cross-validation (raw_cpg_v12; the same estimator and
    tuning as MethylGPT's embedding head), better of mean and nearest-neighbour fill where both exist. J, L, M, O. K is
    reported but flagged not usable: 12 of K's 65 raw_cpg_v12 fold models stopped early (11 carry age and sex only);
    K's complete_v12_paths_mean statistic (10 diseases with complete paths) is the usable K row.
  grimage2_cohort_mean, dunedinpace_cohort_mean: the clock with missing CpGs filled by the development-set mean
    (grimage2_mean, dunedinpace_mean). J, K, L, M, O.
  strongest_single: R's best single-clock or raw-CpG comparator of the scenario (context row).
  noisy_values: N, the noisy values used directly (raw_noisy); strongest: corpus-kNN denoising (knn_denoised).
  raw_mean_imputation, clocks: G, raw CpGs with mean imputation and the clocks; strongest: kNN imputation (raw_knn).
  D and D2 mean filling is not tabulated: an imputed CpG with a constant value gives a correlation of 0 by construction.

INTERVALS: each source's own (J, M: shared participant bootstrap; K, L, O: their bootstraps; G: paired dataset bootstrap,
  A.compare; N: N's participant bootstrap draws, mean over the eight endpoints per draw, percentile interval).
RULE ARITHMETIC (reference): J's rule against the named baseline; not applicable when the baseline value is <= 0.

CHECKS (abort on failure)
  1. Every selected summary row exists, and its stored relative gain equals the recomputed one within 1e-9.
  2. N: the MethylGPT-minus-best-comparator interval recomputed from N's draws equals N_rule_outcome within 1e-9, for
     every scenario of the eight-endpoint set.

AMENDMENT 1 (10 Nov 2026, the first run stopped at check 2 before any table was written): N's best comparator is the
noisy values at Gaussian SD 0.05 and corpus-kNN denoising elsewhere; check 2 and the strongest_single row now use N's
best comparator of each scenario instead of corpus-kNN denoising in every scenario.
  3. strongest_single equals R's best_single_comparator.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-10"
OUT = ART / "large_gain_2026-11-10/W"
COMBINED = "grimage2_dunedinpace_"


def need(p):
    p = ART / p
    if not p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return p


CHECKS, ROWS = [], []


def chk(n, ok, d=""):
    CHECKS.append({"check": n, "passed": bool(ok), "detail": str(d)})


def rule(vm, vb, lo):
    if not (np.isfinite(vb) and vb > 0):
        return "not applicable (baseline <= 0)"
    return "meets" if ((vm - vb) / vb >= 0.10 and lo > 0) else "does not meet"


def add(analysis, scenario, metric, label, model, vm, vb, diff, lo, hi, usable=True, note="", source=""):
    ROWS.append({"analysis": analysis, "scenario": scenario, "metric": metric, "baseline": label, "model": model,
                 "methylgpt_value": vm, "baseline_value": vb, "difference": diff, "ci95_low": lo, "ci95_high": hi,
                 "relative_gain": (vm - vb) / vb if vb > 0 else np.nan, "rule_arithmetic": rule(vm, vb, lo),
                 "usable": usable, "note": note, "source": source})


R = pd.read_csv(need("large_gain_2026-11-08/R/R_rule_outcome_2026-11-08.csv"))
SRC = {"J": ("large_gain_2026-10-28/J/J_summary_2026-10-28.csv", "scenario", "M_model", "M_mgpt2_combined", "C-index gain above age and sex"),
       "L": ("large_gain_2026-10-30/L/L_summary_2026-10-30.csv", "scenario", "H_model", "H_mgpt2_combined", "log hazard ratio per SD"),
       "M": ("large_gain_2026-11-02/M/M_summary_2026-11-02.csv", "scenario", "M_model", "M_mgpt2_combined", "C-index gain above age and sex"),
       "O": ("large_gain_2026-11-08/O/O_summary_2026-11-08.csv", "scenario", "H_model", "H_mgpt2_combined", "log hazard ratio per SD"),
       "K": ("large_gain_2026-10-30/K/K_summary_2026-10-30.csv", "setting", "M_model", "M_mgpt2_combined", "C-index gain above age and sex, mean over diseases")}


def summary_rows():
    for a, (f, key, val, mg, metric) in SRC.items():
        s = pd.read_csv(need(f))
        if a == "K":
            s = s[s.statistic.isin(["mean", "complete_v12_paths_mean"])]
        groups = s.groupby([key, "statistic"]) if a == "K" else s.groupby(key)
        for g, x in groups:
            scen, stat = (g if a == "K" else (g, "mean"))
            dec = x[x.role == "decision comparator"]
            pick = {}
            v12 = dec[dec.model.str.startswith("raw_cpg_v12_")]
            if len(v12):
                pick["elastic_net_cpg"] = v12.loc[v12[val].idxmax()]
            for lab, m in (("grimage2_cohort_mean", "grimage2_mean"), ("dunedinpace_cohort_mean", "dunedinpace_mean")):
                y = dec[dec.model == m]
                if len(y):
                    pick[lab] = y.iloc[0]
            single = dec[~dec.model.str.startswith(COMBINED)]
            best = single.loc[single[val].idxmax()]
            if stat == "mean":
                rr = R[(R.analysis == a) & ((R.setting == scen) & (R.statistic == "mean") if a == "K" else (R.scenario == scen))]
                chk("3 strongest_single equals R", len(rr) and rr.best_single_comparator.iloc[0] == best.model, f"{a} {scen}")
                pick["strongest_single"] = best
            for lab, r in pick.items():
                ok = (not r[val] > 0) or abs(r.relative_gain - (r[mg] - r[val]) / r[val]) < 1e-9
                chk("1 summary row", ok, f"{a} {scen} {stat} {r.model}")
                usable, note = True, ""
                if a == "K" and lab == "elastic_net_cpg" and stat == "mean":
                    usable, note = False, "12 of 65 raw_cpg_v12 fold models stopped early (11 age and sex only); use complete_v12_paths_mean"
                if a == "K" and stat == "complete_v12_paths_mean":
                    note = "10 diseases with complete raw_cpg_v12 paths"
                add(a, scen if stat == "mean" else f"{scen} ({stat})", metric, lab, r.model, r[mg], r[val], r.difference,
                    r.ci95_low, r.ci95_high, usable, note, f)


def n_rows():
    ro = pd.read_csv(need("large_gain_2026-11-02/N/N_rule_outcome.csv"))
    ro = ro[ro.endpoint_set == "eight endpoints"]
    d = pd.read_csv(need("large_gain_2026-11-02/N/N_bootstrap_draws.csv.gz"))
    for scen, x in d.groupby("scenario"):
        m = x.groupby(["method", "draw"]).rho.mean().unstack(0)
        r = ro[ro.scenario == scen]
        if not len(r):
            continue
        r = r.iloc[0]
        bc = r.best_comparator                       # amendment 1: N's best comparator of the scenario
        dk = m["methylgpt"] - m[bc]
        ok = (abs(np.percentile(dk, 2.5) - r.difference_ci95_low) < 1e-9 and abs(np.percentile(dk, 97.5) - r.difference_ci95_high) < 1e-9)
        chk("2 N interval reproduction", ok, scen)
        dn = m["methylgpt"] - m["raw_noisy"]
        add("N", scen, "Spearman correlation of per-CpG Cox z with the full-data z", "noisy_values", "raw_noisy",
            r.methylgpt_value, r.raw_noisy_value, r.methylgpt_value - r.raw_noisy_value, float(np.percentile(dn, 2.5)),
            float(np.percentile(dn, 97.5)), source="large_gain_2026-11-02/N/N_bootstrap_draws.csv.gz")
        add("N", scen, "Spearman correlation of per-CpG Cox z with the full-data z", "strongest_single", bc,
            r.methylgpt_value, r.best_comparator_value, r.difference, r.difference_ci95_low, r.difference_ci95_high,
            source="large_gain_2026-11-02/N/N_rule_outcome.csv")


def g_rows():
    g = json.load(open(need("large_gain_2026-10-25/G/results_2026-10-25.json")))["primary"]
    for lab, c in (("raw_mean_imputation", "raw_mean"), ("clocks", "clocks"), ("strongest_single", "raw_knn")):
        v = g["vs_each"][c]
        add("G", g["set"], "mean AUC - 0.5 over 24 held-out datasets, 50 labelled samples", lab, c, g["M"][g["methylgpt"]],
            g["M"][c], v["difference"], v["ci95"][0], v["ci95"][1],
            note=f"MethylGPT higher in {v['datasets_methylgpt_higher']} of {g['datasets']} datasets",
            source="large_gain_2026-10-25/G/results_2026-10-25.json")


def main():
    summary_rows()
    n_rows()
    g_rows()
    ck = pd.DataFrame(CHECKS)
    OUT.mkdir(parents=True, exist_ok=True)
    ck.to_csv(OUT / f"W_checks_{DATE}.csv", index=False)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("W checks failed; no result table written")
    df = pd.DataFrame(ROWS)
    df.to_csv(OUT / f"W_named_baselines_{DATE}.csv", index=False)
    print(f"W: {len(ck)} checks pass; {len(df)} rows")


if __name__ == "__main__":
    main()
