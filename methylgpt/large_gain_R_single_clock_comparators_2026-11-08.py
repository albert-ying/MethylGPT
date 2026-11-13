#!/usr/bin/env python3
"""Large-gain analysis R: the Generation Scotland (GS) decisions of J, K, L, M and O with single-clock comparators.
Tables only, from committed aggregate tables. No participant-level file is read, no model is fitted, and no estimate,
interval or bootstrap draw is recomputed: R selects among per-model rows that J, K, L, M and O already computed with their
own pre-specified bootstrap, and re-applies J's rule to the selected rows.

Written on 8 Nov 2026, AFTER the results of J, K, L, M, N, O and P were known, to compare
MethylGPT with one clock at a time and never with the model that combines GrimAge2 and DunedinPACE. The
comparator-set change is therefore post hoc and is disclosed as such; every row against the combined-clock models stays
reported (R_combined_clock_contrasts) and the original decisions are carried beside the new ones.

INPUTS (committed, already exported aggregate tables)
  J: artifacts/large_gain_2026-10-28/J/J_summary_2026-10-28.csv, J_rule_outcome_2026-10-28.csv   metric M (C-index gain)
  K: artifacts/large_gain_2026-10-30/K/K_summary_2026-10-30.csv, K_rule_outcome_2026-10-30.csv   metric M over diseases,
     four statistics (mean, event_weighted_mean, sex_restricted_mean, complete_v12_paths_mean)
  L: artifacts/large_gain_2026-10-30/L/L_summary_2026-10-30.csv, L_rule_outcome_2026-10-30.csv   metric H (log HR per SD)
  M: artifacts/large_gain_2026-11-02/M/M_summary_2026-11-02.csv, M_rule_outcome_2026-11-02.csv   metric M
  O: artifacts/large_gain_2026-11-08/O/O_summary_2026-11-08.csv, O_rule_outcome_2026-11-08.csv   metric H
  Each summary holds, per scenario (K: setting and statistic) and model, the model's metric and the paired-bootstrap
  difference MethylGPT combined (mgpt2_combined) minus that model with its 95% interval, from the analysis's own draws.
  P is not re-tabulated: every P comparator and P's decision model contain both clocks (a single-clock stacking analysis
  needs new fits and is a separate pre-specified analysis).

COMPARATOR SETS
  Original decision set = rows with role "decision comparator" (unchanged definitions from each analysis).
  Combined-clock models = any model whose name starts with "grimage2_dunedinpace_" (mean and kNN fill).
  Single-clock decision set S = original decision set minus the combined-clock models: raw_cpg_v12_{mean,knn},
    raw_cpg_gpu_{mean,knn}, grimage2_{mean,knn}, dunedinpace_{mean,knn} where the analysis has them in that scenario.
  Secondary, application and sensitivity rows (in-cohort kNN, factor imputation, corpus-kNN denoising, depth-aware,
    non-vocabulary sensitivity) stay outside the decision, as in the source analyses.

PRIMARY OUTPUT AND DECISION (R_rule_outcome)
  Per analysis and scenario (K: setting and statistic): best_single = the model in S with the largest metric (point
  estimate, as J fixes C*). Rule (J's, unchanged): PASS iff (value_MethylGPT - value_best) / value_best >= 0.10 and the
  lower end of the 95% interval of the difference is > 0; "not applicable" if value_best <= 0; otherwise FAIL. The
  interval is the source analysis's primary interval column (ci95_low, ci95_high; J and M: shared participant resample).
  The decision scenario of each analysis is its own (J hm27; K hm27 mean over primary diseases; L hm27; M depth_10;
  O depth_10); other rows are each scenario against its own best, as the source analyses report them.

NAMED SINGLE-CLOCK CONTRASTS (R_clock_contrasts)
  Per analysis and scenario: MethylGPT against GrimAge2 alone and against DunedinPACE alone, each at the better of its two
  fills in S (largest metric; in complete profiles only the mean fill exists), with the stored difference, interval,
  bootstrap P and the rule arithmetic. The best raw-CpG model is listed in the same table.

CHECKS (R_checks; any failure stops the run before outputs are written)
  1. mgpt2_combined's metric is constant within each scenario of each summary.
  2. The original best decision comparator recomputed as the argmax over the original decision set equals the one in the
     source rule-outcome table, for every row of that table.
  3. The stored relative gain equals (value_MethylGPT - value_model) / value_model within 1e-9 wherever value_model > 0.
  4. The stored per-model rule (rule_against_this_model) equals J's rule recomputed from the stored values for every
     decision comparator row.
  5. No model in S starts with "grimage2_dunedinpace_"; S is non-empty for every scenario.
  6. Each source rule-outcome row's difference and interval equal the summary row of its best comparator within 1e-12.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
OUT = ART / "large_gain_2026-11-08" / "R"
COMBINED_PREFIX = "grimage2_dunedinpace_"

SOURCES = {
    "J": dict(summary="large_gain_2026-10-28/J/J_summary_2026-10-28.csv",
              rule="large_gain_2026-10-28/J/J_rule_outcome_2026-10-28.csv",
              keys=["scenario"], value="M_model", mgpt="M_mgpt2_combined", decision={"scenario": "hm27"},
              metric="M: mean over 8 endpoints of pooled within-fold C-index minus age+sex C-index"),
    "K": dict(summary="large_gain_2026-10-30/K/K_summary_2026-10-30.csv",
              rule="large_gain_2026-10-30/K/K_rule_outcome_2026-10-30.csv",
              keys=["setting", "statistic"], value="M_model", mgpt="M_mgpt2_combined",
              decision={"setting": "hm27", "statistic": "mean"},
              metric="M over the primary disease set (13 individual diseases), per statistic"),
    "L": dict(summary="large_gain_2026-10-30/L/L_summary_2026-10-30.csv",
              rule="large_gain_2026-10-30/L/L_rule_outcome_2026-10-30.csv",
              keys=["scenario"], value="H_model", mgpt="H_mgpt2_combined", decision={"scenario": "hm27"},
              metric="H: mean over 8 endpoints of the log hazard ratio per SD of the risk residualised on age and sex"),
    "M": dict(summary="large_gain_2026-11-02/M/M_summary_2026-11-02.csv",
              rule="large_gain_2026-11-02/M/M_rule_outcome_2026-11-02.csv",
              keys=["scenario"], value="M_model", mgpt="M_mgpt2_combined", decision={"scenario": "depth_10"},
              metric="M (as J) under simulated measurement noise on empirical profiles"),
    "O": dict(summary="large_gain_2026-11-08/O/O_summary_2026-11-08.csv",
              rule="large_gain_2026-11-08/O/O_rule_outcome_2026-11-08.csv",
              keys=["scenario"], value="H_model", mgpt="H_mgpt2_combined", decision={"scenario": "depth_10"},
              metric="H (as L) under simulated measurement noise on empirical profiles"),
}


def rule(v_m, v_b, lo):
    if not np.isfinite(v_b) or v_b <= 0:
        return "not applicable (comparator value <= 0)", np.nan
    g = (v_m - v_b) / v_b
    return ("PASS" if (g >= 0.10 and lo > 0) else "FAIL"), g


def clock_of(model):
    if model.startswith(COMBINED_PREFIX):
        return "GrimAge2+DunedinPACE"
    if model.startswith("grimage2_"):
        return "GrimAge2"
    if model.startswith("dunedinpace_"):
        return "DunedinPACE"
    if model.startswith("raw_cpg_"):
        return "raw CpG"
    return "other"


def main():
    checks, outcome, clocks, combined = [], [], [], []

    def check(analysis, name, ok, detail=""):
        checks.append(dict(analysis=analysis, check=name, passed=bool(ok), detail=detail))

    for a, s in SOURCES.items():
        try:
            summ = pd.read_csv(ART / s["summary"])
            rule_src = pd.read_csv(ART / s["rule"])
        except FileNotFoundError as e:
            sys.exit(f"Cannot proceed - required data file not found: {e.filename}")
        keys, val, mg = s["keys"], s["value"], s["mgpt"]
        dec = summ[summ.role == "decision comparator"].copy()
        for kv, g in summ.groupby(keys):
            kv = kv if isinstance(kv, tuple) else (kv,)
            check(a, "1 mgpt constant", g[mg].nunique() == 1, str(kv))
        for _, r in dec.iterrows():
            if r[val] > 0:
                ok = abs(r.relative_gain - (r[mg] - r[val]) / r[val]) < 1e-9
                check(a, "3 relative gain", ok, f"{tuple(r[k] for k in keys)} {r.model}")
            exp, _ = rule(r[mg], r[val], r.ci95_low)
            got = str(r.rule_against_this_model)
            ok = (exp == got) or (exp.startswith("not applicable") and got.startswith("not applicable"))
            check(a, "4 per-model rule", ok, f"{tuple(r[k] for k in keys)} {r.model} exp={exp} got={got}")
        for _, rr in rule_src.iterrows():
            sel = np.ones(len(dec), bool)
            for k in keys:
                sel &= (dec[k] == rr[k]).values
            g = dec[sel]
            best = g.loc[g[val].idxmax()]
            check(a, "2 original best", best.model == rr.best_comparator,
                  f"{tuple(rr[k] for k in keys)} argmax={best.model} stored={rr.best_comparator}")
            ok6 = all(abs(best[c] - rr[c]) < 1e-12 for c in ["difference", "ci95_low", "ci95_high"])
            check(a, "6 rule row = summary row", ok6, f"{tuple(rr[k] for k in keys)}")

            S = g[~g.model.str.startswith(COMBINED_PREFIX)]
            check(a, "5 single set", len(S) > 0 and not S.model.str.startswith(COMBINED_PREFIX).any(),
                  f"{tuple(rr[k] for k in keys)} n={len(S)}")
            if len(S) == 0:
                continue
            b = S.loc[S[val].idxmax()]
            out, gain = rule(b[mg], b[val], b.ci95_low)
            is_dec = all(rr[k] == v for k, v in s["decision"].items())
            row = dict(analysis=a, **{k: rr[k] for k in keys}, decision_row=is_dec, metric=s["metric"],
                       methylgpt_value=b[mg], best_single_comparator=b.model, best_single_clock_family=clock_of(b.model),
                       best_single_value=b[val], relative_gain=gain, difference=b.difference, ci95_low=b.ci95_low,
                       ci95_high=b.ci95_high, p_boot=b.p_boot, outcome=out,
                       original_best_comparator=rr.best_comparator, original_outcome=rr.outcome,
                       comparator_changed=b.model != rr.best_comparator)
            if "ci95_low_within_fold" in b:
                row.update(ci95_low_within_fold=b.ci95_low_within_fold, ci95_high_within_fold=b.ci95_high_within_fold)
            outcome.append(row)

            for fam in ["GrimAge2", "DunedinPACE", "raw CpG"]:
                F = S[S.model.map(clock_of) == fam]
                if len(F) == 0:
                    continue
                c = F.loc[F[val].idxmax()]
                o, gn = rule(c[mg], c[val], c.ci95_low)
                clocks.append(dict(analysis=a, **{k: rr[k] for k in keys}, decision_row=is_dec, comparator_family=fam,
                                   comparator=c.model, methylgpt_value=c[mg], comparator_value=c[val],
                                   relative_gain=gn, difference=c.difference, ci95_low=c.ci95_low,
                                   ci95_high=c.ci95_high, p_boot=c.p_boot, rule_arithmetic=o))
            for _, c in g[g.model.str.startswith(COMBINED_PREFIX)].iterrows():
                o, gn = rule(c[mg], c[val], c.ci95_low)
                combined.append(dict(analysis=a, **{k: rr[k] for k in keys}, decision_row=is_dec, comparator=c.model,
                                     methylgpt_value=c[mg], comparator_value=c[val], relative_gain=gn,
                                     difference=c.difference, ci95_low=c.ci95_low, ci95_high=c.ci95_high,
                                     p_boot=c.p_boot, rule_arithmetic=o))

    ck = pd.DataFrame(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    ck.to_csv(OUT / "R_checks_2026-11-08.csv", index=False)
    if not ck.passed.all():
        print(ck[~ck.passed].to_string())
        sys.exit("R checks failed; no result table written")
    pd.DataFrame(outcome).to_csv(OUT / "R_rule_outcome_2026-11-08.csv", index=False)
    pd.DataFrame(clocks).to_csv(OUT / "R_clock_contrasts_2026-11-08.csv", index=False)
    pd.DataFrame(combined).to_csv(OUT / "R_combined_clock_contrasts_2026-11-08.csv", index=False)
    res = dict(analysis="R", written="2026-11-08, after J to P results were known (post hoc comparator restriction)",
               n_checks=int(len(ck)), checks_passed=bool(ck.passed.all()),
               decisions=[{k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item()) for k, v in r.items()}
                          for r in outcome if r["decision_row"]])
    (OUT / "R_results_2026-11-08.json").write_text(json.dumps(res, indent=2, default=str))
    print(pd.DataFrame(outcome).drop(columns=["metric"]).to_string())


if __name__ == "__main__":
    main()
