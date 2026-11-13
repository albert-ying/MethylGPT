#!/usr/bin/env python3
"""Finite-resample bootstrap P values for all Generation Scotland contrasts.

The contrast scripts computed P = min(1, 2 * min(#{d* <= 0}, #{d* >= 0}) / B) from B bootstrap differences d*, which
is 0 when no resample crosses zero. A bootstrap with B draws cannot resolve P below 2 / (B + 1), so P is recomputed
with the add-one convention

    P = min(1, 2 * (min(#{d* <= 0}, #{d* >= 0}) + 1) / (B + 1)),

and the Benjamini-Hochberg adjustment is repeated within the same families as before. The smaller tail count is
recovered exactly from the stored P (count = P * B / 2); no bootstrap is re-run and no participant-level data are read.
For the eight primary contrasts of the test-subset plan, which were not adjusted, a Holm adjustment across the eight
superiority tests is added and labelled post hoc.

Inputs (aggregate) -> outputs with the suffix _finite_p_2026-10-09:
  gs_cox/paired_bootstrap_mgpt2_combined_2026-10-01.csv      B = 1,000, one family
  gs_cox/paired_bootstrap_factor_residual_2026-10-01.csv     B = 1,000, one family
  gs_cox/paired_bootstrap_stacked_2026-10-04.csv             B = 1,000, one family
  gs_cox/clinical_covariates_bootstrap_2026-10-03.csv        B = 1,000, one family (stored P floored at 0.001)
  gs_cox/bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09.csv   B = 2,000, one family of 384
  gs_cox/scenario_raw_factor_imputed_2026-10-09_2000draws.csv          B = 2,000, one family of 128
  gs_test_2026-10-05/confirmation.json, validation_subset.json   B = 2,000; per-endpoint contrasts within scenario
Summary: gs_cox/bootstrap_p_finite_2026-10-09.json (families, smallest attainable P, classification changes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, log, require, write_json

G = ARTIFACTS / "gs_cox"
T = ARTIFACTS / "gs_test_2026-10-05"


def finite_p(p, B):
    p = np.asarray(p, float)
    k = p * B / 2
    kr = np.rint(k)
    capped = p >= 1.0
    # the clinical table stored max(P, 1 / 1000): a stored 0.001 with B = 1,000 is a zero count
    assert np.all(capped | (np.abs(k - kr) < 1e-6) | np.isclose(p, 1 / B)), "stored P is not a multiple of 2 / B"
    kr = np.where(np.isclose(p, 1 / B) & (np.abs(k - kr) > 1e-6), 0, kr)
    return np.where(capped, 1.0, np.minimum(1.0, 2 * (kr + 1) / (B + 1)))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n); out[o] = np.minimum(adj, 1.0)
    return out


def main():
    summary = {"formula": "P = min(1, 2 * (min(#{d* <= 0}, #{d* >= 0}) + 1) / (B + 1))", "tables": {}}
    # The two scenario tables were first computed with 300 draws, where the add-one convention changed 14 of 384 and
    # 6 of 128 classifications; both were therefore re-run with 2,000 draws (same scripts, --draws 2000) and the
    # 2,000-draw tables are the reported ones. `prev` names the 300-draw table used only to count changes.
    for name, B, prev in (("paired_bootstrap_mgpt2_combined_2026-10-01", 1000, None), ("paired_bootstrap_factor_residual_2026-10-01", 1000, None),
                          ("paired_bootstrap_stacked_2026-10-04", 1000, None), ("clinical_covariates_bootstrap_2026-10-03", 1000, None),
                          ("bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09", 2000, "bootstrap_scenarios_mgpt2_combined_2026-10-01"),
                          ("scenario_raw_factor_imputed_2026-10-09_2000draws", 2000, "scenario_raw_factor_imputed_2026-10-09")):
        d = pd.read_csv(require(G / f"{name}.csv"))
        if prev is None:
            old_sig = d.q_bh.to_numpy() < 0.05
        else:
            keys = ["scenario", "endpoint", "comparator"]
            o = pd.read_csv(require(G / f"{prev}.csv"))[keys + ["q_bh"]].rename(columns={"q_bh": "q_prev"})
            old_sig = d[keys].merge(o, on=keys, how="left", validate="one_to_one").q_prev.to_numpy() < 0.05
        assert np.allclose(bh(d.p.to_numpy()), d.q_bh.to_numpy(), atol=1e-9), f"{name}: stored q is not BH over the whole table"
        d["p"] = finite_p(d.p, B); d["q_bh"] = bh(d.p.to_numpy())
        new_sig = d.q_bh.to_numpy() < 0.05
        d.to_csv(G / f"{name}_finite_p_2026-10-09.csv", index=False)
        summary["tables"][name] = {"draws": B, "family_size": int(len(d)), "smallest_attainable_p": 2 / (B + 1), "compared_with": prev or "the same table with P = 2 * count / B",
                                   "contrasts_at_smallest_p": int(np.isclose(d.p, 2 / (B + 1)).sum()),
                                   "significant_before": int(old_sig.sum()), "significant_after": int(new_sig.sum()),
                                   "classification_changes": int((old_sig != new_sig).sum()),
                                   "largest_significant_q": float(d.q_bh[new_sig].max()) if new_sig.any() else None,
                                   "smallest_nonsignificant_q": float(d.q_bh[~new_sig].min()) if (~new_sig).any() else None}
        log(f"{name}: {summary['tables'][name]}")
    for name in ("confirmation", "validation_subset"):
        j = json.loads(require(T / f"{name}.json").read_text()); B = int(j["draws"])
        sec = pd.DataFrame(j["secondary"])
        old_q = sec.q_bh.to_numpy(float)
        sec["p"] = finite_p(sec.p, B)
        sec["q_bh"] = np.nan
        for scen, g in sec.groupby("scenario"):
            idx = g.index[g.endpoint != "mean"]
            sec.loc[idx, "q_bh"] = bh(sec.loc[idx, "p"].to_numpy())
        ep = (sec.endpoint != "mean").to_numpy()
        old_sig, new_sig = old_q[ep] < 0.05, sec.q_bh.to_numpy()[ep] < 0.05
        pri = pd.DataFrame(j["primary"])
        pri["p"] = finite_p(pri.p, B)
        pri["p_holm_post_hoc"] = holm(pri.p.to_numpy())
        out = {"draws": B, "note": "P values with the add-one convention; q_bh within scenario over per-endpoint contrasts; p_holm_post_hoc is a post hoc "
                                   "Holm adjustment over the eight primary superiority tests (not part of the plan)",
               "primary": json.loads(pri.to_json(orient="records")), "secondary": json.loads(sec.to_json(orient="records"))}
        write_json(T / f"{name}_finite_p_2026-10-09.json", out)
        summary["tables"][name] = {"draws": B, "smallest_attainable_p": 2 / (B + 1), "secondary_family_sizes": sec[ep].groupby("scenario").size().to_dict(),
                                   "secondary_significant_before": int(old_sig.sum()), "secondary_significant_after": int(new_sig.sum()),
                                   "secondary_classification_changes": int((old_sig != new_sig).sum()),
                                   "primary": [{"id": r.id, "scenario": r.scenario, "endpoint": r.endpoint, "comparator": r.comparator, "difference": r.difference,
                                                "ci95": r.ci95, "p": r.p, "p_holm_post_hoc": r.p_holm_post_hoc} for r in pri.itertuples()]}
        log(f"{name}: secondary changes {summary['tables'][name]['secondary_classification_changes']}; primary " +
            "; ".join(f"{r.id} {r.scenario} {r.endpoint[:5]} vs {r.comparator[:8]} p {r.p:.4f} holm {r.p_holm_post_hoc:.4f}" for r in pri.itertuples()))
    write_json(G / "bootstrap_p_finite_2026-10-09.json", summary)


if __name__ == "__main__":
    main()
