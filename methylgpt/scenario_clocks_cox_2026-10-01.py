#!/usr/bin/env python3
"""Published clocks under the missing-CpG scenarios (GS development, locked outer folds).

For each endpoint and outer fold, an unpenalised Cox model (age, sex, clock score[s]) is fitted on
the fitting subjects' full-coverage scores and applied to the held-out subjects' scores computed
under each scenario (score_gs_clocks_missingness_2026-10-01.py). Pooled within-fold C-index. Held-out risks are also
written (private_clock_scenario_risks.parquet, controlled-data derived) for paired bootstraps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, log, require, write_json


def pooled(ev, du, r, folds):
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], r[s]); t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


def main():
    scores = pd.read_csv(require(ARTIFACTS / "gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv"))
    scores["sample_id"] = scores.sample_id.astype(str)
    preds = pd.read_parquet(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    full = scores[scores.scenario == "full"].set_index("sample_id")
    models = {"grimage2": ["GrimAgeV2"], "dunedinpace": ["DunedinPACE"], "grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"]}
    res = {}
    frames = []
    for scen in scores.scenario.unique():
        sc = scores[scores.scenario == scen].set_index("sample_id")
        block = {}
        for ep, pe in preds.groupby("endpoint"):
            pe = pe.reset_index(drop=True)
            ids = pe.dnam_id.astype(str).to_numpy()
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            cov = pe[["age", "sex"]].to_numpy(float)
            folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
            row = {}
            for name, cols in models.items():
                Mf = np.column_stack([cov] + [full.loc[ids, c].to_numpy(float) for c in cols])
                Ms = np.column_stack([cov] + [sc.loc[ids, c].to_numpy(float) for c in cols])
                r = np.empty(len(pe))
                for held in folds:
                    fit = np.setdiff1d(np.arange(len(pe)), held)
                    m, s = Mf[fit].mean(0), Mf[fit].std(0)
                    cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((Mf[fit] - m) / s, Surv.from_arrays(ev[fit], du[fit]))
                    r[held] = ((Ms[held] - m) / s) @ cx.coef_
                row[name] = pooled(ev, du, r, folds)
                frames.append(pd.DataFrame({"dnam_id": ids, "endpoint": ep, "outer_fold": pe.outer_fold.to_numpy(),
                                            "scenario": scen, "model": name, "risk": r}))
            block[ep] = row
        res[scen] = block
        log(f"{scen}: " + ", ".join(f"{ep[:10]} G {v['grimage2']:.3f} D {v['dunedinpace']:.3f} GD {v['grimage2_dunedinpace']:.3f}" for ep, v in block.items()))
    write_json(ARTIFACTS / "gs_cox/scenario_clocks_cox_2026-10-01.json", res)
    pd.concat(frames).to_parquet(ARTIFACTS / "gs_cox/private_clock_scenario_risks_2026-10-01.parquet", index=False)


if __name__ == "__main__":
    main()
