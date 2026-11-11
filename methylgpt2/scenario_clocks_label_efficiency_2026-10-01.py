#!/usr/bin/env python3
"""Published clocks in the label-efficiency study (GS development, locked outer folds).

Uses the same stratified subsamples of the fitting subjects as
scenario_gs_label_efficiency_2026-10-01.py (identical generator seeds and draw order) and fits
unpenalised Breslow Cox models of standardised age, sex and the clock score(s) on them; risks
are scored on the full held-out fold and pooled within folds. Clock scores as in
cox_clock_comparators_2026-10-01.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, log, require, write_json

SCORES = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv")
MODELS = {"grimage2": ["GrimAgeV2"], "dunedinpace": ["DunedinPACE"], "grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"]}


def pooled(ev, du, r, folds):
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], r[s]); t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.25,0.1,0.05")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/scenario_clocks_label_efficiency_2026-10-01.json"))
    a = ap.parse_args()
    sc = pd.read_csv(require(SCORES))
    wide = sc.pivot_table(index="sample_id", columns="model", values="score")
    wide.index = wide.index.astype(str)
    preds = pd.read_parquet(Path(a.mgpt2_run) / "private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    rows = []
    for frac in [float(x) for x in a.fractions.split(",")]:
        for rep in range(1 if frac == 1.0 else a.repeats):
            for ep in sorted(preds.endpoint.unique()):
                pe = preds[preds.endpoint == ep].reset_index(drop=True)
                ids = pe.dnam_id.astype(str).to_numpy()
                ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
                cov = pe[["age", "sex"]].to_numpy(float)
                risks = {k: np.empty(len(pe)) for k in MODELS}
                folds = []
                for f in sorted(pe.outer_fold.unique()):
                    held = np.flatnonzero(pe.outer_fold.to_numpy() == f)
                    fit = np.flatnonzero(pe.outer_fold.to_numpy() != f)
                    folds.append(held)
                    if frac < 1.0:
                        rng = np.random.default_rng(1000 * rep + 17 * f + int(frac * 1000))
                        keep = []
                        for e_val in (True, False):
                            grp = fit[ev[fit] == e_val]
                            keep.append(rng.choice(grp, max(2, int(round(frac * len(grp)))), replace=False))
                        fit = np.sort(np.concatenate(keep))
                    for name, cols in MODELS.items():
                        M = np.column_stack([cov] + [wide.loc[ids, c].to_numpy(float) for c in cols])
                        m, s = M[fit].mean(0), M[fit].std(0)
                        cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M[fit] - m) / s, Surv.from_arrays(ev[fit], du[fit]))
                        risks[name][held] = ((M[held] - m) / s) @ cx.coef_
                row = {"fraction": frac, "repeat": rep, "endpoint": ep}
                row |= {k: float(pooled(ev, du, r, folds)) for k, r in risks.items()}
                rows.append(row)
                log(f"f={frac} rep={rep} {ep}: " + " ".join(f"{k} {row[k]:.4f}" for k in MODELS))
    write_json(Path(a.out), {"settings": vars(a), "results": rows})


if __name__ == "__main__":
    main()
