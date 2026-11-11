#!/usr/bin/env python3
"""Out-of-fold Cox risks for published clocks in the locked nested protocol (GS development).

Same 9,866 subjects and outer folds as the MethylGPT-2 run (taken from its prediction file).
Within each outer fold an unpenalised Breslow Cox model is fitted on the fitting subjects with
standardised age, sex and the clock score(s); held-out risks are saved in the protocol's
prediction format so paired_bootstrap_gs can compare them with MethylGPT-2.
Clock scores: task_a_gs_clock_scores_fixed_grimage_20260629.csv from the v12 server results
(GrimAgeV2 computed by BioLearn with fixed age 47 and female = 1, so only its DNAm component
varies; DunedinPACE). Extra columns can be added from feature files (e.g. MethylGPT-2 heads).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, log, require, write_json

SCORES = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mgpt2-run", default=str(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01"))
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "gs_cox/clock_comparators_2026-10-01"))
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(require(SCORES))
    wide = sc.pivot_table(index="sample_id", columns="model", values="score")
    wide.index = wide.index.astype(str)
    preds = pd.read_parquet(Path(a.mgpt2_run) / "private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"].copy()
    preds["dnam_id"] = preds.dnam_id.astype(str)
    missing = set(preds.dnam_id) - set(wide.index)
    if missing:
        raise ValueError(f"{len(missing)} development subjects lack clock scores")
    models = {"grimage2": ["GrimAgeV2"], "dunedinpace": ["DunedinPACE"], "grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"]}
    summary = {}
    for name, cols in models.items():
        rows = []
        for ep in sorted(preds.endpoint.unique()):
            pe = preds[preds.endpoint == ep].reset_index(drop=True)
            M = np.column_stack([pe.age.to_numpy(float), pe.sex.to_numpy(float)] + [wide.loc[pe.dnam_id, c].to_numpy(float) for c in cols])
            ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
            r = np.empty(len(pe))
            for f in sorted(pe.outer_fold.unique()):
                held = pe.outer_fold.to_numpy() == f
                fit = ~held
                m, s = M[fit].mean(0), M[fit].std(0)
                cox = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M[fit] - m) / s, Surv.from_arrays(ev[fit], du[fit]))
                r[held] = ((M[held] - m) / s) @ cox.coef_
            q = pe[["dnam_id", "duration", "event", "age", "sex", "endpoint", "age_adjustment", "outer_fold"]].copy()
            q["risk"] = r
            q["baseline_risk"] = pe.baseline_risk.to_numpy()
            rows.append(q)
        d = pd.concat(rows)
        (out / name).mkdir(exist_ok=True)
        d.to_parquet(out / name / "private_nested_predictions.parquet", index=False)
        summary[name] = str(out / name)
        log(f"{name}: wrote out-of-fold risks for {d.endpoint.nunique()} endpoints")
    write_json(out / "manifest.json", {"scores": str(SCORES), "models": models, "outputs": summary})


if __name__ == "__main__":
    main()
