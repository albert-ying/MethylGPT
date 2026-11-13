#!/usr/bin/env python3
"""Two-coefficient Cox models on MethylGPT's distilled GrimAge2 and DunedinPACE in the label-efficiency study.

Same design, subsamples (identical generator seeds and draw order) and model form as
scenario_clocks_label_efficiency_2026-10-01.py: unpenalised Breslow Cox models of standardised age,
sex and the clock score(s), fitted on stratified subsamples of the fitting folds and scored on the full
held-out folds. Scores are MethylGPT's distilled outputs (released model, gs_features_2026-10-01
*_train_mt_bio.embedding.npz) instead of BioLearn's full-array GrimAge2 and DunedinPACE, which are
refitted alongside for a matched comparison. Fraction 1.0 uses all fitting participants.
Output (aggregate only): artifacts/gs_cox/scenario_distilled_clocks_label_efficiency_2026-10-04.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, log, require, write_json

SCORES = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
MODELS = {"grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"], "distilled_grimage2_dunedinpace": ["d_GrimAge2", "d_DunedinPACE"],
          "grimage2": ["GrimAgeV2"], "distilled_grimage2": ["d_GrimAge2"], "dunedinpace": ["DunedinPACE"], "distilled_dunedinpace": ["d_DunedinPACE"]}


def pooled(ev, du, r, folds):
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], r[s]); t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


def main():
    sc = pd.read_csv(require(SCORES))
    wide = sc.pivot_table(index="sample_id", columns="model", values="score")
    wide.index = wide.index.astype(str)
    ck = torch.load(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt", map_location="cpu", weights_only=False)
    z = np.load(require(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mt_bio.embedding.npz"))
    bio = pd.DataFrame(z["embedding"] * np.asarray(ck["bio_scale"]) + np.asarray(ck["bio_median"]), index=z["dnam_id"].astype(str),
                       columns=list(ck["bio_names"]))
    wide = wide.join(bio[["GrimAge2", "DunedinPACE"]].rename(columns={"GrimAge2": "d_GrimAge2", "DunedinPACE": "d_DunedinPACE"}), how="inner")
    preds = pd.read_parquet(ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet")
    preds = preds[preds.age_adjustment == "linear"]
    rows = []
    for frac, reps in ((1.0, 1), (0.25, 2), (0.1, 2), (0.05, 2)):
        for rep in range(reps):
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
    d = pd.DataFrame(rows)
    summ = d.groupby("fraction")[list(MODELS)].mean()
    log("\n" + summ.round(4).to_string())
    write_json(ARTIFACTS / "gs_cox/scenario_distilled_clocks_label_efficiency_2026-10-04.json",
               {"results": rows, "mean_by_fraction": {str(k): v for k, v in summ.to_dict("index").items()}})


if __name__ == "__main__":
    main()
