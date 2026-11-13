#!/usr/bin/env python3
"""Incident-disease value of MethylGPT-2 predicted age in Generation Scotland (development only).

Same subjects, labels and outer folds as the locked nested protocol
(compare_joint_cox_2026-09-28.py: 9,866 original-training subjects, StratifiedKFold(5,
shuffle, seed 20260928 + 1000) on each endpoint's event indicator; the test split is never
loaded). Within each outer fold an unpenalised Breslow Cox model is fitted on the fitting
subjects with standardised covariates:
  age_sex            age, sex
  age_sex_mgpt2age   age, sex, MethylGPT-2 predicted age (equivalent to its age acceleration)
Risk scores of held-out subjects give a combined C-index over within-fold comparable pairs.
Paired bootstrap (subjects resampled within outer folds, 500 draws) gives the difference.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt_lib import ARTIFACTS, log, require, write_json

LABELS = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/results/regenerated_survival_split_label_tables")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
ENDPOINTS = ["Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases", "date_of_death",
             "Endocrine_and_Metabolic_Diseases", "Kidney_Diseases", "Neurological_Diseases", "Respiratory_Diseases"]
SEED = 20260928


def counts(event, time_, risk):
    _, c, d, tr, _ = concordance_index_censored(event, time_, risk)
    return np.array([c, d, tr], float)


def cidx(cnt):
    return (cnt[0] + 0.5 * cnt[2]) / (cnt[0] + cnt[1] + cnt[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_mgpt2_age.embedding.npz"))
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--out", default=str(ARTIFACTS / "gs_cox/mgpt2_predicted_age_2026-10-01.json"))
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(require(Path(a.features)), allow_pickle=False)
    ids, pred_age = z["dnam_id"], z["embedding"][:, 0].astype(float)
    res = {"features": a.features, "endpoints": {}}
    for ep in ENDPOINTS:
        fr = pd.read_parquet(require(LABELS / f"survial_{ep}_{SPLIT}_split_labels.parquet"), filters=[("split", "==", "train")])
        fr["dnam_id"] = fr.dnam_id.astype(str)
        if set(fr.dnam_id) != set(ids):
            raise ValueError(f"Label-feature identity mismatch for {ep}")
        fr = fr.set_index("dnam_id").loc[ids]
        sex = fr.sex.map({"F": 1.0, "M": 0.0, "Female": 1.0, "Male": 0.0, "female": 1.0, "male": 0.0})
        sex = sex.where(sex.notna(), pd.to_numeric(fr.sex, errors="coerce")).to_numpy(float)
        age, ev, du = fr.age.to_numpy(float), fr.event.astype(bool).to_numpy(), fr.duration.to_numpy(float)
        X = {"age_sex": np.column_stack([age, sex]), "age_sex_mgpt2age": np.column_stack([age, sex, pred_age])}
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1000)
        held_sets, risks = [], {k: np.empty(len(ids)) for k in X}
        coefs = {k: [] for k in X}
        for fit_idx, held_idx in outer.split(ids, fr.event):
            held_sets.append(held_idx)
            for k, M in X.items():
                m, s = M[fit_idx].mean(0), M[fit_idx].std(0)
                model = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M[fit_idx] - m) / s, Surv.from_arrays(ev[fit_idx], du[fit_idx]))
                risks[k][held_idx] = ((M[held_idx] - m) / s) @ model.coef_
                coefs[k].append(model.coef_.tolist())
        tot = {k: sum(counts(ev[h], du[h], risks[k][h]) for h in held_sets) for k in X}
        point = {k: cidx(v) for k, v in tot.items()}
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(a.draws):
            cnt = {k: np.zeros(3) for k in X}
            for h in held_sets:
                b = rng.choice(h, len(h), replace=True)
                for k in X:
                    cnt[k] += counts(ev[b], du[b], risks[k][b])
            diffs.append(cidx(cnt["age_sex_mgpt2age"]) - cidx(cnt["age_sex"]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        res["endpoints"][ep] = {"n": int(len(ids)), "events": int(ev.sum()), "c_index": point,
                                "difference": point["age_sex_mgpt2age"] - point["age_sex"], "ci95": [float(lo), float(hi)],
                                "p_two_sided_sign": float(2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))),
                                "mgpt2age_hr_per_sd_by_fold": [c[2] for c in coefs["age_sex_mgpt2age"]]}
        log(f"{ep}: age+sex {point['age_sex']:.4f} +MethylGPT-2 age {point['age_sex_mgpt2age']:.4f} "
            f"diff {res['endpoints'][ep]['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    res["elapsed_s"] = time.time() - t0
    write_json(Path(a.out), res)


if __name__ == "__main__":
    main()
