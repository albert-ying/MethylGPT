#!/usr/bin/env python3
"""Age prediction and distilled-clock agreement in Generation Scotland (aggregate statistics only).

Participants: the training and validation subsets of the locked Generation Scotland split
(9,866 + 3,289); the test subset is not loaded. Chronological age comes from the frame-cache
labels. Predictions:
  MethylGPT (released multitask model)   gs_features_2026-10-01/*_{train,val}_mt_age.embedding.npz
  Age, sex and tissue stage               gs_features_2026-10-01/*_{train,val}_mgpt2_age.embedding.npz
  Horvath multi-tissue, Hannum, PhenoAge  BioLearn full-array predictions (v12 server results, task_b)
  Distilled GrimAge2 / DunedinPACE        gs_features_2026-10-01/*_{train,val}_mt_bio.embedding.npz (rescaled)
  Full-array GrimAge2 / DunedinPACE       BioLearn with age and sex inputs fixed (47 years, female; v12 task_a)
Reported: n, MedAE, MAE, Pearson r and mean offset (predicted minus chronological) for each age
predictor on the participants scored by all of them; Pearson r between distilled and full-array
outputs, and for GrimAge2 also after regressing both on chronological age.
Output: eval/gs_age_clocks_2026-10-02.json (no participant-level values are written).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, require, write_json

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
BIOLEARN_AGE = V12 / "revision_round3_20260703/server_results/task_b/biolearn_gs_clock_predictions.csv"
BIOLEARN_FIXED = V12 / "revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv"
F = ARTIFACTS / "gs_features_2026-10-01"


def feature(kind, names=None):
    frames = []
    for sp in ("train", "val"):
        z = np.load(require(F / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{sp}_{kind}.embedding.npz"))
        frames.append(pd.DataFrame(z["embedding"], index=z["dnam_id"].astype(str), columns=names))
    return pd.concat(frames)


def scores(pred, age):
    e = pred - age
    return {"n": int(len(age)), "medae": float(np.median(np.abs(e))), "mae": float(np.mean(np.abs(e))),
            "pearson_r": float(np.corrcoef(pred, age)[0, 1]), "mean_offset": float(e.mean())}


def main():
    labels = []
    for sp in ("train", "val"):
        lp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{sp}_all_gs_batches_all_probes_*.labels.parquet")))
        labels.append(pd.read_parquet(lp, columns=["dnam_id", "age"]).assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id"))
    age = pd.concat(labels)["age"].astype(float)
    age = age[~age.index.duplicated()]

    preds = pd.DataFrame({"MethylGPT": feature("mt_age").iloc[:, 0], "Age, sex and tissue stage": feature("mgpt2_age").iloc[:, 0]})
    bl = pd.read_csv(BIOLEARN_AGE, usecols=["sample_id", "model", "pred_age"])
    bl = bl[bl.model.isin(["Horvathv1", "Hannum", "PhenoAge"])].pivot_table(index="sample_id", columns="model", values="pred_age")
    bl.index = bl.index.astype(str)
    preds = preds.join(bl.rename(columns={"Horvathv1": "Horvath multi-tissue"}), how="left")
    common = preds.dropna().index.intersection(age.dropna().index)
    out = {"participants": "Generation Scotland training and validation subsets (test subset not loaded)",
           "n_with_age": int(age.notna().sum()), "n_common": int(len(common)), "age": {}}
    for c in preds.columns:
        out["age"][c] = scores(preds.loc[common, c].to_numpy(float), age.loc[common].to_numpy(float))

    ck = torch.load(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt", map_location="cpu", weights_only=False)
    names = list(ck["bio_names"])
    bio = feature("mt_bio", names) * np.asarray(ck["bio_scale"]) + np.asarray(ck["bio_median"])
    fx = pd.read_csv(BIOLEARN_FIXED, usecols=["sample_id", "model", "score", "metadata_mode"])
    # GrimAgeV2 was scored with age and sex inputs fixed; DunedinPACE takes no such inputs (stored as mode "real")
    fx = fx[((fx.model == "GrimAgeV2") & (fx.metadata_mode == "fixed")) | (fx.model == "DunedinPACE")]
    fx = fx.pivot_table(index="sample_id", columns="model", values="score")
    fx.index = fx.index.astype(str)
    out["distilled_vs_full_array"] = {"full_array_inputs": "BioLearn; GrimAge2 with age and sex inputs fixed at 47 years and female"}
    for d_name, f_name in (("GrimAge2", "GrimAgeV2"), ("DunedinPACE", "DunedinPACE")):
        j = (bio[[d_name]].rename(columns={d_name: "distilled"}).join(fx[[f_name]].rename(columns={f_name: "full_array"}), how="inner")
             .join(age.rename("age"), how="inner").dropna())
        r = float(np.corrcoef(j.distilled, j.full_array)[0, 1])
        A = np.c_[np.ones(len(j)), j.age.to_numpy()]
        res = [v - A @ np.linalg.lstsq(A, v, rcond=None)[0] for v in (j.distilled.to_numpy(), j.full_array.to_numpy())]
        out["distilled_vs_full_array"][d_name] = {"n": int(len(j)), "pearson_r": r,
                                                  "pearson_r_age_adjusted": float(np.corrcoef(*res)[0, 1])}
    write_json(ARTIFACTS / "eval/gs_age_clocks_2026-10-02.json", out)
    for k, v in out["age"].items():
        print(f"{k}: MedAE {v['medae']:.2f}, MAE {v['mae']:.2f}, r {v['pearson_r']:.3f}, offset {v['mean_offset']:+.2f} (n {v['n']})")
    for k, v in out["distilled_vs_full_array"].items():
        if isinstance(v, dict):
            print(f"{k}: r {v['pearson_r']:.3f}, age-adjusted {v['pearson_r_age_adjusted']:.3f} (n {v['n']})")


if __name__ == "__main__":
    main()
