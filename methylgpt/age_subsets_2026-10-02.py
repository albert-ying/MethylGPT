#!/usr/bin/env python3
"""AltumAge test-set subsets: dataset overlap with the training split and a whole-blood Zhang 2019 comparison.

Inputs: eval/age_revision_2026-10-01.predictions.npz (MethylGPT and elastic-net predictions, AltumAge
test), corpus_2026-09-30/profiles.parquet (split and dataset group of each profile) and the Zhang 2019
full-matrix predictions used in evaluate_age_revision_2026-10-01.py.
Reports (1) how many AltumAge test profiles belong to datasets whose other profiles were in the
training split, with MedAE for profiles from those datasets and from datasets outside the training
split; (2) MedAE of Zhang 2019, MethylGPT and the elastic net on whole-blood test profiles with at
least 90% of Zhang's CpGs observed (Zhang 2019 was trained on blood and saliva).
Output: eval/age_subsets_2026-10-02.json
"""
import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, write_json

ZHANG = ("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_round3_20260703/revision_analyses/"
         "46_mean_imputed_public_clock_benchmark/results/zhang2019_full_matrix_altumage_predictions.csv")


def medae(p, y):
    return float(np.median(np.abs(p - y)))


def main():
    z = np.load(ARTIFACTS / "eval/age_revision_2026-10-01.predictions.npz")
    ids, y, tis = z["altumage_ids"].astype(str), z["altumage_age"], z["altumage_tissue"].astype(str)
    mg, en = z["MethylGPT|all"], z["ElasticNet (same labels)|all"]
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group", "split"]).set_index("id").reindex(ids)
    in_train_ds = (prof.split == "excluded_downstream").to_numpy()   # withheld profiles of training-split datasets
    out = {"n_test": int(len(ids)), "dataset_overlap": {
        "profiles_in_training_split_datasets": int(in_train_ds.sum()),
        "training_split_datasets": int(prof[in_train_ds].group.nunique()),
        "profiles_in_other_datasets": int((~in_train_ds).sum()),
        "other_datasets": int(prof[~in_train_ds].group.nunique()),
        "other_dataset_splits": prof[~in_train_ds].split.value_counts().to_dict(),
        "medae_in_training_split_datasets": {"MethylGPT": medae(mg[in_train_ds], y[in_train_ds]), "ElasticNet": medae(en[in_train_ds], y[in_train_ds])},
        "medae_in_other_datasets": {"MethylGPT": medae(mg[~in_train_ds], y[~in_train_ds]), "ElasticNet": medae(en[~in_train_ds], y[~in_train_ds])}}}
    zz = pd.read_csv(ZHANG, usecols=["GSM_ID", "Zhang2019_EN_full_matrix_all_samples", "zhang_observed_fraction"]).set_index("GSM_ID").reindex(ids)
    zp, zf = zz.Zhang2019_EN_full_matrix_all_samples.to_numpy(float), zz.zhang_observed_fraction.to_numpy(float)
    m = np.isfinite(zp) & (zf >= 0.9) & (tis == "Whole Blood")
    out["zhang_whole_blood_ge90pct"] = {"n": int(m.sum()), "Zhang 2019": medae(zp[m], y[m]), "MethylGPT": medae(mg[m], y[m]),
                                        "ElasticNet": medae(en[m], y[m])}
    write_json(ARTIFACTS / "eval/age_subsets_2026-10-02.json", out)
    print(out)


if __name__ == "__main__":
    main()
