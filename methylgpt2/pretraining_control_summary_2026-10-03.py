#!/usr/bin/env python3
"""Pretrained versus not pretrained, at the same training stage (age, sex and tissue heads, seed 20261001).

Inputs: eval/age_stage_sweep_2026-10-03.predictions.npz (pretrained: corpus-fitted factor model and
masked-value-trained transformer) and eval/age_nopretrain_2026-10-03.predictions.npz (no pretraining:
random orthonormal factor loadings and a transformer from random weights, encoder learning rate 5e-5), and
eval/age_randominit_sweep_2026-10-03.predictions.npz (fitted factor model, transformer from random weights).
Paired difference in MedAE (no pretraining minus pretrained) on the AltumAge test set for each
missing-CpG scenario, and in MedAE and MAE on the 34 held-out datasets, with 95% intervals from
resampling datasets (2,000 draws, seed 20261003).
Output: eval/pretraining_control_2026-10-03.json
"""
import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, write_json


def paired(a, b, y, g, stat, n=2000, seed=20261003):
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(a - y), np.abs(b - y)
    idx = pd.Series(np.arange(len(y))).groupby(g).apply(np.asarray).tolist()
    d = [stat(ea[k]) - stat(eb[k]) for k in (np.concatenate([idx[i] for i in rng.integers(0, len(idx), len(idx))]) for _ in range(n))]
    return {"no_pretraining": float(stat(ea)), "pretrained": float(stat(eb)), "difference": float(stat(ea) - stat(eb)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    p = np.load(ARTIFACTS / "eval/age_stage_sweep_2026-10-03.predictions.npz")
    q = np.load(ARTIFACTS / "eval/age_nopretrain_2026-10-03.predictions.npz")
    r = np.load(ARTIFACTS / "eval/age_randominit_sweep_2026-10-03.predictions.npz")
    assert (p["altumage_ids"] == r["altumage_ids"]).all()
    assert (p["altumage_ids"] == q["altumage_ids"]).all()
    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "group", "split"])
    gt = prof.set_index("id").reindex(p["altumage_ids"].astype(str)).group.astype(str).to_numpy()
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    gh = prof.group.astype(str).to_numpy()[rows]
    y = p["altumage_age"]
    out = {"altumage_test_medae": {}, "heldout_34_datasets": {}, "transformer_random_weights_vs_pretrained": {"altumage_test_medae": {}}}
    for s in ("all", "random_30", "random_50", "random_70", "random_90", "hm27", "hm450", "epic_v2"):
        out["altumage_test_medae"][s] = paired(q[f"No pretraining lr 5e-5|{s}"], p[f"Age-sex-tissue stage|{s}"], y, gt, np.median)
        out["transformer_random_weights_vs_pretrained"]["altumage_test_medae"][s] = paired(
            r[f"Random initial weights lr 5e-5|{s}"], p[f"Age-sex-tissue stage|{s}"], y, gt, np.median)
    yh = p["heldout_age"]
    for name, stat in (("medae", np.median), ("mae", np.mean)):
        out["heldout_34_datasets"][name] = paired(q["No pretraining lr 5e-5|heldout"], p["Age-sex-tissue stage|heldout"], yh, gh, stat)
    out["transformer_random_weights_vs_pretrained"]["heldout_34_datasets"] = {
        name: paired(r["Random initial weights lr 5e-5|heldout"], p["Age-sex-tissue stage|heldout"], yh, gh, stat)
        for name, stat in (("medae", np.median), ("mae", np.mean))}
    write_json(ARTIFACTS / "eval/pretraining_control_2026-10-03.json", out)
    for k, v in out["transformer_random_weights_vs_pretrained"]["altumage_test_medae"].items():
        print("random-weights transformer", k, round(v["no_pretraining"], 2), round(v["pretrained"], 2), [round(x, 2) for x in v["ci95"]])
    for k, v in out["altumage_test_medae"].items():
        print(k, round(v["no_pretraining"], 2), round(v["pretrained"], 2), [round(x, 2) for x in v["ci95"]])
    for k, v in out["heldout_34_datasets"].items():
        print("heldout", k, round(v["no_pretraining"], 2), round(v["pretrained"], 2), [round(x, 2) for x in v["ci95"]])


if __name__ == "__main__":
    main()
