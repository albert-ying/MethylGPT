#!/usr/bin/env python3
"""Tissue-level age error on all held-out datasets (primary evaluation), beside the unflagged sensitivity analysis.

Figure 4d and the tissue rows of Supplementary Table 4 used only the held-out datasets whose metadata ages were not
flagged as discordant with the reference clock. This script computes the same quantities on all 34 held-out datasets
from the stored predictions (no model is run): MedAE of MethylGPT (three-run mean) and of the elastic net per tissue
with at least 50 profiles, and the paired difference with a 95% interval from resampling profiles within the tissue
(1,000 draws, seed 20261004), exactly as age_ensemble_2026-10-04.py does for the unflagged subset.
Inputs: artifacts/eval/age_ensemble_2026-10-04.predictions.npz, artifacts/eval/age_revision_2026-10-01.predictions.npz
Output: artifacts/eval/age_tissue_all_datasets_2026-10-10.json
"""
import numpy as np

from methylgpt2_lib import ARTIFACTS, log, require, write_json

EN = "ElasticNet (same labels)"


def medae(p, y):
    return float(np.median(np.abs(p - y)))


def tissue_boot(pa, pb, y, n=1000, seed=20261004):
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    d = [np.median(ea[k]) - np.median(eb[k]) for k in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {"difference": float(np.median(ea) - np.median(eb)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    try:
        ze = np.load(require(ARTIFACTS / "eval/age_ensemble_2026-10-04.predictions.npz"))
        z1 = np.load(require(ARTIFACTS / "eval/age_revision_2026-10-01.predictions.npz"))
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    y, flag, tis = ze["heldout_age"], ze["heldout_flag"].astype(bool), ze["heldout_tissue"].astype(str)
    assert np.array_equal(y, z1["heldout_age"]) and np.array_equal(tis, z1["heldout_tissue"].astype(str))
    mg, en = ze["ensemble|heldout"], z1[f"{EN}|heldout"]
    res = {"heldout_profiles": int(len(y)), "flagged_profiles": int(flag.sum()), "tissues": {}}
    for t in sorted(set(tis) - {"unlabelled"}):
        row = {}
        for name, m in (("all_datasets", tis == t), ("unflagged_datasets", (tis == t) & ~flag)):
            if m.sum() >= 50:
                row[name] = {"n": int(m.sum()), "MethylGPT": medae(mg[m], y[m]), "elastic net": medae(en[m], y[m]),
                             "MethylGPT_minus_elastic_net": tissue_boot(mg[m], en[m], y[m])}
        if row:
            res["tissues"][t] = row
            log(f"{t}: " + "; ".join(f"{k} n {v['n']} MethylGPT {v['MethylGPT']:.2f} EN {v['elastic net']:.2f} diff {v['MethylGPT_minus_elastic_net']['difference']:+.2f} "
                                     f"[{v['MethylGPT_minus_elastic_net']['ci95'][0]:+.2f}, {v['MethylGPT_minus_elastic_net']['ci95'][1]:+.2f}]" for k, v in row.items()))
    write_json(ARTIFACTS / "eval/age_tissue_all_datasets_2026-10-10.json", res)


if __name__ == "__main__":
    main()
