#!/usr/bin/env python3
"""Age error by tissue with datasets resampled (Supplementary Fig. 20d).

Written on 11 Nov 2026 before running. Every result is reported. No model is run; no Generation Scotland file is read.

Supplementary Fig. 20d shows MethylGPT minus elastic-net MedAE by tissue on the 34 held-out datasets. Its intervals
resample profiles within the tissue (age_tissue_all_datasets_2026-10-10.py: 1,000 draws, seed 20261004). Here the same
difference gets a 95% interval that resamples datasets within the tissue.
INPUTS  The inputs of age_tissue_all_datasets_2026-10-10.py:
  - artifacts/eval/age_ensemble_2026-10-04.predictions.npz: MethylGPT, mean of three runs; held-out ages, tissues and
    flags;
  - artifacts/eval/age_revision_2026-10-01.predictions.npz: the elastic net.
  Dataset = corpus series of each held-out profile.
TISSUES  Every tissue label with at least 50 profiles, as that script ('unlabelled' excluded). All 34 datasets
  (primary; Supplementary Fig. 20d). The unflagged datasets (metadata ages not discordant with the reference clock;
  Supplementary Table 4) as a secondary.
INTERVAL  Per tissue, 2,000 draws (seed 20261110, one generator across tissues in sorted order, all datasets then
  unflagged). Each draw resamples the tissue's datasets with replacement and recomputes both MedAEs on the resampled
  profiles; percentile 95% interval of the difference. A tissue from one dataset has a single-valued interval
  (reported). The profile-resampling interval of the 2026-10-10 JSON is given beside it.
CHECK (abort on failure)  Point estimates and n equal age_tissue_all_datasets_2026-10-10.json within 1e-12.
OUTPUT  artifacts/eval/age_tissue_dataset_resampled_2026-11-10.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, log, require  # noqa: E402

EN = "ElasticNet (same labels)"
SEED = 20261110


def main():
    ze = np.load(require(ARTIFACTS / "eval/age_ensemble_2026-10-04.predictions.npz"))
    z1 = np.load(require(ARTIFACTS / "eval/age_revision_2026-10-01.predictions.npz"))
    old = json.loads(require(ARTIFACTS / "eval/age_tissue_all_datasets_2026-10-10.json").read_text())["tissues"]
    y, flag, tis = ze["heldout_age"], ze["heldout_flag"].astype(bool), ze["heldout_tissue"].astype(str)
    assert np.array_equal(y, z1["heldout_age"]) and np.array_equal(tis, z1["heldout_tissue"].astype(str))
    ids = ze["heldout_ids"].astype(str)
    grp = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"), columns=["id", "group"]).set_index("id").reindex(ids).group.astype(str).to_numpy()
    mg, en = ze["ensemble|heldout"], z1[f"{EN}|heldout"]
    ea, eb = np.abs(mg - y), np.abs(en - y)
    rng = np.random.default_rng(SEED)
    rows, bad = [], []
    for subset, base in (("all 34 held-out datasets (Supplementary Fig. 20d)", np.ones(len(y), bool)), ("unflagged datasets (Supplementary Table 4)", ~flag)):
        key = "all_datasets" if subset.startswith("all") else "unflagged_datasets"
        for t in sorted(set(tis) - {"unlabelled"}):
            m = (tis == t) & base
            if m.sum() < 50 or t not in old or key not in old[t]:
                continue
            idx = np.flatnonzero(m)
            ds = grp[idx]
            units = np.unique(ds)
            members = [idx[ds == u] for u in units]
            draws = []
            for _ in range(2000):
                k = np.concatenate([members[j] for j in rng.integers(0, len(units), len(units))])
                draws.append(np.median(ea[k]) - np.median(eb[k]))
            diff = float(np.median(ea[m]) - np.median(eb[m]))
            o = old[t][key]
            if abs(diff - o["MethylGPT_minus_elastic_net"]["difference"]) > 1e-12 or int(m.sum()) != o["n"]:
                bad.append(f"{t} {key}")
            rows.append({"profiles": subset, "tissue": t, "n_profiles": int(m.sum()), "n_datasets": int(len(units)),
                         "datasets": ";".join(units), "medae_methylgpt": float(np.median(ea[m])), "medae_elastic_net": float(np.median(eb[m])),
                         "methylgpt_minus_elastic_net": diff,
                         "ci95_low_resampling_datasets": float(np.percentile(draws, 2.5)), "ci95_high_resampling_datasets": float(np.percentile(draws, 97.5)),
                         "ci95_low_resampling_profiles_2026_10_10": o["MethylGPT_minus_elastic_net"]["ci95"][0],
                         "ci95_high_resampling_profiles_2026_10_10": o["MethylGPT_minus_elastic_net"]["ci95"][1],
                         "note": "one dataset: the interval is a single value" if len(units) == 1 else ""})
    log(f"check point estimates and n equal the 2026-10-10 JSON: {'pass' if not bad else 'FAIL ' + str(bad)}")
    if bad:
        sys.exit("check failed")
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "eval/age_tissue_dataset_resampled_2026-11-10.csv", index=False)
    log(out[["profiles", "tissue", "n_profiles", "n_datasets", "methylgpt_minus_elastic_net", "ci95_low_resampling_datasets",
             "ci95_high_resampling_datasets"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
