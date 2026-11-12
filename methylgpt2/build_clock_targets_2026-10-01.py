#!/usr/bin/env python3
"""Align ClockBase full-array clock outputs to the MethylGPT-2 corpus profiles.

ClockBase (/Entropy/kying/GEO_data/GEO_hs_dnam/clock_result/<GSE>.csv) scores each GEO sample
from its full array (450K/EPIC), so many inputs lie outside the 49,156-CpG vocabulary. These
values become distillation targets: the model learns to predict them from vocabulary CpGs.
Columns 14-25 of each file are GrimAge v1 and 26-39 GrimAge v2 (v2 adds logA1C and logCRP);
pandas suffixes the repeated v2 names with ".1". A target is kept only where ClockBase's
missing_percentage file records at most 10% of that clock's CpGs missing for the sample.
"""
from __future__ import annotations

import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from methylgpt2_lib import CORPUS_DIR, Corpus, log, write_json

SRC = "/Entropy/kying/GEO_data/GEO_hs_dnam/clock_result"
MISS = "/Entropy/kying/GEO_data/GEO_hs_dnam/missing_percentage"
MAX_MISSING = 0.10
KEEP = {
    "DunedinPACE": "DunedinPACE", "PhenoAge": "PhenoAge", "Horvathv1": "Horvathv1", "Hannum": "Hannum",
    "Horvathv2": "Horvathv2", "DNAmTL": "DNAmTL", "HRSInCHPhenoAge": "HRSInCHPhenoAge",
    "YingCausAge": "YingCausAge", "YingDamAge": "YingDamAge", "YingAdaptAge": "YingAdaptAge",
    "DNAmGrimAge": "GrimAge1", "DNAmGrimAge.1": "GrimAge2",
    "DNAmADM.1": "ADM", "DNAmB2M.1": "B2M", "DNAmCystatinC.1": "CystatinC", "DNAmGDF15.1": "GDF15",
    "DNAmLeptin.1": "Leptin", "DNAmPACKYRS.1": "PACKYRS", "DNAmPAI1.1": "PAI1", "DNAmTIMP1.1": "TIMP1",
    "DNAmlogA1C": "logA1C", "DNAmlogCRP": "logCRP",
    "AlcoholMcCartney": "Alcohol_McC", "BMI_McCartney": "BMI_McC", "SmokingMcCartney": "Smoking_McC",
}


MISS_COL = {"DunedinPACE": "DunedinPACE", "PhenoAge": "PhenoAge", "Horvathv1": "Horvathv1", "Hannum": "Hannum",
            "Horvathv2": "Horvathv2", "DNAmTL": "DNAmTL", "HRSInCHPhenoAge": "HRSInCHPhenoAge",
            "YingCausAge": "YingCausAge", "YingDamAge": "YingDamAge", "YingAdaptAge": "YingAdaptAge",
            "GrimAge1": "GrimAgeV1", "Alcohol_McC": "AlcoholMcCartney", "BMI_McC": "BMI_McCartney",
            "Smoking_McC": "SmokingMcCartney"}
for _t in ("GrimAge2", "ADM", "B2M", "CystatinC", "GDF15", "Leptin", "PACKYRS", "PAI1", "TIMP1", "logA1C", "logCRP"):
    MISS_COL[_t] = "GrimAgeV2"


def read_one(path):
    try:
        d = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    cols = [c for c in KEEP if c in d.columns]
    if not cols:
        return None
    d = d[cols].rename(columns=KEEP)
    d.index = d.index.astype(str)
    d = d.apply(pd.to_numeric, errors="coerce")
    mpath = os.path.join(MISS, os.path.basename(path))
    if not os.path.exists(mpath):
        return None  # without coverage information the targets are not trusted
    m = pd.read_csv(mpath, index_col=0)
    m.index = m.index.astype(str)
    m = m.reindex(d.index)
    for t in d.columns:
        mc = MISS_COL.get(t)
        if mc is None or mc not in m.columns:
            d[t] = np.nan
        else:
            d.loc[~(pd.to_numeric(m[mc], errors="coerce") <= MAX_MISSING), t] = np.nan
    return d


def main():
    t0 = time.time()
    files = sorted(glob.glob(f"{SRC}/*.csv"))
    with Pool(16) as pool:
        parts = [p for p in pool.map(read_one, files) if p is not None]
    clocks = pd.concat(parts)
    clocks = clocks[~clocks.index.duplicated(keep="first")]
    corpus = Corpus()
    tab = clocks.reindex(corpus.profiles.id.astype(str).to_numpy())
    tab.index = corpus.profiles.index
    tab.insert(0, "id", corpus.profiles.id.to_numpy())
    tab.insert(1, "split", corpus.profiles.split.to_numpy())
    out = CORPUS_DIR / "clock_targets_2026-10-01.parquet"
    tab.to_parquet(out)
    cov = {c: tab[c].notna().groupby(tab.split).sum().astype(int).to_dict() for c in KEEP.values() if c in tab}
    summ = {"source": SRC, "files_read": len(parts), "samples_in_clockbase": int(len(clocks)),
            "coverage_by_split": cov, "elapsed_s": time.time() - t0}
    write_json(CORPUS_DIR / "clock_targets_2026-10-01.json", summ)
    log(f"ClockBase samples {len(clocks)}; train coverage: " + ", ".join(f"{k} {v.get('train', 0)}" for k, v in cov.items()))


if __name__ == "__main__":
    main()
