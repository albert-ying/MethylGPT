#!/usr/bin/env python3
"""Check of the Zhang 2019 elastic-net age predictor used as a comparator.

1. Coefficients: the file distributed with BioLearn (Zhang2019.csv, 514 CpGs) against the authors' released
   en.coef (https://github.com/qzhang314/DNAm-based-age-predictor, downloaded 9 Oct 2026).
2. Predictions: the authors' released script pred.R (unmodified; elastic-net output) is run on the full arrays of
   public AltumAge test datasets (ClockBase processed matrices) and compared with the predictions used in the
   manuscript (v12 full-matrix rerun: beta values standardized within each sample across all probes of the array,
   ddof 1, coefficients plus intercept 65.79295, absent values contributing zero) for the same GEO samples.
   pred.R differs in one step: a value missing for a sample is first replaced by the mean of that probe across the
   samples of the input file.
Public GEO data only. Output: artifacts/eval/zhang2019_reference_check_2026-10-09.json
Usage: python zhang2019_reference_check_2026-10-09.py <dir with pred.R, en.coef, blup.coef> [GSE ...]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, log, require, write_json

PROCESSED = Path("/Entropy/kying/GEO_data/GEO_hs_dnam/processed_data")
STORED = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_round3_20260703/revision_analyses/"
              "46_mean_imputed_public_clock_benchmark/results/zhang2019_full_matrix_altumage_predictions.csv")
BIOLEARN_COEF = Path("/home/kying/anaconda3/envs/integrative/lib/python3.10/site-packages/biolearn/data/Zhang2019.csv")
DEFAULT_GSE = ["GSE49064", "GSE52068", "GSE115797", "GSE61257", "GSE92767"]
R_PREP = r'''
args <- commandArgs(TRUE)
suppressMessages(library(data.table))
d <- fread(args[1])
m <- as.matrix(d[, -1, with = FALSE]); rownames(m) <- d[[1]]
keep <- scan(args[4], what = "", quiet = TRUE)
m <- m[, intersect(colnames(m), keep), drop = FALSE]
saveRDS(t(m), args[2])                                   # individuals x probes, raw beta
write.table(data.frame(ID = colnames(m), age = 0), args[3], row.names = FALSE, quote = FALSE)
'''


def main():
    try:
        ref = Path(sys.argv[1]).resolve()
        for f in ("pred.R", "en.coef", "blup.coef"):
            require(ref / f)
        stored = pd.read_csv(require(STORED))
    except (IndexError, OSError) as e:
        sys.exit(f"Cannot proceed - required data file not found: {e}")
    gses = sys.argv[2:] or DEFAULT_GSE
    a = pd.read_csv(ref / "en.coef", sep=r"\s+")
    b = pd.read_csv(require(BIOLEARN_COEF))
    aa = a.iloc[1:].set_index("probe").coef.astype(float); bb = b.set_index("CpGmarker").CoefficientTraining.astype(float)
    res = {"coefficients": {"authors_en_coef_cpgs": int(len(aa)), "biolearn_cpgs": int(len(bb)), "same_cpg_set": bool(set(aa.index) == set(bb.index)),
                            "max_abs_difference": float((aa - bb.reindex(aa.index)).abs().max()), "authors_intercept": float(a.iloc[0, 1])},
           "stored_predictions": str(STORED), "datasets": {}}
    log(f"coefficients: {res['coefficients']}")
    col, frac = "Zhang2019_EN_full_matrix_all_samples", "zhang_observed_fraction"
    allr, alls = [], []
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as td:
        td = Path(td); (td / "prep.R").write_text(R_PREP)
        for g in gses:
            st = stored[stored.GSM_ID.isin(pd.read_csv(require(PROCESSED / f"{g}.csv.gz"), nrows=0).columns)]
            st = st[np.isfinite(st[col])]
            if st.empty:
                log(f"{g}: no stored predictions"); continue
            (td / "keep.txt").write_text("\n".join(st.GSM_ID))
            try:
                subprocess.run(["Rscript", str(td / "prep.R"), str(PROCESSED / f"{g}.csv.gz"), str(td / "d.rds"), str(td / "age.txt"), str(td / "keep.txt")],
                               check=True, capture_output=True, text=True)
                out = subprocess.run(["Rscript", "pred.R", "-i", str(td / "d.rds"), "-o", str(td / "out.txt"), "-a", str(td / "age.txt")],
                                     cwd=ref, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                log(f"{g}: reference script failed: {e.stderr[-400:]}"); continue
            r = pd.read_csv(td / "out.txt", sep=r"\s+").set_index("ID").enpred.reindex(st.GSM_ID).to_numpy(float)
            s, fr = st[col].to_numpy(float), st[frac].to_numpy(float)
            full = fr >= 1.0
            missing_line = [l for l in out.stdout.splitlines() if "Elastic Net predictor" in l]
            res["datasets"][g] = {"samples": int(len(st)), "samples_with_all_514_cpgs": int(full.sum()),
                                  "max_abs_difference_years": float(np.abs(r - s).max()),
                                  "max_abs_difference_years_samples_with_all_cpgs": float(np.abs(r - s)[full].max()) if full.any() else None,
                                  "reference_script_message": missing_line[0].strip() if missing_line else None}
            allr.append(r); alls.append(s)
            log(f"{g}: {res['datasets'][g]}")
    if allr:
        r, s = np.concatenate(allr), np.concatenate(alls)
        res["all"] = {"samples": int(len(r)), "max_abs_difference_years": float(np.abs(r - s).max()),
                      "median_abs_difference_years": float(np.median(np.abs(r - s))), "pearson_r": float(np.corrcoef(r, s)[0, 1])}
        log(f"all: {res['all']}")
    write_json(ARTIFACTS / "eval/zhang2019_reference_check_2026-10-09.json", res)


if __name__ == "__main__":
    main()
