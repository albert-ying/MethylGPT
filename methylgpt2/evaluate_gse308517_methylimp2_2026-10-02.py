#!/usr/bin/env python3
"""methyLImp2 on the fixed GSE308517 imputation protocol (best-in-class comparator).

Same 136 x 35,676 beta matrix, five donor folds, masks (15/30/50/70%, seeds 20260930-20260932)
and per-sample MAE as evaluate_gse308517_2026-09-30.py. methyLImp2 (Bioconductor 1.4.0,
Plaksienko et al. 2024) regresses each missing CpG on the complete CpGs of the same chromosome
across samples, so each evaluation donor is imputed together with the complete profiles of
the fitting donors of its fold (its standard use with a reference panel). CpG-to-chromosome
assignment comes from the EPIC v1 / HM450 manifests (annotation type "user"). Runs in R with
16 worker processes over donors.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, log, require, write_json

DEV_RESULTS = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RLIB = "/Entropy/kying/tmp/rlib"

R_CODE = r"""
.libPaths(c("RLIB", .libPaths()))
suppressMessages({library(methyLImp2); library(BiocParallel); library(parallel)})
d <- "DIR"
n <- 136; p <- 35676
beta <- matrix(readBin(file.path(d, "beta.bin"), "double", n = n * p), nrow = n, ncol = p, byrow = TRUE)
ids <- readLines(file.path(d, "probes.txt"))
colnames(beta) <- ids; rownames(beta) <- paste0("s", seq_len(n))
ann <- read.csv(file.path(d, "annotation.csv"), stringsAsFactors = FALSE)
folds <- as.integer(readLines(file.path(d, "folds.txt")))
for (cond in readLines(file.path(d, "conditions.txt"))) {
  mask <- matrix(readBin(file.path(d, paste0("mask_", cond, ".bin")), "raw", n = n * p), nrow = n, ncol = p, byrow = TRUE) != as.raw(0)
  res <- mclapply(seq_len(n), function(i) {
    fit <- which(folds != folds[i])
    x <- beta[i, ]; x[mask[i, ]] <- NA
    inp <- rbind(beta[fit, , drop = FALSE], x)
    rownames(inp) <- c(rownames(beta)[fit], "target")
    out <- methyLImp2(inp, type = "user", annotation = ann, BPPARAM = SerialParam(), overwrite_res = TRUE)
    out["target", mask[i, ]]
  }, mc.cores = 16)
  writeBin(as.double(unlist(res)), file.path(d, paste0("pred_", cond, ".bin")))
  cat(cond, "done\n")
}
"""


def main():
    t0 = time.time()
    plan = np.load(require(DEV_RESULTS / "prospective_imputation_2026-09-30/mask_fold_plan.npz"))
    ov = np.load(require(DEV_RESULTS / "geo308517_access_2026-09-29/GSE308517_model_probe_overlap_2026-09-30.npz"))
    beta, folds, ids = ov["beta_values"], plan["folds"], ov["overlap_probe_ids"].astype(str)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet")).set_index("cpg").reindex(ids)
    conds = [f"seed{s}_rate{r}" for r in (15, 30, 50, 70) for s in (20260930, 20260931, 20260932)]
    res = {"method": "methyLImp2 1.4.0 (Bioconductor), annotation type user, one evaluation donor at a time with "
                     "the complete fitting donors of its fold", "conditions": {}}
    with tempfile.TemporaryDirectory(dir="/Entropy/kying/tmp") as d:
        d = Path(d)
        beta.astype(np.float64).tofile(d / "beta.bin")
        (d / "probes.txt").write_text("\n".join(ids) + "\n")
        pd.DataFrame({"cpg": ids, "chr": "chr" + ann.chr.astype(str).to_numpy()}).to_csv(d / "annotation.csv", index=False)
        (d / "folds.txt").write_text("\n".join(map(str, folds)) + "\n")
        (d / "conditions.txt").write_text("\n".join(conds) + "\n")
        for c in conds:
            plan[f"mask_{c}"].astype(np.uint8).tofile(d / f"mask_{c}.bin")
        code = R_CODE.replace("RLIB", RLIB).replace("DIR", str(d))
        log("running methyLImp2 in R")
        r = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
        if r.returncode != 0:
            log(r.stderr[-3000:])
            raise SystemExit("methyLImp2 run failed")
        log(r.stdout.strip())
        for c in conds:
            mask = plan[f"mask_{c}"]
            pred = np.fromfile(d / f"pred_{c}.bin", dtype=np.float64)
            per, k = [], 0
            for i in range(len(mask)):
                m = mask[i]
                y = beta[i, m]
                p_ = pred[k:k + m.sum()]; k += m.sum()
                ok = np.isfinite(p_) & np.isfinite(y)
                per.append(float(np.mean(np.abs(p_[ok] - y[ok]))))
            assert k == len(pred)
            res["conditions"][c] = {"methylimp2": float(np.mean(per))}
            res["conditions"][c + "_per_sample"] = {"methylimp2": per}
    res["seed_averaged_mae_by_rate"] = {str(r): {"methylimp2": float(np.mean([res["conditions"][f"seed{s}_rate{r}"]["methylimp2"]
                                                                            for s in (20260930, 20260931, 20260932)]))}
                                        for r in (15, 30, 50, 70)}
    res["elapsed_s"] = time.time() - t0
    write_json(ARTIFACTS / "eval/gse308517_methylimp2_2026-10-01.json", res)
    log(f"methyLImp2 MAE by rate: {res['seed_averaged_mae_by_rate']}")


if __name__ == "__main__":
    main()
