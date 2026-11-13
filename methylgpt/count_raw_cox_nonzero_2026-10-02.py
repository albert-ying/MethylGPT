#!/usr/bin/env python3
"""Number of nonzero CpG coefficients in the two raw-CpG Cox comparators (40 fits each: 8 endpoints x 5 outer folds).

Original protocol (v12): model files of the nested raw-CpG runs used as the comparator in
paired_bootstrap_gs_2026-10-01.py (coefficient vector = 2 covariates, age and sex, then 49,156 CpGs).
Dense fit: the 'raw_cpg_gpu' lines of logs/gs_factor_residual_cox_2026-10-01.log.
Output: eval/raw_cox_nonzero_2026-10-02.json
"""
import glob
import re
import statistics

import numpy as np

from methylgpt_lib import ARTIFACTS, write_json

V12 = "/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results"
PARTS = ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")


def main():
    files = [f for p in PARTS for f in sorted(glob.glob(f"{V12}/nested_raw_cpg_{p}_2026-09-28/*_linear_outer*_model.npz"))]
    v12 = [int((np.load(f)["coefficient"][2:] != 0).sum()) for f in files]
    log = (ARTIFACTS / "logs/gs_factor_residual_cox_2026-10-01.log").read_text()
    dense = [int(m.group(1)) for m in re.finditer(r"raw_cpg_gpu: alpha index \d+, nonzero (\d+)", log)]
    out = {"original_protocol": {"fits": len(v12), "median_nonzero_cpgs": statistics.median(v12), "min": min(v12), "max": max(v12)},
           "dense_fit": {"fits": len(dense), "median_nonzero_cpgs": statistics.median(dense), "min": min(dense), "max": max(dense)}}
    write_json(ARTIFACTS / "eval/raw_cox_nonzero_2026-10-02.json", out)
    print(out)


if __name__ == "__main__":
    main()
