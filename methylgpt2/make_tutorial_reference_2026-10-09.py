#!/usr/bin/env python3
"""Analysis-code ages for the six public tutorial profiles (GSE54848), for the package agreement test.

The analysis code (load_model of evaluate_age_revision_2026-10-01.py, as used for every age result of the
manuscript) scores package/examples/GSE54848_example_betas.csv with each of the three training runs on GPU; the
reference age is their mean. Output: package/examples/GSE54848_example_analysis_code_ages.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, cuda_preinit, require

HERE = Path(__file__).resolve().parent
RUNS = ["runs/multitask_bio_2026-10-01/final.pt", "runs/multitask_bio_seed20261002/final.pt", "runs/multitask_bio_seed20261003/final.pt"]


def main():
    cuda_preinit()
    spec = importlib.util.spec_from_file_location("ear", HERE / "evaluate_age_revision_2026-10-01.py")
    ear = importlib.util.module_from_spec(spec); spec.loader.exec_module(ear)
    ex = pd.read_csv(require(HERE / "package/examples/GSE54848_example_betas.csv"), index_col=0)
    vocab = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    X = np.clip(ex.reindex(columns=vocab).to_numpy(np.float32), 0, 1)
    runs = np.stack([ear.load_model(require(ARTIFACTS / p), torch.device("cuda:0"))(X) for p in RUNS], 1)
    out = pd.DataFrame({"sample": ex.index, "analysis_code_age": runs.mean(1), **{f"analysis_code_age_run{i + 1}": runs[:, i] for i in range(3)}})
    out.to_csv(HERE / "package/examples/GSE54848_example_analysis_code_ages.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
