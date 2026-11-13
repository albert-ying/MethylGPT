#!/usr/bin/env python3
"""Mean beta value per vocabulary CpG over 8,000 whole-blood training profiles (seed 0).

Used by evaluate_rrbs_gse194154_2026-10-02.py to check the RRBS-to-vocabulary mapping and to fill
missing CpGs of published clocks. This is the code that produced
artifacts/external/corpus_whole_blood_mean_beta.csv on 1 October 2026, saved as a script.
Run: python3 make_whole_blood_reference_2026-10-01.py [output.csv]
"""
import sys

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import Corpus, ARTIFACTS


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else str(ARTIFACTS / "external/corpus_whole_blood_mean_beta.csv")
    c = Corpus()
    p = c.profiles
    rows = np.flatnonzero(((p.split == "train") & (p.tissue == "Whole Blood")).to_numpy())
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(rows, 8000, replace=False))
    X = torch.from_numpy(c.get(rows))
    fin = torch.isfinite(X)
    mean = (torch.where(fin, X, torch.zeros(())).sum(0) / fin.sum(0).clamp_min(1)).numpy()
    vocab = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str)
    pd.DataFrame({"cpg": vocab, "blood_array_mean_beta": mean, "n_obs": fin.sum(0).numpy()}).to_csv(out, index=False)
    print(len(rows), "blood training profiles; saved", out)


if __name__ == "__main__":
    main()
