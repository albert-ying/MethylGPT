#!/usr/bin/env python3
"""Genome-wide CpG matrix of the whole-blood RRBS cohort GSE194154, for models trained on the cohort itself.

The vocabulary-level matrix (artifacts/external/GSE194154/betas_cov10.parquet) holds only MethylGPT's 49,156 CpGs.
A model trained from scratch on the sequencing cohort can use every CpG the assay covers, so this script builds
the matrix of all CpGs with at least 10 reads (both strands summed) in at least 90% of the 110 analysis samples.
Call files and conventions are those documented in artifacts/external/GSE194154/scripts/03_build_matrix.py; beta is
the read-weighted mean of the two strands' freqC / 100 (freqC is truncated to a percent in the call files).
Output: artifacts/external/GSE194154/betas_genomewide_cov10_2026-10-12.npz (beta float32 with NaN, keys, samples)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS, log, require

D = ARTIFACTS / "external/GSE194154"
CHROMS = {f"chr{c}": i for i, c in enumerate(list(range(1, 23)) + ["X", "Y", "M"])}


def one(path):
    t = pd.read_csv(path, sep="\t", usecols=["chrBase", "chr", "base", "coverage", "freqC"],
                    dtype={"chrBase": "category", "chr": "category", "base": np.int64, "coverage": np.int32, "freqC": np.float32})
    t = t[t.chr.isin(list(CHROMS))]
    pos = t.base.to_numpy() - (t.chrBase.to_numpy() == "G")
    key = t.chr.map(CHROMS).to_numpy().astype(np.int64) * (1 << 32) + pos
    cov = t.coverage.to_numpy().astype(np.float64)
    meth = cov * t.freqC.to_numpy() / 100.0
    order = np.argsort(key, kind="stable")
    key, cov, meth = key[order], cov[order], meth[order]
    uk, start = np.unique(key, return_index=True)
    c, m = np.add.reduceat(cov, start), np.add.reduceat(meth, start)
    keep = c >= 10
    return uk[keep], (m[keep] / c[keep]).astype(np.float32)


def main():
    try:
        samples = pd.read_parquet(require(D / "betas_cov10.parquet"), columns=[]).index.to_list()
    except OSError as e:
        raise SystemExit(f"Cannot proceed - required data file not found: {e}")
    files = {p.name.split("_")[0]: p for p in sorted((D / "raw").glob("GSM*.txt.gz"))}
    missing = [g for g in samples if g not in files]
    if missing:
        raise SystemExit(f"Cannot proceed - required data file not found: {D / 'raw'} lacks {missing[:3]}")
    with Pool(22) as pool:
        res = pool.map(one, [files[g] for g in samples])
    log(f"parsed {len(res)} samples; CpGs with >= 10 reads per sample median {int(np.median([len(k) for k, _ in res]))}")
    allk, cnt = np.unique(np.concatenate([k for k, _ in res]), return_counts=True)
    keys = allk[cnt >= int(np.ceil(0.9 * len(samples)))]
    B = np.full((len(samples), len(keys)), np.nan, np.float32)
    for i, (k, b) in enumerate(res):
        j = np.searchsorted(keys, k)
        ok = (j < len(keys)) & (keys[np.minimum(j, len(keys) - 1)] == k)
        B[i, j[ok]] = b[ok]
    log(f"CpGs covered in >= 90% of samples: {len(keys)}; observed fraction {np.isfinite(B).mean():.3f}")
    np.savez(D / "betas_genomewide_cov10_2026-10-12.npz", beta=B, keys=keys, samples=np.array(samples))


if __name__ == "__main__":
    sys.exit(main())
