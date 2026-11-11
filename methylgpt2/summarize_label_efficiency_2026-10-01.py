#!/usr/bin/env python3
"""Summarise the label-efficiency runs (mean over repeats per fraction and endpoint).

Reads every scenario_label_efficiency*_2026-10-01.json under artifacts/gs_cox (the main run and
any fraction run in parallel); a (fraction, repeat, endpoint) row seen twice must agree. Clock
results on the same subsamples (scenario_clocks_label_efficiency_2026-10-01.json) are joined.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from methylgpt2_lib import ARTIFACTS

METHODS = ["age_sex", "raw_cpg", "methylgpt2", "biomarkers", "ensemble"]
CLOCKS = ["grimage2", "dunedinpace", "grimage2_dunedinpace"]


def main():
    rows = []
    for f in sorted((ARTIFACTS / "gs_cox").glob("scenario_label_efficiency*_2026-10-01.json")):
        rows += json.load(open(f))["results"]
    d = pd.DataFrame(rows)
    key = ["fraction", "repeat", "endpoint"]
    dup = d[d.duplicated(key, keep=False)]
    if len(dup):
        spread = dup.groupby(key)[METHODS].agg(lambda s: s.max() - s.min()).to_numpy().max()
        if spread > 1e-6:
            raise ValueError("duplicate label-efficiency rows disagree")
    d = d.drop_duplicates(key)
    ck = pd.DataFrame(json.load(open(ARTIFACTS / "gs_cox/scenario_clocks_label_efficiency_2026-10-01.json"))["results"])
    d = d.merge(ck, on=key, how="left", validate="one_to_one")
    m = d.groupby(["fraction", "endpoint"])[METHODS + CLOCKS + ["mean_fit_subjects"]].mean().reset_index()
    m["n_repeats"] = d.groupby(["fraction", "endpoint"]).size().to_numpy()
    m["best_mgpt2"] = m[["biomarkers", "ensemble"]].max(axis=1)
    m.to_csv(ARTIFACTS / "gs_cox/scenario_label_efficiency_summary_2026-10-01.csv", index=False)
    pd.set_option("display.width", 200)
    for frac, g in m.groupby("fraction", sort=False):
        print(f"\nfraction {frac} (fitting subjects ~{g.mean_fit_subjects.mean():.0f}; repeats {g.n_repeats.min()}-{g.n_repeats.max()})")
        print(g[["endpoint"] + METHODS + CLOCKS].round(4).to_string(index=False))
        best_clock = g[CLOCKS].max(axis=1)
        for name in ("biomarkers", "ensemble", "methylgpt2"):
            print(f"  {name} > raw on {(g[name] > g.raw_cpg).sum()}/{len(g)}; mean diff {np.mean(g[name] - g.raw_cpg):+.4f};"
                  f" > age+sex on {(g[name] > g.age_sex).sum()}/{len(g)}; > best clock on {(g[name] > best_clock).sum()}/{len(g)},"
                  f" mean diff vs GrimAge2+DunedinPACE {np.mean(g[name] - g.grimage2_dunedinpace):+.4f}")


if __name__ == "__main__":
    main()
