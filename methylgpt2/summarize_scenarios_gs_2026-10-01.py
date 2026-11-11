#!/usr/bin/env python3
"""Mean C-index over the eight GS endpoints for every model in every missing-CpG scenario.

Joins scenario_gs_combined_2026-10-01.py (MethylGPT-2 models, raw CpG) with
scenario_clocks_cox_2026-10-01.py (clocks) and, when present, the scenario bootstrap summary.
Writes artifacts/gs_cox/scenario_summary_combined_2026-10-01.csv and prints a Markdown table.
"""
from __future__ import annotations

import json

import pandas as pd

from methylgpt2_lib import ARTIFACTS, require

ORDER = ["full", "hm450", "epic_v2", "random_30", "random_50", "random_70", "hm27", "random_90"]
COLS = {"raw_cpg_v12": "Raw CpG, locked", "raw_cpg_gpu": "Raw CpG, GPU", "grimage2": "GrimAge2", "dunedinpace": "DunedinPACE",
        "grimage2_dunedinpace": "GrimAge2 + DunedinPACE", "ensemble_embed_bio": "MGPT-2 embedding + biomarkers",
        "factor_residual": "MGPT-2 factor+residual", "mgpt2_combined": "MGPT-2 combined"}
BASELINES = ["raw_cpg_v12", "raw_cpg_gpu", "grimage2", "dunedinpace", "grimage2_dunedinpace"]


def main():
    comb = json.load(open(require(ARTIFACTS / "gs_cox/scenario_combined_2026-10-01/scenario_cindex.json")))
    clocks = json.load(open(require(ARTIFACTS / "gs_cox/scenario_clocks_cox_2026-10-01.json")))
    rows = []
    for scen, block in comb["scenarios"].items():
        for ep, v in block.items():
            rows.append({"scenario": scen, "endpoint": ep, "observed_fraction": comb["observed_fraction"][scen], **v, **clocks[scen][ep]})
    d = pd.DataFrame(rows)
    d["best_baseline"] = d[BASELINES].max(axis=1)
    d.to_csv(ARTIFACTS / "gs_cox/scenario_summary_combined_2026-10-01.csv", index=False)
    boot_path = ARTIFACTS / "gs_cox/bootstrap_scenarios_mgpt2_combined_2026-10-01.json"
    boot = json.load(open(boot_path))["summary"] if boot_path.exists() else {}
    scens = [s for s in ORDER if s in set(d.scenario)]
    print("| Scenario (observed vocabulary CpGs) | " + " | ".join(COLS.values()) + " | Combined ahead of every baseline |")
    print("|---" * (len(COLS) + 2) + "|")
    for scen in scens:
        g = d[d.scenario == scen]
        ahead = int((g.mgpt2_combined > g.best_baseline).sum())
        print(f"| {scen} ({g.observed_fraction.iloc[0]:.0%}) | " + " | ".join(f"{g[c].mean():.3f}" for c in COLS) + f" | {ahead} of 8 |")
    if boot:
        print("\nBootstrap: endpoints where the combined model is better / worse (95% CI excludes 0)")
        print("| Scenario | " + " | ".join(COLS[c] for c in BASELINES) + " |")
        print("|---" * (len(BASELINES) + 1) + "|")
        for scen in scens:
            print(f"| {scen} | " + " | ".join(f"{boot[scen][c]['better']} / {boot[scen][c]['worse']}" for c in BASELINES) + " |")


if __name__ == "__main__":
    main()
