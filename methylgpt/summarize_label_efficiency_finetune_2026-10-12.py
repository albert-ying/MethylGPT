#!/usr/bin/env python3
"""Collect the fine-tuning label-efficiency grid (run_label_efficiency_2026-10-12.py) into one aggregate table.

One row per run: arm (pretrained, factor_only, none), number of labelled profiles, label-subset seed, selected step and
MedAE / MAE on the AltumAge test set (complete profiles, 90% of CpGs removed, 27K-array CpGs) and on the 5,811
held-out profiles. Runs with all labels are the controls of 1-3 October (artifacts/eval/pretraining_control_2026-10-03.json).
Outputs: artifacts/eval/pretraining_label_efficiency_finetune_2026-10-12.{csv,json}
"""
import json
import re

import pandas as pd

from methylgpt_lib import ARTIFACTS, log, write_json

D = ARTIFACTS / "label_efficiency_2026-10-12"


def main():
    rows = []
    for f in sorted(D.glob("*/results.json")):
        m = re.fullmatch(r"(finetune|frozen|last1)_(pretrained|factor_only|none)_n(\d+)_s(\d+)", f.parent.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        row = {"mode": m.group(1), "arm": m.group(2), "n_labels": int(m.group(3)), "seed": int(m.group(4)), "n_fit": r["n_fit"],
               "n_selection": r["n_selection"], "selected_step": r["best_step"]}
        for key, name in (("altumage_test", "altumage_test|all"), ("altumage_test_random_90", "altumage_test|random_90"),
                          ("altumage_test_hm27", "altumage_test|hm27"), ("heldout_datasets_all", "heldout|all")):
            row[f"{name}|medae"], row[f"{name}|mae"] = r[key]["medae"], r[key]["mae"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"Cannot proceed - required data file not found: {D}/*/results.json")
    df = pd.DataFrame(rows).sort_values(["mode", "n_labels", "arm", "seed"])
    df.to_csv(ARTIFACTS / "eval/pretraining_label_efficiency_finetune_2026-10-12.csv", index=False)
    cols = [c for c in df.columns if c.endswith("|medae")]
    summ = df.groupby(["mode", "n_labels", "arm"])[cols].agg(["mean", "min", "max"]).round(4)
    summ.columns = [f"{a}|{b}" for a, b in summ.columns]
    seeds = df.groupby(["mode", "n_labels", "arm"]).seed.count().rename("runs")
    out = summ.join(seeds).reset_index()
    write_json(ARTIFACTS / "eval/pretraining_label_efficiency_finetune_2026-10-12.json",
               {"runs": int(len(df)), "arms": {"pretrained": "fitted factor model + transformer after masked-value pretraining",
                                                "factor_only": "fitted factor model + transformer from random weights",
                                                "none": "random factor loadings + transformer from random weights"},
                "summary": out.to_dict(orient="records")})
    for n, g in df[df["mode"] == "finetune"].groupby("n_labels"):
        log(f"n = {n}: " + "; ".join(f"{a} {x['altumage_test|all|medae'].mean():.2f} / 27K {x['altumage_test|hm27|medae'].mean():.2f} ({len(x)} runs)" for a, x in g.groupby("arm")))


if __name__ == "__main__":
    main()
