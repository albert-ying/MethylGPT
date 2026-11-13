#!/usr/bin/env python3
"""Equal-weight ensemble of two or more Cox models' out-of-fold risks (no refitting).

Usage: make_risk_ensemble_2026-10-01.py RUN_DIR RUN_DIR [RUN_DIR ...] OUT_DIR
Within each outer fold each model's held-out risk is standardised and the standardised risks
are summed. The weights are fixed (equal) before evaluation; folds, subjects and baseline risks
are those of the first run.
"""
import sys
from pathlib import Path
import pandas as pd

runs, out = [Path(x) for x in sys.argv[1:-1]], Path(sys.argv[-1])
if len(runs) < 2:
    raise SystemExit("need at least two run directories and an output directory")
frames = []
for r in runs:
    d = pd.read_parquet(r / "private_nested_predictions.parquet")
    d = d[d.age_adjustment == "linear"].copy()
    d["dnam_id"] = d.dnam_id.astype(str)
    frames.append(d)
key = ["dnam_id", "endpoint", "outer_fold"]
z = lambda s: (s - s.mean()) / s.std()
m = frames[0].copy()
m["risk"] = m.groupby(["endpoint", "outer_fold"]).risk.transform(z)
for i, d in enumerate(frames[1:], 1):
    m = m.merge(d[key + ["risk"]].rename(columns={"risk": f"risk_{i}"}), on=key, validate="one_to_one")
    m["risk"] += m.groupby(["endpoint", "outer_fold"])[f"risk_{i}"].transform(z)
    m = m.drop(columns=[f"risk_{i}"])
if len(m) != len(frames[0]):
    raise ValueError("runs do not share subjects, endpoints and folds")
out.mkdir(parents=True, exist_ok=True)
m.to_parquet(out / "private_nested_predictions.parquet", index=False)
print("wrote", out, len(m))
