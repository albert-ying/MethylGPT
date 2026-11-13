#!/usr/bin/env python3
"""GrimAgeV2 and DunedinPACE for the Generation Scotland test subset under the confirmation scenarios.

Pre-specified in the test-confirmation plan of 5 Oct 2026. Copy of score_gs_clocks_missingness_2026-10-01.py
with three changes: participants are the test subset (identifiers written by gs_test_confirmation's
--write-test-ids step; no outcomes are read here); missing clock CpGs are filled with the means of the
9,866 training-subset participants (the development clock-CpG matrix), never with test means; and the
scenarios are full, hm27 and random_90 (default_rng(20261001 + 90) over the test clock-CpG matrix).
Run with the BioLearn environment: /home/kying/anaconda3/envs/integrative/bin/python <this> <out.csv> [ids.csv]
(the optional ids file scores another held-out subset, e.g. the validation subset)
Output (participant-level, controlled-data derived, stays in artifacts/): <out.csv> and <out>.json.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import biolearn
from biolearn.data_library import GeoData
from biolearn.model_gallery import ModelGallery

PROJECT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
TASK_B = PROJECT / "revision_round3_20260703/server_results/task_b/gs_age_external_validation_2026_06_22.py"
HERE = Path(__file__).resolve().parent
DEV_MATRIX = HERE / "artifacts/gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.clock_cpg_matrix.npz"
OUT = Path(sys.argv[1])


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def require(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    return p


def load_task_b():
    spec = importlib.util.spec_from_file_location("task_b_gs_age", require(TASK_B))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["task_b_gs_age"] = mod
    spec.loader.exec_module(mod)
    return mod


def clock_cpgs():
    d = Path(biolearn.__file__).resolve().parent / "data"
    out = {}
    for name, fn, col in (("GrimAgeV2", "GrimAgeV2.csv", "var"), ("DunedinPACE", "DunedinPACE.csv", "CpGmarker")):
        df = pd.read_csv(d / fn)
        col = col if col in df.columns else df.columns[0]
        v = df[col].dropna().astype(str)
        out[name] = v[v.str.startswith("cg")].drop_duplicates().tolist()
    return out


def hm27_set():
    with zipfile.ZipFile(require(PROJECT / "infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip")) as z:
        raw = z.read("infinium-methylationepic-v-1-0-b5-manifest-file.csv").decode()
    epic = pd.read_csv(io.StringIO(raw[raw.index("IlmnID,Name,"):].split("[Controls]")[0]), low_memory=False)
    epic["Name"] = epic["Name"].astype(str)
    return set(epic.Name[epic["Methyl27_Loci"].astype(str).str.upper().eq("TRUE")])


def main():
    t0 = time.time()
    task_b = load_task_b()
    cl = clock_cpgs()
    target = sorted(set(cl["GrimAgeV2"]) | set(cl["DunedinPACE"]))
    dev = np.load(require(DEV_MATRIX), allow_pickle=False)
    if list(dev["cpgs"].astype(str)) != target:
        raise SystemExit("clock CpG list differs from the development matrix")
    means = np.nanmean(dev["beta"], 0)  # training-subset means
    ids = pd.read_csv(require(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT.parent / "gs_test_ids.csv")).dnam_id.astype(str).tolist()
    if set(ids) & set(dev["ids"].astype(str)):
        raise SystemExit("test identifiers overlap the training subset")
    positions, cpgs, target_idx, n_rows = task_b.row_map_for_targets(target)
    M = np.full((len(ids), len(target)), np.nan, dtype=np.float32)
    row = {s: i for i, s in enumerate(ids)}
    for member, sel, batch in task_b.iter_archive_batches(set(ids), positions, n_rows, len(target), target_idx, 25000):
        for j, s in enumerate(sel):
            M[row[s]] = batch[j]
        log(f"loaded {member}: {len(sel)} subjects")
    np.savez(OUT.with_suffix(".clock_cpg_matrix.npz"), ids=np.asarray(ids, dtype=str), cpgs=np.asarray(target, dtype=str), beta=M)
    on27 = hm27_set()
    meta = pd.DataFrame({"sample_id": ids, "age": 47.0, "Age": 47.0, "sex": 1.0, "Female": 1.0}).set_index("sample_id")
    gallery = ModelGallery()
    rows, observed = [], {}
    for name in ("full", "hm27", "random_90"):
        X = M.copy()
        if name == "random_90":
            r = np.random.default_rng(20261001 + 90)
            X[r.random(X.shape) < 0.9] = np.nan
        elif name == "hm27":
            X[:, ~np.array([c in on27 for c in target])] = np.nan
        observed[name] = float(np.isfinite(X).mean())
        X = np.where(np.isfinite(X), X, means[None, :])
        data = GeoData(meta, pd.DataFrame(X.T, index=target, columns=ids))
        out = pd.DataFrame({"sample_id": ids, "scenario": name})
        for model_name, col in (("GrimAgeV2", "DNAmGrimAge"), ("DunedinPACE", "DunedinPACE")):
            pred = gallery.get(model_name, imputation_method="averaging").predict(data)
            out[model_name] = pred[col].reindex(ids).to_numpy() if col in pred.columns else pred.iloc[:, 0].reindex(ids).to_numpy()
        rows.append(out)
        log(f"{name}: scored; observed clock-CpG fraction {observed[name]:.3f}")
    pd.concat(rows).to_csv(OUT, index=False)
    json.dump({"participants": len(ids), "clock_cpgs": {k: len(v) for k, v in cl.items()}, "target_cpgs": len(target),
               "captured_in_archive": len(set(cpgs)), "observed_clock_cpg_fraction": observed,
               "fill": "training-subset means", "elapsed_s": time.time() - t0}, open(OUT.with_suffix(".json"), "w"), indent=1)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
