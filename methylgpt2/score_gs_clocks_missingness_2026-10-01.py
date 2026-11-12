#!/usr/bin/env python3
"""GrimAgeV2 and DunedinPACE scores for GS development subjects under the missing-CpG scenarios.

Run with the BioLearn environment (/home/kying/anaconda3/envs/integrative/bin/python). Clock
CpG values are streamed from gs.tar.1.gz with the v12 Task B helper (M-values converted to
beta). Scenarios mirror scenario_gs_missingness: random_<p> removes p% of each subject's observed
clock CpGs (seeded); epic_v2 / hm450 / hm27 keep only clock CpGs on that platform. Missing clock
CpGs are filled with the development-set mean of the full data (as the v12 Task F analysis
filled with training means); BioLearn's averaging handles CpGs absent from the archive.
GrimAgeV2 uses fixed age 47 and female = 1, matching the v12 scores.
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
CACHE = PROJECT / "task_a_private_training_20260629/_frame_cache"
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
OUT = Path(sys.argv[1])


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_task_b():
    path = TASK_B
    spec = importlib.util.spec_from_file_location("task_b_gs_age", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["task_b_gs_age"] = mod
    spec.loader.exec_module(mod)
    return mod, path


def clock_cpgs():
    d = Path(biolearn.__file__).resolve().parent / "data"
    out = {}
    for name, fn, col in (("GrimAgeV2", "GrimAgeV2.csv", "var"), ("DunedinPACE", "DunedinPACE.csv", "CpGmarker")):
        df = pd.read_csv(d / fn)
        col = col if col in df.columns else df.columns[0]
        v = df[col].dropna().astype(str)
        out[name] = v[v.str.startswith("cg")].drop_duplicates().tolist()
    return out


def platform_sets():
    with zipfile.ZipFile(PROJECT / "infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip") as z:
        raw = z.read("infinium-methylationepic-v-1-0-b5-manifest-file.csv").decode()
    epic = pd.read_csv(io.StringIO(raw[raw.index("IlmnID,Name,"):].split("[Controls]")[0]), low_memory=False)
    epic["Name"] = epic["Name"].astype(str)
    v2 = pd.read_csv("/Entropy/kying/09_shallowBS_is_all_you_need/data/EPICv2.hg38.manifest.tsv.gz", sep="\t", usecols=["Probe_ID"])
    return {"epic_v2": set(v2.Probe_ID.astype(str).str.replace(r"_.*$", "", regex=True)),
            "hm450": set(epic.Name[epic["Methyl450_Loci"].astype(str).str.upper().eq("TRUE")]),
            "hm27": set(epic.Name[epic["Methyl27_Loci"].astype(str).str.upper().eq("TRUE")])}


def main():
    t0 = time.time()
    task_b, tb_path = load_task_b()
    cl = clock_cpgs()
    target = sorted(set(cl["GrimAgeV2"]) | set(cl["DunedinPACE"]))
    lab = pd.read_csv(OUT.parent / "gs_dev_ids.csv")  # written from the frame-cache labels (no pyarrow here)
    ids = lab.dnam_id.astype(str).tolist()
    positions, cpgs, target_idx, n_rows = task_b.row_map_for_targets(target)
    M = np.full((len(ids), len(target)), np.nan, dtype=np.float32)
    row = {s: i for i, s in enumerate(ids)}
    for member, sel, batch in task_b.iter_archive_batches(set(ids), positions, n_rows, len(target), target_idx, 25000):
        for j, s in enumerate(sel):
            M[row[s]] = batch[j]
        log(f"loaded {member}: {len(sel)} subjects")
    np.savez(OUT.with_suffix(".clock_cpg_matrix.npz"), ids=np.asarray(ids, dtype=str), cpgs=np.asarray(target, dtype=str), beta=M)
    means = np.nanmean(M, 0)
    plats = platform_sets()
    meta = pd.DataFrame({"sample_id": ids, "age": 47.0, "Age": 47.0, "sex": 1.0, "Female": 1.0}).set_index("sample_id")
    gallery = ModelGallery()
    scenarios = {"full": None, **{f"random_{p}": p / 100 for p in (30, 50, 70, 90)}, **{k: k for k in plats}}
    rows = []
    for name, spec in scenarios.items():
        X = M.copy()
        if isinstance(spec, float):
            r = np.random.default_rng(20261001 + int(spec * 100))
            X[r.random(X.shape) < spec] = np.nan
        elif isinstance(spec, str):
            keep = np.array([c in plats[spec] for c in target])
            X[:, ~keep] = np.nan
        X = np.where(np.isfinite(X), X, means[None, :])
        beta = pd.DataFrame(X.T, index=target, columns=ids)
        data = GeoData(meta, beta)
        out = pd.DataFrame({"sample_id": ids, "scenario": name})
        for model_name, col in (("GrimAgeV2", "DNAmGrimAge"), ("DunedinPACE", "DunedinPACE")):
            pred = gallery.get(model_name, imputation_method="averaging").predict(data)
            out[model_name] = pred[col].reindex(ids).to_numpy() if col in pred.columns else pred.iloc[:, 0].reindex(ids).to_numpy()
        rows.append(out)
        log(f"{name}: scored; observed clock-CpG fraction {np.isfinite(np.where(np.isfinite(M), M, np.nan) if spec is None else X).mean():.3f}")
    pd.concat(rows).to_csv(OUT, index=False)
    json.dump({"task_b_helper": str(tb_path), "clock_cpgs": {k: len(v) for k, v in cl.items()}, "target_cpgs": len(target),
               "captured_in_archive": len(set(cpgs)), "scenarios": list(scenarios), "elapsed_s": time.time() - t0},
              open(OUT.with_suffix(".json"), "w"), indent=1)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
