#!/usr/bin/env python3
"""Build the retraining corpus: a float16 memory map of all pretraining profiles and a
dataset-level train/dev/test split, fixed before any model is fitted.

Inputs (empirical, read-only):
  processed_dataset/processed_type3_parquet_shuffled/*.parquet  (169,120 profiles x 49,156 CpGs)
  QCed_samples_type3.csv            sample -> dataset name
  compiled_metadata.csv             GSM -> tissue, age, sex
  /Entropy/kying/GEO_data/gse_me_hs.csv   GSE -> platform (GPL)
  finetuning/age_prediction/age_data/altumage_metadata/altumage_{train,valid,test}.parquet

Outputs (artifacts/corpus_2026-09-30/):
  beta_f16.npy       N x 49,156 float16, NaN = not measured; rows in sorted-block order
  profiles.parquet   one row per profile with id, group, metadata and split
  split_manifest.json

Split rules
  * Dataset names are reduced to their base accession (EWAS Data Hub chunk suffixes such
    as "-1250" removed); base datasets that share any sample are merged into one group.
  * Groups are shuffled with a fixed seed and assigned to test until 8% of profiles and
    then to dev until 2%; groups larger than 2% of the corpus always go to train.
  * AltumAge validation and test profiles never enter pretraining: if their group is in
    train they are marked "excluded_downstream".
  * Observed values outside [0, 1] are clipped; profiles with more than 1% of observed
    values outside [0, 1] are marked "excluded_qc".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
PARQUET_DIR = PROJECT / "processed_dataset/processed_type3_parquet_shuffled"
PROBES = PROJECT / "processed_dataset/probe_ids_type3.csv"
QC_TABLE = PROJECT / "QCed_samples_type3.csv"
META = PROJECT / "compiled_metadata.csv"
GSE_PLATFORM = Path("/Entropy/kying/GEO_data/gse_me_hs.csv")
ALTUMAGE = PROJECT / "finetuning/age_prediction/age_data/altumage_metadata"
N_CPG = 49156


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def require(path: Path):
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def file_record(path: Path, hash_content: bool):
    st = path.stat()
    rec = {"path": str(path), "bytes": st.st_size, "mtime": st.st_mtime}
    if hash_content:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        rec["sha256"] = h.hexdigest()
    return rec


def write_block(args):
    path, offset, n_rows, out_path = args
    table = pq.read_table(path)
    col = table.column("data").combine_chunks()
    lengths = np.diff(col.offsets.to_numpy())
    if len(table) != n_rows or not np.all(lengths == N_CPG):
        raise ValueError(f"Unexpected shape in {path}")
    X = col.values.to_numpy(zero_copy_only=False).reshape(n_rows, N_CPG)
    finite = np.isfinite(X)
    oor = finite & ((X < 0) | (X > 1))
    oor_n = oor.sum(1).astype(np.int32)
    with np.errstate(invalid="ignore"):
        row_min = np.nanmin(np.where(finite, X, np.nan), axis=1)
        row_max = np.nanmax(np.where(finite, X, np.nan), axis=1)
    mm = np.load(out_path, mmap_mode="r+")
    mm[offset:offset + n_rows] = np.clip(X, 0, 1).astype(np.float16)  # NaN stays NaN
    mm.flush()
    del mm
    return ([str(x) for x in table.column("id").to_pylist()], finite.sum(1).astype(np.int32),
            oor_n, row_min, row_max)


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "artifacts/corpus_2026-09-30"))
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for p in (PARQUET_DIR, PROBES, QC_TABLE, META, GSE_PLATFORM, ALTUMAGE):
        require(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    blocks = sorted(f for f in os.listdir(PARQUET_DIR) if f.endswith(".parquet"))
    rows = [pq.ParquetFile(PARQUET_DIR / f).metadata.num_rows for f in blocks]
    offsets = np.concatenate([[0], np.cumsum(rows)[:-1]])
    n = int(sum(rows))
    beta_path = out / "beta_f16.npy"
    mm = np.lib.format.open_memmap(beta_path, mode="w+", dtype=np.float16, shape=(n, N_CPG))
    del mm
    log(f"writing {n} x {N_CPG} float16 memmap from {len(blocks)} blocks")
    jobs = [(PARQUET_DIR / f, int(o), int(r), beta_path) for f, o, r in zip(blocks, offsets, rows)]
    with Pool(a.workers) as pool:
        results = pool.map(write_block, jobs)
    ids = [i for r in results for i in r[0]]
    n_obs = np.concatenate([r[1] for r in results])
    oor_n = np.concatenate([r[2] for r in results])
    row_min = np.concatenate([r[3] for r in results])
    row_max = np.concatenate([r[4] for r in results])
    block_of_row = np.repeat(np.arange(len(blocks)), rows)
    if len(set(ids)) != n:
        raise ValueError("Duplicate profile IDs across blocks")
    log("memmap written")

    qc = pd.read_csv(QC_TABLE, usecols=["dataset_name", "sample_name"])
    qc["base"] = qc.dataset_name.astype(str).str.replace(r"-\d+$", "", regex=True)
    uf = UnionFind()
    for _, g in qc.groupby("sample_name").base:
        bases = sorted(set(g))
        for b in bases[1:]:
            uf.union(bases[0], b)
        uf.find(bases[0])
    sample_base = qc.groupby("sample_name").base.first()
    group = [uf.find(sample_base[i]) if i in sample_base.index else f"unmapped:{i}" for i in ids]
    profiles = pd.DataFrame({"row": np.arange(n), "id": ids, "block": [blocks[b] for b in block_of_row],
                             "group": group, "n_obs": n_obs, "n_out_of_range": oor_n,
                             "raw_min": row_min, "raw_max": row_max})

    meta = pd.read_csv(META, usecols=["GSM_ID", "sex", "age", "tissue"]).drop_duplicates("GSM_ID")
    meta["age"] = pd.to_numeric(meta.age, errors="coerce")
    profiles = profiles.merge(meta.rename(columns={"GSM_ID": "id"}), on="id", how="left")
    plat = pd.read_csv(GSE_PLATFORM, usecols=["accession", "platform"]).dropna()
    plat_map = dict(zip(plat.accession.astype(str), plat.platform.astype(str)))
    profiles["gpl"] = [plat_map.get(g.split(":")[0]) for g in profiles.group]

    alt = {}
    for split in ("train", "valid", "test"):
        t = pq.read_table(ALTUMAGE / f"altumage_{split}.parquet", columns=["id"])
        for i in t.column("id").to_pylist():
            alt[str(i)] = split
    profiles["altumage_split"] = profiles.id.map(alt)

    rng = np.random.default_rng(a.seed)
    sizes = profiles.groupby("group").size()
    cap = 0.02 * n
    order = rng.permutation(sizes.index.to_numpy())
    assign, n_test, n_dev = {}, 0, 0
    for g in order:
        s = int(sizes[g])
        if s > cap:
            assign[g] = "train"
        elif n_test < a.test_frac * n:
            assign[g] = "test"; n_test += s
        elif n_dev < a.dev_frac * n:
            assign[g] = "dev"; n_dev += s
        else:
            assign[g] = "train"
    profiles["split"] = profiles.group.map(assign)
    excl = (profiles.split == "train") & profiles.altumage_split.isin(["valid", "test"])
    profiles.loc[excl, "split"] = "excluded_downstream"
    # Profiles with more than 1% of observed values outside [0, 1] are not on the beta scale.
    bad = profiles.n_out_of_range > 0.01 * profiles.n_obs
    profiles.loc[bad, "split"] = "excluded_qc"
    profiles.to_parquet(out / "profiles.parquet", index=False)

    comp = {}
    for col in ("split",):
        comp["profiles"] = profiles[col].value_counts().to_dict()
        comp["groups"] = profiles.groupby(col).group.nunique().to_dict()
    tissue_top = profiles.tissue.value_counts().head(8).index
    comp["tissue_by_split"] = (profiles[profiles.tissue.isin(tissue_top)]
                               .groupby(["split", "tissue"]).size().unstack(fill_value=0).to_dict())
    comp["gpl_by_split"] = profiles.groupby(["split", "gpl"]).size().unstack(fill_value=0).to_dict()
    comp["altumage_by_split"] = profiles.groupby(["split", "altumage_split"]).size().unstack(fill_value=0).to_dict()
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": a.seed,
        "rules": __doc__.split("Split rules")[1].strip(),
        "n_profiles": n, "n_cpg": N_CPG, "blocks": blocks, "rows_per_block": rows,
        "inputs": {"qc_table": file_record(QC_TABLE, True), "probes": file_record(PROBES, True),
                   "metadata": file_record(META, True), "gse_platform": file_record(GSE_PLATFORM, True),
                   "parquet_blocks": [file_record(PARQUET_DIR / f, False) for f in blocks]},
        "outputs": {"beta_f16": file_record(beta_path, False), "profiles": file_record(out / "profiles.parquet", True)},
        "composition": comp,
        "out_of_range": {"values_total": int(oor_n.sum()), "profiles_with_any": int((oor_n > 0).sum()),
                         "profiles_excluded_over_1pct": int(bad.sum()),
                         "raw_min": float(np.nanmin(row_min)), "raw_max": float(np.nanmax(row_max)),
                         "handling": "values clipped to [0, 1]; profiles with >1% out of range excluded"},
        "test_groups": sorted(g for g, s in assign.items() if s == "test"),
        "dev_groups": sorted(g for g, s in assign.items() if s == "dev"),
        "elapsed_s": time.time() - t0,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"profiles by split: {comp['profiles']}; groups: {comp['groups']}")
    log(f"AltumAge by split: {comp['altumage_by_split']}")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
