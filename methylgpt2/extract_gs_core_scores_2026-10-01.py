#!/usr/bin/env python3
"""Standardised PPCA factor scores (s / sqrt(lam)) of GS development subjects under a given core.

Writes <prefix>_core.embedding.npz for the train and val splits of the locked protocol (same
naming as extract_gs_features_2026-10-01.py, so compare_joint_cox_mgpt2_2026-10-01.py can read
them with --embedding-tag). Each subject's posterior uses only its observed CpGs. Also reports,
on the train split, how much of each CpG's GS variance the core reconstructs, for the CpGs that
carry the endocrine/metabolic and smoking signal. The test split is not read. Outputs stay in
artifacts/ (derived from controlled data).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, PPCACore, log, require, write_json

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
MARKERS = {"cg06500161": "ABCG1", "cg19693031": "TXNIP", "cg27243685": "ABCG1", "cg11024682": "SREBF1",
           "cg00574958": "CPT1A", "cg03636183": "F2RL3", "cg05575921": "AHRR", "cg21566642": "2q37.1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "gs_features_2026-10-01"))
    a = ap.parse_args()
    cuda_preinit()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    core = PPCACore.load(require(Path(a.core)), rank=a.rank).to(dev)
    vocab = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    out = Path(a.out_dir)
    record = {"core": a.core, "rank": core.rank, "splits": {}}
    for split in ("train", "val"):
        mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
        ids = pd.read_parquet(mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
        with np.load(mpath, allow_pickle=False) as z:
            X = z["matrix"].astype(np.float32)
        S, sse, sum_, sum2, n = [], 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X), 64):
                x = torch.from_numpy(X[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                s = core.posterior(x, obs.float())
                S.append((s / core.lam.sqrt()).cpu().numpy())
                if split == "train":
                    xf = torch.where(obs, x, torch.zeros((), device=dev)).double()
                    res = torch.where(obs, x - core.predict(s), torch.zeros((), device=dev)).double()
                    sse = sse + (res ** 2).sum(0); sum_ = sum_ + xf.sum(0); sum2 = sum2 + (xf ** 2).sum(0)
                    n = n + obs.double().sum(0)
        S = np.concatenate(S).astype(np.float32)
        np.savez(out / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{a.prefix}_core.embedding.npz", dnam_id=ids.astype(str), embedding=S)
        record["splits"][split] = {"n": int(len(ids)), "dim": int(S.shape[1]), "observed_fraction": float(np.isfinite(X).mean())}
        if split == "train":
            nn_ = n.clamp_min(1)
            var = sum2 / nn_ - (sum_ / nn_) ** 2
            r2 = (1 - (sse / nn_) / var.clamp_min(1e-12)).cpu().numpy()
            ok = (n > 0.5 * len(X)).cpu().numpy()
            record["median_cpg_r2_gs"] = float(np.median(r2[ok]))
            record["marker_r2_gs"] = {f"{c} ({g})": float(r2[np.flatnonzero(vocab == c)[0]]) for c, g in MARKERS.items()}
        log(f"{split}: {S.shape}")
    log(f"median CpG R2 in GS {record['median_cpg_r2_gs']:.3f}; markers {record['marker_r2_gs']}")
    write_json(out / f"extraction_record_{a.prefix}_core.json", record)


if __name__ == "__main__":
    main()
