#!/usr/bin/env python3
"""MethylGPT-2 features for Generation Scotland development subjects (train and val splits of
the locked incident-disease protocol). GS was never in the pretraining corpus.

Features (one .embedding.npz per split and kind, keys dnam_id and embedding):
  embed  [standardised core scores (1,536), pooled transformer latents (512)] from the
         aging-head encoder
  age    MethylGPT-2 predicted age (years), one column
Inputs are the frame-cache beta matrices (5,257 vocabulary CpGs absent from GS are
unobserved). No outcome enters the features. Outputs stay in artifacts/ (git-ignored)
because they are derived from controlled data.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, Finv, log, require, write_json
from methylgpt2_model import BioHeads, Heads, embed_batch, load_checkpoint

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging", default=str(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "gs_features_2026-10-01"))
    ap.add_argument("--prefix", default="mgpt2", help="feature tag prefix")
    a = ap.parse_args()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ag = torch.load(a.aging, map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    model.load_state_dict(ag["model"]); model.eval()
    multitask = "bio_names" in ag
    if multitask:
        heads = BioHeads(core.rank + model.latents.shape[1], len(ag["tissues"]) + 1, len(ag["bio_names"])).to(dev)
    else:
        heads = Heads(core.rank + model.latents.shape[1], len(ag["tissues"]) + 1).to(dev)
    heads.load_state_dict(ag["heads"]); heads.eval()
    record = {"aging_checkpoint": a.aging, "aging_step": ag["step"], "splits": {}}
    for split in ("train", "val"):
        mpath = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
        lpath = mpath.with_name(mpath.name.replace(".matrix.npz", ".labels.parquet"))
        ids = np.asarray(pd.read_parquet(lpath, columns=["dnam_id"])["dnam_id"].astype(str).tolist(), dtype=str)
        with np.load(mpath, allow_pickle=False) as z:
            X = z["matrix"].astype(np.float32)
        if X.shape != (len(ids), 49156):
            raise ValueError(f"Unexpected GS matrix shape {X.shape}")
        emb, age, bio = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                x = torch.from_numpy(X[i:i + 32]).to(dev)
                e = embed_batch(model, core, x, torch.isfinite(x))
                outs = heads(e)
                pa = outs[0]
                if multitask:
                    bio.append(outs[3].float().cpu().numpy())
                emb.append(e.float().cpu().numpy())
                age.append(Finv(pa.float().cpu().numpy() * ag["fs"] + ag["fm"]))
        emb, age = np.concatenate(emb), np.concatenate(age)
        P = a.prefix
        np.savez(out / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{P}_embed.embedding.npz", dnam_id=ids, embedding=emb.astype(np.float32))
        np.savez(out / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{P}_age.embedding.npz", dnam_id=ids, embedding=age[:, None].astype(np.float32))
        if multitask:
            bio = np.concatenate(bio)
            np.savez(out / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{P}_bio.embedding.npz", dnam_id=ids, embedding=bio.astype(np.float32))
            np.savez(out / f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_{P}_embedbio.embedding.npz", dnam_id=ids,
                     embedding=np.hstack([emb, bio]).astype(np.float32))
            record.setdefault("bio_names", ag["bio_names"])
        record["splits"][split] = {"n": int(len(ids)), "matrix": str(mpath), "observed_fraction": float(np.isfinite(X).mean()),
                                   "embedding_dim": int(emb.shape[1]), "predicted_age_mean": float(age.mean())}
        log(f"{split}: {len(ids)} subjects, embedding {emb.shape}, predicted age mean {age.mean():.2f}")
    record["elapsed_s"] = time.time() - t0
    write_json(out / "extraction_record.json", record)


if __name__ == "__main__":
    main()
