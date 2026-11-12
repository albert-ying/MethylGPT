#!/usr/bin/env python3
"""Add the age-stage encoder to release bundle methylgpt2-1.1 and check that the package reproduces the disease embedding.

The disease models of the manuscript (development folds, test and validation subsets) were fitted on the embedding
of run 1 at the end of the age, sex and tissue stage (artifacts/runs/aging_heads_clean_2026-10-01/best.pt, step
8,000), extracted by extract_gs_features_2026-10-01.py. The released weights.pt is the same run after the
clock-distillation stage. This script writes weights_age_stage.pt (encoder only) into the bundle, records it in
config.json under "embedding_stages", and compares package embed(stage="age_stage") with the stored Generation
Scotland development embedding (aggregate differences only; no participant-level output).
Output: artifacts/eval/disease_embedding_reproduction_2026-10-09.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
SRC = "runs/aging_heads_clean_2026-10-01/best.pt"
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"


def main():
    cuda_preinit()
    ag = torch.load(require(ARTIFACTS / SRC), map_location="cpu", weights_only=False)
    rel = torch.load(require(BUNDLE / "weights.pt"), map_location="cpu", weights_only=True)
    cfg = json.loads((BUNDLE / "config.json").read_text())
    run1 = torch.load(require(HERE / cfg["source_checkpoint"]), map_location="cpu", weights_only=False)
    assert ag["ckpt"] == run1["ckpt"], "the age-stage and clock-stage checkpoints do not share the pretrained encoder"
    state = {k: v for k, v in ag["model"].items() if not k.startswith("core.")}
    assert set(state) == set(rel["encoder"]), "age-stage encoder keys differ from the released encoder"
    torch.save({"encoder": state}, BUNDLE / "weights_age_stage.pt")
    cfg["embedding_stages"] = {"age_stage": {
        "weights": "weights_age_stage.pt", "source_checkpoint": f"artifacts/{SRC}", "step": int(ag["step"]),
        "description": "run 1 at the end of the age, sex and tissue stage, before clock distillation; "
                       "the embedding of the disease models in the MethylGPT paper"}}
    (BUNDLE / "config.json").write_text(json.dumps(cfg, indent=1))
    log(f"wrote {BUNDLE / 'weights_age_stage.pt'} (step {ag['step']})")

    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT, __version__
    model = MethylGPT.from_pretrained(BUNDLE, device="cuda:0")
    res = {"package_version": __version__, "bundle": BUNDLE.name, "source_checkpoint": f"artifacts/{SRC}", "step": int(ag["step"]), "splits": {}}
    for split in ("train",):
        mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
        ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
        with np.load(mp, allow_pickle=False) as z:
            X = z["matrix"].astype(np.float32)
        st = np.load(require(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_mgpt2_embed.embedding.npz"))
        assert (st["dnam_id"].astype(str) == ids).all()
        ref = st["embedding"].astype(np.float64)
        df = pd.DataFrame(X, columns=model.vocab)
        out = {}
        for tf32 in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            E = model.embed(df, stage="age_stage").to_numpy(np.float64)
            d = np.abs(E - ref)
            out[f"tf32_{'on' if tf32 else 'off'}"] = {
                "max_abs_difference": float(d.max()), "mean_abs_difference": float(d.mean()),
                "min_per_dimension_pearson_r": float(min(np.corrcoef(E[:, j], ref[:, j])[0, 1] for j in range(E.shape[1]) if ref[:, j].std() > 0)),
                "embedding_sd_overall": float(ref.std())}
        torch.backends.cuda.matmul.allow_tf32 = False
        F = model.embed(df).to_numpy(np.float64)
        out["released_final_stage_embedding_vs_disease_embedding"] = {
            "mean_abs_difference": float(np.abs(F - ref).mean()),
            "median_per_dimension_pearson_r_encoder_part": float(np.median([np.corrcoef(F[:, j], ref[:, j])[0, 1] for j in range(1536, F.shape[1])]))}
        res["splits"][split] = {"participants": int(len(ids)), "dimensions": int(ref.shape[1]), **out}
        log(f"{split}: {res['splits'][split]}")
    write_json(ARTIFACTS / "eval/disease_embedding_reproduction_2026-10-09.json", res)


if __name__ == "__main__":
    main()
