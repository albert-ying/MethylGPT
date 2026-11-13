#!/usr/bin/env python3
"""Predicted age during iPSC reprogramming (GSE54848, 450K).

GSE54848 was not in the MethylGPT-2 corpus. The 30 methylation samples are dermal fibroblasts
(day 0), reprogramming intermediates (days 3-28), iPSCs after passages 1-3 (days 35, 42, 49),
H9 ESCs and the 201B7 iPSC line. MethylGPT-2 predicts age from the vocabulary CpGs; Horvath
2013 and Horvath skin & blood are computed by BioLearn from all 450K CpGs (integrative env).
Every replicate is reported; there are 2 or 3 per time point.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt_lib import cuda_preinit, ARTIFACTS, Finv, log, require, write_json
from methylgpt_model import BioHeads, embed_batch, load_checkpoint

D = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/dataset/GSE54848")
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"
DAY = {"p1": 35, "p2": 42, "p3": 49}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=f"MethylGPT-2={ARTIFACTS / 'runs/multitask_bio_2026-10-01/final.pt'}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(ARTIFACTS / "eval/reprogramming_gse54848_2026-10-01.csv"))
    a = ap.parse_args()
    cuda_preinit()
    meta = pd.read_csv(require(D / "metadata.csv"))
    meta = meta[meta.platform_id == "GPL13534"].reset_index(drop=True)
    gsm = meta.geo_accession.tolist()
    m = pd.read_csv(require(D / "expression_matrix.csv"), usecols=["ID_REF"] + gsm, index_col=0)
    m = m[m.index.astype(str).str.startswith("cg")].astype(float)
    log(f"{len(gsm)} methylation samples, {len(m)} CpG rows")
    t = meta.title.astype(str)
    day = [float(re.match(r"Day(\d+)_", s).group(1)) if s.startswith("Day") else DAY.get(s.split("_")[0], np.nan) for s in t]
    out = pd.DataFrame({"gsm": gsm, "title": t, "cell_type": meta["characteristics_ch1.0.cell type"], "day": day})

    vocab = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str)
    X = m.reindex(vocab).T.to_numpy(np.float32)
    X = np.where(np.isfinite(X), np.clip(X, 0, 1), np.nan).astype(np.float32)
    out["vocab_cpgs_observed"] = np.isfinite(X).sum(1)
    dev = torch.device(a.device)
    for spec in a.models.split(","):
        name, path = spec.split("=", 1)
        ck = torch.load(require(Path(path)), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(ck["ckpt"]), None, dev)
        model.load_state_dict(ck["model"]); model.eval()
        heads = BioHeads(core.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(ck["bio_names"])).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(dev)
            age = heads(embed_batch(model, core, x, torch.isfinite(x)))[0].float().cpu().numpy()
        out[name] = Finv(age * ck["fs"] + ck["fm"])

    tmp = Path(a.out).with_suffix(".betas.csv")
    m.to_csv(tmp)
    code = (
        "import pandas as pd, sys\n"
        "from biolearn.data_library import GeoData\nfrom biolearn.model_gallery import ModelGallery\n"
        f"b = pd.read_csv('{tmp}', index_col=0)\n"
        "meta = pd.DataFrame(index=b.columns)\n"
        "g = ModelGallery(); out = pd.DataFrame(index=b.columns)\n"
        "for n in ['Horvathv1', 'Horvathv2']:\n"
        "    p = g.get(n, imputation_method='averaging').predict(GeoData(meta, b))\n"
        "    out[n] = p.iloc[:, 0].reindex(b.columns)\n"
        f"out.to_csv('{tmp.with_suffix('.clocks.csv')}')\n")
    subprocess.run([BIOLEARN_PY, "-c", code], check=True, capture_output=True)
    ck_ = pd.read_csv(tmp.with_suffix(".clocks.csv"), index_col=0).reindex(gsm)
    out["Horvath 2013"] = ck_.Horvathv1.to_numpy()
    out["Horvath skin & blood"] = ck_.Horvathv2.to_numpy()
    tmp.unlink()
    out.to_csv(a.out, index=False)
    summ = out.groupby("day", dropna=True)[[c for c in out.columns if c not in ("gsm", "title", "cell_type", "day", "vocab_cpgs_observed")]].agg(["mean", "std", "count"])
    summ.columns = [f"{c} {s}" for c, s in summ.columns]
    write_json(Path(a.out).with_suffix(".json"), {"per_day": {str(k): v for k, v in summ.round(3).to_dict(orient="index").items()},
                                                   "n_samples": len(out)})
    log(summ.round(2).to_string())


if __name__ == "__main__":
    main()
