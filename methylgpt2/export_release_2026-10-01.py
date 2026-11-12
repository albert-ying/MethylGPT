#!/usr/bin/env python3
"""Export the final MethylGPT model as a self-contained bundle for the methylgpt2 package.

Bundle (artifacts/release/methylgpt2-1.0): config.json (architecture, age and clock scaling,
names), vocab.txt (49,156 CpG IDs in model order), core.npz (PPCA mean, loadings, variances,
noise variance at rank 1,536) and weights.pt (encoder and head state dicts, core buffers
removed). The package is then checked against the research pipeline on 200 AltumAge test
profiles (age, clocks, embedding) and on 50 held-out profiles with 30% of CpGs masked
(imputation); the largest absolute differences are recorded in verification.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, Corpus, Finv, PPCACore, load_altumage_split, log, require
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint

HERE = Path(__file__).resolve().parent


def main():
    cuda_preinit()
    dev = torch.device("cuda:0")
    final = require(ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt")
    out = ARTIFACTS / "release/methylgpt2-1.0"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(final, map_location="cpu", weights_only=False)
    enc = torch.load(require(HERE / ck["ckpt"]), map_location="cpu", weights_only=False)
    c = enc["config"]
    core = PPCACore.load(Path(c["core"]), rank=c["rank"])
    vocab = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str)
    np.savez(out / "core.npz", mu=core.mu.numpy(), V=core.V.numpy(), lam=core.lam.numpy(), sigma2=np.float64(core.sigma2))
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    cfg = {"model": "MethylGPT 1.0 (retrained October 2026)", "n_cpgs": int(len(vocab)), "rank": int(c["rank"]),
           "architecture": {k: c[k] if k in c else c.get(k.replace("_", "-")) for k in
                            ("d_tok", "d", "heads", "latents", "score_tokens", "self_layers", "d_read", "encoder",
                             "d_dense", "dense_tokens")},
           "age_center": float(ck["fm"]), "age_scale": float(ck["fs"]), "age_transform": "Horvath log-linear, adult age 20",
           "tissues": list(ck["tissues"]), "clock_names": list(ck["bio_names"]),
           "clock_median": [float(v) for v in ck["bio_median"]], "clock_scale": [float(v) for v in ck["bio_scale"]],
           "source_checkpoint": str(final.relative_to(HERE))}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    state = {k: v for k, v in ck["model"].items() if not k.startswith("core.")}
    torch.save({"encoder": state, "heads": ck["heads"]}, out / "weights.pt")
    log(f"bundle written to {out}")

    # check the package against the research pipeline
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    pkg = MethylGPT.from_pretrained(out, device="cuda:0")
    model, rcore, _ = load_checkpoint(HERE / ck["ckpt"], None, dev)
    model.load_state_dict(ck["model"]); model.eval()
    heads = BioHeads(rcore.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(ck["bio_names"])).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    ids, X, y = load_altumage_split("test")
    X, ids = X[:200], ids[:200]
    with torch.no_grad():
        x = torch.from_numpy(X).to(dev)
        e = torch.cat([embed_batch(model, rcore, x[i:i + 32], torch.isfinite(x[i:i + 32])) for i in range(0, len(x), 32)])
        age, _, _, bio = heads(e)
    ref_age = Finv(age.float().cpu().numpy() * ck["fs"] + ck["fm"])
    ref_bio = bio.float().cpu().numpy() * ck["bio_scale"] + ck["bio_median"]
    df = pd.DataFrame(X, index=ids, columns=vocab)
    p = pkg.predict(df)
    emb = pkg.embed(df).to_numpy()
    ver = {"age_max_abs_diff_years": float(np.abs(p.age.to_numpy() - ref_age).max()),
           "clock_max_abs_diff_in_training_sd_units": float((np.abs(p[cfg["clock_names"]].to_numpy() - ref_bio)
                                                               / np.asarray(cfg["clock_scale"])).max()),
           "embedding_max_abs_diff": float(np.abs(emb - e.float().cpu().numpy()).max())}
    corpus = Corpus()
    rows = np.sort(np.random.default_rng(7).choice(corpus.rows("test"), 50, replace=False))
    Xh = corpus.get(rows)
    m = np.random.default_rng(8).random(Xh.shape) < 0.3
    Xm = np.where(m, np.nan, Xh)
    imp = pkg.impute(pd.DataFrame(Xm, columns=vocab)).to_numpy()
    tgt = m & np.isfinite(Xh)
    ver["impute_masked_mse"] = float(np.mean((imp[tgt] - Xh[tgt]) ** 2))
    ver["impute_observed_unchanged"] = bool(np.array_equal(imp[np.isfinite(Xm)], Xm[np.isfinite(Xm)]))
    (out / "verification.json").write_text(json.dumps(ver, indent=1))
    log(f"verification: {ver}")


if __name__ == "__main__":
    main()
