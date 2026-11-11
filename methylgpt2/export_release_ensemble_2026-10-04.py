#!/usr/bin/env python3
"""Export methylgpt2 bundle 1.1: run 1 as in 1.0 plus two extra training runs for an age ensemble.

Inputs
  artifacts/runs/multitask_bio_2026-10-01/final.pt      run 1 (seed 20261001; the 1.0 release model)
  artifacts/runs/multitask_bio_seed20261002/final.pt    run 2 (seed 20261002)
  artifacts/runs/multitask_bio_seed20261003/final.pt    run 3 (seed 20261003)
  artifacts/release/methylgpt2-1.0                      read only, for the imputation and 1.0 checks

Method
  1. Check that the three runs share the factor model core (same encoder checkpoint, same PPCA
     core path and rank, identical core buffers in the saved state) and the same age scaling
     (fm, fs), tissue list and clock names; abort otherwise.
  2. Write run 1's files as export_release_2026-10-01.py does (config.json, vocab.txt, core.npz,
     weights.pt) and the encoder + heads state dicts of runs 2 and 3 (core buffers removed) as
     weights_run2.pt and weights_run3.pt. config.json lists the three members under
     "age_ensemble"; predicted age is their mean in years.
  3. Verify the package (package/src) on 200 AltumAge test profiles against the research code:
     ensemble age vs the mean of load_model() predictions (evaluate_age_revision_2026-10-01.py)
     of the three checkpoints, each run's age, run 1's clocks and embedding; imputation of 50
     held-out profiles with 30% of CpGs masked, compared between bundles 1.1 and 1.0; and the
     1.0 bundle loaded with the new API against research run 1.

Outputs
  artifacts/release/methylgpt2-1.1/{config.json, vocab.txt, core.npz, weights.pt,
  weights_run2.pt, weights_run3.pt, verification.json}.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import cuda_preinit, ARTIFACTS, Corpus, Finv, PPCACore, load_altumage_split, log, require
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint

HERE = Path(__file__).resolve().parent
RUNS = [("run1", 20261001, "runs/multitask_bio_2026-10-01/final.pt", "weights.pt"),
        ("run2", 20261002, "runs/multitask_bio_seed20261002/final.pt", "weights_run2.pt"),
        ("run3", 20261003, "runs/multitask_bio_seed20261003/final.pt", "weights_run3.pt")]


def research_load_model():
    """load_model() from evaluate_age_revision_2026-10-01.py (file name is not importable directly)."""
    spec = importlib.util.spec_from_file_location("evaluate_age_revision", HERE / "evaluate_age_revision_2026-10-01.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_model


def check_shared(cks):
    """Confirm all runs share run 1's core and age scaling; return the per-run summary."""
    ref = cks[0]
    out = []
    for (name, seed, path, _), ck in zip(RUNS, cks):
        enc = torch.load(require(HERE / ck["ckpt"]), map_location="cpu", weights_only=False)
        core_path = (ck.get("config") or {}).get("core") or enc["config"]["core"]
        row = {"name": name, "seed": seed, "checkpoint": f"artifacts/{path}", "encoder_checkpoint": ck["ckpt"],
               "core": str(core_path), "rank": int(enc["config"]["rank"]), "fm": float(ck["fm"]), "fs": float(ck["fs"]),
               "core_buffers_equal_run1": all(torch.equal(ck["model"][k], ref["model"][k])
                                              for k in ref["model"] if k.startswith("core.")),
               "tissues_equal_run1": list(ck["tissues"]) == list(ref["tissues"]),
               "clock_names_equal_run1": list(ck["bio_names"]) == list(ref["bio_names"])}
        out.append(row)
    r1 = out[0]
    for row in out:
        same = (row["core"] == r1["core"] and row["rank"] == r1["rank"] and row["encoder_checkpoint"] == r1["encoder_checkpoint"]
                and row["fm"] == r1["fm"] and row["fs"] == r1["fs"] and row["core_buffers_equal_run1"]
                and row["tissues_equal_run1"] and row["clock_names_equal_run1"])
        if not same:
            raise SystemExit(f"run {row['name']} does not share run 1's core or age scaling: {row}")
    return out


def main():
    cuda_preinit()
    dev = torch.device("cuda:0")
    out = ARTIFACTS / "release/methylgpt2-1.1"
    old = require(ARTIFACTS / "release/methylgpt2-1.0")
    out.mkdir(parents=True, exist_ok=True)
    cks = [torch.load(require(ARTIFACTS / p), map_location="cpu", weights_only=False) for _, _, p, _ in RUNS]
    shared = check_shared(cks)
    log(f"shared core and age scaling confirmed: {[(r['name'], r['core'], r['fm'], r['fs']) for r in shared]}")

    ck = cks[0]
    enc = torch.load(require(HERE / ck["ckpt"]), map_location="cpu", weights_only=False)
    c = enc["config"]
    core = PPCACore.load(Path(c["core"]), rank=c["rank"])
    vocab = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str)
    np.savez(out / "core.npz", mu=core.mu.numpy(), V=core.V.numpy(), lam=core.lam.numpy(), sigma2=np.float64(core.sigma2))
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    cfg = {"model": "MethylGPT 1.1 (retrained October 2026; age from three training runs)", "n_cpgs": int(len(vocab)),
           "rank": int(c["rank"]),
           "architecture": {k: c[k] if k in c else c.get(k.replace("_", "-")) for k in
                            ("d_tok", "d", "heads", "latents", "score_tokens", "self_layers", "d_read", "encoder",
                             "d_dense", "dense_tokens")},
           "age_center": float(ck["fm"]), "age_scale": float(ck["fs"]), "age_transform": "Horvath log-linear, adult age 20",
           "tissues": list(ck["tissues"]), "clock_names": list(ck["bio_names"]),
           "clock_median": [float(v) for v in ck["bio_median"]], "clock_scale": [float(v) for v in ck["bio_scale"]],
           "source_checkpoint": str((ARTIFACTS / RUNS[0][2]).relative_to(HERE)),
           "age_ensemble": {"combine": "mean of the members' predicted ages in years; sex, tissue, clocks, "
                                       "embeddings, factor scores and imputation come from the first member",
                            "members": [{"name": n, "seed": s, "weights": w, "age_center": float(k["fm"]),
                                         "age_scale": float(k["fs"]),
                                         "source_checkpoint": str((ARTIFACTS / p).relative_to(HERE))}
                                        for (n, s, p, w), k in zip(RUNS, cks)]}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    for (_, _, _, w), k in zip(RUNS, cks):
        state = {key: v for key, v in k["model"].items() if not key.startswith("core.")}
        torch.save({"encoder": state, "heads": k["heads"]}, out / w)
    log(f"bundle written to {out}")
    del cks, enc

    # package vs research pipeline
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    pkg = MethylGPT.from_pretrained(out, device="cuda:0")
    ids, X, y = load_altumage_split("test")
    X, ids = X[:200], ids[:200]
    df = pd.DataFrame(X, index=ids, columns=vocab)
    p = pkg.predict(df)
    runs = pkg.predict_age_runs(df)
    emb = pkg.embed(df).to_numpy()

    load_model = research_load_model()
    ref_runs = np.stack([load_model(ARTIFACTS / path, dev)(X) for _, _, path, _ in RUNS], 1)
    ref_age = ref_runs.mean(1)

    ck = torch.load(ARTIFACTS / RUNS[0][2], map_location="cpu", weights_only=False)
    model, rcore, _ = load_checkpoint(HERE / ck["ckpt"], None, dev)
    model.load_state_dict(ck["model"]); model.eval()
    heads = BioHeads(rcore.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(ck["bio_names"])).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).to(dev)
        e = torch.cat([embed_batch(model, rcore, x[i:i + 32], torch.isfinite(x[i:i + 32])) for i in range(0, len(x), 32)])
        _, sex, tissue, bio = heads(e)
    ref_bio = bio.float().cpu().numpy() * ck["bio_scale"] + ck["bio_median"]
    ref_pmale = torch.sigmoid(sex).float().cpu().numpy()
    del model, heads, rcore
    torch.cuda.empty_cache()

    ver = {"n_profiles": int(len(X)), "profiles": "first 200 of load_altumage_split('test')",
           "shared_core_check": shared,
           "ensemble_members": [m.name for m in pkg.age_members],
           "ensemble_age_max_abs_diff_years_vs_mean_of_research_load_model": float(np.abs(p.age.to_numpy() - ref_age).max()),
           "predict_age_equals_predict_age_column": bool(np.array_equal(pkg.predict_age(df).to_numpy(), p.age.to_numpy())),
           "age_column_equals_mean_of_age_runs": float(np.abs(runs.to_numpy().mean(1) - p.age.to_numpy()).max()),
           "per_run_age_max_abs_diff_years_vs_research": {n: float(np.abs(runs[n].to_numpy() - ref_runs[:, j]).max())
                                                          for j, (n, _, _, _) in enumerate(RUNS)},
           "between_run_age_sd_years_median": float(np.median(ref_runs.std(1, ddof=1))),
           "run1_clock_max_abs_diff_in_training_sd_units": float((np.abs(p[cfg["clock_names"]].to_numpy() - ref_bio)
                                                                   / np.asarray(cfg["clock_scale"])).max()),
           "run1_p_male_max_abs_diff": float(np.abs(p.p_male.to_numpy() - ref_pmale).max()),
           "run1_tissue_agreement": float(np.mean(p.tissue.to_numpy() == np.asarray(pkg.tissues)[tissue.argmax(1).cpu().numpy()])),
           "run1_embedding_max_abs_diff": float(np.abs(emb - e.float().cpu().numpy()).max())}

    # bundle 1.0 with the new API: one member, age equal to research run 1
    pkg0 = MethylGPT.from_pretrained(old, device="cuda:0")
    p0 = pkg0.predict(df)
    ver["bundle_1_0_members"] = [m.name for m in pkg0.age_members]
    ver["bundle_1_0_age_max_abs_diff_years_vs_research_run1"] = float(np.abs(p0.age.to_numpy() - ref_runs[:, 0]).max())
    ver["bundle_1_0_vs_1_1_clock_p_male_max_abs_diff"] = float(np.abs(p0[["p_male"] + cfg["clock_names"]].to_numpy()
                                                                     - p[["p_male"] + cfg["clock_names"]].to_numpy()).max())
    ver["bundle_1_0_vs_1_1_tissue_identical"] = bool((p0.tissue == p.tissue).all())
    ver["bundle_1_0_vs_1_1_embedding_max_abs_diff"] = float(np.abs(pkg0.embed(df).to_numpy() - emb).max())

    corpus = Corpus()
    rows = np.sort(np.random.default_rng(7).choice(corpus.rows("test"), 50, replace=False))
    Xh = corpus.get(rows)
    m = np.random.default_rng(8).random(Xh.shape) < 0.3
    Xm = np.where(m, np.nan, Xh)
    imp = pkg.impute(pd.DataFrame(Xm, columns=vocab)).to_numpy()
    imp0 = pkg0.impute(pd.DataFrame(Xm, columns=vocab)).to_numpy()
    tgt = m & np.isfinite(Xh)
    ver["impute_masked_mse"] = float(np.mean((imp[tgt] - Xh[tgt]) ** 2))
    ver["impute_observed_unchanged"] = bool(np.array_equal(imp[np.isfinite(Xm)], Xm[np.isfinite(Xm)]))
    ver["impute_max_abs_diff_vs_bundle_1_0"] = float(np.abs(imp - imp0).max())
    ver["impute_identical_to_bundle_1_0"] = bool(np.array_equal(imp, imp0))
    (out / "verification.json").write_text(json.dumps(ver, indent=1))
    log(f"verification: {json.dumps({k: v for k, v in ver.items() if k != 'shared_core_check'})}")


if __name__ == "__main__":
    main()
