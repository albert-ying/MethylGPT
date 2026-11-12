#!/usr/bin/env python3
"""Raw-CpG disease models with factor-model imputation under the missing-CpG scenarios (GS development, locked folds).

In the scenario analysis the raw-CpG Cox models fill missing CpGs with
fitting-fold means, whereas MethylGPT computes its features from the corpus-fitted factor model. Here the same
fitted raw-CpG fold models (original protocol and dense fit; unchanged) are applied to held-out participants
whose missing CpGs are filled with the factor model's reconstruction from the remaining CpGs (posterior factor
scores -> mu + V s, clipped to [0, 1]), i.e. the imputation of the released package. Masks are those of
scenario_gs_combined_2026-10-01.py. The complete-profile scenario also fills the natively missing CpGs.

Comparison: MethylGPT combined model (stored scenario risks) minus each raw-CpG model with factor-model
imputation; pooled within-fold C-index; paired bootstrap (300 draws within outer folds, seed 20261009), Benjamini-
Hochberg over all scenario x endpoint x comparator contrasts of this analysis; mean over endpoints resampled per
endpoint. Development participants only (9,866); the validation and test subsets are not read.
Outputs: artifacts/gs_cox/scenario_raw_factor_imputed_2026-10-09<tag>.{json,csv} (reported: --draws 2000 --tag _2000draws); private risks in
artifacts/gs_cox/scenario_raw_factor_imputed_2026-10-09/private_risks.parquet.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, cuda_preinit, log, require, write_json
from methylgpt2_model import load_checkpoint

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stk", HERE / "gs_stacked_2026-10-04.py")
stk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stk)

SPLIT, CACHE = stk.SPLIT, stk.CACHE
G = ARTIFACTS / "gs_cox"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_DIRS = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]
SCEN = ["full", "epic_v2", "hm450", "random_30", "random_50", "random_70", "hm27", "random_90"]


def locked_risk(model, feats, cov):
    z = (np.where(np.isfinite(feats), feats, model["mean"]) - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return c @ model["coefficient"][:2] + z @ model["coefficient"][2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--tag", default="", help="suffix for the aggregate output files, e.g. _2000draws")
    a = ap.parse_args()
    cuda_preinit(); torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time(); dev = a.device
    out = G / "scenario_raw_factor_imputed_2026-10-09"; out.mkdir(parents=True, exist_ok=True)
    mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    pos = {s: i for i, s in enumerate(ids)}
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    ag = torch.load(ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt", map_location=dev, weights_only=False)
    _, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    stored = pd.read_parquet(require(G / "scenario_combined_2026-10-01/private_scenario_risks.parquet"))
    stored["dnam_id"] = stored.dnam_id.astype(str)
    endpoints = sorted(stored.endpoint.unique())
    fdir = G / "factor_residual_2026-10-01/models"

    frames, observed = [], {}
    for scen in SCEN:
        Xs = X.copy()
        if scen.startswith("random_"):
            p = int(scen.split("_")[1]) / 100
            Xs[np.random.default_rng(20261001 + int(p * 100)).random(Xs.shape) < p] = np.nan
        elif scen != "full":
            Xs[:, ~plat[scen]] = np.nan
        observed[scen] = float(np.isfinite(Xs).mean())
        with torch.no_grad():
            for i in range(0, len(Xs), 64):
                x = torch.from_numpy(Xs[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                rec = core.predict(core.posterior(x, obs.float())).clamp(0, 1)
                Xs[i:i + 64] = torch.where(obs, x, rec).float().cpu().numpy()
        assert np.isfinite(Xs).all()
        st = stored[stored.scenario == scen]
        for ep in endpoints:
            pe = st[st.endpoint == ep].reset_index(drop=True)
            idx = np.array([pos[s] for s in pe.dnam_id])
            cov = pe[["age", "sex"]].to_numpy(float)
            r_gpu, r_v12 = np.empty(len(pe)), np.empty(len(pe))
            for f in sorted(pe.outer_fold.unique()):
                held = np.flatnonzero(pe.outer_fold.to_numpy() == f)
                gm = dict(np.load(require(fdir / f"raw_cpg_gpu_{ep}_outer{f}.npz"), allow_pickle=False))
                zf = (Xs[idx[held]] - gm["mean"]) / gm["scale"]
                r_gpu[held] = np.hstack([((cov[held] - gm["cov_mean"]) / gm["cov_scale"]).astype(np.float32), zf.astype(np.float32)]) @ gm["coefficient"]
                rdir = [d for d in RAW_DIRS if (d / f"{ep}_linear_outer{f}_model.npz").exists()][0]
                rm = dict(np.load(rdir / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))
                r_v12[held] = locked_risk(rm, Xs[idx[held]].astype(np.float64), cov[held])
            q = pe[["dnam_id", "endpoint", "outer_fold", "event", "duration", "scenario", "mgpt2_combined", "raw_cpg_gpu", "raw_cpg_v12"]].copy()
            q["raw_cpg_gpu_factor_imputed"] = r_gpu; q["raw_cpg_v12_factor_imputed"] = r_v12
            frames.append(q)
        log(f"{scen} (observed {observed[scen]:.3f}) scored ({time.time() - t0:.0f}s)")
    R = pd.concat(frames, ignore_index=True)
    R.to_parquet(out / "private_risks.parquet", index=False)

    models = ["mgpt2_combined", "raw_cpg_gpu", "raw_cpg_v12", "raw_cpg_gpu_factor_imputed", "raw_cpg_v12_factor_imputed"]
    rng = np.random.default_rng(20261009)
    rows, boot, point = [], {}, {}
    for scen in SCEN:
        for ep in endpoints:
            g = R[(R.scenario == scen) & (R.endpoint == ep)].reset_index(drop=True)
            ev, du, of = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float), g.outer_fold.to_numpy()
            tot = np.zeros((len(models), a.draws, 3)); tot0 = np.zeros((len(models), 3))
            for f in sorted(np.unique(of)):
                s = np.flatnonzero(of == f)
                pc = stk.PairCounter(ev[s], du[s], dev)
                ii = rng.integers(0, len(s), size=(a.draws, len(s)))
                W = np.zeros((a.draws, len(s)), np.float32); np.add.at(W, (np.repeat(np.arange(a.draws), len(s)), ii.ravel()), 1)
                M = np.stack([g[m].to_numpy(float)[s] for m in models])
                tot += pc.counts(M, W); tot0 += pc.counts(M)
            boot[(scen, ep)] = {m: stk.cidx(tot[j]) for j, m in enumerate(models)}
            point[(scen, ep)] = {m: float(stk.cidx(tot0[j])) for j, m in enumerate(models)}
            for c in ("raw_cpg_gpu_factor_imputed", "raw_cpg_v12_factor_imputed"):
                d = boot[(scen, ep)]["mgpt2_combined"] - boot[(scen, ep)][c]
                rows.append({"scenario": scen, "endpoint": ep, "comparator": c, "c_mgpt2_combined": point[(scen, ep)]["mgpt2_combined"],
                             "c_comparator": point[(scen, ep)][c], "difference": point[(scen, ep)]["mgpt2_combined"] - point[(scen, ep)][c],
                             "ci95_low": float(np.percentile(d, 2.5)), "ci95_high": float(np.percentile(d, 97.5)),
                             "p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))})
        log(f"bootstrap {scen} done")
    df = pd.DataFrame(rows); df["q_bh"] = stk.bh(df.p.to_numpy())
    means = []
    for scen in SCEN:
        row = {"scenario": scen, "observed_fraction": observed[scen], **{m: float(np.mean([point[(scen, e)][m] for e in endpoints])) for m in models}}
        for c in ("raw_cpg_gpu_factor_imputed", "raw_cpg_v12_factor_imputed", "raw_cpg_gpu", "raw_cpg_v12"):
            d = np.mean([boot[(scen, e)]["mgpt2_combined"] - boot[(scen, e)][c] for e in endpoints], 0)
            row[f"combined_minus_{c}"] = row["mgpt2_combined"] - row[c]
            row[f"combined_minus_{c}_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        g_ = df[df.scenario == scen]
        for c in ("raw_cpg_gpu_factor_imputed", "raw_cpg_v12_factor_imputed"):
            h = g_[g_.comparator == c]
            row[f"endpoints_better_worse_vs_{c}"] = [int(((h.q_bh < 0.05) & (h.difference > 0)).sum()), int(((h.q_bh < 0.05) & (h.difference < 0)).sum())]
        means.append(row)
        log(f"{scen}: combined {row['mgpt2_combined']:.4f}; dense raw mean-filled {row['raw_cpg_gpu']:.4f}, factor-imputed {row['raw_cpg_gpu_factor_imputed']:.4f} "
            f"(diff {row['combined_minus_raw_cpg_gpu_factor_imputed']:+.4f} {np.round(row['combined_minus_raw_cpg_gpu_factor_imputed_ci95'], 4).tolist()}); "
            f"original raw mean-filled {row['raw_cpg_v12']:.4f}, factor-imputed {row['raw_cpg_v12_factor_imputed']:.4f}")
    df.to_csv(G / f"scenario_raw_factor_imputed_2026-10-09{a.tag}.csv", index=False)
    write_json(G / f"scenario_raw_factor_imputed_2026-10-09{a.tag}.json", {"draws": a.draws, "mean_over_endpoints": means, "elapsed_s": time.time() - t0})


if __name__ == "__main__":
    main()
