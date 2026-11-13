#!/usr/bin/env python3
"""Large-gain analysis A, variant ADDED AFTER THE PRIMARY RESULT: disease classification from few labelled samples
with 27K-array CpGs only.

Written on 24 Oct 2026 after the primary result of large_gain_A_disease_few_labels_2026-10-24.py failed the rule
(MethylGPT embedding 0.287 against raw CpGs 0.312, mean AUC - 0.5 at n = 50 over 21 primary datasets). The agent rules
allow at most one further variant per analysis, restricted to the data-limited settings the plan names; this is that
variant for A (missing CpGs). It is committed before it is run and is reported alongside the primary result, labelled
as added after the primary result. Every result is reported, whichever representation it favours.

DATA
  The primary datasets frozen by analysis A (groups test, val, external; A/datasets_frozen_2026-10-24.csv) that were
  measured on the 450K, EPIC or EPIC v2 array (27K datasets are already 27K) and that have a clock set in A (so every
  comparator is available). Same contrasts, profiles, training draws (A's draws_for), sizes (20, 50, 100 with A's
  skip rule), classifier, grid and selection (A's fit_select_score).
  Reduction: every profile keeps only the 22,231 vocabulary CpGs of the 27K array (platform_vocab_masks_2026-10-01.npz,
  hm27: the 27K scenario of the paper); all other CpGs are unobserved.

REPRESENTATIONS
  embedding   MethylGPT, PRIMARY of the variant: A's embedding (age-stage encoder) computed from the reduced profiles.
  latent      MethylGPT, secondary: the first 1,536 dimensions of that embedding (latent scores).
  raw         comparator: the reduced profiles, unobserved CpGs filled with corpus training means (as A).
  clocks      comparator: the clocks of the dataset's clock set in A (GrimAge2, DunedinPACE, PhenoAge, Horvath as
              available), computed with biolearn 0.7.1 (A's conventions: model default imputation; GrimAge2 with the
              Horvath age and all profiles male) from the dataset's deposited full-array matrix (A's clock input:
              WORK/clockcpg_<GSE>.parquet, or the deposited matrix read with A's load_deposited when A used ClockBase
              values) restricted to clock CpGs in the 27K vocabulary set; every other clock CpG is set, for all
              profiles, to its mean over all profiles of that matrix (as the Generation Scotland 27K scenario filled
              missing clock CpGs with full-data means). The 10% missing-CpG rule is not applied (the reduction is the
              point of the variant). Inputs clipped to [0, 1] as in A.

METRIC AND DECISION (as A's primary)
  M(rep) = mean over the variant datasets evaluable at n = 50 of (AUC - 0.5); best comparator = raw or clocks with the
  larger M; relative gain (M(embedding) - M(best)) / M(best); 95% interval of the difference by paired bootstrap over
  datasets (2,000 draws, seed 20261024). Meets the rule if and only if the gain is >= 10% and the lower end of the
  interval is > 0. Secondary: n = 20 and 100; latent scores; MethylGPT against each comparator; per-dataset AUC.

Stages: embed (GPU), clocks (/home/kying/anaconda3/bin/python), fit (GPU, --shard k/K), analyse.
Outputs: artifacts/large_gain_2026-10-24/A/variant27k_*.csv / .json (aggregates); working files in
artifacts/large_gain_2026-10-24/work_AB/variant27k/.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lgA", HERE / "large_gain_A_disease_few_labels_2026-10-24.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

DATE = A.DATE
VW = A.WORK / "variant27k"
TAG = "variant27k"
REPS = ["embedding", "latent", "raw", "clocks"]


def datasets():
    fr = pd.read_csv(A.need(A.OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included & fr.group.isin(A.PRIMARY_GROUPS) & fr.array.isin(["450K", "EPIC", "EPIC v2"]) & fr.clocks_used.notna()]
    return fr.sort_values("accession")


def hm27():
    return np.load(A.need(A.MASKS))["hm27"]


def reduced_X(acc):
    X = np.load(A.need(A.WORK / f"X_{acc}.npy")).astype(np.float32)
    X[:, ~hm27()] = np.nan
    return X


def stage_embed(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    from methylgpt_model import load_checkpoint
    fp = A._module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    VW.mkdir(parents=True, exist_ok=True)
    ag = torch.load(A.need(A.AGE_STAGE), map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(HERE / ag["ckpt"], None, dev)
    model.load_state_dict(ag["model"])
    model.eval()
    for acc in datasets().accession:
        f = VW / f"emb_{acc}.npy"
        if not f.exists():
            np.save(f, fp.embed(model, core, reduced_X(acc), dev, batch=32))
            A.log(f"{acc}: reduced-profile embeddings")


def stage_clocks():
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery
    gal = ModelGallery()
    VW.mkdir(parents=True, exist_ok=True)
    vocab = np.load(A.need(A.MASKS))["vocab"].astype(str)
    keep27 = set(vocab[hm27()])
    for _, d in datasets().iterrows():
        acc = d.accession
        f = VW / f"clocks_{acc}.csv"
        if f.exists():
            continue
        fc = A.WORK / f"clockcpg_{acc}.parquet"
        if fc.exists():
            dnam = pd.read_parquet(fc)
        else:
            dnam, _ = A.load_deposited(acc, A.clock_cpg_set())
        dnam = dnam.astype(np.float64).clip(0, 1)
        means = dnam.mean(axis=1)
        off = ~dnam.index.isin(list(keep27))
        dnam.loc[off, :] = np.repeat(means[off].to_numpy()[:, None], dnam.shape[1], axis=1)
        meta0 = pd.DataFrame({"age": np.nan, "sex": 2}, index=dnam.columns)
        out = pd.DataFrame(index=dnam.columns)
        horv = gal.get("Horvathv1").predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
        for clock in str(d.clocks_used).split(";"):
            name = A.BIOLEARN_NAME[clock]
            if clock == "Horvathv1":
                out[clock] = horv
            elif clock == "GrimAge2":
                meta = pd.DataFrame({"age": horv.to_numpy(float), "sex": 2}, index=dnam.columns)
                out[clock] = gal.get(name).predict(GeoData(meta, dnam.copy()))["DNAmGrimAge"].reindex(dnam.columns)
            else:
                out[clock] = gal.get(name).predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
        out.to_csv(f)
        A.log(f"{acc}: 27K-CpG clocks ({dnam.shape[1]} profiles, {int((~off).sum())} clock CpGs kept)")


def features(acc, rep, mu):
    pt, keep = A.load_contrast(acc)
    y = (pt.role == "case").to_numpy(np.float32)
    if rep in ("embedding", "latent"):
        E = np.load(A.need(VW / f"emb_{acc}.npy"))[keep]
        return (E if rep == "embedding" else E[:, :1536]), y
    if rep == "raw":
        X = reduced_X(acc)[keep]
        return np.where(np.isfinite(X), X, mu[None, :]).astype(np.float32), y
    c = pd.read_csv(A.need(VW / f"clocks_{acc}.csv"), index_col=0).reindex(pt.id.astype(str).to_numpy())
    return c.to_numpy(np.float32), y


def stage_fit(device, shard=None):
    import torch
    dev = torch.device(device)
    mu = np.load(A.need(A.CORE))["mu"].astype(np.float32)
    out = VW / "fit"
    out.mkdir(parents=True, exist_ok=True)
    ds = datasets().reset_index(drop=True)
    if shard:
        k, K = map(int, shard.split("/"))
        ds = ds[ds.index % K == k]
    t0 = time.time()
    for _, d in ds.iterrows():
        acc = d.accession
        f = out / f"{acc}.csv"
        if f.exists():
            continue
        sizes = [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
        rows = []
        for rep in REPS:
            Z, y = features(acc, rep, mu)
            for n in sizes:
                trs = A.draws_for(int(d.accession_number), n, y)
                res = A.fit_select_score(Z, y, trs, dev, fold_seeds=list(range(A.N_DRAWS)))
                for r_i, (tr, rr) in enumerate(zip(trs, res)):
                    mask = np.ones(len(y), bool)
                    mask[tr] = False
                    rows.append({"accession": acc, "representation": rep, "n_train": n, "draw": r_i,
                                 "auc": A.auc(rr["score"], y[mask]), "alpha": rr["alpha"], "lambda_index": rr["lambda_index"],
                                 "nonzero": rr["nonzero"], "features": int(Z.shape[1])})
        pd.DataFrame(rows).to_csv(f, index=False)
        A.log(f"{acc}: variant fitted ({time.time() - t0:.0f} s)")


def stage_analyse():
    fr = datasets().set_index("accession")
    per_draw = pd.concat([pd.read_csv(f) for f in sorted((VW / "fit").glob("*.csv"))], ignore_index=True)
    per_draw.to_csv(A.OUTA / f"{TAG}_per_draw_auc_{DATE}.csv.gz", index=False)
    g = per_draw.groupby(["accession", "representation", "n_train"])
    pdx = g.auc.agg(["mean", "std", "count"]).rename(columns={"mean": "auc_mean", "std": "auc_sd", "count": "draws"})
    pdx["null_model_draws"] = g.nonzero.apply(lambda s: int((s == 0).sum()))
    pdx = pdx.reset_index().join(fr[["group", "disease", "tissue", "array", "n_case", "n_ctrl", "clocks_used"]], on="accession")
    pdx.to_csv(A.OUTA / f"{TAG}_per_dataset_auc_{DATE}.csv", index=False)
    rows, res = [], {"primary": None, "secondary": {}}
    for n in A.SIZES:
        w = pdx[pdx.n_train == n].pivot(index="accession", columns="representation", values="auc_mean") - 0.5
        for mg in ("embedding", "latent"):
            r = A.compare(w, REPS, mg, A.COMPARATORS, f"{TAG}|n={n}|{mg}")
            if r is None:
                continue
            decision = n == A.PRIMARY_SIZE and mg == "embedding"
            if decision:
                res["primary"] = r
            else:
                res["secondary"][f"n={n}|{mg}"] = r
            rows.append({"set": TAG, "n_train": n, "methylgpt_representation": mg, "datasets": r["datasets"],
                         "M_embedding": r["M"].get("embedding"), "M_latent": r["M"].get("latent"), "M_raw": r["M"].get("raw"),
                         "M_clocks": r["M"].get("clocks"), "best_comparator": r["best_comparator"],
                         "best_comparator_M": r["best_comparator_M"], "difference": r["difference"],
                         "difference_ci95_low": r["difference_ci95"][0], "difference_ci95_high": r["difference_ci95"][1],
                         "relative_gain": r["relative_gain"], "meets_rule": r["pass"], "variant_decision_row": decision})
    pd.DataFrame(rows).to_csv(A.OUTA / f"{TAG}_summary_{DATE}.csv", index=False)
    p = res["primary"]
    pd.DataFrame([{"analysis": "A variant (added after the primary result): 27K-array CpGs only, n = 50",
                   "datasets": p["datasets"], "methylgpt_value": p["methylgpt_M"], "best_comparator": p["best_comparator"],
                   "best_comparator_value": p["best_comparator_M"], "relative_gain": p["relative_gain"],
                   "difference": p["difference"], "difference_ci95_low": p["difference_ci95"][0],
                   "difference_ci95_high": p["difference_ci95"][1],
                   "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
                   "outcome": "meets rule" if p["pass"] else "does not meet rule"}]).to_csv(A.OUTA / f"{TAG}_rule_outcome_{DATE}.csv", index=False)
    A.write_json(A.OUTA / f"{TAG}_results_{DATE}.json", res)
    A.log(f"VARIANT: {p['methylgpt_M']:.4f} vs {p['best_comparator']} {p['best_comparator_M']:.4f}, gain {p['relative_gain']:.3f}, "
          f"CI {p['difference_ci95']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["embed", "clocks", "fit", "analyse"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard")
    a = ap.parse_args()
    if a.stage == "embed":
        stage_embed(a.device)
    elif a.stage == "clocks":
        stage_clocks()
    elif a.stage == "fit":
        stage_fit(a.device, a.shard)
    else:
        stage_analyse()


if __name__ == "__main__":
    main()
