#!/usr/bin/env python3
"""Large-gain analysis B: disease classifiers trained on one array and tested on another.

Written on 24 Oct 2026, before any model is fitted, for the large-gain analysis plan of 23 Oct 2026 (Decision rule,
analysis B, Outputs). Data, labels, representations and the classifier are those of analysis A
(large_gain_A_disease_few_labels_2026-10-24.py, imported). Every result is reported, whichever representation it favours.

DATA
  The datasets frozen by analysis A (A/datasets_frozen_2026-10-24.csv, primary and train groups) with A's contrasts,
  profiles, arrays, clocks and representations.

PAIRS (stage `pairs`, written to B/pairs_frozen_2026-10-24.csv and committed before `fit`)
  Two frozen datasets with the same disease label (A's normalization and synonym map; the non-specific labels "tumor" and
  "cancer" never pair) and the same specific tissue label (the tissue labels "Other", "Cells" and "unlabelled" never pair),
  measured on different arrays (A's dataset array: 27K, 450K, EPIC, EPIC v2). Both directions are scored.
  Clocks for train-group datasets that are sources of a primary pair and lack ClockBase values are computed from their
  deposited GEO matrix as in A (downloaded to artifacts/external/<GSE>/, file named in artifacts/external/<GSE>/DEPOSITED.txt:
  the processed beta-value matrix of the supplementary files; stage `downloads` lists them before A's final freeze).

DESIGN
  Train on all contrast profiles of the source dataset: hyperparameters by stratified 5-fold CV within the source
  (random_state 0), A's grid, deviance criterion and solver; refit on all source profiles; standardization with source
  statistics. Score: AUC on all contrast profiles of the target dataset.

REPRESENTATIONS
  embedding     MethylGPT, PRIMARY (A's embedding).
  latent        MethylGPT, secondary (A's latent scores).
  raw_shared    comparator: vocabulary CpGs observed in >= 90% of the contrast profiles of both datasets; remaining missing
                values filled with corpus training means.
  raw_meanfill  comparator: all 49,156 vocabulary CpGs, unobserved values filled with corpus training means (A's raw).
  clocks        comparator: the clocks (GrimAge2, DunedinPACE, PhenoAge, Horvathv1) that enter both datasets' clock sets in A,
                computed on each dataset's own array.
  Comparators are what a user would otherwise do; the latent scores are a MethylGPT output.

PRIMARY METRIC AND DECISION
  Primary set: ordered pairs whose target dataset is outside the training split (A's groups test, val or external), with
  every representation available. M(rep) = mean over these ordered pairs of (target AUC - 0.5). Best comparator C* =
  the comparator with the largest M. Relative gain = (M(embedding) - M(C*)) / M(C*). 95% interval of
  M(embedding) - M(C*): paired bootstrap over unordered dataset pairs (both directions of a pair resampled together), 2,000
  draws, seed 20261024, C* fixed as on the full set. PASS if and only if the relative gain is >= 10% and the lower end of
  the interval is > 0. If there is no qualifying pair, B reports that, with the search (the pair rule above applied to
  A's frozen datasets).

SECONDARY (all reported): all ordered pairs including train-group targets; pairs by array combination (27K with 450K or
EPIC; 450K with EPIC or EPIC v2); latent scores in place of the embedding; MethylGPT minus each comparator; per-pair AUC
of every representation; pairs that lack clocks, scored without the clock comparator.

Stages: pairs, downloads (lists the deposited files needed for clocks), fit (GPU), analyse.
Outputs: artifacts/large_gain_2026-10-24/B/ (README.md, pairs, per-pair AUC, summary, rule outcome).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lgA", HERE / "large_gain_A_disease_few_labels_2026-10-24.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

OUTB = A.OUT / "B"
DATE = A.DATE
NO_PAIR_DISEASE = {"tumor", "cancer"}
NO_PAIR_TISSUE = {"Other", "Cells", "unlabelled"}
REPS_B = ["embedding", "latent", "raw_shared", "raw_meanfill", "clocks"]
MGPT_B, COMP_B = "embedding", ["raw_shared", "raw_meanfill", "clocks"]


def stage_pairs():
    OUTB.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(A.need(A.OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included].copy()
    fr["clocks_used"] = fr.clocks_used.fillna("")
    rows = []
    elig = fr[~fr.disease.isin(NO_PAIR_DISEASE) & ~fr.tissue.isin(NO_PAIR_TISSUE)]
    for (dis, tis), b in elig.groupby(["disease", "tissue"]):
        for (_, d1), (_, d2) in itertools.combinations(b.sort_values("accession").iterrows(), 2):
            if d1["array"] == d2["array"]:
                continue
            shared = sorted(set(d1.clocks_used.split(";")) & set(d2.clocks_used.split(";")) - {""})
            pid = f"{d1.accession}|{d2.accession}"
            for s, t in ((d1, d2), (d2, d1)):
                rows.append({"pair_id": pid, "disease": dis, "tissue": tis, "source": s.accession, "target": t.accession,
                             "source_array": s["array"], "target_array": t["array"], "source_group": s.group, "target_group": t.group,
                             "source_n_case": s.n_case, "source_n_ctrl": s.n_ctrl, "target_n_case": t.n_case,
                             "target_n_ctrl": t.n_ctrl, "clocks_shared": ";".join(shared),
                             "primary": t.group in A.PRIMARY_GROUPS,
                             "array_combination": "27K with 450K/EPIC" if "27K" in (s["array"], t["array"]) else "450K with EPIC/EPIC v2"})
    pr = pd.DataFrame(rows)
    pr.to_csv(OUTB / f"pairs_frozen_{DATE}.csv", index=False)
    A.log(f"pairs: {pr.pair_id.nunique() if len(pr) else 0} unordered, {len(pr)} ordered, primary ordered {int(pr.primary.sum()) if len(pr) else 0}")


def stage_downloads():
    """Train-group sources of primary pairs whose clocks are incomplete (fewer clocks than the array allows)."""
    pr = pd.read_csv(A.need(OUTB / f"pairs_frozen_{DATE}.csv"))
    fr = pd.read_csv(A.need(A.OUTA / f"datasets_frozen_{DATE}.csv")).set_index("accession")
    need = sorted({s for s, g in zip(pr[pr.primary].source, pr[pr.primary].source_group) if g == "train"})
    rows = []
    for acc in need:
        used = str(fr.loc[acc, "clocks_used"]) if isinstance(fr.loc[acc, "clocks_used"], str) else ""
        rows.append({"accession": acc, "array": fr.loc[acc, "array"], "clocks_used": used,
                     "deposited": [str(f) for f in (A.deposited_file(acc) or [])]})
    print(json.dumps(rows, indent=1))


def pair_features(src, tgt, rep, mu):
    if rep in ("embedding", "latent", "raw_meanfill"):
        r = "raw" if rep == "raw_meanfill" else rep
        Zs, ys = A.features(src, r, mu)
        Zt, yt = A.features(tgt, r, mu)
        return Zs, ys, Zt, yt
    if rep == "raw_shared":
        out = []
        obs = []
        for acc in (src, tgt):
            pt, keep = A.load_contrast(acc)
            X = np.load(A.need(A.WORK / f"X_{acc}.npy"), mmap_mode="r")[keep].astype(np.float32)
            obs.append(np.isfinite(X).mean(0) >= A.AVAIL)
            out.append((X, (pt.role == "case").to_numpy(np.float32)))
        cols = obs[0] & obs[1]
        (Xs, ys), (Xt, yt) = out
        f = lambda X: np.where(np.isfinite(X[:, cols]), X[:, cols], mu[None, cols]).astype(np.float32)
        return f(Xs), ys, f(Xt), yt
    if rep == "clocks":
        ps, _ = A.load_contrast(src)
        pt, _ = A.load_contrast(tgt)
        cols = [f"clock_{k}" for k in A.CLOCKS if ps[f"clock_{k}"].notna().mean() >= A.AVAIL and pt[f"clock_{k}"].notna().mean() >= A.AVAIL]
        if not cols:
            return None, None, None, None
        return (ps[cols].to_numpy(np.float32), (ps.role == "case").to_numpy(np.float32),
                pt[cols].to_numpy(np.float32), (pt.role == "case").to_numpy(np.float32))


def stage_fit(device):
    import torch
    dev = torch.device(device)
    pr = pd.read_csv(A.need(OUTB / f"pairs_frozen_{DATE}.csv"))
    mu = np.load(A.need(A.CORE))["mu"].astype(np.float32)
    outdir = A.WORK / "fit_B"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for _, r in pr.sort_values(["primary", "pair_id"], ascending=[False, True]).iterrows():
        f = outdir / f"{r.source}__{r.target}.csv"
        if f.exists():
            continue
        rows = []
        for rep in REPS_B:
            Zs, ys, Zt, yt = pair_features(r.source, r.target, rep, mu)
            if Zs is None:
                continue
            res = A.fit_select_score(Zs, ys, [np.arange(len(ys))], dev, fold_seeds=[0], score_sets=[np.arange(len(yt))], Zscore=Zt)[0]
            rows.append({"pair_id": r.pair_id, "source": r.source, "target": r.target, "representation": rep,
                         "auc": A.auc(res["score"], yt), "alpha": res["alpha"], "lambda_index": res["lambda_index"],
                         "nonzero": res["nonzero"], "features": int(Zs.shape[1])})
        pd.DataFrame(rows).to_csv(f, index=False)
        A.log(f"{r.source} -> {r.target}: fitted ({time.time() - t0:.0f} s)")


def boot_pairs(wide, reps, draws=A.BOOT, seed=A.SEED):
    rng = np.random.default_rng(seed)
    pids = wide.pair_id.unique()
    idx = {p: np.flatnonzero(wide.pair_id.to_numpy() == p) for p in pids}
    out = {r: [] for r in reps}
    V = {r: wide[r].to_numpy(float) for r in reps}
    for _ in range(draws):
        take = np.concatenate([idx[pids[i]] for i in rng.integers(0, len(pids), len(pids))])
        for r in reps:
            out[r].append(V[r][take].mean())
    return {r: np.asarray(v) for r, v in out.items()}


def compare_pairs(wide, mg, comps, label):
    wide = wide.dropna(subset=[mg] + comps)
    if not len(wide):
        return None
    reps = [r for r in REPS_B if r in wide and wide[r].notna().all()]
    M = {r: float(wide[r].mean()) for r in reps}
    B = boot_pairs(wide, reps)
    best = max(comps, key=lambda c: M[c])
    d = B[mg] - B[best]
    res = {"set": label, "ordered_pairs": int(len(wide)), "unordered_pairs": int(wide.pair_id.nunique()), "M": M,
           "best_comparator": best, "best_comparator_M": M[best], "methylgpt_M": M[mg], "difference": M[mg] - M[best],
           "difference_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
           "relative_gain": (M[mg] - M[best]) / M[best] if M[best] > 0 else float("nan"), "vs_each": {}}
    for c in [r for r in reps if r != mg]:
        dd = B[mg] - B[c]
        res["vs_each"][c] = {"difference": M[mg] - M[c], "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                             "relative_gain": (M[mg] - M[c]) / M[c] if M[c] > 0 else float("nan"),
                             "pairs_methylgpt_higher": int((wide[mg] > wide[c]).sum())}
    res["pass"] = bool(np.isfinite(res["relative_gain"]) and res["relative_gain"] >= 0.10 and res["difference_ci95"][0] > 0)
    return res


def stage_analyse():
    pr = pd.read_csv(A.need(OUTB / f"pairs_frozen_{DATE}.csv"))
    files = sorted((A.WORK / "fit_B").glob("*.csv"))
    res = {"primary": None, "secondary": {}}
    if not len(pr):
        pd.DataFrame([{"analysis": "B", "outcome": "no qualifying pairs"}]).to_csv(OUTB / f"rule_outcome_{DATE}.csv", index=False)
        A.write_json(OUTB / f"results_{DATE}.json", {"primary": "no qualifying pairs"})
        return
    long = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    wide = long.pivot_table(index=["pair_id", "source", "target"], columns="representation", values="auc").reset_index()
    wide = wide.merge(pr, on=["pair_id", "source", "target"])
    per_pair = wide.copy()
    per_pair.to_csv(OUTB / f"per_pair_auc_{DATE}.csv", index=False)
    for r in REPS_B:
        if r in wide:
            wide[r] = wide[r] - 0.5
    sets = {"primary": wide[wide.primary], "all ordered pairs": wide}
    for comb in sorted(wide.array_combination.unique()):
        sets[f"all|{comb}"] = wide[wide.array_combination == comb]
        sets[f"primary|{comb}"] = wide[wide.primary & (wide.array_combination == comb)]
    rows = []
    for name, w in sets.items():
        for mg in ("embedding", "latent"):
            for comps, tag in ((COMP_B, "all comparators"), (["raw_shared", "raw_meanfill"], "without clocks")):
                r = compare_pairs(w, mg, comps, f"{name}|{mg}|{tag}")
                if r is None:
                    continue
                if name == "primary" and mg == MGPT_B and tag == "all comparators":
                    res["primary"] = r
                else:
                    res["secondary"][f"{name}|{mg}|{tag}"] = r
                rows.append({"set": name, "methylgpt_representation": mg, "comparators": tag, "ordered_pairs": r["ordered_pairs"],
                             "unordered_pairs": r["unordered_pairs"], "methylgpt_M": r["methylgpt_M"],
                             **{f"M_{k}": v for k, v in r["M"].items()}, "best_comparator": r["best_comparator"],
                             "best_comparator_M": r["best_comparator_M"], "difference": r["difference"],
                             "difference_ci95_low": r["difference_ci95"][0], "difference_ci95_high": r["difference_ci95"][1],
                             "relative_gain": r["relative_gain"], "meets_rule": r["pass"],
                             "decision_set": name == "primary" and mg == MGPT_B and tag == "all comparators"})
    pd.DataFrame(rows).to_csv(OUTB / f"summary_{DATE}.csv", index=False)
    p = res["primary"]
    if p is None:
        out = {"analysis": "B", "outcome": "no qualifying primary pairs with every representation"}
    else:
        out = {"analysis": "B: disease classifier trained on one array, scored on another (targets outside the training split)",
               "primary_metric": "mean over ordered pairs of (target AUC - 0.5)", "ordered_pairs": p["ordered_pairs"],
               "unordered_pairs": p["unordered_pairs"], "methylgpt_representation": "embedding", "methylgpt_value": p["methylgpt_M"],
               "best_comparator": p["best_comparator"], "best_comparator_value": p["best_comparator_M"],
               "relative_gain": p["relative_gain"], "difference": p["difference"], "difference_ci95_low": p["difference_ci95"][0],
               "difference_ci95_high": p["difference_ci95"][1],
               "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
               "outcome": "PASS" if p["pass"] else "FAIL"}
    pd.DataFrame([out]).to_csv(OUTB / f"rule_outcome_{DATE}.csv", index=False)
    A.write_json(OUTB / f"results_{DATE}.json", res)
    A.log(f"PRIMARY B: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pairs", "downloads", "fit", "analyse"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    {"pairs": stage_pairs, "downloads": stage_downloads, "analyse": stage_analyse}.get(a.stage, lambda: stage_fit(a.device))()


if __name__ == "__main__":
    main()
