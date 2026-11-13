#!/usr/bin/env python3
"""Fig. 4e with the AltumAge-style MLP trained on the 41,845 corpus profiles.

Written on 11 Nov 2026 before running. Every result is reported. No Generation Scotland file is read; nothing is trained.

MODEL  The MLP corpus labels of missing_cpgs_comparators_2026-10-18.py: the AltumAge network, 49,156 inputs, trained on
  the 41,845 corpus training-split profiles with clean ages. Three seeds; the age is the mean of the seeds.
PREDICTIONS
  - AltumAge test set (4,626): the saved predictions "MLP corpus labels|all" of
    artifacts/eval/missing_cpgs_comparators_2026-10-18.predictions.npz.
  - AltumAge validation set (1,366): not saved, so computed here from the saved weights
    (artifacts/runs/mlp_altumage_style_2026-10-18/weights.pt, corpus_labels) with the stored standardiser. These are
    inputs filled with training means and standardised, float32, as for the test predictions. No retraining. The
    validation set chose this MLP's settings, so the column validation_used_for_selection is True on those rows.
  - MethylGPT (mean of three runs), for the paired differences: test, age_ensemble_2026-10-04.predictions.npz;
    validation, the bundle predictions of export_figure_tables_2026-11-10 (work file extract.npz).
TABLE  Exactly the format of fig4e_medae_by_dataset_2026-11-10.csv: columns set, method, validation_used_for_selection,
  dataset, n_scored, medae, n_datasets. method = "MLP (corpus labels)". Dataset = corpus series. Datasets with at least
  10 scored profiles, plus a row 'all'.
SUMMARY  For each set:
  - overall MedAE with a 95% interval resampling datasets (cluster_ci of evaluate_age_revision_2026-10-01.py, 2,000
    draws, seed 20261001: the interval of the MethylGPT bar of Fig. 4e);
  - the paired difference MLP minus MethylGPT with its 95% interval (paired_ci, 2,000 draws, seed 20261002);
  - the same for the 5,461-profile MLP of Fig. 4e, as a reference.
CHECKS (abort on failure)
  1. Test predictions recomputed from the saved weights equal the saved ones, per seed, within 1e-3 years.
  2. The validation MedAE of each seed with half-precision inputs, as in training, equals its stored best_valid_medae
     within 1e-3.
  3. The test MedAE equals the stored 1.7607 (missing_cpgs_comparators table, 4 decimals) within 5e-5.
  4. The recomputed MethylGPT test interval equals age_ensemble_2026-10-04.json medae_ci95_dataset_bootstrap within 1e-12.
OUTPUTS  artifacts/figure_tables_2026-11-10/fig4e_corpus_mlp_by_dataset_2026-11-10.csv and
  fig4e_corpus_mlp_summary_2026-11-10.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt_lib import ARTIFACTS, CORPUS_DIR, Finv, log, require, write_json  # noqa: E402

DATE = "2026-11-10"
EVAL = ARTIFACTS / "eval"
FT = ARTIFACTS / f"figure_tables_{DATE}"
CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    log(f"check {name}: {'pass' if ok else 'FAIL'} {detail}")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    import torch
    from methylgpt_lib import cuda_preinit, load_altumage_split
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = torch.device("cuda:0")
    mc = _load("missing_cpgs_comparators", "missing_cpgs_comparators_2026-10-18.py")
    ar = mc.ar
    zm = np.load(require(EVAL / "missing_cpgs_comparators_2026-10-18.predictions.npz"), allow_pickle=True)
    ze = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    zx = np.load(require(ARTIFACTS / f"figure_tables_{DATE}_work/extract.npz"), allow_pickle=True)
    ids_t, Xt, yt = load_altumage_split("test")
    ids_v, Xv, yv = load_altumage_split("valid")
    assert (zm["altumage_ids"].astype(str) == ids_t).all() and (ze["altumage_ids"].astype(str) == ids_t).all()
    assert (zx["e_valid_ids"].astype(str) == ids_v).all() and np.array_equal(zx["e_valid_age"], yv)
    W = torch.load(require(ARTIFACTS / "runs/mlp_altumage_style_2026-10-18/weights.pt"), map_location=d, weights_only=False)["corpus_labels"]
    std = mc.Standardiser(W["mu"], W["sd"], W["used"], d)
    tm, ts = float(W["target_center"]), float(W["target_scale"])
    tj = json.loads(require(EVAL / "missing_cpgs_comparators_2026-10-18_training.json").read_text())["mlp_corpus_labels"]["seeds"]
    best = {int(s["seed"]): float(s["best_valid_medae"]) for s in tj}
    pv = []
    for k in (1, 2, 3):
        net = mc.make_mlp(Xt.shape[1]).to(d)
        net.load_state_dict(W["seeds"][k])
        net.eval()
        a = Finv(mc.predict_mlp(net, std(Xt)) * ts + tm)
        dk = float(np.abs(a - zm[f"MLP corpus labels seed {k}|all"]).max())
        chk(f"1 seed {k} test predictions equal the saved ones", dk <= 1e-3, f"max abs diff {dk:.3g} years")
        pv.append(Finv(mc.predict_mlp(net, std(Xv)) * ts + tm))
        vh = float(np.median(np.abs(Finv(mc.predict_mlp(net, std(Xv).half()) * ts + tm) - yv)))
        chk(f"2 seed {k} validation MedAE equals best_valid_medae", abs(vh - best[k]) <= 1e-3, f"{vh:.6f} vs {best[k]:.6f}")
    P = {"AltumAge test": (ids_t, yt, zm["MLP corpus labels|all"], ze["ensemble|all"], zm["MLP|all"]),
         "AltumAge validation": (ids_v, yv, np.mean(pv, 0), zx["e_valid|MethylGPT"], zx["e_valid|MLP"])}
    tb = pd.read_csv(require(EVAL / "missing_cpgs_comparators_2026-10-18_table.csv"))
    ref = float(tb[(tb.part == "overall") & (tb.scenario == "all") & (tb.model_or_contrast == "MLP corpus labels")].estimate.iloc[0])
    mt = float(np.median(np.abs(P["AltumAge test"][2] - yt)))
    chk("3 test MedAE equals the stored value", abs(mt - ref) <= 5e-5, f"{mt:.6f} vs {ref:.4f}")
    ens = json.loads(require(EVAL / "age_ensemble_2026-10-04.json").read_text())["a_altumage_test"]["ensemble"]["medae_ci95_dataset_bootstrap"]
    pid = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"), columns=["id", "group"]).set_index("id")
    rows, summ = [], {}
    for split, (ids, y, p, pm, p5) in P.items():
        grp = pid.reindex(ids).group.astype(str).to_numpy()
        e = np.abs(p - y)
        base = {"set": split, "method": "MLP (corpus labels)", "validation_used_for_selection": split == "AltumAge validation"}
        rows.append({**base, "dataset": "all", "n_scored": int(len(y)), "medae": float(np.median(e)), "n_datasets": int(len(set(grp)))})
        per = pd.DataFrame({"g": grp, "e": e}).groupby("g").e.agg(["size", "median"])
        for g, r in per[per["size"] >= 10].iterrows():
            rows.append({**base, "dataset": g, "n_scored": int(r["size"]), "medae": float(r["median"]), "n_datasets": 1})
        ci_m = ar.cluster_ci(pm, y, grp)
        if split == "AltumAge test":
            chk("4 MethylGPT test interval equals Fig. 4e", max(abs(ci_m[0] - ens[0]), abs(ci_m[1] - ens[1])) <= 1e-12, f"{ci_m} vs {ens}")
        summ[split] = {"n_profiles": int(len(y)), "n_datasets": int(len(set(grp))), "datasets_with_at_least_10": int((per["size"] >= 10).sum()),
                       "MLP (corpus labels)": {"medae": float(np.median(e)), "ci95_resampling_datasets": ar.cluster_ci(p, y, grp)},
                       "MethylGPT": {"medae": float(np.median(np.abs(pm - y))), "ci95_resampling_datasets": ci_m},
                       "MLP (5,461 AltumAge profiles)": {"medae": float(np.median(np.abs(p5 - y))), "ci95_resampling_datasets": ar.cluster_ci(p5, y, grp)},
                       "MLP (corpus labels) minus MethylGPT": ar.paired_ci(p, pm, y, grp),
                       "MLP (5,461 AltumAge profiles) minus MethylGPT": ar.paired_ci(p5, pm, y, grp),
                       "datasets where MethylGPT has the lower MedAE (>= 10 scored)": int(
                           (pd.DataFrame({"g": grp, "a": np.abs(pm - y), "b": e}).groupby("g").agg(n=("a", "size"), a=("a", "median"), b=("b", "median"))
                            .query("n >= 10").pipe(lambda x: (x.a < x.b).sum())))}
        log(f"{split}: {summ[split]['MLP (corpus labels)']} vs MethylGPT {summ[split]['MethylGPT']}; difference {summ[split]['MLP (corpus labels) minus MethylGPT']}")
    if not all(c["passed"] for c in CHECKS):
        sys.exit("checks failed")
    out = pd.DataFrame(rows)
    out.to_csv(FT / f"fig4e_corpus_mlp_by_dataset_{DATE}.csv", index=False)
    write_json(FT / f"fig4e_corpus_mlp_summary_{DATE}.json", {"summary": summ, "checks": CHECKS, "rows": int(len(out)), "definitions": __doc__})
    log(f"written {len(out)} rows")


if __name__ == "__main__":
    main()
