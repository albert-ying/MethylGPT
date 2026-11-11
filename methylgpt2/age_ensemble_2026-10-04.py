#!/usr/bin/env python3
"""Three-run MethylGPT age ensemble, evaluated everywhere age is reported.

Ensemble age = mean of the age predictions of three multitask runs (same recipe, different seeds):
  run 1 (released model)  artifacts/runs/multitask_bio_2026-10-01/final.pt
  seed 2                  artifacts/runs/multitask_bio_seed20261002/final.pt
  seed 3                  artifacts/runs/multitask_bio_seed20261003/final.pt
Each analysis reports the ensemble next to run 1 and the elastic net trained on the same labels,
always on identical profiles.

Inputs
  eval/age_revision_2026-10-01.predictions.npz         run 1 and elastic-net predictions (AltumAge test, all
                                                       missing-CpG scenarios; held-out test-split datasets)
  eval/age_seed{2,3}_scenarios_2026-10-04.predictions.npz  seed 2 / seed 3 predictions for every scenario and
                                                       held-out (evaluate_age_revision_2026-10-01.py with the seed
                                                       run as first model); their elastic-net predictions are
                                                       checked against the run-1 file to confirm identical masks
  corpus_2026-09-30/profiles.parquet, profiles_age_clean_2026-10-01.parquet   dataset groups, split, held-out rows
  ClockBase and Zhang 2019 predictions via clock_table() of evaluate_age_revision_2026-10-01.py
  gs_features_2026-10-01/*_mt_age.embedding.npz (run 1) and gs_features_seeds_2026-10-04/*_mtseed{2,3}_age
  external/GSE194154/betas_cov10.parquet + samples.csv (RRBS); GSE54848 450K matrix (reprogramming)

Method
  a  AltumAge test: MedAE, MAE, r; MedAE 95% CI from resampling datasets (cluster_ci); paired MedAE difference
     vs elastic net with dataset-bootstrap CI (paired_ci).
  b  Missing-CpG scenarios: MedAE/MAE/r per scenario.
  c  Held-out (34 datasets; 21 unflagged): MedAE, MAE, paired differences vs elastic net (2,000 dataset draws,
     seed 20261002, logic of age_heldout_paired_2026-10-03.py), datasets where the model has lower MedAE.
  d  Tissues (>= 50 profiles; held-out unflagged and AltumAge test): MedAE difference vs elastic net with a
     profile-bootstrap 95% CI (1,000 draws, seed 20261004, resampling profiles within the tissue).
  e  Published clocks on the same AltumAge test profiles (as evaluate_age_revision_2026-10-01.py, Zhang 2019 on
     the >= 90%-CpG subset and its whole-blood part as age_subsets_2026-10-02.py) and on held-out datasets
     (age_heldout_clocks_2026-10-04.py logic, constant-output datasets excluded).
  f  AltumAge test profiles from datasets outside the training split (split != 'excluded_downstream').
  g  Generation Scotland training+validation participants: MedAE, MAE, r, mean offset (aggregate only).
  h  RRBS GSE194154: each run loaded with load_model(), r, mean bias, fivefold recalibrated MAE (same folds),
     with and without GSM5829631.
  i  Reprogramming GSE54848: per-sample ensemble age, day means, day 0, range of day means from day 20, day 15.

Outputs
  artifacts/eval/age_ensemble_2026-10-04.json
  artifacts/eval/age_ensemble_2026-10-04.predictions.npz   (ensemble per profile: AltumAge scenarios, held-out)
"""
from __future__ import annotations

import importlib.util
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, Finv, age_scores, cuda_preinit, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
RUNS = {"run 1": ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt",
        "seed 2": ARTIFACTS / "runs/multitask_bio_seed20261002/final.pt",
        "seed 3": ARTIFACTS / "runs/multitask_bio_seed20261003/final.pt"}
SCEN = ["all", "random_30", "random_50", "random_70", "random_90", "hm27", "hm450", "epic_v2"]
EN = "ElasticNet (same labels)"
FLAGGED = "GSM5829631"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ar = _load("age_revision", "evaluate_age_revision_2026-10-01.py")          # cluster_ci, paired_ci, clock_table, load_model
hp = _load("age_heldout_paired", "age_heldout_paired_2026-10-03.py")      # paired (seed 20261002)
hc = _load("age_heldout_clocks", "age_heldout_clocks_2026-10-04.py")      # paired_medae (seed 20261004)


def medae(p, y):
    return float(np.median(np.abs(np.asarray(p) - np.asarray(y))))


def tissue_boot(pa, pb, y, n=1000, seed=20261004):
    """MedAE difference (a - b) with a 95% interval from resampling profiles (paired)."""
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(pa - y), np.abs(pb - y)
    d = []
    for _ in range(n):
        k = rng.integers(0, len(y), len(y))
        d.append(np.median(ea[k]) - np.median(eb[k]))
    return {"difference": float(np.median(ea) - np.median(eb)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def main():
    t0 = time.time()
    res = {"ensemble": "mean of the age predictions of run 1 (released), seed 2 and seed 3",
           "checkpoints": {k: str(v) for k, v in RUNS.items()}}
    z1 = np.load(require(EVAL / "age_revision_2026-10-01.predictions.npz"))
    zs = {"seed 2": np.load(require(EVAL / "age_seed2_scenarios_2026-10-04.predictions.npz")),
          "seed 3": np.load(require(EVAL / "age_seed3_scenarios_2026-10-04.predictions.npz"))}
    ids, y, tis_te = z1["altumage_ids"].astype(str), z1["altumage_age"], z1["altumage_tissue"].astype(str)
    yho, flag, tis_ho = z1["heldout_age"], z1["heldout_flag"], z1["heldout_tissue"].astype(str)

    # ---- consistency checks: identical profiles, masks and checkpoints -------------------------------------
    chk = {}
    for s, z in zs.items():
        assert (z["altumage_ids"].astype(str) == ids).all() and np.array_equal(z["altumage_age"], y)
        assert np.array_equal(z["heldout_age"], yho) and np.array_equal(z["heldout_flag"], flag)
        chk[s] = {"elasticnet_scenario_max_abs_diff_vs_run1_file": {sc: float(np.max(np.abs(z[f"{EN}|{sc}"] - z1[f"{EN}|{sc}"])))
                                                                     for sc in SCEN},
                  "elasticnet_heldout_max_abs_diff": float(np.max(np.abs(z[f"{EN}|heldout"] - z1[f"{EN}|heldout"]))),
                  "seed_all_max_abs_diff_vs_2026-10-01_file": float(np.max(np.abs(z[f"MethylGPT {s}|all"] - z1[f"MethylGPT {s}|all"]))),
                  "seed_heldout_max_abs_diff_vs_2026-10-01_file": float(np.max(np.abs(z[f"MethylGPT {s}|heldout"] - z1[f"MethylGPT {s}|heldout"])))}
        for sc in SCEN:
            if not np.array_equal(z[f"{EN}|{sc}"], z1[f"{EN}|{sc}"]):
                raise SystemExit(f"elastic-net predictions differ for {s} {sc}: masks not identical")
    res["consistency_checks"] = chk
    log(f"consistency: {chk}")

    P = {"run 1": {sc: z1[f"MethylGPT|{sc}"] for sc in SCEN}}
    for s, z in zs.items():
        P[s] = {sc: z[f"MethylGPT {s}|{sc}"] for sc in SCEN}
    ens = {sc: np.mean([P[r][sc] for r in RUNS], axis=0) for sc in SCEN}
    en = {sc: z1[f"{EN}|{sc}"] for sc in SCEN}
    ho = {"run 1": z1["MethylGPT|heldout"], **{s: z[f"MethylGPT {s}|heldout"] for s, z in zs.items()}}
    ens_ho, en_ho = np.mean([ho[r] for r in RUNS], axis=0), z1[f"{EN}|heldout"]
    models_te = {"ensemble": ens["all"], "run 1": P["run 1"]["all"], "elastic net": en["all"]}
    models_ho = {"ensemble": ens_ho, "run 1": ho["run 1"], "elastic net": en_ho}

    prof = pd.read_parquet(CORPUS_DIR / "profiles.parquet", columns=["id", "split", "group"])
    lab = pd.read_parquet(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet")
    pte = prof.set_index("id").reindex(ids)
    grp_te = pte.group.astype(str).to_numpy()
    rows = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    assert len(rows) == len(yho)
    ho_ids = prof.id.to_numpy()[rows].astype(str)
    grp_ho = prof.group.astype(str).to_numpy()[rows]

    # ---- a. AltumAge test ---------------------------------------------------------------------------------
    a = {}
    for k, p in models_te.items():
        a[k] = {**age_scores(p, y), "medae_ci95_dataset_bootstrap": ar.cluster_ci(p, y, grp_te)}
    a["paired_medae_vs_elastic_net"] = {k: ar.paired_ci(models_te[k], en["all"], y, grp_te) for k in ("ensemble", "run 1")}
    a["paired_medae_ensemble_vs_run1"] = ar.paired_ci(ens["all"], P["run 1"]["all"], y, grp_te)
    a["seed 2"], a["seed 3"] = age_scores(P["seed 2"]["all"], y), age_scores(P["seed 3"]["all"], y)
    res["a_altumage_test"] = a

    # ---- b. missing-CpG scenarios -------------------------------------------------------------------------
    res["b_missing_cpg_scenarios"] = {sc: {"ensemble": age_scores(ens[sc], y), "run 1": age_scores(P["run 1"][sc], y),
                                           "seed 2": age_scores(P["seed 2"][sc], y), "seed 3": age_scores(P["seed 3"][sc], y),
                                           "elastic net": age_scores(en[sc], y)} for sc in SCEN}

    # ---- c. held-out datasets -----------------------------------------------------------------------------
    c = {}
    eb = np.abs(en_ho - yho)
    for name, m in (("all_34_datasets", np.ones(len(yho), bool)), ("reliable_ages_21_datasets", ~flag)):
        d = {"n_profiles": int(m.sum()), "n_datasets": int(pd.Series(grp_ho[m]).nunique())}
        for k in ("ensemble", "run 1"):
            ea = np.abs(models_ho[k] - yho)
            per = pd.DataFrame({"g": grp_ho[m], "a": ea[m], "b": eb[m]}).groupby("g").median()
            md, ma = hp.paired(ea[m], eb[m], grp_ho[m], np.median), hp.paired(ea[m], eb[m], grp_ho[m], np.mean)
            ren = lambda r: {"model": r["methylgpt"], "elastic_net": r["elastic_net"], "difference": r["difference"], "ci95": r["ci95"]}
            d[k] = {"datasets_lower_medae_than_elastic_net": int((per.a < per.b).sum()), "medae": ren(md), "mae": ren(ma),
                    "pearson_r": float(np.corrcoef(models_ho[k][m], yho[m])[0, 1])}
        d["seed 2 medae"], d["seed 3 medae"] = medae(ho["seed 2"][m], yho[m]), medae(ho["seed 3"][m], yho[m])
        c[name] = d
    res["c_heldout"] = c

    # ---- d. tissues ---------------------------------------------------------------------------------------
    dt = {}
    for t in sorted(set(tis_te)):
        m = tis_te == t
        if m.sum() >= 50:
            dt[f"altumage_test:{t}"] = {"n": int(m.sum()), **{k: medae(v[m], y[m]) for k, v in models_te.items()},
                                        "ensemble_minus_elastic_net": tissue_boot(ens["all"][m], en["all"][m], y[m]),
                                        "run1_minus_elastic_net": tissue_boot(P["run 1"]["all"][m], en["all"][m], y[m])}
    for t in sorted(set(tis_ho) - {"unlabelled"}):      # evaluate_age_revision drops profiles without a tissue label
        m = (tis_ho == t) & ~flag
        if m.sum() >= 50:
            dt[f"heldout_unflagged:{t}"] = {"n": int(m.sum()), **{k: medae(v[m], yho[m]) for k, v in models_ho.items()},
                                            "ensemble_minus_elastic_net": tissue_boot(ens_ho[m], en_ho[m], yho[m]),
                                            "run1_minus_elastic_net": tissue_boot(ho["run 1"][m], en_ho[m], yho[m])}
    res["d_tissue"] = dt

    # ---- e. published clocks ------------------------------------------------------------------------------
    ct = ar.clock_table(list(ids))
    zf = ct.zhang_observed_fraction.to_numpy(float)
    clock_masks = {}
    for cname in [c_ for c_ in ct.columns if c_ not in ("zhang_observed_fraction", "Zhang 2019")]:
        ok = np.isfinite(ct[cname].to_numpy(float))
        if ok.sum() >= 50:
            clock_masks[cname] = (ct[cname].to_numpy(float), ok)
    zp = ct["Zhang 2019"].to_numpy(float)
    clock_masks["Zhang 2019 (>= 90% of its CpGs observed)"] = (zp, np.isfinite(zp) & (zf >= 0.9))
    clock_masks["Zhang 2019 (>= 90%, whole blood)"] = (zp, np.isfinite(zp) & (zf >= 0.9) & (tis_te == "Whole Blood"))
    ec = {}
    for cname, (v, ok) in clock_masks.items():
        ec[cname] = {"n": int(ok.sum()), "clock": age_scores(v[ok], y[ok]),
                     **{k: age_scores(p[ok], y[ok]) for k, p in models_te.items()}}
    res["e_clocks_altumage_test"] = ec

    cth = ar.clock_table(list(ho_ids))
    eh = {}
    for cname in [c_ for c_ in cth.columns if c_ not in ("zhang_observed_fraction", "Zhang 2019")]:
        v = cth[cname].to_numpy(float).copy()
        const = pd.Series(v).groupby(grp_ho).transform(lambda s: s.dropna().nunique() <= 1 and s.notna().sum() > 1).to_numpy(bool)
        eh.setdefault("excluded_constant_output_datasets", {})[cname] = sorted(set(grp_ho[const & np.isfinite(v)]))
        v[const] = np.nan
        for name, m0 in (("all", np.ones(len(yho), bool)), ("reliable_ages", ~flag)):
            m = m0 & np.isfinite(v)
            if m.sum() < 50:
                continue
            eh.setdefault(cname, {})[name] = {"n_profiles": int(m.sum()), "n_datasets": int(pd.Series(grp_ho[m]).nunique()),
                                              "clock_medae": medae(v[m], yho[m]),
                                              **{f"{k}_medae": medae(p[m], yho[m]) for k, p in models_ho.items()},
                                              "ensemble_minus_clock_ci95": hc.paired_medae(ens_ho[m], v[m], yho[m], grp_ho[m]),
                                              "run1_minus_clock_ci95": hc.paired_medae(ho["run 1"][m], v[m], yho[m], grp_ho[m])}
    res["e_clocks_heldout"] = eh

    # ---- f. AltumAge test profiles outside training-split datasets ----------------------------------------
    outside = (pte.split != "excluded_downstream").to_numpy()
    f = {"n_profiles": int(outside.sum()), "n_datasets": int(pd.Series(grp_te[outside]).nunique()),
         **{k: age_scores(p[outside], y[outside]) for k, p in models_te.items()}, "clocks": {}}
    for cname, (v, ok) in clock_masks.items():
        m = ok & outside
        if m.sum() == 0:
            continue
        f["clocks"][cname] = {"n": int(m.sum()), "clock_medae": medae(v[m], y[m]),
                              **{f"{k}_medae": medae(p[m], y[m]) for k, p in models_te.items()}}
    res["f_altumage_test_outside_training_datasets"] = f

    np.savez_compressed(EVAL / "age_ensemble_2026-10-04.predictions.npz", altumage_ids=ids, altumage_age=y,
                        altumage_tissue=tis_te, heldout_ids=ho_ids, heldout_age=yho, heldout_flag=flag, heldout_tissue=tis_ho,
                        **{f"ensemble|{sc}": v for sc, v in ens.items()}, **{"ensemble|heldout": ens_ho})

    # ---- g. Generation Scotland (aggregate only) ----------------------------------------------------------
    gsm = _load("gs_age", "evaluate_gs_age_clocks_2026-10-02.py")
    labels = []
    for sp in ("train", "val"):
        lp = require(next(gsm.CACHE.glob(f"COMMON_SAMPLE_SPLIT_{gsm.SPLIT}_{sp}_all_gs_batches_all_probes_*.labels.parquet")))
        labels.append(pd.read_parquet(lp, columns=["dnam_id", "age"]).assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id"))
    age = pd.concat(labels)["age"].astype(float)
    age = age[~age.index.duplicated()]

    def gs_feat(folder, kind):
        fr = []
        for sp in ("train", "val"):
            zz = np.load(require(folder / f"COMMON_SAMPLE_SPLIT_{gsm.SPLIT}_{sp}_{kind}.embedding.npz"))
            fr.append(pd.Series(zz["embedding"][:, 0], index=zz["dnam_id"].astype(str)))
        return pd.concat(fr)
    gp = pd.DataFrame({"run 1": gs_feat(ARTIFACTS / "gs_features_2026-10-01", "mt_age"),
                       "seed 2": gs_feat(ARTIFACTS / "gs_features_seeds_2026-10-04", "mtseed2_age"),
                       "seed 3": gs_feat(ARTIFACTS / "gs_features_seeds_2026-10-04", "mtseed3_age")})
    gp["ensemble"] = gp[list(RUNS)].mean(axis=1)
    common = gp.dropna().index.intersection(age.dropna().index)
    res["g_generation_scotland"] = {"participants": "training and validation subsets (test subset not loaded)",
                                    "n_common": int(len(common)),
                                    **{k: gsm.scores(gp.loc[common, k].to_numpy(float), age.loc[common].to_numpy(float))
                                       for k in ("ensemble", "run 1", "seed 2", "seed 3")}}
    del gp, age, labels

    # ---- h/i. models on GPU ------------------------------------------------------------------------------
    cuda_preinit()
    dev = torch.device("cuda:0")
    vocab = np.load(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz")["vocab"].astype(str)
    zen = np.load(require(EVAL / "baseline_corpus_age_clean_rerun_2026-10-01.enet_model.npz"))

    def enet(X):
        T = np.where(np.isfinite(X), X, zen["mu"])
        return Finv(((T - zen["mu"]) / zen["sd"]) @ zen["w"] * float(zen["ys"]) + float(zen["ym"]))

    D = ARTIFACTS / "external/GSE194154"
    betas = pd.read_parquet(require(D / "betas_cov10.parquet"))
    samples = pd.read_csv(require(D / "samples.csv")).set_index("gsm").loc[betas.index]
    Xr = betas.reindex(columns=vocab).to_numpy(np.float32)
    Xr = np.where(np.isfinite(Xr), np.clip(Xr, 0, 1), np.nan).astype(np.float32)     # as the release API (_matrix)
    yr = samples["age"].astype(float).to_numpy()

    gd = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/dataset/GSE54848")
    meta = pd.read_csv(require(gd / "metadata.csv"))
    meta = meta[meta.platform_id == "GPL13534"].reset_index(drop=True)
    gsm_ids = meta.geo_accession.tolist()
    mm = pd.read_csv(require(gd / "expression_matrix.csv"), usecols=["ID_REF"] + gsm_ids, index_col=0)
    mm = mm[mm.index.astype(str).str.startswith("cg")].astype(float)
    Xp = mm.reindex(vocab).T.to_numpy(np.float32)
    Xp = np.where(np.isfinite(Xp), np.clip(Xp, 0, 1), np.nan).astype(np.float32)
    titles = meta.title.astype(str)
    DAY = {"p1": 35, "p2": 42, "p3": 49}
    day = np.array([float(re.match(r"Day(\d+)_", s).group(1)) if s.startswith("Day") else DAY.get(s.split("_")[0], np.nan) for s in titles])

    rr, rp = {}, {}
    for r, path in RUNS.items():
        fpred = ar.load_model(require(path), dev)
        rr[r], rp[r] = fpred(Xr), fpred(Xp)
        del fpred
        torch.cuda.empty_cache()
    rr["ensemble"], rp["ensemble"] = np.mean([rr[r] for r in RUNS], axis=0), np.mean([rp[r] for r in RUNS], axis=0)
    rr["elastic net"], rp["elastic net"] = enet(Xr), enet(Xp)

    # checks against the existing per-sample outputs (run 1 and elastic net)
    old_r = pd.read_csv(EVAL / "rrbs_gse194154_2026-10-02.csv").set_index("gsm").loc[betas.index]
    old_p = pd.read_csv(EVAL / "reprogramming_gse54848_2026-10-01.csv").set_index("gsm").loc[gsm_ids]
    res["consistency_checks"]["rrbs_run1_load_model_vs_release_api_max_abs_diff"] = float(np.max(np.abs(rr["run 1"] - old_r["MethylGPT"].to_numpy())))
    res["consistency_checks"]["rrbs_elastic_net_max_abs_diff"] = float(np.max(np.abs(rr["elastic net"] - old_r[EN].to_numpy())))
    res["consistency_checks"]["reprogramming_run1_max_abs_diff_vs_2026-10-01_csv"] = float(np.max(np.abs(rp["run 1"] - old_p["MethylGPT-2"].to_numpy())))

    from sklearn.model_selection import KFold

    def recalibrated(pred, yy, seed=20261002):
        pred, yy = np.asarray(pred, float), np.asarray(yy, float)
        cal = np.empty(len(yy))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(pred):
            b, a_ = np.polyfit(pred[tr], yy[tr], 1)
            cal[te] = a_ + b * pred[te]
        return cal

    keep = betas.index.to_numpy() != FLAGGED
    h = {"n_samples": int(len(yr))}
    for k in ("ensemble", "run 1", "seed 2", "seed 3", "elastic net"):
        p = rr[k]
        h[k] = {"all": age_scores(p, yr), f"without_{FLAGGED}": age_scores(p[keep], yr[keep]),
                "mean_bias_years": float(np.mean(p - yr)), f"mean_bias_years_without_{FLAGGED}": float(np.mean(p[keep] - yr[keep])),
                "recalibrated_5fold": age_scores(recalibrated(p, yr), yr),
                f"recalibrated_5fold_without_{FLAGGED}": age_scores(recalibrated(p[keep], yr[keep]), yr[keep])}
    res["h_rrbs_gse194154"] = h

    per = pd.DataFrame({"gsm": gsm_ids, "title": titles, "day": day, **{k: rp[k] for k in ("ensemble", "run 1", "seed 2", "seed 3", "elastic net")}})
    i = {"n_samples": int(len(per)), "per_sample": per.round(4).astype(object).where(per.notna(), None).to_dict(orient="records")}
    dm = per.dropna(subset=["day"]).groupby("day")[["ensemble", "run 1", "seed 2", "seed 3", "elastic net"]].mean()
    i["day_means"] = {str(int(d_)): row.to_dict() for d_, row in dm.iterrows()}
    for k in ("ensemble", "run 1", "elastic net"):
        late = dm.loc[dm.index >= 20, k]
        i[k] = {"day0_mean": float(dm.loc[0.0, k]), "day_ge20_day_means_range": [float(late.min()), float(late.max())],
                "day15_replicates": [float(v) for v in per.loc[per.day == 15, k]]}
    res["i_reprogramming_gse54848"] = i
    res["elapsed_s"] = time.time() - t0
    write_json(EVAL / "age_ensemble_2026-10-04.json", res)
    log(f"done in {res['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
