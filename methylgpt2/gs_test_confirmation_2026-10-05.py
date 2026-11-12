#!/usr/bin/env python3
"""One-time Generation Scotland test-subset confirmation (pre-specified in the test-confirmation plan of 5 Oct 2026).

--mode dryrun  development check: fitting set = training-subset participants outside outer fold 1 of each
               endpoint, evaluation set = outer fold 1. No test file is read. (The stacked weights were
               chosen with all five development folds, so the dry run is a code check, not an estimate.)
--mode test    fitting set = all 9,866 training-subset participants, evaluation set = the 3,289 test-subset
               participants. Refuses to run if its output already exists.
--mode val     added 6 October 2026, after the test run: the same frozen models evaluated on the 3,289 validation-subset
               participants, which no disease analysis of this revision had used (post hoc, reported as such).

Models, scenarios, outcomes and the interpretation rule are those of the plan. Elastic-net models use
gpu_coxnet_2026_10_01.select_and_fit (seed 20260928); clocks and age + sex use unpenalized Breslow Cox
models of standardized covariates; MethylGPT combined and stacked models standardize each component risk
within the evaluation set. C-index: scikit-survival definition on the evaluation set; 2,000 bootstrap
resamples of evaluation participants (default_rng(20261005)), the same resample for every model.
Outputs (aggregate): artifacts/gs_test_2026-10-05/{dryrun,confirmation}.json; participant-level risks
stay in artifacts/gs_test_2026-10-05/private_<mode>_risks.parquet (git-ignored, controlled data).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from methylgpt2_lib import ARTIFACTS, Finv, cuda_preinit, log, require, write_json
from methylgpt2_model import BioHeads, embed_batch, load_checkpoint
from gpu_coxnet_2026_10_01 import select_and_fit

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stk", HERE / "gs_stacked_2026-10-04.py")
stk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stk)

CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
LABELS = V12 / "revision_server_20260622/server_results/task_a/results/regenerated_survival_split_label_tables"
SCORES = V12 / "revision_server_20260622/server_results/task_a/task_a_gs_clock_scores_fixed_grimage_20260629.csv"
BIOLEARN_AGE = V12 / "revision_round3_20260703/server_results/task_b/biolearn_gs_clock_predictions.csv"
OUT = ARTIFACTS / "gs_test_2026-10-05"
PREDS = ARTIFACTS / "gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet"
AGING = ARTIFACTS / "runs/aging_heads_clean_2026-10-01/best.pt"
RUNS = {"run 1": ARTIFACTS / "runs/multitask_bio_2026-10-01/final.pt",
        "seed 2": ARTIFACTS / "runs/multitask_bio_seed20261002/final.pt",
        "seed 3": ARTIFACTS / "runs/multitask_bio_seed20261003/final.pt"}
SCEN = ("complete", "hm27", "random_90")
COMPS = ("E", "B", "R")
MODELS = ("age_sex", "raw_cpg_dense", "grimage2", "dunedinpace", "grimage2_dunedinpace", "E", "B", "R",
          "mgpt_combined_equal", "mgpt_stacked")
CLOCK_MODELS = {"grimage2": ["GrimAgeV2"], "dunedinpace": ["DunedinPACE"], "grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"]}
PRIMARY = [("P1", "complete", "mean", "mgpt_stacked", c) for c in ("raw_cpg_dense", "grimage2_dunedinpace")] + \
          [("P2", "complete", "Endocrine_and_Metabolic_Diseases", "mgpt_stacked", c) for c in ("raw_cpg_dense", "grimage2_dunedinpace")] + \
          [("P3", s, "mean", "mgpt_combined_equal", c) for s in ("hm27", "random_90") for c in ("raw_cpg_dense", "grimage2_dunedinpace")]
NI_MARGIN = -0.01


def matrix(split):
    mp = require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{split}_all_gs_batches_all_probes_*.matrix.npz")))
    ids = pd.read_parquet(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet")), columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    if X.shape != (len(ids), 49156):
        raise ValueError(f"unexpected matrix shape {X.shape}")
    return ids, X


def degrade(X, scen, plat):
    Xs = X.copy()
    if scen == "random_90":
        Xs[np.random.default_rng(20261001 + 90).random(Xs.shape) < 0.9] = np.nan
    elif scen == "hm27":
        Xs[:, ~plat["hm27"]] = np.nan
    return Xs


class Featurizer:
    def __init__(self, dev):
        self.dev = dev
        ag = torch.load(AGING, map_location=dev, weights_only=False)
        self.model, self.core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
        self.model.load_state_dict(ag["model"]); self.model.eval()
        self.mt = self._multitask(RUNS["run 1"])

    def _multitask(self, path):
        mt = torch.load(path, map_location=self.dev, weights_only=False)
        m, c, _ = load_checkpoint(Path(mt["ckpt"]), None, self.dev)
        m.load_state_dict(mt["model"]); m.eval()
        h = BioHeads(c.rank + m.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(self.dev)
        h.load_state_dict(mt["heads"]); h.eval()
        return m, c, h, float(mt["fs"]), float(mt["fm"])

    def features(self, X):
        m, c, h, _, _ = self.mt
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), 32):
                x = torch.from_numpy(X[i:i + 32]).to(self.dev)
                obs = torch.isfinite(x)
                E.append(embed_batch(self.model, self.core, x, obs).float().cpu().numpy())
                B.append(h(embed_batch(m, c, x, obs))[3].float().cpu().numpy())
                s = self.core.posterior(x, obs.float())
                R.append(torch.where(obs, x - self.core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        E, B, R = np.concatenate(E), np.concatenate(B), np.concatenate(R)
        return {"E": E, "B": B, "R": np.hstack([E, B, R]).astype(np.float32)}

    def ages(self, X):
        out = {}
        for name, path in RUNS.items():
            m, c, h, fs, fm = self._multitask(path)
            a = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    x = torch.from_numpy(X[i:i + 32]).to(self.dev)
                    a.append(Finv(h(embed_batch(m, c, x, torch.isfinite(x)))[0].float().cpu().numpy() * fs + fm))
            out[name] = np.concatenate(a).ravel()
            del m, c, h
        out["ensemble"] = np.mean([out[k] for k in RUNS], axis=0)
        return out


def cox_unpen(M_fit, ev, du):
    m, s = M_fit.mean(0), M_fit.std(0)
    cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((M_fit - m) / s, Surv.from_arrays(ev, du))
    return lambda M: ((M - m) / s) @ cx.coef_


def z(r):
    return (r - r.mean()) / r.std(ddof=1)


def c_index(ev, du, r):
    return concordance_index_censored(ev, du, r)[0]


def test_labels(endpoint, ids, split="test"):
    frame = pd.read_parquet(require(LABELS / f"survial_{endpoint}_{SPLIT}_split_labels.parquet"), filters=[("split", "==", split)])
    frame["dnam_id"] = frame["dnam_id"].astype(str)
    keep = frame[frame.dnam_id.isin(set(ids))].set_index("dnam_id")
    mapped = keep["sex"].map({"F": 1.0, "M": 0.0, "Female": 1.0, "Male": 0.0, "female": 1.0, "male": 0.0})
    keep["sex"] = mapped.where(mapped.notna(), pd.to_numeric(keep["sex"], errors="coerce"))
    for col in ("age", "sex", "duration", "event"):
        if not np.isfinite(keep[col].to_numpy(float)).all():
            raise ValueError(f"non-finite {col} in test labels for {endpoint}")
    if (keep.duration <= 0).any() or not keep.event.isin([0, 1]).all():
        raise ValueError(f"invalid survival labels for {endpoint}")
    return keep.reset_index()[["dnam_id", "age", "sex", "duration", "event"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dryrun", "test", "val"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    out_json = OUT / {"test": "confirmation.json", "dryrun": "dryrun.json", "val": "validation_subset.json"}[a.mode]
    risk_path = OUT / f"private_{a.mode}_risks.parquet"
    if a.mode in ("test", "val") and (out_json.exists() or risk_path.exists()):
        raise SystemExit("The test confirmation has already been run; it is run once (see the plan).")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True  # as in feature extraction for development
    dev = a.device
    weights = json.loads(require(OUT / "final_stacked_weights.json").read_text())["weights"]
    plat = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))

    preds = pd.read_parquet(require(PREDS))
    preds = preds[preds.age_adjustment == "linear"].copy()
    preds["dnam_id"] = preds.dnam_id.astype(str)
    endpoints = sorted(preds.endpoint.unique())

    tr_ids, Xtr = matrix("train")
    tr_pos = {s: i for i, s in enumerate(tr_ids)}
    fz = Featurizer(dev)
    log("features: training subset, complete profiles")
    Ftr = fz.features(Xtr)
    checks = {}
    for nm, key in (("E", "mgpt2_embed"), ("B", "mt_bio")):
        ref = np.load(require(ARTIFACTS / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_train_{key}.embedding.npz"), allow_pickle=False)
        assert (ref["dnam_id"].astype(str) == tr_ids).all()
        checks[f"{nm}_train_vs_stored_max_abs_diff"] = float(np.abs(Ftr[nm] - ref["embedding"]).max())
    log(f"feature checks {checks}")

    sc = pd.read_csv(require(SCORES)); sc["sample_id"] = sc.sample_id.astype(str)
    full_clock = sc.pivot_table(index="sample_id", columns="model", values="score")
    if a.mode == "dryrun":
        ev_ids = None  # endpoint-specific outer fold 1
        Xev_all, ev_pos = Xtr, tr_pos
        scen_clock = pd.read_csv(require(ARTIFACTS / "gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv"))
    else:
        te_ids, Xev_all = matrix(a.mode)
        ev_pos = {s: i for i, s in enumerate(te_ids)}
        if set(te_ids) & set(tr_ids):
            raise SystemExit("test and training identifiers overlap")
        scen_clock = pd.read_csv(require(OUT / f"private_{a.mode}_clock_scores.csv"))
        sfull = scen_clock[scen_clock.scenario == "full"].set_index("sample_id")
        sfull.index = sfull.index.astype(str)
        checks["test_full_clock_rescored_vs_v12_max_abs_diff"] = float(np.nanmax(np.abs(
            sfull[["GrimAgeV2", "DunedinPACE"]].to_numpy(float) - full_clock.loc[sfull.index, ["GrimAgeV2", "DunedinPACE"]].to_numpy(float))))
    scen_clock["sample_id"] = scen_clock.sample_id.astype(str)
    clock_by_scen = {"complete": full_clock, "hm27": scen_clock[scen_clock.scenario == "hm27"].set_index("sample_id"),
                     "random_90": scen_clock[scen_clock.scenario == "random_90"].set_index("sample_id")}

    # evaluation features per scenario (test mode: the whole test matrix once; dry run: training matrix rows)
    Fev = {}
    for s in SCEN:
        log(f"features: evaluation set, {s}")
        Xs = degrade(Xev_all, s, plat)
        Fev[s] = {**fz.features(Xs), "raw": Xs}
        checks[f"observed_fraction_{s}"] = float(np.isfinite(Xs).mean())
    ages = fz.ages(Xev_all) if a.mode in ("test", "val") else None

    rows, frames = [], []
    for ep in endpoints:
        pe = preds[preds.endpoint == ep].reset_index(drop=True)
        if a.mode == "dryrun":
            fit_df, ev_df = pe[pe.outer_fold != 1].reset_index(drop=True), pe[pe.outer_fold == 1].reset_index(drop=True)
        else:
            fit_df, ev_df = pe, test_labels(ep, list(ev_pos), a.mode)
        fi = np.array([tr_pos[s] for s in fit_df.dnam_id])
        ei = np.array([ev_pos[s] for s in ev_df.dnam_id])
        fev, fdu = fit_df.event.astype(bool).to_numpy(), fit_df.duration.to_numpy(float)
        eev, edu = ev_df.event.astype(bool).to_numpy(), ev_df.duration.to_numpy(float)
        cov_f, cov_e = fit_df[["age", "sex"]].to_numpy(np.float32), ev_df[["age", "sex"]].to_numpy(np.float32)
        cm, cs = stk.fit_scaler(cov_f, dev)
        risk = {s: {} for s in SCEN}
        info = {}
        for name, Ffit in (("E", Ftr["E"]), ("B", Ftr["B"]), ("R", Ftr["R"]), ("raw_cpg_dense", Xtr)):
            fm, fs = stk.fit_scaler(Ffit[fi], dev)
            Xf = np.hstack([stk.apply_scaler(cov_f, cm, cs, dev), stk.apply_scaler(Ffit[fi], fm, fs, dev)])
            w, alpha, k = select_and_fit(Xf, fdu, fev, seed=20260928, dev=dev, iters=a.iters)
            del Xf
            info[name] = {"alpha": alpha, "alpha_index": k, "nonzero": int((w[2:] != 0).sum())}
            key = "raw" if name == "raw_cpg_dense" else name
            for s in SCEN:
                Xe = np.hstack([stk.apply_scaler(cov_e, cm, cs, dev), stk.apply_scaler(Fev[s][key][ei], fm, fs, dev)])
                risk[s][name] = Xe @ w
        f_as = cox_unpen(cov_f.astype(float), fev, fdu)
        for s in SCEN:
            risk[s]["age_sex"] = f_as(cov_e.astype(float))
        for nm, cols in CLOCK_MODELS.items():
            Mf = np.column_stack([cov_f, full_clock.loc[fit_df.dnam_id, cols].to_numpy(float)])
            fcl = cox_unpen(Mf, fev, fdu)
            for s in SCEN:
                risk[s][nm] = fcl(np.column_stack([cov_e, clock_by_scen[s].loc[ev_df.dnam_id, cols].to_numpy(float)]))
        wt = weights[ep]
        for s in SCEN:
            zc = {c: z(risk[s][c]) for c in COMPS}
            risk[s]["mgpt_combined_equal"] = zc["E"] + zc["B"] + zc["R"]
            risk[s]["mgpt_stacked"] = sum(wt[c] * zc[c] for c in COMPS)
        q = ev_df[["dnam_id", "event", "duration"]].assign(endpoint=ep)
        for s in SCEN:
            for m in MODELS:
                q[f"{s}|{m}"] = risk[s][m]
            rows += [{"endpoint": ep, "scenario": s, "model": m, "n": len(ev_df), "events": int(eev.sum()),
                      "c_index": float(c_index(eev, edu, risk[s][m]))} for m in MODELS]
        frames.append(q)
        log(f"{ep} (n {len(ev_df)}, events {int(eev.sum())}): " + ", ".join(
            f"{m} {c_index(eev, edu, risk['complete'][m]):.3f}" for m in ("age_sex", "raw_cpg_dense", "grimage2_dunedinpace", "mgpt_combined_equal", "mgpt_stacked")))
        log(f"  components: {info}")
    R = pd.concat(frames, ignore_index=True)
    R.to_parquet(risk_path, index=False)
    cdf = pd.DataFrame(rows)

    # ---- bootstrap: same participant resample for every model; shared across endpoints when the sets match
    sets = {ep: set(R[R.endpoint == ep].dnam_id) for ep in endpoints}
    shared = all(sets[ep] == sets[endpoints[0]] for ep in endpoints)
    ref_ids = sorted(sets[endpoints[0]]) if shared else None
    rng = np.random.default_rng(20261005)
    cols = [f"{s}|{m}" for s in SCEN for m in MODELS]
    boot = {}
    Wshared = None
    if shared:
        idx = rng.integers(0, len(ref_ids), size=(a.draws, len(ref_ids)))
        Wshared = np.zeros((a.draws, len(ref_ids)), np.float32)
        np.add.at(Wshared, (np.repeat(np.arange(a.draws), len(ref_ids)), idx.ravel()), 1)
        rpos = {s: i for i, s in enumerate(ref_ids)}
    for ep in endpoints:
        g = R[R.endpoint == ep].reset_index(drop=True)
        if shared:
            W = Wshared[:, np.array([rpos[s] for s in g.dnam_id])]
        else:
            ii = rng.integers(0, len(g), size=(a.draws, len(g)))
            W = np.zeros((a.draws, len(g)), np.float32)
            np.add.at(W, (np.repeat(np.arange(a.draws), len(g)), ii.ravel()), 1)
        pc = stk.PairCounter(g.event.astype(bool).to_numpy(), g.duration.to_numpy(float), dev)
        cnt = pc.counts(np.stack([g[c].to_numpy(float) for c in cols]), W)  # (m, b, 3)
        boot[ep] = {c: stk.cidx(cnt[j]) for j, c in enumerate(cols)}
        obs = pc.counts(np.stack([g[c].to_numpy(float) for c in cols]))
        for j, c in enumerate(cols):  # exactness check against sksurv on the observed data
            s_, m_ = c.split("|")
            ref = float(cdf[(cdf.endpoint == ep) & (cdf.scenario == s_) & (cdf.model == m_)].c_index.iloc[0])
            checks["max_abs_gpu_vs_sksurv_c"] = max(checks.get("max_abs_gpu_vs_sksurv_c", 0.0), abs(float(stk.cidx(obs[j])) - ref))
        log(f"bootstrap {ep} done ({time.time() - t0:.0f}s)")

    def contrast(s, ep, m, c):
        if ep == "mean":
            d = np.mean([boot[e][f"{s}|{m}"] - boot[e][f"{s}|{c}"] for e in endpoints], 0)
            pt = float(np.mean([cdf[(cdf.endpoint == e) & (cdf.scenario == s) & (cdf.model == m)].c_index.iloc[0] -
                                cdf[(cdf.endpoint == e) & (cdf.scenario == s) & (cdf.model == c)].c_index.iloc[0] for e in endpoints]))
        else:
            d = boot[ep][f"{s}|{m}"] - boot[ep][f"{s}|{c}"]
            pt = float(cdf[(cdf.endpoint == ep) & (cdf.scenario == s) & (cdf.model == m)].c_index.iloc[0] -
                       cdf[(cdf.endpoint == ep) & (cdf.scenario == s) & (cdf.model == c)].c_index.iloc[0])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
        return {"difference": pt, "ci95": [float(lo), float(hi)], "p": p}

    primary = []
    for pid, s, ep, m, c in PRIMARY:
        r = contrast(s, ep, m, c)
        r.update({"id": pid, "scenario": s, "endpoint": ep, "model": m, "comparator": c,
                  "superiority_confirmed": bool(r["ci95"][0] > 0), "non_inferiority_confirmed": bool(r["ci95"][0] > NI_MARGIN)})
        primary.append(r)
        log(f"{pid} {s} {ep}: {m} - {c} = {r['difference']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] p {r['p']:.3f} "
            f"superiority {r['superiority_confirmed']} non-inferiority {r['non_inferiority_confirmed']}")
    secondary = []
    for s in SCEN:
        block = []
        for m in ("mgpt_stacked", "mgpt_combined_equal"):
            for c in ("age_sex", "raw_cpg_dense", "grimage2", "dunedinpace", "grimage2_dunedinpace"):
                for ep in endpoints + ["mean"]:
                    block.append({"scenario": s, "endpoint": ep, "model": m, "comparator": c, **contrast(s, ep, m, c)})
        ps = [b["p"] for b in block if b["endpoint"] != "mean"]
        qs = iter(stk.bh(ps))
        for b in block:
            b["q_bh"] = float(next(qs)) if b["endpoint"] != "mean" else None
        secondary += block
    means = cdf.groupby(["scenario", "model"]).c_index.mean().unstack("model").reindex(SCEN)

    res = {"mode": a.mode, "plan": "pre-specified test-confirmation plan, 5 Oct 2026", "draws": a.draws, "shared_resample": bool(shared),
           "checks": checks, "stacked_weights": weights, "c_index": rows, "mean_c_index": means.round(6).to_dict("index"),
           "primary": primary, "secondary": secondary}
    if ages is not None:
        lab = pd.read_parquet(require(next(CACHE.glob(f"COMMON_SAMPLE_SPLIT_{SPLIT}_{a.mode}_all_gs_batches_all_probes_*.labels.parquet"))),
                              columns=["dnam_id", "age"]).assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id")
        y = lab.loc[list(ev_pos), "age"].to_numpy(float)
        bl = pd.read_csv(require(BIOLEARN_AGE), usecols=["sample_id", "model", "pred_age"])
        bl["sample_id"] = bl.sample_id.astype(str)
        bl = bl[bl.model.isin(["Horvathv1", "Hannum"])].pivot_table(index="sample_id", columns="model", values="pred_age")
        P = pd.DataFrame({**{f"MethylGPT {k}": v for k, v in ages.items()}}, index=list(ev_pos)).join(
            bl.rename(columns={"Horvathv1": "Horvath multi-tissue"}), how="left")
        ok = np.isfinite(y) & P.notna().all(axis=1).to_numpy()
        res["age"] = {"n": int(ok.sum()), **{k: {"medae": float(np.median(np.abs(P[k].to_numpy()[ok] - y[ok]))),
                                                 "mae": float(np.mean(np.abs(P[k].to_numpy()[ok] - y[ok]))),
                                                 "pearson_r": float(np.corrcoef(P[k].to_numpy()[ok], y[ok])[0, 1]),
                                                 "mean_offset": float(np.mean(P[k].to_numpy()[ok] - y[ok]))} for k in P.columns}}
    res["elapsed_s"] = time.time() - t0
    write_json(out_json, res)
    log(f"mean C-index by scenario:\n{means.round(4).to_string()}")
    log(f"wrote {out_json} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
