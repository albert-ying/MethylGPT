#!/usr/bin/env python3
"""Large-gain analysis O: Generation Scotland (GS) risk stratification beyond age and sex from analysis M's stored held-out
risks, with analysis L's design unchanged. Tables only. No model is fitted to predict disease and no risk is recomputed: O
re-scores the held-out fold risks that M stored. SIMULATED MEASUREMENT NOISE ON EMPIRICAL PROFILES in every noise scenario
(M's scenarios: measured GS development profiles to which noise was added by the shared generator
large_gain_noise_profiles_2026-11-02.py, commit 729f99e; no profile, outcome or model is simulated). full is the complete
measured profiles.

WHEN AND WHY (stated plainly)
  O was written on 8 Nov 2026, AFTER the results of analyses J, K, L, M and N were known, including M's C-index results. At
  M's primary scenario depth_10 every decision comparator's C-index fell below that of age+sex, so M's C-index rule gave no
  outcome (M_rule_outcome_2026-11-02.csv: not applicable). No risk-stratification metric had been computed on M's noisy risks
  before this design was committed. Before the commit only schemas, shapes and file listings of M's stored risks and tables
  were looked at, L's solvers were timed on synthetic random arrays, and O's stages (with L's) were run end to end on
  synthetic random participants, events and risks with every path redirected to a scratch directory and 40 draws (code
  timing and code tests only; no GS file was read by these tests and nothing from them is reported).
  Reason for O: the C-index compresses differences in risk separation, and M's rule gave no outcome because the comparators
  fell below age+sex. L's metrics were pre-specified on 30 Oct 2026 (commit d59d694, with amendments 196380e and ebec663) for
  J's risks and measure separation beyond age and sex directly: each held-out risk is residualised on age and sex within the
  outer fold, then the log hazard ratio per SD, deciles, event capture and Kaplan-Meier curves are reported. O applies that
  design, unchanged, to M's stored held-out risks. Every result is reported, whichever model it favours.

DESIGN: L's design and code (large_gain_L_gs_risk_stratification_2026-10-30.py as committed, including both of its
  amendments), unchanged except for the items under CHANGES FROM L. L's numerical functions are imported from the committed L
  script and used as they are: prepare (age and sex residual, within-fold standardisation, constant-risk test, deciles),
  CoxData and cox_newton (stratified Breslow Cox with frequency weights), design, km_at (weighted Kaplan-Meier), release_row,
  release_group and release_km (disclosure control), pct and pboot. L's constants are read from L (constant-risk factor 1e-5,
  10 years = 3,652.5 days, yearly grid 0 to 17, anchor year 10, minimum cell 5, chance 0.10, tolerances). The stage code
  (prep, checks, boot, analyse, sensitivity, verify) is L's, copied and adapted only as listed below.
  As in L: 9,866 GS development participants, 8 endpoints (Autoimmune_Diseases, Cancers, Cardiovascular_Diseases,
  Endocrine_and_Metabolic_Diseases, Kidney_Diseases, Neurological_Diseases, Respiratory_Diseases, date_of_death), every
  participant at risk for every endpoint, outer folds assigned separately per endpoint. Preparation per endpoint, model,
  scenario and outer fold; primary metric beta(model, e) = log hazard ratio per SD of the prepared risk in a Cox model on all
  9,866 participants with age, sex and the prepared risk (Breslow ties, stratified by outer fold), H(model) = mean over the 8
  endpoints of beta (an endpoint with no estimable fold counts as beta = 0 for H, flagged, with H over the estimable
  endpoints beside it); secondary metrics log_hr_per_sd, log_hr_top_vs_bottom, event_capture_top10,
  event_capture_excess_over_10pct, abs_risk_10y_top, abs_risk_10y_bottom, abs_risk_10y_difference, per endpoint and as the
  mean over endpoints; contrasts (a) C*, (b) the scenario's own best decision comparator on H, (c) GrimAge2 with kNN fill,
  (d) each endpoint's own best decision comparator on that metric; not-estimable handling, chance_exact per row, L's
  disclosure control (primary and complementary rules, 4-decimal rounding of survival and absolute risks) and L's verify
  stage (identifier scan, released counts at least 5, fold-level remainder, KM subtraction rules with the whole-follow-up
  total). Stage sensitivity is L's (fold_weighted and leave_one_endpoint_out readings; added in L after its primary result,
  pre-specified here, outside the decision).

CHANGES FROM L (the only ones)
  Risk source: M.assemble(scenario, preds), with M loaded by L's load_module pattern (M itself reads J). M's stored risks:
    artifacts/large_gain_2026-11-02_private/M/{mgpt,clocks,knn}/private_risks_{mgpt,clocks,knn}_<scenario>.parquet and
    cohort/private_cohortknn_<scenario>__<endpoint>.parquet. Participants, events, follow-up, age, sex and the locked outer
    folds are J's (J.load_preds(): artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet,
    age_adjustment "linear"), as in L. No validation or test participant is read.
  Scenarios: depth_10 PRIMARY (read sampling at mean depth 10). Secondary: depth_5, depth_30, gauss_0.05, gauss_0.10, depth_2,
    depth_1, gauss_0.02 (M's generator: reads n ~ Poisson(depth), methylated reads ~ Binomial(n, beta), value m / n; Gaussian:
    clip(beta + sigma e, 0, 1)). full is the reference, not a decision: M's full risks equal J's exactly (M check 2), so O's
    full must reproduce L's stored full results (CHECKS 8).
    Runtime estimate, made before this commit: L's solvers timed on synthetic random arrays of the same size (9,866 rows, 5
    strata, 2,000 frequency-weight draws) took about 28 s per model and endpoint in one process (continuous Cox 11 s, decile
    Cox 17 s, Kaplan-Meier under 1 s) at a machine load average of about 950 on 256 cores. O has 1,792 model-endpoint units
    (8 noise scenarios x 26 models x 8 endpoints, plus 16 models x 8 endpoints in full), about 14 CPU-hours, so about 0.5 h of
    wall time with 32 workers at that speed, and about 2 h if each worker gets only a quarter of a core. The checks stage
    (sksurv C-index for 1,864 model-scenario-endpoints, about 320 PHReg fits) adds well under 1 h. This is below 12 h, so no
    scenario is dropped: all 8 noise scenarios and full are run.
  Models: M.models_for(scenario) without age_sex, the MethylGPT components (J.COMPONENTS: embedding, biomarkers,
    factor_residual, ensemble_embed_bio) and their _depthaware versions. Each noise scenario has 26 models: mgpt2_combined; the
    10 decision comparators; raw_cpg_{v12,gpu}_cohortknn, grimage2_cohortknn, dunedinpace_cohortknn,
    grimage2_dunedinpace_cohortknn; raw_cpg_{v12,gpu}_factor_imputed; the five *_knndenoise; raw_cpg_{v12,gpu}_factor_denoised;
    mgpt2_combined_depthaware. full has 16: L's 8 (mgpt2_combined, raw_cpg_{v12,gpu}_mean, raw_cpg_{v12,gpu}_factor_imputed,
    grimage2_mean, dunedinpace_mean, grimage2_dunedinpace_mean), the five *_knndenoise, the two *_factor_denoised and
    mgpt2_combined_depthaware, which M computed on complete profiles (there the depth-aware risk equals mgpt2_combined
    exactly, M check 3). The decision model is mgpt2_combined. Decision comparators: M.decision_for(scenario) = J's DECISION
    set (raw_cpg_{v12,gpu}_{mean,knn}, grimage2_{mean,knn}, dunedinpace_{mean,knn}, grimage2_dunedinpace_{mean,knn}); mean fill
    only in full (the five _mean models). Every other model keeps M's role label (M.role) and stays outside the decision:
    mgpt2_combined_depthaware, *_knndenoise, *_cohortknn, *_factor_imputed and *_factor_denoised. M recorded the depth-aware
    identity as holding (private_wcheck.json, identity_holds true); stage prep stops otherwise.
  Decision: L's rule on H, evaluated at depth_10. C* = the decision comparator with the largest H at depth_10 (identity fixed
    at the point estimate). PASS iff (H_mgpt2_combined - H_C*) / H_C* >= 0.10 and the lower end of the 95% interval of
    H_mgpt2_combined - H_C* is > 0; otherwise FAIL. If H_C* <= 0 this is stated, no ratio is reported and the rule is not
    applicable. Following L, contrast (a) uses that C* in every scenario (its _mean counterpart in full), and the rule rows give
    each scenario against its own best decision comparator; only the depth_10 row is the decision.
  Bootstrap: L's procedure and seed, unchanged: paired bootstrap of participants, 2,000 draws, seed 20261030, one shared
    resample per draw, I = default_rng(20261030).integers(0, 9866, size=(2000, 9866)), w_b = bincount(I[b], minlength=9866)
    indexed by the frame-cache order; preparation, decile membership, C* and each endpoint's own best comparator fixed in the
    draws; percentile 95% intervals; bootstrap P as in J. These are L's weights (CHECKS 7), so full reproduces L's intervals.
  KM tables: mgpt2_combined, mgpt2_combined_depthaware, the depth_10 C* (its _mean counterpart in full) and GrimAge2 with kNN
    fill (grimage2_knn; grimage2_mean in full), duplicates removed, at depth_10, depth_30 and full, every endpoint plus the mean
    over endpoints, on L's yearly grid. C* is known only from the depth_10 point estimates, so stage boot computes the KM
    aggregates for mgpt2_combined, mgpt2_combined_depthaware, the GrimAge2 reference and every decision comparator of those
    scenarios, and stage analyse releases only the models above. As in L for its two KM models, the abs_risk_10y cells of
    every released KM model follow the 10-year row of the KM table.
  Labels: every noise scenario is labelled "simulated measurement noise on empirical profiles" (with its depth or SD) in every
    table and in the README; full is "complete measured profiles (reference, not a decision)".

CHECKS (L's, adapted, plus two; written to O_checks before any result is used; stage boot refuses to run unless all pass).
Tolerances stated here, before running.
  (1) M's stored pooled within-fold C-indices (M_cindex_per_endpoint_2026-11-02.csv, column c_index) reproduce from the stored
      risks within 1e-6 with J.sksurv_pooled, for every O model and age_sex, 8 endpoints, 9 scenarios (L's check 1 with M's
      table). Also at full: M's full risks of L's 8 models and age_sex equal J's stored full risks (J.assemble) exactly (maximum
      absolute difference 0), and J's stored full C-indices reproduce from M's full risks within 1e-6 (L's check 1 at full).
  (2) Identifiers: 9,866 unique participants per endpoint, the same set as the frame-cache training labels; 5 folds each.
  (3) Preparation, every scenario: in every estimable fold |mean(x)| <= 1e-9, |SD(x) - 1| <= 1e-9, |corr(x, age)| and
      |corr(x, sex)| <= 1e-9. Positive control: the age_sex risk is not estimable in all 40 endpoint-folds of every scenario.
      Also at full: the prepared risks and decile groups of L's 8 models equal L's stored ones (private_prepared_full.npz):
      groups identical, prepared risk within 1e-9.
  (4) Decile membership, every scenario, model, endpoint and estimable fold: top = bottom = floor(n_fold / 10), middle =
      n_fold - 2k, the groups partition the fold and are ordered by x; ties across a boundary counted.
  (5) Cox against statsmodels PHReg(ties="breslow", strata = outer fold), 1e-5 absolute on every coefficient: per scenario
      and endpoint, mgpt2_combined and GrimAge2 with kNN fill (144 pairs), plus 12 further (scenario, model, endpoint)
      triples from default_rng(20261030), each for the continuous and the decile model; frequency weights against expanded
      rows for 3 pairs and 2 draws (depth_10 mgpt2_combined, full grimage2_mean, depth_30 mgpt2_combined; L's endpoints and
      draws), same tolerance. A sampled triple with no estimable fold is recorded as not applicable (L's amendment).
  (6) Kaplan-Meier against sksurv.nonparametric.kaplan_meier_estimator, mgpt2_combined at depth_10, every endpoint, top and
      bottom deciles, yearly grid, 1e-10; frequency weights against expanded rows for 2 draws, 1e-10.
  (7) Bootstrap: every draw sums to 9,866; SHA-256 of the weight matrix recorded and equal to the SHA-256 that L recorded in
      L_checks_2026-10-30.csv.
  (8) Full reproduction (stage reproduce, after analyse and sensitivity, before verify; table O_full_reproduction): O's full
      scenario reproduces L's stored public full results within 1e-9 absolute. O's full inputs, code, seed and draws for L's 8
      models are L's, so the expected difference is 0; 1e-9 allows only for last-digit differences in floating-point
      summation order. Compared: the L_rule_outcome full row; L_summary full rows (every column except label and is_C_star,
      which depend on O's labels and C*); L_metrics full rows of L's 8 models (values, intervals, hazard ratios, counts and
      suppression flags, including the per-endpoint betas); L_contrasts full rows for scenario_best, grimage2_knn and
      endpoint_best, and for C_star when O's full C* equals L's (grimage2_mean); L_km_curves full rows of mgpt2_combined and
      grimage2_mean; L_preparation full rows of L's 8 models and age_sex; L_sensitivity_posthoc full rows; L_results full H;
      and the per-draw private estimates of L's 8 models (private_fit_full__<endpoint>.npz). Public cells released by both
      are compared as released; every cell is also compared through the unsuppressed private tables. Text fields
      (comparators, outcomes, flags) must be equal. One difference is expected by design and is allowed only there: if O's
      full C* is one of L's models other than mgpt2_combined and grimage2_mean, it is a released KM model in O but not in
      L, so its abs_risk_10y cells (and the means and contrasts built on them) follow the stricter KM release rule; those
      cells may be suppressed in O where L released them, and their values are compared through the private tables. Stage
      verify refuses to run unless every reproduction row passes.

DISCLOSURE CONTROL: L's, unchanged, applied to every public file and verified by stage verify before anything is committed:
  any aggregate cell with fewer than 5 events or fewer than 5 people at risk is suppressed (rules per metric and the KM
  subtraction and complementary rules as in L's docstring); a mean over endpoints is blank if any of its 8 cells is
  suppressed; a contrast is blank if either cell is; survival and absolute risks rounded to 4 decimals; no participant
  identifier or participant-level value goes to a public file. Unsuppressed aggregates stay in the private directory on
  Chronos. verify also checks that the KM table holds only the pre-specified KM models.

STAGES (resumable; done-markers artifacts/logs/large_gain_O_<stage>_2026-11-08.done; every stage refuses to run unless this
  script, J's, L's and M's scripts are committed unchanged): prep -> checks -> boot -> analyse -> sensitivity -> reproduce ->
  verify. CPU only.
OUTPUTS
  Public: artifacts/large_gain_2026-11-08/O/ (README.md, O_rule_outcome, O_summary, O_metrics, O_contrasts, O_km_curves,
  O_preparation, O_checks, O_sensitivity, O_full_reproduction, O_results, O_disclosure_check). Private, never committed:
  artifacts/large_gain_2026-11-08_private/O/.
Run: python3 large_gain_O_gs_noise_risk_stratification_2026-11-08.py --stage <stage> [--workers N]  (N <= 32)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-08"
OUT = ART / "large_gain_2026-11-08/O"
PRIV = ART / "large_gain_2026-11-08_private/O"
LOGS = ART / "logs"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"
L_SCRIPT = HERE / "large_gain_L_gs_risk_stratification_2026-10-30.py"
M_SCRIPT = HERE / "large_gain_M_gs_measurement_noise_2026-11-02.py"
L_DATE = "2026-10-30"
M_DATE = "2026-11-02"
LOUT = ART / "large_gain_2026-10-30/L"
LPRIV = ART / "large_gain_2026-10-30_private/L"
MOUT = ART / "large_gain_2026-11-02/M"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for _p in (J_SCRIPT, L_SCRIPT, M_SCRIPT):
    if not _p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {_p}")
M = load_module("large_gain_M", M_SCRIPT)      # M reads J
J = M.J
L = load_module("large_gain_L", L_SCRIPT)      # L's numerical functions and constants, used unchanged

SEED = 20261030
N_BOOT = 2000
N = 9866
if (L.SEED, L.N_BOOT, L.N) != (SEED, N_BOOT, N):
    sys.exit("L's bootstrap settings differ from the pre-specified ones")
SCENARIOS = ["depth_10", "depth_5", "depth_30", "gauss_0.05", "gauss_0.10", "depth_2", "depth_1", "gauss_0.02", "full"]
PRIMARY = "depth_10"
KM_SCENARIOS = ["depth_10", "depth_30", "full"]
YEAR, T10, YEARS, ANCHOR, MIN_CELL, CHANCE = L.YEAR, L.T10, L.YEARS, L.ANCHOR, L.MIN_CELL, L.CHANCE
TOL_C, TOL_PREP, TOL_COX, TOL_KM = L.TOL_C, L.TOL_PREP, L.TOL_COX, L.TOL_KM
TOL_REPRO = 1e-9
GROUPS, COUNT_NAMES, ID_PATTERN, ID_COLUMNS = L.GROUPS, L.COUNT_NAMES, L.ID_PATTERN, L.ID_COLUMNS
METRICS, DIRECTION = L.METRICS, L.DIRECTION
prepare, CoxData, cox_newton, design, km_at = L.prepare, L.CoxData, L.cox_newton, L.design, L.km_at
release_row, release_group, release_km, pct, pboot = L.release_row, L.release_group, L.release_km, L.pct, L.pboot
need, write_json, atomic_csv, atomic_npz = L.need, L.write_json, L.atomic_csv, L.atomic_npz
ENDPOINTS = J.ENDPOINTS
OURS = J.OURS
DA = M.DA
DESC = {"depth_10": "read sampling at mean depth 10; primary", "depth_5": "read sampling at mean depth 5",
        "depth_30": "read sampling at mean depth 30", "depth_2": "read sampling at mean depth 2",
        "depth_1": "read sampling at mean depth 1", "gauss_0.02": "Gaussian noise, SD 0.02",
        "gauss_0.05": "Gaussian noise, SD 0.05", "gauss_0.10": "Gaussian noise, SD 0.10"}
LABELS = {s: f"{M.LABEL} ({d})" for s, d in DESC.items()}
LABELS["full"] = M.label("full")
RULE = ("PASS iff (H_MethylGPT - H_best) / H_best >= 0.10 and the lower end of the 95% interval of H_MethylGPT - H_best is > 0; "
        "H = mean over the 8 endpoints of the log hazard ratio per SD of the prepared risk; best = decision comparator with "
        "the largest H in the scenario (the decision is depth_10)")
L_FULL = L.l_models("full")[0]
L_KM_FULL = [OURS, "grimage2_mean"]


def done_path(stage):
    return LOGS / f"large_gain_O_{stage}_{DATE}.done"


def done_marker(stage, info=None):
    write_json(done_path(stage), {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})})


def require_done(*stages):
    for s in stages:
        if not done_path(s).exists():
            sys.exit(f"stage {s} has not completed ({done_path(s)} missing)")


def require_committed():
    for p in (Path(__file__).resolve(), J_SCRIPT, L_SCRIPT, M_SCRIPT):
        if subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode or \
                subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(p)]).returncode:
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before any result-producing stage runs")


def o_models(scen):
    """(models, decision comparators) of a scenario, in M's order."""
    ms = [m for m in M.models_for(scen) if m != "age_sex" and m not in J.COMPONENTS and m not in M.DA_COMPONENTS]
    dec = M.decision_for(scen)
    return ms, [m for m in ms if m in dec]


def grimage_ref(scen):
    return "grimage2_mean" if scen == "full" else "grimage2_knn"


def in_scenario(name, scen):
    return name.replace("_knn", "_mean") if scen == "full" else name


def km_candidates(scen):
    """Models whose KM aggregates stage boot computes (C* is fixed only in stage analyse)."""
    return list(dict.fromkeys([OURS, DA, grimage_ref(scen)] + o_models(scen)[1]))


def km_models(scen, cstar):
    """Pre-specified released KM models of a scenario."""
    return list(dict.fromkeys([OURS, DA, in_scenario(cstar, scen), grimage_ref(scen)]))


def load_ids():
    mp = J.need_glob(J.CACHE, f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")
    return pd.read_parquet(need(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet"))),
                           columns=["dnam_id"]).dnam_id.astype(str).to_numpy()


# ------------------------------------------------------------------ stage prep (L's, risk source M)
def stage_prep():
    require_committed()
    if not M.da_ok():
        sys.exit("M did not record the depth-aware identity as holding; mgpt2_combined_depthaware is not available")
    t0 = time.time()
    PRIV.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds = J.load_preds()
    ids = load_ids()
    pos = {s: i for i, s in enumerate(ids)}
    arr = {k: [] for k in ("event", "duration", "age", "sex", "fold", "position")}
    for ep in ENDPOINTS:
        pe = preds[ep]
        if len(pe) != N or pe.dnam_id.nunique() != N or set(pe.dnam_id) != set(ids):
            sys.exit(f"{ep}: participants differ from the 9,866 development participants")
        arr["event"].append(pe.event.astype(bool).to_numpy())
        arr["duration"].append(pe.duration.to_numpy(float))
        arr["age"].append(pe.age.to_numpy(float))
        arr["sex"].append(pe.sex.to_numpy(float))
        arr["fold"].append(pe.outer_fold.to_numpy(int))
        arr["position"].append(np.array([pos[s] for s in pe.dnam_id]))
    T = {k: np.stack(v) for k, v in arr.items()}
    atomic_npz(PRIV / "private_endpoint_arrays.npz", **T)
    I = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))
    W = np.stack([np.bincount(I[b], minlength=N) for b in range(N_BOOT)]).astype(np.uint16)
    np.save(PRIV / "private_boot_weights.npy", W)
    rows = []
    for scen in SCENARIOS:
        A = M.assemble(scen, preds)
        models, _ = o_models(scen)
        X = np.zeros((len(models), len(ENDPOINTS), N))
        G = np.zeros((len(models), len(ENDPOINTS), N), np.int8)
        for j, ep in enumerate(ENDPOINTS):
            g = A[A.endpoint == ep]
            if not np.array_equal(g.dnam_id.to_numpy(), preds[ep].dnam_id.to_numpy()):
                sys.exit(f"{scen} {ep}: stored risks are not in the endpoint-table order")
            for k, m in enumerate(["age_sex"] + models):
                x, gr, rec = prepare(g[m].to_numpy(float), T["age"][j], T["sex"][j], T["fold"][j])
                if m != "age_sex":
                    X[k - 1, j], G[k - 1, j] = x, gr
                for r in rec:
                    rows.append({"scenario": scen, "endpoint": ep, "model": m, "role": M.role(m), **r})
        atomic_npz(PRIV / f"private_prepared_{scen}.npz", X=X, G=G, models=np.array(models))
        log(f"{scen}: {len(models)} models prepared")
    atomic_csv(OUT / f"O_preparation_{DATE}.csv", pd.DataFrame(rows))
    done_marker("prep", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage checks
def _phreg(args):
    """One comparison with statsmodels: unweighted, or weighted against expanded rows (draw is not None). L's code."""
    from statsmodels.duration.hazard_regression import PHReg
    scen, model, j, form, draw = args
    T = np.load(PRIV / "private_endpoint_arrays.npz")
    P = np.load(PRIV / f"private_prepared_{scen}.npz")
    k = list(P["models"].astype(str)).index(model)
    x, g = P["X"][k, j], P["G"][k, j]
    ev, du, fold, age, sex = T["event"][j], T["duration"][j], T["fold"][j], T["age"][j], T["sex"][j]
    if (g == -1).all():
        return {"scenario": scen, "model": model, "endpoint": ENDPOINTS[j], "form": form, "draw": draw, "worst": np.nan,
                "note": "not applicable: constant risk in all 5 folds, no coefficient to compare"}
    if form == "continuous":
        Z = design(age, sex, x)
    else:
        if ((g == 2) & ev).sum() == 0 or ((g == 0) & ev).sum() == 0:
            return {"scenario": scen, "model": model, "endpoint": ENDPOINTS[j], "form": form, "draw": draw, "worst": np.nan, "note": "zero events in a group"}
        Z = design(age, sex, (g == 2).astype(float), (g == 0).astype(float))
    cd = CoxData(ev, du, fold)
    if draw is None:
        mine = cox_newton(cd, Z)[0]
        res = PHReg(du, Z, status=ev.astype(int), strata=fold, ties="breslow").fit(disp=0)
    else:
        w = np.load(PRIV / "private_boot_weights.npy", mmap_mode="r")[draw, T["position"][j]].astype(float)
        mine = cox_newton(cd, Z, W=w[None, :], beta0=cox_newton(cd, Z)[0])[0]
        rep = np.repeat(np.arange(len(w)), w.astype(int))
        res = PHReg(du[rep], Z[rep], status=ev[rep].astype(int), strata=fold[rep], ties="breslow").fit(disp=0)
    return {"scenario": scen, "model": model, "endpoint": ENDPOINTS[j], "form": form, "draw": draw,
            "worst": float(np.max(np.abs(mine - res.params))), "beta": float(mine[-1]), "note": ""}


def stage_checks(workers):
    from sksurv.nonparametric import kaplan_meier_estimator
    require_committed()
    require_done("prep")
    t0 = time.time()
    preds = J.load_preds()
    ids = load_ids()
    rows = []

    def add(check, scenario, item, stored, recomputed, tol, passed=None, note=""):
        diff = abs(float(stored) - float(recomputed)) if isinstance(stored, (int, float, np.floating)) and \
            isinstance(recomputed, (int, float, np.floating)) else ""
        ok = passed if passed is not None else (diff != "" and diff <= tol)
        rows.append({"check": check, "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "abs_diff": diff, "tolerance": tol, "pass": bool(ok), "note": note})

    T = np.load(PRIV / "private_endpoint_arrays.npz")
    # (1) M's C-indices from the stored risks; at full also J's risks and J's C-indices
    mc = pd.read_csv(need(MOUT / f"M_cindex_per_endpoint_{M_DATE}.csv")).set_index(["scenario", "endpoint", "model"]).sort_index()
    jc = pd.read_csv(need(L.JDIR / "J_cindex_per_endpoint_2026-10-28.csv")).set_index(["scenario", "endpoint", "model"]).sort_index()
    for scen in SCENARIOS:
        A = M.assemble(scen, preds)
        models, _ = o_models(scen)
        for j, ep in enumerate(ENDPOINTS):
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            folds = [np.flatnonzero(g.outer_fold.to_numpy() == f) for f in sorted(g.outer_fold.unique())]
            for m in ["age_sex"] + models:
                c = float(J.sksurv_pooled(ev, du, g[m].to_numpy(float), folds))
                add("1 M C-index reproduces from the stored risks", scen, f"{ep}:{m}", float(mc.loc[(scen, ep, m), "c_index"]), c, TOL_C)
                if scen == "full" and (m == "age_sex" or m in L_FULL):
                    add("1 J full C-index reproduces from M's full risks", scen, f"{ep}:{m}", float(jc.loc[("full", ep, m), "c_index"]), c, TOL_C)
        if scen == "full":
            JA = J.assemble("full", preds)
            same = all(np.array_equal(A[c].astype(str).to_numpy(), JA[c].astype(str).to_numpy()) for c in ("dnam_id", "endpoint")) and \
                np.array_equal(A.outer_fold.to_numpy(), JA.outer_fold.to_numpy())
            for m in ["age_sex"] + L_FULL:
                d = float(np.max(np.abs(A[m].to_numpy(float) - JA[m].to_numpy(float)))) if same else np.inf
                add("1 M full risks equal J's stored full risks", scen, m, 0.0, d, 0, note="max abs difference over 8 x 9,866 rows")
        log(f"check 1 {scen} done ({time.time() - t0:.0f} s)")
    # (2) identifiers
    for j, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        add("2 participants", "all", ep, f"{N} unique, frame-cache training labels, 5 folds",
            f"{pe.dnam_id.nunique()} unique, {pe.outer_fold.nunique()} folds", "",
            passed=len(pe) == N and pe.dnam_id.nunique() == N and set(pe.dnam_id) == set(ids) and pe.outer_fold.nunique() == 5)
    # (3) preparation, (4) deciles
    prep = pd.read_csv(need(OUT / f"O_preparation_{DATE}.csv"))
    for scen in SCENARIOS:
        pc = prep[(prep.scenario == scen) & (prep.model == "age_sex")]
        add("3 positive control: age_sex risk not estimable", scen, "endpoint-folds flagged", 40, int((~pc.estimable).sum()), 0,
            note=f"largest residual_to_raw_sd {pc.residual_to_raw_sd.max():.3g}")
        ps = prep[(prep.scenario == scen) & (prep.model != "age_sex")]
        add("3 not-estimable folds recorded", scen, "model-endpoint-folds not estimable", "", int((~ps.estimable).sum()), "", passed=True,
            note=f"of {len(ps)}; smallest estimable residual_to_raw_sd {ps[ps.estimable].residual_to_raw_sd.min():.3g}; "
                 f"largest not-estimable {ps[~ps.estimable].residual_to_raw_sd.max() if (~ps.estimable).any() else float('nan'):.3g}")
        P = np.load(PRIV / f"private_prepared_{scen}.npz")
        X, G = P["X"], P["G"]
        wmean = wsd = wage = wsex = 0.0
        bad = ties = 0
        for k in range(X.shape[0]):
            for j in range(len(ENDPOINTS)):
                for f in np.unique(T["fold"][j]):
                    s = T["fold"][j] == f
                    gs = G[k, j][s]
                    if (gs == -1).all():
                        continue
                    z = X[k, j][s]
                    wmean = max(wmean, abs(z.mean()))
                    wsd = max(wsd, abs(z.std() - 1))
                    wage = max(wage, abs(np.corrcoef(z, T["age"][j][s])[0, 1]))
                    wsex = max(wsex, abs(np.corrcoef(z, T["sex"][j][s])[0, 1]))
                    q = int(s.sum()) // 10
                    t_, m_, b_ = z[gs == 2], z[gs == 1], z[gs == 0]
                    if not (len(t_) == q and len(b_) == q and len(m_) == s.sum() - 2 * q and (gs >= 0).all()
                            and b_.max() <= m_.min() and m_.max() <= t_.min()):
                        bad += 1
                    ties += int(b_.max() == m_.min()) + int(m_.max() == t_.min())
        add("3 prepared risk", scen, "max |mean| in a fold", 0.0, wmean, TOL_PREP)
        add("3 prepared risk", scen, "max |SD - 1| in a fold", 0.0, wsd, TOL_PREP)
        add("3 prepared risk", scen, "max |corr with age| in a fold", 0.0, wage, TOL_PREP)
        add("3 prepared risk", scen, "max |corr with sex| in a fold", 0.0, wsex, TOL_PREP)
        add("4 decile membership counts", scen, "estimable folds with wrong counts or order", 0, bad, 0, note=f"ties across a decile boundary: {ties}")
    Lp = np.load(need(LPRIV / "private_prepared_full.npz"))
    Op = np.load(PRIV / "private_prepared_full.npz")
    lm, om = list(Lp["models"].astype(str)), list(Op["models"].astype(str))
    for m in lm:
        a, b = lm.index(m), om.index(m)
        add("3 full prepared risk equals L's", "full", m, 0.0, float(np.max(np.abs(Lp["X"][a] - Op["X"][b]))), TOL_REPRO)
        add("3 full decile groups equal L's", "full", m, 0, int((Lp["G"][a] != Op["G"][b]).sum()), 0, note="participants with a different group")
    log(f"checks 2-4 done ({time.time() - t0:.0f} s)")
    # (5) Cox against PHReg
    pairs = [(s, m, j) for s in SCENARIOS for j in range(len(ENDPOINTS)) for m in (OURS, grimage_ref(s))]
    rest = [(s, m, j) for s in SCENARIOS for m in o_models(s)[0] for j in range(len(ENDPOINTS)) if (s, m, j) not in pairs]
    pick = np.random.default_rng(SEED).choice(len(rest), 12, replace=False)
    pairs += [rest[i] for i in pick]
    tasks = [(s, m, j, form, None) for (s, m, j) in pairs for form in ("continuous", "decile")]
    tasks += [("depth_10", OURS, 0, "continuous", 0), ("depth_10", OURS, 4, "decile", 1), ("full", grimage_ref("full"), 7, "continuous", 0),
              ("full", grimage_ref("full"), 7, "decile", 1), ("depth_30", OURS, 5, "continuous", 1), ("depth_30", OURS, 2, "decile", 0)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_phreg, tasks))
    for r in res:
        name = "5 Cox equals PHReg" if r["draw"] is None else "5 weighted Cox equals PHReg on expanded rows"
        if not np.isfinite(r["worst"]):
            add(name, r["scenario"], f"{r['endpoint']}:{r['model']}:{r['form']}", "", "", TOL_COX, passed=True, note=r["note"])
        else:
            add(name, r["scenario"], f"{r['endpoint']}:{r['model']}:{r['form']}" + ("" if r["draw"] is None else f":draw{r['draw']}"),
                0.0, r["worst"], TOL_COX, note="max abs coefficient difference")
    log(f"check 5 done ({time.time() - t0:.0f} s)")
    # (6) Kaplan-Meier
    P = np.load(PRIV / f"private_prepared_{PRIMARY}.npz")
    k = list(P["models"].astype(str)).index(OURS)
    days = YEARS * YEAR
    Wall = np.load(PRIV / "private_boot_weights.npy", mmap_mode="r")
    for j, ep in enumerate(ENDPOINTS):
        ev, du = T["event"][j], T["duration"][j]
        for gv, gn in ((2, "top_decile"), (0, "bottom_decile")):
            s = P["G"][k, j] == gv
            mine = km_at(du[s], ev[s], days)[0]
            tt, ss = kaplan_meier_estimator(ev[s], du[s])
            ref = np.concatenate([[1.0], ss])[np.searchsorted(tt, days, side="right")]
            ok = np.isfinite(mine)
            add("6 Kaplan-Meier equals sksurv", PRIMARY, f"{ep}:{gn}", 0.0, float(np.max(np.abs(mine[ok] - ref[ok]))), TOL_KM)
            for b in (0, 1):
                w = Wall[b, T["position"][j]][s].astype(int)
                rep = np.repeat(np.arange(int(s.sum())), w)
                a = km_at(du[s], ev[s], days, W=w[None, :].astype(float))[0]
                c = km_at(du[s][rep], ev[s][rep], days)[0]
                fin = np.isfinite(a) & np.isfinite(c)
                add("6 weighted Kaplan-Meier equals expanded rows", PRIMARY, f"{ep}:{gn}:draw{b}", 0.0,
                    float(np.max(np.abs(a[fin] - c[fin]))), TOL_KM)
    # (7) bootstrap
    W = np.load(PRIV / "private_boot_weights.npy")
    sha = hashlib.sha256(W.tobytes()).hexdigest()
    add("7 bootstrap weights", "all", "every draw sums to 9,866", N, int(W.sum(1).min()), 0,
        passed=bool((W.sum(1) == N).all()), note="sha256 " + sha)
    lc = pd.read_csv(need(LOUT / f"L_checks_{L_DATE}.csv"))
    lsha = str(lc.loc[lc.check == "7 bootstrap weights", "note"].iloc[0]).replace("sha256 ", "").strip()
    add("7 bootstrap weights equal L's", "all", "SHA-256 of the weight matrix", lsha, sha, "", passed=lsha == sha)
    tab = pd.DataFrame(rows)
    atomic_csv(OUT / f"O_checks_{DATE}.csv", tab)
    nf = int((~tab["pass"]).sum())
    log(f"checks: {len(tab)} rows, {nf} failed ({time.time() - t0:.0f} s)")
    if nf:
        print(tab[~tab["pass"]].to_string())
        sys.exit("checks failed; no result is used")
    done_marker("checks", {"rows": len(tab), "failed": 0})


# ------------------------------------------------------------------ stage boot (L's; KM for every candidate KM model)
def task_fit(args):
    scen, j = args
    fo = PRIV / f"private_fit_{scen}__{ENDPOINTS[j]}.npz"
    if fo.exists():
        return str(fo.name)
    t0 = time.time()
    T = np.load(PRIV / "private_endpoint_arrays.npz")
    P = np.load(PRIV / f"private_prepared_{scen}.npz")
    models = list(P["models"].astype(str))
    X, G = P["X"][:, j], P["G"][:, j]
    ev, du, fold, age, sex = T["event"][j], T["duration"][j], T["fold"][j], T["age"][j], T["sex"][j]
    W = np.load(PRIV / "private_boot_weights.npy", mmap_mode="r")[:, T["position"][j]].astype(np.float64)
    cd = CoxData(ev, du, fold)
    nm = len(models)
    nan_p, nan_b = lambda: np.full(nm, np.nan), lambda: np.full((nm, N_BOOT), np.nan)
    out = {k: nan_p() for k in ("beta_pt", "tb_pt", "cap_pt", "rt_pt", "rb_pt")}
    out.update({k: nan_b() for k in ("beta_boot", "tb_boot", "cap_boot", "rt_boot", "rb_boot")})
    counts = np.zeros((nm, len(COUNT_NAMES)))
    for k, m in enumerate(models):
        x, g = X[k], G[k]
        top, bot, est = g == 2, g == 0, g >= 0
        counts[k] = [top.sum(), bot.sum(), (ev & est).sum(), (ev & top).sum(), (ev & bot).sum(), (ev & top & (du <= T10)).sum(),
                     (ev & bot & (du <= T10)).sum(), (top & (du >= T10)).sum(), (bot & (du >= T10)).sum(),
                     len(np.unique(fold[est])), est.sum()]
        if not est.any():
            continue
        Z = design(age, sex, x)
        b0 = cox_newton(cd, Z)[0]
        out["beta_pt"][k] = b0[2]
        out["beta_boot"][k] = cox_newton(cd, Z, W=W, beta0=b0)[:, 2]
        wt, wb = W[:, ev & top].sum(1), W[:, ev & bot].sum(1)
        if (ev & top).any() and (ev & bot).any():
            Z = design(age, sex, top.astype(float), bot.astype(float))
            b0 = cox_newton(cd, Z)[0]
            out["tb_pt"][k] = b0[2] - b0[3]
            bb = cox_newton(cd, Z, W=W, beta0=b0, valid=(wt > 0) & (wb > 0))
            out["tb_boot"][k] = bb[:, 2] - bb[:, 3]
        out["cap_pt"][k] = (ev & top).sum() / (ev & est).sum()
        out["cap_boot"][k] = wt / W[:, ev & est].sum(1)
        out["rt_pt"][k] = 1 - km_at(du[top], ev[top], np.array([T10]))[0, 0]
        out["rb_pt"][k] = 1 - km_at(du[bot], ev[bot], np.array([T10]))[0, 0]
        out["rt_boot"][k] = 1 - km_at(du[top], ev[top], np.array([T10]), W=W[:, top])[:, 0]
        out["rb_boot"][k] = 1 - km_at(du[bot], ev[bot], np.array([T10]), W=W[:, bot])[:, 0]
    if scen in KM_SCENARIOS:
        days = YEARS * YEAR
        kms = km_candidates(scen)
        nk = len(kms)
        Sp, Sb = np.full((nk, 3, len(YEARS)), np.nan), np.full((nk, 3, N_BOOT, len(YEARS)), np.nan)
        atr, cum, tot, ng = np.zeros((nk, 3, len(YEARS))), np.zeros((nk, 3, len(YEARS))), np.zeros((nk, 3)), np.zeros((nk, 3))
        for a, m in enumerate(kms):
            g = G[models.index(m)]
            for b, gv in enumerate((2, 1, 0)):
                s = g == gv
                if not s.any():
                    continue
                Sp[a, b] = km_at(du[s], ev[s], days)[0]
                Sb[a, b] = km_at(du[s], ev[s], days, W=W[:, s])
                atr[a, b] = [(du[s] >= d).sum() for d in days]
                cum[a, b] = [(ev[s] & (du[s] <= d)).sum() for d in days]
                tot[a, b], ng[a, b] = ev[s].sum(), s.sum()
        out.update(km_S=Sp, km_boot=Sb, km_atrisk=atr, km_cum=cum, km_total=tot, km_n=ng, km_models=np.array(kms))
    atomic_npz(fo, counts=counts, models=np.array(models), **out)
    log(f"{scen} {ENDPOINTS[j]}: {nm} models, point and {N_BOOT} draws ({time.time() - t0:.0f} s)")
    return str(fo.name)


def stage_boot(workers):
    require_committed()
    require_done("checks")
    t0 = time.time()
    tasks = [(s, j) for s in SCENARIOS for j in range(len(ENDPOINTS))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(task_fit, tasks))
    done_marker("boot", {"draws": N_BOOT, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage analyse
def load_fit(scen):
    fits = [np.load(need(PRIV / f"private_fit_{scen}__{ep}.npz")) for ep in ENDPOINTS]
    F = {"models": list(fits[0]["models"].astype(str))}
    for key in ("beta", "tb", "cap", "rt", "rb"):
        F[f"{key}_pt"] = np.stack([f[f"{key}_pt"] for f in fits], 1)        # (m, e)
        F[f"{key}_boot"] = np.stack([f[f"{key}_boot"] for f in fits], 1)    # (m, e, B)
    F["counts"] = np.stack([f["counts"] for f in fits], 1)                  # (m, e, c)
    if scen in KM_SCENARIOS:
        F["km_models"] = list(fits[0]["km_models"].astype(str))
        for key in ("km_S", "km_boot", "km_atrisk", "km_cum", "km_total", "km_n"):
            F[key] = np.stack([f[key] for f in fits], 0)                    # (e, candidate model, group, ...)
    return F


def metric_values(F):
    return {"log_hr_per_sd": (F["beta_pt"], F["beta_boot"]),
            "log_hr_top_vs_bottom": (F["tb_pt"], F["tb_boot"]),
            "event_capture_top10": (F["cap_pt"], F["cap_boot"]),
            "event_capture_excess_over_10pct": (F["cap_pt"] - CHANCE, F["cap_boot"] - CHANCE),
            "abs_risk_10y_top": (F["rt_pt"], F["rt_boot"]),
            "abs_risk_10y_bottom": (F["rb_pt"], F["rb_boot"]),
            "abs_risk_10y_difference": (F["rt_pt"] - F["rb_pt"], F["rt_boot"] - F["rb_boot"])}


def suppression(F, scen, km_rel, kmm):
    """Boolean (m, e) per metric and a reason per metric. L's rules; kmm = the released KM models of the scenario."""
    c = {n: F["counts"][..., i] for i, n in enumerate(COUNT_NAMES)}
    models = F["models"]
    sup_tb = (c["events_top"] < MIN_CELL) | (c["events_bottom"] < MIN_CELL)
    sup_cap = (c["events_top"] < MIN_CELL) | (c["events_total"] - c["events_top"] < MIN_CELL)
    nm, ne = sup_tb.shape
    sup_rt, sup_rb = np.zeros((nm, ne), bool), np.zeros((nm, ne), bool)
    for k in range(nm):
        for j in range(ne):
            sup_rt[k, j] = not release_row(c["events_10y_top"][k, j], c["at_risk_10y_top"][k, j], c["events_top"][k, j])
            sup_rb[k, j] = not release_row(c["events_10y_bottom"][k, j], c["at_risk_10y_bottom"][k, j], c["events_bottom"][k, j])
    if km_rel is not None:
        for a, m in enumerate(kmm):
            k = models.index(m)
            for j in range(ne):
                sup_rt[k, j] = not km_rel[j, a, 0, ANCHOR]
                sup_rb[k, j] = not km_rel[j, a, 2, ANCHOR]
    sup = {"log_hr_per_sd": np.zeros((nm, ne), bool), "log_hr_top_vs_bottom": sup_tb, "event_capture_top10": sup_cap,
           "event_capture_excess_over_10pct": sup_cap, "abs_risk_10y_top": sup_rt, "abs_risk_10y_bottom": sup_rb,
           "abs_risk_10y_difference": sup_rt | sup_rb}
    allnot = c["n_folds_estimable"] == 0
    for metric in sup:
        sup[metric] = sup[metric] & ~allnot
    why = {"log_hr_per_sd": "", "log_hr_top_vs_bottom": "fewer than 5 events in the top or bottom decile",
           "event_capture_top10": "fewer than 5 events in or outside the top decile",
           "event_capture_excess_over_10pct": "fewer than 5 events in or outside the top decile",
           "abs_risk_10y_top": "fewer than 5 events or people at risk at 10 years, or a subtraction would reveal such a cell",
           "abs_risk_10y_bottom": "fewer than 5 events or people at risk at 10 years, or a subtraction would reveal such a cell",
           "abs_risk_10y_difference": "an input cell is suppressed"}
    return sup, why


def stage_analyse():
    require_committed()
    require_done("boot")
    t0 = time.time()
    FITS, VALS = {}, {}
    for scen in SCENARIOS:
        FITS[scen] = load_fit(scen)
        VALS[scen] = metric_values(FITS[scen])
    # H, C*, rule
    Hp, Hb, Hest, Hnot, best = {}, {}, {}, {}, {}
    for scen in SCENARIOS:
        F = FITS[scen]
        b, bb = F["beta_pt"], F["beta_boot"]
        allnot = ~np.isfinite(b)
        Hp[scen] = np.where(allnot, 0.0, b).mean(1)
        Hb[scen] = np.where(allnot[..., None], 0.0, bb).mean(1)
        Hest[scen] = np.nanmean(b, 1)
        Hnot[scen] = allnot.sum(1)
        _, dec = o_models(scen)
        ms = F["models"]
        best[scen] = max(dec, key=lambda m: Hp[scen][ms.index(m)])
    cstar = best[PRIMARY]
    # KM release (pre-specified KM models, which need C*) and suppression
    KMM, KMREL, SUP = {}, {}, {}
    for scen in SCENARIOS:
        F = FITS[scen]
        rel = None
        if scen in KM_SCENARIOS:
            KMM[scen] = km_models(scen, cstar)
            ci = [F["km_models"].index(m) for m in KMM[scen]]
            rel = np.stack([np.stack([release_km(F["km_cum"][j, c], F["km_atrisk"][j, c], F["km_total"][j, c]) for c in ci])
                            for j in range(len(ENDPOINTS))])     # (e, released KM model, group, Y)
        KMREL[scen] = rel
        SUP[scen] = suppression(F, scen, rel, KMM.get(scen))
    summ, rules = [], []
    res = {"settings": {"seed": SEED, "draws": N_BOOT, "rule": RULE, "primary": PRIMARY, "C_star": cstar, "minimum_cell": MIN_CELL,
                        "label": M.LABEL, "km_models": {s: KMM[s] for s in KM_SCENARIOS}}, "scenarios": {}}
    for scen in SCENARIOS:
        ms = FITS[scen]["models"]
        io = ms.index(OURS)
        _, dec = o_models(scen)
        P, B = Hp[scen], Hb[scen]
        for k, m in enumerate(ms):
            if m == OURS:
                continue
            d = B[io] - B[k]
            lo, hi, _ = pct(d)
            rel = (P[io] - P[k]) / P[k] if P[k] > 0 else np.nan
            outcome = "not applicable (H <= 0)" if not P[k] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
            summ.append({"scenario": scen, "label": LABELS[scen], "model": m, "role": M.role(m),
                         "is_best_decision_comparator": m == best[scen], "is_C_star": m == in_scenario(cstar, scen),
                         "H_model": P[k], "H_model_ci95_low": pct(B[k])[0], "H_model_ci95_high": pct(B[k])[1],
                         "exp_H_model": float(np.exp(P[k])), "H_mgpt2_combined": P[io], "difference": P[io] - P[k],
                         "ci95_low": lo, "ci95_high": hi, "p_boot": pboot(d), "relative_gain": rel,
                         "rule_against_this_model": outcome, "n_endpoints_not_estimable": int(Hnot[scen][k]),
                         "H_over_estimable_endpoints": Hest[scen][k] if Hnot[scen][k] else np.nan,
                         "n_folds_not_estimable": int(40 - FITS[scen]["counts"][k, :, COUNT_NAMES.index("n_folds_estimable")].sum())})
        kb = ms.index(best[scen])
        d = B[io] - B[kb]
        lo, hi, _ = pct(d)
        rel = (P[io] - P[kb]) / P[kb] if P[kb] > 0 else np.nan
        outcome = "not applicable (H_best <= 0)" if not P[kb] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
        rules.append({"analysis": "O", "scenario": scen, "primary": scen == PRIMARY, "label": LABELS[scen],
                      "metric": "mean over 8 endpoints of the log hazard ratio per SD of the prepared risk",
                      "methylgpt": OURS, "H_methylgpt": P[io], "H_methylgpt_ci95_low": pct(B[io])[0],
                      "H_methylgpt_ci95_high": pct(B[io])[1], "best_comparator": best[scen], "H_best": P[kb],
                      "H_best_ci95_low": pct(B[kb])[0], "H_best_ci95_high": pct(B[kb])[1], "relative_gain": rel,
                      "difference": P[io] - P[kb], "ci95_low": lo, "ci95_high": hi,
                      "interval": f"paired bootstrap, one shared participant resample per draw, preparation fixed, {N_BOOT} draws, seed {SEED}",
                      "p_boot": pboot(d), "rule": RULE, "outcome": outcome})
        res["scenarios"][scen] = {"H": {m: float(P[k]) for k, m in enumerate(ms)}, "best": best[scen], "relative_gain": rel,
                                  "difference": float(P[io] - P[kb]), "ci95": [lo, hi], "outcome": outcome}
        log(f"{scen}: H {OURS} {P[io]:.4f}; best {best[scen]} {P[kb]:.4f}; gain {rel:+.1%}; diff {P[io] - P[kb]:+.4f} [{lo:.4f}, {hi:.4f}] -> {outcome}")

    # metric tables (all rows computed; the public copy blanks suppressed cells)
    mrows, crows = [], []
    for scen in SCENARIOS:
        F = FITS[scen]
        ms = F["models"]
        io = ms.index(OURS)
        _, dec = o_models(scen)
        c = {n: F["counts"][..., i] for i, n in enumerate(COUNT_NAMES)}
        sup, why = SUP[scen]
        for metric in METRICS:
            P, B = VALS[scen][metric]
            S = sup[metric]
            for k, m in enumerate(ms):
                for j, ep in enumerate(ENDPOINTS):
                    lo, hi, nd = pct(B[k, j])
                    row = {"scenario": scen, "label": LABELS[scen], "endpoint": ep, "model": m, "role": M.role(m), "metric": metric,
                           "value": P[k, j], "ci95_low": lo, "ci95_high": hi, "n_draws_estimable": nd,
                           "events_total": c["events_total"][k, j], "events_top": np.nan, "events_bottom": np.nan,
                           "events_by_10y": np.nan, "at_risk_10y": np.nan, "chance_exact": np.nan,
                           "n_folds_estimable": c["n_folds_estimable"][k, j], "n_endpoints_suppressed": int(S[k, j]),
                           "suppressed": bool(S[k, j]), "reason": why[metric] if S[k, j] else ("not estimable" if not np.isfinite(P[k, j]) else "")}
                    if metric == "log_hr_top_vs_bottom":
                        row.update(events_top=c["events_top"][k, j], events_bottom=c["events_bottom"][k, j])
                    elif metric.startswith("event_capture"):
                        row.update(events_top=c["events_top"][k, j],
                                   chance_exact=c["n_top"][k, j] / c["n_estimable"][k, j] if c["n_estimable"][k, j] else np.nan)
                    elif metric == "abs_risk_10y_top":
                        row.update(events_by_10y=c["events_10y_top"][k, j], at_risk_10y=c["at_risk_10y_top"][k, j])
                    elif metric == "abs_risk_10y_bottom":
                        row.update(events_by_10y=c["events_10y_bottom"][k, j], at_risk_10y=c["at_risk_10y_bottom"][k, j])
                    mrows.append(row)
                bm = B[k].mean(0)
                lo, hi, nd = pct(bm)
                nsup = int(S[k].sum())
                mrows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "model": m, "role": M.role(m),
                              "metric": metric, "value": float(P[k].mean()), "ci95_low": lo, "ci95_high": hi, "n_draws_estimable": nd,
                              "n_endpoints_suppressed": nsup, "suppressed": nsup > 0,
                              "reason": (f"{nsup} of 8 per-endpoint cells suppressed" if nsup else ("not estimable" if not np.isfinite(P[k]).all() else ""))})

            def targets(j):
                """(contrast, comparator) of endpoint j for this metric and scenario."""
                sgn = DIRECTION[metric]
                vals = [(sgn * P[ms.index(d), j], d) for d in dec if np.isfinite(P[ms.index(d), j])]
                return [("C_star", in_scenario(cstar, scen)), ("scenario_best", best[scen]), ("grimage2_knn", grimage_ref(scen)),
                        ("endpoint_best", max(vals)[1] if vals else None)]

            per = {}
            for j, ep in enumerate(ENDPOINTS):
                for contrast, cmp_ in targets(j):
                    per.setdefault(contrast, []).append(cmp_)
                    if cmp_ is None:
                        crows.append({"scenario": scen, "label": LABELS[scen], "endpoint": ep, "metric": metric, "contrast": contrast,
                                      "suppressed": False, "reason": "no comparator estimable"})
                        continue
                    k = ms.index(cmp_)
                    d = B[io, j] - B[k, j]
                    lo, hi, nd = pct(d)
                    s_ = bool(S[io, j] or S[k, j])
                    dp = P[io, j] - P[k, j]
                    crows.append({"scenario": scen, "label": LABELS[scen], "endpoint": ep, "metric": metric, "contrast": contrast,
                                  "comparator": cmp_, "value_methylgpt": P[io, j], "value_comparator": P[k, j], "difference": dp,
                                  "ci95_low": lo, "ci95_high": hi, "p_boot": pboot(d), "n_draws_estimable": nd,
                                  "relative_gain": dp / P[k, j] if P[k, j] > 0 else np.nan, "n_endpoints_suppressed": int(s_),
                                  "suppressed": s_, "reason": "a cell of the contrast is suppressed" if s_ else ("not estimable" if not np.isfinite(dp) else "")})
            for contrast, cmps in per.items():
                if any(c_ is None for c_ in cmps):
                    crows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "metric": metric,
                                  "contrast": contrast, "suppressed": False, "reason": "no comparator estimable at an endpoint"})
                    continue
                ks = [ms.index(c_) for c_ in cmps]
                jr = np.arange(len(ENDPOINTS))
                vo, vc = P[io].mean(), P[ks, jr].mean()
                d = B[io].mean(0) - B[ks, jr].mean(0)
                lo, hi, nd = pct(d)
                nsup = int((S[io] | S[ks, jr]).sum())
                crows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "metric": metric, "contrast": contrast,
                              "comparator": cmps[0] if len(set(cmps)) == 1 else "per-endpoint best", "value_methylgpt": vo, "value_comparator": vc,
                              "difference": vo - vc, "ci95_low": lo, "ci95_high": hi, "p_boot": pboot(d), "n_draws_estimable": nd,
                              "relative_gain": (vo - vc) / vc if vc > 0 else np.nan, "n_endpoints_suppressed": nsup, "suppressed": nsup > 0,
                              "reason": f"{nsup} of 8 per-endpoint contrasts suppressed" if nsup else ("not estimable" if not np.isfinite(vo - vc) else "")})
    mdf, cdf = pd.DataFrame(mrows), pd.DataFrame(crows)
    for col in ("hr", "hr_ci95_low", "hr_ci95_high"):
        src = {"hr": "value", "hr_ci95_low": "ci95_low", "hr_ci95_high": "ci95_high"}[col]
        mdf[col] = np.where(mdf.metric.str.startswith("log_hr"), np.exp(mdf[src].to_numpy(float)), np.nan)
    cdf["hr_ratio"] = np.where(cdf.metric.str.startswith("log_hr"), np.exp(cdf["difference"].to_numpy(float)), np.nan)

    # KM table: the pre-specified released KM models only
    krows = []
    for scen in KM_SCENARIOS:
        F = FITS[scen]
        rel = KMREL[scen]
        for a, m in enumerate(KMM[scen]):
            cm = F["km_models"].index(m)
            for b, gn in enumerate(GROUPS):
                for y in range(len(YEARS)):
                    for j, ep in enumerate(ENDPOINTS):
                        lo, hi, nd = pct(F["km_boot"][j, cm, b, :, y])
                        ok = bool(rel[j, a, b, y])
                        krows.append({"scenario": scen, "label": LABELS[scen], "endpoint": ep, "model": m, "role": M.role(m), "group": gn,
                                      "year": int(YEARS[y]), "n_group": F["km_n"][j, cm, b], "survival": F["km_S"][j, cm, b, y],
                                      "survival_ci95_low": lo, "survival_ci95_high": hi, "n_at_risk": F["km_atrisk"][j, cm, b, y],
                                      "cum_events": F["km_cum"][j, cm, b, y], "suppressed": not ok,
                                      "reason": "" if ok else "fewer than 5 events or people at risk, or a subtraction would reveal such a cell"})
                    mb = F["km_boot"][:, cm, b, :, y].mean(0)
                    lo, hi, nd = pct(mb)
                    nsup = int((~rel[:, a, b, y]).sum())
                    krows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "model": m, "role": M.role(m),
                                  "group": gn, "year": int(YEARS[y]), "survival": float(F["km_S"][:, cm, b, y].mean()), "survival_ci95_low": lo,
                                  "survival_ci95_high": hi, "suppressed": nsup > 0,
                                  "reason": f"{nsup} of 8 per-endpoint rows suppressed" if nsup else ""})
    kdf = pd.DataFrame(krows)

    # private: every aggregate; public: suppressed cells blank
    atomic_csv(PRIV / "private_metrics_unsuppressed.csv", mdf)
    atomic_csv(PRIV / "private_contrasts_unsuppressed.csv", cdf)
    atomic_csv(PRIV / "private_km_unsuppressed.csv", kdf)
    pm, pc, pk = mdf.copy(), cdf.copy(), kdf.copy()
    blank = ["value", "ci95_low", "ci95_high", "hr", "hr_ci95_low", "hr_ci95_high", "events_top", "events_bottom", "events_by_10y", "at_risk_10y"]
    pm.loc[pm.suppressed, blank] = np.nan
    for col in ("events_top", "events_bottom", "events_by_10y", "at_risk_10y", "events_total"):
        pm.loc[pm[col] < MIN_CELL, col] = np.nan
    pc.loc[pc.suppressed, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high", "p_boot", "relative_gain", "hr_ratio"]] = np.nan
    pk.loc[pk.suppressed, ["survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"]] = np.nan
    risk_rows = pm.metric.str.startswith("abs_risk")
    pm.loc[risk_rows, ["value", "ci95_low", "ci95_high"]] = pm.loc[risk_rows, ["value", "ci95_low", "ci95_high"]].round(4)
    risk_rows = pc.metric.str.startswith("abs_risk")
    pc.loc[risk_rows, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high"]] = \
        pc.loc[risk_rows, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high"]].round(4)
    pk[["survival", "survival_ci95_low", "survival_ci95_high"]] = pk[["survival", "survival_ci95_low", "survival_ci95_high"]].round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / f"O_metrics_{DATE}.csv", pm)
    atomic_csv(OUT / f"O_contrasts_{DATE}.csv", pc)
    atomic_csv(OUT / f"O_km_curves_{DATE}.csv", pk)
    atomic_csv(OUT / f"O_summary_{DATE}.csv", pd.DataFrame(summ))
    atomic_csv(OUT / f"O_rule_outcome_{DATE}.csv", pd.DataFrame(rules))
    write_json(OUT / f"O_results_{DATE}.json", res)
    write_readme(res)
    done_marker("analyse", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage sensitivity (L's; pre-specified here, not part of the decision)
def stage_sensitivity():
    require_committed()
    require_done("analyse")
    rows = []

    def row(scen, reading, dropped, cmp_, Hm, Hc, d):
        lo, hi, _ = pct(d)
        rel = (Hm - Hc) / Hc if Hc > 0 else np.nan
        rows.append({"scenario": scen, "label": LABELS[scen], "reading": reading, "endpoint_left_out": dropped, "comparator": cmp_,
                     "H_methylgpt": Hm, "H_comparator": Hc, "relative_gain": rel, "difference": Hm - Hc, "ci95_low": lo, "ci95_high": hi,
                     "p_boot": pboot(d), "rule_arithmetic": "not applicable" if not Hc > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL"),
                     "status": "pre-specified in O (added in L after its primary result); not part of the decision"})

    for scen in SCENARIOS:
        F = load_fit(scen)
        ms = F["models"]
        dec = o_models(scen)[1]
        io = ms.index(OURS)
        b, bb = np.where(np.isfinite(F["beta_pt"]), F["beta_pt"], 0.0), np.where(np.isfinite(F["beta_boot"]), F["beta_boot"], 0.0)
        nf = F["counts"][..., COUNT_NAMES.index("n_folds_estimable")] / 5
        Pw, Bw = (b * nf).mean(1), (bb * nf[..., None]).mean(1)
        kw = ms.index(max(dec, key=lambda m: Pw[ms.index(m)]))
        row(scen, "fold_weighted", "", ms[kw], Pw[io], Pw[kw], Bw[io] - Bw[kw])
        kb = ms.index(max(dec, key=lambda m: b[ms.index(m)].mean()))
        for j, ep in enumerate(ENDPOINTS):
            keep = [q for q in range(len(ENDPOINTS)) if q != j]
            row(scen, "leave_one_endpoint_out", ep, ms[kb], b[io, keep].mean(), b[kb, keep].mean(), bb[io, keep].mean(0) - bb[kb, keep].mean(0))
    atomic_csv(OUT / f"O_sensitivity_{DATE}.csv", pd.DataFrame(rows))
    done_marker("sensitivity", {"rows": len(rows)})


# ------------------------------------------------------------------ stage reproduce (CHECKS 8: full against L's stored results)
def _norm(df, keys):
    df = df.copy()
    for k in keys:
        df[k] = df[k].fillna("").astype(str)
    return df


def _compare(rows, table, Lpub, Opub, keys, num, txt, Lpriv=None, Opriv=None, exempt=None, note=""):
    """One row per column: L's public cells against O's public cells where both are released; L's private cells against O's
    private cells everywhere (NaN patterns equal); text columns equal; release status equal except on exempt rows."""
    Lpub, Opub = _norm(Lpub, keys), _norm(Opub, keys)
    mp = Lpub.merge(Opub, on=keys, how="left", suffixes=("_L", "_O"), indicator=True)
    miss = int((mp["_merge"] != "both").sum())
    ex = np.zeros(len(mp), bool) if exempt is None else np.asarray(exempt(mp), bool)
    if Lpriv is not None:
        Lpriv, Opriv = _norm(Lpriv, keys), _norm(Opriv, keys)
        mq = Lpriv.merge(Opriv, on=keys, how="left", suffixes=("_L", "_O"), indicator=True)
        miss += int((mq["_merge"] != "both").sum())
    for col in num:
        a, b = mp[f"{col}_L"].to_numpy(float), mp[f"{col}_O"].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        dpub = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        l_only = np.isfinite(a) & ~np.isfinite(b)
        o_only = ~np.isfinite(a) & np.isfinite(b)
        n_l_only, n_l_only_ex, n_o_only = int(l_only.sum()), int((l_only & ex).sum()), int(o_only.sum())
        dpriv, nan_mis, n_priv = 0.0, 0, 0
        if Lpriv is not None:
            c, d = mq[f"{col}_L"].to_numpy(float), mq[f"{col}_O"].to_numpy(float)
            bp = np.isfinite(c) & np.isfinite(d)
            dpriv = float(np.max(np.abs(c[bp] - d[bp]))) if bp.any() else 0.0
            nan_mis, n_priv = int((np.isfinite(c) != np.isfinite(d)).sum()), int(bp.sum())
        ok = miss == 0 and dpub <= TOL_REPRO and dpriv <= TOL_REPRO and nan_mis == 0 and n_o_only == 0 and n_l_only == n_l_only_ex
        if Lpriv is None and n_l_only:
            ok = False
        rows.append({"table": table, "column": col, "rows_L": len(Lpub), "rows_missing_in_O": miss, "n_compared_public": int(both.sum()),
                     "max_abs_diff_public": dpub, "n_released_L_only": n_l_only, "n_released_L_only_by_design": n_l_only_ex,
                     "n_released_O_only": n_o_only, "n_compared_private": n_priv, "max_abs_diff_private": dpriv,
                     "n_nan_pattern_mismatch_private": nan_mis, "n_text_mismatch": "", "tolerance": TOL_REPRO, "pass": bool(ok), "note": note})
    for col in txt:
        a, b = mp[f"{col}_L"], mp[f"{col}_O"]
        if col == "suppressed":
            mis = a.astype(str).str.lower().to_numpy() != b.astype(str).str.lower().to_numpy()
            n_mis, n_mis_ex = int(mis.sum()), int((mis & ex).sum())
            ok = miss == 0 and n_mis == n_mis_ex
            nt = f"{n_mis} ({n_mis_ex} by design)"
        else:
            mis = a.fillna("").astype(str).str.lower().to_numpy() != b.fillna("").astype(str).str.lower().to_numpy()
            ok, nt = miss == 0 and not mis.any(), str(int(mis.sum()))
        rows.append({"table": table, "column": col, "rows_L": len(Lpub), "rows_missing_in_O": miss, "n_compared_public": len(mp),
                     "max_abs_diff_public": "", "n_released_L_only": "", "n_released_L_only_by_design": "", "n_released_O_only": "",
                     "n_compared_private": "", "max_abs_diff_private": "", "n_nan_pattern_mismatch_private": "", "n_text_mismatch": nt,
                     "tolerance": 0, "pass": bool(ok), "note": note})


def stage_reproduce():
    require_committed()
    require_done("analyse", "sensitivity")
    t0 = time.time()
    res = json.loads((OUT / f"O_results_{DATE}.json").read_text())
    cstar = res["settings"]["C_star"]
    lres = json.loads(need(LOUT / f"L_results_{L_DATE}.json").read_text())
    l_cfull, o_cfull = in_scenario(lres["settings"]["C_star"], "full"), in_scenario(cstar, "full")
    extra = [m for m in km_models("full", cstar) if m in L_FULL and m not in L_KM_FULL]
    note_ex = (f"O's full C* {o_cfull} is a released KM model in O only: its abs_risk_10y cells follow the KM release rule"
               if extra else "")
    rows = []
    rd = lambda d, f: pd.read_csv(need(d / f))
    full = lambda df: df[df.scenario == "full"].reset_index(drop=True)
    # rule outcome
    _compare(rows, "rule_outcome", full(rd(LOUT, f"L_rule_outcome_{L_DATE}.csv")), full(rd(OUT, f"O_rule_outcome_{DATE}.csv")), ["scenario"],
             ["H_methylgpt", "H_methylgpt_ci95_low", "H_methylgpt_ci95_high", "H_best", "H_best_ci95_low", "H_best_ci95_high",
              "relative_gain", "difference", "ci95_low", "ci95_high", "p_boot"], ["methylgpt", "best_comparator", "metric", "interval", "outcome"])
    # summary
    ls, os_ = full(rd(LOUT, f"L_summary_{L_DATE}.csv")), full(rd(OUT, f"O_summary_{DATE}.csv"))
    _compare(rows, "summary", ls, os_, ["model"],
             ["H_model", "H_model_ci95_low", "H_model_ci95_high", "exp_H_model", "H_mgpt2_combined", "difference", "ci95_low", "ci95_high",
              "p_boot", "relative_gain", "n_endpoints_not_estimable", "H_over_estimable_endpoints", "n_folds_not_estimable"],
             ["role", "is_best_decision_comparator", "rule_against_this_model"])
    # metrics (per-endpoint betas included)
    def ex_metrics(mp):
        return mp["model"].isin(extra).to_numpy() & mp["metric"].str.startswith("abs_risk").to_numpy()
    lm = full(rd(LOUT, f"L_metrics_{L_DATE}.csv"))
    om = full(rd(OUT, f"O_metrics_{DATE}.csv"))
    om = om[om.model.isin(L_FULL)].reset_index(drop=True)
    lmp = full(rd(LPRIV, "private_metrics_unsuppressed.csv"))
    omp = full(rd(PRIV, "private_metrics_unsuppressed.csv"))
    omp = omp[omp.model.isin(L_FULL)].reset_index(drop=True)
    num_m = ["value", "ci95_low", "ci95_high", "n_draws_estimable", "events_total", "events_top", "events_bottom", "events_by_10y",
             "at_risk_10y", "chance_exact", "n_folds_estimable", "hr", "hr_ci95_low", "hr_ci95_high"]
    for metric in METRICS:
        _compare(rows, f"metrics:{metric}", lm[lm.metric == metric], om[om.metric == metric], ["endpoint", "model", "metric"], num_m,
                 ["role", "suppressed"], lmp[lmp.metric == metric], omp[omp.metric == metric], ex_metrics, note_ex if metric.startswith("abs_risk") else "")
    # contrasts
    keep = ["scenario_best", "grimage2_knn", "endpoint_best"] + (["C_star"] if o_cfull == l_cfull else [])
    lc_, oc_ = full(rd(LOUT, f"L_contrasts_{L_DATE}.csv")), full(rd(OUT, f"O_contrasts_{DATE}.csv"))
    lcp, ocp = full(rd(LPRIV, "private_contrasts_unsuppressed.csv")), full(rd(PRIV, "private_contrasts_unsuppressed.csv"))
    sel = lambda d: d[d.contrast.isin(keep)].reset_index(drop=True)
    lc_, oc_, lcp, ocp = sel(lc_), sel(oc_), sel(lcp), sel(ocp)
    inv = set(map(tuple, lcp[lcp.comparator.isin(extra)][["metric", "contrast"]].to_numpy())) if extra else set()

    def ex_contrasts(mp):
        risk = mp["metric"].str.startswith("abs_risk").to_numpy()
        hit = np.array([(m, c) in inv for m, c in zip(mp["metric"], mp["contrast"])], bool)
        return risk & hit

    _compare(rows, "contrasts", lc_, oc_, ["endpoint", "metric", "contrast"],
             ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high", "p_boot", "n_draws_estimable", "relative_gain", "hr_ratio"],
             ["comparator", "suppressed"], lcp, ocp, ex_contrasts,
             f"contrasts compared: {', '.join(keep)}" + ("" if o_cfull == l_cfull else f" (C_star differs: L {l_cfull}, O {o_cfull})") +
             (f"; {note_ex}" if extra else ""))
    # KM curves of L's two KM models
    lk, ok_ = full(rd(LOUT, f"L_km_curves_{L_DATE}.csv")), full(rd(OUT, f"O_km_curves_{DATE}.csv"))
    lkp, okp = full(rd(LPRIV, "private_km_unsuppressed.csv")), full(rd(PRIV, "private_km_unsuppressed.csv"))
    ok_, okp = ok_[ok_.model.isin(L_KM_FULL)].reset_index(drop=True), okp[okp.model.isin(L_KM_FULL)].reset_index(drop=True)
    _compare(rows, "km_curves", lk, ok_, ["endpoint", "model", "group", "year"],
             ["n_group", "survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"], ["role", "suppressed"], lkp, okp)
    # preparation
    lp_, op_ = full(rd(LOUT, f"L_preparation_{L_DATE}.csv")), full(rd(OUT, f"O_preparation_{DATE}.csv"))
    op_ = op_[op_.model.isin(["age_sex"] + L_FULL)].reset_index(drop=True)
    _compare(rows, "preparation", lp_, op_, ["endpoint", "model", "outer_fold"],
             ["n_fold", "raw_sd", "residual_sd", "residual_to_raw_sd", "n_top", "n_middle", "n_bottom", "boundary_ties"], ["role", "estimable"])
    # sensitivity
    _compare(rows, "sensitivity", full(rd(LOUT, f"L_sensitivity_posthoc_{L_DATE}.csv")), full(rd(OUT, f"O_sensitivity_{DATE}.csv")),
             ["reading", "endpoint_left_out"], ["H_methylgpt", "H_comparator", "relative_gain", "difference", "ci95_low", "ci95_high", "p_boot"],
             ["comparator", "rule_arithmetic"])
    # results JSON
    lh, oh = lres["scenarios"]["full"]["H"], res["scenarios"]["full"]["H"]
    d = max(abs(float(lh[m]) - float(oh[m])) for m in L_FULL)
    rows.append({"table": "results_json:H", "column": "H", "rows_L": len(L_FULL), "rows_missing_in_O": sum(m not in oh for m in L_FULL),
                 "n_compared_public": len(L_FULL), "max_abs_diff_public": d, "tolerance": TOL_REPRO, "pass": bool(d <= TOL_REPRO), "note": ""})
    # per-draw private estimates of L's 8 models
    agg = {}
    for ep in ENDPOINTS:
        Lf, Of = np.load(need(LPRIV / f"private_fit_full__{ep}.npz")), np.load(PRIV / f"private_fit_full__{ep}.npz")
        lmod, omod = list(Lf["models"].astype(str)), list(Of["models"].astype(str))
        idx = [omod.index(m) for m in lmod]
        okm = list(Of["km_models"].astype(str))
        kidx = [okm.index(m) for m in L_KM_FULL]
        for key in ("beta_pt", "beta_boot", "tb_pt", "tb_boot", "cap_pt", "cap_boot", "rt_pt", "rt_boot", "rb_pt", "rb_boot", "counts",
                    "km_S", "km_boot", "km_atrisk", "km_cum", "km_total", "km_n"):
            a = Lf[key]
            b = Of[key][kidx] if key.startswith("km_") else Of[key][idx]
            fa, fb = np.isfinite(a), np.isfinite(b)
            r = agg.setdefault(key, [0.0, 0, 0])
            r[0] = max(r[0], float(np.max(np.abs(a[fa & fb] - b[fa & fb]))) if (fa & fb).any() else 0.0)
            r[1] += int((fa != fb).sum())
            r[2] += int((fa & fb).sum())
    for key, (dmax, nm_, nc) in agg.items():
        rows.append({"table": "private_per_draw_estimates", "column": key, "rows_L": "", "rows_missing_in_O": 0, "n_compared_private": nc,
                     "max_abs_diff_private": dmax, "n_nan_pattern_mismatch_private": nm_, "tolerance": TOL_REPRO,
                     "pass": bool(dmax <= TOL_REPRO and nm_ == 0), "note": "8 endpoints; L's 8 models; KM arrays for mgpt2_combined and grimage2_mean"})
    tab = pd.DataFrame(rows)
    atomic_csv(OUT / f"O_full_reproduction_{DATE}.csv", tab)
    nf = int((~tab["pass"]).sum())
    log(f"full reproduction: {len(tab)} rows, {nf} failed ({time.time() - t0:.0f} s)")
    if nf:
        print(tab[~tab["pass"]].to_string())
        sys.exit("full reproduction failed")
    done_marker("reproduce", {"rows": len(tab), "failed": 0})


# ------------------------------------------------------------------ stage verify (disclosure control; L's)
def stage_verify():
    require_committed()
    require_done("analyse", "sensitivity", "reproduce")
    bad = []
    ids = load_ids()
    idset = set(map(str, ids))
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != f"O_disclosure_check_{DATE}.json")
    scan = []
    for p in files:
        raw = gzip.open(p, "rt").read() if p.name.endswith(".gz") else p.read_text()
        toks = set(re.split(r"[,\s\"'{}\[\]:;|`()]+", raw))
        header = [c.strip().strip('"') for c in raw.split("\n", 1)[0].split(",")] if p.suffix == ".csv" else []
        rec = {"file": p.name, "bytes": p.stat().st_size, "rows": raw.count("\n"), "n_participant_ids_found": len(toks & idset),
               "n_barcode_pattern_matches": len(ID_PATTERN.findall(raw)), "identifier_column_in_header": bool(set(header) & ID_COLUMNS)}
        scan.append(rec)
        if rec["n_participant_ids_found"] or rec["n_barcode_pattern_matches"] or rec["identifier_column_in_header"]:
            bad.append(f"{p.name}: identifier found")
        if rec["rows"] >= N and p.name not in (f"O_km_curves_{DATE}.csv", f"O_metrics_{DATE}.csv", f"O_contrasts_{DATE}.csv"):
            bad.append(f"{p.name}: as many rows as participants")
    m = pd.read_csv(OUT / f"O_metrics_{DATE}.csv")
    c = pd.read_csv(OUT / f"O_contrasts_{DATE}.csv")
    k = pd.read_csv(OUT / f"O_km_curves_{DATE}.csv")
    pre = pd.read_csv(OUT / f"O_preparation_{DATE}.csv")
    rep = pd.read_csv(OUT / f"O_full_reproduction_{DATE}.csv")
    res = json.loads((OUT / f"O_results_{DATE}.json").read_text())
    cstar = res["settings"]["C_star"]
    summary = {}
    if not rep["pass"].all():
        bad.append("full reproduction has a failing row")
    # small counts
    for name, df, cols in (("metrics", m, ["events_total", "events_top", "events_bottom", "events_by_10y", "at_risk_10y"]),
                           ("km", k, ["n_at_risk", "cum_events"])):
        for col in cols:
            v = df[col].dropna()
            summary[f"{name}.{col}: smallest released value"] = float(v.min()) if len(v) else None
            if len(v) and v.min() < MIN_CELL:
                bad.append(f"{name}.{col}: released value below {MIN_CELL}")
    # the endpoint total minus the events of the estimable folds is 0 or at least the minimum cell
    totals = dict(zip(ENDPOINTS, np.load(PRIV / "private_endpoint_arrays.npz")["event"].sum(1)))
    et = m[(m.endpoint != "mean_over_endpoints") & m.events_total.notna()]
    rem = et.endpoint.map(totals).to_numpy(float) - et.events_total.to_numpy(float)
    summary["fold-level remainder of events_total: smallest nonzero"] = float(rem[rem > 0].min()) if (rem > 0).any() else None
    if ((rem > 0) & (rem < MIN_CELL)).any():
        bad.append("metrics.events_total: endpoint total minus events_total reveals a cell below the minimum")
    # suppressed rows are blank
    if m.loc[m.suppressed, ["value", "ci95_low", "ci95_high", "hr", "events_top", "events_bottom", "events_by_10y", "at_risk_10y"]].notna().any().any():
        bad.append("metrics: suppressed row with a value")
    if c.loc[c.suppressed, ["value_methylgpt", "value_comparator", "difference", "ci95_low", "ci95_high", "relative_gain", "p_boot"]].notna().any().any():
        bad.append("contrasts: suppressed row with a value")
    if k.loc[k.suppressed, ["survival", "survival_ci95_low", "survival_ci95_high", "n_at_risk", "cum_events"]].notna().any().any():
        bad.append("km: suppressed row with a value")
    # means blank whenever an input is suppressed
    ep = m[m.endpoint != "mean_over_endpoints"].groupby(["scenario", "model", "metric"]).suppressed.sum()
    mm = m[m.endpoint == "mean_over_endpoints"].set_index(["scenario", "model", "metric"])
    for key, n in ep.items():
        if (n > 0) and pd.notna(mm.loc[key, "value"]):
            bad.append(f"metrics mean released with a suppressed input: {key}")
    for (s, mt, ct), g in c.groupby(["scenario", "metric", "contrast"]):
        me = g[g.endpoint == "mean_over_endpoints"]
        if g[g.endpoint != "mean_over_endpoints"].suppressed.any() and me.difference.notna().any():
            bad.append(f"contrast mean released with a suppressed input: {s} {mt} {ct}")
    # only the pre-specified KM models are released
    for s in KM_SCENARIOS:
        got = set(k[k.scenario == s].model.unique())
        if got != set(km_models(s, cstar)):
            bad.append(f"km: models released in {s} differ from the pre-specified KM models: {sorted(got)}")
    if set(k.scenario.unique()) != set(KM_SCENARIOS):
        bad.append("km: scenarios differ from the pre-specified KM scenarios")
    # KM subtraction rules on the released rows
    fits = {s: load_fit(s) for s in KM_SCENARIOS}
    ke = k[k.endpoint != "mean_over_endpoints"]
    for (s, e_, mo, gr), g in ke.groupby(["scenario", "endpoint", "model", "group"]):
        g = g.sort_values("year")
        cum = g.cum_events.dropna().to_numpy()
        if len(cum) and (cum[0] < MIN_CELL or (np.diff(cum) < MIN_CELL).any()):
            bad.append(f"km: released rows {MIN_CELL} events apart failed: {s} {e_} {mo} {gr}")
        F = fits[s]
        tot = F["km_total"][ENDPOINTS.index(e_), F["km_models"].index(mo), GROUPS.index(gr)]
        if len(cum) and 0 < tot - cum[-1] < MIN_CELL:
            bad.append(f"km: remainder after the last released row: {s} {e_} {mo} {gr}")
    hid = ke.groupby(["scenario", "endpoint", "model", "year"]).suppressed.sum()
    if (hid == 1).any():
        bad.append("km: a year with exactly one hidden group")
    mk = k[k.endpoint == "mean_over_endpoints"].set_index(["scenario", "model", "group", "year"])
    for key, n in ke.groupby(["scenario", "model", "group", "year"]).suppressed.sum().items():
        if n > 0 and pd.notna(mk.loc[key, "survival"]):
            bad.append(f"km mean released with a suppressed input: {key}")
    # released abs-risk cells of the KM models equal the 10-year KM rows in release status
    for s in KM_SCENARIOS:
        for mo in km_models(s, cstar):
            for mt, gr in (("abs_risk_10y_top", "top_decile"), ("abs_risk_10y_bottom", "bottom_decile")):
                a = m[(m.scenario == s) & (m.model == mo) & (m.metric == mt) & (m.endpoint != "mean_over_endpoints")].set_index("endpoint").value
                b = 1 - ke[(ke.scenario == s) & (ke.model == mo) & (ke.group == gr) & (ke.year == ANCHOR)].set_index("endpoint").survival
                if not (a.notna() == b.reindex(a.index).notna()).all():
                    bad.append(f"abs risk and KM 10-year row differ in release status: {s} {mo} {mt}")
    summary.update({"metrics rows": len(m), "metrics rows suppressed": int(m.suppressed.sum()), "contrast rows": len(c),
                    "contrast rows suppressed": int(c.suppressed.sum()), "km rows": len(k), "km rows suppressed": int(k.suppressed.sum()),
                    "preparation rows": len(pre), "full reproduction rows": len(rep)})
    F = {"files": scan, "summary": summary, "problems": bad, "passed": not bad,
         "method": "every token of every public file compared with the 9,866 development participant IDs; regex for the GS barcode "
                   "pattern; headers scanned for identifier columns; released counts at least 5; suppressed rows blank; means blank when "
                   "an input is suppressed; KM rows at least 5 events apart and from the whole-follow-up total; no KM year with exactly one "
                   "hidden group; only the pre-specified KM models released; full reproduction passed. Tables are per scenario, model, "
                   "endpoint, fold or year; none has one row per participant."}
    write_json(OUT / f"O_disclosure_check_{DATE}.json", F)
    log(f"disclosure check over {len(files)} files: {'passed' if not bad else 'FAILED'}; {len(bad)} problems")
    if bad:
        for b in bad[:30]:
            print(b)
        sys.exit("disclosure check failed; do not commit")
    write_readme(res)
    done_marker("verify", {"files": len(files)})


# ------------------------------------------------------------------ README
def write_readme(res):
    def rows(fn):
        return "-" if fn.endswith(".json") else str(len(pd.read_csv(OUT / fn)))

    def cols(fn):
        return "-" if fn.endswith(".json") else ", ".join(pd.read_csv(OUT / fn, nrows=1).columns)

    desc = {
        f"O_rule_outcome_{DATE}.csv": "One row per scenario: H of MethylGPT with interval, best decision comparator of the scenario with H, relative gain, difference with 95% interval, bootstrap P, rule, outcome. The decision is the depth_10 row.",
        f"O_summary_{DATE}.csv": "Per scenario and model: H with interval, exp(H), difference from MethylGPT with interval and bootstrap P, relative gain, rule arithmetic against this model, best-comparator and C* flags, endpoints and folds not estimable, role (M's role labels; only decision comparators enter the decision).",
        f"O_metrics_{DATE}.csv": "Per scenario, endpoint (8 plus mean_over_endpoints), model and metric: value, interval, hazard ratio where the metric is a log hazard ratio, cell counts, suppression flag and reason. Metrics: " + ", ".join(METRICS) + ".",
        f"O_contrasts_{DATE}.csv": "Per scenario, endpoint (8 plus mean_over_endpoints) and metric: MethylGPT against C* (the depth_10 best on H; contrast C_star), the scenario's best on H (scenario_best), GrimAge2 with kNN fill (grimage2_knn; grimage2_mean in full) and the endpoint's own best decision comparator (endpoint_best); values, difference, interval, bootstrap P, relative gain, suppression flag.",
        f"O_km_curves_{DATE}.csv": "Kaplan-Meier aggregates on the yearly grid 0 to 17 for the top decile, middle 80% and bottom decile: survival with interval, numbers at risk, cumulative events, for mgpt2_combined, mgpt2_combined_depthaware, the depth_10 C* and GrimAge2 with kNN fill (grimage2_mean in full), every endpoint plus the mean over endpoints, at depth_10, depth_30 and full.",
        f"O_preparation_{DATE}.csv": "Per scenario, endpoint, model and outer fold: fold size, SD of the stored risk, SD of its age and sex residual, their ratio, estimable flag, decile sizes, ties across a decile boundary. age_sex is the positive control of the constant-risk test.",
        f"O_checks_{DATE}.csv": "Every pre-specified check before the bootstrap, with reference and recomputed values, tolerance and pass.",
        f"O_sensitivity_{DATE}.csv": "L's sensitivity readings, pre-specified in O, not part of the decision. fold_weighted: beta x (estimable folds / 5), best comparator reselected. leave_one_endpoint_out: H over the other 7 endpoints, comparator fixed at the scenario's best on all 8. Rule arithmetic is for reference only.",
        f"O_full_reproduction_{DATE}.csv": "CHECKS 8: O's full scenario against L's stored full results, per table and column: cells compared, largest absolute difference (public and private), release-status and text mismatches, tolerance 1e-9, pass.",
        f"O_results_{DATE}.json": "Settings, C*, released KM models, and per scenario H of every model, best comparator, gain, difference, interval and outcome.",
        f"O_disclosure_check_{DATE}.json": "Scan of every public file for participant identifiers and verification of the disclosure rules, run before commit.",
    }
    pr = res["scenarios"][PRIMARY]
    gain = "not reported (H of the best comparator <= 0)" if pr["H"][pr["best"]] <= 0 else f"{pr['relative_gain']:+.1%}"
    lines = ["# Large-gain analysis O: GS risk stratification beyond age and sex from M's stored held-out risks", "",
             "Script: `large_gain_O_gs_noise_risk_stratification_2026-11-08.py` (pre-specified design in its docstring). L's design and code applied "
             "unchanged to analysis M's stored held-out risks. No disease model is fitted and no risk is recomputed. Every noise scenario is "
             f"**{M.LABEL}**; full is complete measured profiles. 9,866 GS development participants, 8 endpoints, J's locked outer folds, "
             "M's models and roles.", "",
             "Timing, stated plainly: O was written on 8 Nov 2026, after the results of J, K, L, M and N were known, including M's C-index results "
             "(at depth_10 every decision comparator fell below age+sex, so M's C-index rule gave no outcome). No risk-stratification metric had been "
             "computed on M's noisy risks before O's design was committed. L's metrics were pre-specified on 30 Oct 2026.", "",
             "## Primary result (depth_10)", "",
             "- H = mean over the 8 endpoints of the log hazard ratio per SD of the prepared risk (age and sex residual, standardised within fold; Cox with age and sex, Breslow ties, stratified by outer fold).",
             f"- MethylGPT combined H = {pr['H'][OURS]:.4f}; best decision comparator (C*) {pr['best']} H = {pr['H'][pr['best']]:.4f}.",
             f"- Relative gain {gain}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             "", "## Every scenario (each against its own best decision comparator; only depth_10 is the decision)", "",
             "| Scenario | Label | H MethylGPT | Best comparator | H best | Gain | Difference [95% interval] | Outcome |", "|---|---|---:|---|---:|---:|---|---|"]
    for s in SCENARIOS:
        r = res["scenarios"][s]
        hb = r["H"][r["best"]]
        g = "-" if hb <= 0 else f"{r['relative_gain']:+.1%}"
        lines.append(f"| {s} | {LABELS[s]} | {r['H'][OURS]:.4f} | {r['best']} | {hb:.4f} | {g} | {r['difference']:+.4f} [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {r['outcome']} |")
    lines += ["", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUT / fn).exists():
            lines.append(f"| `{fn}` | {rows(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              "- Stored held-out risks of analysis M: `artifacts/large_gain_2026-11-02_private/M/{mgpt,clocks,knn}/private_risks_*_<scenario>.parquet`, `cohort/private_cohortknn_<scenario>__<endpoint>.parquet` (read through M.assemble).",
              "- Endpoints, follow-up, age, sex and locked outer folds: `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet` (J.load_preds).",
              "- Reference C-indices: `artifacts/large_gain_2026-11-02/M/M_cindex_per_endpoint_2026-11-02.csv`; J's full C-indices `artifacts/large_gain_2026-10-28/J/J_cindex_per_endpoint_2026-10-28.csv`.",
              "- Full reproduction reference: `artifacts/large_gain_2026-10-30/L/` and L's private per-draw estimates on Chronos.",
              "", "## Notes", "",
              "- Interval: L's paired bootstrap, one participant resample per draw shared by every model, endpoint and scenario, 2,000 draws, seed 20261030 (L's weights, checked by SHA-256); prepared risks and decile membership are fixed in the draws; C* and each endpoint's best comparator are fixed at the point estimates.",
              f"- Disclosure control (L's): cells with fewer than {MIN_CELL} events or fewer than {MIN_CELL} people at risk are blank, as are cells that subtraction would reveal (see the docstring): a mean over endpoints is blank if any of its 8 cells is suppressed, a contrast is blank if either cell is, and a KM year with one hidden group hides a second one. Survival and absolute risks are rounded to 4 decimals.",
              "- Absolute risks are Kaplan-Meier estimates with other causes censored, not competing-risk estimates.",
              "- Models outside the decision keep M's role labels: mgpt2_combined_depthaware, *_knndenoise, *_cohortknn, *_factor_imputed, *_factor_denoised.",
              "- Constant risk: a fold where the stored risk is a linear function of age and sex is not estimable for that model and endpoint (`O_preparation`). Its stratum informs age and sex only; an endpoint with no estimable fold has blank metrics and counts as beta = 0 for H, with H over the estimable endpoints beside it in `O_summary`.",
              "- Unsuppressed aggregates, prepared risks, bootstrap weights and per-draw estimates stay in `artifacts/large_gain_2026-11-08_private/O/` on Chronos and are never committed."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "checks", "boot", "analyse", "sensitivity", "reproduce", "verify"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.workers > 32:
        sys.exit("at most 32 workers")
    os.chdir(HERE)
    LOGS.mkdir(parents=True, exist_ok=True)
    {"prep": stage_prep, "checks": lambda: stage_checks(a.workers), "boot": lambda: stage_boot(a.workers),
     "analyse": stage_analyse, "sensitivity": stage_sensitivity, "reproduce": stage_reproduce, "verify": stage_verify}[a.stage]()


if __name__ == "__main__":
    main()
