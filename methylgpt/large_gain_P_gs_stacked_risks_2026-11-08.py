#!/usr/bin/env python3
"""Large-gain analysis P: does the MethylGPT risk carry disease information in Generation Scotland (GS) beyond the clocks
and the raw-CpG models? Stored held-out risks of analyses J and M are combined in a Cox model fitted by cross-fitting
inside each held-out fold, so that no participant's outcome sets its own weights. CPU only; no base model is refitted.

Written on 8 Nov 2026 for the large-gain analysis plan of 23 Oct 2026 (Decision rule) and the analysis brief for
P, BEFORE any stage of this analysis runs. Every result is reported, whichever set it favours.

WHY, AND WHEN THIS ANALYSIS WAS WRITTEN: P was written on 8 Nov 2026, after the results of analyses J, K, L, M and N were
known. On complete profiles, J found MethylGPT combined 7.5% above the best clock comparator (GrimAge2 + DunedinPACE; M
0.0320 vs 0.0298, FAIL), and K found MethylGPT 24% below its best comparator on individual diseases (FAIL). J compares
models side by side; it does not ask whether the MethylGPT risk adds information to the comparators' risks. P asks that
question. No set of P, and no model that combines the MethylGPT risk with a clock or raw-CpG risk, had been evaluated
before this commit. Two earlier analyses combined risks in other ways and are not P's question: gs_stacked_2026-10-04.py
learned weights over MethylGPT's own components (embedding, biomarkers, residuals) by inner cross-validation, and
gs_clinical_covariates_2026-10-03.py added equal-weight methylation risks (MethylGPT combined, or GrimAge2 + DunedinPACE)
to a clinical covariate model. Before this commit only schemas and shapes of the stored risk tables were inspected, and
the code was tested on synthetic data only: the Cox solvers on random arrays, stages boot and analyse with random numbers
in place of the cross-fitted linear predictors (that run scored only J's and M's stored unstacked risks, reproducing their
published M), and stages prep and checks in a fully synthetic cohort. No stacked set was fitted to or scored on GS
outcomes, and no outcome metric of a stacked set was computed.

INPUTS (participant-level, Chronos only; loaded through J's and M's own functions)
  J and M are loaded with M's load_module pattern (J = large_gain_J_gs_missing_cpgs_knn_2026-10-28.py, M =
  large_gain_M_gs_measurement_noise_2026-11-02.py; P uses the J instance that M loads). Participants, endpoints and outer
  folds: J.load_preds() (artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet, age_adjustment
  "linear"): the 9,866 GS development participants, J's 8 endpoints (J.ENDPOINTS order), J's locked outer folds (labels 1
  to 5, assigned separately per endpoint). Age and sex: the same table. Validation and test participants are not read.
  Scenarios and stored risk tables:
    full        PRIMARY. J.assemble("full", preds): complete profiles.
    hm27, random_90   J.assemble(scen, preds): simulated missingness on empirical profiles (J's label).
    depth_10, depth_5, depth_30, gauss_0.05   M.assemble(scen, preds): "simulated measurement noise on empirical profiles"
                (M's label; measured GS profiles with noise from the shared generator; no outcome or model is simulated).
  Stored risk columns used: mgpt2_combined; grimage2_F, dunedinpace_F, raw_cpg_v12_F, raw_cpg_gpu_F (F = mean, and F = knn
  outside full); mgpt2_combined_depthaware (M's scenarios); age_sex (J's stored baseline_risk: the reference of M and the
  positive control). Each is the held-out risk of a model fitted on the other four outer folds (J's and M's fold models);
  P never refits them.

PREPARATION, per scenario, endpoint e, outer fold f and the fold's held-out participants H (n about 1,973)
  Standardisation: every risk column r is standardised within H, z = (r - mean_H r) / sd_H r (ddof 0). Outcome-free.
  Constant-risk test (L's prepare(), large_gain_L_gs_risk_stratification_2026-10-30.py): residual e of r on [1, age, sex]
  by least squares within H; the column is NOT ESTIMABLE in that fold iff sd(e) <= 1e-5 x sd(r) (np.std, ddof 0). A
  non-estimable column is dropped from every inner fit of that outer fold in every set that contains it; a set left with no
  risk column in a fold is fitted with age and sex only there. Every (scenario, endpoint, fold, column) is recorded in
  P_preparation (raw SD, residual SD, ratio, estimable, sets containing the column). age_sex is prepared too, as the
  positive control only (it is in no set).

NESTED SPLIT (cross-fitting) inside H only
  Inner folds: 10, stratified by event. Let E_H and N_H be the positions within H (endpoint-table order) of the event and
  the non-event participants, k the endpoint index in J.ENDPOINTS order (0 to 7) and f the stored outer-fold label (1 to
  5). perm = default_rng([20261108, k, f, 0]).permutation(len(E_H)) and inner[E_H[perm[p]]] = p mod 10; likewise for N_H
  with default_rng([20261108, k, f, 1]). The inner split depends only on the endpoint and the outer fold, so every set and
  every scenario uses the same split (all contrasts are paired).
  Fit: for inner fold j, a Breslow Cox model (ties = "breslow", no strata, no weights) is fitted on T = H minus inner fold
  j. Design: standardised age (mean and SD of age over T, ddof 0) and sex (0/1 as stored), both unpenalised, plus the set's
  estimable standardised risk columns z. Objective: maximise log partial likelihood(beta) - 0.5 x sum of squared risk-
  column coefficients (a fixed ridge, a unit-variance Gaussian prior on the log hazard ratio per SD; fixed, not tuned).
  Solver: Newton-Raphson from beta = 0 with step halving whenever the penalised objective would decrease; converged when
  max |step| < 1e-10; at most 100 iterations. A fit that does not converge aborts stage prep (no fallback).
  Prediction on inner fold j: the linear predictor with every covariate centred at its mean over H (outcome-free; it fixes
  the arbitrary additive constant of each inner fit to the same reference population): lp = b_age (age - mean_H age) /
  sd_T age + b_sex (sex - mean_H sex) + sum_c b_c z_c (z is already centred over H).
  Every participant in H is predicted exactly once, by the fit that excluded their inner fold. Participants of other outer
  folds are never used, for any fit or any standardisation.

MODEL SETS (every set includes age and sex; F = the clock/raw fill variant: in full only F = mean; elsewhere F = mean and
F = knn; a set uses one F for all its members)
  Decision model  P1 = {mgpt2_combined, grimage2_mean, dunedinpace_mean}. Its clocks use mean fill in every scenario
                  (fixed a priori).
  Decision comparators (no MethylGPT): C1_F {grimage2_F}; C2_F {dunedinpace_F}; C3_F {grimage2_F, dunedinpace_F};
                  C4_F {raw_cpg_v12_F}; C5_F {raw_cpg_gpu_F}; C6_F {grimage2_F, dunedinpace_F, raw_cpg_v12_F};
                  C7_F {grimage2_F, dunedinpace_F, raw_cpg_gpu_F}; C8_F {grimage2_F, dunedinpace_F, raw_cpg_v12_F,
                  raw_cpg_gpu_F}. full: 8 comparators (F = mean); every other scenario: 16 (F = mean and F = knn).
  Secondary sets, outside the decision: P1_knn {mgpt2_combined, grimage2_knn, dunedinpace_knn} (not in full); S1
                  {mgpt2_combined} (recalibrated MethylGPT); S2 {mgpt2_combined, grimage2_mean, dunedinpace_mean,
                  raw_cpg_v12_mean, raw_cpg_gpu_mean}; S0 = age and sex only (cross-fitted reference); in the noise
                  scenarios S3 {mgpt2_combined_depthaware, grimage2_mean, dunedinpace_mean}.
  In gauss_0.05 nothing is missing, so every _knn column equals its _mean column (M's note) and the _knn sets repeat the
  _mean sets; they are kept as specified.

METRIC
  C(set, e) = J's pooled within-fold C-index of the cross-fitted linear predictor on endpoint e (J.fold_counts arithmetic,
  sksurv concordance_index_censored convention, tied_tol 1e-8; comparable pairs only within an outer fold, summed over the
  5 outer folds). M(set) = mean over the 8 endpoints of C(set, e) - C(age_sex, e), age_sex = J's stored baseline risk (the
  reference of J and M). Unstacked stored risks (J's and M's models of the same names) are scored the same way.

DECISION (full scenario only)
  C* = the decision comparator (C1_mean to C8_mean) with the largest M. PASS iff (M_P1 - M_C*) / M_C* >= 0.10 and the lower
  end of the 95% interval of M_P1 - M_C* is > 0; otherwise FAIL. If M_C* <= 0 the rule is not applicable. Every secondary
  scenario gets its own row with its own C* (chosen among its 16 comparators); there the rule arithmetic is for reference
  only, not a decision.

INTERVALS
  Shared participant bootstrap: 2,000 draws, seed 20261108, I = default_rng(20261108).integers(0, 9866, size=(2000, 9866)),
  w_b = bincount(I[b], minlength=9866) in frame-cache order (labels parquet of the frame cache, as J and L), pair weights
  w_i w_j within each outer fold of each endpoint (J's scheme); one resample shared by every set, endpoint and scenario.
  The cross-fitted risks are fixed in the draws (J's and M's convention: risks are not refitted). Sensitivity: J's
  within-fold resample (per endpoint k in J.ENDPOINTS order and outer fold in sorted order, multiplicities from
  default_rng([20261108, 1, k]).integers(0, n_f, size=(2000, n_f))). Percentile 95% intervals; bootstrap P = min(1, 2 (k +
  1) / (B + 1)), k = min(#{d_b <= 0}, #{d_b >= 0}) (J.pboot). C* is chosen on the point estimates and kept fixed in the
  draws.
  Execution note: the bootstrap counts use a copy of J.fold_counts that skips the weighted tied-pair product of a risk
  vector with no tied comparable pair in the fold (that product is exactly zero); CHECKS 6 shows it equals J.fold_counts
  exactly.

SECONDARY OUTPUTS (all pre-specified, all reported; none changes the decision)
  - Every set and every unstacked stored model, every scenario: M with interval, P1 minus it with both intervals, P,
    relative gain and rule arithmetic against it (P_summary).
  - Per endpoint: C, gain above age_sex, P1 minus each set with interval, P and Benjamini-Hochberg q across the 8
    endpoints (P_cindex_per_endpoint); P1 minus the scenario's C* and minus the endpoint's own best decision comparator
    (chosen by the point estimate on that endpoint, fixed in the draws), each with interval, P and q, contribution to the
    M difference, and rule arithmetic against the own best for reference (P_endpoint_contrasts).
  - Leave-one-endpoint-out: M over the other 7 endpoints with C* re-chosen on those 7 and with C* fixed at the 8-endpoint
    choice; same draws; rule arithmetic for reference only (P_leave_one_endpoint_out).
  - Event-weighted mean over endpoints (weights = incident events), C* re-chosen, rule arithmetic for reference only
    (P_event_weighted).
  - Named contrasts, outside the decision (P_named_contrasts): P1 vs C3_mean (MethylGPT combined added to the two clocks;
    P1 = C3_mean plus mgpt2_combined), and P1_knn vs C3_knn where knn exists; P1 vs S1 (the two clocks added to
    MethylGPT combined); S2 vs C8_mean (MethylGPT combined added to the two clocks and both raw-CpG models).
  - Recalibration (P_recalibration): each single-model stacked set against J's or M's unstacked model of the same name,
    scored from the stored risk with the same fold_counts: S1 vs mgpt2_combined, C1_F vs grimage2_F, C2_F vs dunedinpace_F,
    C4_F vs raw_cpg_v12_F, C5_F vs raw_cpg_gpu_F, and S0 vs age_sex; this is the effect of recalibrating a model on the
    target data by cross-fitting.
  - Rule arithmetic on every non-decision row is for reference only.

CHECKS (written to P_checks before any result is used; stage boot refuses to run unless all pass; abort on failure)
  (1) J's and M's stored pooled C-indices reproduce from the stored risks: every stored column P uses, plus age_sex, every
      scenario and endpoint, against J_cindex_per_endpoint / M_cindex_per_endpoint within 1e-12 (CSV round trip), and the
      stored M of every such model against J_results / M_results within 1e-12.
  (2) Identifiers: per endpoint 9,866 distinct participants equal to the frame-cache participants, 5 outer folds labelled
      1 to 5; in every scenario the assembled rows of each endpoint are in the endpoint-table order with the same event,
      duration and outer fold.
  (3) Cross-fit integrity: the stored inner assignment equals an independent recomputation (inverse permutation); inner
      folds partition H (every H participant in exactly one of 10); the stored linear predictors of the sampled outer folds
      (below) equal a recomputation bitwise; per fit, training rows = H minus the inner fold and
      test rows = the inner fold (SHA-256 of positions against an independent recomputation), so no other-fold participant
      is used; every participant is predicted exactly once per scenario, set and endpoint (finite linear predictor from
      exactly one fit); events per inner fold are recorded (private file; public: number of inner folds with fewer than 5
      events, never the small counts) and differ by at most 1 within an outer fold; outcome-perturbation test on a sample
      (per scenario and endpoint, outer fold 1 + k mod 5): permuting the outcomes of every participant outside H leaves
      every linear predictor of H unchanged bitwise, and permuting the outcomes of inner fold j = k mod 10 leaves that
      inner fold's linear predictors unchanged bitwise.
  (4) Cox solver, on a sample of fits (per scenario and endpoint: outer fold 1, inner fold 0 for P1, C8_mean and S0, plus
      one fit drawn by default_rng([20261108, 4, scenario index, k])): with penalty 0, P's solver against statsmodels
      PHReg(ties="breslow") on the same design, max abs coefficient difference <= 1e-6; with the ridge, the stored
      coefficients against an independent scipy.optimize.minimize (BFGS) of the penalised Breslow partial likelihood
      coded separately (explicit risk-set matrix), max abs difference <= 1e-5. Every fit converged (count and maximum
      iterations recorded). Weighted fits are not needed: risks are not refitted in the draws.
  (5) Constant-risk handling: the positive control age_sex is flagged not estimable in every scenario, endpoint and fold.
  (6) Bootstrap: every shared draw sums to 9,866 (SHA-256 recorded; stage boot recomputes it and must match); the
      counting function equals J.fold_counts exactly (one endpoint-fold of full, every column, both resamples).
  (7) Privacy (stage privacy, appended after the outputs, as J, K and M): no public file holds a participant identifier,
      an identifier column or an event cell below 5.

STAGES (resumable; atomic writes; done-markers artifacts/logs/large_gain_P_<stage>_2026-11-08.done)
  prep (CPU, one worker per scenario) -> checks (CPU) -> boot (CPU, sharded by scenario) -> analyse (CPU) -> privacy
  (CPU). Every stage refuses to run unless this script, J's script and M's script are committed unchanged, and stage prep
  requires J's and M's analyse done-markers. Execution-only changes after the commit go in a dated AMENDMENT paragraph
  below, committed before their output is used. Run inside the transient user unit large-gain-P-2026-11-08.
OUTPUTS
  Aggregate tables: artifacts/large_gain_2026-11-08/P/ (README.md, P_rule_outcome, P_summary, P_cindex_per_endpoint,
  P_endpoint_contrasts, P_leave_one_endpoint_out, P_event_weighted, P_named_contrasts, P_recalibration, P_preparation,
  P_checks, P_results.json, P_bootstrap_draws.csv.gz). Participant-level linear predictors, inner folds, fit records and
  bootstrap arrays: artifacts/large_gain_2026-11-08_private/P/ (git-ignored, never committed, stay on Chronos). Tables
  only: no figures, no manuscript text.
AMENDMENT 1 (8 Nov 2026; verification only; after the first run of stage checks, before stage boot has produced any
  output): 3 of the 1,316 check rows failed, all in check 4's penalty-0 comparison and all for C8_mean on Cancers, outer
  fold 1 (full inner fold 0; random_90 inner folds 0 and 5). statsmodels PHReg's default Newton optimiser returned
  non-finite coefficients (overflow, "invalid value encountered in divide") on these unpenalised designs, in which the
  standardised age and the clock and raw-CpG risk columns correlate at 0.96 to 0.998; P's solver converged in 8
  iterations. Refitted with PHReg's own BFGS optimiser (method="bfgs", gtol 1e-10, maxiter 5000), PHReg reaches P's
  solution within 1.2e-9, 2.2e-8 and 8.2e-8. Check 4 therefore refits PHReg with method="bfgs" whenever the default fit
  returns a non-finite coefficient and compares that fit, with the same tolerance (1e-6); the note records the optimiser.
  In the first run the ridge comparison with scipy passed for all 224 sampled fits and every other check passed. No fit,
  linear predictor, estimate, table or decision changes.
Run: python3 large_gain_P_gs_stacked_risks_2026-11-08.py --stage <stage> [--workers n] [--shard i --nshards n]
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
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-08"
OUTP = ART / "large_gain_2026-11-08/P"
PRIV = ART / "large_gain_2026-11-08_private/P"
LOGS = ART / "logs"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"
M_SCRIPT = HERE / "large_gain_M_gs_measurement_noise_2026-11-02.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


J = load_module("large_gain_J", J_SCRIPT)
MM = load_module("large_gain_M", M_SCRIPT)
J = MM.J                       # one J instance: the one M loaded
log, need, write_json = J.log, J.need, J.write_json
atomic_csv, atomic_npz, sha256_bytes = J.atomic_csv, J.atomic_npz, J.sha256_bytes
ENDPOINTS = J.ENDPOINTS

SEED = 20261108
N_BOOT = 2000
N = 9866
N_INNER = 10
RIDGE = 1.0                    # objective: log PL - 0.5 * RIDGE * sum(beta_risk^2)
CONST_RATIO = 1e-5
NEWTON_TOL = 1e-10
NEWTON_MAXIT = 100
TOL_C = 1e-12
TOL_PHREG = 1e-6
TOL_SCIPY = 1e-5
SMALL_CELL = 5
SCEN_J = ["full", "hm27", "random_90"]
SCEN_M = ["depth_10", "depth_5", "depth_30", "gauss_0.05"]
ALL_SCEN = SCEN_J + SCEN_M
PRIMARY = "full"
OURS = "mgpt2_combined"
DA = "mgpt2_combined_depthaware"
BASES = ["grimage2", "dunedinpace", "raw_cpg_v12", "raw_cpg_gpu"]
COMP_DEF = {"C1": ["grimage2"], "C2": ["dunedinpace"], "C3": ["grimage2", "dunedinpace"], "C4": ["raw_cpg_v12"],
            "C5": ["raw_cpg_gpu"], "C6": ["grimage2", "dunedinpace", "raw_cpg_v12"],
            "C7": ["grimage2", "dunedinpace", "raw_cpg_gpu"], "C8": ["grimage2", "dunedinpace", "raw_cpg_v12", "raw_cpg_gpu"]}
RULE = ("PASS iff (M_P1 - M_C*) / M_C* >= 0.10 and the lower end of the 95% interval of M_P1 - M_C* is > 0; M = mean over "
        "the 8 endpoints of (pooled within-fold C-index of the cross-fitted linear predictor - age+sex C-index); C* = decision "
        "comparator (C1 to C8, no MethylGPT) with the largest M in the scenario; decision = full scenario only")


# ------------------------------------------------------------------ scenarios, sets, roles
def fills(scen):
    return ["mean"] if scen == "full" else ["mean", "knn"]


def label(scen):
    if scen == "full":
        return "complete profiles"
    return J.LABEL if scen in SCEN_J else MM.LABEL


def source(scen):
    return "J" if scen in SCEN_J else "M"


def sets_for(scen):
    """Ordered {set name: (members, role)}."""
    s = {"P1": ([OURS, "grimage2_mean", "dunedinpace_mean"], "decision model")}
    for F in fills(scen):
        for c, mem in COMP_DEF.items():
            s[f"{c}_{F}"] = ([f"{b}_{F}" for b in mem], "decision comparator")
    if scen != "full":
        s["P1_knn"] = ([OURS, "grimage2_knn", "dunedinpace_knn"], "secondary set (P1 with kNN-filled clocks)")
    s["S1"] = ([OURS], "secondary set (recalibrated MethylGPT combined)")
    s["S2"] = ([OURS, "grimage2_mean", "dunedinpace_mean", "raw_cpg_v12_mean", "raw_cpg_gpu_mean"],
               "secondary set (MethylGPT combined, both clocks, both raw-CpG models)")
    s["S0"] = ([], "secondary set (cross-fitted age and sex only)")
    if scen in SCEN_M:
        s["S3"] = ([DA, "grimage2_mean", "dunedinpace_mean"], "secondary set (P1 with depth-aware MethylGPT)")
    return s


def decision_sets(scen):
    return [f"{c}_{F}" for F in fills(scen) for c in COMP_DEF]


def stored_for(scen):
    """Unstacked stored risks scored alongside the sets (recalibration references)."""
    return [OURS] + [f"{b}_{F}" for F in fills(scen) for b in BASES]


def columns_for(scen):
    cols = []
    for mem, _ in sets_for(scen).values():
        cols += [c for c in mem if c not in cols]
    return cols


def models_for(scen):
    return list(sets_for(scen)) + stored_for(scen) + ["age_sex"]


def role(scen, m):
    s = sets_for(scen)
    if m in s:
        return s[m][1]
    if m == "age_sex":
        return "covariate model (J's stored age+sex risk, reference of M)"
    return f"unstacked stored risk ({source(scen)}'s model)"


def members(scen, m):
    s = sets_for(scen)
    return "age, sex" + "".join(f", {c}" for c in s[m][0]) if m in s else m


def named_contrasts(scen):
    out = [("P1 vs C3_mean: MethylGPT combined added to GrimAge2 and DunedinPACE (mean fill)", "P1", "C3_mean")]
    if scen != "full":
        out.append(("P1_knn vs C3_knn: MethylGPT combined added to GrimAge2 and DunedinPACE (kNN fill)", "P1_knn", "C3_knn"))
    out += [("P1 vs S1: GrimAge2 and DunedinPACE added to MethylGPT combined", "P1", "S1"),
            ("S2 vs C8_mean: MethylGPT combined added to both clocks and both raw-CpG models (mean fill)", "S2", "C8_mean")]
    return out


def recal_pairs(scen):
    out = [("S1", OURS)]
    for F in fills(scen):
        out += [(f"C1_{F}", f"grimage2_{F}"), (f"C2_{F}", f"dunedinpace_{F}"), (f"C4_{F}", f"raw_cpg_v12_{F}"),
                (f"C5_{F}", f"raw_cpg_gpu_{F}")]
    out.append(("S0", "age_sex"))
    return out


# ------------------------------------------------------------------ gates and paths
def done_path(stage):
    return LOGS / f"large_gain_P_{stage}_{DATE}.done"


def done_marker(stage, info=None):
    write_json(done_path(stage), {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})})


def require_done(*stages):
    for s in stages:
        if not done_path(s).exists():
            sys.exit(f"stage {s} has not completed ({done_path(s)} missing)")


def committed(path):
    r = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(path)], capture_output=True)
    if r.returncode:
        return False
    return subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0


def require_committed():
    for p in (Path(__file__).resolve(), J_SCRIPT, M_SCRIPT):
        if not committed(p):
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before any result-producing stage runs")


def shard_of(items, shard, nshards):
    return [x for i, x in enumerate(items) if i % nshards == shard]


def load_ids():
    mp = J.need_glob(J.CACHE, f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")
    return pd.read_parquet(need(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet"))),
                           columns=["dnam_id"]).dnam_id.astype(str).to_numpy()


def assemble(scen, preds):
    A = J.assemble(scen, preds) if scen in SCEN_J else MM.assemble(scen, preds)
    miss = [c for c in columns_for(scen) + stored_for(scen) + ["age_sex"] if c not in A.columns]
    if miss:
        sys.exit(f"{scen}: stored risk columns missing: {miss}")
    for ep in ENDPOINTS:
        g = A[A.endpoint == ep]
        if not np.array_equal(g.dnam_id.to_numpy(), preds[ep].dnam_id.to_numpy()):
            sys.exit(f"{scen} {ep}: stored risks are not in the endpoint-table order")
    return A


def pos_sha(a):
    return sha256_bytes(np.sort(np.asarray(a, np.int64)).tobytes())


# ------------------------------------------------------------------ preparation and inner split
def estimability(r, age, sex):
    """L's constant-risk test on one fold: (raw SD, residual SD, estimable)."""
    A = np.column_stack([np.ones(len(r)), age, sex])
    e = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    sd_r, sd_e = float(r.std()), float(e.std())
    return sd_r, sd_e, bool(sd_e > CONST_RATIO * sd_r)


def inner_assignment(ev_h, k, f):
    """Inner fold (0..9) of each participant of H (positions in H order)."""
    inner = np.full(len(ev_h), -1, np.int64)
    for flag, idx in ((0, np.flatnonzero(ev_h)), (1, np.flatnonzero(~ev_h))):
        perm = np.random.default_rng([SEED, k, f, flag]).permutation(len(idx))
        inner[idx[perm]] = np.arange(len(idx)) % N_INNER
    return inner


def fold_columns(g, H, cols):
    """Standardised columns on H and the constant-risk record of each column."""
    age, sex = g.age_v[H], g.sex_v[H]
    z, est, rec = {}, {}, []
    for c in cols:
        r = g.R[c][H]
        sd_r, sd_e, ok = estimability(r, age, sex)
        est[c] = ok
        z[c] = (r - r.mean()) / sd_r if sd_r > 0 else np.zeros(len(r))
        rec.append({"column": c, "n_fold": len(H), "raw_sd": sd_r, "residual_sd": sd_e,
                    "residual_to_raw_sd": sd_e / sd_r if sd_r > 0 else np.nan, "estimable": ok})
    return z, est, rec


class EP:
    """One endpoint's arrays in endpoint-table order."""

    def __init__(self, A, preds, ep, cols):
        g = A[A.endpoint == ep]
        pe = preds[ep]
        self.ev = g.event.astype(bool).to_numpy()
        self.du = g.duration.to_numpy(float)
        self.fold = g.outer_fold.to_numpy()
        self.age_v = pe.age.to_numpy(float)
        self.sex_v = pe.sex.to_numpy(float)
        self.R = {c: g[c].to_numpy(float) for c in cols}
        self.ids = g.dnam_id.to_numpy().astype(str)


# ------------------------------------------------------------------ Cox (Breslow, ridge on risk columns)
def cox_fit(Z, du, ev, pen, ridge=RIDGE, maxit=NEWTON_MAXIT, tol=NEWTON_TOL):
    """Newton-Raphson for the Breslow partial likelihood minus 0.5 * ridge * sum(pen * beta^2).
    Returns (beta, iterations, converged, penalised log likelihood)."""
    o = np.argsort(-du, kind="stable")
    Zs, d, e = Z[o], du[o], ev[o].astype(bool)
    last = np.searchsorted(-d, -d, side="right") - 1        # risk set of an event at d_i = rows 0..last[i]
    ei = np.flatnonzero(e)
    ends = last[ei]
    p = Z.shape[1]
    P = np.diag(np.asarray(pen, float) * ridge)
    sZe = Zs[ei].sum(0)

    def evalf(beta, full):
        eta = Zs @ beta
        c = eta.max()
        w = np.exp(eta - c)
        S0 = np.cumsum(w)[ends]
        ll = float(eta[ei].sum() - (np.log(S0) + c).sum() - 0.5 * beta @ P @ beta)
        if not full:
            return ll
        wZ = w[:, None] * Zs
        S1 = np.cumsum(wZ, 0)[ends]
        S2 = np.cumsum(wZ[:, :, None] * Zs[:, None, :], 0)[ends]
        m1 = S1 / S0[:, None]
        grad = sZe - m1.sum(0) - P @ beta
        info = (S2 / S0[:, None, None]).sum(0) - m1.T @ m1 + P
        return ll, grad, info

    beta = np.zeros(p)
    ll, grad, info = evalf(beta, True)
    for it in range(1, maxit + 1):
        step = np.linalg.solve(info, grad)
        t = 1.0
        while True:
            nb = beta + t * step
            nll = evalf(nb, False)
            if nll >= ll - 1e-12 * max(1.0, abs(ll)) or t < 1e-8:
                break
            t *= 0.5
        beta = nb
        ll, grad, info = evalf(beta, True)
        if np.max(np.abs(t * step)) < tol:
            return beta, it, True, ll
    return beta, maxit, False, ll


def neg_pen_pl_independent(beta, Z, du, ev, pen, ridge=RIDGE):
    """Independent coding (explicit risk-set matrix) of minus the penalised Breslow log partial likelihood and its gradient."""
    eta = Z @ beta
    E = np.flatnonzero(ev)
    Rm = (du[None, :] >= du[E][:, None]).astype(float)
    w = np.exp(eta)
    den = Rm @ w
    pen = np.asarray(pen, float)
    ll = eta[E].sum() - np.log(den).sum() - 0.5 * ridge * (pen * beta ** 2).sum()
    gr = Z[E].sum(0) - ((Rm * w[None, :]) @ Z / den[:, None]).sum(0) - ridge * pen * beta
    return -ll, -gr


def design(g, rows, z_rows, use, mu_age_T, sd_age_T, centre=None):
    """Design on the given endpoint rows: [age standardised with the fit's mean/SD, sex, z columns]. centre = (age mean, sex
    mean) over H for prediction (lp centred over H); None for fitting."""
    a = g.age_v[rows]
    s = g.sex_v[rows]
    if centre is None:
        cols = [(a - mu_age_T) / sd_age_T, s]
    else:
        cols = [(a - centre[0]) / sd_age_T, s - centre[1]]
    return np.column_stack(cols + [z_rows[c] for c in use])


def crossfit_fold(g, H, inner, z, est, sets, ev=None, du=None):
    """Cross-fitted linear predictors on H for every set; fit records. ev/du override the endpoint outcomes (checks only)."""
    ev = g.ev if ev is None else ev
    du = g.du if du is None else du
    LP = np.full((len(sets), len(H)), np.nan)
    recs = []
    centre = (g.age_v[H].mean(), g.sex_v[H].mean())
    for j in range(N_INNER):
        tr, te = np.flatnonzero(inner != j), np.flatnonzero(inner == j)
        T, Te = H[tr], H[te]
        mu, sd = g.age_v[T].mean(), g.age_v[T].std()
        ztr = {c: v[tr] for c, v in z.items()}
        zte = {c: v[te] for c, v in z.items()}
        for si, (name, (mem, _)) in enumerate(sets.items()):
            use = [c for c in mem if est[c]]
            Ztr = design(g, T, ztr, use, mu, sd)
            pen = np.r_[0.0, 0.0, np.ones(len(use))]
            beta, it, conv, ll = cox_fit(Ztr, du[T], ev[T], pen)
            LP[si, te] = design(g, Te, zte, use, mu, sd, centre) @ beta
            recs.append({"inner_fold": j, "set": name, "n_train": len(T), "n_test": len(Te), "events_train": int(ev[T].sum()),
                         "events_test": int(ev[Te].sum()), "columns_used": ";".join(use), "n_dropped": len(mem) - len(use),
                         "iterations": it, "converged": bool(conv), "penalised_loglik": ll,
                         "beta": json.dumps([float(b) for b in beta]), "train_sha256": pos_sha(T), "test_sha256": pos_sha(Te)})
    return LP, recs


def crossfit_scenario(scen):
    t0 = time.time()
    preds = J.load_preds()
    A = assemble(scen, preds)
    sets = sets_for(scen)
    cols = columns_for(scen)
    innerz = np.load(PRIV / "private_inner_folds.npz")
    LP = np.full((len(sets), len(ENDPOINTS), N), np.nan)
    prep_rows, fit_rows = [], []
    containing = {c: ";".join(n for n, (mem, _) in sets.items() if c in mem) for c in cols}
    for k, ep in enumerate(ENDPOINTS):
        g = EP(A, preds, ep, cols + ["age_sex"])
        inner_all = innerz[ep]
        for f in sorted(np.unique(g.fold)):
            H = np.flatnonzero(g.fold == f)
            z, est, rec = fold_columns(g, H, cols + ["age_sex"])
            for r in rec:
                prep_rows.append({"scenario": scen, "label": label(scen), "endpoint": ep, "outer_fold": int(f), **r,
                                  "positive_control": r["column"] == "age_sex",
                                  "sets_containing_column": containing.get(r["column"], "none (positive control only)")})
            lp, recs = crossfit_fold(g, H, inner_all[H], z, est, sets)
            LP[:, k, H] = lp
            for r in recs:
                fit_rows.append({"scenario": scen, "endpoint": ep, "outer_fold": int(f), **r})
    fits = pd.DataFrame(fit_rows)
    atomic_npz(PRIV / f"private_crossfit_{scen}.npz", LP=LP, sets=np.array(list(sets)))
    tmp = PRIV / f"private_fits_{scen}.tmp.parquet"
    fits.to_parquet(tmp, index=False)
    os.replace(tmp, PRIV / f"private_fits_{scen}.parquet")
    log(f"{scen}: {len(fits)} fits, {int((~fits.converged).sum())} not converged, max iterations {fits.iterations.max()}, "
        f"{time.time() - t0:.0f} s")
    return prep_rows, {"scenario": scen, "fits": len(fits), "not_converged": int((~fits.converged).sum()),
                       "max_iterations": int(fits.iterations.max()), "finite_lp": bool(np.isfinite(LP).all()),
                       "seconds": round(time.time() - t0)}


def stage_prep(workers):
    require_committed()
    for s in ("large_gain_J_analyse_2026-10-28", "large_gain_M_analyse_2026-11-02"):
        need(LOGS / f"{s}.done")
    t0 = time.time()
    PRIV.mkdir(parents=True, exist_ok=True)
    OUTP.mkdir(parents=True, exist_ok=True)
    preds = J.load_preds()
    inner, ev_counts = {}, []
    for k, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        ev, fold = pe.event.astype(bool).to_numpy(), pe.outer_fold.to_numpy()
        a = np.full(len(pe), -1, np.int64)
        for f in sorted(np.unique(fold)):
            H = np.flatnonzero(fold == f)
            a[H] = inner_assignment(ev[H], k, int(f))
            for j in range(N_INNER):
                m = H[a[H] == j]
                ev_counts.append({"endpoint": ep, "outer_fold": int(f), "inner_fold": j, "n": len(m), "events": int(ev[m].sum())})
        inner[ep] = a
    atomic_npz(PRIV / "private_inner_folds.npz", **inner)
    atomic_csv(PRIV / "private_inner_fold_events.csv", pd.DataFrame(ev_counts))
    from multiprocessing import Pool
    with Pool(min(workers, len(ALL_SCEN))) as pool:
        res = pool.map(crossfit_scenario, ALL_SCEN)
    prep = pd.DataFrame([r for rows, _ in res for r in rows])
    atomic_csv(OUTP / f"P_preparation_{DATE}.csv", prep)
    info = [i for _, i in res]
    write_json(PRIV / "private_prep_record.json", {"scenarios": info})
    bad = [i for i in info if i["not_converged"] or not i["finite_lp"]]
    if bad:
        sys.exit(f"cross-fitting incomplete (non-converged fit or non-finite linear predictor): {bad}")
    done_marker("prep", {"seconds": round(time.time() - t0), "fits": int(sum(i["fits"] for i in info))})


# ------------------------------------------------------------------ pair counting (J.fold_counts, tied product skipped when zero)
def fold_counts(ev, du, R, Ws):
    """J.fold_counts with one execution-only change: the weighted tied-pair product of a risk vector is skipped when that
    vector has no tied comparable pair in the fold (the product is then exactly zero)."""
    e = np.flatnonzero(ev)
    comp = (du[None, :] > du[e][:, None]) | ((du[None, :] == du[e][:, None]) & ~ev[None, :])
    compf = comp.astype(np.float64)
    out = []
    for W in Ws:
        tot = compf.sum() if W is None else ((W[:, e] @ compf) * W).sum(1)
        out.append([tot, np.zeros((len(R),) + (() if W is None else (W.shape[0],)) + (2,))])
    for k, r in enumerate(R):
        dd = r[None, :] - r[e][:, None]
        tie = np.abs(dd) <= 1e-8
        con = (comp & (dd < 0) & ~tie).astype(np.float64)
        tiedb = comp & tie
        anytie = bool(tiedb.any())
        tied = tiedb.astype(np.float64)
        for (tot, arr), W in zip(out, Ws):
            if W is None:
                arr[k] = [con.sum(), tied.sum()]
            else:
                We = W[:, e]
                arr[k, :, 0] = ((We @ con) * W).sum(1)
                if anytie:
                    arr[k, :, 1] = ((We @ tied) * W).sum(1)
    return out


def pooled_point(ev, du, fold, R):
    con = np.zeros(len(R)); tied = np.zeros(len(R)); tot = 0.0
    for f in sorted(np.unique(fold)):
        s = np.flatnonzero(fold == f)
        (t_, arr), = fold_counts(ev[s], du[s], R[:, s], [None])
        con += arr[:, 0]; tied += arr[:, 1]; tot += t_
    return J.cidx_from(con, tied, tot)


def shared_weights():
    I = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))
    return np.stack([np.bincount(I[b], minlength=N) for b in range(N_BOOT)]).astype(np.float64)


def within_fold_weights(preds):
    sens = {}
    for k, (ep, pe) in enumerate(preds.items()):
        rng = np.random.default_rng([SEED, 1, k])
        sens[ep] = {}
        for f in sorted(pe.outer_fold.unique()):
            n_f = int((pe.outer_fold == f).sum())
            Jm = rng.integers(0, n_f, size=(N_BOOT, n_f))
            sens[ep][f] = np.stack([np.bincount(Jm[b], minlength=n_f) for b in range(N_BOOT)]).astype(np.float64)
    return sens


def risk_matrix(scen, A, ep, k, LPz):
    """Rows = models_for(scen): cross-fitted lp of each set, stored risks, age_sex (endpoint-table order)."""
    g = A[A.endpoint == ep]
    sets = list(LPz["sets"].astype(str))
    if sets != list(sets_for(scen)):
        sys.exit(f"{scen}: stored set list differs from the design")
    rows = [LPz["LP"][i, k] for i in range(len(sets))] + [g[m].to_numpy(float) for m in stored_for(scen) + ["age_sex"]]
    return np.stack(rows)


# ------------------------------------------------------------------ stage checks
def stage_checks():
    require_committed()
    require_done("prep")
    t0 = time.time()
    from scipy.optimize import minimize
    from statsmodels.duration.hazard_regression import PHReg
    preds = J.load_preds()
    ids = load_ids()
    rows = []

    def add(check, scenario, item, stored, recomputed, tol, passed, note=""):
        num = (int, float, np.integer, np.floating)
        diff = abs(float(stored) - float(recomputed)) if isinstance(stored, num) and isinstance(recomputed, num) and \
            not isinstance(stored, (bool, np.bool_)) else ""
        rows.append({"check": check, "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "abs_diff": diff, "tolerance": tol, "pass": bool(passed), "note": note})

    A = {s: assemble(s, preds) for s in ALL_SCEN}
    # (1) stored C-indices and M reproduce
    jc = pd.read_csv(need(J.OUTJ / f"J_cindex_per_endpoint_{J.DATE}.csv")).set_index(["scenario", "endpoint", "model"])
    mc = pd.read_csv(need(MM.OUTM / f"M_cindex_per_endpoint_{MM.DATE}.csv")).set_index(["scenario", "endpoint", "model"])
    jr = json.loads(need(J.OUTJ / f"J_results_{J.DATE}.json").read_text())["scenarios"]
    mr = json.loads(need(MM.OUTM / f"M_results_{MM.DATE}.json").read_text())["scenarios"]
    for scen in ALL_SCEN:
        ref, refM = (jc, jr) if scen in SCEN_J else (mc, mr)
        cols = [c for c in dict.fromkeys(stored_for(scen) + columns_for(scen) + ["age_sex"])]
        Cm = np.zeros((len(cols), len(ENDPOINTS)))
        for j, ep in enumerate(ENDPOINTS):
            g = A[scen][A[scen].endpoint == ep]
            R = np.stack([g[c].to_numpy(float) for c in cols])
            Cm[:, j] = pooled_point(g.event.astype(bool).to_numpy(), g.duration.to_numpy(float), g.outer_fold.to_numpy(), R)
            for i, c in enumerate(cols):
                st = float(ref.loc[(scen, ep, c), "c_index"])
                add("1 stored C-index reproduces from the stored risk", scen, f"{ep}:{c}", st, float(Cm[i, j]), TOL_C,
                    abs(st - Cm[i, j]) <= TOL_C, f"{source(scen)}_cindex_per_endpoint")
        Mv = (Cm - Cm[cols.index("age_sex")]).mean(1)
        for i, c in enumerate(cols):
            if c != "age_sex":
                st = float(refM[scen]["M"][c])
                add("1 stored M reproduces from the stored risk", scen, c, st, float(Mv[i]), TOL_C, abs(st - Mv[i]) <= TOL_C,
                    f"{source(scen)}_results")
    # (2) identifiers
    idset = set(ids)
    for ep, pe in preds.items():
        ok = len(pe) == N and pe.dnam_id.nunique() == N and set(pe.dnam_id) == idset
        add("2 identifiers: 9,866 distinct participants equal to the frame cache", "all", ep, N, int(pe.dnam_id.nunique()), 0, ok)
        fl = sorted(int(x) for x in pe.outer_fold.unique())
        add("2 identifiers: 5 outer folds labelled 1 to 5", "all", ep, "1;2;3;4;5", ";".join(map(str, fl)), "", fl == [1, 2, 3, 4, 5])
    for scen in ALL_SCEN:
        ok = True
        for ep in ENDPOINTS:
            g, pe = A[scen][A[scen].endpoint == ep], preds[ep]
            ok &= np.array_equal(g.dnam_id.to_numpy(), pe.dnam_id.to_numpy()) and \
                np.array_equal(g.event.to_numpy(), pe.event.to_numpy()) and \
                np.array_equal(g.duration.to_numpy(), pe.duration.to_numpy()) and \
                np.array_equal(g.outer_fold.to_numpy(), pe.outer_fold.to_numpy())
        add("2 identifiers: assembled rows in endpoint-table order with the same event, duration and fold", scen, "8 endpoints",
            "", "", "", ok)
    # (3) cross-fit integrity
    innerz = np.load(need(PRIV / "private_inner_folds.npz"))
    evc = pd.read_csv(need(PRIV / "private_inner_fold_events.csv"))
    expected_sha = {}
    for k, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        ev, fold = pe.event.astype(bool).to_numpy(), pe.outer_fold.to_numpy()
        a = innerz[ep]
        ok_assign, ok_part, ok_strat = True, True, True
        for f in sorted(np.unique(fold)):
            H = np.flatnonzero(fold == f)
            indep = np.full(len(H), -1)
            for flag, idx in ((0, np.flatnonzero(ev[H])), (1, np.flatnonzero(~ev[H]))):
                perm = np.random.default_rng([SEED, k, int(f), flag]).permutation(len(idx))
                indep[idx] = np.argsort(perm) % N_INNER
            ok_assign &= np.array_equal(indep, a[H])
            ok_part &= bool(((a[H] >= 0) & (a[H] < N_INNER)).all()) and sum(int((a[H] == j).sum()) for j in range(N_INNER)) == len(H)
            ce = np.array([int(ev[H][a[H] == j].sum()) for j in range(N_INNER)])
            cn = np.array([int((~ev[H])[a[H] == j].sum()) for j in range(N_INNER)])
            ok_strat &= ce.max() - ce.min() <= 1 and cn.max() - cn.min() <= 1 and ce.min() >= 1
            for j in range(N_INNER):
                expected_sha[(ep, int(f), j)] = (pos_sha(H[a[H] != j]), pos_sha(H[a[H] == j]), len(H) - int((a[H] == j).sum()))
        ok_part &= bool((a >= 0).all())
        add("3 inner assignment equals the independent recomputation", "all", ep, "", "", "", ok_assign, "inverse-permutation formula")
        add("3 inner folds partition each outer fold", "all", ep, "", "", "", ok_part)
        e_ep = evc[evc.endpoint == ep]
        nsmall = int((e_ep.events < SMALL_CELL).sum())
        add("3 events per inner fold recorded; stratified (differ by at most 1), at least 1 per inner fold", "all", ep, 50, len(e_ep), "",
            ok_strat and len(e_ep) == 50,
            f"inner folds with fewer than {SMALL_CELL} events: {nsmall} of 50 (counts in the private file only)")
    for scen in ALL_SCEN:
        fits = pd.read_parquet(need(PRIV / f"private_fits_{scen}.parquet"))
        LPz = np.load(need(PRIV / f"private_crossfit_{scen}.npz"))
        sets = sets_for(scen)
        ok_sha = all((r.train_sha256, r.test_sha256, r.n_train) == expected_sha[(r.endpoint, int(r.outer_fold), int(r.inner_fold))]
                     for r in fits.itertuples())
        n_expected = len(sets) * len(ENDPOINTS) * 5 * N_INNER
        add("3 per fit: training rows = H minus the inner fold, test rows = the inner fold (no other-fold participant)", scen,
            "fits", n_expected, len(fits), 0, ok_sha and len(fits) == n_expected)
        cnt = fits.groupby(["set", "endpoint"]).n_test.sum()
        once = bool((cnt == N).all()) and len(cnt) == len(sets) * len(ENDPOINTS) and bool(np.isfinite(LPz["LP"]).all())
        add("3 every participant predicted exactly once per set and endpoint (finite lp, test rows sum to 9,866)", scen,
            "set-endpoints", len(sets) * len(ENDPOINTS), int((cnt == N).sum()), 0, once)
        add("4 every inner fit converged", scen, "fits", len(fits), int(fits.converged.sum()), 0, bool(fits.converged.all()),
            f"maximum Newton iterations {int(fits.iterations.max())}")
    # (3) outcome perturbation and (4) solver checks on samples; (5) positive control
    for si, scen in enumerate(ALL_SCEN):
        sets = sets_for(scen)
        cols = columns_for(scen)
        fits = pd.read_parquet(need(PRIV / f"private_fits_{scen}.parquet")).set_index(["endpoint", "outer_fold", "inner_fold", "set"])
        LPz = np.load(need(PRIV / f"private_crossfit_{scen}.npz"))
        prep = pd.read_csv(need(OUTP / f"P_preparation_{DATE}.csv"))
        pc = prep[(prep.scenario == scen) & (prep.column == "age_sex")]
        add("5 positive control (age_sex) not estimable in every endpoint and fold", scen, "endpoint-folds", 40,
            int((~pc.estimable.astype(bool)).sum()), 0, len(pc) == 40 and not pc.estimable.astype(bool).any(),
            f"max residual/raw SD {pc.residual_to_raw_sd.max():.3g}")
        for k, ep in enumerate(ENDPOINTS):
            g = EP(A[scen], preds, ep, cols)
            a = innerz[ep]
            # outcome perturbation
            f = 1 + k % 5
            j = k % N_INNER
            H = np.flatnonzero(g.fold == f)
            z, est, _ = fold_columns(g, H, cols)
            lp_ref = LPz["LP"][:, k, H]
            lp0, _ = crossfit_fold(g, H, a[H], z, est, sets)
            rng = np.random.default_rng([SEED, 3, si, k])
            out = np.flatnonzero(g.fold != f)
            ev1, du1 = g.ev.copy(), g.du.copy()
            p_ = rng.permutation(len(out)); ev1[out] = g.ev[out][p_]; du1[out] = g.du[out][rng.permutation(len(out))]
            lp1, _ = crossfit_fold(g, H, a[H], z, est, sets, ev=ev1, du=du1)
            hj = H[a[H] == j]
            loc = np.flatnonzero(a[H] == j)
            ev2, du2 = g.ev.copy(), g.du.copy()
            ev2[hj] = g.ev[hj][rng.permutation(len(hj))]; du2[hj] = rng.permutation(g.du[hj]) * 1.5
            lp2, _ = crossfit_fold(g, H, a[H], z, est, sets, ev=ev2, du=du2)
            add("3 recomputed cross-fit equals the stored linear predictors", scen, f"{ep} fold {f}", 0.0,
                float(np.max(np.abs(lp0 - lp_ref))), 0, np.array_equal(lp0, lp_ref))
            add("3 outcomes outside H permuted: every lp of H unchanged bitwise", scen, f"{ep} fold {f}", 0.0,
                float(np.max(np.abs(lp1 - lp0))), 0, np.array_equal(lp1, lp0))
            add("3 outcomes of inner fold j permuted: lp of inner fold j unchanged bitwise", scen, f"{ep} fold {f} inner {j}", 0.0,
                float(np.max(np.abs(lp2[:, loc] - lp0[:, loc]))), 0, np.array_equal(lp2[:, loc], lp0[:, loc]))
            # solver sample
            rng4 = np.random.default_rng([SEED, 4, si, k])
            extra = (int(rng4.integers(1, 6)), int(rng4.integers(0, N_INNER)), list(sets)[int(rng4.integers(0, len(sets)))])
            for (ff, jj, sname) in [(1, 0, "P1"), (1, 0, "C8_mean"), (1, 0, "S0"), extra]:
                Hf = np.flatnonzero(g.fold == ff)
                zf, estf, _ = fold_columns(g, Hf, cols)
                inn = a[Hf]
                tr = np.flatnonzero(inn != jj)
                T = Hf[tr]
                mu, sd = g.age_v[T].mean(), g.age_v[T].std()
                use = [c for c in sets[sname][0] if estf[c]]
                Z = design(g, T, {c: v[tr] for c, v in zf.items()}, use, mu, sd)
                pen = np.r_[0.0, 0.0, np.ones(len(use))]
                b0, _, conv0, _ = cox_fit(Z, g.du[T], g.ev[T], np.zeros(len(pen)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ph = PHReg(g.du[T], Z, status=g.ev[T].astype(int), ties="breslow").fit(disp=0)
                meth = "newton (default)"
                if not np.all(np.isfinite(ph.params)):      # AMENDMENT 1
                    ph = PHReg(g.du[T], Z, status=g.ev[T].astype(int), ties="breslow").fit(disp=0, method="bfgs", maxiter=5000,
                                                                                           gtol=1e-10)
                    meth = "bfgs (default Newton returned non-finite coefficients; amendment 1)"
                d0 = float(np.max(np.abs(b0 - ph.params)))
                add("4 penalty 0: P's solver equals statsmodels PHReg (Breslow)", scen, f"{ep} fold {ff} inner {jj} {sname}", 0.0, d0,
                    TOL_PHREG, conv0 and d0 <= TOL_PHREG, f"{Z.shape[1]} coefficients; PHReg optimiser {meth}")
                stored_b = np.array(json.loads(fits.loc[(ep, ff, jj, sname), "beta"]))
                rs = minimize(neg_pen_pl_independent, np.zeros(Z.shape[1]), args=(Z, g.du[T], g.ev[T].astype(bool), pen), jac=True,
                              method="BFGS", options={"gtol": 1e-10, "maxiter": 20000})
                d1 = float(np.max(np.abs(stored_b - rs.x)))
                add("4 ridge: stored coefficients equal an independent scipy minimisation", scen, f"{ep} fold {ff} inner {jj} {sname}",
                    0.0, d1, TOL_SCIPY, d1 <= TOL_SCIPY, f"BFGS: {rs.message}")
        log(f"checks: {scen} samples done ({time.time() - t0:.0f} s)")
    # (6) bootstrap weights and counting function
    Wp = shared_weights()
    hw = sha256_bytes(Wp.tobytes())
    add("6 bootstrap weights: every shared draw sums to 9,866", "all", "2,000 draws", N, int(Wp.sum(1).min()), 0,
        bool((Wp.sum(1) == N).all()), f"sha256 {hw}")
    write_json(PRIV / "private_boot_weights.json", {"sha256": hw})
    pos = {s: i for i, s in enumerate(ids)}
    sens = within_fold_weights(preds)
    LPz = np.load(need(PRIV / "private_crossfit_full.npz"))
    ep = ENDPOINTS[0]
    g = A["full"][A["full"].endpoint == ep]
    R = risk_matrix("full", A["full"], ep, 0, LPz)
    s = np.flatnonzero(g.outer_fold.to_numpy() == 1)
    ev, du = g.event.astype(bool).to_numpy()[s], g.duration.to_numpy(float)[s]
    xr = np.array([pos[x] for x in g.dnam_id])
    Ws = [None, Wp[:, xr[s]], sens[ep][1]]
    mine, ref = fold_counts(ev, du, R[:, s], Ws), J.fold_counts(ev, du, R[:, s], Ws)
    same = all(np.array_equal(np.asarray(a_[0]), np.asarray(b_[0])) and np.array_equal(a_[1], b_[1]) for a_, b_ in zip(mine, ref))
    add("6 counting function equals J.fold_counts exactly", "full", f"{ep} fold 1, {len(R)} columns, point and both resamples", "", "",
        0, same)
    tab = pd.DataFrame(rows)
    atomic_csv(OUTP / f"P_checks_{DATE}.csv", tab)
    nf = int((~tab["pass"].astype(bool)).sum())
    log(f"checks: {len(tab)} rows, {nf} failed ({time.time() - t0:.0f} s)")
    if nf:
        print(tab[~tab["pass"].astype(bool)].to_string())
        sys.exit("checks failed; no new number is used")
    done_marker("checks", {"rows": len(tab), "failed": 0, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage boot (sharded by scenario)
def stage_boot(shard, nshards):
    require_committed()
    require_done("checks")
    t0 = time.time()
    preds = J.load_preds()
    ids = load_ids()
    pos = {s: i for i, s in enumerate(ids)}
    Wp = shared_weights()
    if sha256_bytes(Wp.tobytes()) != json.loads(need(PRIV / "private_boot_weights.json").read_text())["sha256"]:
        sys.exit("bootstrap weights differ from those checked")
    sens = within_fold_weights(preds)
    (PRIV / "boot").mkdir(parents=True, exist_ok=True)
    for scen in shard_of(ALL_SCEN, shard, nshards):
        fo = PRIV / "boot" / f"private_boot_{scen}.npz"
        if fo.exists():
            continue
        A = assemble(scen, preds)
        LPz = np.load(need(PRIV / f"private_crossfit_{scen}.npz"))
        models = models_for(scen)
        point = np.zeros((len(models), len(ENDPOINTS)))
        bprim = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        bsens = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        for j, ep in enumerate(ENDPOINTS):
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            xr = np.array([pos[s] for s in g.dnam_id])
            R = risk_matrix(scen, A, ep, j, LPz)
            acc = {"p": [np.zeros(len(models)), np.zeros(len(models)), 0.0],
                   "b": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)],
                   "s": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]}
            for f in sorted(g.outer_fold.unique()):
                s = np.flatnonzero(g.outer_fold.to_numpy() == f)
                res = fold_counts(ev[s], du[s], R[:, s], [None, Wp[:, xr[s]], sens[ep][f]])
                for key, (tot, arr) in zip(("p", "b", "s"), res):
                    acc[key][0] += arr[..., 0]; acc[key][1] += arr[..., 1]; acc[key][2] = acc[key][2] + tot
            point[:, j] = J.cidx_from(*acc["p"])
            bprim[:, j] = J.cidx_from(acc["b"][0], acc["b"][1], acc["b"][2][None, :])
            bsens[:, j] = J.cidx_from(acc["s"][0], acc["s"][1], acc["s"][2][None, :])
            log(f"{scen}: {ep} done ({time.time() - t0:.0f} s)")
        tmp = fo.with_name(fo.stem + ".tmp.npz")
        np.savez(tmp, models=np.array(models), point=point, boot=bprim, boot_within_fold=bsens)
        os.replace(tmp, fo)
        log(f"{scen}: point and bootstrap C-indices ({time.time() - t0:.0f} s)")
    done_marker(f"boot_shard{shard}of{nshards}", {"seconds": round(time.time() - t0)})
    if all((PRIV / "boot" / f"private_boot_{s}.npz").exists() for s in ALL_SCEN):
        done_marker("boot", {"draws": N_BOOT})


# ------------------------------------------------------------------ stage analyse
def pct(d, q):
    return float(np.percentile(d, q))


def outcome_of(rel, lo, mbest, na="not applicable (M_best <= 0)"):
    return na if not mbest > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")


def ref_text(o):
    return o.replace("PASS", "meets").replace("FAIL", "does not meet")


def stage_analyse():
    require_committed()
    require_done("boot")
    preds = J.load_preds()
    events = np.array([int(preds[ep].event.sum()) for ep in ENDPOINTS], float)
    jr = json.loads(need(J.OUTJ / f"J_results_{J.DATE}.json").read_text())["scenarios"]
    mr = json.loads(need(MM.OUTM / f"M_results_{MM.DATE}.json").read_text())["scenarios"]
    prep = pd.read_csv(need(OUTP / f"P_preparation_{DATE}.csv"))
    per, cons, summ, rules, ew, loeo, named, recal, draws = [], [], [], [], [], [], [], [], []
    res = {"settings": {"seed": SEED, "draws": N_BOOT, "inner_folds": N_INNER, "ridge": "0.5 x sum of squared risk-column coefficients",
                        "rule": RULE, "primary": PRIMARY,
                        "written": "8 Nov 2026, after the results of analyses J, K, L, M and N were known; no stacked set evaluated before the design commit"},
           "scenarios": {}}
    for scen in ALL_SCEN:
        z = np.load(need(PRIV / "boot" / f"private_boot_{scen}.npz"), allow_pickle=False)
        models = list(z["models"].astype(str))
        if models != models_for(scen):
            sys.exit(f"{scen}: bootstrap models differ from the design")
        P, B, S = z["point"], z["boot"], z["boot_within_fold"]
        ia, io = models.index("age_sex"), models.index("P1")
        dP, dB, dS = P - P[ia], B - B[ia], S - S[ia]
        M, MB, MS = dP.mean(1), dB.mean(1), dS.mean(1)
        dec = decision_sets(scen)
        dec_i = [models.index(m) for m in dec]
        best = max(dec, key=lambda m: M[models.index(m)])
        kb = models.index(best)
        is_dec = scen == PRIMARY
        row_status = "decision" if is_dec else "secondary scenario: rule arithmetic for reference only"
        others = [m for m in models if m not in ("P1", "age_sex")]
        for k, m in enumerate(models):
            draws.append(pd.DataFrame({"scenario": scen, "model": m, "draw": np.arange(N_BOOT), "M_shared": MB[k], "M_within_fold": MS[k]}))
        # per endpoint, every model
        pmat = np.full((len(models), len(ENDPOINTS)), np.nan)
        qmat = np.full_like(pmat, np.nan)
        for k in range(len(models)):
            if k not in (io, ia):
                pmat[k] = [J.pboot(dB[io, j] - dB[k, j]) for j in range(len(ENDPOINTS))]
                qmat[k] = MM.bh(pmat[k])
        for k, m in enumerate(models):
            for j, ep in enumerate(ENDPOINTS):
                d = dB[io, j] - dB[k, j]
                per.append({"scenario": scen, "label": label(scen), "endpoint": ep, "model": m, "role": role(scen, m), "members": members(scen, m),
                            "n": len(preds[ep]), "events": int(events[j]), "c_index": P[k, j], "c_age_sex": P[ia, j], "delta_c": dP[k, j],
                            "P1_minus_model": dP[io, j] - dP[k, j], "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5),
                            "p_boot": pmat[k, j], "q_bh_across_endpoints": qmat[k, j]})
        # per endpoint contrasts with C* and own best
        own = [dec_i[int(np.argmax(dP[dec_i, j]))] for j in range(len(ENDPOINTS))]
        p_own = [J.pboot(dB[io, j] - dB[own[j], j]) for j in range(len(ENDPOINTS))]
        q_own = MM.bh(p_own)
        tot_diff = M[io] - M[kb]
        for j, ep in enumerate(ENDPOINTS):
            db, do = dB[io, j] - dB[kb, j], dB[io, j] - dB[own[j], j]
            diff, diffo = dP[io, j] - dP[kb, j], dP[io, j] - dP[own[j], j]
            cons.append({"scenario": scen, "label": label(scen), "endpoint": ep, "events": int(events[j]), "c_age_sex": P[ia, j],
                         "c_P1": P[io, j], "delta_c_P1": dP[io, j], "best_comparator": best, "c_best": P[kb, j], "delta_c_best": dP[kb, j],
                         "difference_vs_best": diff, "ci95_low_vs_best": pct(db, 2.5), "ci95_high_vs_best": pct(db, 97.5),
                         "p_boot_vs_best": pmat[kb, j], "q_bh_vs_best": qmat[kb, j],
                         "contribution_to_M_difference": diff / len(ENDPOINTS),
                         "share_of_M_difference": (diff / len(ENDPOINTS)) / tot_diff if tot_diff != 0 else np.nan,
                         "own_best_comparator": models[own[j]], "delta_c_own_best": dP[own[j], j], "difference_vs_own_best": diffo,
                         "ci95_low_vs_own_best": pct(do, 2.5), "ci95_high_vs_own_best": pct(do, 97.5), "p_boot_vs_own_best": p_own[j],
                         "q_bh_vs_own_best": q_own[j],
                         "relative_gain_vs_own_best": diffo / dP[own[j], j] if dP[own[j], j] > 0 else np.nan,
                         "rule_arithmetic_vs_own_best_for_reference": ("undefined (own best gain <= 0)" if not dP[own[j], j] > 0 else
                                                                       ("meets" if diffo / dP[own[j], j] >= 0.10 and pct(do, 2.5) > 0 else "does not meet"))})
        # summary: every set and stored model against P1
        for k, m in enumerate(models):
            if m == "age_sex":
                continue
            base = {"scenario": scen, "label": label(scen), "model": m, "members": members(scen, m), "role": role(scen, m),
                    "is_best_decision_comparator": m == best, "M": M[k], "M_ci95_low": pct(MB[k], 2.5), "M_ci95_high": pct(MB[k], 97.5),
                    "M_ci95_low_within_fold": pct(MS[k], 2.5), "M_ci95_high_within_fold": pct(MS[k], 97.5), "M_P1": M[io]}
            if m == "P1":
                summ.append(base)
                continue
            d, ds = MB[io] - MB[k], MS[io] - MS[k]
            rel = (M[io] - M[k]) / M[k] if M[k] > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                relb = d / MB[k]
            lo = pct(d, 2.5)
            o = outcome_of(rel, lo, M[k], "not applicable (M <= 0)")
            summ.append({**base, "difference_P1_minus_model": M[io] - M[k], "ci95_low": lo, "ci95_high": pct(d, 97.5),
                         "ci95_low_within_fold": pct(ds, 2.5), "ci95_high_within_fold": pct(ds, 97.5), "p_boot": J.pboot(d),
                         "relative_gain": rel,
                         "relative_gain_ci95_low": float(np.nanpercentile(relb, 2.5)) if M[k] > 0 else np.nan,
                         "relative_gain_ci95_high": float(np.nanpercentile(relb, 97.5)) if M[k] > 0 else np.nan,
                         "rule_arithmetic_against_this_model": o if (is_dec and m == best) else ref_text(o) + " (reference only)"})
        # rule outcome
        d, ds = MB[io] - MB[kb], MS[io] - MS[kb]
        rel = (M[io] - M[kb]) / M[kb] if M[kb] > 0 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            relb = d / MB[kb]
        lo, hi = pct(d, 2.5), pct(d, 97.5)
        outcome = outcome_of(rel, lo, M[kb])
        rules.append({"analysis": "P", "scenario": scen, "decision_row": is_dec, "row_status": row_status, "label": label(scen),
                      "metric": "mean over 8 endpoints of (pooled within-fold C-index of the cross-fitted linear predictor - age+sex C-index)",
                      "decision_model": "P1", "decision_model_members": members(scen, "P1"), "M_P1": M[io],
                      "M_P1_ci95_low": pct(MB[io], 2.5), "M_P1_ci95_high": pct(MB[io], 97.5), "best_comparator": best,
                      "best_comparator_members": members(scen, best), "M_best": M[kb], "M_best_ci95_low": pct(MB[kb], 2.5),
                      "M_best_ci95_high": pct(MB[kb], 97.5), "relative_gain": rel,
                      "relative_gain_ci95_low": float(np.nanpercentile(relb, 2.5)) if M[kb] > 0 else np.nan,
                      "relative_gain_ci95_high": float(np.nanpercentile(relb, 97.5)) if M[kb] > 0 else np.nan,
                      "difference": M[io] - M[kb], "ci95_low": lo, "ci95_high": hi,
                      "interval": f"paired bootstrap, one shared participant resample per draw, {N_BOOT} draws, seed {SEED}; cross-fitted risks fixed",
                      "ci95_low_within_fold": pct(ds, 2.5), "ci95_high_within_fold": pct(ds, 97.5), "p_boot": J.pboot(d),
                      "rule": RULE, "outcome": outcome if is_dec else f"{outcome} (reference only)"})
        sres = {"label": label(scen), "decision_row": is_dec, "M": {m: float(M[models.index(m)]) for m in models if m != "age_sex"},
                "best": best, "relative_gain": rel, "difference": float(M[io] - M[kb]), "ci95": [lo, hi],
                "ci95_within_fold": [pct(ds, 2.5), pct(ds, 97.5)], "outcome": outcome, "row_status": row_status}
        # event-weighted
        wt = events / events.sum()
        Mw, MBw = (dP * wt[None, :]).sum(1), (dB * wt[None, :, None]).sum(1)
        bw = max(dec, key=lambda m: Mw[models.index(m)])
        kw = models.index(bw)
        dw = MBw[io] - MBw[kw]
        relw = (Mw[io] - Mw[kw]) / Mw[kw] if Mw[kw] > 0 else np.nan
        ew.append({"scenario": scen, "label": label(scen), "statistic": "event-weighted mean over 8 endpoints (secondary)", "M_P1": Mw[io],
                   "best_comparator": bw, "M_best": Mw[kw], "relative_gain": relw, "difference": Mw[io] - Mw[kw], "ci95_low": pct(dw, 2.5),
                   "ci95_high": pct(dw, 97.5), "p_boot": J.pboot(dw),
                   "rule_arithmetic_for_reference": ref_text(outcome_of(relw, pct(dw, 2.5), Mw[kw]))})
        # leave one endpoint out
        for j, ep in enumerate(ENDPOINTS):
            keep = np.arange(len(ENDPOINTS)) != j
            Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
            bl = max(dec, key=lambda m: Ml[models.index(m)])
            kl = models.index(bl)
            dl, df_ = MBl[io] - MBl[kl], MBl[io] - MBl[kb]
            rell = (Ml[io] - Ml[kl]) / Ml[kl] if Ml[kl] > 0 else np.nan
            relf = (Ml[io] - Ml[kb]) / Ml[kb] if Ml[kb] > 0 else np.nan
            loeo.append({"scenario": scen, "left_out": ep, "n_endpoints": 7, "M_P1": Ml[io],
                         "best_comparator_rechosen": bl, "M_best_rechosen": Ml[kl], "relative_gain_rechosen": rell,
                         "difference_rechosen": Ml[io] - Ml[kl], "ci95_low_rechosen": pct(dl, 2.5), "ci95_high_rechosen": pct(dl, 97.5),
                         "p_boot_rechosen": J.pboot(dl),
                         "rule_arithmetic_rechosen_for_reference": ref_text(outcome_of(rell, pct(dl, 2.5), Ml[kl])),
                         "best_comparator_fixed": best, "M_best_fixed": Ml[kb], "relative_gain_fixed": relf, "difference_fixed": Ml[io] - Ml[kb],
                         "ci95_low_fixed": pct(df_, 2.5), "ci95_high_fixed": pct(df_, 97.5), "p_boot_fixed": J.pboot(df_),
                         "rule_arithmetic_fixed_for_reference": ref_text(outcome_of(relf, pct(df_, 2.5), Ml[kb]))})
        # named contrasts
        sres["named_contrasts"] = {}
        for name, a, b_ in named_contrasts(scen):
            ka, kb_ = models.index(a), models.index(b_)
            dd, dds = MB[ka] - MB[kb_], MS[ka] - MS[kb_]
            r_ = (M[ka] - M[kb_]) / M[kb_] if M[kb_] > 0 else np.nan
            named.append({"scenario": scen, "label": label(scen), "contrast": name, "set_a": a, "members_a": members(scen, a), "set_b": b_,
                          "members_b": members(scen, b_), "M_a": M[ka], "M_b": M[kb_], "difference": M[ka] - M[kb_],
                          "ci95_low": pct(dd, 2.5), "ci95_high": pct(dd, 97.5), "ci95_low_within_fold": pct(dds, 2.5),
                          "ci95_high_within_fold": pct(dds, 97.5), "p_boot": J.pboot(dd), "relative_gain": r_,
                          "rule_arithmetic_for_reference": ref_text(outcome_of(r_, pct(dd, 2.5), M[kb_], "not applicable (M_b <= 0)"))})
            sres["named_contrasts"][f"{a} vs {b_}"] = {"difference": float(M[ka] - M[kb_]), "ci95": [pct(dd, 2.5), pct(dd, 97.5)]}
        # recalibration
        sres["recalibration"] = {}
        pub = jr if scen in SCEN_J else mr
        for a, b_ in recal_pairs(scen):
            ka, kb_ = models.index(a), models.index(b_)
            dd, dds = MB[ka] - MB[kb_], MS[ka] - MS[kb_]
            recal.append({"scenario": scen, "label": label(scen), "stacked_set": a, "members": members(scen, a), "stored_model": b_,
                          "stored_source": source(scen), "M_stacked": M[ka], "M_stored": M[kb_],
                          "M_stored_published": float(pub[scen]["M"][b_]) if b_ != "age_sex" else 0.0,
                          "difference_stacked_minus_stored": M[ka] - M[kb_], "ci95_low": pct(dd, 2.5), "ci95_high": pct(dd, 97.5),
                          "ci95_low_within_fold": pct(dds, 2.5), "ci95_high_within_fold": pct(dds, 97.5), "p_boot": J.pboot(dd)})
            sres["recalibration"][f"{a} vs {b_}"] = {"difference": float(M[ka] - M[kb_]), "ci95": [pct(dd, 2.5), pct(dd, 97.5)]}
        ne = prep[(prep.scenario == scen) & (~prep.positive_control.astype(bool)) & (~prep.estimable.astype(bool))]
        sres["non_estimable_columns"] = [f"{r.endpoint} fold {r.outer_fold}: {r.column}" for r in ne.itertuples()]
        res["scenarios"][scen] = sres
        log(f"{scen}: M P1 {M[io]:.4f}; C* {best} {M[kb]:.4f}; gain {rel:+.1%}; diff {M[io] - M[kb]:+.4f} [{lo:.4f}, {hi:.4f}] -> {outcome}")
    res["decision"] = {k: res["scenarios"][PRIMARY][k] for k in ("best", "relative_gain", "difference", "ci95", "outcome")}
    files = {f"P_cindex_per_endpoint_{DATE}.csv": per, f"P_endpoint_contrasts_{DATE}.csv": cons, f"P_summary_{DATE}.csv": summ,
             f"P_rule_outcome_{DATE}.csv": rules, f"P_event_weighted_{DATE}.csv": ew, f"P_leave_one_endpoint_out_{DATE}.csv": loeo,
             f"P_named_contrasts_{DATE}.csv": named, f"P_recalibration_{DATE}.csv": recal}
    for fn, rows_ in files.items():
        atomic_csv(OUTP / fn, pd.DataFrame(rows_))
    atomic_csv(OUTP / f"P_bootstrap_draws_{DATE}.csv.gz", pd.concat(draws, ignore_index=True), compression="gzip")
    write_json(OUTP / f"P_results_{DATE}.json", res)
    write_readme(res)
    done_marker("analyse")


def write_readme(res):
    def nrow(fn):
        return "-" if fn.endswith(".json") or fn.endswith(".md") else str(len(pd.read_csv(OUTP / fn, dtype=str)))

    def cols(fn):
        return "-" if fn.endswith(".json") else ", ".join(pd.read_csv(OUTP / fn, nrows=1).columns)

    desc = {
        f"P_rule_outcome_{DATE}.csv": "One row per scenario; decision_row marks full, the pre-specified decision (other rows: rule arithmetic for reference only). M of P1, C* and its M, relative gain, difference, 95% intervals (shared and within-fold), P, rule, outcome.",
        f"P_summary_{DATE}.csv": "Per scenario, every stacked set and every unstacked stored model: members, role, M with shared and within-fold intervals, P1 minus it with intervals, P, relative gain with interval, rule arithmetic against it.",
        f"P_cindex_per_endpoint_{DATE}.csv": "Per scenario, endpoint and model: pooled within-fold C-index, age+sex C-index, gain above age+sex, P1 minus model with 95% interval, bootstrap P and Benjamini-Hochberg q across the 8 endpoints.",
        f"P_endpoint_contrasts_{DATE}.csv": "Per scenario and endpoint: P1 against the scenario's C* (difference, interval, P, q, contribution to the M difference) and against the endpoint's own best decision comparator (chosen by point estimate, fixed in the draws).",
        f"P_leave_one_endpoint_out_{DATE}.csv": "Per scenario and left-out endpoint: M over the other 7 endpoints with C* re-chosen and fixed; intervals from the same draws; rule arithmetic for reference only.",
        f"P_event_weighted_{DATE}.csv": "Per scenario: event-weighted mean gain, C* re-chosen, difference with interval, rule arithmetic for reference only.",
        f"P_named_contrasts_{DATE}.csv": "Outside the decision: P1 vs C3_mean (MethylGPT combined added to the two clocks), P1_knn vs C3_knn, P1 vs S1, S2 vs C8_mean; difference with both intervals, P, rule arithmetic for reference only.",
        f"P_recalibration_{DATE}.csv": "Each single-model stacked set (S1, C1, C2, C4, C5; S0) against J's or M's unstacked model of the same name scored from the stored risk: M of both, the published M, difference with intervals, P.",
        f"P_preparation_{DATE}.csv": "Per scenario, endpoint, outer fold and risk column: raw SD, SD of the residual on age and sex, ratio, estimable flag (L's constant-risk test, 1e-5), sets containing the column; age_sex is the positive control.",
        f"P_checks_{DATE}.csv": "Every pre-specified check with stored and recomputed values, tolerance and pass (stored C-index reproduction, identifiers, cross-fit integrity, Cox solver, positive control, bootstrap, privacy).",
        f"P_bootstrap_draws_{DATE}.csv.gz": "Per scenario, model and draw: M under the shared resample and under the within-fold resample (2,000 draws).",
        f"P_results_{DATE}.json": "Settings and, per scenario, M of every model, C*, gain, difference, intervals, outcome, named contrasts, recalibration and non-estimable columns.",
    }
    pr = res["scenarios"][PRIMARY]
    ne_all = {s: v["non_estimable_columns"] for s, v in res["scenarios"].items() if v["non_estimable_columns"]}
    lines = ["# Large-gain analysis P: GS incident disease, stored held-out risks combined by cross-fitting", "",
             "Script: `large_gain_P_gs_stacked_risks_2026-11-08.py` (pre-specified design in its docstring). J's and M's stored held-out risks "
             "(9,866 GS development participants, 8 endpoints, J's locked outer folds) are combined in a ridge Cox model fitted by 10-fold "
             "cross-fitting inside each held-out outer fold, so no participant's outcome sets its own weights. No base model is refitted. "
             "full = complete profiles (decision); hm27 and random_90 = simulated missingness on empirical profiles (J); depth_10, depth_5, "
             "depth_30 and gauss_0.05 = simulated measurement noise on empirical profiles (M). P was written after the results of analyses "
             "J, K, L, M and N were known.", "",
             "## Decision (full)", "",
             f"- P1 (MethylGPT combined + GrimAge2 + DunedinPACE, cross-fitted) M = {pr['M']['P1']:.4f}; C* = {pr['best']} M = {pr['M'][pr['best']]:.4f}.",
             f"- Relative gain {pr['relative_gain']:+.1%}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             "", "## Secondary scenarios (rule arithmetic for reference only)", ""]
    for s, v in res["scenarios"].items():
        if s != PRIMARY:
            rg = f"{v['relative_gain']:+.1%}" if v["relative_gain"] == v["relative_gain"] else "undefined"
            lines.append(f"- {s}: P1 M = {v['M']['P1']:.4f}; C* = {v['best']} M = {v['M'][v['best']]:.4f}; gain {rg}; "
                         f"difference {v['difference']:+.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]; {v['outcome']}.")
    lines += ["", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUTP / fn).exists():
            lines.append(f"| `{fn}` | {nrow(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              "- Stored held-out risks: J's private risk tables (`artifacts/large_gain_2026-10-28_private/J/`, via `J.assemble`) and M's "
              "(`artifacts/large_gain_2026-11-02_private/M/`, via `M.assemble`); endpoint table `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/"
              "private_nested_predictions.parquet` (J's locked folds, age, sex, age+sex risk).",
              "- Stored C-indices and M for the reproduction checks: `J_cindex_per_endpoint_2026-10-28.csv`, `J_results_2026-10-28.json`, "
              "`M_cindex_per_endpoint_2026-11-02.csv`, `M_results_2026-11-02.json`.",
              "", "## Notes", "",
              "- Interval: one participant resample per draw shared by every set, endpoint and scenario (2,000 draws, seed 20261108); the "
              "cross-fitted risks are fixed in the draws, so the intervals do not include the variability of the inner fits. J's within-fold "
              "resample is the sensitivity.",
              "- Every set includes age and sex (unpenalised); risk columns carry a fixed ridge (unit-variance Gaussian prior on the log hazard "
              "ratio per SD). Inner folds are stratified by event and shared by every set and scenario.",
              "- Non-estimable columns (L's constant-risk test) are dropped from that fold's fits: "
              + ("; ".join(f"{s}: {', '.join(v)}" for s, v in ne_all.items()) if ne_all else "none.") ,
              "- Inner-fold event counts stay in the private directory (some inner folds have fewer than 5 events).",
              "- Participant-level linear predictors, inner folds, fit records and bootstrap arrays stay in "
              "`artifacts/large_gain_2026-11-08_private/P/` on Chronos and are never committed."]
    (OUTP / "README.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ stage privacy
def stage_privacy():
    require_committed()
    require_done("analyse")
    ids = set(load_ids())
    token = re.compile(r"[A-Za-z0-9_]+")
    forbidden = {"dnam_id", "sample_id", "participant", "participant_id", "row", "id"}
    rows = []
    for p in sorted(OUTP.iterdir()):
        text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
        hits = len(set(token.findall(text)) & ids)
        bad_cols, small = [], 0
        if ".csv" in p.name:
            df = pd.read_csv(p, dtype=str)
            bad_cols = sorted(set(c.lower() for c in df.columns) & forbidden)
            for c in [c for c in df.columns if c.lower().startswith("events")]:
                v = pd.to_numeric(df[c], errors="coerce")
                small += int((v < SMALL_CELL).sum())
        rows.append({"check": "7 privacy of public files", "scenario": "all", "item": p.name, "stored": 0, "recomputed": hits + len(bad_cols) + small,
                     "abs_diff": "", "tolerance": 0, "pass": hits == 0 and not bad_cols and small == 0,
                     "note": f"identifier tokens {hits}; identifier columns {bad_cols}; event cells below {SMALL_CELL}: {small}"})
    ck = pd.read_csv(need(OUTP / f"P_checks_{DATE}.csv"))
    ck = ck[ck.check != "7 privacy of public files"]
    out = pd.concat([ck, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(OUTP / f"P_checks_{DATE}.csv", out)
    write_readme(json.loads(need(OUTP / f"P_results_{DATE}.json").read_text()))
    nf = sum(not r["pass"] for r in rows)
    log(f"privacy: {len(rows)} public files, {nf} failed")
    if nf:
        print(pd.DataFrame(rows).query("~`pass`").to_string())
        sys.exit("privacy check failed; do not commit the outputs")
    done_marker("privacy", {"files": len(rows)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "checks", "boot", "analyse", "privacy"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    os.chdir(HERE)
    LOGS.mkdir(parents=True, exist_ok=True)
    {"prep": lambda: stage_prep(a.workers), "checks": stage_checks, "boot": lambda: stage_boot(a.shard, a.nshards),
     "analyse": stage_analyse, "privacy": stage_privacy}[a.stage]()


if __name__ == "__main__":
    main()
