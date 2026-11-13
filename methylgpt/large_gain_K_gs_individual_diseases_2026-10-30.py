#!/usr/bin/env python3
"""Large-gain analysis K: Generation Scotland (GS) individual incident diseases, MethylGPT against J's comparators, with
27K-array CpGs as the primary setting. SIMULATED MISSINGNESS ON EMPIRICAL PROFILES at hm27 and random_90: every input is a
measured GS development profile from which CpGs are removed by J's fixed masks; no methylation value is simulated. The
complete-profile setting uses the measured profiles unchanged.

Written on 30 Oct 2026 for the large-gain analysis plan of 23 Oct 2026 (Decision rule) and the analysis brief for
K, BEFORE any result of this analysis is computed. Every result is reported, whichever model it favours.

WHY: analysis J compared MethylGPT with comparators on 8 broad GS endpoints; the mean over endpoints gives a small absolute
C-index difference (hm27: 0.0074 against GrimAge2 with kNN fill), and most endpoints are broad categories that pool many
diseases. K asks, with comparators unchanged, whether individual diseases show larger gaps than the broad-category means.

DATA AND INCLUSION
  Participants: the 9,866 GS development participants of the locked protocol (frame-cache matrix, J's load_gs). Validation
  and test participants are never read: the label code builds tables on all 18,859 rows because that is how it works and
  the checks need it, but only development rows are kept, and no methylation or model output of any other row is loaded.
  Constituent diseases: mapped_disease_columns of each broad endpoint in artifacts/gs_cox/gs_endpoint_definitions_common_
  split_2026-06-29_audit.json (autoimmune 19, cancers 13, cardiovascular 10, respiratory 7, neurological 4; 53 diseases).
  Labels: build_endpoint_table (regenerate_disease_survival_data.py, imported from revision_server_20260622/server_results/
  task_a/code, unchanged; label mode incident_positive_duration; prepare_task_a_common_split_2026_06_22.py calls it the same
  way), given a one-row mapping that names the single disease column, so the event date is that column's date and every
  other line of the label code is unchanged.
  At risk: every development participant is free of prevalent disease in all broad categories, so all are at risk for each
  constituent. The freeze verifies this (each constituent table holds all 9,866 development participants). A participant
  who has another constituent disease first stays at risk for this one (the broad-endpoint label code has no competing-risk
  handling and none is added). Sex-specific diseases keep all participants, as the brief fixes; see SENSITIVITIES.

ELIGIBILITY AND PRIMARY SET (frozen before any fit)
  Constituents of the five multi-disease categories with >= 50 incident events among the development participants. Type 2
  diabetes, kidney disease and death are outside the primary set because K was prompted by the type 2 diabetes result; they
  are context only: their rows are copied from J's per-endpoint table (same models, J's locked folds and fold models), no
  refit, and they enter no statistic of K.
  The stage freeze writes K_disease_table (constituent, category, role, n_at_risk, events, eligible, fold seed and fold
  hash; cells below 5 events shown as "<5") and K_freeze_checks; both are committed before any fit (stage features and every
  later stage refuse to run otherwise). If fewer than 5 diseases are eligible the analysis stops and reports rather than
  lowering the threshold.

FOLDS
  For each disease: 5 outer folds, StratifiedKFold(n_splits=5, shuffle=True, random_state=s) on the event indicator over the
  participants in frame-cache order, the locked protocol's outer splitter, with s = first 4 bytes (big-endian) of SHA-256 of
  "20261030:<disease column name>", so the seeded generator is keyed by the disease name. The same assignment is used by
  every model of that disease; folds differ between diseases. This differs from J (locked folds of broad endpoints), which
  cannot serve an individual disease because stratification is by that disease's own events.

MODELS (each fitted per disease and outer fold on the complete profiles of the training folds, applied unchanged to the
held-out participants, whose inputs come from J's masks; nothing held-out enters a fit, a scaler or a fill reference)
  age_sex: fit_baseline of compare_joint_cox_mgpt2_2026-10-01.py (Breslow Cox on standardised age and sex). If its strict
    warning-to-error filter raises because sex separates events (a sex-specific disease), the same estimator is refitted
    without that filter, as J's clock Cox fits run, and the warning is recorded in K_fit_record.
  MethylGPT combined (mgpt2_combined): the same three feature sets and the same equal-weight combination as J's mgpt2_combined:
    embedding (2,048, runs/aging_heads_clean_2026-10-01/best.pt) and biomarkers (25, runs/multitask_bio_2026-10-01/final.pt):
      the locked nested elastic-net Cox of compare_joint_cox_mgpt2_2026-10-01.py (sksurv Coxnet, age and sex unpenalised,
      l1_ratio 0.5, 12 alphas down to 0.05 alpha_max, tol 1e-7, max_iter 100,000; inner StratifiedKFold(3, shuffle,
      random_state 20260928 + outer fold), selection = largest mean inner C-index, refit on the training folds). Fit inputs
      are the stored full-profile features (gs_features_2026-10-01/*_mgpt2_embed, *_mt_bio), as the locked runs.
    factor_residual ([embedding, biomarkers, core residuals], 51,229 columns): gs_factor_residual_cox_2026-10-01.py (GPU
      elastic net of gpu_coxnet_2026_10_01.py, inner 3 folds seed 20260928 + outer fold, 12 alphas, 200 FISTA iterations
      per alpha). The locked run leaves the power iteration's start unseeded; K seeds torch per disease, model and fold.
    Held-out features are recomputed from each setting's kept CpGs by J's stage_mgpt code (batch 32, TF32); missing residuals
    and any non-finite feature take the fold mean. mgpt2_combined = sum of the three risks, each standardised within the
    held-out fold, as J. Components and ensemble_embed_bio are tabulated per disease, as J.
  Decision comparators, identical in kind to J's:
    raw_cpg_v12_{mean,knn}: the locked raw-CpG nested Cox (compare_joint_cox, the same as the embedding model on 49,156 CpGs,
      max_iter 1,000,000, 2 threads), held-out missing CpGs = fold means, or J's corpus-kNN fill.
    raw_cpg_gpu_{mean,knn}: the GPU elastic net above on the raw CpGs, same two fills.
    grimage2_{mean,knn}, dunedinpace_{mean,knn}, grimage2_dunedinpace_{mean,knn}: J's unpenalised Breslow Cox (n_iter 200) on
      standardised age, sex and clock score(s), fitted on the training folds' stored full-profile scores, applied to the
      held-out participants' scores of the setting (J's clock_scores, mean-filled and corpus-kNN-filled).
  Secondary comparator (reported, outside the decision set): in-cohort kNN (suffix _cohortknn) as in J: k = 10, reference =
    complete profiles of the training folds of the participant's outer fold for that disease. Because K's folds differ from
    J's, these fills and the clock scores built on them cannot be reused and are recomputed by J's code path (stage_cohort,
    clockin, clocks) with K's folds.
  MethylGPT application (reported, not a comparator): raw_cpg_v12_factor_imputed, raw_cpg_gpu_factor_imputed: the same raw
    fold models on J's factor-imputed profiles (mu + V s from the kept CpGs, clipped to [0, 1]), every setting.
  No comparator of J is removed. J's _nvobs variants exist only in sequencing scenarios, which K does not run. If the
  raw-CpG protocols are too expensive for both to run within about 24 h, one disease is measured first and a dated
  amendment states, before any result is computed, which raw protocol stays and why, keeping the stronger of the two on
  complete profiles (raw_cpg_gpu, M 0.0297 in J).
  Reuse: J's masks, corpus kNN fills (full 49,156 CpGs and clock CpGs) and clock scores of full, hm27 and random_90, from
  artifacts/large_gain_2026-10-28_private/J, after verifying them against J_masks and J_knn_record (CHECKS).

SETTINGS
  hm27 PRIMARY; full (complete profiles) and random_90 secondary, with J's masks and seeds. Labels: hm27 and random_90 are
  simulated missingness on empirical profiles; full is the measured data.

PRIMARY METRIC AND DECISION (plan, Decision rule for discrimination)
  C(model, d) = pooled within-fold C-index of disease d (concordant + tied/2 over comparable pairs summed over its outer
  folds; sksurv concordance_index_censored arithmetic, tied_tol 1e-8; J's fold_counts).
  M(model) = mean over primary-set diseases of (C(model, d) - C(age_sex, d)).
  Decision set: the 4 raw-CpG and 6 clock models above. C* = the decision comparator with the largest M at hm27.
  PASS iff (M_MethylGPT - M_C*) / M_C* >= 0.10 and the lower end of the 95% interval of M_MethylGPT - M_C* is > 0; otherwise
  FAIL. If M_C* <= 0 the relative gain is undefined and the rule is reported as not applicable. The same quantities are
  reported for full and random_90 (each with its own C*; full uses the mean-fill models only, as J), as secondary rows.

INTERVALS: paired bootstrap of participants, 2,000 draws, seed 20261030: I = default_rng(20261030).integers(0, 9866, size=
  (2000, 9866)); multiplicities w_b = bincount(I[b], minlength=9866) in frame-cache order, one resample shared by every model,
  disease and setting in a draw; per disease the pooled within-fold C-index is recomputed over its folds with pair weights
  w_i w_j (J's fold_counts; a participant and its copies are never comparable; per-fold sizes vary between draws). Risks
  are not refitted. Percentile 95% intervals, statistic recomputed in every draw. Bootstrap P = min(1, 2 (k + 1) / (B + 1)),
  k = min(#{d_b <= 0}, #{d_b >= 0}).

SECONDARY (all reported)
  Per disease and setting: C-index of every model; gain above age+sex; difference from that disease's own best decision
  comparator (largest gain at that disease and setting; the point estimate picks it, the draws keep it fixed, so that
  interval does not include the selection) and from GrimAge2 (grimage2_knn and grimage2_mean at hm27 and random_90,
  grimage2_mean at full) with its interval; Benjamini-Hochberg q across primary-set diseases within each setting and contrast.
  Event-weighted mean as a sensitivity (weights = incident events of each disease, C* re-chosen).
  Number of diseases where MethylGPT meets the rule against its own best comparator (gain >= 10% with interval above 0; own
  best gain <= 0 is counted as undefined), and the number where a comparator beats MethylGPT (point estimate, and interval
  below 0). Selection of "top" diseases is not allowed in the primary; every eligible disease is tabulated.
SENSITIVITIES (added by this script, outside the brief's list; reported only, no decision)
  sex_restricted: for Prostate_cancer (men), Breast_cancer, Ovary_cancer and Uterine_corpus_cancer (women), a list fixed here
  from the definitions of these diseases and not from data, if eligible: the same risks, with comparable pairs restricted
  to participants of that sex, C and age+sex C both, then M as above (C* re-chosen). All-participant pairs in these diseases
  include the between-sex pairs that sex alone orders, which shrinks every model's gain.
  complete_v12_paths_mean: see the second amendment 2026-10-30.

CHECKS (written to K_checks before any new number is used; stage boot refuses to run unless all pass)
  (1) Labels: constituents pooled per participant (event = any, duration = earliest event or censoring) equal the stored
      endpoint event and duration of all 9,866 participants for the five categories, event counts equal the stored counts,
      and the label code with the true mapping reproduces the audit's rows and events for all 8 endpoints.
  (2) At risk: every constituent table holds all 9,866 development participants; no prevalent or same-month event among them;
      no disease column is in two categories.
  (3) Reuse: mask SHA-256 equal J_masks (vocabulary and clock masks of full, hm27, random_90); corpus reference row SHA-256
      recomputed and equal J_knn_record and G's; fill counts equal J_knn_record, kept values unchanged, masked values finite;
      J's fold models applied to the fills reproduce J's stored kNN risks on Neurological_Diseases (|delta C| <= 1e-6); J's
      mean-fill clock scores equal the stored scores within 1e-6.
  (4) Features: recomputed full-profile embedding and biomarkers equal the stored features within 1e-3; J's fold models on
      K's features reproduce J's stored risks on Neurological_Diseases at every setting (|delta C| <= 1e-6).
  (5) Fit reproduction: the fitting and applying code of K, run on Neurological_Diseases with its locked folds (target
      REPRO_Neurological_Diseases, rows in the locked order for the GPU fits), against J's stored risks at full, hm27 and
      random_90. TOLERANCES STATED BEFORE RUNNING: sksurv and Cox models (age_sex, embedding, biomarkers, ensemble_embed_bio,
      raw_cpg_v12 with mean, knn and factor_imputed fills, clocks with mean and knn fills): |delta C| <= 1e-5 and maximum
      absolute risk difference <= 1e-3 (1 + max|stored risk|). GPU-fitted models and anything built on them, and every
      in-cohort kNN model (factor_residual, raw_cpg_gpu in every fill, mgpt2_combined, *_cohortknn): |delta C| <= 1e-3 and
      minimum within-fold Pearson r >= 0.99, because the locked GPU fits start an unseeded power iteration and the kNN
      search runs in a different reference row order.
  (6) Leakage: per target, family and fold, the training rows recorded in the model equal the fold's training rows, are
      disjoint from its held-out rows, and the two cover all 9,866; in-cohort references equal the training rows; no GS
      participant identifier is in the corpus kNN reference.
  (7) Folds: per disease every fold holds its share of events (max - min <= 1) and the fold hash equals the frozen one.
  (8) Pair counting equals sksurv's pooled C-index for every primary disease and model at hm27; the weighted counts equal
      gs_stacked_2026-10-04.PairCounter on one disease-fold for 5 draws.
  (9) Privacy (stage privacy, appended after the outputs): no public file has a participant identifier, an identifier column
      or a cell below 5 events.

EXECUTION
  Stages (resumable, atomic writes; done-markers artifacts/logs/large_gain_K_<stage>_2026-10-30.done): freeze (CPU) ->
  features (GPU) -> fit --family cheap|gpu|v12 (sharded; each fold model is an atomic file) -> cohort (GPU, sharded) ->
  clockin (CPU) -> clocks (biolearn interpreter) -> apply (CPU, sharded) -> checks -> boot -> analyse -> privacy.
  repcheck is an optional early look at the REPRO target only. A fit failure is written to private/failures and stops the
  apply stage until a dated amendment resolves it. No C-index of a K disease is computed before stage checks, whose item (8)
  compares two computations of it and records only their maximum difference; the C-indices themselves come from stage boot. Every stage
  refuses to run unless this script and J's are committed unchanged. Run through run_large_gain_K_2026-10-30.sh in a
  transient systemd user unit. Execution-only changes after the commit go in a dated AMENDMENT paragraph below, committed
  before their output is used.
OUTPUTS
  Aggregate tables: artifacts/large_gain_2026-10-30/K/ (README.md, K_disease_table, K_freeze_checks, K_checks, K_fit_record,
  K_cindex_per_disease, K_disease_contrasts, K_direction_counts, K_summary, K_rule_outcome, K_context_endpoints,
  K_bootstrap_draws, K_results). Participant-level labels, features, models, fills, clock inputs and scores, risks and
  bootstrap arrays: artifacts/large_gain_2026-10-30_private/K/ (inside the git-ignored artifacts/, never committed, stay on
  Chronos). Tables only: no figures, no manuscript edits.
Run: python3 large_gain_K_gs_individual_diseases_2026-10-30.py --stage <stage> [--family f] [--shard i --nshards n] [--device cuda:N]
     (stage clocks: /home/kying/anaconda3/envs/integrative/bin/python ... --stage clocks [--workers 4])

AMENDMENTS (dated, committed before their output is used)
  2026-10-30, numerical, before any C-index of a K disease is computed. raw_cpg_v12 on Inflammatory_Bowel_Disease raised
  sksurv's ArithmeticError ("weights are too large") at weak penalties (31 events in an inner fit, 49,158 columns); the error
  discards the whole path, so the docstring's rule stopped that disease until this amendment. Change to fit_path_k only: on
  ArithmeticError the path is refitted with n_alphas = k and alpha_min_ratio = 0.05^((k - 1) / 11), which are the first k
  points of the locked 12-point grid, with k the largest value that succeeds, found by bisection. Verified on the failing fit
  with no outcome and no held-out row used: largest k = 7, and the k = 6 and k = 7 paths agree at their common points
  (relative alpha difference 9e-16, maximum absolute coefficient difference 2e-10), so a truncated path equals the locked
  path up to where it stops. Selection already works over the alpha indices common to the three inner paths, and the final
  fit is limited in the same way; K_fit_record gives every path length. Paths that complete are unchanged, so fold models
  already fitted stay valid. raw_cpg_v12 stays for every disease; no comparator is removed. The brief's cost option is not
  used: measured v12 folds take 130-290 s, so both raw protocols run within the budget.
  2026-10-30, execution only: stage fit takes --only-target and --only-fold, so that one target's folds can run in parallel.
  2026-10-30, numerical, second amendment, before any C-index of a K disease is computed. The six v12 workers of fit_main
  started before the first amendment and recorded ArithmeticError on 12 folds (Inflammatory_Bowel_Disease 5,
  Congestive_heart_failure 4, Depression 3). Rerun with the first amendment, Inflammatory_Bowel_Disease folds 1-3 stopped with
  sksurv's convergence warning, which also marks a failing point of the path. Change to fit_path_k only: the prefix search
  treats that warning as it treats ArithmeticError, so every kept path is converged at all its points. Measured on training
  rows of 4 of the failing folds, no held-out row and no C-index: of 12 inner paths, 5 stop short (lengths 1, 1, 4, 1, 1 in
  fits with 30-32 events; the 4-point path equals the 3-point path at their common points). The cause is not established;
  the largest training |z| of a CpG is about 64 in these fits, and also in a fit whose path completes. Refitting through a
  finer warm-started grid does not help: on all 4 inner fits tested sksurv ended that grid after 6 points with no CpG in the
  model (subdivisions 4 and 16), so it is not used. A path of length 1 holds the largest penalty only, where no CpG has a
  coefficient, so a fold with such an inner path selects alpha index 0 and its raw_cpg_v12 model has age and sex only.
  raw_cpg_v12 is therefore weaker in those folds than a completed path could be; raw_cpg_gpu uses another solver and all its
  folds completed; both stay in the decision set. Reported
  only, no decision: statistic complete_v12_paths_mean is the mean gain over the diseases whose raw_cpg_v12 paths all have
  12 points (read from the model files, C* re-chosen), so the comparison can be read without any truncated path.
  K_fit_record gives every path length and selected index.
  2026-11-01, after the primary result (hm27 mean FAIL, +4.1%), outside the decision; no fit, estimate, table or decision
  changes. One disease carries a large share of M in either direction (Inflammatory_Bowel_Disease, age+sex C 0.425; COPD), so
  stage sensitivity writes K_leave_one_disease_out: M over the other 12 diseases at every setting, C* re-chosen, interval from
  the same draws. Rule arithmetic is shown for reference only. Post hoc: it does not change the outcome of K.
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
DATE = "2026-10-30"
OUTK = ART / "large_gain_2026-10-30/K"
PRIV = ART / "large_gain_2026-10-30_private/K"
FEAT = PRIV / "features"
MODELS = PRIV / "models"
RISKS = PRIV / "risks"
LOGS = ART / "logs"
AUDIT = ART / "gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"
CJ_SCRIPT = HERE / "compare_joint_cox_mgpt2_2026-10-01.py"
FR_SCRIPT = HERE / "gs_factor_residual_cox_2026-10-01.py"
FEATS_DIR = ART / "gs_features_2026-10-01"
SEED = 20261030
SEED_LOCKED = 20260928
N_BOOT = 2000
MIN_EVENTS = 50
MIN_DISEASES = 5
GPU_ITERS = 200
SMALL_CELL = 5
FOLDS = [1, 2, 3, 4, 5]
PRIMARY = "hm27"
SETTINGS = ["hm27", "random_90"]
ALL_SCEN = ["full"] + SETTINGS
CATEGORIES = ["Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases", "Neurological_Diseases", "Respiratory_Diseases"]
CONTEXT = ["Endocrine_and_Metabolic_Diseases", "Kidney_Diseases", "date_of_death"]
REPRO_EP = "Neurological_Diseases"
REPRO = f"REPRO_{REPRO_EP}"
SEX_RESTRICTED = {"Prostate_cancer": 0.0, "Breast_cancer": 1.0, "Ovary_cancer": 1.0, "Uterine_corpus_cancer": 1.0}  # female = 1
SKSURV_REPS = {"embedding": 100000, "biomarkers": 100000}
V12_REP = {"raw_cpg_v12": 1000000}
GPU_REPS = ["factor_residual", "raw_cpg_gpu"]
LABEL = "simulated missingness on empirical profiles"
OURS = "mgpt2_combined"
RULE = ("PASS iff (M_MethylGPT - M_best) / M_best >= 0.10 and the lower end of the 95% interval of M_MethylGPT - M_best is "
        "> 0; M = mean over primary-set diseases of (pooled within-fold C-index - age+sex C-index); best = decision comparator "
        "with the largest M in the setting")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


J = load_module("large_gain_J", J_SCRIPT)   # no torch or sksurv at import, so the biolearn interpreter can load it too
log, need, need_glob, write_json = J.log, J.need, J.need_glob, J.write_json
atomic_npy, atomic_npz, atomic_parquet, atomic_csv, sha256_bytes = J.atomic_npy, J.atomic_npz, J.atomic_parquet, J.atomic_csv, J.sha256_bytes
CLOCK_MODELS, RAW, COMPONENTS, DECISION, COHORT, FACTOR = J.CLOCK_MODELS, J.RAW, J.COMPONENTS, J.DECISION, J.COHORT, J.FACTOR
models_for, role = J.models_for, J.role


# ------------------------------------------------------------------ gates and paths
def done_path(stage):
    return LOGS / f"large_gain_K_{stage}_{DATE}.done"


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
    for p in (Path(__file__).resolve(), J_SCRIPT):
        if not committed(p):
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before any result-producing stage runs")


def table_path():
    return OUTK / f"K_disease_table_{DATE}.csv"


def require_frozen():
    require_committed()
    require_done("freeze")
    if not committed(table_path()):
        sys.exit("Cannot proceed - the frozen disease table must be committed before any fit")


def read_table():
    return pd.read_csv(need(table_path()), dtype={"events": str})


def primary_diseases():
    t = read_table()
    return t.loc[t.role == "primary", "constituent"].tolist()


def all_targets():
    return [REPRO] + primary_diseases()


def shard_of(items, shard, nshards):
    return items[shard::nshards]


def model_path(t, rep, f):
    return MODELS / t / f"{rep}_outer{f}.npz"


def load_model(t, rep, f):
    with np.load(need(model_path(t, rep, f)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rows_sha(rows):
    return sha256_bytes(np.sort(np.asarray(rows, dtype=np.int64)).tobytes())


def fold_seed(disease):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{disease}".encode()).digest()[:4], "big")


def gpu_seed(t, rep, f):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{t}:{rep}:{f}".encode()).digest()[:4], "big")


def failure(t, rep, f, err):
    d = PRIV / "failures"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{t}__{rep}__outer{f}.txt").write_text(repr(err)[:2000] + "\n")
    log(f"FAILED {t} {rep} fold {f}: {err!r}"[:400])


def load_ids():
    mp = need_glob(J.CACHE, f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")
    return pd.read_parquet(need(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet"))),
                           columns=["dnam_id"]).dnam_id.astype(str).to_numpy()


def covariates(ids):
    pe = J.load_preds()[J.ENDPOINTS[0]].set_index("dnam_id").loc[ids]
    return pe[["age", "sex"]].to_numpy(float)


def target_data(t, ids):
    """(event bool, duration, outer fold, row order of the GPU fits); arrays in frame-cache order. K's diseases fit in
    frame-cache order; the REPRO target fits in the locked run's row order so that its inner splits are the locked ones."""
    if t == REPRO:
        pe = J.load_preds()[REPRO_EP]
        pos = {s: i for i, s in enumerate(ids)}
        order = np.array([pos[s] for s in pe.dnam_id])
        ev, du, fold = np.zeros(len(ids), bool), np.zeros(len(ids)), np.zeros(len(ids), int)
        ev[order], du[order], fold[order] = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float), pe.outer_fold.to_numpy(int)
        return ev, du, fold, order
    lab = pd.read_parquet(need(PRIV / "labels" / f"private_labels_{t}.parquet"))
    if not np.array_equal(lab.dnam_id.astype(str).to_numpy(), ids):
        sys.exit(f"{t}: labels are not in frame-cache order")
    return lab.event.astype(bool).to_numpy(), lab.duration.to_numpy(float), lab.outer_fold.to_numpy(int), np.arange(len(ids))


# ------------------------------------------------------------------ stage freeze
def stage_freeze():
    from sklearn.model_selection import StratifiedKFold
    require_committed()
    t0 = time.time()
    lab = load_module("regen_labels", need(J.PROJECT / "revision_server_20260622/server_results/task_a/code/regenerate_disease_survival_data.py"))
    _, phenotype, mapping, _ = lab.load_metadata()
    audit = json.loads(need(AUDIT).read_text())
    ids = load_ids()
    preds = J.load_preds()
    checks = []

    def add(check, item, stored, recomputed, passed, note=""):
        checks.append({"check": check, "scenario": "labels", "item": item, "stored": stored, "recomputed": recomputed,
                       "abs_diff": "", "tolerance": 0, "pass": bool(passed), "note": note})

    # the label code with the true mapping reproduces the audit (all rows of the code, event counts only)
    for spec in lab.ENDPOINTS:
        _, s = lab.build_endpoint_table(phenotype, mapping, spec, label_mode="incident_positive_duration")
        a = audit["label_summaries"][spec.endpoint]
        add("1 label code reproduces the audit", spec.endpoint, f"{a['rows']} rows / {a['events']} events",
            f"{s['rows']} rows / {s['events']} events", s["rows"] == a["rows"] and s["events"] == a["events"])

    def constituent(col):
        spec = lab.EndpointSpec(col, col, col)
        tb, _ = lab.build_endpoint_table(phenotype, pd.DataFrame({"disease": [col], "category": [col]}), spec,
                                         label_mode="incident_positive_duration")
        d = tb.assign(dnam_id=tb.dnam_id.astype(str)).drop_duplicates("dnam_id").set_index("dnam_id").reindex(ids)
        return d

    rows, labels, seen = [], {}, {}
    for cat in CATEGORIES:
        cols = audit["label_summaries"][cat]["mapped_disease_columns"].split(";")
        evs, dus = [], []
        for col in cols:
            if col in seen:
                sys.exit(f"{col} is in two categories ({seen[col]}, {cat})")
            seen[col] = cat
            d = constituent(col)
            missing = int(d.event.isna().sum())
            add("2 all development participants at risk", col, 9866, 9866 - missing, missing == 0,
                "constituent table holds every development participant (no prevalent or same-month event among them)")
            ev, du = d.event.to_numpy(float), d.duration.to_numpy(float)
            add("2 positive durations", col, "all > 0", f"min > 0: {bool(np.nanmin(du) > 0)}", bool(np.nanmin(du) > 0))
            labels[col] = (ev.astype(bool), du)
            evs.append(ev.astype(bool))
            dus.append(du)
            rows.append({"constituent": col, "category": cat, "n_at_risk": 9866 - missing, "events_n": int(np.nansum(ev))})
        # pooled constituents against the stored endpoint
        E, D = np.array(evs), np.array(dus)
        pe = preds[cat].set_index("dnam_id").loc[ids]
        anyev = E.any(0)
        earliest = np.where(E, D, np.inf).min(0)
        censor = D[0]
        pooled_du = np.where(anyev, earliest, censor)
        same_censor = bool((D[:, ~anyev] == censor[~anyev]).all())
        add("1 pooled constituents equal the stored endpoint (events)", cat, int(pe.event.sum()), int(anyev.sum()),
            bool(np.array_equal(anyev, pe.event.astype(bool).to_numpy())), "participant-level equality; counts shown")
        add("1 pooled constituents equal the stored endpoint (durations)", cat, "equal for all 9,866", "",
            bool(np.array_equal(pooled_du, pe.duration.to_numpy(float))) and same_censor, "participant-level equality")
    t = pd.DataFrame(rows)
    t["eligible"] = t.events_n >= MIN_EVENTS
    t["role"] = np.where(t.eligible, "primary", f"ineligible (fewer than {MIN_EVENTS} events)")
    ctx = []
    for ep in CONTEXT:
        ctx.append({"constituent": audit["label_summaries"][ep]["mapped_disease_columns"], "category": ep, "n_at_risk": 9866,
                    "events_n": int(preds[ep].event.sum()), "eligible": False,
                    "role": "context (outside the primary set; rows copied from J)"})
    t = pd.concat([t, pd.DataFrame(ctx)], ignore_index=True)
    (PRIV / "labels").mkdir(parents=True, exist_ok=True)
    sh, se = [], []
    for _, r in t.iterrows():
        if r.role != "primary":
            sh.append(""); se.append("")
            continue
        ev, du = labels[r.constituent]
        s = fold_seed(r.constituent)
        fold = np.zeros(len(ids), np.int64)
        for k, (_, held) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=s).split(ids, ev.astype(int)), 1):
            fold[held] = k
        atomic_parquet(PRIV / "labels" / f"private_labels_{r.constituent}.parquet",
                       pd.DataFrame({"dnam_id": ids, "event": ev.astype(int), "duration": du, "outer_fold": fold}))
        se.append(s)
        sh.append(sha256_bytes(fold.tobytes()))
    t["fold_seed"], t["sha256_folds"] = se, sh
    t["events"] = [str(e) if e >= SMALL_CELL else f"<{SMALL_CELL}" for e in t.events_n]
    pub = t[["constituent", "category", "role", "n_at_risk", "events", "eligible", "fold_seed", "sha256_folds"]]
    OUTK.mkdir(parents=True, exist_ok=True)
    atomic_csv(table_path(), pub)
    ck = pd.DataFrame(checks)
    atomic_csv(OUTK / f"K_freeze_checks_{DATE}.csv", ck)
    n_el = int(t.eligible.sum())
    log(f"freeze: {len(rows)} constituents, {n_el} eligible (>= {MIN_EVENTS} events); checks {len(ck)} rows, {int((~ck['pass']).sum())} failed")
    if (~ck["pass"]).any():
        print(ck[~ck["pass"]].to_string())
        sys.exit("freeze checks failed")
    if n_el < MIN_DISEASES:
        sys.exit(f"STOP: only {n_el} diseases are eligible (fewer than {MIN_DISEASES}); the threshold is not lowered")
    done_marker("freeze", {"constituents": len(rows), "eligible": n_el, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage features
def stage_features(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    from methylgpt_model import BioHeads, embed_batch, load_checkpoint
    require_frozen()
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = J.pick_device(device)
    ids, X = J.load_gs()
    FEAT.mkdir(parents=True, exist_ok=True)
    if not (FEAT / "private_X.npy").exists():
        atomic_npy(FEAT / "private_X.npy", X)
    ag = torch.load(need(J.AGING), map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    model.load_state_dict(ag["model"]); model.eval()
    mt = torch.load(need(J.MULTITASK), map_location=dev, weights_only=False)
    mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
    mt_model.load_state_dict(mt["model"]); mt_model.eval()
    mt_heads = BioHeads(mt_core.rank + mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
    mt_heads.load_state_dict(mt["heads"]); mt_heads.eval()
    _, fi_core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)   # scenario_gs_raw_factor_imputed's core

    def features(Xs):   # J.stage_mgpt.features (scenario_gs_combined.main.features)
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                obs = torch.isfinite(x)
                E.append(embed_batch(model, core, x, obs).float().cpu().numpy())
                B.append(mt_heads(embed_batch(mt_model, mt_core, x, obs))[3].float().cpu().numpy())
                s = core.posterior(x, obs.float())
                R.append(torch.where(obs, x - core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        return np.concatenate(E), np.concatenate(B), np.concatenate(R)

    def factor_impute(Xs):   # J.stage_mgpt.factor_impute (scenario_gs_raw_factor_imputed loop)
        Xf = Xs.copy()
        with torch.no_grad():
            for i in range(0, len(Xf), 64):
                x = torch.from_numpy(Xf[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                rec = fi_core.predict(fi_core.posterior(x, obs.float())).clamp(0, 1)
                Xf[i:i + 64] = torch.where(obs, x, rec).float().cpu().numpy()
        return Xf

    for scen in ALL_SCEN:
        fr_p, fi_p = FEAT / f"private_FR_{scen}.npy", FEAT / f"private_Xfi_{scen}.npy"
        if fr_p.exists() and fi_p.exists():
            log(f"{scen}: features exist, skipped")
            continue
        t1 = time.time()
        Xs = np.where(J.load_mask(scen), X, np.nan).astype(np.float32)
        E, B, R = features(Xs)
        atomic_npy(fr_p, np.hstack([E, B, R]).astype(np.float32))
        del E, B, R
        Xf = factor_impute(Xs)
        if not np.isfinite(Xf).all():
            sys.exit(f"{scen}: factor imputation left missing values")
        atomic_npy(fi_p, Xf)
        log(f"{scen}: features written ({time.time() - t1:.0f} s)")
    done_marker("features", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ fit helpers
class UnconvergedPath(RuntimeError):
    pass


def fit_path_k(x, y, max_iter):
    """coxnet_path over the locked 12-point grid; on sksurv's ArithmeticError or a convergence warning at weak penalties, the
    longest prefix of that grid that succeeds and converges, found by bisection (amendments 2026-10-30)."""
    try:
        return coxnet_path(x, y, 12, max_iter)
    except (ArithmeticError, UnconvergedPath):
        pass
    lo, hi, best = 0, 12, None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try:
            best, lo = coxnet_path(x, y, mid, max_iter), mid
        except (ArithmeticError, UnconvergedPath):
            hi = mid
    if best is None:
        raise ArithmeticError("no prefix of the locked alpha grid succeeds")
    return best


def coxnet_path(x, y, k, max_iter):
    """compare_joint_cox fit_path with the locked settings on the first k of the 12 grid points, except that a path sksurv
    ends early is kept (recorded in the model file) instead of raising; a convergence warning still raises."""
    from sklearn.exceptions import ConvergenceWarning
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    penalty = np.ones(x.shape[1])
    penalty[:2] = 0.0
    model = CoxnetSurvivalAnalysis(n_alphas=k, alpha_min_ratio=0.05 ** ((k - 1) / 11), l1_ratio=0.5, penalty_factor=penalty,
                                   normalize=False, tol=1e-7, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x, y)
    for w in captured:
        if issubclass(w.category, ConvergenceWarning):
            raise UnconvergedPath(f"Unconverged Cox path: {w.message}")
    if not np.isfinite(model.coef_).all():
        raise ValueError("Non-finite Cox coefficients")
    return model


def fit_sksurv_family(t, rep, F, cov, ev, du, fold, max_iter, cj, folds=None):
    """compare_joint_cox_mgpt2.nested_development for one target and representation; rows in frame-cache order."""
    from sklearn.model_selection import StratifiedKFold
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    y = Surv.from_arrays(ev, du)
    for f in folds or FOLDS:
        out = model_path(t, rep, f)
        if out.exists():
            continue
        t1 = time.time()
        try:
            train_idx, held_idx = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED_LOCKED + f)
            scores = []
            for lfit, lev in inner.split(train_idx, ev[train_idx].astype(int)):
                fi, ei = train_idx[lfit], train_idx[lev]
                Ff = np.asarray(F[fi])
                tr = cj.TrainTransform().fit(Ff, cov[fi])
                fx = tr.transform(Ff, cov[fi])
                ex = tr.transform(np.asarray(F[ei]), cov[ei])
                del Ff
                m = fit_path_k(fx, y[fi], max_iter)
                scores.append([concordance_index_censored(ev[ei], du[ei], m.predict(ex, alpha=a))[0] for a in m.alphas_])
                del fx, ex, m
            lens = [len(s) for s in scores]
            means = np.asarray([s[:min(lens)] for s in scores]).mean(0)
            sel = int(np.argmax(means))
            Ff = np.asarray(F[train_idx])
            tr = cj.TrainTransform().fit(Ff, cov[train_idx])
            fx = tr.transform(Ff, cov[train_idx])
            del Ff
            m = fit_path_k(fx, y[train_idx], max_iter)
            k = min(sel, len(m.alphas_) - 1)
            coef = m.coef_[:, k].copy()
            atomic_npz(out, coefficient=coef, alpha=float(m.alphas_[k]), selected=k, inner_path_lengths=np.array(lens),
                       final_path_length=len(m.alphas_), mean=tr.mean, scale=tr.scale, cov_mean=tr.cov_mean, cov_scale=tr.cov_scale,
                       train_sha=np.array(rows_sha(train_idx)), held_sha=np.array(rows_sha(held_idx)),
                       seconds=time.time() - t1, warnings_note=np.array(""))
            log(f"{t} {rep} fold {f}: alpha index {k}, nonzero {int((coef[2:] != 0).sum())}, paths {lens + [len(m.alphas_)]}, {time.time() - t1:.0f} s")
            del fx, m
        except Exception as e:      # recorded, the other folds and targets continue; apply refuses until it is resolved
            failure(t, rep, f, e)


def fit_age_sex(t, cov, ev, du, fold, cj):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    y = Surv.from_arrays(ev, du)
    for f in FOLDS:
        out = model_path(t, "age_sex", f)
        if out.exists():
            continue
        try:
            train_idx, held_idx = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            cm, cs = cov[train_idx].mean(0), cov[train_idx].std(0)
            cs = np.where(cs < 1e-8, 1.0, cs)
            z = (cov[train_idx] - cm) / cs
            note = ""
            try:
                coef = cj.fit_baseline(z, y[train_idx]).coef_
            except Exception as e:  # sex separates the events: the same estimator, warnings recorded (docstring)
                with warnings.catch_warnings(record=True) as cap:
                    warnings.simplefilter("always")
                    coef = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit(z, y[train_idx]).coef_
                note = f"strict filter raised {type(e).__name__}; refitted with warnings recorded: " + "; ".join(sorted({str(w.message)[:80] for w in cap}))
            atomic_npz(out, coefficient=coef, cov_mean=cm, cov_scale=cs, train_sha=np.array(rows_sha(train_idx)),
                       held_sha=np.array(rows_sha(held_idx)), seconds=0.0, warnings_note=np.array(note))
        except Exception as e:
            failure(t, "age_sex", f, e)


def fit_clocks(t, full, ids, cov, ev, du, fold):
    """J.stage_cox fit: fitted on the training participants' stored full-profile scores."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    for name, cols in CLOCK_MODELS.items():
        Mf = np.column_stack([cov] + [full.loc[ids, c].to_numpy(float) for c in cols])
        for f in FOLDS:
            out = model_path(t, f"clock_{name}", f)
            if out.exists():
                continue
            try:
                train_idx, held_idx = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                m, s = Mf[train_idx].mean(0), Mf[train_idx].std(0)
                with warnings.catch_warnings(record=True) as cap:
                    warnings.simplefilter("always")
                    cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((Mf[train_idx] - m) / s, Surv.from_arrays(ev[train_idx], du[train_idx]))
                atomic_npz(out, m=m, s=s, coef=cx.coef_.copy(), train_sha=np.array(rows_sha(train_idx)), held_sha=np.array(rows_sha(held_idx)),
                           seconds=0.0, warnings_note=np.array("; ".join(sorted({str(w.message)[:80] for w in cap}))))
            except Exception as e:
                failure(t, f"clock_{name}", f, e)


def fit_gpu_family(t, rep, F, cov, ev, du, fold, order, dev, fr, select_and_fit):
    """gs_factor_residual_cox_2026-10-01.main, one target and representation; rows in `order`."""
    import torch
    ev_o, du_o, fold_o, cov_o = ev[order], du[order], fold[order], cov[order].astype(np.float32)
    for f in FOLDS:
        out = model_path(t, rep, f)
        if out.exists():
            continue
        t1 = time.time()
        try:
            held, fit = np.flatnonzero(fold_o == f), np.flatnonzero(fold_o != f)
            cm, cs = fr.fit_scaler(cov_o[fit], dev)
            Ff = np.asarray(F[order[fit]])
            fm, fs = fr.fit_scaler(Ff, dev)
            Xf = np.hstack([fr.apply_scaler(cov_o[fit], cm, cs, dev), fr.apply_scaler(Ff, fm, fs, dev)])
            del Ff
            torch.manual_seed(gpu_seed(t, rep, f))
            w, alpha, k = select_and_fit(Xf, du_o[fit], ev_o[fit], seed=SEED_LOCKED + int(f), dev=dev, iters=GPU_ITERS)
            del Xf
            torch.cuda.empty_cache()
            atomic_npz(out, cov_mean=cm, cov_scale=cs, mean=fm, scale=fs, coefficient=w.astype(np.float32), alpha=alpha, selected=k,
                       train_sha=np.array(rows_sha(order[fit])), held_sha=np.array(rows_sha(order[held])), seconds=time.time() - t1,
                       warnings_note=np.array(""))
            log(f"{t} {rep} fold {f}: alpha index {k}, nonzero {int((w[2:] != 0).sum())}, {time.time() - t1:.0f} s")
        except Exception as e:
            failure(t, rep, f, e)


# ------------------------------------------------------------------ stage fit
def stage_fit(family, shard, nshards, device, only_target=None, only_fold=None):
    require_frozen()
    require_done("features")
    t0 = time.time()
    ids = load_ids()
    cov = covariates(ids)
    targets = [only_target] if only_target else shard_of(all_targets(), shard, nshards)
    log(f"fit {family} shard {shard}/{nshards}: {len(targets)} targets")
    for t in targets:
        (MODELS / t).mkdir(parents=True, exist_ok=True)
    if family == "cheap":
        from threadpoolctl import threadpool_limits
        cj = load_module("compare_joint_cox_mgpt2", CJ_SCRIPT)
        emb = np.load(need(FEATS_DIR / f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_mgpt2_embed.embedding.npz"), allow_pickle=False)
        bio = np.load(need(FEATS_DIR / f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_mt_bio.embedding.npz"), allow_pickle=False)
        if not (np.array_equal(emb["dnam_id"].astype(str), ids) and np.array_equal(bio["dnam_id"].astype(str), ids)):
            sys.exit("stored features are not in frame-cache order")
        E, B = emb["embedding"], bio["embedding"]
        sc = pd.read_csv(need(J.CLOCK_SCORES))
        sc["sample_id"] = sc.sample_id.astype(str)
        full = sc[sc.scenario == "full"].set_index("sample_id")
        with threadpool_limits(limits=4):
            for t in targets:
                ev, du, fold, _ = target_data(t, ids)
                fit_age_sex(t, cov, ev, du, fold, cj)
                fit_clocks(t, full, ids, cov, ev, du, fold)
                fit_sksurv_family(t, "embedding", E, cov, ev, du, fold, SKSURV_REPS["embedding"], cj)
                fit_sksurv_family(t, "biomarkers", B, cov, ev, du, fold, SKSURV_REPS["biomarkers"], cj)
    elif family == "v12":
        from threadpoolctl import threadpool_limits
        cj = load_module("compare_joint_cox_mgpt2", CJ_SCRIPT)
        X = np.load(need(FEAT / "private_X.npy"), mmap_mode="r")
        with threadpool_limits(limits=2):
            for t in targets:
                ev, du, fold, _ = target_data(t, ids)
                fit_sksurv_family(t, "raw_cpg_v12", X, cov, ev, du, fold, V12_REP["raw_cpg_v12"], cj, [only_fold] if only_fold else None)
    elif family == "gpu":
        import torch
        sys.path.insert(0, str(HERE))
        from methylgpt_lib import cuda_preinit
        from gpu_coxnet_2026_10_01 import select_and_fit
        cuda_preinit()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_num_threads(4)
        dev = J.pick_device(device)
        fr = load_module("gs_factor_residual", FR_SCRIPT)
        X = np.load(need(FEAT / "private_X.npy"), mmap_mode="r")
        FR = np.load(need(FEAT / "private_FR_full.npy"), mmap_mode="r")
        for t in targets:
            ev, du, fold, order = target_data(t, ids)
            fit_gpu_family(t, "factor_residual", FR, cov, ev, du, fold, order, dev, fr, select_and_fit)
            fit_gpu_family(t, "raw_cpg_gpu", X, cov, ev, du, fold, order, dev, fr, select_and_fit)
    if not only_target:
        done_marker(f"fit_{family}_{shard}of{nshards}", {"targets": targets, "seconds": round(time.time() - t0)})


def all_reps():
    return (["age_sex", "embedding", "biomarkers"] + [f"clock_{c}" for c in CLOCK_MODELS] + list(V12_REP) + GPU_REPS)


def require_fits(reps=None):
    miss = [(t, r, f) for t in all_targets() for r in (reps or all_reps()) for f in FOLDS if not model_path(t, r, f).exists()]
    fails = sorted((PRIV / "failures").glob("*.txt")) if (PRIV / "failures").exists() else []
    if fails:
        sys.exit(f"{len(fails)} fit failures need a dated amendment first: {[p.stem for p in fails][:6]}")
    if miss:
        sys.exit(f"{len(miss)} fold models missing, e.g. {miss[:3]}")


# ------------------------------------------------------------------ stage cohort (in-cohort kNN with K's folds)
def stage_cohort(device, shard, nshards):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    require_frozen()
    require_fits(list(V12_REP) + ["raw_cpg_gpu"])
    cuda_preinit()
    t0 = time.time()
    dev = J.pick_device(device)
    ids = load_ids()
    X = np.load(need(FEAT / "private_X.npy"))
    gsobs = np.isfinite(X)
    cov_all = covariates(ids)
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    _, cpgs, _ = J.load_clock_matrix()
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    rd = load_module("recon_diag", J.RECON)
    mu = np.load(need(J.CORE_MU))["mu"].astype(np.float32)
    mu_h = torch.as_tensor(mu, device=dev).half()
    obs_all = {s: J.load_mask(s) for s in SETTINGS}
    cm_all = {s: J.clock_masked_vocab(s, ic) for s in SETTINGS}
    (PRIV / "cohort").mkdir(parents=True, exist_ok=True)
    for t in shard_of(all_targets(), shard, nshards):
        outs = {s: PRIV / "cohort" / f"private_cohortknn_{s}__{t}.parquet" for s in SETTINGS}
        refp = PRIV / "cohort" / f"private_reference_check_{t}.csv"
        if refp.exists() and all(o.exists() and o.with_name(o.stem + "_clockfill.npy").exists() for o in outs.values()):
            log(f"{t}: in-cohort kNN exists, skipped")
            continue
        t1 = time.time()
        ev, du, fold, _ = target_data(t, ids)
        res = {s: {"v12": np.full(len(ids), np.nan), "gpu": np.full(len(ids), np.nan),
                   "fill": np.full((len(ids), len(ic)), np.nan, np.float32)} for s in SETTINGS}
        refcheck = []
        for f in FOLDS:
            held, fit = np.flatnonzero(fold == f), np.flatnonzero(fold != f)
            refcheck.append({"target": t, "outer_fold": f, "reference_rows": len(fit), "held_out_in_reference": len(np.intersect1d(held, fit)),
                             "sha256_reference_rows": rows_sha(fit)})
            rm, gm = load_model(t, "raw_cpg_v12", f), load_model(t, "raw_cpg_gpu", f)
            Rh = torch.from_numpy(X[fit]).to(dev).half()
            Xh = X[held]
            cov = cov_all[held]
            for s in SETTINGS:                                    # J.stage_cohort loop
                obs = obs_all[s][held]
                Xs = np.where(obs, Xh, np.nan).astype(np.float32)
                masked = gsobs[held] & ~obs
                Xk = J.knn_fill(rd, Rh, mu_h, Xs, obs, masked, dev)
                if not (np.array_equal(Xk[obs], Xh[obs]) and np.isfinite(Xk[masked]).all()):
                    sys.exit(f"{s} {t} fold {f}: in-cohort kNN fill check failed")
                res[s]["v12"][held] = J.locked_risk(rm, Xk.astype(np.float64), cov)
                res[s]["gpu"][held] = J.gpu_risk(gm, Xk, cov.astype(np.float32))
                cm = cm_all[s][held]
                ctx_c = obs.copy()
                ctx_c[:, vj] &= ~cm
                fill = np.full(cm.shape, np.nan, np.float32)
                if np.array_equal(ctx_c, obs):
                    fill[cm] = Xk[:, vj][cm]
                else:
                    mc = np.zeros(obs.shape, bool)
                    mc[:, vj] = cm
                    Xc = J.knn_fill(rd, Rh, mu_h, np.where(ctx_c, Xh, np.nan).astype(np.float32), ctx_c, mc, dev)
                    fill[cm] = Xc[:, vj][cm]
                if not np.isfinite(fill[cm]).all():
                    sys.exit(f"{s} {t} fold {f}: in-cohort clock fills incomplete")
                res[s]["fill"][held] = fill
            del Rh
            torch.cuda.empty_cache()
        for s in SETTINGS:
            fr_ = pd.DataFrame({"row": np.arange(len(ids)), "raw_cpg_v12_cohortknn": res[s]["v12"], "raw_cpg_gpu_cohortknn": res[s]["gpu"]})
            if not (np.isfinite(res[s]["v12"]).all() and np.isfinite(res[s]["gpu"]).all()):
                sys.exit(f"{s} {t}: in-cohort risks incomplete")
            atomic_npy(outs[s].with_name(outs[s].stem + "_clockfill.npy"), res[s]["fill"])
            atomic_parquet(outs[s], fr_)
        atomic_csv(refp, pd.DataFrame(refcheck))
        log(f"{t}: in-cohort kNN for {len(SETTINGS)} settings written ({time.time() - t1:.0f} s)")
    done_marker(f"cohort_{shard}of{nshards}", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stages clockin, clocks
def stage_clockin():
    require_frozen()
    t0 = time.time()
    cids, cpgs, M = J.load_clock_matrix()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    means = np.nanmean(M, 0)
    d = PRIV / "clockin"
    d.mkdir(parents=True, exist_ok=True)
    atomic_csv(d / "private_ids.csv", pd.DataFrame({"sample_id": cids}))
    atomic_csv(d / "private_cpgs.csv", pd.DataFrame({"cpg": cpgs}))
    for s in SETTINGS:
        cobs = J.load_mask(s, "cobs")
        cm = np.isfinite(M[:, ic]) & ~cobs[:, ic]
        base = np.where(cobs, M, means[None, :]).astype(np.float32)   # J.stage_clockin.filled
        for t in all_targets():
            fo = d / f"private_clockin_{s}__cohortknn__{t}.npy"
            if fo.exists():
                continue
            fill = np.load(need(PRIV / "cohort" / f"private_cohortknn_{s}__{t}_clockfill.npy"))
            if not np.array_equal(np.isfinite(fill), cm):
                sys.exit(f"{s} {t}: clock fills do not match the clock-masked vocabulary CpGs")
            Z = base.copy()
            sub = Z[:, ic]
            ok = np.isfinite(fill)
            sub[ok] = fill[ok]
            Z[:, ic] = sub
            atomic_npy(fo, Z)
    done_marker("clockin", {"seconds": round(time.time() - t0)})


def score_variant(v):
    """J.score_variant with K's directories."""
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery
    d = PRIV / "clockin"
    ids = pd.read_csv(d / "private_ids.csv").sample_id.astype(str).tolist()
    target = pd.read_csv(d / "private_cpgs.csv").cpg.astype(str).tolist()
    X = np.load(d / f"private_clockin_{v}.npy")
    meta = pd.DataFrame({"sample_id": ids, "age": 47.0, "Age": 47.0, "sex": 1.0, "Female": 1.0}).set_index("sample_id")
    data = GeoData(meta, pd.DataFrame(X.T, index=target, columns=ids))
    gallery = ModelGallery()
    out = pd.DataFrame({"sample_id": ids})
    for model_name, col in (("GrimAgeV2", "DNAmGrimAge"), ("DunedinPACE", "DunedinPACE")):
        pred = gallery.get(model_name, imputation_method="averaging").predict(data)
        out[model_name] = pred[col].reindex(ids).to_numpy() if col in pred.columns else pred.iloc[:, 0].reindex(ids).to_numpy()
    atomic_csv(PRIV / "clock_scores" / f"private_scores_{v}.csv", out)
    return v


def stage_clocks(workers):
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    require_frozen()
    require_done("clockin")
    import biolearn
    t0 = time.time()
    (PRIV / "clock_scores").mkdir(parents=True, exist_ok=True)
    variants = [f"{s}__cohortknn__{t}" for s in SETTINGS for t in all_targets()]
    todo = []
    for v in variants:
        fo = PRIV / "clock_scores" / f"private_scores_{v}.csv"
        try:
            ok = len(pd.read_csv(fo)) == 9866
        except Exception:
            ok = False
        if not ok:
            todo.append(v)
    log(f"biolearn {getattr(biolearn, '__version__', '?')}: {len(todo)} clock variants to score with {workers} workers")
    with Pool(workers) as pool:
        for v in pool.imap_unordered(score_variant, todo):
            log(f"scored {v}")
    done_marker("clocks", {"variants": len(variants), "biolearn": getattr(biolearn, "__version__", "?"), "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage apply
class Context:
    """Shared read-only inputs of the apply code (memory-mapped matrices)."""

    def __init__(self, scens):
        self.ids = load_ids()
        self.cov = covariates(self.ids)
        self.X = np.load(need(FEAT / "private_X.npy"), mmap_mode="r")
        self.FR = {s: np.load(need(FEAT / f"private_FR_{s}.npy"), mmap_mode="r") for s in scens}
        self.Xfi = {s: np.load(need(FEAT / f"private_Xfi_{s}.npy"), mmap_mode="r") for s in scens}
        self.Xk = {s: np.load(need(J.PRIV / f"private_knn_corpus_{s}.npy"), mmap_mode="r") for s in scens if s != "full"}
        self.obs = {s: J.load_mask(s) for s in scens if s != "full"}
        self._sc = {}

    def scores(self, scen, fl, t):
        p = (PRIV / "clock_scores" / f"private_scores_{scen}__cohortknn__{t}.csv") if fl == "cohortknn" else \
            (J.PRIV / "clock_scores" / f"private_scores_{scen}__{fl}.csv")
        if p not in self._sc:
            s = pd.read_csv(need(p))
            s["sample_id"] = s.sample_id.astype(str)
            self._sc[p] = s.set_index("sample_id")
        return self._sc[p]


def sub_model(model):
    """The fold model restricted to its nonzero CpG or feature coefficients; identical arithmetic, fewer columns read."""
    c = model["coefficient"]
    nz = np.flatnonzero(c[2:])
    return nz, {"mean": model["mean"][nz], "scale": model["scale"][nz], "cov_mean": model["cov_mean"], "cov_scale": model["cov_scale"],
                "coefficient": np.r_[c[:2], c[2:][nz]]}


def apply_target(t, scen, C, cohort=True):
    ev, du, fold, _ = target_data(t, C.ids)
    N = len(ev)
    names = ["age_sex", "embedding", "biomarkers", "factor_residual"] + [f"{r}_mean" for r in RAW] + FACTOR
    if scen != "full":
        names += [f"{r}_knn" for r in RAW]
    R = {k: np.full(N, np.nan) for k in names}
    FRs = C.FR[scen]
    for f in FOLDS:
        held = np.flatnonzero(fold == f)
        cov, cov32 = C.cov[held], C.cov[held].astype(np.float32)
        b = load_model(t, "age_sex", f)
        R["age_sex"][held] = ((cov - b["cov_mean"]) / b["cov_scale"]) @ b["coefficient"]
        for rep, col0 in (("embedding", 0), ("biomarkers", 2048)):
            nz, sub = sub_model(load_model(t, rep, f))
            R[rep][held] = J.locked_risk(sub, np.asarray(FRs[np.ix_(held, nz + col0)], dtype=np.float64), cov)
        nz, sub = sub_model(load_model(t, "factor_residual", f))
        R["factor_residual"][held] = J.gpu_risk(sub, np.asarray(FRs[np.ix_(held, nz)]), cov32)
        for r, fn, dt, cx in (("raw_cpg_v12", J.locked_risk, np.float64, cov), ("raw_cpg_gpu", J.gpu_risk, np.float32, cov32)):
            nz, sub = sub_model(load_model(t, r, f))
            cells = np.ix_(held, nz)
            vals = np.asarray(C.X[cells])
            if scen != "full":
                vals = np.where(C.obs[scen][cells], vals, np.nan)
            R[f"{r}_mean"][held] = fn(sub, vals.astype(dt), cx)
            R[f"{r}_factor_imputed"][held] = fn(sub, np.asarray(C.Xfi[scen][cells]).astype(dt), cx)
            if scen != "full":
                R[f"{r}_knn"][held] = fn(sub, np.asarray(C.Xk[scen][cells]).astype(dt), cx)
    for cname, parts in J.COMBOS.items():         # sum of the within-fold standardised risks (J.stage_mgpt)
        v = np.zeros(N)
        for f in FOLDS:
            held = np.flatnonzero(fold == f)
            for k in parts:
                x_ = R[k][held]
                v[held] += (x_ - x_.mean()) / x_.std()
        R[cname] = v
    fills = ["mean"] if scen == "full" else (["mean", "knn"] + (["cohortknn"] if cohort else []))
    for fl in fills:
        S = C.scores(scen, fl, t)
        for name, cols in CLOCK_MODELS.items():
            Ms = np.column_stack([C.cov] + [S.loc[C.ids, c].to_numpy(float) for c in cols])
            r = np.full(N, np.nan)
            for f in FOLDS:
                held = np.flatnonzero(fold == f)
                z = load_model(t, f"clock_{name}", f)
                r[held] = ((Ms[held] - z["m"]) / z["s"]) @ z["coef"]
            R[f"{name}_{fl}"] = r
    if scen != "full" and cohort:
        c = pd.read_parquet(need(PRIV / "cohort" / f"private_cohortknn_{scen}__{t}.parquet"))
        for m in ("raw_cpg_v12_cohortknn", "raw_cpg_gpu_cohortknn"):
            R[m] = c[m].to_numpy(float)
    out = pd.DataFrame({"row": np.arange(N), "outer_fold": fold, **R})
    bad = [m for m in R if not np.isfinite(R[m]).all()]
    if bad:
        sys.exit(f"{t} {scen}: risks not finite for {bad}")
    return out


def stage_apply(shard, nshards):
    require_frozen()
    require_fits()
    require_done("clocks")
    t0 = time.time()
    C = Context(ALL_SCEN)
    RISKS.mkdir(parents=True, exist_ok=True)
    for t in shard_of(all_targets(), shard, nshards):
        for scen in ALL_SCEN:
            fo = RISKS / f"private_risks_{t}__{scen}.parquet"
            if fo.exists():
                continue
            t1 = time.time()
            atomic_parquet(fo, apply_target(t, scen, C))
            log(f"{t} {scen}: risks written ({time.time() - t1:.0f} s)")
    done_marker(f"apply_{shard}of{nshards}", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ reproduction against J's stored risks
def j_stored(scen, ids, cohort=True):
    """J's stored risks of REPRO_EP at a setting, frame-cache order."""
    preds = J.load_preds()
    pe = preds[REPRO_EP].set_index("dnam_id").loc[ids]
    out = pd.DataFrame({"age_sex": pe.baseline_risk.to_numpy(float)}, index=ids)
    parts = [J.PRIV / f"private_risks_mgpt_{scen}.parquet", J.PRIV / f"private_risks_clocks_{scen}.parquet"]
    if scen != "full":
        parts.append(J.PRIV / f"private_risks_knn_{scen}.parquet")
        if cohort:
            parts.append(J.PRIV / "cohort" / f"private_cohortknn_{scen}__{REPRO_EP}.parquet")
    for p in parts:
        t = pd.read_parquet(need(p))
        t = t[t.endpoint == REPRO_EP].assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id").loc[ids]
        for c in t.columns:
            if c not in ("endpoint", "outer_fold"):
                out[c] = t[c].to_numpy(float)
    return out


def repro_family(m):
    gpu = m.startswith("raw_cpg_gpu") or m in ("factor_residual", OURS) or m.endswith("_cohortknn")
    return "gpu_or_inCohort" if gpu else "sksurv_or_cox"


def repro_rows(scen, A, ids, ev, du, fold):
    st = j_stored(scen, ids, cohort=any(c.endswith("_cohortknn") for c in A.columns))
    folds = [np.flatnonzero(fold == f) for f in FOLDS]
    rows = []
    for m in [c for c in A.columns if c not in ("row", "outer_fold")]:
        ref, mine = st[m].to_numpy(float), A[m].to_numpy(float)
        cr, cm = J.sksurv_pooled(ev, du, ref, folds), J.sksurv_pooled(ev, du, mine, folds)
        rd = float(np.max(np.abs(ref - mine)))
        rmin = float(min(np.corrcoef(ref[s], mine[s])[0, 1] for s in folds))
        fam = repro_family(m)
        if fam == "gpu_or_inCohort":
            ok, tol = abs(cr - cm) <= 1e-3 and rmin >= 0.99, "abs dC <= 1e-3; min within-fold r >= 0.99"
        else:
            ok, tol = abs(cr - cm) <= 1e-5 and rd <= 1e-3 * (1 + np.abs(ref).max()), "abs dC <= 1e-5; max abs risk diff <= 1e-3 (1 + max|ref|)"
        rows.append({"check": "5 fitting code reproduces stored results on a locked endpoint", "scenario": scen, "item": f"{REPRO_EP}:{m}",
                     "stored": cr, "recomputed": cm, "abs_diff": abs(cr - cm), "tolerance": tol, "pass": bool(ok),
                     "note": f"family {fam}; max abs risk diff {rd:.3g}; min within-fold r {rmin:.6f}"})
    return rows


def stage_repcheck():
    """Early look at the REPRO target only, no K disease; the formal check is in stage checks."""
    require_frozen()
    have = [r for r in all_reps() if all(model_path(REPRO, r, f).exists() for f in FOLDS)]
    if set(have) != set(all_reps()):
        sys.exit(f"REPRO fold models incomplete; have {have}")
    C = Context(["full", "hm27"])
    ev, du, fold, _ = target_data(REPRO, C.ids)
    rows = []
    for scen in ("full", "hm27"):
        rows += repro_rows(scen, apply_target(REPRO, scen, C, cohort=False), C.ids, ev, du, fold)
    tab = pd.DataFrame(rows)
    atomic_csv(PRIV / "private_repcheck.csv", tab)
    print(tab[["scenario", "item", "stored", "recomputed", "abs_diff", "pass", "note"]].to_string())
    log(f"repcheck: {len(tab)} rows, {int((~tab['pass']).sum())} failed")


# ------------------------------------------------------------------ stage checks
def stage_checks():
    require_frozen()
    require_fits()
    miss = [(t, s) for t in all_targets() for s in ALL_SCEN if not (RISKS / f"private_risks_{t}__{s}.parquet").exists()]
    if miss:
        sys.exit(f"risks missing: {miss[:4]}")
    ids, X = J.load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    preds = J.load_preds()
    cov = covariates(ids)
    rows = []

    def add(check, scenario, item, stored, recomputed, tol, passed, note=""):
        diff = abs(float(stored) - float(recomputed)) if isinstance(stored, (int, float, np.floating)) and \
            isinstance(recomputed, (int, float, np.floating)) else ""
        rows.append({"check": check, "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "abs_diff": diff, "tolerance": tol, "pass": bool(passed), "note": note})

    jm = pd.read_csv(need(J.OUTJ / f"J_masks_{J.DATE}.csv")).set_index("scenario")
    kr = json.loads(need(J.OUTJ / f"J_knn_record_{J.DATE}.json").read_text())
    gsobs = np.isfinite(X)
    cids, cpgs, M = J.load_clock_matrix()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    # (3) masks
    for scen in ALL_SCEN:
        z = np.load(need(J.PRIV / "masks" / f"private_mask_{scen}.npz"), allow_pickle=False)
        for key, col in (("obs", "sha256_vocab_mask"), ("cobs", "sha256_clock_mask")):
            h = sha256_bytes(z[key].tobytes())
            add("3 mask hash equals J_masks", scen, key, jm.loc[scen, col], h, 0, h == jm.loc[scen, col])
    # (3) corpus reference rows, recomputed without loading the reference
    G = load_module("large_gain_G", J.G_SCRIPT)
    F = G.load_f()
    prof = pd.read_parquet(need(G.CORPUS_DIR / "profiles.parquet"))
    rrows = prof.index[(prof.split == "train").to_numpy()].to_numpy()
    drop = set(pd.read_csv(need(F.OUTF / f"duplicates_removed_{F.DATE}.csv")).corpus_row.astype(int))
    rrows = np.sort(np.array([r for r in rrows if r not in drop]))
    h = sha256_bytes(rrows.astype(np.int64).tobytes())
    add("3 kNN reference rows equal J_knn_record", "all", "row SHA-256", kr["reference_rows_sha256"], h, 0,
        h == kr["reference_rows_sha256"] and h == kr["G_rows_sha256"] and kr["identical_to_G_heldout_reference"], f"{len(rrows)} rows")
    gs_in_ref = len(set(prof.id.iloc[rrows].astype(str)) & set(ids))
    add("6 no GS participant in the corpus kNN reference", "all", "identifier overlap", 0, gs_in_ref, 0, gs_in_ref == 0)
    # (3) fills, and J's fold models on them
    for scen in SETTINGS:
        obs = J.load_mask(scen)
        masked = gsobs & ~obs
        Xk = np.load(need(J.PRIV / f"private_knn_corpus_{scen}.npy"))
        rec = kr["scenarios"][scen]
        add("3 kNN fills equal J_knn_record", scen, "masked values filled", rec["masked_values_filled"], int(masked.sum()), 0,
            rec["masked_values_filled"] == int(masked.sum()) and np.isfinite(Xk[masked]).all() and np.array_equal(Xk[obs], X[obs]))
        fill = np.load(need(J.PRIV / f"private_knn_corpus_clockfill_{scen}.npy"))
        add("3 kNN fills equal J_knn_record", scen, "clock-masked values filled", rec["clock_masked_vocab_values"], int(np.isfinite(fill).sum()), 0,
            rec["clock_masked_vocab_values"] == int(np.isfinite(fill).sum()))
        r_v12, r_gpu = J.raw_risks_filled(Xk, preds, pos, REPRO_EP)
        pe = preds[REPRO_EP]
        st = pd.read_parquet(need(J.PRIV / f"private_risks_knn_{scen}.parquet"))
        st = st[st.endpoint == REPRO_EP].assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id").loc[pe.dnam_id]
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in FOLDS]
        for m, r in (("raw_cpg_v12_knn", r_v12), ("raw_cpg_gpu_knn", r_gpu)):
            c1, c2 = J.sksurv_pooled(ev, du, st[m].to_numpy(float), folds), J.sksurv_pooled(ev, du, r, folds)
            add("3 J's fold models on the reused fills reproduce J's kNN risks", scen, f"{REPRO_EP}:{m}", c1, c2, 1e-6, abs(c1 - c2) <= 1e-6)
        del Xk, obs, masked
    # (3) clock scores
    sc = pd.read_csv(need(J.CLOCK_SCORES))
    sc["sample_id"] = sc.sample_id.astype(str)
    for scen in ALL_SCEN:
        mine = pd.read_csv(need(J.PRIV / "clock_scores" / f"private_scores_{scen}__mean.csv"))
        mine["sample_id"] = mine.sample_id.astype(str)
        ref = sc[sc.scenario == scen].set_index("sample_id").loc[mine.sample_id]
        for c in ("GrimAgeV2", "DunedinPACE"):
            d = float(np.max(np.abs(ref[c].to_numpy(float) - mine[c].to_numpy(float))))
            add("3 J's mean-fill clock scores equal the stored scores", scen, c, 0.0, d, 1e-6, d <= 1e-6, "max abs difference")
        if scen != "full":
            kn = pd.read_csv(need(J.PRIV / "clock_scores" / f"private_scores_{scen}__knn.csv"))
            add("3 J's kNN clock scores complete", scen, "rows", 9866, len(kn), 0, len(kn) == 9866 and np.isfinite(kn[["GrimAgeV2", "DunedinPACE"]].to_numpy()).all())
    # (4) features
    FRf = np.load(need(FEAT / "private_FR_full.npy"), mmap_mode="r")
    for name, tag, a, b in (("embedding", "mgpt2_embed", 0, 2048), ("biomarkers", "mt_bio", 2048, 2073)):
        z = np.load(need(FEATS_DIR / f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_{tag}.embedding.npz"), allow_pickle=False)
        d = float(np.abs(z["embedding"] - np.asarray(FRf[:, a:b])).max())
        add("4 recomputed full-profile features equal the stored features", "full", name, 0.0, d, 1e-3, d <= 1e-3, "max abs difference")
    pe = preds[REPRO_EP]
    idx = np.array([pos[s] for s in pe.dnam_id])
    c_pe = pe[["age", "sex"]].to_numpy(float)
    ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
    held_sets = {f: np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in FOLDS}
    folds = list(held_sets.values())
    for scen in ALL_SCEN:
        FRs = np.load(need(FEAT / f"private_FR_{scen}.npy"), mmap_mode="r")
        Xfi = np.load(need(FEAT / f"private_Xfi_{scen}.npy"), mmap_mode="r")
        st = pd.read_parquet(need(J.PRIV / f"private_risks_mgpt_{scen}.parquet"))
        st = st[st.endpoint == REPRO_EP].assign(dnam_id=lambda d: d.dnam_id.astype(str)).set_index("dnam_id").loc[pe.dnam_id]
        risk = {k: np.empty(len(pe)) for k in ("embedding", "biomarkers", "factor_residual", "raw_cpg_v12_factor_imputed", "raw_cpg_gpu_factor_imputed")}
        for f, h_ in held_sets.items():
            rows_f = idx[h_]
            mm = dict(np.load(need(J.MGPT2_RUN / f"{REPRO_EP}_linear_outer{f}_model.npz"), allow_pickle=False))
            bm = dict(np.load(need(J.BIO_RUN / f"{REPRO_EP}_linear_outer{f}_model.npz"), allow_pickle=False))
            risk["embedding"][h_] = J.locked_risk(mm, np.asarray(FRs[rows_f, :2048], dtype=np.float64), c_pe[h_])
            risk["biomarkers"][h_] = J.locked_risk(bm, np.asarray(FRs[rows_f, 2048:2073], dtype=np.float64), c_pe[h_])
            risk["factor_residual"][h_] = J.gpu_risk(J.gpu_model("factor_residual", REPRO_EP, f), np.asarray(FRs[rows_f]), c_pe[h_].astype(np.float32))
            rm, gm = J.raw_v12_model(REPRO_EP, f), J.gpu_model("raw_cpg_gpu", REPRO_EP, f)
            risk["raw_cpg_v12_factor_imputed"][h_] = J.fi_locked_risk(rm, np.asarray(Xfi[rows_f], dtype=np.float64), c_pe[h_])
            risk["raw_cpg_gpu_factor_imputed"][h_] = J.gpu_risk(gm, np.asarray(Xfi[rows_f]), c_pe[h_].astype(np.float32))
        for m, r in risk.items():
            c1, c2 = J.sksurv_pooled(ev, du, st[m].to_numpy(float), folds), J.sksurv_pooled(ev, du, r, folds)
            add("4 J's fold models on K's features reproduce J's risks", scen, f"{REPRO_EP}:{m}", c1, c2, 1e-6, abs(c1 - c2) <= 1e-6,
                f"max abs risk diff {np.max(np.abs(st[m].to_numpy(float) - r)):.3g}")
    # (5) fit reproduction
    ev_t, du_t, fold_t, _ = target_data(REPRO, ids)
    for scen in ALL_SCEN:
        A = pd.read_parquet(need(RISKS / f"private_risks_{REPRO}__{scen}.parquet"))
        rows.extend(repro_rows(scen, A, ids, ev_t, du_t, fold_t))
    # (6), (7) folds and leakage
    tab = read_table().set_index("constituent")
    for t in all_targets():
        ev, du, fold, _ = target_data(t, ids)
        ok_all, n_files = True, 0
        for f in FOLDS:
            tr, he = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            st_, sh_ = rows_sha(tr), rows_sha(he)
            if len(np.intersect1d(tr, he)) or len(tr) + len(he) != len(ids):
                ok_all = False
            for r in all_reps():
                z = load_model(t, r, f)
                n_files += 1
                if str(z["train_sha"]) != st_ or str(z["held_sha"]) != sh_:
                    ok_all = False
        add("6 training rows of every model equal the fold's training rows, disjoint from held-out", t, "models checked", n_files, n_files, 0, ok_all)
        rc = pd.read_csv(need(PRIV / "cohort" / f"private_reference_check_{t}.csv"))
        okr = all(rows_sha(np.flatnonzero(fold != r_.outer_fold)) == r_.sha256_reference_rows and r_.held_out_in_reference == 0 for r_ in rc.itertuples())
        add("6 in-cohort reference = training folds, held-out fold excluded", t, "folds", 5, len(rc), 0, okr and len(rc) == 5)
        if t != REPRO:
            ec = np.bincount(fold[ev], minlength=6)[1:]
            add("7 folds stratified by event status", t, "max - min events per fold", 1, int(ec.max() - ec.min()), 1, ec.max() - ec.min() <= 1)
            h = sha256_bytes(fold.astype(np.int64).tobytes())
            add("7 fold hash equals the frozen hash", t, "sha256", tab.loc[t, "sha256_folds"], h, 0, h == tab.loc[t, "sha256_folds"])
    # (8) pair counting
    worst = 0.0
    models = models_for(PRIMARY)
    for t in primary_diseases():
        ev, du, fold, _ = target_data(t, ids)
        A = pd.read_parquet(need(RISKS / f"private_risks_{t}__{PRIMARY}.parquet"))
        R = np.stack([A[m].to_numpy(float) for m in models])
        fs = [np.flatnonzero(fold == f) for f in FOLDS]
        con = np.zeros(len(R)); tied = np.zeros(len(R)); tot = 0.0
        for s in fs:
            (t_, arr), = J.fold_counts(ev[s], du[s], R[:, s], [None])
            con += arr[:, 0]; tied += arr[:, 1]; tot += t_
        for k in range(len(models)):
            worst = max(worst, abs(J.cidx_from(con[k], tied[k], tot) - J.sksurv_pooled(ev, du, R[k], fs)))
    add("8 pair counting equals sksurv (point)", PRIMARY, "max abs C difference over diseases and models", 0.0, worst, 1e-12, worst <= 1e-12)
    import torch
    stk = load_module("stk", J.STK)
    t = primary_diseases()[0]
    ev, du, fold, _ = target_data(t, ids)
    A = pd.read_parquet(need(RISKS / f"private_risks_{t}__{PRIMARY}.parquet"))
    s = np.flatnonzero(fold == 1)
    R = np.stack([A[m].to_numpy(float)[s] for m in (OURS, "age_sex")])
    W = np.stack([np.bincount(np.random.default_rng(b).integers(0, len(s), len(s)), minlength=len(s)) for b in range(5)]).astype(np.float64)
    (tot, arr), = J.fold_counts(ev[s], du[s], R, [W])
    ref = stk.PairCounter(ev[s], du[s], torch.device("cpu")).counts(R, W)
    d = float(max(np.abs(arr[..., 0] - ref[..., 0]).max(), np.abs(arr[..., 1] - ref[..., 2]).max(), np.abs(tot[None, :] - ref.sum(-1)).max()))
    add("8 pair counting equals stk.PairCounter (weighted)", PRIMARY, "one disease, fold 1, 5 draws", 0.0, d, 1e-9, d <= 1e-9)
    fz = pd.read_csv(need(OUTK / f"K_freeze_checks_{DATE}.csv"))
    out = pd.concat([fz, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(OUTK / f"K_checks_{DATE}.csv", out)
    write_fit_record()
    nf = int((~out["pass"].astype(bool)).sum())
    log(f"checks: {len(out)} rows, {nf} failed")
    if nf:
        print(out[~out["pass"].astype(bool)].to_string())
        sys.exit("checks failed; no new number is used")
    done_marker("checks", {"rows": len(out), "failed": 0})


def write_fit_record():
    """Per target, family and fold: selected penalty, nonzero coefficients, seconds, path lengths, recorded warnings. No
    participant-level value and no fold-level event count."""
    rows = []
    for t in all_targets():
        for r in all_reps():
            for f in FOLDS:
                z = load_model(t, r, f)
                row = {"target": t, "family": r, "outer_fold": f, "seconds": round(float(z["seconds"]), 1), "warnings_note": str(z["warnings_note"])}
                if "selected" in z:
                    row.update(alpha_index=int(z["selected"]), nonzero_coefficients=int((z["coefficient"][2:] != 0).sum()))
                if "final_path_length" in z:
                    row.update(path_lengths=" ".join(str(int(x)) for x in np.r_[z["inner_path_lengths"], z["final_path_length"]]))
                rows.append(row)
    atomic_csv(OUTK / f"K_fit_record_{DATE}.csv", pd.DataFrame(rows))


# ------------------------------------------------------------------ stage boot
def stage_boot():
    require_frozen()
    require_done("checks")
    t0 = time.time()
    ids = load_ids()
    N = len(ids)
    cov = covariates(ids)
    I = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))
    Wp = np.stack([np.bincount(I[b], minlength=N) for b in range(N_BOOT)]).astype(np.float64)
    del I
    diseases = primary_diseases()
    out = {}
    for scen in ALL_SCEN:
        models = models_for(scen)
        P, B = np.zeros((len(models), len(diseases))), np.zeros((len(models), len(diseases), N_BOOT))
        Ps, Bs = P.copy(), B.copy()     # sex_restricted view (equal to the primary view except in SEX_RESTRICTED diseases)
        for j, t in enumerate(diseases):
            ev, du, fold, _ = target_data(t, ids)
            A = pd.read_parquet(need(RISKS / f"private_risks_{t}__{scen}.parquet"))
            R = np.stack([A[m].to_numpy(float) for m in models])
            acc = [np.zeros(len(models)), np.zeros(len(models)), 0.0, np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]
            acs = [np.zeros(len(models)), np.zeros(len(models)), 0.0, np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]
            for f in FOLDS:
                s = np.flatnonzero(fold == f)
                (t_p, a_p), (t_b, a_b) = J.fold_counts(ev[s], du[s], R[:, s], [None, Wp[:, s]])
                acc[0] += a_p[:, 0]; acc[1] += a_p[:, 1]; acc[2] += t_p
                acc[3] += a_b[..., 0]; acc[4] += a_b[..., 1]; acc[5] += t_b
                if t in SEX_RESTRICTED:
                    s2 = s[cov[s, 1] == SEX_RESTRICTED[t]]
                    (t_p, a_p), (t_b, a_b) = J.fold_counts(ev[s2], du[s2], R[:, s2], [None, Wp[:, s2]])
                    acs[0] += a_p[:, 0]; acs[1] += a_p[:, 1]; acs[2] += t_p
                    acs[3] += a_b[..., 0]; acs[4] += a_b[..., 1]; acs[5] += t_b
            P[:, j] = J.cidx_from(acc[0], acc[1], acc[2])
            B[:, j] = J.cidx_from(acc[3], acc[4], acc[5][None, :])
            if t in SEX_RESTRICTED:
                Ps[:, j] = J.cidx_from(acs[0], acs[1], acs[2])
                Bs[:, j] = J.cidx_from(acs[3], acs[4], acs[5][None, :])
            else:
                Ps[:, j], Bs[:, j] = P[:, j], B[:, j]
        out[scen] = {"models": np.array(models), "point": P, "boot": B, "point_sex": Ps, "boot_sex": Bs}
        log(f"{scen}: point and bootstrap C-indices ({time.time() - t0:.0f} s)")
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez(PRIV / "private_bootstrap_cindex.npz", diseases=np.array(diseases), **{f"{s}__{k}": v for s, d in out.items() for k, v in d.items()})
    done_marker("boot", {"draws": N_BOOT, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage analyse
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in zip(range(n, 0, -1), o[::-1]):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def pct(d, q):
    return float(np.percentile(d, q))


def decision_for(scen):
    return [f"{r}_mean" for r in RAW] + [f"{c}_mean" for c in CLOCK_MODELS] if scen == "full" else DECISION


def complete_v12_weights(diseases):
    """Equal weights over the diseases whose raw_cpg_v12 paths all have the 12 locked points (second amendment 2026-10-30)."""
    ok = []
    for d in diseases:
        lens = []
        for f in FOLDS:
            with np.load(model_path(d, "raw_cpg_v12", f), allow_pickle=False) as m:
                lens += [int(x) for x in m["inner_path_lengths"]] + [int(m["final_path_length"])]
        ok.append(min(lens) == 12)
    ok = np.array(ok, float)
    return ok / ok.sum()


def stage_analyse():
    require_frozen()
    require_done("boot")
    z = np.load(need(PRIV / "private_bootstrap_cindex.npz"), allow_pickle=False)
    diseases = list(z["diseases"].astype(str))
    tab = read_table().set_index("constituent")
    events = np.array([int(tab.loc[d, "events"]) for d in diseases], float)
    ncat = {d: tab.loc[d, "category"] for d in diseases}
    jm = pd.read_csv(need(J.OUTJ / f"J_masks_{J.DATE}.csv")).set_index("scenario")
    per, cons, summ, rules, counts, draws = [], [], [], [], [], []
    res = {"settings": {"seed": SEED, "draws": N_BOOT, "rule": RULE, "label": LABEL, "primary": PRIMARY, "diseases": diseases,
                        "min_events": MIN_EVENTS}, "scenarios": {}}
    W = {"mean": np.ones(len(diseases)) / len(diseases), "event_weighted_mean": events / events.sum(), "sex_restricted_mean": np.ones(len(diseases)) / len(diseases),
         "complete_v12_paths_mean": complete_v12_weights(diseases)}
    for scen in ALL_SCEN:
        models = list(z[f"{scen}__models"].astype(str))
        P, B = z[f"{scen}__point"], z[f"{scen}__boot"]
        ia, io = models.index("age_sex"), models.index(OURS)
        dP, dB = P - P[ia], B - B[ia]
        dec = decision_for(scen)
        dec_i = [models.index(m) for m in dec]
        # per disease and model
        pmat = np.ones((len(models), len(diseases)))
        for k in range(len(models)):
            if k == io:
                continue
            for j in range(len(diseases)):
                pmat[k, j] = J.pboot(dB[io, j] - dB[k, j])
        for k, m in enumerate(models):
            q = bh(pmat[k]) if k != io and m != "age_sex" else np.full(len(diseases), np.nan)
            for j, d in enumerate(diseases):
                dd = dB[io, j] - dB[k, j]
                per.append({"setting": scen, "disease": d, "category": ncat[d], "model": m, "role": role(m), "n": 9866, "events": int(events[j]),
                            "c_index": P[k, j], "c_age_sex": P[ia, j], "delta_c": dP[k, j], "mgpt2_combined_minus_model": dP[io, j] - dP[k, j],
                            "ci95_low": pct(dd, 2.5), "ci95_high": pct(dd, 97.5), "p_boot": pmat[k, j] if k != io else np.nan, "q_bh": q[j]})
        # per disease contrasts
        best_k = [dec_i[int(np.argmax(dP[dec_i, j]))] for j in range(len(diseases))]
        p_best = [J.pboot(dB[io, j] - dB[best_k[j], j]) for j in range(len(diseases))]
        q_best = bh(p_best)
        gk = {g: models.index(g) for g in ("grimage2_knn", "grimage2_mean") if g in models}
        qg = {g: bh([pmat[k, j] for j in range(len(diseases))]) for g, k in gk.items()}
        for j, d in enumerate(diseases):
            kb = best_k[j]
            db = dB[io, j] - dB[kb, j]
            diff = dP[io, j] - dP[kb, j]
            rel = diff / dP[kb, j] if dP[kb, j] > 0 else np.nan
            lo, hi = pct(db, 2.5), pct(db, 97.5)
            meets = "undefined (own best gain <= 0)" if not dP[kb, j] > 0 else bool(rel >= 0.10 and lo > 0)
            row = {"setting": scen, "disease": d, "category": ncat[d], "events": int(events[j]), "c_age_sex": P[ia, j], "c_mgpt2_combined": P[io, j],
                   "delta_c_mgpt2_combined": dP[io, j], "own_best_comparator": models[kb], "c_own_best": P[kb, j], "delta_c_own_best": dP[kb, j],
                   "difference_own_best": diff, "own_best_ci95_low": lo, "own_best_ci95_high": hi, "own_best_p_boot": p_best[j], "own_best_q_bh": q_best[j],
                   "relative_gain_own_best": rel, "meets_rule_against_own_best": meets}
            for g, k in gk.items():
                dg = dB[io, j] - dB[k, j]
                row.update({f"delta_c_{g}": dP[k, j], f"difference_{g}": dP[io, j] - dP[k, j], f"{g}_ci95_low": pct(dg, 2.5), f"{g}_ci95_high": pct(dg, 97.5),
                            f"{g}_p_boot": pmat[k, j], f"{g}_q_bh": qg[g][j]})
            cons.append(row)
        # summaries
        scen_res = {}
        for stat, wt in W.items():
            Pv, Bv = (z[f"{scen}__point_sex"], z[f"{scen}__boot_sex"]) if stat == "sex_restricted_mean" else (P, B)
            dPv, dBv = Pv - Pv[ia], Bv - Bv[ia]
            M, MB = (dPv * wt[None, :]).sum(1), (dBv * wt[None, :, None]).sum(1)
            for k, m in enumerate(models):
                draws.append(pd.DataFrame({"setting": scen, "statistic": stat, "model": m, "draw": np.arange(N_BOOT), "M": MB[k]}))
            comps = [m for m in models if m not in (OURS, "age_sex")]
            best = max(dec, key=lambda m: M[models.index(m)])
            for m in comps:
                k = models.index(m)
                d = MB[io] - MB[k]
                rel = (M[io] - M[k]) / M[k] if M[k] > 0 else np.nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    relb = d / MB[k]
                lo, hi = pct(d, 2.5), pct(d, 97.5)
                outcome = "not applicable (M <= 0)" if not M[k] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
                summ.append({"setting": scen, "statistic": stat, "model": m, "role": role(m), "is_best_decision_comparator": m == best,
                             "M_model": M[k], "M_model_ci95_low": pct(MB[k], 2.5), "M_model_ci95_high": pct(MB[k], 97.5), "M_mgpt2_combined": M[io],
                             "difference": M[io] - M[k], "ci95_low": lo, "ci95_high": hi, "p_boot": J.pboot(d), "relative_gain": rel,
                             "relative_gain_ci95_low": float(np.nanpercentile(relb, 2.5)) if M[k] > 0 else np.nan,
                             "relative_gain_ci95_high": float(np.nanpercentile(relb, 97.5)) if M[k] > 0 else np.nan, "rule_against_this_model": outcome})
            kb = models.index(best)
            d = MB[io] - MB[kb]
            rel = (M[io] - M[kb]) / M[kb] if M[kb] > 0 else np.nan
            lo, hi = pct(d, 2.5), pct(d, 97.5)
            outcome = "not applicable (M_best <= 0)" if not M[kb] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
            lab = "complete profiles (reference, not a decision)" if scen == "full" else LABEL
            rules.append({"analysis": "K", "setting": scen, "statistic": stat, "decision_row": scen == PRIMARY and stat == "mean", "primary_setting": scen == PRIMARY,
                          "label": lab, "kept_vocab_cpgs_per_participant_mean": float(jm.loc[scen, "kept_mean"]), "n_diseases": int((wt > 0).sum()),
                          "metric": {"mean": "mean over primary-set diseases of (pooled within-fold C-index - age+sex C-index)",
                                     "event_weighted_mean": "event-weighted mean of the same gains (sensitivity)",
                                     "sex_restricted_mean": "mean of the same gains, pairs within the at-risk sex in sex-specific cancers (sensitivity)",
                                     "complete_v12_paths_mean": "mean of the same gains over diseases with every raw_cpg_v12 path complete (sensitivity)"}[stat],
                          "methylgpt": OURS, "M_methylgpt": M[io], "M_methylgpt_ci95_low": pct(MB[io], 2.5), "M_methylgpt_ci95_high": pct(MB[io], 97.5),
                          "best_comparator": best, "M_best": M[kb], "relative_gain": rel, "difference": M[io] - M[kb], "ci95_low": lo, "ci95_high": hi,
                          "interval": f"paired bootstrap of participants, one shared resample per draw, {N_BOOT} draws, seed {SEED}",
                          "p_boot": J.pboot(d), "rule": RULE, "outcome": outcome})
            scen_res[stat] = {"M": {m: float(M[models.index(m)]) for m in models if m != "age_sex"}, "best": best, "relative_gain": rel,
                              "difference": float(M[io] - M[kb]), "ci95": [lo, hi], "outcome": outcome}
            log(f"{scen} {stat}: M {OURS} {M[io]:.4f}; best {best} {M[kb]:.4f}; gain {rel:+.1%}; diff {M[io] - M[kb]:+.4f} [{lo:.4f}, {hi:.4f}] -> {outcome}")
        res["scenarios"][scen] = scen_res
        # counts by direction
        cs = [c for c in cons if c["setting"] == scen]
        lst = [("own_best", "difference_own_best", "own_best_ci95_low", "own_best_ci95_high", "own_best_q_bh")]
        lst += [(g, f"difference_{g}", f"{g}_ci95_low", f"{g}_ci95_high", f"{g}_q_bh") for g in gk]
        for name, dc, lc, hc, qc in lst:
            counts.append({"setting": scen, "contrast": name, "n_diseases": len(cs),
                           "n_mgpt2_combined_above": sum(c[dc] > 0 for c in cs), "n_comparator_above": sum(c[dc] < 0 for c in cs),
                           "n_interval_above_zero": sum(c[lc] > 0 for c in cs), "n_interval_below_zero": sum(c[hc] < 0 for c in cs),
                           "n_q05_favouring_mgpt2_combined": sum(c[qc] < 0.05 and c[dc] > 0 for c in cs),
                           "n_q05_favouring_comparator": sum(c[qc] < 0.05 and c[dc] < 0 for c in cs),
                           "n_meets_rule": sum(c["meets_rule_against_own_best"] is True for c in cs) if name == "own_best" else "",
                           "n_rule_undefined": sum(isinstance(c["meets_rule_against_own_best"], str) for c in cs) if name == "own_best" else ""})
    jt = pd.read_csv(need(J.OUTJ / f"J_cindex_per_endpoint_{J.DATE}.csv"))
    jt = jt[jt.scenario.isin(ALL_SCEN) & jt.endpoint.isin(CONTEXT)].copy()
    jt.insert(0, "source", "analysis J: locked folds and fold models; outside K's primary set and statistics")
    for name, df in ((f"K_cindex_per_disease_{DATE}.csv", pd.DataFrame(per)), (f"K_disease_contrasts_{DATE}.csv", pd.DataFrame(cons)),
                     (f"K_direction_counts_{DATE}.csv", pd.DataFrame(counts)), (f"K_summary_{DATE}.csv", pd.DataFrame(summ)),
                     (f"K_rule_outcome_{DATE}.csv", pd.DataFrame(rules)), (f"K_context_endpoints_{DATE}.csv", jt)):
        atomic_csv(OUTK / name, df)
    atomic_csv(OUTK / f"K_bootstrap_draws_{DATE}.csv.gz", pd.concat(draws, ignore_index=True), compression="gzip")
    write_json(OUTK / f"K_results_{DATE}.json", res)
    write_readme(res)
    done_marker("analyse")


# ------------------------------------------------------------------ stage sensitivity (added after the primary result; not part of the decision)
def stage_sensitivity():
    require_done("analyse")
    z = np.load(need(PRIV / "private_bootstrap_cindex.npz"), allow_pickle=False)
    diseases = list(z["diseases"].astype(str))
    rows = []
    for scen in ALL_SCEN:
        models = list(z[f"{scen}__models"].astype(str))
        P, B = z[f"{scen}__point"], z[f"{scen}__boot"]
        ia, io = models.index("age_sex"), models.index(OURS)
        dP, dB = P - P[ia], B - B[ia]
        dec = decision_for(scen)
        for j, d in enumerate(diseases):
            keep = np.arange(len(diseases)) != j
            M, MB = dP[:, keep].mean(1), dB[:, keep].mean(1)
            best = max(dec, key=lambda m: M[models.index(m)])
            kb = models.index(best)
            dd = MB[io] - MB[kb]
            rel = (M[io] - M[kb]) / M[kb] if M[kb] > 0 else np.nan
            lo, hi = pct(dd, 2.5), pct(dd, 97.5)
            rows.append({"setting": scen, "left_out": d, "n_diseases": int(keep.sum()), "M_mgpt2_combined": M[io], "best_comparator": best, "M_best": M[kb],
                         "relative_gain": rel, "difference": M[io] - M[kb], "ci95_low": lo, "ci95_high": hi, "p_boot": J.pboot(dd),
                         "rule_arithmetic_for_reference": ("not applicable (M_best <= 0)" if not M[kb] > 0 else "meets" if rel >= 0.10 and lo > 0 else "does not meet")})
    atomic_csv(OUTK / f"K_leave_one_disease_out_{DATE}.csv", pd.DataFrame(rows))
    log(f"sensitivity: {len(rows)} rows")


def write_readme(res):
    def nrow(fn):
        return "-" if fn.endswith(".json") else str(len(pd.read_csv(OUTK / fn, dtype=str)))

    def cols(fn):
        return "-" if fn.endswith(".json") else ", ".join(pd.read_csv(OUTK / fn, nrows=1).columns)

    desc = {
        f"K_disease_table_{DATE}.csv": "Frozen before any fit: each constituent disease, its category, role (primary, ineligible, context), participants at risk, incident events (cells below 5 shown as <5), eligibility, fold seed and fold hash.",
        f"K_freeze_checks_{DATE}.csv": "Label checks written at the freeze (also inside K_checks).",
        f"K_checks_{DATE}.csv": "Every pre-specified check with stored and recomputed values, tolerance and pass.",
        f"K_fit_record_{DATE}.csv": "Per target, family and outer fold: seconds, selected alpha index, nonzero coefficients, path lengths and recorded warnings.",
        f"K_cindex_per_disease_{DATE}.csv": "Per setting, primary-set disease and model: role, n, events, pooled within-fold C-index, age+sex C-index, gain above age+sex (delta_c), MethylGPT combined minus model with 95% interval, bootstrap P, Benjamini-Hochberg q across primary-set diseases within setting and contrast.",
        f"K_disease_contrasts_{DATE}.csv": "One row per setting and disease: MethylGPT, its own best decision comparator (chosen by point estimate) and GrimAge2 (kNN and mean fill), each difference with interval, P and q; relative gain; whether the disease meets the rule against its own best comparator.",
        f"K_direction_counts_{DATE}.csv": "Per setting and contrast: number of diseases by direction, by interval and by q < 0.05; number meeting the rule.",
        f"K_summary_{DATE}.csv": "Per setting, statistic (mean, event-weighted mean, sex-restricted mean, mean over diseases with complete raw_cpg_v12 paths) and comparator: M with interval, MethylGPT's M, difference with interval, bootstrap P, relative gain with interval, rule against this comparator, best-decision-comparator flag.",
        f"K_rule_outcome_{DATE}.csv": "One row per setting and statistic; decision_row marks hm27 with the plain mean, the pre-specified decision.",
        f"K_context_endpoints_{DATE}.csv": "Type 2 diabetes, kidney disease and death: rows copied from J's per-endpoint table; context only.",
        f"K_bootstrap_draws_{DATE}.csv.gz": "Per setting, statistic, model and draw: M (2,000 draws).",
        f"K_results_{DATE}.json": "Settings and, per setting and statistic, M of every model, best comparator, gain, difference, interval and outcome.",
    }
    sens = f"K_leave_one_disease_out_{DATE}.csv"
    if (OUTK / sens).exists():
        desc[sens] = ("Added after the primary result; not part of the decision. Per setting and left-out disease: M over the other diseases, "
                      "best comparator re-chosen, difference with interval. Rule arithmetic is shown for reference only.")
    pr = res["scenarios"][PRIMARY]["mean"]
    n = len(res["settings"]["diseases"])
    fr = pd.read_csv(OUTK / f"K_fit_record_{DATE}.csv")
    v12 = fr[(fr.family == "raw_cpg_v12") & fr.target.isin(res["settings"]["diseases"])]
    short = v12.path_lengths.astype(str).map(lambda p: min(int(x) for x in p.split()) < 12)
    con = pd.read_csv(OUTK / f"K_disease_contrasts_{DATE}.csv")
    low = con[(con.setting == PRIMARY) & (con.c_age_sex < 0.5)]
    lines = ["# Large-gain analysis K: GS individual incident diseases", "",
             f"Script: `large_gain_K_gs_individual_diseases_2026-10-30.py` (pre-specified design and amendments in its docstring). hm27 and random_90 are {LABEL}: "
             "measured GS development profiles (9,866 participants) with CpGs removed by J's fixed masks; no methylation value is simulated. full is the measured data.", "",
             "## Primary result (hm27)", "",
             f"- {n} diseases in the primary set (constituents of five categories, at least {MIN_EVENTS} incident events).",
             f"- MethylGPT combined M = {pr['M'][OURS]:.4f}; best decision comparator {pr['best']} M = {pr['M'][pr['best']]:.4f}.",
             f"- Relative gain {pr['relative_gain']:+.1%}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             "", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUTK / fn).exists():
            lines.append(f"| `{fn}` | {nrow(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              f"- GS development frame-cache matrix, endpoint table `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet`, constituent columns `{AUDIT.relative_to(HERE)}`, label code `regenerate_disease_survival_data.py` (revision_server_20260622).",
              "- Stored full-profile features `artifacts/gs_features_2026-10-01/*_mgpt2_embed*, *_mt_bio*`; checkpoints `artifacts/runs/aging_heads_clean_2026-10-01/best.pt`, `artifacts/runs/multitask_bio_2026-10-01/final.pt`; stored full-profile clock scores `artifacts/gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv`.",
              "- J's masks, corpus kNN fills and clock scores `artifacts/large_gain_2026-10-28_private/J`, verified against `J_masks` and `J_knn_record`; J's per-endpoint table for the context rows.",
              "", "## Notes", "",
              "- Folds: per disease, 5 outer folds stratified by event status, seeded by the disease name, the same for every model of that disease.",
              "- Interval: one participant resample per draw shared by every model, disease and setting.",
              "- Decision comparators: raw-CpG models (locked v12 protocol, GPU elastic net) with fold-mean or corpus-kNN fill, and GrimAge2, DunedinPACE and both with mean- or corpus-kNN-filled clock CpGs. In-cohort kNN is a secondary comparator; raw models with MethylGPT factor imputation are MethylGPT applications.",
              "- The own-best-comparator contrast picks the comparator by point estimate and keeps it fixed in the draws, so its interval does not include that selection.",
              f"- raw_cpg_v12: {int(short.sum())} of {len(v12)} fold models have at least one path that stops before the 12th grid point ({', '.join(sorted(v12[short].target.unique()))}); "
              f"{int((v12[short].nonzero_coefficients == 0).sum())} of those {int(short.sum())} hold no CpG coefficient, so they carry age and sex only. raw_cpg_gpu is unaffected. "
              "Statistic complete_v12_paths_mean omits the affected diseases.",
              "- Age+sex C-index below 0.5 at the primary setting: " + ("; ".join(f"{r.disease} {r.c_age_sex:.3f}" for r in low.itertuples()) or "none") + ". Gains in such a disease are measured from a baseline that orders held-out participants worse than chance.",
              "- Participant-level labels, features, models, fills, scores and risks stay in `artifacts/large_gain_2026-10-30_private/K/` on Chronos and are never committed."]
    (OUTK / "README.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ stage privacy
def stage_privacy():
    require_frozen()
    require_done("analyse")
    ids = set(load_ids())
    token = re.compile(r"[A-Za-z0-9_]+")
    forbidden = {"dnam_id", "sample_id", "participant", "participant_id", "row", "id"}
    rows = []
    for p in sorted(OUTK.iterdir()):
        text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
        hits = len(set(token.findall(text)) & ids)
        bad_cols, small = [], 0
        if ".csv" in p.name:
            df = pd.read_csv(p, dtype=str)
            bad_cols = sorted(set(c.lower() for c in df.columns) & forbidden)
            if "events" in df.columns:
                v = pd.to_numeric(df["events"], errors="coerce")
                small = int((v < SMALL_CELL).sum())
        rows.append({"check": "9 privacy of public files", "scenario": "all", "item": p.name, "stored": 0, "recomputed": hits + len(bad_cols) + small,
                     "abs_diff": "", "tolerance": 0, "pass": hits == 0 and not bad_cols and small == 0,
                     "note": f"identifier tokens {hits}; identifier columns {bad_cols}; event cells below {SMALL_CELL}: {small}"})
    ck = pd.read_csv(need(OUTK / f"K_checks_{DATE}.csv"))
    ck = ck[ck.check != "9 privacy of public files"]
    out = pd.concat([ck, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(OUTK / f"K_checks_{DATE}.csv", out)
    write_readme(json.loads(need(OUTK / f"K_results_{DATE}.json").read_text()))
    nf = sum(not r["pass"] for r in rows)
    log(f"privacy: {len(rows)} public files, {nf} failed")
    if nf:
        print(pd.DataFrame(rows).query("~`pass`").to_string())
        sys.exit("privacy check failed; do not commit the outputs")
    done_marker("privacy", {"files": len(rows)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["freeze", "features", "fit", "cohort", "clockin", "clocks", "apply", "repcheck", "checks", "boot", "analyse", "sensitivity", "privacy"])
    ap.add_argument("--family", choices=["cheap", "gpu", "v12"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only-target", default=None)
    ap.add_argument("--only-fold", type=int, default=None)
    a = ap.parse_args()
    os.chdir(HERE)          # checkpoint configs hold paths relative to the repository directory
    LOGS.mkdir(parents=True, exist_ok=True)
    {"freeze": stage_freeze, "features": lambda: stage_features(a.device), "fit": lambda: stage_fit(a.family, a.shard, a.nshards, a.device, a.only_target, a.only_fold),
     "cohort": lambda: stage_cohort(a.device, a.shard, a.nshards), "clockin": stage_clockin, "clocks": lambda: stage_clocks(a.workers),
     "apply": lambda: stage_apply(a.shard, a.nshards), "repcheck": stage_repcheck, "checks": stage_checks, "boot": stage_boot,
     "analyse": stage_analyse, "sensitivity": stage_sensitivity, "privacy": stage_privacy}[a.stage]()


if __name__ == "__main__":
    main()
