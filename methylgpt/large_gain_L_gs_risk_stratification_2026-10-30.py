#!/usr/bin/env python3
"""Large-gain analysis L: Generation Scotland (GS) risk stratification of incident disease from analysis J's stored held-out
risks. Tables only. No model is fitted to predict disease and no risk is recomputed: L re-scores the held-out fold risks J
stored. SIMULATED MISSINGNESS ON EMPIRICAL PROFILES for hm27 and random_90 (J's scenarios: measured GS development profiles
with CpGs removed by fixed masks; no methylation value is simulated); full is complete profiles.

Written on 30 Oct 2026 for the large-gain analysis plan of 23 Oct 2026 and the analysis brief for L, BEFORE any
result of this analysis is computed. Only the schema and the follow-up scale of the endpoint table were looked at (days,
range 28 to 6,482, month resolution, heavy ties); no risk has been compared with an outcome here. Every result is reported,
whichever model it favours.

WHY: J measured discrimination by the C-index, which compresses differences in risk separation. L asks how far the same
held-out risks separate people who develop disease, on metrics clinical readers use, with every J comparator kept.

DATA (J's inputs, read through J's own loaders; large_gain_J_gs_missing_cpgs_knn_2026-10-28.py)
  9,866 GS development participants, 8 endpoints (Autoimmune_Diseases, Cancers, Cardiovascular_Diseases,
  Endocrine_and_Metabolic_Diseases, Kidney_Diseases, Neurological_Diseases, Respiratory_Diseases, date_of_death). Every
  participant is at risk for every endpoint. Events, duration, age, sex and the locked outer folds: J.load_preds(). The
  outer folds were assigned separately per endpoint, so every per-fold step below uses that endpoint's folds. Stored
  held-out risks: J.assemble(scenario, preds), i.e. artifacts/large_gain_2026-10-28_private/J/private_risks_{mgpt,clocks,knn}
  _<scenario>.parquet and cohort/private_cohortknn_<scenario>__<endpoint>.parquet. No validation or test participant is
  read: only the training-subset label table of the frame cache is read, to check the 9,866 identifiers.

MODELS, in scenarios hm27 (PRIMARY), full and random_90
  MethylGPT combined (mgpt2_combined).
  Decision comparators (J's decision set): hm27 and random_90: raw_cpg_v12_{mean,knn}, raw_cpg_gpu_{mean,knn},
    grimage2_{mean,knn}, dunedinpace_{mean,knn}, grimage2_dunedinpace_{mean,knn}. full: J has no kNN variants on complete
    profiles (no CpG is removed), so its decision set there is raw_cpg_v12_mean, raw_cpg_gpu_mean, grimage2_mean,
    dunedinpace_mean, grimage2_dunedinpace_mean.
  J secondary: *_cohortknn (5 models; not in full, as in J). J applications: raw_cpg_{v12,gpu}_factor_imputed (every
    scenario). J's components and age_sex are not L models; the age_sex risk is the positive control of the constant-risk
    test (CHECKS).
  "GrimAge2 with kNN fill" = grimage2_knn in hm27 and random_90. In full it is grimage2_mean (complete profiles, nothing to
    fill): the same mapping _knn -> _mean applies to C* in full.

PREPARATION, per endpoint, model and scenario, within each outer fold, on that fold's held-out participants
  r = stored held-out risk. Fit r ~ 1 + age + sex by least squares in the fold; e = residual. Constant-risk test: the fold
  is NOT ESTIMABLE for this model if SD(e) <= 1e-5 x SD(r) (this includes SD(r) = 0; the factor is above the float32
  rounding noise of the GPU fold models, whose risk is computed in single precision). Otherwise the prepared risk is
  x = (e - mean(e)) / SD(e), SD with ddof = 0 (J's within-fold standardisation).
  Deciles, within each estimable fold: k = floor(n_fold / 10) (197 for every fold; 985 participants in each pooled group,
  so the exact chance share is 985 / 9,866 = 9.98%). Participants sorted by x ascending, ties broken by their position in
  the endpoint table: bottom decile = first k, top decile = last k, middle 80% = the rest.
  Not estimable folds: recorded per endpoint, model, scenario and fold (L_preparation). In the Cox models x (or the decile
  indicators) is set to 0 in that fold, so the fold's stratum informs age and sex only. Its participants have no decile and
  are left out of event capture and of the Kaplan-Meier groups. If every fold of an endpoint is not estimable, the
  endpoint's per-endpoint metrics are blank; for H only, beta = 0 is used for that endpoint (the risk adds nothing beyond
  age and sex), the row is flagged, and H over the estimable endpoints is reported beside it.

PRIMARY METRIC AND DECISION
  beta(model, e) = log hazard ratio per SD of x in a Cox model on all 9,866 participants with age, sex and x (Breslow ties,
  stratified by the outer fold of endpoint e). H(model) = mean over the 8 endpoints of beta (all 8; no endpoint is picked).
  C* = the decision comparator with the largest H at hm27 (identity fixed at the point estimate, as J does).
  PASS iff (H_MethylGPT - H_C*) / H_C* >= 0.10 and the lower end of the 95% interval of H_MethylGPT - H_C* is > 0; otherwise
  FAIL. If H_C* <= 0 this is stated, no ratio is reported and the rule is not applicable. The decision is hm27 only. The
  rule table also gives each secondary scenario against that scenario's own best decision comparator (J's convention);
  the full row is a reference.

INTERVALS: paired bootstrap of participants, 2,000 draws, seed 20261030, percentile 95% intervals (np.percentile, J's).
  I = default_rng(20261030).integers(0, 9866, size=(2000, 9866)); w_b = bincount(I[b], minlength=9866), indexed by the frame
  cache order. One resample per draw, shared by every model, endpoint and scenario. Every statistic is recomputed in every
  draw with frequency weights w_b (weighted Breslow Cox refitted to convergence per draw; weighted event share; weighted
  Kaplan-Meier). The preparation step is held fixed in the draws: x and the decile membership are not recomputed. C* and
  each endpoint's own best comparator are fixed at the point estimates. Means over endpoints are recomputed in every draw.
  A top-versus-bottom draw with zero resampled events in the top or bottom decile has no finite estimate: that endpoint
  and its means are not estimable in that draw, the interval is taken over the remaining draws and n_draws_estimable is
  reported. Bootstrap P = min(1, 2 (k + 1) / (B + 1)), k = min(#{d_b <= 0}, #{d_b >= 0}), as in J.

SECONDARY (per endpoint and as the mean over the 8 endpoints, every model and scenario, intervals from the same draws)
  log_hr_per_sd: beta; hazard ratio per SD = exp(beta); the mean row is the mean of beta and exp of it (geometric mean HR).
  log_hr_top_vs_bottom: one Cox model on all 9,866 participants with age, sex, a top-decile and a bottom-decile indicator
    (reference: middle 80%), Breslow ties, stratified by outer fold, deciles formed within fold. Contrast = b_top - b_bottom;
    hazard ratio = exp of it; mean row as above.
  event_capture_top10: share of the endpoint's incident events (whole follow-up) that fall in the top decile.
    event_capture_excess_over_10pct = that share - 0.10 (chance 10%; chance_exact = 985 / 9,866 is also given).
  abs_risk_10y_top, abs_risk_10y_bottom: 1 - S(10 years), Kaplan-Meier pooled over folds within the decile; 10 years =
    3,652.5 days; other causes censor (no competing-risk model). abs_risk_10y_difference = top - bottom.
  KM curves: aggregate table on the yearly grid 0, 1, ..., 17 years (365.25-day years; follow-up ends at 17.7 years) for the
    top decile, the middle 80% and the bottom decile: survival with interval, numbers at risk (duration >= t), cumulative
    events (duration <= t), for mgpt2_combined and GrimAge2 with kNN fill, every endpoint, hm27 and full, plus the mean over
    endpoints per year and group.
  Contrasts: every metric, per endpoint and as the mean over endpoints, MethylGPT minus (a) C* (the hm27 best on H, in every
    scenario), (b) the scenario's own best decision comparator on H, (c) GrimAge2 with kNN fill, (d) each endpoint's own best
    decision comparator on that metric in that scenario (largest value; smallest for abs_risk_10y_bottom; the mean row
    averages the per-endpoint differences). Interval from the same draws.

CHECKS, written to L_checks before any result is used; the boot stage refuses to run unless all pass. Tolerances stated here,
before running.
  (1) J's pooled within-fold C-indices (J_cindex_per_endpoint c_index) reproduce from the stored risks to 1e-6 with
      J.sksurv_pooled, for every L model and age_sex, 8 endpoints, hm27, full and random_90.
  (2) Identifiers: 9,866 unique participants per endpoint, the same set as the frame-cache training labels; 5 folds each.
  (3) Preparation: in every estimable fold |mean(x)| <= 1e-9, |SD(x) - 1| <= 1e-9, |corr(x, age)| and |corr(x, sex)| <= 1e-9.
      Positive control: the age_sex risk is flagged not estimable in all 40 endpoint-folds of every scenario.
  (4) Decile membership: for every scenario, model, endpoint and estimable fold, top = bottom = floor(n_fold / 10), middle =
      n_fold - 2 k, the groups partition the fold and are ordered by x; ties across a boundary are counted.
  (5) Cox against statsmodels PHReg(ties="breslow", strata = outer fold) on the same design. Tolerance: 1e-5 absolute on every
      coefficient. Sample: per scenario and endpoint, mgpt2_combined and GrimAge2 with kNN fill (48 pairs), plus 12 further
      (scenario, model, endpoint) triples from default_rng(20261030); each for the continuous and the decile Cox model.
      Frequency weights: for 3 pairs and 2 draws the weighted fit is compared with an unweighted PHReg on the expanded rows,
      same tolerance.
  (6) Kaplan-Meier against sksurv.nonparametric.kaplan_meier_estimator, mgpt2_combined hm27, every endpoint, top and bottom
      deciles, on the yearly grid, tolerance 1e-10; frequency weights against expanded rows for 2 draws, tolerance 1e-10.
  (7) Bootstrap: every draw sums to 9,866; SHA-256 of the weight matrix recorded.

DISCLOSURE CONTROL, applied to every public file, verified by stage verify before anything is committed
  Primary: any aggregate cell with fewer than 5 events or fewer than 5 people at risk is suppressed.
    log_hr_top_vs_bottom: fewer than 5 events in the top or bottom decile (whole follow-up).
    event_capture: fewer than 5 events in the top decile, or fewer than 5 outside it.
    abs_risk_10y: fewer than 5 events by 10 years or fewer than 5 at risk at 10 years in the decile, or 1 to 4 events after 10
      years in the decile (total minus the 10-year count would reveal them).
    KM rows: as abs_risk at each year, in addition (subtraction between released rows): a row is released only if at least 5
      events separate it from the previous released row of that group and from the anchor row at 10 years. A row hides
      survival, its interval, numbers at risk and cumulative events together, because n minus at risk reveals events before
      censoring starts.
    log_hr_per_sd is continuous over the whole cohort (at least 135 events per endpoint) and has no small cell.
  Complementary (subtraction across cells): the group total per year can be derived from another released model at that
    year, and the endpoint total is public, so a KM year with exactly one group hidden hides a second one (the middle 80%
    if top or bottom is hidden, otherwise the bottom decile). A mean over endpoints is blank if any of its 8 per-endpoint
    cells is suppressed (one suppressed cell could be recovered from the mean and the other seven); n_endpoints_suppressed
    gives the count. A contrast is blank if either of its two cells is suppressed. For mgpt2_combined and the GrimAge2
    reference, abs_risk_10y cells follow the 10-year row of the KM table.
  Public survival and absolute risks are rounded to 4 decimals. No participant identifier or participant-level value goes
  to a public file. Unsuppressed aggregates stay in the private directory on Chronos.

STAGES (resumable; done-markers artifacts/logs/large_gain_L_<stage>_2026-10-30.done; every stage refuses to run unless this
  script is committed unchanged)
  prep -> checks -> boot -> analyse -> verify. CPU only.
OUTPUTS
  Public: artifacts/large_gain_2026-10-30/L/ (README.md, L_rule_outcome, L_summary, L_metrics, L_contrasts, L_km_curves,
  L_preparation, L_checks, L_results, L_disclosure_check). Private, uncommitted: artifacts/large_gain_2026-10-30_private/L/.
Run: python3 large_gain_L_gs_risk_stratification_2026-10-30.py --stage <stage> [--workers N]

AMENDMENT 30 Oct 2026, execution only, after the first checks run stopped and before any Cox, Kaplan-Meier or check output
existed: L_preparation showed that the constant-risk branch of the design occurs (raw_cpg_gpu_* on Neurological_Diseases in
all 5 folds; raw_cpg_gpu_mean on Kidney_Diseases at hm27 in all 5 folds; other raw models and mgpt2_combined in some folds).
The test separates cleanly: largest flagged SD ratio 4.9e-8, smallest estimable 0.0054. No outcome had been compared with any
risk. Stage checks stopped because check 5's sample, as drawn, holds one triple with no estimable fold (hm27 raw_cpg_gpu_knn
Neurological_Diseases): its prepared-risk column is all zeros, so there is no coefficient to compare and PHReg raises a singular
matrix. Changes: (a) check 5 keeps the sample as drawn and records such a triple as not applicable; (b) cells of an endpoint
with no estimable fold are labelled "not estimable" instead of falling under the small-cell rule (they stay blank; the H rule
for them is unchanged); (c) chance_exact uses each row's own group sizes (n_top / participants in estimable folds), because a
not estimable fold changes the pooled group size. Design, primary metric, rule, comparators, seed and tolerances are unchanged.

AMENDMENT 30 Oct 2026, verification and reporting only, AFTER THE PRIMARY RESULT WAS KNOWN, before anything from analyse was
committed. No fit, estimate, table of the analyse stage, or decision changes. (a) Stage verify stopped on two KM groups: it took
the group total from the yearly grid (year 17), but follow-up runs to 17.7 years and the release rule uses the whole-follow-up
total. With that total every released group passes (no true remainder of 1 to 4 events), so verify now reads km_total. (b)
Verify also requires that the endpoint total minus events_total, which a not estimable fold exposes as a fold-level count, is 0
or at least 5. (c) Stage sensitivity is ADDED AFTER THE PRIMARY RESULT and is NOT part of the decision; it reports two readings of
how far H depends on one handling choice and on one endpoint. fold_weighted: beta x (estimable folds / 5), counting a constant-
risk fold as carrying no signal rather than leaving it out (the pre-specified H gives such an endpoint beta = 0 only when all 5
folds are constant), best decision comparator reselected on this quantity. leave_one_endpoint_out: H over the other 7 endpoints,
each endpoint in turn, comparator fixed at the scenario's best on all 8. All 8 endpoints stay in H and in every table; no
endpoint is dropped from any result. Same draws, seed and interval.
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
DATE = "2026-10-30"
OUT = ART / "large_gain_2026-10-30/L"
PRIV = ART / "large_gain_2026-10-30_private/L"
LOGS = ART / "logs"
JDIR = ART / "large_gain_2026-10-28/J"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"

SEED = 20261030
N_BOOT = 2000
N = 9866
SCENARIOS = ["hm27", "full", "random_90"]
PRIMARY = "hm27"
KM_SCENARIOS = ["hm27", "full"]
CONST_RATIO = 1e-5
YEAR = 365.25
T10 = 10 * YEAR
YEARS = np.arange(18)
ANCHOR = 10
MIN_CELL = 5
CHANCE = 0.10
TOL_C = 1e-6
TOL_PREP = 1e-9
TOL_COX = 1e-5
TOL_KM = 1e-10
GROUPS = ["top_decile", "middle_80pct", "bottom_decile"]
COUNT_NAMES = ["n_top", "n_bottom", "events_total", "events_top", "events_bottom", "events_10y_top", "events_10y_bottom",
               "at_risk_10y_top", "at_risk_10y_bottom", "n_folds_estimable", "n_estimable"]
ID_PATTERN = re.compile(r"\b\d{12}_R\d{2}C\d{2}\b")
ID_COLUMNS = {"dnam_id", "sample_id", "participant_id", "id", "sample", "participant", "barcode"}
RULE = ("PASS iff (H_MethylGPT - H_best) / H_best >= 0.10 and the lower end of the 95% interval of H_MethylGPT - H_best is > 0; "
        "H = mean over the 8 endpoints of the log hazard ratio per SD of the prepared risk; best = decision comparator with "
        "the largest H in the scenario (the decision is hm27)")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


if not J_SCRIPT.exists():
    sys.exit(f"Cannot proceed - required data file not found: {J_SCRIPT}")
J = load_module("large_gain_J", J_SCRIPT)
ENDPOINTS = J.ENDPOINTS
OURS = J.OURS
LABELS = {"hm27": J.LABEL + " (27K-array CpGs kept; primary)", "random_90": J.LABEL + " (90% of CpGs removed at random)",
          "full": "complete profiles (reference, not a decision)"}
METRICS = ["log_hr_per_sd", "log_hr_top_vs_bottom", "event_capture_top10", "event_capture_excess_over_10pct", "abs_risk_10y_top",
           "abs_risk_10y_bottom", "abs_risk_10y_difference"]
DIRECTION = {m: (-1 if m == "abs_risk_10y_bottom" else 1) for m in METRICS}


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def atomic_csv(path, df):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def done_path(stage):
    return LOGS / f"large_gain_L_{stage}_{DATE}.done"


def done_marker(stage, info=None):
    write_json(done_path(stage), {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})})


def require_done(*stages):
    for s in stages:
        if not done_path(s).exists():
            sys.exit(f"stage {s} has not completed ({done_path(s)} missing)")


def require_committed():
    me = Path(__file__).resolve()
    if subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(me)], capture_output=True).returncode or \
            subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(me)]).returncode:
        sys.exit("Cannot proceed - the script must be committed unchanged before any result-producing stage runs")


def l_models(scen):
    """(models, decision comparators) of a scenario, in J's order."""
    ms = [m for m in J.models_for(scen) if m != "age_sex" and m not in J.COMPONENTS]
    full_dec = [f"{r}_mean" for r in J.RAW] + [f"{c}_mean" for c in J.CLOCK_MODELS]
    return ms, [m for m in ms if m in (full_dec if scen == "full" else J.DECISION)]


def grimage_ref(scen):
    return "grimage2_mean" if scen == "full" else "grimage2_knn"


def in_scenario(name, scen):
    return name.replace("_knn", "_mean") if scen == "full" else name


def load_ids():
    mp = J.need_glob(J.CACHE, f"COMMON_SAMPLE_SPLIT_{J.SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")
    return pd.read_parquet(need(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet"))),
                           columns=["dnam_id"]).dnam_id.astype(str).to_numpy()


# ------------------------------------------------------------------ preparation
def prepare(raw, age, sex, fold):
    """Prepared risk x, decile group g (2 top, 1 middle, 0 bottom, -1 not estimable) and one record per fold."""
    x, g, rec = np.zeros(len(raw)), np.full(len(raw), -1, np.int8), []
    for f in sorted(np.unique(fold)):
        s = np.flatnonzero(fold == f)
        A = np.column_stack([np.ones(len(s)), age[s], sex[s]])
        r = raw[s]
        e = r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
        sd_r, sd_e = float(r.std()), float(e.std())
        est = bool(sd_e > CONST_RATIO * sd_r)
        k = len(s) // 10
        ties = 0
        n_top = n_mid = n_bot = 0
        if est:
            z = (e - e.mean()) / sd_e
            x[s] = z
            o = np.lexsort((s, z))
            g[s] = 1
            g[s[o[:k]]] = 0
            g[s[o[-k:]]] = 2
            ties = int(z[o[k - 1]] == z[o[k]]) + int(z[o[-k - 1]] == z[o[-k]])
            n_top, n_mid, n_bot = int((g[s] == 2).sum()), int((g[s] == 1).sum()), int((g[s] == 0).sum())
        rec.append({"outer_fold": int(f), "n_fold": len(s), "raw_sd": sd_r, "residual_sd": sd_e,
                    "residual_to_raw_sd": sd_e / sd_r if sd_r > 0 else np.nan, "estimable": est, "n_top": n_top,
                    "n_middle": n_mid, "n_bottom": n_bot, "boundary_ties": ties})
    return x, g, rec


def stage_prep():
    require_committed()
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
        A = J.assemble(scen, preds)
        models, _ = l_models(scen)
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
                    rows.append({"scenario": scen, "endpoint": ep, "model": m, "role": J.role(m), **r})
        atomic_npz(PRIV / f"private_prepared_{scen}.npz", X=X, G=G, models=np.array(models))
        log(f"{scen}: {len(models)} models prepared")
    atomic_csv(OUT / f"L_preparation_{DATE}.csv", pd.DataFrame(rows))
    done_marker("prep", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ Cox (Breslow ties, strata, frequency weights)
class CoxData:
    """Risk sets of one endpoint, per stratum: rows sorted by descending duration; a risk set is a prefix of them."""

    def __init__(self, ev, du, fold):
        self.strata = []
        for f in sorted(np.unique(fold)):
            s = np.flatnonzero(fold == f)
            o = s[np.argsort(-du[s], kind="stable")]
            d = du[o]
            ep = np.flatnonzero(ev[o])
            last = np.searchsorted(-d, -d, side="right") - 1
            _, first = np.unique(-d[ep], return_index=True)
            self.strata.append((o, ep, first, last[ep[first]]))


def cox_newton(cd, Z, W=None, beta0=None, valid=None, maxit=50, tol=1e-9):
    """Newton solver for the stratified Breslow partial likelihood with frequency weights.
    Z (n, p); W (B, n) or None; beta0 (p); valid (B) False skips a draw. Returns beta (B, p), NaN if skipped or not converged."""
    p = Z.shape[1]
    B = 1 if W is None else W.shape[0]
    beta = np.zeros((B, p)) if beta0 is None else np.tile(beta0, (B, 1))
    skip = np.zeros(B, bool) if valid is None else ~valid
    conv = np.zeros(B, bool)
    for _ in range(maxit):
        act = np.flatnonzero(~(skip | conv))
        if act.size == 0:
            break
        g = np.zeros((act.size, p))
        H = np.zeros((act.size, p, p))
        for o, ep, first, ends in cd.strata:
            Zs = Z[o]
            Ws = np.ones((act.size, len(o))) if W is None else W[np.ix_(act, o)]
            r = Ws * np.exp(beta[act] @ Zs.T)
            c0 = np.cumsum(r, 1)[:, ends]
            inv = np.divide(1.0, c0, out=np.zeros_like(c0), where=c0 > 0)
            We = Ws[:, ep]
            De = np.add.reduceat(We, first, axis=1)
            m1 = [np.cumsum(r * Zs[:, a], 1)[:, ends] * inv for a in range(p)]
            for a in range(p):
                g[:, a] += (We * Zs[ep, a]).sum(1) - (De * m1[a]).sum(1)
                for b in range(a, p):
                    v = (De * (np.cumsum(r * Zs[:, a] * Zs[:, b], 1)[:, ends] * inv - m1[a] * m1[b])).sum(1)
                    H[:, a, b] += v
                    if b != a:
                        H[:, b, a] += v
        try:
            delta = np.linalg.solve(H, g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            delta = np.stack([np.linalg.lstsq(H[i], g[i], rcond=None)[0] for i in range(act.size)])
        delta = np.clip(delta, -1.5, 1.5)
        beta[act] += delta
        conv[act] = np.abs(delta).max(1) < tol
    beta[~conv] = np.nan
    return beta


def design(age, sex, *cols):
    return np.column_stack([(age - age.mean()) / age.std(), sex - sex.mean()] + list(cols))


# ------------------------------------------------------------------ Kaplan-Meier with frequency weights
def km_at(du, ev, times, W=None):
    """S(t) at the given times, (B, len(times)); NaN where no weight is at risk at t (duration >= t)."""
    o = np.argsort(du, kind="stable")
    d, e = du[o], ev[o].astype(bool)
    Ws = np.ones((1, len(o))) if W is None else np.asarray(W)[:, o].astype(float)
    risk = np.cumsum(Ws[:, ::-1], 1)[:, ::-1]
    risk_end = np.hstack([risk, np.zeros((len(risk), 1))])
    ep = np.flatnonzero(e)
    if ep.size:
        ut, first = np.unique(d[ep], return_index=True)
        n_t = risk[:, np.searchsorted(d, ut, side="left")]
        d_t = np.add.reduceat(Ws[:, ep], first, axis=1)
        frac = np.divide(d_t, n_t, out=np.zeros_like(d_t), where=n_t > 0)
        S_t = np.hstack([np.ones((len(Ws), 1)), np.cumprod(1 - frac, axis=1)])
        S = S_t[:, np.searchsorted(ut, times, side="right")]
    else:
        S = np.ones((len(Ws), len(times)))
    atr = risk_end[:, np.searchsorted(d, times, side="left")]
    return np.where(atr > 0, S, np.nan)


# ------------------------------------------------------------------ stage checks
def _phreg(args):
    """One comparison with statsmodels: unweighted, or weighted against expanded rows (draw is not None)."""
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
    # (1) J's C-indices from the stored risks
    jc = pd.read_csv(need(JDIR / "J_cindex_per_endpoint_2026-10-28.csv")).set_index(["scenario", "endpoint", "model"]).sort_index()
    for scen in SCENARIOS:
        A = J.assemble(scen, preds)
        models, _ = l_models(scen)
        for j, ep in enumerate(ENDPOINTS):
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            folds = [np.flatnonzero(g.outer_fold.to_numpy() == f) for f in sorted(g.outer_fold.unique())]
            for m in ["age_sex"] + models:
                add("1 J C-index reproduces from the stored risks", scen, f"{ep}:{m}", float(jc.loc[(scen, ep, m), "c_index"]),
                    float(J.sksurv_pooled(ev, du, g[m].to_numpy(float), folds)), TOL_C)
    log(f"check 1 done ({time.time() - t0:.0f} s)")
    # (2) identifiers
    for j, ep in enumerate(ENDPOINTS):
        pe = preds[ep]
        add("2 participants", "all", ep, f"{N} unique, frame-cache training labels, 5 folds",
            f"{pe.dnam_id.nunique()} unique, {pe.outer_fold.nunique()} folds", "",
            passed=len(pe) == N and pe.dnam_id.nunique() == N and set(pe.dnam_id) == set(ids) and pe.outer_fold.nunique() == 5)
    # (3) preparation, (4) deciles
    prep = pd.read_csv(need(OUT / f"L_preparation_{DATE}.csv"))
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
    log(f"checks 2-4 done ({time.time() - t0:.0f} s)")
    # (5) Cox against PHReg
    pairs = [(s, m, j) for s in SCENARIOS for j in range(len(ENDPOINTS)) for m in (OURS, grimage_ref(s))]
    rest = [(s, m, j) for s in SCENARIOS for m in l_models(s)[0] for j in range(len(ENDPOINTS)) if (s, m, j) not in pairs]
    pick = np.random.default_rng(SEED).choice(len(rest), 12, replace=False)
    pairs += [rest[i] for i in pick]
    tasks = [(s, m, j, form, None) for (s, m, j) in pairs for form in ("continuous", "decile")]
    tasks += [("hm27", OURS, 0, "continuous", 0), ("hm27", OURS, 4, "decile", 1), ("full", grimage_ref("full"), 7, "continuous", 0),
              ("full", grimage_ref("full"), 7, "decile", 1), ("random_90", OURS, 5, "continuous", 1), ("random_90", OURS, 2, "decile", 0)]
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
    P = np.load(PRIV / "private_prepared_hm27.npz")
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
            add("6 Kaplan-Meier equals sksurv", "hm27", f"{ep}:{gn}", 0.0, float(np.max(np.abs(mine[ok] - ref[ok]))), TOL_KM)
            for b in (0, 1):
                w = Wall[b, T["position"][j]][s].astype(int)
                rep = np.repeat(np.arange(int(s.sum())), w)
                a = km_at(du[s], ev[s], days, W=w[None, :].astype(float))[0]
                c = km_at(du[s][rep], ev[s][rep], days)[0]
                fin = np.isfinite(a) & np.isfinite(c)
                add("6 weighted Kaplan-Meier equals expanded rows", "hm27", f"{ep}:{gn}:draw{b}", 0.0,
                    float(np.max(np.abs(a[fin] - c[fin]))), TOL_KM)
    # (7) bootstrap
    W = np.load(PRIV / "private_boot_weights.npy")
    add("7 bootstrap weights", "all", "every draw sums to 9,866", N, int(W.sum(1).min()), 0,
        passed=bool((W.sum(1) == N).all()), note="sha256 " + hashlib.sha256(W.tobytes()).hexdigest())
    tab = pd.DataFrame(rows)
    atomic_csv(OUT / f"L_checks_{DATE}.csv", tab)
    nf = int((~tab["pass"]).sum())
    log(f"checks: {len(tab)} rows, {nf} failed ({time.time() - t0:.0f} s)")
    if nf:
        print(tab[~tab["pass"]].to_string())
        sys.exit("checks failed; no result is used")
    done_marker("checks", {"rows": len(tab), "failed": 0})


# ------------------------------------------------------------------ stage boot
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
        kms = [OURS, grimage_ref(scen)]
        Sp, Sb = np.full((2, 3, len(YEARS)), np.nan), np.full((2, 3, N_BOOT, len(YEARS)), np.nan)
        atr, cum, tot, ng = np.zeros((2, 3, len(YEARS))), np.zeros((2, 3, len(YEARS))), np.zeros((2, 3)), np.zeros((2, 3))
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
        out.update(km_S=Sp, km_boot=Sb, km_atrisk=atr, km_cum=cum, km_total=tot, km_n=ng)
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


# ------------------------------------------------------------------ disclosure helpers
def release_row(cum, atr, tot):
    """Primary rule for one row: events, people at risk and the events after this row."""
    rem = tot - cum
    return bool(cum >= MIN_CELL and atr >= MIN_CELL and (rem == 0 or rem >= MIN_CELL))


def release_group(cum, atr, tot):
    """Rows of one KM group that may be released; subtraction between released rows never gives fewer than 5 events."""
    ok = np.array([release_row(c, a, tot) for c, a in zip(cum, atr)])
    rel = np.zeros(len(cum), bool)
    anchor = ANCHOR if ok[ANCHOR] else None
    if anchor is not None:
        rel[anchor] = True
    last = 0.0
    for y in range(len(cum)):
        if y == anchor:
            last = cum[y]
            continue
        if not ok[y] or cum[y] - last < MIN_CELL:
            continue
        if anchor is not None and y < anchor and cum[anchor] - cum[y] < MIN_CELL:
            continue
        rel[y] = True
        last = cum[y]
    return rel


def release_km(cum, atr, tot):
    """(3, Y) release mask for top, middle, bottom, with the complementary rule."""
    rel = np.stack([release_group(cum[g], atr[g], tot[g]) for g in range(3)])
    for y in range(rel.shape[1]):
        if rel[:, y].sum() == 2:
            hidden = int(np.flatnonzero(~rel[:, y])[0])
            rel[1 if hidden != 1 else 2, y] = False
    return rel


def pct(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return np.nan, np.nan, 0
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), int(a.size)


def pboot(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return J.pboot(d) if d.size else np.nan


# ------------------------------------------------------------------ stage analyse
def load_fit(scen):
    fits = [np.load(need(PRIV / f"private_fit_{scen}__{ep}.npz")) for ep in ENDPOINTS]
    F = {"models": list(fits[0]["models"].astype(str))}
    for key in ("beta", "tb", "cap", "rt", "rb"):
        F[f"{key}_pt"] = np.stack([f[f"{key}_pt"] for f in fits], 1)        # (m, e)
        F[f"{key}_boot"] = np.stack([f[f"{key}_boot"] for f in fits], 1)    # (m, e, B)
    F["counts"] = np.stack([f["counts"] for f in fits], 1)                  # (m, e, c)
    if scen in KM_SCENARIOS:
        for key in ("km_S", "km_boot", "km_atrisk", "km_cum", "km_total", "km_n"):
            F[key] = np.stack([f[key] for f in fits], 0)                    # (e, model, group, ...)
    return F


def metric_values(F):
    return {"log_hr_per_sd": (F["beta_pt"], F["beta_boot"]),
            "log_hr_top_vs_bottom": (F["tb_pt"], F["tb_boot"]),
            "event_capture_top10": (F["cap_pt"], F["cap_boot"]),
            "event_capture_excess_over_10pct": (F["cap_pt"] - CHANCE, F["cap_boot"] - CHANCE),
            "abs_risk_10y_top": (F["rt_pt"], F["rt_boot"]),
            "abs_risk_10y_bottom": (F["rb_pt"], F["rb_boot"]),
            "abs_risk_10y_difference": (F["rt_pt"] - F["rb_pt"], F["rt_boot"] - F["rb_boot"])}


def suppression(F, scen, km_rel):
    """Boolean (m, e) per metric and a reason per metric."""
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
        for a, m in enumerate([OURS, grimage_ref(scen)]):
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
    FITS, KMREL, VALS, SUP = {}, {}, {}, {}
    for scen in SCENARIOS:
        F = FITS[scen] = load_fit(scen)
        rel = None
        if scen in KM_SCENARIOS:
            rel = np.stack([np.stack([release_km(F["km_cum"][j, a], F["km_atrisk"][j, a], F["km_total"][j, a]) for a in range(2)])
                            for j in range(len(ENDPOINTS))])     # (e, model, group, Y)
        KMREL[scen] = rel
        VALS[scen] = metric_values(F)
        SUP[scen] = suppression(F, scen, rel)
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
        _, dec = l_models(scen)
        ms = F["models"]
        best[scen] = max(dec, key=lambda m: Hp[scen][ms.index(m)])
    cstar = best[PRIMARY]
    summ, rules, res = [], [], {"settings": {"seed": SEED, "draws": N_BOOT, "rule": RULE, "primary": PRIMARY, "C_star": cstar,
                                             "minimum_cell": MIN_CELL, "label": J.LABEL}, "scenarios": {}}
    for scen in SCENARIOS:
        ms = FITS[scen]["models"]
        io = ms.index(OURS)
        _, dec = l_models(scen)
        P, B = Hp[scen], Hb[scen]
        for k, m in enumerate(ms):
            if m == OURS:
                continue
            d = B[io] - B[k]
            lo, hi, _ = pct(d)
            rel = (P[io] - P[k]) / P[k] if P[k] > 0 else np.nan
            outcome = "not applicable (H <= 0)" if not P[k] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
            summ.append({"scenario": scen, "label": LABELS[scen], "model": m, "role": J.role(m),
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
        rules.append({"analysis": "L", "scenario": scen, "primary": scen == PRIMARY, "label": LABELS[scen],
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
        _, dec = l_models(scen)
        c = {n: F["counts"][..., i] for i, n in enumerate(COUNT_NAMES)}
        sup, why = SUP[scen]
        for metric in METRICS:
            P, B = VALS[scen][metric]
            S = sup[metric]
            for k, m in enumerate(ms):
                for j, ep in enumerate(ENDPOINTS):
                    lo, hi, nd = pct(B[k, j])
                    row = {"scenario": scen, "label": LABELS[scen], "endpoint": ep, "model": m, "role": J.role(m), "metric": metric,
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
                mrows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "model": m, "role": J.role(m),
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

    # KM table
    krows = []
    for scen in KM_SCENARIOS:
        F = FITS[scen]
        rel = KMREL[scen]
        for a, m in enumerate([OURS, grimage_ref(scen)]):
            for b, gn in enumerate(GROUPS):
                for y in range(len(YEARS)):
                    for j, ep in enumerate(ENDPOINTS):
                        lo, hi, nd = pct(F["km_boot"][j, a, b, :, y])
                        ok = bool(rel[j, a, b, y])
                        krows.append({"scenario": scen, "label": LABELS[scen], "endpoint": ep, "model": m, "role": J.role(m), "group": gn,
                                      "year": int(YEARS[y]), "n_group": F["km_n"][j, a, b], "survival": F["km_S"][j, a, b, y],
                                      "survival_ci95_low": lo, "survival_ci95_high": hi, "n_at_risk": F["km_atrisk"][j, a, b, y],
                                      "cum_events": F["km_cum"][j, a, b, y], "suppressed": not ok,
                                      "reason": "" if ok else "fewer than 5 events or people at risk, or a subtraction would reveal such a cell"})
                    mb = F["km_boot"][:, a, b, :, y].mean(0)
                    lo, hi, nd = pct(mb)
                    nsup = int((~rel[:, a, b, y]).sum())
                    krows.append({"scenario": scen, "label": LABELS[scen], "endpoint": "mean_over_endpoints", "model": m, "role": J.role(m),
                                  "group": gn, "year": int(YEARS[y]), "survival": float(F["km_S"][:, a, b, y].mean()), "survival_ci95_low": lo,
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
    atomic_csv(OUT / f"L_metrics_{DATE}.csv", pm)
    atomic_csv(OUT / f"L_contrasts_{DATE}.csv", pc)
    atomic_csv(OUT / f"L_km_curves_{DATE}.csv", pk)
    atomic_csv(OUT / f"L_summary_{DATE}.csv", pd.DataFrame(summ))
    atomic_csv(OUT / f"L_rule_outcome_{DATE}.csv", pd.DataFrame(rules))
    write_json(OUT / f"L_results_{DATE}.json", res)
    write_readme(res)
    done_marker("analyse", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage sensitivity (added after the primary result; not part of the decision)
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
                     "status": "added after the primary result; not part of the decision"})

    for scen in SCENARIOS:
        F = load_fit(scen)
        ms = F["models"]
        dec = l_models(scen)[1]
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
    atomic_csv(OUT / f"L_sensitivity_posthoc_{DATE}.csv", pd.DataFrame(rows))
    done_marker("sensitivity", {"rows": len(rows)})


# ------------------------------------------------------------------ stage verify (disclosure control)
def stage_verify():
    require_committed()
    require_done("analyse", "sensitivity")
    F = {}
    bad = []
    ids = load_ids()
    idset = set(map(str, ids))
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != f"L_disclosure_check_{DATE}.json")
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
        if rec["rows"] >= N and p.name != f"L_km_curves_{DATE}.csv" and p.name != f"L_metrics_{DATE}.csv" and p.name != f"L_contrasts_{DATE}.csv":
            bad.append(f"{p.name}: as many rows as participants")
    m = pd.read_csv(OUT / f"L_metrics_{DATE}.csv")
    c = pd.read_csv(OUT / f"L_contrasts_{DATE}.csv")
    k = pd.read_csv(OUT / f"L_km_curves_{DATE}.csv")
    pre = pd.read_csv(OUT / f"L_preparation_{DATE}.csv")
    summary = {}
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
    # KM subtraction rules on the released rows
    km_total = {s: load_fit(s)["km_total"] for s in KM_SCENARIOS}     # (e, model, group): whole-follow-up totals
    ke = k[k.endpoint != "mean_over_endpoints"]
    for (s, e_, mo, gr), g in ke.groupby(["scenario", "endpoint", "model", "group"]):
        g = g.sort_values("year")
        cum = g.cum_events.dropna().to_numpy()
        if len(cum) and (cum[0] < MIN_CELL or (np.diff(cum) < MIN_CELL).any()):
            bad.append(f"km: released rows {MIN_CELL} events apart failed: {s} {e_} {mo} {gr}")
        tot = km_total[s][ENDPOINTS.index(e_), [OURS, grimage_ref(s)].index(mo), GROUPS.index(gr)]
        if len(cum) and 0 < tot - cum[-1] < MIN_CELL:
            bad.append(f"km: remainder after the last released row: {s} {e_} {mo} {gr}")
    hid = ke.groupby(["scenario", "endpoint", "model", "year"]).suppressed.sum()
    if (hid == 1).any():
        bad.append("km: a year with exactly one hidden group")
    mk = k[k.endpoint == "mean_over_endpoints"].set_index(["scenario", "model", "group", "year"])
    for key, n in ke.groupby(["scenario", "model", "group", "year"]).suppressed.sum().items():
        if n > 0 and pd.notna(mk.loc[key, "survival"]):
            bad.append(f"km mean released with a suppressed input: {key}")
    # released abs-risk cells of the KM models equal the 10-year KM rows
    for s in KM_SCENARIOS:
        for mo in (OURS, grimage_ref(s)):
            for mt, gr in (("abs_risk_10y_top", "top_decile"), ("abs_risk_10y_bottom", "bottom_decile")):
                a = m[(m.scenario == s) & (m.model == mo) & (m.metric == mt) & (m.endpoint != "mean_over_endpoints")].set_index("endpoint").value
                b = 1 - ke[(ke.scenario == s) & (ke.model == mo) & (ke.group == gr) & (ke.year == ANCHOR)].set_index("endpoint").survival
                if not (a.notna() == b.reindex(a.index).notna()).all():
                    bad.append(f"abs risk and KM 10-year row differ in release status: {s} {mo} {mt}")
    summary.update({"metrics rows": len(m), "metrics rows suppressed": int(m.suppressed.sum()), "contrast rows": len(c),
                    "contrast rows suppressed": int(c.suppressed.sum()), "km rows": len(k), "km rows suppressed": int(k.suppressed.sum()),
                    "preparation rows": len(pre)})
    F = {"files": scan, "summary": summary, "problems": bad, "passed": not bad,
         "method": "every token of every public file compared with the 9,866 development participant IDs; regex for the GS barcode "
                   "pattern; headers scanned for identifier columns; released counts at least 5; suppressed rows blank; means blank when "
                   "an input is suppressed; KM rows at least 5 events apart and from the total; no KM year with exactly one hidden group. "
                   "Tables are per scenario, model, endpoint, fold or year; none has one row per participant."}
    write_json(OUT / f"L_disclosure_check_{DATE}.json", F)
    log(f"disclosure check over {len(files)} files: {'passed' if not bad else 'FAILED'}; {len(bad)} problems")
    if bad:
        for b in bad[:30]:
            print(b)
        sys.exit("disclosure check failed; do not commit")
    write_readme(json.loads((OUT / f"L_results_{DATE}.json").read_text()))
    done_marker("verify", {"files": len(files)})


# ------------------------------------------------------------------ README
def write_readme(res):
    def rows(fn):
        return "-" if fn.endswith(".json") else str(len(pd.read_csv(OUT / fn)))

    def cols(fn):
        return "-" if fn.endswith(".json") else ", ".join(pd.read_csv(OUT / fn, nrows=1).columns)

    desc = {
        f"L_rule_outcome_{DATE}.csv": "One row per scenario: H of MethylGPT with interval, best decision comparator of the scenario with H, relative gain, difference with 95% interval, bootstrap P, rule, outcome. The decision is the hm27 row.",
        f"L_summary_{DATE}.csv": "Per scenario and model: H with interval, exp(H), difference from MethylGPT with interval and bootstrap P, relative gain, rule outcome against this model, best-comparator and C* flags, endpoints and folds not estimable.",
        f"L_metrics_{DATE}.csv": "Per scenario, endpoint (8 plus mean_over_endpoints), model and metric: value, interval, hazard ratio where the metric is a log hazard ratio, cell counts, suppression flag and reason. Metrics: " + ", ".join(METRICS) + ".",
        f"L_contrasts_{DATE}.csv": "Per scenario, endpoint (8 plus mean_over_endpoints) and metric: MethylGPT against C* (contrast C_star), the scenario's best on H (scenario_best), GrimAge2 with kNN fill (grimage2_knn; grimage2_mean in full) and the endpoint's own best decision comparator (endpoint_best); values, difference, interval, bootstrap P, relative gain, suppression flag.",
        f"L_km_curves_{DATE}.csv": "Kaplan-Meier aggregates on the yearly grid 0 to 17 for the top decile, middle 80% and bottom decile: survival with interval, numbers at risk, cumulative events, for MethylGPT combined and GrimAge2 with kNN fill (grimage2_mean in full), every endpoint plus the mean over endpoints, hm27 and full.",
        f"L_preparation_{DATE}.csv": "Per scenario, endpoint, model and outer fold: fold size, SD of the stored risk, SD of its age and sex residual, their ratio, estimable flag, decile sizes, ties across a decile boundary. age_sex is the positive control of the constant-risk test.",
        f"L_checks_{DATE}.csv": "Every pre-specified check, with reference and recomputed values, tolerance and pass.",
        f"L_sensitivity_posthoc_{DATE}.csv": "Added after the primary result; not part of the decision. fold_weighted: beta x (estimable folds / 5), counting a constant-risk fold as carrying no signal, best comparator reselected. leave_one_endpoint_out: H over the other 7 endpoints, comparator fixed at the scenario's best on all 8. Rule arithmetic is shown for reference only.",
        f"L_results_{DATE}.json": "Settings, C*, and per scenario H of every model, best comparator, gain, difference, interval and outcome.",
        f"L_disclosure_check_{DATE}.json": "Scan of every public file for participant identifiers and verification of the disclosure rules, run before commit.",
    }
    pr = res["scenarios"][PRIMARY]
    gain = "not reported (H of the best comparator <= 0)" if pr["H"][pr["best"]] <= 0 else f"{pr['relative_gain']:+.1%}"
    lines = ["# Large-gain analysis L: GS risk stratification from J's stored held-out risks", "",
             "Script: `large_gain_L_gs_risk_stratification_2026-10-30.py` (pre-specified design in its docstring). No disease model is fitted and no risk is recomputed. "
             f"hm27 and random_90 are {J.LABEL}; full is complete profiles. 9,866 GS development participants, 8 endpoints, models and scenarios as in J.", "",
             "## Primary result (hm27)", "",
             f"- H = mean over the 8 endpoints of the log hazard ratio per SD of the prepared risk (age and sex residual, standardized within fold; Cox with age and sex, Breslow ties, stratified by outer fold).",
             f"- MethylGPT combined H = {pr['H'][OURS]:.4f}; best decision comparator {pr['best']} H = {pr['H'][pr['best']]:.4f}.",
             f"- Relative gain {gain}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             "", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUT / fn).exists():
            lines.append(f"| `{fn}` | {rows(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              "- Stored held-out risks of analysis J: `artifacts/large_gain_2026-10-28_private/J/private_risks_{mgpt,clocks,knn}_<scenario>.parquet`, `cohort/private_cohortknn_<scenario>__<endpoint>.parquet`.",
              "- Endpoints, follow-up, age, sex and locked outer folds: `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet`.",
              "- Reference C-indices: `artifacts/large_gain_2026-10-28/J/J_cindex_per_endpoint_2026-10-28.csv`.",
              "", "## Notes", "",
              "- Interval: one participant resample per draw shared by every model, endpoint and scenario, 2,000 draws, seed 20261030; prepared risks and decile membership are fixed in the draws; C* and each endpoint's best comparator are fixed at the point estimates.",
              f"- Disclosure control: cells with fewer than {MIN_CELL} events or fewer than {MIN_CELL} people at risk are blank, as are cells that subtraction would reveal (see the docstring): a mean over endpoints is blank if any of its 8 cells is suppressed, a contrast is blank if either cell is, and a KM year with one hidden group hides a second one. Survival and absolute risks are rounded to 4 decimals.",
              "- Absolute risks are Kaplan-Meier estimates with other causes censored, not competing-risk estimates.",
              "- `L_sensitivity_posthoc` was added after the primary result and is not part of the decision. All 8 endpoints stay in H and in every table.",
              "- Constant risk: a fold where the stored risk is a linear function of age and sex is not estimable for that model and endpoint (`L_preparation`). Its stratum informs age and sex only, so that model's estimates at that endpoint come from its other folds, and event capture and the Kaplan-Meier groups cover fewer participants (`n_folds_estimable`). An endpoint with no estimable fold has blank metrics; for H only it counts as beta = 0, and `L_summary` gives H over the estimable endpoints beside it.",
              "- Unsuppressed aggregates, prepared risks, bootstrap weights and per-draw estimates stay in `artifacts/large_gain_2026-10-30_private/L/` on Chronos and are never committed."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "checks", "boot", "analyse", "sensitivity", "verify"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.chdir(HERE)
    LOGS.mkdir(parents=True, exist_ok=True)
    {"prep": stage_prep, "checks": lambda: stage_checks(a.workers), "boot": lambda: stage_boot(a.workers),
     "analyse": stage_analyse, "sensitivity": stage_sensitivity, "verify": stage_verify}[a.stage]()


if __name__ == "__main__":
    main()
