#!/usr/bin/env python3
"""Large-gain analysis M: Generation Scotland (GS) incident disease when the deployed input is measured with noise, MethylGPT
against analysis J's comparators. SIMULATED MEASUREMENT NOISE ON EMPIRICAL PROFILES: every input is a measured GS
development profile to which noise is added by the shared generator large_gain_noise_profiles_2026-11-02.py (commit
729f99e); no profile, outcome or model is simulated. The complete-profile reference uses the measured profiles unchanged.

Written on 2 Nov 2026 for the large-gain analysis plan of 23 Oct 2026 (Decision rule) and the analysis brief for M,
BEFORE any result of this analysis is computed. No disease model is refitted: J's fold models are applied unchanged. Every
result is reported, whichever model it favours.

WHY, AND WHEN THIS SETTING WAS CHOSEN: in analysis J, MethylGPT's advantage grew as the deployed input lost CpGs (complete
profiles +7.5% over the best comparator, 27K-array CpGs +66%). Analysis K showed that the C-index gap does not widen on
individual diseases (hm27 +4.1%, FAIL), and in analysis L one endpoint carried most of the hm27 gain. Measurement noise is
the other way a deployed input degrades, and low-depth bisulfite or enzymatic sequencing is the cheapest route to
cohort-scale methylation. This setting was chosen after the results of J, K and L were known; the results file repeats this.

DESIGN: J's design unchanged except the input
  Participants, endpoints and folds: the 9,866 GS development participants, J's 8 endpoints and J's locked outer folds
  (J.load_preds; artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet, age_adjustment
  "linear"). Validation and test participants are not read.
  Fold models: J's, fitted on complete profiles (mgpt2_embed_linear, mt_bio_linear, factor_residual and raw_cpg_gpu GPU
  models, v12 raw-CpG models) and applied to the held-out participants' noisy profiles by J's functions (locked_risk,
  gpu_risk, fi_locked_risk). Clock Cox models: J's stage_cox code (unpenalised Breslow Cox, n_iter 200, on standardised age,
  sex and clock score(s), fitted per endpoint and outer fold on the fitting participants' stored full-profile scores),
  applied to the held-out participants' scores from the noisy clock matrix. No disease model is refitted.
  Input of scenario s: the vocabulary matrix (9,866 x 49,156, frame-cache order) and the clock matrix (9,866 x 1,195, J's
  clock-matrix order) of private_noisy_<s>.npz. Observed value = the noisy value. A CpG with no read (NaN in the noisy
  matrix) is missing and is filled exactly as J fills a missing CpG: fold mean (raw models), corpus kNN (G's held-out
  reference, J's knn_predict_lowmem call, k = 10, context = the participant's noisy observed CpGs), in-cohort kNN (J's
  secondary: reference = complete profiles of the training folds of the participant's outer fold for that endpoint), and for
  the clocks J's development-set mean (np.nanmean of the measured clock matrix) with corpus-kNN or in-cohort-kNN values at the
  vocabulary clock CpGs that GS measures and that have no read. CpGs absent from GS stay missing everywhere, as in J.
  Gaussian scenarios have no read-level missingness, so their kNN and in-cohort fills change nothing (recorded).
  Label of every noise scenario: "simulated measurement noise on empirical profiles".

SCENARIOS (generator, seeds and summaries in large_gain_noise_profiles_2026-11-02.py and its noise_summary table)
  depth_10   PRIMARY. Read sampling, reads n ~ Poisson(10), methylated reads m ~ Binomial(n, beta), value m / n.
  depth_1, depth_2, depth_5, depth_30   secondary, the same generator at mean depth 1, 2, 5, 30.
  gauss_0.02, gauss_0.05, gauss_0.10    secondary, clip(beta + sigma e, 0, 1).
  full       complete measured profiles: reference, not a decision. Must reproduce J's stored full-profile values (CHECKS 2).

MODELS (J's model set and roles)
  MethylGPT combined (mgpt2_combined): J's stage_mgpt code path unchanged (features from the noisy observed CpGs, batch 32,
    TF32; locked fold models; equal-weight sum of the within-fold standardised embedding, biomarker and factor_residual
    risks). Components embedding, biomarkers, factor_residual and ensemble_embed_bio are tabulated per endpoint, as J.
  Decision comparators (J's DECISION set): raw_cpg_v12_{mean,knn}, raw_cpg_gpu_{mean,knn}, grimage2_{mean,knn},
    dunedinpace_{mean,knn}, grimage2_dunedinpace_{mean,knn}.
  Secondary comparator (outside the decision): in-cohort kNN, suffix _cohortknn, for both raw models and the three clocks.
  MethylGPT applications (not comparators): raw_cpg_v12_factor_imputed, raw_cpg_gpu_factor_imputed (J's: noisy observed
    values kept, missing CpGs = mu + V s clipped to [0, 1]).
  Covariate model: age_sex = the stored age+sex Cox risk (baseline_risk).
  ADDED, all pre-specified here, all reported, all outside the decision:
  (a) kNN DENOISING comparators, suffix _knndenoise: every GS-measured vocabulary CpG (read or not) is replaced by the
      corpus-kNN prediction from the noisy observed context; CpGs absent from GS take the fold mean. Applied to both raw
      models and to the vocabulary clock CpGs of the three clocks; non-vocabulary clock CpGs keep their noisy values (J's
      development-set mean when they have no read). One neighbour search per participant serves the fill and the denoising,
      because the neighbours depend only on the context; the kNN fill takes the predictions at the no-read CpGs (CHECKS 8
      compares this with J's knn_fill call on the first 512 participants).
  (b) MethylGPT applications raw_cpg_v12_factor_denoised, raw_cpg_gpu_factor_denoised: every vocabulary CpG replaced by the
      factor reconstruction mu + V s (s from the noisy observed CpGs; J's factor-imputation core and loop, clipped to [0,
      1]), then J's raw fold models.
  (c) mgpt2_combined_depthaware (components embedding_depthaware, biomarkers_depthaware, factor_residual_depthaware,
      ensemble_embed_bio_depthaware): the factor-model posterior weights each observed CpG by w = sigma2 / (sigma2 + v),
      sigma2 = the factor model's own residual variance (PPCACore.sigma2 of the core in use: the aging checkpoint's core for
      the embedding and residuals, the multitask checkpoint's core for the biomarkers), v = p (1 - p) / n with
      p = (m + 0.5) / (n + 1) for depth scenarios (n reads, m = rint(value n) methylated reads) and v = sigma_noise^2 for
      Gaussian scenarios; w = 0 where there is no read. Code reading, done before this docstring was written:
      PPCACore.posterior(x, w) (methylgpt_lib.py) forms xc = where(w > 0, x - mu, 0) * w, b = xc V / sigma2 and
      A = (V * w)^T V / sigma2 + diag(1 / lam), so w enters only as a multiplicative weight per observation, and these are the
      posterior equations for observation noise variance sigma2 / w. In J's feature code every component uses the posterior:
      embed_batch calls core.posterior(x, ctx.float()) and the encoder inputs and the embedding use s; the residuals are
      x - mu - V s. The depth-aware path replaces only the posterior's w argument (ctx.float() -> w); the attention mask, the
      token values and every fold model are unchanged. w is computed as 1 / (1 + v / sigma2) in float32 (the same value;
      sigma2 / (sigma2 + v) evaluates to 1 - 2^-24 at v = 0 in torch), so v = 0 at observed CpGs gives w = 1 exactly and
      the path must reproduce J's features bitwise: stage wcheck compares both paths on the first 256 participants of
      full and depth_10 before any depth-aware feature is used, and the full scenario is computed by both paths (CHECKS 3).
      If the identity does not hold, mgpt2_combined_depthaware is recorded as "not run" with the reason, and nothing else
      is tried.

PRIMARY METRIC AND DECISION (J's, plan Decision rule for discrimination)
  C(model, e) = pooled within-fold C-index of endpoint e (concordant + tied/2 over comparable pairs summed over e's outer
  folds; J's fold_counts, tied_tol 1e-8). M(model) = mean over the 8 endpoints of (C(model, e) - C(age_sex, e)).
  C* = the decision comparator with the largest M in that scenario (complete profiles: J's mean-fill models only).
  PASS iff (M_MethylGPT - M_C*) / M_C* >= 0.10 and the lower end of the 95% interval of M_MethylGPT - M_C* is > 0; otherwise
  FAIL; M_C* <= 0 is "not applicable". M_MethylGPT = M(mgpt2_combined). The decision is the depth_10 row; every other
  scenario is a secondary row with its own C*.
INTERVALS: J's pre-specified shared participant bootstrap, 2,000 draws, seed 20261102: I = default_rng(20261102).integers(0,
  9866, size=(2000, 9866)), w_b = bincount(I[b], minlength=9866) in frame-cache order, one resample per draw shared by every
  model, endpoint and scenario; per endpoint the pooled within-fold C-index is recomputed with pair weights w_i w_j.
  Percentile 95% intervals; bootstrap P = min(1, 2 (k + 1) / (B + 1)), k = min(#{d_b <= 0}, #{d_b >= 0}). Sensitivity: J's
  within-fold resample (per endpoint k in sorted order and outer fold, default_rng([20261102, 1, k]).integers(0, n_f, size=
  (2000, n_f))). Risks are not refitted in the draws.

SECONDARY (all pre-specified and reported; none changes the decision)
  - Every comparator, every scenario: M with interval, difference from mgpt2_combined with both intervals, P, relative gain,
    rule arithmetic against that comparator (J's summary).
  - Per endpoint: C-index of every model, gain above age+sex, mgpt2_combined minus model with interval and P,
    Benjamini-Hochberg q across the 8 endpoints within scenario and model; per endpoint contrast with the scenario's C*
    (difference, interval, q, contribution to the M difference = difference / 8) and with the endpoint's own best decision
    comparator (chosen by point estimate and kept fixed in the draws).
  - Direction counts per scenario and contrast (C*, own best, every comparator): endpoints by sign, by interval and by q.
  - Leave-one-endpoint-out M for every scenario and endpoint (pre-specified now because in L one endpoint carried most of
    the gain): M over the other 7 endpoints with C* re-chosen on those 7, and with C* fixed at the 8-endpoint C*; same draws;
    rule arithmetic for reference only.
  - Event-weighted mean (weights = incident events per endpoint, C* re-chosen), rule arithmetic for reference only.
  - Fraction of each model's complete-profile M retained, M_scenario / M_full, descriptive only. M_full of a model = the same
    model on the complete profiles; for _knn and _cohortknn models nothing is filled on complete profiles, so their M_full is
    that of the _mean model with the same base; _knndenoise, _factor_denoised and the depth-aware models are computed on
    the complete profiles (depth-aware with w = 1). Undefined when M_full <= 0.
  - Added contrasts, outside the decision, rule arithmetic for reference only: mgpt2_combined vs C*_ext (best of the
    decision set and the five _knndenoise comparators); mgpt2_combined_depthaware vs C*, vs C*_ext and vs mgpt2_combined;
    raw_cpg_{v12,gpu}_factor_denoised vs C*; each <base>_knndenoise vs <base>_knn (complete profiles: vs <base>_mean).

CHECKS (written to M_checks before any new number is used; stage boot refuses to run unless all pass; abort on failure)
  (1) Noise inputs: done_noise.json lists the 8 scenarios; noise_checks all pass; the noise script is committed and equals
      its blob at 729f99e; the SHA-256 of every array (vocab, clock, reads_vocab, reads_clock) equals noise_summary (verified
      again at every load); noisy vocabulary clock columns equal the noisy clock matrix there; CpGs absent from GS are NaN;
      depth scenarios: observed iff reads > 0, and value n is an integer m (max |rint(value n) - value n| <= 1e-3).
  (2) Full scenario reproduces J exactly: every risk column of J's private_risks_mgpt_full and private_risks_clocks_full
      equals the recomputed one (maximum absolute difference 0); C-index of every J model and endpoint and J's M equal J's
      stored values within 1e-12 (CSV round trip); the full-profile clock scores equal J's private scores exactly and the
      stored scores within 1e-6 (J's tolerance); age_sex C-index equals J's within 1e-12.
  (3) Depth-aware identity: stage wcheck (first 256 participants of full and depth_10) and the full scenario (all
      participants): with w = 1 the depth-aware features equal J's path bitwise, and the full depth-aware risks equal the
      standard risks exactly. Weights: w in [0, 1], w = 0 exactly where nothing is observed, w = 1 on complete profiles.
  (4) Fold models: J recorded no fold-model hashes, so M records the SHA-256 of all 200 fold-model files it reads (5 families
      x 8 endpoints x 5 folds) at stage inputs (public table M_fold_models); at stage checks they are recomputed and must be
      equal, every file must predate J's first stage (J's masks done-marker time), and CHECKS 2 shows that they reproduce
      J's stored full-profile risks exactly.
  (5) Leakage: the fit_ids of every embedding, biomarker and v12 fold model equal the training folds' participants and
      exclude the held-out fold; in-cohort references equal the training folds (v12 fit_ids) and exclude the held-out fold;
      clock Cox fits use the complement of the held-out fold; the GPU fold models store no fit rows (recorded; covered by
      CHECKS 2); no GS participant identifier is in the corpus kNN reference.
  (6) Corpus kNN reference: 145,316 rows, row SHA-256 equal to J's (J_knn_record) and G's; checked before any query.
  (7) age_sex: identical in every scenario (maximum absolute difference 0 between scenarios), equal to the stored
      baseline_risk; endpoint table SHA-256 recorded and the file predates J's first stage.
  (8) kNN integrity, per scenario: fills leave observed values unchanged, fill every no-read GS-measured CpG and leave CpGs
      absent from GS missing; denoised profiles are finite at every GS-measured CpG; single search = J's knn_fill on the
      first 512 participants (maximum absolute difference 0, where those participants have a no-read CpG); clock fills sit
      exactly at the no-read vocabulary clock CpGs; in-cohort fills complete.
  (9) Pair counting equals sksurv's pooled C-index for every model of full, and gs_stacked_2026-10-04.PairCounter's
      weighted counts on one endpoint-fold for 5 draws (J's check 10).
  (10) Bootstrap weights: every shared draw sums to 9,866 (SHA-256 recorded; stage boot recomputes it and must match).
  (11) Privacy (stage privacy, appended after the outputs, as J and K): no public file holds a participant identifier, an
      identifier column or an event cell below 5.

STAGES (resumable; atomic writes; done-markers artifacts/logs/large_gain_M_<stage>_2026-11-02.done)
  inputs (CPU) -> wcheck (GPU) -> mgpt (GPU, sharded by scenario) | knn (GPU) | cohort (GPU, sharded by endpoint) ->
  clockin (CPU) -> clocks (biolearn interpreter /home/kying/anaconda3/envs/integrative/bin/python) -> cox (CPU) -> checks
  (CPU) -> boot (CPU, sharded by scenario) -> analyse (CPU) -> privacy (CPU). In-cohort kNN queries only the held-out
  participants that have a CpG to fill (the others keep their noisy profile, which is what the fill returns for them).
  Every stage refuses to run unless this script, J's script and the noise script are committed unchanged. Execution-only
  changes after the commit go in a dated AMENDMENT paragraph below, committed before their output is used. Run through
  run_large_gain_M_2026-11-02.sh inside the transient user unit large-gain-M-2026-11-02.
OUTPUTS
  Aggregate tables: artifacts/large_gain_2026-11-02/M/ (README.md; M_inputs, M_fold_models, M_checks, M_knn_record,
  M_depthaware_record, M_cindex_per_endpoint, M_endpoint_contrasts, M_direction_counts, M_summary, M_rule_outcome (one row
  per scenario, decision_row = depth_10), M_event_weighted, M_leave_one_endpoint_out, M_retention, M_added_contrasts,
  M_bootstrap_draws, M_results). Participant-level fills, clock inputs and scores, risks and bootstrap arrays:
  artifacts/large_gain_2026-11-02_private/M/ (git-ignored, never committed, stay on Chronos). Tables only: no figures, no
  manuscript edits.
Run: python3 large_gain_M_gs_measurement_noise_2026-11-02.py --stage <stage> [--device cuda:N] [--shard i --nshards n]
     (stage clocks: /home/kying/anaconda3/envs/integrative/bin/python ... --stage clocks [--workers 6])
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
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-11-02"
OUTM = ART / "large_gain_2026-11-02/M"
PRIV = ART / "large_gain_2026-11-02_private/M"
LOGS = ART / "logs"
NOISE_OUT = ART / "large_gain_2026-11-02/noise"
NOISE_PRIV = ART / "large_gain_2026-11-02_private/noise"
NOISE_SCRIPT = HERE / "large_gain_noise_profiles_2026-11-02.py"
NOISE_COMMIT = "729f99e"
J_SCRIPT = HERE / "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


J = load_module("large_gain_J", J_SCRIPT)   # no torch or sksurv at import, so the biolearn interpreter can load it too
log, need, write_json = J.log, J.need, J.write_json
atomic_npy, atomic_parquet, atomic_csv, sha256_bytes = J.atomic_npy, J.atomic_parquet, J.atomic_csv, J.sha256_bytes
ENDPOINTS, CLOCK_MODELS, RAW, COMPONENTS, DECISION, COHORT, FACTOR, OURS = (
    J.ENDPOINTS, J.CLOCK_MODELS, J.RAW, J.COMPONENTS, J.DECISION, J.COHORT, J.FACTOR, J.OURS)

SEED = 20261102
N_BOOT = 2000
SMALL_CELL = 5
WCHECK_N = 256
NOISE = ["depth_10", "depth_1", "depth_2", "depth_5", "depth_30", "gauss_0.02", "gauss_0.05", "gauss_0.10"]
KIND = {s: ("depth", float(s.split("_")[1])) if s.startswith("depth") else ("gauss", float(s.split("_")[1])) for s in NOISE}
ALL_SCEN = ["full"] + NOISE
PRIMARY = "depth_10"
LABEL = "simulated measurement noise on empirical profiles"
KNNDEN = [f"{r}_knndenoise" for r in RAW] + [f"{c}_knndenoise" for c in CLOCK_MODELS]
FDEN = [f"{r}_factor_denoised" for r in RAW]
DA = "mgpt2_combined_depthaware"
DA_COMPONENTS = [f"{c}_depthaware" for c in COMPONENTS]
DA_ALL = [DA] + DA_COMPONENTS
COMBOS_DA = {f"{k}_depthaware": [f"{p}_depthaware" for p in v] for k, v in J.COMBOS.items()}
FULL_DECISION = [f"{r}_mean" for r in RAW] + [f"{c}_mean" for c in CLOCK_MODELS]
J_FULL_MODELS = J.models_for("full")
BASES = RAW + list(CLOCK_MODELS)
RULE = ("PASS iff (M_MethylGPT - M_best) / M_best >= 0.10 and the lower end of the 95% interval of M_MethylGPT - M_best "
        "is > 0; M = mean over the 8 endpoints of (pooled within-fold C-index - age+sex C-index); best = decision "
        "comparator with the largest M in the scenario")


# ------------------------------------------------------------------ gates and paths
def done_path(stage):
    return LOGS / f"large_gain_M_{stage}_{DATE}.done"


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
    for p in (Path(__file__).resolve(), J_SCRIPT, NOISE_SCRIPT):
        if not committed(p):
            sys.exit(f"Cannot proceed - {p.name} must be committed unchanged before any result-producing stage runs")


def shard_of(items, shard, nshards):
    return [x for i, x in enumerate(items) if i % nshards == shard]


def da_status():
    p = PRIV / "private_wcheck.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def da_ok():
    s = da_status()
    return bool(s and s["identity_holds"])


def models_for(scen, with_da=None):
    with_da = da_ok() if with_da is None else with_da
    da = DA_ALL if with_da else []
    if scen == "full":
        return J_FULL_MODELS + KNNDEN + FDEN + da
    return ["age_sex", OURS] + COMPONENTS + DECISION + COHORT + FACTOR + KNNDEN + FDEN + da


def decision_for(scen):
    return FULL_DECISION if scen == "full" else DECISION


def role(m):
    if m in KNNDEN:
        return "added comparator (corpus-kNN denoising), outside the decision"
    if m in FDEN:
        return "MethylGPT application (factor denoising), outside the decision"
    if m == DA:
        return "MethylGPT combined with the depth-aware posterior, outside the decision"
    if m in DA_COMPONENTS:
        return "MethylGPT component (depth-aware posterior)"
    return J.role(m)


def label(scen):
    return "complete measured profiles (reference, not a decision)" if scen == "full" else LABEL


# ------------------------------------------------------------------ noisy inputs
def noise_table():
    return pd.read_csv(need(NOISE_OUT / f"noise_summary_{DATE}.csv")).set_index("scenario")


def array_keys(scen):
    return ["vocab", "clock"] + (["reads_vocab", "reads_clock"] if KIND[scen][0] == "depth" else [])


def load_noisy(scen, keys=("vocab",)):
    """Arrays of private_noisy_<scen>.npz; every SHA-256 is checked against noise_summary before use (abort on mismatch)."""
    if not (NOISE_PRIV / "done_noise.json").exists():
        sys.exit(f"Cannot proceed - required data file not found: {NOISE_PRIV / 'done_noise.json'}")
    ref = noise_table()
    out = {}
    with np.load(need(NOISE_PRIV / f"private_noisy_{scen}.npz"), allow_pickle=False) as z:
        for k in keys:
            a = z[k]
            h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            if h != ref.loc[scen, f"sha256_{k}"]:
                sys.exit(f"{scen}: SHA-256 of {k} differs from noise_summary; stopped before use")
            out[k] = a
    return out


def scenario_vocab(scen, X):
    return X if scen == "full" else load_noisy(scen, ("vocab",))["vocab"]


def noise_variance(scen, Xs):
    """v per vocabulary value (float32): 0 at observed values of complete profiles; inf where nothing is observed.
    Computed in row blocks of 1,024 (memory only)."""
    kind, par = KIND.get(scen, ("full", 0.0))
    n_all = load_noisy(scen, ("reads_vocab",))["reads_vocab"] if kind == "depth" else None
    out = np.empty(Xs.shape, np.float32)
    for s in range(0, len(Xs), 1024):
        e = slice(s, min(s + 1024, len(Xs)))
        x = Xs[e]
        obs = np.isfinite(x)
        if kind == "full":
            v = np.where(obs, 0.0, np.inf)
        elif kind == "gauss":
            v = np.where(obs, par ** 2, np.inf)
        else:
            n = n_all[e].astype(np.float64)
            m = np.rint(np.where(obs, x, 0.0) * n)
            p = (m + 0.5) / (n + 1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(obs, p * (1 - p) / n, np.inf)
        out[e] = v
    return out


def weight_np(sigma2, Vn):
    """w = sigma2 / (sigma2 + v), evaluated as 1 / (1 + v / sigma2) in float32 (as on the GPU)."""
    return (np.float32(1.0) / (np.float32(1.0) + Vn / np.float32(sigma2))).astype(np.float32)


def fold_model_files():
    rows = []
    for ep in ENDPOINTS:
        for f in range(1, 6):
            d = [d for d in J.RAW_DIRS if (d / f"{ep}_linear_outer{f}_model.npz").exists()]
            paths = {"embedding": J.MGPT2_RUN / f"{ep}_linear_outer{f}_model.npz", "biomarkers": J.BIO_RUN / f"{ep}_linear_outer{f}_model.npz",
                     "factor_residual": J.FR_DIR / "models" / f"factor_residual_{ep}_outer{f}.npz",
                     "raw_cpg_gpu": J.FR_DIR / "models" / f"raw_cpg_gpu_{ep}_outer{f}.npz",
                     "raw_cpg_v12": (d[0] if d else J.RAW_DIRS[0]) / f"{ep}_linear_outer{f}_model.npz"}
            for fam, p in paths.items():
                rows.append({"family": fam, "endpoint": ep, "outer_fold": f, "path": str(need(p))})
    return rows


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def j_start_time():
    t = json.loads(need(J.done_path("masks")).read_text())["time"]
    return time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------------ stage inputs
def stage_inputs():
    require_committed()
    t0 = time.time()
    rows = []
    done = json.loads(need(NOISE_PRIV / "done_noise.json").read_text())
    nck = pd.read_csv(need(NOISE_OUT / f"noise_checks_{DATE}.csv"))
    blob = subprocess.run(["git", "-C", str(HERE), "show", f"{NOISE_COMMIT}:./{NOISE_SCRIPT.name}"], capture_output=True).stdout
    same_blob = blob == NOISE_SCRIPT.read_bytes()

    def add(scenario, item, stored, recomputed, passed, note=""):
        rows.append({"check": "1 noise inputs", "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "pass": bool(passed), "note": note})

    add("all", "done_noise.json scenarios", " ".join(NOISE), " ".join(done.get("scenarios", [])), sorted(done.get("scenarios", [])) == sorted(NOISE))
    add("all", "noise_checks all pass", len(nck), int(nck["pass"].astype(bool).sum()), bool(nck["pass"].astype(bool).all()))
    add("all", f"noise script committed and equal to its blob at {NOISE_COMMIT}", "", "", committed(NOISE_SCRIPT) and same_blob)
    ids, X = J.load_gs()
    gsobs = np.isfinite(X)
    cids, cpgs, Mc = J.load_clock_matrix()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    cfin = np.isfinite(Mc)
    ref = noise_table()
    summ = []
    for scen in NOISE:
        keys = array_keys(scen)
        with np.load(need(NOISE_PRIV / f"private_noisy_{scen}.npz"), allow_pickle=False) as z:
            arr = {k: z[k] for k in keys}
        for k in keys:
            h = hashlib.sha256(np.ascontiguousarray(arr[k]).tobytes()).hexdigest()
            add(scen, f"SHA-256 {k}", ref.loc[scen, f"sha256_{k}"], h, h == ref.loc[scen, f"sha256_{k}"])
        V, C = arr["vocab"], arr["clock"]
        obs, cobs = np.isfinite(V), np.isfinite(C)
        add(scen, "shapes", "(9866, 49156) (9866, 1195)", f"{V.shape} {C.shape}", V.shape == X.shape and C.shape == Mc.shape)
        add(scen, "CpGs absent from GS are NaN", 0, int((obs & ~gsobs).sum() + (cobs & ~cfin).sum()), not (obs & ~gsobs).any() and not (cobs & ~cfin).any())
        add(scen, "noisy vocabulary clock columns equal the noisy clock matrix", 0, "",
            np.array_equal(np.nan_to_num(V[:, vj], nan=-1), np.nan_to_num(C[:, ic], nan=-1)))
        info = {"scenario": scen, "label": LABEL, "kind": KIND[scen][0], "parameter": KIND[scen][1], "role": "primary" if scen == PRIMARY else "secondary",
                "vocab_observed_per_participant_mean": float(obs.sum(1).mean()), "vocab_observed_per_participant_min": int(obs.sum(1).min()),
                "vocab_observed_per_participant_max": int(obs.sum(1).max()),
                "gs_measured_vocab_values_with_no_read": int((gsobs & ~obs).sum()),
                "fraction_gs_measured_vocab_values_with_no_read": float((gsobs & ~obs).sum() / gsobs.sum()),
                "participants_with_a_no_read_vocab_cpg": int((gsobs & ~obs).any(1).sum()),
                "clock_vocab_observed_per_participant_mean": float(cobs[:, ic].sum(1).mean()),
                "clock_nonvocab_observed_per_participant_mean": float(cobs[:, ~invoc].sum(1).mean()),
                "rmse_noisy_vs_measured": float(ref.loc[scen, "rmse_noisy_vs_measured"]),
                "mean_within_participant_pearson_r": float(ref.loc[scen, "mean_within_participant_pearson_r"])}
        if KIND[scen][0] == "depth":
            n = arr["reads_vocab"]
            add(scen, "observed iff reads > 0 (vocabulary and clock)", 0, int((obs != ((n > 0) & gsobs)).sum()),
                np.array_equal(obs, (n > 0) & gsobs) and np.array_equal(cobs, (arr["reads_clock"] > 0) & cfin))
            vn = np.where(obs, V, 0).astype(np.float64) * n
            dev = float(np.abs(vn - np.rint(vn)).max())
            add(scen, "value x reads is an integer", 0, dev, dev <= 1e-3, "max |rint(value n) - value n|")
            info["mean_reads_gs_measured_vocab"] = float(n[gsobs].mean())
        summ.append(info)
        del arr, V, C, obs, cobs
        log(f"inputs: {scen} verified")
    # fold-model files and the endpoint table
    t_j = j_start_time()
    fm = []
    for r in fold_model_files():
        st = os.stat(r["path"])
        fm.append({**r, "bytes": st.st_size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                   "predates_J_first_stage": st.st_mtime < t_j, "sha256": file_sha(r["path"])})
    st = os.stat(J.PREDS)
    fm.append({"family": "endpoint_table", "endpoint": "all", "outer_fold": 0, "path": str(J.PREDS), "bytes": st.st_size,
               "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)), "predates_J_first_stage": st.st_mtime < t_j,
               "sha256": file_sha(J.PREDS)})
    OUTM.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUTM / f"M_fold_models_{DATE}.csv", pd.DataFrame(fm))
    atomic_csv(OUTM / f"M_inputs_{DATE}.csv", pd.DataFrame(summ))
    PRIV.mkdir(parents=True, exist_ok=True)
    atomic_csv(PRIV / "private_inputs_checks.csv", pd.DataFrame(rows))
    nf = sum(not r["pass"] for r in rows)
    log(f"inputs: {len(rows)} checks, {nf} failed; {len(fm)} files hashed")
    if nf:
        print(pd.DataFrame(rows).query("~`pass`").to_string())
        sys.exit("noise input checks failed; stopped before any use")
    done_marker("inputs", {"checks": len(rows), "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ MethylGPT feature code (J's stage_mgpt, plus the depth-aware path)
class Mgpt:
    def __init__(self, dev):
        import torch
        sys.path.insert(0, str(HERE))
        from methylgpt_model import BioHeads, load_checkpoint
        torch.backends.cuda.matmul.allow_tf32 = True
        self.torch, self.dev = torch, dev
        ag = torch.load(need(J.AGING), map_location=dev, weights_only=False)
        self.model, self.core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
        self.model.load_state_dict(ag["model"]); self.model.eval()
        mt = torch.load(need(J.MULTITASK), map_location=dev, weights_only=False)
        self.mt_model, self.mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
        self.mt_model.load_state_dict(mt["model"]); self.mt_model.eval()
        self.mt_heads = BioHeads(self.mt_core.rank + self.mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
        self.mt_heads.load_state_dict(mt["heads"]); self.mt_heads.eval()
        _, self.fi_core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)   # scenario_gs_raw_factor_imputed's core
        self.record = {"aging_ckpt": str(ag["ckpt"]), "multitask_ckpt": str(mt["ckpt"]), "sigma2_aging_core": self.core.sigma2,
                       "sigma2_multitask_core": self.mt_core.sigma2, "rank": int(self.core.rank)}

    def features(self, Xs):   # J.stage_mgpt.features, unchanged
        from methylgpt_model import embed_batch
        torch, dev = self.torch, self.dev
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                obs = torch.isfinite(x)
                E.append(embed_batch(self.model, self.core, x, obs).float().cpu().numpy())
                B.append(self.mt_heads(embed_batch(self.mt_model, self.mt_core, x, obs))[3].float().cpu().numpy())
                s = self.core.posterior(x, obs.float())
                R.append(torch.where(obs, x - self.core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        return np.concatenate(E), np.concatenate(B), np.concatenate(R)

    def _embed_w(self, model, core, x, ctx, w):   # methylgpt_model.embed_batch with the posterior's weight argument replaced
        torch = self.torch
        s = core.posterior(x, w)
        with torch.autocast("cuda", enabled=False):
            xs, c, e = model.inputs(x, ctx, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.encode(xs, c, e, ctx, s)
        return model.embedding(z, s)

    def features_depthaware(self, Xs, Vn):
        """Same as features with w = sigma2 / (sigma2 + v) passed to every posterior (each core's own sigma2)."""
        torch, dev = self.torch, self.dev
        E, B, R = [], [], []
        with torch.no_grad():
            for i in range(0, len(Xs), 32):
                x = torch.from_numpy(Xs[i:i + 32]).to(dev)
                v = torch.from_numpy(Vn[i:i + 32]).to(dev)
                obs = torch.isfinite(x)
                wa = torch.reciprocal(1.0 + v / self.core.sigma2)        # = sigma2 / (sigma2 + v); exactly 1 at v = 0
                wm = torch.reciprocal(1.0 + v / self.mt_core.sigma2)
                E.append(self._embed_w(self.model, self.core, x, obs, wa).float().cpu().numpy())
                B.append(self.mt_heads(self._embed_w(self.mt_model, self.mt_core, x, obs, wm))[3].float().cpu().numpy())
                s = self.core.posterior(x, wa)
                R.append(torch.where(obs, x - self.core.predict(s), torch.full_like(x, float("nan"))).cpu().numpy())
        return np.concatenate(E), np.concatenate(B), np.concatenate(R)

    def factor_impute_and_reconstruct(self, Xs):   # J.stage_mgpt.factor_impute, also returning the reconstruction
        torch, dev = self.torch, self.dev
        Xf = Xs.copy()
        Xr = np.empty_like(Xs)
        with torch.no_grad():
            for i in range(0, len(Xf), 64):
                x = torch.from_numpy(Xf[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                rec = self.fi_core.predict(self.fi_core.posterior(x, obs.float())).clamp(0, 1)
                Xf[i:i + 64] = torch.where(obs, x, rec).float().cpu().numpy()
                Xr[i:i + 64] = rec.float().cpu().numpy()
        return Xf, Xr


def stage_wcheck(device):
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    require_committed()
    require_done("inputs")
    cuda_preinit()
    dev = J.pick_device(device)
    ids, X = J.load_gs()
    mg = Mgpt(dev)
    rec = {**mg.record, "code_reading": ("PPCACore.posterior: xc = where(w > 0, x - mu, 0) * w; b = xc V / sigma2; A = (V * w)^T V / sigma2 + "
                                         "diag(1/lam): w enters only as a multiplicative weight per observation (noise variance sigma2 / w)."),
           "participants": WCHECK_N, "tests": {}}
    ok = True
    for scen in ("full", PRIMARY):
        Xs = np.ascontiguousarray(scenario_vocab(scen, X)[:WCHECK_N]).astype(np.float32)
        obs = np.isfinite(Xs)
        Vn = np.where(obs, 0.0, np.inf).astype(np.float32)      # w = 1 at every observed CpG
        a = mg.features(Xs)
        b = mg.features_depthaware(Xs, Vn)
        d = {k: float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0 for k, x, y in zip(("embedding", "biomarkers", "residuals"), a, b)}
        nan_same = all(np.array_equal(np.isfinite(x), np.isfinite(y)) for x, y in zip(a, b))
        bitwise = all(np.array_equal(np.nan_to_num(x, nan=7.0), np.nan_to_num(y, nan=7.0)) for x, y in zip(a, b)) and nan_same
        rec["tests"][scen] = {"max_abs_difference": d, "bitwise_equal": bitwise}
        ok &= bitwise
        log(f"wcheck {scen}: w = 1 depth-aware features bitwise equal to J's path: {bitwise} {d}")
    rec["identity_holds"] = bool(ok)
    rec["status"] = "run" if ok else "not run: with w = 1 the depth-aware path does not reproduce J's features bitwise"
    PRIV.mkdir(parents=True, exist_ok=True)
    write_json(PRIV / "private_wcheck.json", rec)
    write_json(OUTM / f"M_depthaware_record_{DATE}.json", {"wcheck": rec, "scenarios": {}})
    done_marker("wcheck", {"identity_holds": bool(ok)})


def stage_mgpt(device, shard, nshards):
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    require_committed()
    require_done("inputs", "wcheck")
    cuda_preinit()
    t0 = time.time()
    dev = J.pick_device(device)
    ids, X = J.load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    preds = J.load_preds()
    mg = Mgpt(dev)
    with_da = da_ok()
    bio_stored = pd.read_parquet(need(J.BIO_RUN / "private_nested_predictions.parquet"))
    bio_stored = bio_stored[bio_stored.age_adjustment == "linear"]
    bio_stored = bio_stored.assign(dnam_id=bio_stored.dnam_id.astype(str))
    fr_stored = {n: pd.read_parquet(need(J.FR_DIR / n / "private_nested_predictions.parquet")) for n in ("factor_residual", "raw_cpg_gpu")}
    (PRIV / "mgpt").mkdir(parents=True, exist_ok=True)
    for scen in shard_of(ALL_SCEN, shard, nshards):
        fo = PRIV / "mgpt" / f"private_risks_mgpt_{scen}.parquet"
        if fo.exists():
            log(f"{scen}: MethylGPT risks exist, skipped")
            continue
        t1 = time.time()
        Xs = np.ascontiguousarray(scenario_vocab(scen, X)).astype(np.float32)
        E, B, R = mg.features(Xs)
        FR = np.hstack([E, B, R]).astype(np.float32)
        del R
        Xf, Xr = mg.factor_impute_and_reconstruct(Xs)
        if not (np.isfinite(Xf).all() and np.isfinite(Xr).all()):
            sys.exit(f"{scen}: factor imputation left missing values")
        wrec = {}
        if with_da:
            Vn = noise_variance(scen, Xs)
            w = weight_np(mg.core.sigma2, Vn)
            o = np.isfinite(Xs)
            wrec = {"w_mean_observed": float(w[o].mean()), "w_min_observed": float(w[o].min()), "w_max_observed": float(w[o].max()),
                    "w_zero_where_unobserved": bool((w[~o] == 0).all()), "w_one_everywhere_observed": bool((w[o] == 1).all()),
                    "w_in_unit_interval": bool(((w >= 0) & (w <= 1)).all())}
            del w, o
            Ew, Bw, Rw = mg.features_depthaware(Xs, Vn)
            FRw = np.hstack([Ew, Bw, Rw]).astype(np.float32)
            del Rw, Vn
        frames = []
        for ep, pe in preds.items():
            idx = np.array([pos[s] for s in pe.dnam_id])
            cov = pe[["age", "sex"]].to_numpy(float)
            folds = sorted(pe.outer_fold.unique())
            held_sets = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in folds]
            keys = ["embedding", "biomarkers", "factor_residual", "raw_cpg_v12_mean", "raw_cpg_gpu_mean", "raw_cpg_v12_factor_imputed",
                    "raw_cpg_gpu_factor_imputed", "raw_cpg_v12_factor_denoised", "raw_cpg_gpu_factor_denoised"]
            if with_da:
                keys += ["embedding_depthaware", "biomarkers_depthaware", "factor_residual_depthaware"]
            risk = {k: np.empty(len(pe)) for k in keys}
            for f, held in zip(folds, held_sets):
                mm = dict(np.load(need(J.MGPT2_RUN / f"{ep}_linear_outer{f}_model.npz"), allow_pickle=False))
                bm = dict(np.load(need(J.BIO_RUN / f"{ep}_linear_outer{f}_model.npz"), allow_pickle=False))
                rm = J.raw_v12_model(ep, f)
                risk["embedding"][held] = J.locked_risk(mm, E[idx[held]].astype(np.float64), cov[held])
                risk["biomarkers"][held] = J.locked_risk(bm, B[idx[held]].astype(np.float64), cov[held])
                risk["raw_cpg_v12_mean"][held] = J.locked_risk(rm, Xs[idx[held]].astype(np.float64), cov[held])
                gfr = J.gpu_model("factor_residual", ep, f)
                risk["factor_residual"][held] = J.gpu_risk(gfr, FR[idx[held]], cov[held].astype(np.float32))
                gm = J.gpu_model("raw_cpg_gpu", ep, f)
                risk["raw_cpg_gpu_mean"][held] = J.gpu_risk(gm, Xs[idx[held]], cov[held].astype(np.float32))
                for tag, Xa in (("factor_imputed", Xf), ("factor_denoised", Xr)):     # scenario_gs_raw_factor_imputed application
                    zf = (Xa[idx[held]] - gm["mean"]) / gm["scale"]
                    risk[f"raw_cpg_gpu_{tag}"][held] = np.hstack(
                        [((cov[held] - gm["cov_mean"]) / gm["cov_scale"]).astype(np.float32), zf.astype(np.float32)]) @ gm["coefficient"]
                    risk[f"raw_cpg_v12_{tag}"][held] = J.fi_locked_risk(rm, Xa[idx[held]].astype(np.float64), cov[held])
                if with_da:
                    risk["embedding_depthaware"][held] = J.locked_risk(mm, Ew[idx[held]].astype(np.float64), cov[held])
                    risk["biomarkers_depthaware"][held] = J.locked_risk(bm, Bw[idx[held]].astype(np.float64), cov[held])
                    risk["factor_residual_depthaware"][held] = J.gpu_risk(gfr, FRw[idx[held]], cov[held].astype(np.float32))
            if scen == "full":   # J's (scenario_gs_combined's) own check against the stored out-of-fold risks
                checks = {"embedding": pe.risk.to_numpy(float),
                          "biomarkers": bio_stored[bio_stored.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)}
                for name, col in (("factor_residual", "factor_residual"), ("raw_cpg_gpu", "raw_cpg_gpu_mean")):
                    s_ = fr_stored[name]
                    s_ = s_.assign(dnam_id=s_.dnam_id.astype(str))
                    checks[col] = s_[s_.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)
                for name, ref in checks.items():
                    if np.max(np.abs(ref - risk[name])) > 1e-3 * (1 + np.abs(ref).max()):
                        sys.exit(f"recomputed {name} risks differ from the stored nested predictions for {ep}")
            combos = dict(J.COMBOS, **(COMBOS_DA if with_da else {}))
            for cname, parts in combos.items():
                v = np.zeros(len(pe))
                for held in held_sets:
                    for k in parts:
                        x_ = risk[k][held]; v[held] += (x_ - x_.mean()) / x_.std()
                risk[cname] = v
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            for k, v in risk.items():
                fr_[k] = v
            frames.append(fr_)
        atomic_parquet(fo, pd.concat(frames, ignore_index=True))
        write_json(PRIV / "mgpt" / f"private_record_{scen}.json", {"seconds": round(time.time() - t1, 1), "depthaware": with_da, "weights": wrec})
        log(f"{scen}: MethylGPT, depth-aware, raw mean-fill, factor-imputed and factor-denoised risks written ({time.time() - t1:.0f} s)")
        del Xs, E, B, FR, Xf, Xr
        if with_da:
            del Ew, Bw, FRw
    done_marker(f"mgpt_shard{shard}of{nshards}", {"scenarios": shard_of(ALL_SCEN, shard, nshards), "seconds": round(time.time() - t0)})
    if all(done_path(f"mgpt_shard{i}of{nshards}").exists() for i in range(nshards)):
        done_marker("mgpt", {"nshards": nshards})


# ------------------------------------------------------------------ stage knn (corpus reference: fill and denoising)
def stage_knn(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    require_committed()
    require_done("inputs")
    cuda_preinit()
    t0 = time.time()
    dev = J.pick_device(device)
    ids, X = J.load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    gsobs = np.isfinite(X)
    preds = J.load_preds()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    _, cpgs, Mc = J.load_clock_matrix()
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    G = load_module("large_gain_G", J.G_SCRIPT)
    F = G.load_f()
    rd = load_module("recon_diag", J.RECON)
    mu = np.load(need(J.CORE_MU))["mu"].astype(np.float32)
    jrec = json.loads(need(J.OUTJ / f"J_knn_record_{J.DATE}.json").read_text())
    gref = json.loads(need(J.G_REF_JSON).read_text())
    free, total = torch.cuda.mem_get_info(dev)
    log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
    R, rrows = G.knn_reference(F, "heldout", dev)
    sha = sha256_bytes(rrows.astype(np.int64).tobytes())
    ref_ok = sha == jrec["reference_rows_sha256"] == gref["rows_sha256"] and len(rrows) == gref["rows"] == 145316
    record = {"reference_rows": int(len(rrows)), "reference_rows_sha256": sha, "J_rows_sha256": jrec["reference_rows_sha256"],
              "G_rows_sha256": gref["rows_sha256"], "identical_to_J_and_G": bool(ref_ok), "mu": str(J.CORE_MU.relative_to(HERE)),
              "k": J.KNN_K, "query_batch": J.KNN_BATCH, "reference_chunk": J.KNN_CHUNK, "scenarios": {}}
    OUTM.mkdir(parents=True, exist_ok=True)
    write_json(OUTM / f"M_knn_record_{DATE}.json", record)
    if not ref_ok:
        sys.exit("kNN reference differs from J's and G's held-out reference; stopped before any query")
    mu_h = torch.as_tensor(mu, device=dev).half()
    (PRIV / "knn").mkdir(parents=True, exist_ok=True)
    gm_t = torch.from_numpy(gsobs).to(dev)
    for scen in ALL_SCEN:
        fo = PRIV / "knn" / f"private_risks_knn_{scen}.parquet"
        if fo.exists():
            record["scenarios"][scen] = json.loads((PRIV / "knn" / f"private_record_{scen}.json").read_text())
            log(f"{scen}: corpus kNN exists, skipped")
            continue
        t1 = time.time()
        Xs = np.ascontiguousarray(scenario_vocab(scen, X)).astype(np.float32)
        ctx = np.isfinite(Xs)
        nr = gsobs & ~ctx                                    # GS-measured CpGs with no read
        Xt, ct = torch.from_numpy(Xs).to(dev), torch.from_numpy(ctx).to(dev)
        pr = rd.knn_predict_lowmem(R, mu_h, Xt, ct, gm_t, k=J.KNN_K, batch=J.KNN_BATCH, chunk=J.KNN_CHUNK)
        del Xt, ct
        Xd = np.full(Xs.shape, np.nan, np.float32)           # denoised: kNN prediction at every GS-measured CpG
        for i, p in enumerate(pr):
            Xd[i, gsobs[i]] = p
        del pr
        Xk = Xs.copy()                                       # fill: kNN prediction at the no-read CpGs only
        Xk[nr] = Xd[nr]
        ok_kept = bool(np.array_equal(Xk[ctx], Xs[ctx]))
        ok_fill = bool(np.isfinite(Xk[gsobs]).all() and not np.isfinite(Xk[~gsobs]).any())
        ok_den = bool(np.isfinite(Xd[gsobs]).all() and not np.isfinite(Xd[~gsobs]).any())
        sub = nr[:J.KNN_BATCH]
        if sub.any():                                        # J's knn_fill call on the first query batch
            Xj = J.knn_fill(rd, R, mu_h, Xs[:J.KNN_BATCH], ctx[:J.KNN_BATCH], sub, dev)
            jdiff = float(np.abs(Xj[sub] - Xk[:J.KNN_BATCH][sub]).max())
            j_note = f"{int(sub.sum())} no-read values in the first {J.KNN_BATCH} participants"
            del Xj
        else:
            jdiff, j_note = None, f"no no-read value in the first {J.KNN_BATCH} participants (not applicable)"
        if not (ok_kept and ok_fill and ok_den) or (jdiff is not None and jdiff != 0):
            sys.exit(f"{scen}: kNN check failed (kept {ok_kept}, filled {ok_fill}, denoised {ok_den}, J difference {jdiff})")
        cm = np.isfinite(Mc[:, ic]) & nr[:, vj]
        fill = np.where(cm, Xk[:, vj], np.nan).astype(np.float32)
        den = np.where(np.isfinite(Mc[:, ic]), Xd[:, vj], np.nan).astype(np.float32)
        atomic_npy(PRIV / "knn" / f"private_clockfill_{scen}.npy", fill)
        atomic_npy(PRIV / "knn" / f"private_clockdenoise_{scen}.npy", den)
        frames = []
        for ep, pe in preds.items():
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            if scen != "full":
                fr_["raw_cpg_v12_knn"], fr_["raw_cpg_gpu_knn"] = J.raw_risks_filled(Xk, preds, pos, ep)
            fr_["raw_cpg_v12_knndenoise"], fr_["raw_cpg_gpu_knndenoise"] = J.raw_risks_filled(Xd, preds, pos, ep)
            frames.append(fr_)
        rec = {"no_read_values_filled": int(nr.sum()), "kept_values_unchanged": ok_kept, "all_no_read_filled_absent_missing": ok_fill,
               "denoised_values": int(gsobs.sum()), "denoised_complete": ok_den, "J_knn_fill_max_abs_difference_first_batch": jdiff,
               "J_knn_fill_note": j_note, "clock_fill_values": int(cm.sum()), "clock_denoised_values": int(np.isfinite(den).sum()),
               "sha256_filled": sha256_bytes(Xk.tobytes()), "sha256_denoised": sha256_bytes(Xd.tobytes()), "seconds": round(time.time() - t1, 1)}
        write_json(PRIV / "knn" / f"private_record_{scen}.json", rec)
        atomic_parquet(fo, pd.concat(frames, ignore_index=True))
        record["scenarios"][scen] = {k: v for k, v in rec.items() if not k.startswith("sha256")}
        write_json(OUTM / f"M_knn_record_{DATE}.json", record)
        log(f"{scen}: corpus kNN fill and denoising, raw kNN risks written ({time.time() - t1:.0f} s)")
        del Xs, Xk, Xd, ctx, nr
    write_json(OUTM / f"M_knn_record_{DATE}.json", record)
    done_marker("knn", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage cohort (in-cohort kNN, J's secondary)
def stage_cohort(device, shard, nshards):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    require_committed()
    require_done("inputs")
    cuda_preinit()
    t0 = time.time()
    dev = J.pick_device(device)
    ids, X = J.load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    gsobs = np.isfinite(X)
    preds = J.load_preds()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    _, cpgs, Mc = J.load_clock_matrix()
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    rd = load_module("recon_diag", J.RECON)
    mu_h = torch.as_tensor(np.load(need(J.CORE_MU))["mu"].astype(np.float32), device=dev).half()
    eps = shard_of(ENDPOINTS, shard, nshards)
    d = PRIV / "cohort"
    d.mkdir(parents=True, exist_ok=True)
    for scen in NOISE:
        outs = {ep: d / f"private_cohortknn_{scen}__{ep}.parquet" for ep in eps}
        if all(o.exists() and o.with_name(o.stem + "_clockfill.npy").exists() for o in outs.values()):
            log(f"{scen}: in-cohort kNN exists, skipped")
            continue
        Xn = np.ascontiguousarray(scenario_vocab(scen, X)).astype(np.float32)
        for ep in eps:
            if outs[ep].exists() and outs[ep].with_name(outs[ep].stem + "_clockfill.npy").exists():
                continue
            t1 = time.time()
            pe = preds[ep]
            idx = np.array([pos[s] for s in pe.dnam_id])
            fold = pe.outer_fold.to_numpy()
            cov_all = pe[["age", "sex"]].to_numpy(float)
            r_v12, r_gpu = np.full(len(pe), np.nan), np.full(len(pe), np.nan)
            fill = np.full((len(X), len(ic)), np.nan, np.float32)
            refcheck = []
            for f in sorted(np.unique(fold)):
                held, fit = idx[fold == f], idx[fold != f]
                hrows = np.flatnonzero(fold == f)
                rm, gm = J.raw_v12_model(ep, f), J.gpu_model("raw_cpg_gpu", ep, f)
                Xs = Xn[held]
                obs = np.isfinite(Xs)
                masked = gsobs[held] & ~obs
                q = np.flatnonzero(masked.any(1))
                Xk = Xs.copy()
                if len(q):
                    Rh = torch.from_numpy(X[fit]).to(dev).half()
                    Xk[q] = J.knn_fill(rd, Rh, mu_h, np.ascontiguousarray(Xs[q]), np.ascontiguousarray(obs[q]), np.ascontiguousarray(masked[q]), dev)
                    del Rh
                    torch.cuda.empty_cache()
                if not (np.array_equal(Xk[obs], Xs[obs]) and np.isfinite(Xk[gsobs[held]]).all()):
                    sys.exit(f"{scen} {ep} fold {f}: in-cohort kNN fill check failed")
                refcheck.append({"scenario": scen, "endpoint": ep, "outer_fold": int(f), "reference_rows": int(len(fit)),
                                 "equals_v12_fit_ids": set(ids[fit]) == set(rm["fit_ids"].astype(str)),
                                 "held_out_in_reference": int(len(np.intersect1d(held, fit))), "participants_queried": int(len(q)),
                                 "values_filled": int(masked.sum())})
                r_v12[hrows] = J.locked_risk(rm, Xk.astype(np.float64), cov_all[hrows])
                r_gpu[hrows] = J.gpu_risk(gm, Xk, cov_all[hrows].astype(np.float32))
                cm = np.isfinite(Mc[held][:, ic]) & masked[:, vj]
                fill[held] = np.where(cm, Xk[:, vj], np.nan)
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            fr_["raw_cpg_v12_cohortknn"], fr_["raw_cpg_gpu_cohortknn"] = r_v12, r_gpu
            if not (np.isfinite(r_v12).all() and np.isfinite(r_gpu).all()):
                sys.exit(f"{scen} {ep}: in-cohort risks incomplete")
            atomic_csv(d / f"private_reference_check_{scen}__{ep}.csv", pd.DataFrame(refcheck))
            atomic_npy(outs[ep].with_name(outs[ep].stem + "_clockfill.npy"), fill)
            atomic_parquet(outs[ep], fr_)
            log(f"{scen} {ep}: in-cohort kNN written ({time.time() - t1:.0f} s, {sum(r['participants_queried'] for r in refcheck)} participants queried)")
        del Xn
    done_marker(f"cohort_shard{shard}of{nshards}", {"endpoints": eps, "seconds": round(time.time() - t0)})
    if all(done_path(f"cohort_shard{i}of{nshards}").exists() for i in range(nshards)):
        done_marker("cohort", {"nshards": nshards})


# ------------------------------------------------------------------ clock stages
def clock_variants():
    v = ["full__mean", "full__knndenoise"]
    for s in NOISE:
        v += [f"{s}__mean", f"{s}__knn", f"{s}__knndenoise"] + [f"{s}__cohortknn__{e}" for e in ENDPOINTS]
    return v


def stage_clockin():
    require_committed()
    require_done("inputs", "knn", "cohort")
    t0 = time.time()
    cids, cpgs, Mc = J.load_clock_matrix()
    vocab = np.load(need(J.MASKS))["vocab"].astype(str)
    invoc, ic, vj = J.clock_vocab_index(cpgs, vocab)
    means = np.nanmean(Mc, 0)
    d = PRIV / "clockin"
    d.mkdir(parents=True, exist_ok=True)
    atomic_csv(d / "private_ids.csv", pd.DataFrame({"sample_id": cids}))
    atomic_csv(d / "private_cpgs.csv", pd.DataFrame({"cpg": cpgs}))
    cache = {}
    for v in clock_variants():
        fo = d / f"private_clockin_{v}.npy"
        if fo.exists():
            continue
        parts = v.split("__")
        scen, kind = parts[0], parts[1]
        if scen not in cache:
            cache.clear()
            cache[scen] = Mc if scen == "full" else load_noisy(scen, ("clock",))["clock"]
        C = cache[scen]
        Z = np.where(np.isfinite(C), C, means[None, :]).astype(np.float32)    # J.stage_clockin.filled: missing -> development mean
        nrc = np.isfinite(Mc[:, ic]) & ~np.isfinite(C[:, ic])
        fill = None
        if kind == "knn":
            fill = np.load(need(PRIV / "knn" / f"private_clockfill_{scen}.npy"))
            if not np.array_equal(np.isfinite(fill), nrc):
                sys.exit(f"{v}: clock fills do not sit at the no-read vocabulary clock CpGs")
        elif kind == "cohortknn":
            fill = np.load(need(PRIV / "cohort" / f"private_cohortknn_{scen}__{parts[2]}_clockfill.npy"))
            if not np.array_equal(np.isfinite(fill), nrc):
                sys.exit(f"{v}: clock fills do not sit at the no-read vocabulary clock CpGs")
        elif kind == "knndenoise":
            fill = np.load(need(PRIV / "knn" / f"private_clockdenoise_{scen}.npy"))
            if not np.array_equal(np.isfinite(fill), np.isfinite(Mc[:, ic])):
                sys.exit(f"{v}: denoised clock values do not cover the GS-measured vocabulary clock CpGs")
        if fill is not None:
            sub = Z[:, ic]
            ok = np.isfinite(fill)
            sub[ok] = fill[ok]
            Z[:, ic] = sub
        atomic_npy(fo, Z)
    done_marker("clockin", {"variants": len(clock_variants()), "seconds": round(time.time() - t0)})


def score_variant(v):
    """J.score_variant with M's directories."""
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
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    require_committed()
    require_done("clockin")
    import biolearn
    t0 = time.time()
    (PRIV / "clock_scores").mkdir(parents=True, exist_ok=True)
    todo = []
    for v in clock_variants():
        try:
            ok = len(pd.read_csv(PRIV / "clock_scores" / f"private_scores_{v}.csv")) == 9866
        except Exception:
            ok = False
        if not ok:
            todo.append(v)
    log(f"biolearn {getattr(biolearn, '__version__', '?')}: {len(todo)} clock variants to score with {workers} workers")
    with Pool(workers) as pool:
        for v in pool.imap_unordered(score_variant, todo):
            log(f"scored {v}")
    done_marker("clocks", {"variants": len(clock_variants()), "biolearn": getattr(biolearn, "__version__", "?"), "seconds": round(time.time() - t0)})


def stage_cox():
    """J.stage_cox: fitted per endpoint and outer fold on the fitting participants' stored full-profile scores."""
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    require_committed()
    require_done("clocks")
    t0 = time.time()
    preds = J.load_preds()
    sc = pd.read_csv(need(J.CLOCK_SCORES))
    sc["sample_id"] = sc.sample_id.astype(str)
    full = sc[sc.scenario == "full"].set_index("sample_id")
    fits, fitrec = {}, []
    for ep, pe in preds.items():
        ids = pe.dnam_id.to_numpy()
        ev, du = pe.event.astype(bool).to_numpy(), pe.duration.to_numpy(float)
        cov = pe[["age", "sex"]].to_numpy(float)
        folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
        for name, cols in CLOCK_MODELS.items():
            Mf = np.column_stack([cov] + [full.loc[ids, c].to_numpy(float) for c in cols])
            for k, held in enumerate(folds):
                fit = np.setdiff1d(np.arange(len(pe)), held)
                m, s = Mf[fit].mean(0), Mf[fit].std(0)
                cx = CoxPHSurvivalAnalysis(ties="breslow", n_iter=200).fit((Mf[fit] - m) / s, Surv.from_arrays(ev[fit], du[fit]))
                fits[(ep, k, name)] = (m, s, cx.coef_.copy())
                fitrec.append({"endpoint": ep, "outer_fold_index": k, "model": name, "fit_rows": int(len(fit)),
                               "held_out_in_fit": int(len(np.intersect1d(fit, held))), "fit_plus_held": int(len(fit) + len(held))})
    scores = {}

    def get_scores(v):
        if v not in scores:
            t = pd.read_csv(need(PRIV / "clock_scores" / f"private_scores_{v}.csv"))
            t["sample_id"] = t.sample_id.astype(str)
            scores.clear()
            scores[v] = t.set_index("sample_id")
        return scores[v]

    (PRIV / "clocks").mkdir(parents=True, exist_ok=True)
    for scen in ALL_SCEN:
        fills = ["mean", "knndenoise"] if scen == "full" else ["mean", "knn", "knndenoise", "cohortknn"]
        frames = []
        for ep, pe in preds.items():
            ids = pe.dnam_id.to_numpy()
            cov = pe[["age", "sex"]].to_numpy(float)
            folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            for fl in fills:
                S = get_scores(f"{scen}__cohortknn__{ep}" if fl == "cohortknn" else f"{scen}__{fl}")
                for name, cols in CLOCK_MODELS.items():
                    Ms = np.column_stack([cov] + [S.loc[ids, c].to_numpy(float) for c in cols])
                    r = np.empty(len(pe))
                    for k, held in enumerate(folds):
                        m, s, coef = fits[(ep, k, name)]
                        r[held] = ((Ms[held] - m) / s) @ coef
                    fr_[f"{name}_{fl}"] = r
            frames.append(fr_)
        atomic_parquet(PRIV / "clocks" / f"private_risks_clocks_{scen}.parquet", pd.concat(frames, ignore_index=True))
        log(f"{scen}: clock Cox risks written")
    atomic_csv(PRIV / "clocks" / "private_clock_fit_rows.csv", pd.DataFrame(fitrec))
    done_marker("cox", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ assembly
def assemble(scen, preds):
    base = pd.concat([pe[["dnam_id", "endpoint", "outer_fold", "event", "duration", "baseline_risk"]] for pe in preds.values()],
                     ignore_index=True).rename(columns={"baseline_risk": "age_sex"})
    keys = ["dnam_id", "endpoint", "outer_fold"]
    parts = [PRIV / "mgpt" / f"private_risks_mgpt_{scen}.parquet", PRIV / "clocks" / f"private_risks_clocks_{scen}.parquet",
             PRIV / "knn" / f"private_risks_knn_{scen}.parquet"]
    tabs = [pd.read_parquet(need(p)) for p in parts]
    if scen != "full":
        tabs.append(pd.concat([pd.read_parquet(need(PRIV / "cohort" / f"private_cohortknn_{scen}__{ep}.parquet")) for ep in ENDPOINTS], ignore_index=True))
    for t in tabs:
        t["dnam_id"] = t.dnam_id.astype(str)
        base = base.merge(t, on=keys, how="left", validate="one_to_one")
    miss = [m for m in models_for(scen) if m not in base.columns or not np.isfinite(base[m].to_numpy(float)).all()]
    if miss:
        sys.exit(f"{scen}: risks missing or not finite for {miss}")
    return base


def pooled_c(g, cols):
    """J's point C-index per model (fold_counts pooled over the endpoint's outer folds), g = one endpoint's rows."""
    ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
    R = np.stack([g[m].to_numpy(float) for m in cols])
    con = np.zeros(len(R)); tied = np.zeros(len(R)); tot = 0.0
    for f in sorted(g.outer_fold.unique()):
        s = np.flatnonzero(g.outer_fold.to_numpy() == f)
        (t_, arr), = J.fold_counts(ev[s], du[s], R[:, s], [None])
        con += arr[:, 0]; tied += arr[:, 1]; tot += t_
    return J.cidx_from(con, tied, tot)


def shared_weights():
    I = np.random.default_rng(SEED).integers(0, 9866, size=(N_BOOT, 9866))
    Wp = np.stack([np.bincount(I[b], minlength=9866) for b in range(N_BOOT)]).astype(np.float64)
    return Wp


# ------------------------------------------------------------------ stage checks
def stage_checks():
    require_committed()
    require_done("inputs", "wcheck", "mgpt", "knn", "cohort", "cox")
    t0 = time.time()
    preds = J.load_preds()
    ids, X = J.load_gs()
    rows = []

    def add(check, scenario, item, stored, recomputed, tol, passed, note=""):
        diff = abs(float(stored) - float(recomputed)) if isinstance(stored, (int, float, np.integer, np.floating)) and \
            isinstance(recomputed, (int, float, np.integer, np.floating)) and not isinstance(stored, bool) else ""
        rows.append({"check": check, "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "abs_diff": diff, "tolerance": tol, "pass": bool(passed), "note": note})

    # (1) noise inputs
    for r in pd.read_csv(need(PRIV / "private_inputs_checks.csv"), keep_default_na=False).to_dict("records"):
        add("1 noise inputs", r["scenario"], r["item"], r["stored"], r["recomputed"], 0, str(r["pass"]) == "True", r["note"])
    # (2) full reproduces J exactly
    A = {s: assemble(s, preds) for s in ALL_SCEN}
    keys = ["dnam_id", "endpoint", "outer_fold"]
    for fn in ("private_risks_mgpt_full.parquet", "private_risks_clocks_full.parquet"):
        jt = pd.read_parquet(need(J.PRIV / fn)).assign(dnam_id=lambda d: d.dnam_id.astype(str))
        mine = A["full"].set_index(keys).loc[pd.MultiIndex.from_frame(jt[keys])]
        for c in [c for c in jt.columns if c not in keys]:
            d = float(np.max(np.abs(jt[c].to_numpy(float) - mine[c].to_numpy(float))))
            add("2 full reproduces J's stored risks exactly", "full", c, 0.0, d, 0, d == 0, f"{fn}; max abs difference over {len(jt)} rows")
    jc = pd.read_csv(need(J.OUTJ / f"J_cindex_per_endpoint_{J.DATE}.csv"))
    jc = jc[jc.scenario == "full"].set_index(["endpoint", "model"])
    for ep in ENDPOINTS:
        g = A["full"][A["full"].endpoint == ep]
        cs = pooled_c(g, J_FULL_MODELS)
        for m, c in zip(J_FULL_MODELS, cs):
            ref = float(jc.loc[(ep, m), "c_index"])
            add("2 full reproduces J's stored C-index", "full", f"{ep}:{m}", ref, float(c), 1e-12, abs(ref - c) <= 1e-12)
            if m == "age_sex":
                add("7 age_sex C-index equals J's", "full", ep, float(jc.loc[(ep, m), "c_age_sex"]), float(c), 1e-12,
                    abs(float(jc.loc[(ep, m), "c_age_sex"]) - c) <= 1e-12)
    jr = json.loads(need(J.OUTJ / f"J_results_{J.DATE}.json").read_text())["scenarios"]["full"]["M"]
    cfull = np.stack([pooled_c(A["full"][A["full"].endpoint == ep], J_FULL_MODELS) for ep in ENDPOINTS], 1)
    Mfull = (cfull - cfull[J_FULL_MODELS.index("age_sex")]).mean(1)
    for m, v in zip(J_FULL_MODELS, Mfull):
        if m != "age_sex":
            add("2 full reproduces J's M", "full", m, float(jr[m]), float(v), 1e-12, abs(float(jr[m]) - v) <= 1e-12)
    sc = pd.read_csv(need(J.CLOCK_SCORES)); sc["sample_id"] = sc.sample_id.astype(str)
    mine = pd.read_csv(need(PRIV / "clock_scores" / "private_scores_full__mean.csv")); mine["sample_id"] = mine.sample_id.astype(str)
    jsc = pd.read_csv(need(J.PRIV / "clock_scores" / "private_scores_full__mean.csv")); jsc["sample_id"] = jsc.sample_id.astype(str)
    ref = sc[sc.scenario == "full"].set_index("sample_id").loc[mine.sample_id]
    jsc = jsc.set_index("sample_id").loc[mine.sample_id]
    for c in ("GrimAgeV2", "DunedinPACE"):
        d1 = float(np.max(np.abs(jsc[c].to_numpy(float) - mine[c].to_numpy(float))))
        d2 = float(np.max(np.abs(ref[c].to_numpy(float) - mine[c].to_numpy(float))))
        add("2 full clock scores equal J's private scores exactly", "full", c, 0.0, d1, 0, d1 == 0, "max abs difference")
        add("2 full clock scores equal the stored scores", "full", c, 0.0, d2, 1e-6, d2 <= 1e-6, "max abs difference (J's tolerance)")
    # (3) depth-aware identity and weights
    ws = da_status()
    for scen, t in ws["tests"].items():
        add("3 depth-aware path with w = 1 equals J's features bitwise", scen, f"first {ws['participants']} participants", "", json.dumps(t["max_abs_difference"]), 0,
            t["bitwise_equal"], "stage wcheck")
    if ws["identity_holds"]:
        for m in DA_ALL:
            base = m.replace("_depthaware", "")
            d = float(np.max(np.abs(A["full"][m].to_numpy(float) - A["full"][base].to_numpy(float))))
            add("3 full depth-aware risks equal the standard risks exactly (w = 1)", "full", m, 0.0, d, 0, d == 0)
        for scen in ALL_SCEN:
            w = json.loads(need(PRIV / "mgpt" / f"private_record_{scen}.json").read_text())["weights"]
            ok = w["w_zero_where_unobserved"] and w["w_in_unit_interval"] and (w["w_one_everywhere_observed"] if scen == "full" else True)
            add("3 depth-aware weights", scen, "w in [0, 1], 0 where unobserved, 1 on complete profiles", "", f"mean {w['w_mean_observed']:.4f}", "", ok)
    else:
        add("3 depth-aware identity", "all", "status", "", ws["status"], "", True, "depth-aware models not run (recorded)")
    # (4) fold models
    fm = pd.read_csv(need(OUTM / f"M_fold_models_{DATE}.csv"))
    same = [file_sha(r.path) == r.sha256 for r in fm.itertuples()]
    add("4 fold-model and endpoint-table files unchanged since stage inputs", "all", "files", len(fm), int(sum(same)), 0, all(same) and len(fm) == 201)
    add("4 fold-model and endpoint-table files predate J's first stage", "all", "files", len(fm), int(fm.predates_J_first_stage.sum()), 0,
        bool(fm.predates_J_first_stage.all()), "J recorded no fold-model hashes; CHECKS 2 shows these files reproduce J's stored risks exactly")
    # (5) leakage
    pos = {s: i for i, s in enumerate(ids)}
    for ep, pe in preds.items():
        fold = pe.outer_fold.to_numpy()
        pid = pe.dnam_id.to_numpy().astype(str)
        ok, n = True, 0
        for f in sorted(np.unique(fold)):
            tr, he = set(pid[fold != f]), set(pid[fold == f])
            for p in (J.MGPT2_RUN / f"{ep}_linear_outer{f}_model.npz", J.BIO_RUN / f"{ep}_linear_outer{f}_model.npz"):
                fi = set(np.load(p, allow_pickle=False)["fit_ids"].astype(str))
                ok &= fi == tr and not (fi & he); n += 1
            fi = set(J.raw_v12_model(ep, f)["fit_ids"].astype(str))
            ok &= fi == tr and not (fi & he); n += 1
        add("5 fit_ids of embedding, biomarker and v12 fold models = training folds, held-out fold excluded", "all", ep, n, n, 0, ok)
    fr = pd.read_csv(need(PRIV / "clocks" / "private_clock_fit_rows.csv"))
    add("5 clock Cox fits use the complement of the held-out fold", "all", "endpoint-fold-models", 120, len(fr), 0,
        len(fr) == 120 and int(fr.held_out_in_fit.sum()) == 0 and bool((fr.fit_plus_held == 9866).all()))
    add("5 GPU fold models (factor_residual, raw_cpg_gpu) store no fit rows", "all", "files", 80, 80, 0, True,
        "not checkable from the files; covered by CHECKS 2 (exact reproduction of J's stored risks)")
    rc = pd.concat([pd.read_csv(need(PRIV / "cohort" / f"private_reference_check_{s}__{ep}.csv")) for s in NOISE for ep in ENDPOINTS], ignore_index=True)
    add("5 in-cohort references = training folds (v12 fit_ids), held-out fold excluded", "all", "scenario-endpoint-folds", 8 * 40, len(rc), 0,
        len(rc) == 320 and bool(rc.equals_v12_fit_ids.all()) and int(rc.held_out_in_reference.sum()) == 0)
    G = load_module("large_gain_G", J.G_SCRIPT)
    F = G.load_f()
    prof = pd.read_parquet(need(G.CORPUS_DIR / "profiles.parquet"))
    rrows = prof.index[(prof.split == "train").to_numpy()].to_numpy()
    drop = set(pd.read_csv(need(F.OUTF / f"duplicates_removed_{F.DATE}.csv")).corpus_row.astype(int))
    rrows = np.sort(np.array([r for r in rrows if r not in drop]))
    gs_in_ref = len(set(prof.id.iloc[rrows].astype(str)) & set(ids))
    add("5 no GS participant in the corpus kNN reference", "all", "identifier overlap", 0, gs_in_ref, 0, gs_in_ref == 0)
    # (6) corpus reference
    kr = json.loads(need(OUTM / f"M_knn_record_{DATE}.json").read_text())
    h = sha256_bytes(rrows.astype(np.int64).tobytes())
    add("6 corpus kNN reference rows equal J's and G's", "all", "rows / SHA-256", f"145316 / {kr['J_rows_sha256']}", f"{len(rrows)} / {h}", "",
        kr["identical_to_J_and_G"] and h == kr["J_rows_sha256"] == kr["G_rows_sha256"] and len(rrows) == 145316)
    # (7) age_sex identical in every scenario
    for s in NOISE:
        d = float(np.max(np.abs(A[s].age_sex.to_numpy(float) - A["full"].age_sex.to_numpy(float))))
        same_rows = A[s][keys].equals(A["full"][keys])
        add("7 age_sex rows unchanged", s, "max abs difference from full", 0.0, d, 0, d == 0 and same_rows)
    # (8) kNN integrity
    for s, r in kr["scenarios"].items():
        add("8 corpus kNN fill and denoising integrity", s, "kept unchanged, no-read filled, absent missing, denoised complete", "",
            f"{r['no_read_values_filled']} filled, {r['denoised_values']} denoised", "",
            r["kept_values_unchanged"] and r["all_no_read_filled_absent_missing"] and r["denoised_complete"])
        jd = r["J_knn_fill_max_abs_difference_first_batch"]
        add("8 single search equals J's knn_fill (first 512 participants)", s, "max abs difference", 0.0, jd if jd is not None else "", 0,
            jd is None or jd == 0, r["J_knn_fill_note"])
    add("8 in-cohort fills complete", "all", "scenario-endpoint-folds", 320, len(rc), "", len(rc) == 320,
        f"{int(rc.participants_queried.sum())} participant queries, {int(rc.values_filled.sum())} values filled")
    # (9) pair counting
    import torch
    stk = load_module("stk", J.STK)
    worst = 0.0
    for ep in ENDPOINTS:
        g = A["full"][A["full"].endpoint == ep]
        ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
        folds = [np.flatnonzero(g.outer_fold.to_numpy() == f) for f in sorted(g.outer_fold.unique())]
        cs = pooled_c(g, models_for("full"))
        for k, m in enumerate(models_for("full")):
            worst = max(worst, abs(cs[k] - J.sksurv_pooled(ev, du, g[m].to_numpy(float), folds)))
    add("9 pair counting equals sksurv (point)", "full", "max abs C difference over endpoints and models", 0.0, worst, 1e-12, worst <= 1e-12)
    g = A["full"][A["full"].endpoint == ENDPOINTS[0]]
    s = np.flatnonzero(g.outer_fold.to_numpy() == 1)
    ev, du = g.event.astype(bool).to_numpy()[s], g.duration.to_numpy(float)[s]
    R = np.stack([g[m].to_numpy(float)[s] for m in (OURS, "age_sex")])
    W = np.stack([np.bincount(np.random.default_rng(b).integers(0, len(s), len(s)), minlength=len(s)) for b in range(5)]).astype(np.float64)
    (tot, arr), = J.fold_counts(ev, du, R, [W])
    refc = stk.PairCounter(ev, du, torch.device("cpu")).counts(R, W)
    d = float(max(np.abs(arr[..., 0] - refc[..., 0]).max(), np.abs(arr[..., 1] - refc[..., 2]).max(), np.abs(tot[None, :] - refc.sum(-1)).max()))
    add("9 pair counting equals stk.PairCounter (weighted)", "full", f"{ENDPOINTS[0]} fold 1, 5 draws", 0.0, d, 1e-9, d <= 1e-9)
    # (10) bootstrap weights
    Wp = shared_weights()
    hw = sha256_bytes(Wp.tobytes())
    add("10 bootstrap weights", "all", "every shared draw sums to 9,866", 9866, int(Wp.sum(1).min()), 0, bool((Wp.sum(1) == 9866).all()), f"sha256 {hw}")
    write_json(PRIV / "private_boot_weights.json", {"sha256": hw})
    tab = pd.DataFrame(rows)
    atomic_csv(OUTM / f"M_checks_{DATE}.csv", tab)
    nf = int((~tab["pass"].astype(bool)).sum())
    log(f"checks: {len(tab)} rows, {nf} failed ({time.time() - t0:.0f} s)")
    if nf:
        print(tab[~tab["pass"].astype(bool)].to_string())
        sys.exit("checks failed; no new number is used")
    done_marker("checks", {"rows": len(tab), "failed": 0})


# ------------------------------------------------------------------ stage boot (sharded by scenario)
def stage_boot(shard, nshards):
    require_committed()
    require_done("checks")
    t0 = time.time()
    preds = J.load_preds()
    ids, _ = J.load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    Wp = shared_weights()
    if sha256_bytes(Wp.tobytes()) != json.loads(need(PRIV / "private_boot_weights.json").read_text())["sha256"]:
        sys.exit("bootstrap weights differ from those checked")
    sens = {}
    for k, (ep, pe) in enumerate(preds.items()):
        rng = np.random.default_rng([SEED, 1, k])
        sens[ep] = {}
        for f in sorted(pe.outer_fold.unique()):
            n_f = int((pe.outer_fold == f).sum())
            Jm = rng.integers(0, n_f, size=(N_BOOT, n_f))
            sens[ep][f] = np.stack([np.bincount(Jm[b], minlength=n_f) for b in range(N_BOOT)]).astype(np.float64)
    (PRIV / "boot").mkdir(parents=True, exist_ok=True)
    for scen in shard_of(ALL_SCEN, shard, nshards):
        fo = PRIV / "boot" / f"private_boot_{scen}.npz"
        if fo.exists():
            continue
        A = assemble(scen, preds)
        models = models_for(scen)
        point = np.zeros((len(models), len(ENDPOINTS)))
        bprim = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        bsens = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        for j, ep in enumerate(ENDPOINTS):     # J.stage_boot
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            xr = np.array([pos[s] for s in g.dnam_id])
            R = np.stack([g[m].to_numpy(float) for m in models])
            acc = {"p": [np.zeros(len(models)), np.zeros(len(models)), 0.0],
                   "b": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)],
                   "s": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]}
            for f in sorted(g.outer_fold.unique()):
                s = np.flatnonzero(g.outer_fold.to_numpy() == f)
                res = J.fold_counts(ev[s], du[s], R[:, s], [None, Wp[:, xr[s]], sens[ep][f]])
                for key, (tot, arr) in zip(("p", "b", "s"), res):
                    acc[key][0] += arr[..., 0]; acc[key][1] += arr[..., 1]; acc[key][2] = acc[key][2] + tot
            point[:, j] = J.cidx_from(*acc["p"])
            bprim[:, j] = J.cidx_from(acc["b"][0], acc["b"][1], acc["b"][2][None, :])
            bsens[:, j] = J.cidx_from(acc["s"][0], acc["s"][1], acc["s"][2][None, :])
        tmp = fo.with_name(fo.stem + ".tmp.npz")
        np.savez(tmp, models=np.array(models), point=point, boot=bprim, boot_within_fold=bsens)
        os.replace(tmp, fo)
        log(f"{scen}: point and bootstrap C-indices ({time.time() - t0:.0f} s)")
    done_marker(f"boot_shard{shard}of{nshards}", {"seconds": round(time.time() - t0)})
    if all((PRIV / "boot" / f"private_boot_{s}.npz").exists() for s in ALL_SCEN):
        done_marker("boot", {"draws": N_BOOT})


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


def outcome_of(rel, lo, mbest, applicable_text="not applicable (M_best <= 0)"):
    return applicable_text if not mbest > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")


def full_counterpart(m):
    for suf in ("_knn", "_cohortknn"):
        if m.endswith(suf) and not m.endswith("_knndenoise"):
            return m[: -len(suf)] + "_mean"
    return m


def stage_analyse():
    require_committed()
    require_done("boot")
    preds = J.load_preds()
    events = np.array([int(preds[ep].event.sum()) for ep in ENDPOINTS], float)
    inp = pd.read_csv(need(OUTM / f"M_inputs_{DATE}.csv")).set_index("scenario")
    Z = {s: np.load(need(PRIV / "boot" / f"private_boot_{s}.npz"), allow_pickle=False) for s in ALL_SCEN}
    ws = da_status()
    per, cons, counts, summ, rules, ew, loeo, ret, added, draws = [], [], [], [], [], [], [], [], [], []
    res = {"settings": {"seed": SEED, "draws": N_BOOT, "rule": RULE, "label": LABEL, "primary": PRIMARY,
                        "chosen_after": "this setting was chosen after the results of analyses J, K and L were known",
                        "depthaware_status": ws["status"]}, "scenarios": {}}
    Mfull = {}
    for scen in ALL_SCEN:
        z = Z[scen]
        models = list(z["models"].astype(str))
        P, B, S = z["point"], z["boot"], z["boot_within_fold"]
        ia, io = models.index("age_sex"), models.index(OURS)
        dP, dB, dS = P - P[ia], B - B[ia], S - S[ia]
        M, MB, MS = dP.mean(1), dB.mean(1), dS.mean(1)
        if scen == "full":
            Mfull = {m: float(M[k]) for k, m in enumerate(models)}
        dec = decision_for(scen)
        dec_i = [models.index(m) for m in dec]
        best = max(dec, key=lambda m: M[models.index(m)])
        kb = models.index(best)
        ext = dec + KNNDEN
        best_ext = max(ext, key=lambda m: M[models.index(m)])
        comps = [m for m in models if m not in (OURS, "age_sex")]
        for k, m in enumerate(models):
            draws.append(pd.DataFrame({"scenario": scen, "model": m, "draw": np.arange(N_BOOT), "M_shared": MB[k], "M_within_fold": MS[k]}))
        # per endpoint and model
        pmat = np.full((len(models), len(ENDPOINTS)), np.nan)
        for k in range(len(models)):
            if k not in (io, ia):
                pmat[k] = [J.pboot(dB[io, j] - dB[k, j]) for j in range(len(ENDPOINTS))]
        qmat = np.full_like(pmat, np.nan)
        for k in range(len(models)):
            if k not in (io, ia):
                qmat[k] = bh(pmat[k])
        for k, m in enumerate(models):
            for j, ep in enumerate(ENDPOINTS):
                d = dB[io, j] - dB[k, j]
                per.append({"scenario": scen, "label": label(scen), "endpoint": ep, "model": m, "role": role(m), "n": len(preds[ep]), "events": int(events[j]),
                            "c_index": P[k, j], "c_age_sex": P[ia, j], "delta_c": dP[k, j], "mgpt2_combined_minus_model": dP[io, j] - dP[k, j],
                            "ci95_low": pct(d, 2.5), "ci95_high": pct(d, 97.5), "p_boot": pmat[k, j], "q_bh_across_endpoints": qmat[k, j]})
        # per endpoint contrasts with C* and own best
        own = [dec_i[int(np.argmax(dP[dec_i, j]))] for j in range(len(ENDPOINTS))]
        p_own = [J.pboot(dB[io, j] - dB[own[j], j]) for j in range(len(ENDPOINTS))]
        q_own = bh(p_own)
        tot_diff = M[io] - M[kb]
        for j, ep in enumerate(ENDPOINTS):
            db, do = dB[io, j] - dB[kb, j], dB[io, j] - dB[own[j], j]
            diff, diffo = dP[io, j] - dP[kb, j], dP[io, j] - dP[own[j], j]
            cons.append({"scenario": scen, "label": label(scen), "endpoint": ep, "events": int(events[j]), "c_age_sex": P[ia, j],
                         "delta_c_mgpt2_combined": dP[io, j], "best_comparator": best, "delta_c_best": dP[kb, j], "difference_vs_best": diff,
                         "ci95_low_vs_best": pct(db, 2.5), "ci95_high_vs_best": pct(db, 97.5), "p_boot_vs_best": pmat[kb, j], "q_bh_vs_best": qmat[kb, j],
                         "contribution_to_M_difference": diff / len(ENDPOINTS),
                         "share_of_M_difference": (diff / len(ENDPOINTS)) / tot_diff if tot_diff != 0 else np.nan,
                         "own_best_comparator": models[own[j]], "delta_c_own_best": dP[own[j], j], "difference_vs_own_best": diffo,
                         "ci95_low_vs_own_best": pct(do, 2.5), "ci95_high_vs_own_best": pct(do, 97.5), "p_boot_vs_own_best": p_own[j],
                         "q_bh_vs_own_best": q_own[j],
                         "relative_gain_vs_own_best": diffo / dP[own[j], j] if dP[own[j], j] > 0 else np.nan,
                         "meets_rule_against_own_best": ("undefined (own best gain <= 0)" if not dP[own[j], j] > 0
                                                         else bool(diffo / dP[own[j], j] >= 0.10 and pct(do, 2.5) > 0))})
        # direction counts
        def count_row(name, comp, diffs, los, his, qs):
            return {"scenario": scen, "contrast": name, "comparator": comp, "n_endpoints": len(ENDPOINTS),
                    "n_mgpt2_combined_above": int((np.array(diffs) > 0).sum()), "n_comparator_above": int((np.array(diffs) < 0).sum()),
                    "n_interval_above_zero": int((np.array(los) > 0).sum()), "n_interval_below_zero": int((np.array(his) < 0).sum()),
                    "n_q05_favouring_mgpt2_combined": int(((np.array(qs) < 0.05) & (np.array(diffs) > 0)).sum()),
                    "n_q05_favouring_comparator": int(((np.array(qs) < 0.05) & (np.array(diffs) < 0)).sum())}
        cs_ = [c for c in cons if c["scenario"] == scen]
        counts.append(count_row("best decision comparator (C*)", best, [c["difference_vs_best"] for c in cs_], [c["ci95_low_vs_best"] for c in cs_],
                                [c["ci95_high_vs_best"] for c in cs_], [c["q_bh_vs_best"] for c in cs_]))
        r_own = count_row("own best decision comparator per endpoint", "per endpoint", [c["difference_vs_own_best"] for c in cs_],
                          [c["ci95_low_vs_own_best"] for c in cs_], [c["ci95_high_vs_own_best"] for c in cs_], [c["q_bh_vs_own_best"] for c in cs_])
        r_own["n_meets_rule_against_own_best"] = sum(c["meets_rule_against_own_best"] is True for c in cs_)
        r_own["n_rule_undefined"] = sum(isinstance(c["meets_rule_against_own_best"], str) for c in cs_)
        counts.append(r_own)
        for m in comps:
            k = models.index(m)
            lo_ = [pct(dB[io, j] - dB[k, j], 2.5) for j in range(len(ENDPOINTS))]
            hi_ = [pct(dB[io, j] - dB[k, j], 97.5) for j in range(len(ENDPOINTS))]
            counts.append(count_row("comparator", m, dP[io] - dP[k], lo_, hi_, qmat[k]))
        # summary against every comparator (J's)
        for m in comps:
            k = models.index(m)
            d, ds = MB[io] - MB[k], MS[io] - MS[k]
            rel = (M[io] - M[k]) / M[k] if M[k] > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                relb = d / MB[k]
            lo, hi = pct(d, 2.5), pct(d, 97.5)
            summ.append({"scenario": scen, "label": label(scen), "model": m, "role": role(m), "is_best_decision_comparator": m == best,
                         "M_model": M[k], "M_model_ci95_low": pct(MB[k], 2.5), "M_model_ci95_high": pct(MB[k], 97.5), "M_mgpt2_combined": M[io],
                         "difference": M[io] - M[k], "ci95_low": lo, "ci95_high": hi, "ci95_low_within_fold": pct(ds, 2.5),
                         "ci95_high_within_fold": pct(ds, 97.5), "p_boot": J.pboot(d), "relative_gain": rel,
                         "relative_gain_ci95_low": float(np.nanpercentile(relb, 2.5)) if M[k] > 0 else np.nan,
                         "relative_gain_ci95_high": float(np.nanpercentile(relb, 97.5)) if M[k] > 0 else np.nan,
                         "rule_against_this_model": outcome_of(rel, lo, M[k], "not applicable (M <= 0)")})
        # rule outcome (one row per scenario)
        d, ds = MB[io] - MB[kb], MS[io] - MS[kb]
        rel = (M[io] - M[kb]) / M[kb] if M[kb] > 0 else np.nan
        lo, hi = pct(d, 2.5), pct(d, 97.5)
        outcome = outcome_of(rel, lo, M[kb])
        row = {"analysis": "M", "scenario": scen, "decision_row": scen == PRIMARY, "label": label(scen),
               "metric": "mean over 8 endpoints of (pooled within-fold C-index - age+sex C-index)", "methylgpt": OURS, "M_methylgpt": M[io],
               "M_methylgpt_ci95_low": pct(MB[io], 2.5), "M_methylgpt_ci95_high": pct(MB[io], 97.5), "best_comparator": best, "M_best": M[kb],
               "relative_gain": rel, "difference": M[io] - M[kb], "ci95_low": lo, "ci95_high": hi,
               "interval": f"paired bootstrap, one shared participant resample per draw, {N_BOOT} draws, seed {SEED}",
               "ci95_low_within_fold": pct(ds, 2.5), "ci95_high_within_fold": pct(ds, 97.5), "p_boot": J.pboot(d), "rule": RULE, "outcome": outcome}
        if scen != "full":
            row.update({"fraction_gs_measured_vocab_values_with_no_read": float(inp.loc[scen, "fraction_gs_measured_vocab_values_with_no_read"]),
                        "rmse_noisy_vs_measured": float(inp.loc[scen, "rmse_noisy_vs_measured"]),
                        "mean_within_participant_pearson_r": float(inp.loc[scen, "mean_within_participant_pearson_r"])})
        rules.append(row)
        sres = {"M": {m: float(M[models.index(m)]) for m in models if m != "age_sex"}, "best": best, "relative_gain": rel,
                "difference": float(M[io] - M[kb]), "ci95": [lo, hi], "outcome": outcome, "best_extended": best_ext}
        # event-weighted mean (sensitivity)
        wt = events / events.sum()
        Mw, MBw = (dP * wt[None, :]).sum(1), (dB * wt[None, :, None]).sum(1)
        bw = max(dec, key=lambda m: Mw[models.index(m)])
        kw = models.index(bw)
        dw = MBw[io] - MBw[kw]
        relw = (Mw[io] - Mw[kw]) / Mw[kw] if Mw[kw] > 0 else np.nan
        ew.append({"scenario": scen, "label": label(scen), "statistic": "event-weighted mean over 8 endpoints (sensitivity)", "M_mgpt2_combined": Mw[io],
                   "best_comparator": bw, "M_best": Mw[kw], "relative_gain": relw, "difference": Mw[io] - Mw[kw], "ci95_low": pct(dw, 2.5),
                   "ci95_high": pct(dw, 97.5), "p_boot": J.pboot(dw),
                   "rule_arithmetic_for_reference": outcome_of(relw, pct(dw, 2.5), Mw[kw]).replace("PASS", "meets").replace("FAIL", "does not meet")})
        # leave one endpoint out
        for j, ep in enumerate(ENDPOINTS):
            keep = np.arange(len(ENDPOINTS)) != j
            Ml, MBl = dP[:, keep].mean(1), dB[:, keep].mean(1)
            bl = max(dec, key=lambda m: Ml[models.index(m)])
            kl = models.index(bl)
            dl, df_ = MBl[io] - MBl[kl], MBl[io] - MBl[kb]
            rell = (Ml[io] - Ml[kl]) / Ml[kl] if Ml[kl] > 0 else np.nan
            relf = (Ml[io] - Ml[kb]) / Ml[kb] if Ml[kb] > 0 else np.nan
            loeo.append({"scenario": scen, "left_out": ep, "n_endpoints": 7, "M_mgpt2_combined": Ml[io],
                         "best_comparator_rechosen": bl, "M_best_rechosen": Ml[kl], "relative_gain_rechosen": rell, "difference_rechosen": Ml[io] - Ml[kl],
                         "ci95_low_rechosen": pct(dl, 2.5), "ci95_high_rechosen": pct(dl, 97.5), "p_boot_rechosen": J.pboot(dl),
                         "rule_arithmetic_rechosen_for_reference": outcome_of(rell, pct(dl, 2.5), Ml[kl]).replace("PASS", "meets").replace("FAIL", "does not meet"),
                         "best_comparator_fixed": best, "M_best_fixed": Ml[kb], "relative_gain_fixed": relf, "difference_fixed": Ml[io] - Ml[kb],
                         "ci95_low_fixed": pct(df_, 2.5), "ci95_high_fixed": pct(df_, 97.5), "p_boot_fixed": J.pboot(df_)})
        # added contrasts (outside the decision)
        def contrast(name, a, b_):
            ka, kb_ = models.index(a), models.index(b_)
            dd = MB[ka] - MB[kb_]
            r_ = (M[ka] - M[kb_]) / M[kb_] if M[kb_] > 0 else np.nan
            added.append({"scenario": scen, "contrast": name, "model_a": a, "role_a": role(a), "model_b": b_, "role_b": role(b_), "M_a": M[ka], "M_b": M[kb_],
                          "difference": M[ka] - M[kb_], "ci95_low": pct(dd, 2.5), "ci95_high": pct(dd, 97.5), "p_boot": J.pboot(dd), "relative_gain": r_,
                          "rule_arithmetic_for_reference": outcome_of(r_, pct(dd, 2.5), M[kb_], "not applicable (M_b <= 0)").replace("PASS", "meets").replace("FAIL", "does not meet")})
        contrast("mgpt2_combined vs best of decision set and kNN-denoising comparators (C*_ext)", OURS, best_ext)
        if DA in models:
            contrast("depth-aware MethylGPT combined vs C*", DA, best)
            contrast("depth-aware MethylGPT combined vs C*_ext", DA, best_ext)
            contrast("depth-aware vs standard MethylGPT combined", DA, OURS)
        for m in FDEN:
            contrast("factor-denoised raw model vs C*", m, best)
        for b_ in BASES:
            contrast("kNN denoising vs kNN fill of the same comparator", f"{b_}_knndenoise", f"{b_}_mean" if scen == "full" else f"{b_}_knn")
        res["scenarios"][scen] = sres
        log(f"{scen}: M {OURS} {M[io]:.4f}; best {best} {M[kb]:.4f}; gain {rel:+.1%}; diff {M[io] - M[kb]:+.4f} [{lo:.4f}, {hi:.4f}] -> {outcome}")
    # retention (descriptive)
    for scen in NOISE:
        for m, v in res["scenarios"][scen]["M"].items():
            fm_ = full_counterpart(m)
            mf = Mfull.get(fm_, np.nan)
            ret.append({"scenario": scen, "model": m, "role": role(m), "M_scenario": v, "M_full_reference_model": fm_, "M_full": mf,
                        "fraction_retained": v / mf if mf > 0 else np.nan, "note": "descriptive only" if mf > 0 else "undefined (M_full <= 0)"})
    files = {f"M_cindex_per_endpoint_{DATE}.csv": per, f"M_endpoint_contrasts_{DATE}.csv": cons, f"M_direction_counts_{DATE}.csv": counts,
             f"M_summary_{DATE}.csv": summ, f"M_rule_outcome_{DATE}.csv": rules, f"M_event_weighted_{DATE}.csv": ew,
             f"M_leave_one_endpoint_out_{DATE}.csv": loeo, f"M_retention_{DATE}.csv": ret, f"M_added_contrasts_{DATE}.csv": added}
    for fn, rows_ in files.items():
        atomic_csv(OUTM / fn, pd.DataFrame(rows_))
    atomic_csv(OUTM / f"M_bootstrap_draws_{DATE}.csv.gz", pd.concat(draws, ignore_index=True), compression="gzip")
    write_json(OUTM / f"M_results_{DATE}.json", res)
    dr = {"wcheck": ws, "scenarios": {s: json.loads(need(PRIV / "mgpt" / f"private_record_{s}.json").read_text())["weights"] for s in ALL_SCEN}}
    write_json(OUTM / f"M_depthaware_record_{DATE}.json", dr)
    write_readme(res)
    done_marker("analyse")


def write_readme(res):
    def nrow(fn):
        return "-" if fn.endswith(".json") else str(len(pd.read_csv(OUTM / fn, dtype=str)))

    def cols(fn):
        return "-" if fn.endswith(".json") else ", ".join(pd.read_csv(OUTM / fn, nrows=1).columns)

    desc = {
        f"M_inputs_{DATE}.csv": "Per noise scenario: kind and parameter, observed vocabulary CpGs per participant, GS-measured values with no read, clock CpGs observed, mean reads, RMSE and within-participant r of the noisy against the measured values (from noise_summary).",
        f"M_fold_models_{DATE}.csv": "Every fold-model file and the endpoint table read by M: family, endpoint, fold, path, size, modification time, whether it predates J's first stage, SHA-256.",
        f"M_checks_{DATE}.csv": "Every pre-specified check with stored and recomputed values, tolerance and pass (noise inputs, exact reproduction of J's full-profile values, depth-aware identity, fold models, leakage, corpus reference, age+sex rows, kNN integrity, pair counting, bootstrap weights, privacy).",
        f"M_knn_record_{DATE}.json": "Corpus kNN reference rows and SHA-256 against J's and G's; per scenario fill and denoising counts, integrity flags, comparison with J's knn_fill, timing.",
        f"M_depthaware_record_{DATE}.json": "Depth-aware posterior: code reading, sigma2 of each core, w = 1 identity tests (first 256 participants of full and depth_10), status, and per scenario the mean, minimum and maximum weight over observed values.",
        f"M_cindex_per_endpoint_{DATE}.csv": "Per scenario, endpoint and model: role, n, events, pooled within-fold C-index, age+sex C-index, gain above age+sex, MethylGPT combined minus model with 95% interval, bootstrap P and Benjamini-Hochberg q across the 8 endpoints.",
        f"M_endpoint_contrasts_{DATE}.csv": "Per scenario and endpoint: MethylGPT combined against the scenario's best decision comparator (difference, interval, P, q, contribution to the M difference) and against the endpoint's own best decision comparator (chosen by point estimate, fixed in the draws).",
        f"M_direction_counts_{DATE}.csv": "Per scenario and contrast (C*, own best, every comparator): endpoints by sign of the difference, by interval and by q < 0.05.",
        f"M_summary_{DATE}.csv": "Per scenario and comparator: M with interval, MethylGPT combined's M, difference with shared and within-fold intervals, bootstrap P, relative gain with interval, rule arithmetic against that comparator.",
        f"M_rule_outcome_{DATE}.csv": "One row per scenario; decision_row marks depth_10, the pre-specified decision. M of MethylGPT combined, best decision comparator and its M, relative gain, difference, 95% intervals (shared and within-fold), P, rule, outcome.",
        f"M_event_weighted_{DATE}.csv": "Per scenario: event-weighted mean gain (sensitivity), best comparator re-chosen, difference with interval, rule arithmetic for reference only.",
        f"M_leave_one_endpoint_out_{DATE}.csv": "Per scenario and left-out endpoint: M over the other 7 endpoints with the best comparator re-chosen and with it fixed at the 8-endpoint choice; intervals from the same draws; rule arithmetic for reference only.",
        f"M_retention_{DATE}.csv": "Per noise scenario and model: M, the complete-profile M of the same model (or of its mean-fill counterpart for kNN and in-cohort fills, which fill nothing on complete profiles), fraction retained; descriptive only.",
        f"M_added_contrasts_{DATE}.csv": "Outside the decision: MethylGPT combined against the best of the decision set and the kNN-denoising comparators; depth-aware MethylGPT combined against C*, C*_ext and standard MethylGPT combined; factor-denoised raw models against C*; kNN denoising against kNN fill of each comparator. Rule arithmetic for reference only.",
        f"M_bootstrap_draws_{DATE}.csv.gz": "Per scenario, model and draw: M under the shared resample and under the within-fold resample (2,000 draws).",
        f"M_results_{DATE}.json": "Settings (including when the setting was chosen and the depth-aware status) and, per scenario, M of every model, best comparator, gain, difference, interval and outcome.",
    }
    pr = res["scenarios"][PRIMARY]
    lines = ["# Large-gain analysis M: GS incident disease with measurement noise", "",
             f"Script: `large_gain_M_gs_measurement_noise_2026-11-02.py` (pre-specified design in its docstring). Every noise scenario is {LABEL}: "
             "measured GS development profiles (9,866 participants) with noise added by the shared generator `large_gain_noise_profiles_2026-11-02.py` "
             "(read sampling at mean depth 10 primary; depth 1, 2, 5, 30 and Gaussian SD 0.02, 0.05, 0.10 secondary). J's fold models are applied "
             "unchanged; no disease model is refitted. full is the measured data (reference). This setting was chosen after the results of analyses J, K and L were known.", "",
             "## Primary result (depth_10)", "",
             f"- MethylGPT combined M = {pr['M'][OURS]:.4f}; best decision comparator {pr['best']} M = {pr['M'][pr['best']]:.4f}.",
             f"- Relative gain {pr['relative_gain']:+.1%}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             f"- Depth-aware posterior: {res['settings']['depthaware_status']}.",
             "", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUTM / fn).exists():
            lines.append(f"| `{fn}` | {nrow(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              "- Noisy profiles: `artifacts/large_gain_2026-11-02_private/noise/private_noisy_<scenario>.npz` (SHA-256 in `artifacts/large_gain_2026-11-02/noise/noise_summary_2026-11-02.csv`, verified at every load).",
              "- GS development frame-cache matrix and endpoint table `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet` (J's locked folds).",
              "- J's fold models (`artifacts/gs_cox/mgpt2_embed_linear_2026-10-01`, `mt_bio_linear_2026-10-01`, `factor_residual_2026-10-01/models`, v12 raw-CpG runs under `/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results`), hashes in M_fold_models.",
              "- Clock CpG matrix and stored scores `artifacts/gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.*`; J's stored private risks and C-indices (reproduction checks).",
              "- kNN reference: G's held-out reference (`large_gain_G_sparse_inputs_2026-10-25.py`), row SHA-256 equal to J's.",
              "", "## Notes", "",
              "- Interval: one participant resample per draw shared by every model, endpoint and scenario, 2,000 draws, seed 20261102; J's within-fold resample as a sensitivity.",
              "- Decision comparators: J's DECISION set (raw-CpG fold models with fold-mean or corpus-kNN fill of no-read CpGs; GrimAge2, DunedinPACE and both with mean- or corpus-kNN-filled clock CpGs). In-cohort kNN is a secondary comparator; raw models with MethylGPT factor imputation or factor denoising are MethylGPT applications; kNN-denoising comparators and the depth-aware MethylGPT model are pre-specified additions outside the decision.",
              "- Gaussian scenarios and depth_30 have no or almost no read-level missingness, so their kNN and in-cohort fills change nothing or almost nothing; such comparators then equal their mean-fill counterparts.",
              "- Participant-level fills, clock inputs, scores, risks and bootstrap arrays stay in `artifacts/large_gain_2026-11-02_private/M/` on Chronos and are never committed."]
    (OUTM / "README.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ stage privacy
def stage_privacy():
    require_committed()
    require_done("analyse")
    ids = set(J.load_gs()[0])
    token = re.compile(r"[A-Za-z0-9_]+")
    forbidden = {"dnam_id", "sample_id", "participant", "participant_id", "row", "id"}
    rows = []
    for p in sorted(OUTM.iterdir()):
        text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
        hits = len(set(token.findall(text)) & ids)
        bad_cols, small = [], 0
        if ".csv" in p.name:
            df = pd.read_csv(p, dtype=str)
            bad_cols = sorted(set(c.lower() for c in df.columns) & forbidden)
            if "events" in df.columns:
                v = pd.to_numeric(df["events"], errors="coerce")
                small = int((v < SMALL_CELL).sum())
        rows.append({"check": "11 privacy of public files", "scenario": "all", "item": p.name, "stored": 0, "recomputed": hits + len(bad_cols) + small,
                     "abs_diff": "", "tolerance": 0, "pass": hits == 0 and not bad_cols and small == 0,
                     "note": f"identifier tokens {hits}; identifier columns {bad_cols}; event cells below {SMALL_CELL}: {small}"})
    ck = pd.read_csv(need(OUTM / f"M_checks_{DATE}.csv"))
    ck = ck[ck.check != "11 privacy of public files"]
    out = pd.concat([ck, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(OUTM / f"M_checks_{DATE}.csv", out)
    write_readme(json.loads(need(OUTM / f"M_results_{DATE}.json").read_text()))
    nf = sum(not r["pass"] for r in rows)
    log(f"privacy: {len(rows)} public files, {nf} failed")
    if nf:
        print(pd.DataFrame(rows).query("~`pass`").to_string())
        sys.exit("privacy check failed; do not commit the outputs")
    done_marker("privacy", {"files": len(rows)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["inputs", "wcheck", "mgpt", "knn", "cohort", "clockin", "clocks", "cox", "checks", "boot", "analyse", "privacy"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.chdir(HERE)          # checkpoint configs hold paths relative to the repository directory
    LOGS.mkdir(parents=True, exist_ok=True)
    {"inputs": stage_inputs, "wcheck": lambda: stage_wcheck(a.device), "mgpt": lambda: stage_mgpt(a.device, a.shard, a.nshards),
     "knn": lambda: stage_knn(a.device), "cohort": lambda: stage_cohort(a.device, a.shard, a.nshards), "clockin": stage_clockin,
     "clocks": lambda: stage_clocks(a.workers), "cox": stage_cox, "checks": stage_checks, "boot": lambda: stage_boot(a.shard, a.nshards),
     "analyse": stage_analyse, "privacy": stage_privacy}[a.stage]()


if __name__ == "__main__":
    main()
