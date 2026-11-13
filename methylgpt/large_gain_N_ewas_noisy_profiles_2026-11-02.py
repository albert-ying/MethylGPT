#!/usr/bin/env python3
"""Large-gain analysis N: disease EWAS from noisy measurements of every CpG (Generation Scotland). SIMULATED MEASUREMENT
NOISE ON EMPIRICAL PROFILES: every profile is a measured GS development profile; the noise is the shared simulated noise of
large_gain_noise_profiles_2026-11-02.py (commit 729f99e). No noise is drawn here. Tables only (no figures).

Written on 2 Nov 2026 from the analysis brief for N, BEFORE any stage that produces a result of N is run. Every result is
reported, whichever method it favours.

Why
  Analyses D (large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py; incident type 2 diabetes) and D2
  (large_gain_D2_ewas_unmeasured_cpgs_endpoints_2026-10-28.py; the other seven endpoints) showed that MethylGPT recovers disease
  associations at CpGs a profile does not measure (27K setting: Spearman rho of per-CpG Cox z with the full-data z 0.666 against
  0.311 for corpus nearest neighbours in D; mean over D2's seven endpoints 0.768 against 0.585). N asks whether MethylGPT recovers
  them from noisy measurements of every CpG, as simulated low-depth sequencing produces. The setting (measurement noise) was
  chosen on 2 Nov 2026 after analyses J, K and L had been run and their results seen (J: MethylGPT's advantage grew as the
  deployed input lost CpGs; K: no widening on individual diseases; L: risk stratification); this is recorded here, in the
  README and in the results file. Label of every output: "simulated measurement noise on empirical profiles".

What is D's and D2's, unchanged (imported with importlib, functions used as they are)
  D: load_gs (participants, frame-cache matrix, type 2 diabetes cross-checks), load_sets (D's frozen CpG sets,
     artifacts/large_gain_2026-10-24/D/D_cpg_sets.csv.gz), CoxBatch (the estimator; used for the in-memory comparison and as the
     fallback), compare and zfill (primary metric and undefined-z rule), hits, recall_precision and bh (hits), accuracy (value
     accuracy), pct (percentile interval), boot_p (bootstrap P), sha256_file, BUNDLE and CHECKSUMS.
  D2: load_endpoints (event, follow-up, age and sex of the eight endpoints with D2's freeze checks), privacy_check,
     ALL_ENDPOINTS, PRIMARY_ENDPOINTS, wait_for_gpu.
  si_reconstruction_diagnostics_2026-10-18.py: knn_predict_lowmem (D's nearest-neighbour code).

Data and inclusion (exactly D and D2)
  Participants: D's 9,866 GS development participants (training subset of the locked incident-disease protocol, frame-cache
  order; frame-cache matrix COMMON_SAMPLE_SPLIT_full18859_common_task_a_incident_positive_duration_20260629_train_...; 43,899 of
  the 49,156 vocabulary CpGs observed in every participant). Nobody is excluded.
  Endpoints: all eight endpoints of the locked protocol (D2_frozen_endpoints.csv): Endocrine_and_Metabolic_Diseases (incident
  type 2 diabetes, D's endpoint), Autoimmune_Diseases, Cancers, Cardiovascular_Diseases, Kidney_Diseases,
  Neurological_Diseases, Respiratory_Diseases, date_of_death; events 135 to 687; every development participant at risk for
  every endpoint. Event, follow-up, age and sex from D2.load_endpoints (locked endpoint table, age_adjustment linear), with
  D2's checks; event counts and the event/duration SHA-256 must equal D2_frozen_endpoints.csv.
  Target CpGs: D's 22,322 GS-observed vocabulary CpGs outside the 27K array (target_hm27 of D_cpg_sets.csv.gz). The set is fixed
  by D, not chosen on any outcome. Every method gives a value at every target CpG for every participant.

Noisy input (shared; read only, never modified)
  artifacts/large_gain_2026-11-02_private/noise/private_noisy_<scenario>.npz written by large_gain_noise_profiles_2026-11-02.py
  (commit 729f99e; generator, seeds and scenarios in its docstring): vocab (float32, 9,866 x 49,156, frame-cache order; NaN where
  a CpG is absent from GS or has no read) and, for depth scenarios, reads_vocab (uint8 reads n; the observed value is m / n with
  m methylated reads). The inputs stage waits for done_noise.json, checks that the noise script is unchanged since 729f99e,
  that its checks table passes, and recomputes the SHA-256 of every array of the five N scenarios against
  noise_summary_2026-11-02.csv (abort on mismatch); the values stage verifies the hashes again when it loads an array.
  Scenarios of N: PRIMARY depth_10 (simulated bisulfite read sampling, reads ~ Poisson(10)); SECONDARY, all pre-specified and
  all run and reported: depth_2, depth_5, depth_30, gauss_0.05 (clip(beta + 0.05 e, 0, 1)). depth_1, gauss_0.02 and gauss_0.10
  of the shared input are not part of N. The noise realisation is fixed (one draw of the shared generator); the bootstrap
  resamples participants only, so intervals do not include noise-realisation variance.

Methods (each value at a target CpG uses only that participant's noisy profile, except the stated raw_noisy fill)
  raw_noisy (comparator)   the noisy value; a no-read value (depth scenarios) is filled with that CpG's mean over the
                           participants with a read in that scenario (no outcome used).
  knn_denoised (comparator) D's corpus nearest neighbours (knn_predict_lowmem; reference = all 145,700 corpus training-split
                           profiles, np.sort(Corpus().rows("train")), float16, unobserved reference values filled with the bundle's
                           core mu; k = 10; squared Euclidean distance over the query's context CpGs, TF32 for the search only;
                           query batch 1,024, reference chunk 8,192), with context = the noisy observed vocabulary CpGs (every
                           vocabulary CpG with a value in the noisy profile) and prediction at every target CpG = the mean of the
                           neighbours' observed values (mean-filled value if none is observed); the measured noisy value is
                           replaced.
  methylgpt (MethylGPT)    D's factor-model procedure (released bundle artifacts/release/methylgpt2-1.1, checksums verified;
                           MethylGPT.impute(method="factor") arithmetic: input built by MethylGPT._matrix, posterior factor scores
                           core.posterior(x, isfinite(x)) from the noisy observed vocabulary CpGs, reconstruction core.predict(s)
                           = mu + V s clipped to [0, 1], batch 8, TF32 off), taken at every target CpG (the measured noisy value
                           is replaced).
  Applications of MethylGPT, outside the decision, reported:
  methylgpt_depthaware     as methylgpt with posterior weights w = sigma2 / (sigma2 + v) at observed CpGs (0 where no read):
                           sigma2 = the factor model's residual variance (core.sigma2 of the bundle, a scalar); depth scenarios
                           v = p (1 - p) / n with p = (m + 0.5) / (n + 1), n = reads, m = round(value x n); gauss_0.05 v = 0.05^2.
                           Run only if (i) the code of PPCACore.posterior uses w only as a multiplicative weight (the two lines
                           xc = torch.where(w > 0, x - mu, 0) * w and G = (V * w[i, :, None]).T @ V; checked on the source text)
                           and (ii) w = 1 at observed CpGs reproduces the locked scores exactly (torch.equal on the depth_10 scores
                           of the first 64 participants); otherwise recorded as "not run" with the reason.
  methylgpt_blend          posterior-mean shrinkage of each noisy value x toward the locked methylgpt reconstruction r:
                           r + (sigma2 / (sigma2 + v)) (x - r), with the same v; r where there is no read.

Reference z: the stored full-data z at the target CpGs (D2_ewas_per_cpg_hm27.csv.gz, column full_z, all eight endpoints; for
  type 2 diabetes it must equal D's D_ewas_per_cpg_hm27.csv.gz exactly). Both files are checked to be tracked and unchanged in git
  and their SHA-256 are recorded. Point estimates use these stored z; bootstrap draws refit the full data (D2's procedure).

EWAS estimator: D's per-CpG model, unchanged (Cox proportional hazards of the endpoint on the CpG value, age and sex; Breslow ties;
  CpG, age and sex standardized over the participants, frequency-weighted in bootstrap draws; Newton-Raphson in float64 from 0 with
  step halving (12), stop at a largest step < 1e-9 or after 50 iterations; not converged if the last step > 1e-6, the estimate
  is not finite or the information is singular; SD < 1e-8 gives an undefined z; beta = log HR per SD, se from the inverse observed
  information, Wald z). Implementation: FastCox below, a GPU kernel (numba.cuda, float64) that evaluates the Breslow log partial
  likelihood, score and information of every CpG in one pass over the rows, with the same standardization, step halving and
  stopping rule as D's CoxBatch; a CpG stops iterating once its step is < 1e-9 (D stops a batch when every step is); bootstrap
  draws start Newton from the CpG's point estimate (the maximum is unique, so this changes the path, not the estimate). It is used
  only if, before any noisy-profile z is used (--stage coxcheck), it reproduces D's and D2's stored full-data z at every target CpG
  and endpoint within |delta z| <= 1e-4 with the same undefined pattern (8 x 22,322 values against D2, 22,322 against D); also
  recorded: its largest |delta z| against D's CoxBatch in memory (type 2 diabetes, 22,322 CpGs), and the draw-level rho of D2's
  bootstrap draws 0-2 recomputed with warm-started FastCox on D's private hm27 imputations (MethylGPT and nearest neighbours, all
  eight endpoints) against D2_bootstrap_draws_hm27.csv (tolerance 1e-6) with the resample hashes equal to D2's. If any of these
  fails, the stage stops, FastCox is not used, and the fit runs D2's code (D.CoxBatch, 12,000 CpGs per batch, no warm start;
  --cox d2) under a dated amendment.

Primary metric (PRIMARY scenario depth_10)
  rho_e(v) = Spearman correlation over the 22,322 target CpGs between the per-CpG z of method v and the full-data z for endpoint
  e, with D's undefined-z rule (undefined z set to 0; a method whose z are all undefined has rho 0). rho_bar(v) = mean of rho_e(v)
  over the eight endpoints.
Comparators: raw_noisy and knn_denoised. Best comparator = the one with the higher rho_bar in that scenario (point estimate).
Decision rule: PASS iff (rho_bar_MethylGPT - rho_bar_best) / rho_bar_best >= 0.10 and the lower end of the 95% interval of
  rho_bar_MethylGPT - rho_bar_best is > 0; otherwise FAIL; if rho_bar_best <= 0 the gain is undefined and the rule is not
  applicable (fail). The decision is the depth_10 row (decision_row = True); the same rule is applied to each secondary
  scenario and reported (one row per scenario in N_rule_outcome.csv).
Interval: D2's pre-specified procedure, unchanged. D2's docstring: "D's participant bootstrap: 1,000 draws; numpy
  default_rng(20261024); draw b uses the b-th call of rng.integers(0, 9866, 9866) on the frame-cache participant order (bincount =
  frequency weights), which is D's draw b. The same resample is shared by every endpoint, setting and version in a draw ... In
  each draw the full-data, MethylGPT and nearest-neighbour EWAS are refitted at the target CpGs for every endpoint, rho_e
  recomputed, and the difference of the endpoint means formed within the draw. Imputed values are not recomputed (as D).
  Percentile 95% intervals; bootstrap P = min(1, 2(k + 1)/(B + 1)) (D's boot_p)." In N every method of every scenario is refitted
  in each draw against the draw's full-data refit; method values are not recomputed. Caveat about percentile intervals of single
  correlations: D2's README defers to D's README, which states: "The primary interval, of the difference in Spearman correlation
  between the two imputed versions, compares versions within each draw. In the 27K setting the bootstrap median of the MethylGPT
  correlation lies below its point estimate (0.62 against 0.67) and that of nearest neighbours slightly above it (0.33 against
  0.31)." The percentile intervals of single correlations are therefore reported but are not centred on the point estimates; the
  decision uses the interval of the difference.
Secondary (all reported; descriptive)
  (a) Per endpoint and scenario: rho_e of every method with intervals, difference MethylGPT minus the endpoint's best comparator
      with interval and bootstrap P, relative gain, the rule arithmetic for that endpoint alone.
  (b) The same metric over the seven endpoints other than type 2 diabetes (best comparator chosen on the seven-endpoint mean),
      per scenario, with the rule arithmetic.
  (c) Hit overlap: hits are BH q < 0.05 over the 22,322 targets (D.hits); per scenario, method and endpoint the number of
      full-data hits (stored z), hits of the method, hits shared with the same sign, recall and precision. D2's caveat applies:
      FDR on reconstructed or denoised values is not calibrated (the values share information across CpGs and participants), so
      hit counts are descriptive and no new hit is claimed.
  (d) The two MethylGPT applications: rho_bar (eight and seven endpoints) with intervals, difference against the scenario's best
      comparator and against methylgpt with intervals and the rule arithmetic (outside the decision).
  (e) Value accuracy at the target CpGs against the measured values (D.accuracy: MSE, MAE, pooled within-CpG R^2, median per-CpG
      Pearson r) per scenario and method.
  (f) Number of undefined (not converged) z per scenario, method and endpoint.
Seeds and resamples: bootstrap default_rng(20261024), 1,000 draws (D2's). No other random number is drawn.

Checks before any new number is used (N_checks.csv; the stage aborts on any failure)
  1 noise input: script unchanged since 729f99e, noise checks table all pass, label, done marker, SHA-256 of every array of the
    five scenarios equal to noise_summary_2026-11-02.csv, shapes, dtypes, NaN pattern (absent CpGs NaN; depth: NaN iff no read at
    GS CpGs; gauss: no NaN at GS CpGs), participant order (noise generator's load = D.load_gs), hashes again at use.
  2 D and D2 inputs: stored tables tracked and unchanged in git (SHA-256 recorded); type 2 diabetes z of D2 equal to D's; D2's
    endpoint freeze reproduced (event counts and event/duration SHA-256); D's CpG-set counts.
  3 full-data z reproduction by FastCox (above), the in-memory comparison with D's CoxBatch and the D2 draw reproduction.
  4 kNN reference: row SHA-256 (int64 row indices, G's convention) of N's reference equals D's. D recorded no row hash; D's
    reference is defined by D's code (np.sort(Corpus().rows("train")), 145,700 rows, the count J's record gives for D's reference).
    N builds it with that code and checks it is D's by the row count and by reproducing D's stored hm27 nearest-neighbour fills
    of the first query batch (1,024 participants, max |diff| <= 1e-6); the SHA-256 is recorded.
  5 MethylGPT code path: N's methylgpt function reproduces D's stored hm27 MethylGPT imputations of the first 64 participants
    (max |diff| <= 1e-6); the depth-aware conditions above.
  6 values: shapes, finite, within [0, 1]; SHA-256 recorded and verified when the fit stage loads them; the fit stage's own
    full-data refit equals the stored z within 1e-4 for every endpoint.
  7 draws: one resample per draw shared by every endpoint, scenario and method, equal to D2's resample hash of that draw; 1,000
    draws per endpoint, scenario and method.
  8 disclosure: no participant identifier in any public file (D2.privacy_check: token match against the 9,866 IDs, barcode regex,
    identifier columns); event cells below 5 suppressed, as D and D2 (their tables hold endpoint totals only, all >= 135; N writes
    endpoint totals only and suppresses any count below 5; the number suppressed is reported); every *.csv.gz is gzip-compressed
    (D2 wrote uncompressed working files under a .gz name; N writes gzip explicitly and checks the magic bytes).

Compute and order
  Run the primary scenario first, then every secondary. The fit stage processes group "primary" (depth_10: full data plus five
  methods) for its endpoints before group "secondary" (four scenarios x five methods), on one worker per GPU (endpoints split in a
  fixed order; speed only), checkpointing every 10 draws. The coxcheck stage records the measured time per draw; if compute forces
  a reduction, it is decided in a dated amendment before any primary result exists. Logs carry progress and timing only.

Stages (each requires this script committed unchanged; outputs of earlier stages required by later ones)
  --stage inputs    (CPU) checks 1 and 2; waits for the noise done marker.
  --stage coxcheck  (GPU) check 3; timing.
  --stage values    (GPU) checks 4 and 5; method values of every scenario (primary first); value accuracy.
  --stage fit       (GPU) per endpoint and group: point fits and 1,000 bootstrap draws (workers per device).
  --stage analyze   (CPU) tables, rule outcome, checks table, README, disclosure checks.
Outputs: aggregate and per-CpG summary tables in artifacts/large_gain_2026-11-02/N/ (README.md lists each file, rows, columns and
  inputs; N_rule_outcome.csv one row per scenario, decision_row = depth_10). Participant-level values, working files and logs in
  artifacts/large_gain_2026-11-02_private/N/ (git-ignored, never committed, stay on Chronos).
Run: python3 large_gain_N_ewas_noisy_profiles_2026-11-02.py --stage {inputs,coxcheck,values,fit,analyze} [--devices 0,1]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, log, require, write_json

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


D = _load("large_gain_D", "large_gain_D_ewas_unmeasured_cpgs_2026-10-24.py")
D2 = _load("large_gain_D2", "large_gain_D2_ewas_unmeasured_cpgs_endpoints_2026-10-28.py")

DATE = "2026-11-02"
LABEL = "simulated measurement noise on empirical profiles"
OUT = ARTIFACTS / "large_gain_2026-11-02/N"
PRIV = ARTIFACTS / "large_gain_2026-11-02_private/N"
WORK = PRIV / "work"
VALS = PRIV / "values"
LOGS = PRIV / "logs"
NOISE_SCRIPT = HERE / "large_gain_noise_profiles_2026-11-02.py"
NOISE_COMMIT = "729f99e"
NOISE_PRIV = ARTIFACTS / "large_gain_2026-11-02_private/noise"
NOISE_OUT = ARTIFACTS / "large_gain_2026-11-02/noise"
NOISE_SUMMARY = NOISE_OUT / "noise_summary_2026-11-02.csv"
NOISE_CHECKS = NOISE_OUT / "noise_checks_2026-11-02.csv"
J_KNN_RECORD = ARTIFACTS / "large_gain_2026-10-28/J/J_knn_record_2026-10-28.json"
RECON = HERE / "si_reconstruction_diagnostics_2026-10-18.py"
D_PER_CPG = D.OUT / "D_ewas_per_cpg_hm27.csv.gz"
D2_PER_CPG = D2.OUT / "D2_ewas_per_cpg_hm27.csv.gz"
D2_DRAWS = D2.OUT / "D2_bootstrap_draws_hm27.csv"
D2_FROZEN = D2.OUT / "D2_frozen_endpoints.csv"
D_FROZEN = D.OUT / "D_frozen_analysis_set.csv"
D_CPG_SETS = D.OUT / "D_cpg_sets.csv.gz"
STORED = (D_PER_CPG, D2_PER_CPG, D2_DRAWS, D2_FROZEN, D_FROZEN, D_CPG_SETS)

SCENARIOS = ("depth_10", "depth_2", "depth_5", "depth_30", "gauss_0.05")       # primary first
PRIMARY = "depth_10"
KIND = {"depth_10": ("depth", 10.0), "depth_2": ("depth", 2.0), "depth_5": ("depth", 5.0), "depth_30": ("depth", 30.0),
        "gauss_0.05": ("gauss", 0.05)}
GROUPS = {"primary": ("depth_10",), "secondary": ("depth_2", "depth_5", "depth_30", "gauss_0.05")}
METHODS = ("raw_noisy", "knn_denoised", "methylgpt", "methylgpt_depthaware", "methylgpt_blend")
COMPARATORS = ("raw_noisy", "knn_denoised")
DECISION = "methylgpt"
APPLICATIONS = ("methylgpt_depthaware", "methylgpt_blend")
ENDPOINTS = D2.ALL_ENDPOINTS                       # type 2 diabetes first, then D2's seven
SEVEN = D2.PRIMARY_ENDPOINTS
T2D = D2.CHECK_ENDPOINT
SEED, N_BOOT, FDR = D.SEED, D.N_BOOT, D.FDR       # 20261024, 1,000, 0.05
Z_TOL = 1e-4
DRAW_TOL = 1e-6
VALUE_TOL = 1e-6
N_CHECK_DRAWS = 3
D_REF_ROWS = 145700
CKPT_EVERY = 10
TPB = 128
RULE = ("PASS iff (rho_bar_MethylGPT - rho_bar_best) / rho_bar_best >= 0.10 and the lower end of the 95% bootstrap interval of "
        "rho_bar_MethylGPT - rho_bar_best is > 0; rho_bar = mean over the eight endpoints of the Spearman correlation of per-CpG "
        "Cox z with the full-data z at D's 22,322 target CpGs; best = the better of raw_noisy and knn_denoised on rho_bar")
INTERVAL = ("percentile, 1,000 participant bootstrap draws (default_rng(20261024), D2's draws), one resample per draw shared by "
            "all endpoints, scenarios and methods; full data and every method refitted in each draw; difference of endpoint means "
            "within each draw")
CAVEAT_INTERVAL = ("D's README (to which D2's README defers): the primary interval, of the difference in Spearman correlation, "
                   "compares versions within each draw; bootstrap medians of single correlations are not centred on their point "
                   "estimates (D, 27K: MethylGPT 0.62 against 0.67, nearest neighbours 0.33 against 0.31), so percentile intervals "
                   "of single correlations are reported but the decision uses the interval of the difference")
CAVEAT_FDR = ("D2's caveat: FDR on reconstructed or denoised values is not calibrated (values share information across CpGs and "
              "participants); hit counts are descriptive and no new hit is claimed")


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def committed():
    rel = str(Path(__file__).resolve().relative_to(HERE))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=HERE, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=HERE).returncode == 0
    return tracked and clean


def require_committed():
    if not committed():
        sys.exit("Cannot proceed - the script must be committed unchanged before any stage runs")


def git_unchanged(path):
    rel = str(Path(path).resolve().relative_to(HERE))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=HERE, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=HERE).returncode == 0
    return tracked, clean


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_ids(ids):
    return hashlib.sha256("\n".join(np.sort(np.asarray(ids, str))).encode()).hexdigest()


def resample_hash(cnt):
    return "h" + hashlib.sha256(cnt.astype(np.int64).tobytes()).hexdigest()[:16]      # D2's convention


def atomic_csv(df, path, **kw):
    """Write a CSV atomically; *.gz is written with gzip explicitly (D2 wrote plain CSV under .gz names)."""
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    if path.name.endswith(".gz"):
        kw = {**kw, "compression": "gzip"}
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    write_json(tmp, obj)
    os.replace(tmp, path)


def is_gzip(path):
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


class Checks:
    """Checks table of one stage; aborts on the first failure (after writing the table)."""

    def __init__(self, stage):
        self.stage, self.rows = stage, []
        self.path = WORK / f"checks_{stage}.json"

    def add(self, name, ok, expected="", observed="", note=""):
        self.rows.append({"stage": self.stage, "check": name, "pass": bool(ok), "expected": str(expected), "observed": str(observed),
                          "note": note})
        atomic_json(self.path, self.rows)
        if not ok:
            sys.exit(f"check failed ({self.stage}): {name}; expected {expected}, observed {observed} {note}")


def stage_passed(stage):
    p = WORK / f"checks_{stage}.json"
    if not p.exists():
        return False
    rows = json.loads(p.read_text())
    return bool(rows) and all(r["pass"] for r in rows) and (WORK / f"done_{stage}.json").exists()


def require_stage(stage):
    if not stage_passed(stage):
        sys.exit(f"Cannot proceed - stage {stage} has not passed (see {WORK / f'checks_{stage}.json'})")


def mark_done(stage, **info):
    atomic_json(WORK / f"done_{stage}.json", {"stage": stage, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **info})


def target_info():
    sets = D.load_sets()
    tgt = sets["target_hm27"].to_numpy(bool)
    return sets, tgt, np.flatnonzero(tgt), sets.cpg_id.astype(str).to_numpy()[tgt]


def stored_reference():
    """Stored full-data z at the 22,322 targets: dict endpoint -> z (D2's table), CpG order of D's sets."""
    d2 = pd.read_csv(require(D2_PER_CPG), usecols=["endpoint", "cpg_id", "full_z"])
    _, _, _, tid = target_info()
    ref = {}
    for e in ENDPOINTS:
        de = d2[d2.endpoint == e]
        if not np.array_equal(de.cpg_id.to_numpy(str), tid):
            raise ValueError(f"{e}: D2's per-CpG CpG order differs from D's target set")
        ref[e] = de.full_z.to_numpy(float)
    return ref


def noise_arrays(scen):
    kind = KIND[scen][0]
    return ("vocab", "clock", "reads_vocab", "reads_clock") if kind == "depth" else ("vocab", "clock")


def load_noisy(scen, verify=True):
    """Noisy vocabulary matrix (and reads) of one scenario, SHA-256 verified against the noise summary."""
    summ = pd.read_csv(require(NOISE_SUMMARY)).set_index("scenario")
    with np.load(require(NOISE_PRIV / f"private_noisy_{scen}.npz")) as z:
        V = z["vocab"]
        R = z["reads_vocab"] if KIND[scen][0] == "depth" else None
    if verify:
        if sha_arr(V) != summ.loc[scen, "sha256_vocab"]:
            sys.exit(f"{scen}: SHA-256 of the noisy vocabulary matrix differs from the noise summary")
        if R is not None and sha_arr(R) != summ.loc[scen, "sha256_reads_vocab"]:
            sys.exit(f"{scen}: SHA-256 of the read counts differs from the noise summary")
    return V, R


def depth_variance(scen, V, R):
    """Measurement variance v of each noisy value (float64; inf where no read / unobserved)."""
    kind, par = KIND[scen]
    obs = np.isfinite(V)
    if kind == "depth":
        n = R.astype(np.float64)
        m = np.rint(np.where(obs, V.astype(np.float64), 0.0) * n)
        p = (m + 0.5) / (n + 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(obs & (n > 0), p * (1 - p) / np.maximum(n, 1.0), np.inf)
    else:
        v = np.where(obs, par ** 2, np.inf)
    return v


# ---------------------------------------------------------------------------------------------------------------------
# FastCox: D's per-CpG Cox model, one GPU kernel pass per evaluation (numba.cuda, float64)
# ---------------------------------------------------------------------------------------------------------------------
_K = {}


def _kernels():
    if _K:
        return _K
    import math
    from numba import cuda

    @cuda.jit
    def std_kernel(X, w, sw, mx, sd):
        j = cuda.grid(1)
        if j < X.shape[1]:
            s = 0.0
            for i in range(X.shape[0]):
                s += w[i] * X[i, j]
            mean = s / sw
            q = 0.0
            for i in range(X.shape[0]):
                d = X[i, j] - mean
                q += w[i] * d * d
            mx[j] = mean
            sd[j] = math.sqrt(q / (sw - 1.0))

    @cuda.jit
    def cox_kernel(X, mx, sx, Z1, Z2, w, ew, blk, W, th, active, grad, out):
        j = cuda.grid(1)
        if j >= X.shape[1]:
            return
        if active[j] == 0:
            return
        t0 = th[j, 0]
        t1 = th[j, 1]
        t2 = th[j, 2]
        mj = mx[j]
        sj = sx[j]
        c = 0.0
        started = 0
        s0 = 0.0
        sxv = 0.0
        sa = 0.0
        sb = 0.0
        sxx = 0.0
        sxa = 0.0
        sxb = 0.0
        saa = 0.0
        sab = 0.0
        sbb = 0.0
        e_eta = 0.0
        e_x = 0.0
        acc = 0.0
        ux = 0.0
        ua = 0.0
        ub = 0.0
        ixx = 0.0
        ixa = 0.0
        ixb = 0.0
        iaa = 0.0
        iab = 0.0
        ibb = 0.0
        for i in range(X.shape[0]):
            x = (X[i, j] - mj) / sj
            a = Z1[i]
            b = Z2[i]
            eta = t0 * x + t1 * a + t2 * b
            wi = w[i]
            if wi > 0.0:
                if started == 0:
                    c = eta
                    started = 1
                elif eta > c:
                    f = math.exp(c - eta)
                    s0 *= f
                    if grad != 0:
                        sxv *= f
                        sa *= f
                        sb *= f
                        sxx *= f
                        sxa *= f
                        sxb *= f
                        saa *= f
                        sab *= f
                        sbb *= f
                    c = eta
                r = wi * math.exp(eta - c)
                s0 += r
                if grad != 0:
                    rx = r * x
                    sxv += rx
                    sa += r * a
                    sb += r * b
                    sxx += rx * x
                    sxa += rx * a
                    sxb += rx * b
                    saa += r * a * a
                    sab += r * a * b
                    sbb += r * b * b
            e = ew[i]
            if e > 0.0:
                e_eta += e * eta
                e_x += e * x
            k = blk[i]
            if k >= 0:
                wk = W[k]
                if wk > 0.0:
                    acc += wk * (math.log(s0) + c)
                    if grad != 0:
                        inv = 1.0 / s0
                        mx_ = sxv * inv
                        ma = sa * inv
                        mb = sb * inv
                        ux += wk * mx_
                        ua += wk * ma
                        ub += wk * mb
                        ixx += wk * (sxx * inv - mx_ * mx_)
                        ixa += wk * (sxa * inv - mx_ * ma)
                        ixb += wk * (sxb * inv - mx_ * mb)
                        iaa += wk * (saa * inv - ma * ma)
                        iab += wk * (sab * inv - ma * mb)
                        ibb += wk * (sbb * inv - mb * mb)
        out[j, 0] = e_eta - acc
        if grad != 0:
            out[j, 1] = e_x - ux
            out[j, 2] = -ua
            out[j, 3] = -ub
            out[j, 4] = ixx
            out[j, 5] = ixa
            out[j, 6] = ixb
            out[j, 7] = iaa
            out[j, 8] = iab
            out[j, 9] = ibb

    _K["std"], _K["cox"] = std_kernel, cox_kernel
    return _K


class FastCox:
    """D's CoxBatch model (CpG, age, sex; Breslow; float64; frequency weights) with a one-pass GPU kernel per evaluation."""

    def __init__(self, duration, event, age, sex, dev):
        import torch
        self.t, self.dev = torch, dev
        self.order = np.argsort(-np.asarray(duration, float), kind="stable")          # as D.CoxBatch
        neg = -np.asarray(duration, float)[self.order]
        end = np.searchsorted(neg, neg, side="right") - 1                             # last row of each tie block
        evs = np.asarray(event)[self.order] > 0
        evpos = np.flatnonzero(evs)
        ends = np.unique(end[evpos])
        blk = np.full(len(neg), -1, np.int32)
        blk[ends] = np.arange(len(ends), dtype=np.int32)
        self.blk = torch.as_tensor(blk, device=dev)
        self.n_blocks = len(ends)
        self.evpos = torch.as_tensor(evpos, device=dev)
        self.evblk = torch.as_tensor(np.searchsorted(ends, end[evpos]), device=dev)
        self.evf = torch.as_tensor(evs.astype(np.float64), device=dev)
        self.age = torch.as_tensor(np.asarray(age, float)[self.order], dtype=torch.float64, device=dev)
        self.sex = torch.as_tensor(np.asarray(sex, float)[self.order], dtype=torch.float64, device=dev)
        self.n = len(neg)

    def sort(self, M):
        return self.t.as_tensor(np.ascontiguousarray(M[self.order]), device=self.dev)

    def weights(self, w=None):
        w = np.ones(self.n) if w is None else np.asarray(w, float)
        return self.t.as_tensor(w[self.order], dtype=self.t.float64, device=self.dev)

    @staticmethod
    def _grid(m):
        return (m + TPB - 1) // TPB

    def fit(self, Xs, w, th0=None, max_iter=50, tol=1e-9, halvings=12):
        """Xs: (n, m) float32 tensor in time order; w: (n,) float64 weights in time order; th0: optional (m, 3) start.
        Returns numpy beta, se, z, converged, sd and the (m, 3) float64 tensor of estimates (standardized units)."""
        t = self.t
        K = _kernels()
        n, m = Xs.shape
        f64 = dict(dtype=t.float64, device=self.dev)
        sw = float(w.sum())
        mx, sd = t.empty(m, **f64), t.empty(m, **f64)
        K["std"][self._grid(m), TPB](Xs, w, sw, mx, sd)
        ok = sd > 1e-8
        sx = t.where(ok, sd, t.ones_like(sd))
        swt = w.sum()
        ma = (w * self.age).sum() / swt
        sa = t.sqrt((w * (self.age - ma) ** 2).sum() / (swt - 1))
        ms = (w * self.sex).sum() / swt
        ss = t.sqrt((w * (self.sex - ms) ** 2).sum() / (swt - 1))
        Z1 = ((self.age - ma) / sa).contiguous()
        Z2 = ((self.sex - ms) / ss).contiguous()
        ew = (w * self.evf).contiguous()
        W = t.zeros(self.n_blocks, **f64)
        W.index_add_(0, self.evblk, w[self.evpos])
        ev1, ev2 = (ew * Z1).sum(), (ew * Z2).sum()
        th = t.zeros((m, 3), **f64) if th0 is None else th0.to(t.float64).clone().contiguous()
        act = t.ones(m, dtype=t.int8, device=self.dev)
        out = t.zeros((m, 10), **f64)

        def run(theta, a, grad):
            K["cox"][self._grid(m), TPB](Xs, mx, sx, Z1, Z2, w, ew, self.blk, W, theta.contiguous(), a, 1 if grad else 0, out)

        def unpack():
            U = t.stack([out[:, 1], out[:, 2] + ev1, out[:, 3] + ev2], 1)
            I = t.empty((m, 3, 3), **f64)
            I[:, 0, 0], I[:, 0, 1], I[:, 0, 2] = out[:, 4], out[:, 5], out[:, 6]
            I[:, 1, 0], I[:, 1, 1], I[:, 1, 2] = out[:, 5], out[:, 7], out[:, 8]
            I[:, 2, 0], I[:, 2, 1], I[:, 2, 2] = out[:, 6], out[:, 8], out[:, 9]
            return out[:, 0].clone(), U, I

        run(th, act, True)
        ll, U, I = unpack()
        last = t.full((m,), float("inf"), **f64)
        it = 0
        for it in range(1, max_iter + 1):
            a = act.bool()
            delta, info = t.linalg.solve_ex(I, U)
            delta = t.where((info == 0)[:, None] & t.isfinite(delta) & a[:, None], delta, t.zeros_like(delta))
            step = t.ones(m, **f64)
            pend = act.clone()
            for _ in range(halvings):
                run(th + step[:, None] * delta, pend, False)
                lln = out[:, 0]
                bad = pend.bool() & (~t.isfinite(lln) | (lln < ll - 1e-12 * ll.abs()))
                if not bool(bad.any()):
                    break
                step = t.where(bad, step / 2, step)
                pend = bad.to(t.int8)
            upd = step[:, None] * delta
            th = t.where(a[:, None], th + upd, th)
            last = t.where(a, upd.abs().max(1).values, last)
            run(th, act, True)
            ll2, U2, I2 = unpack()
            ll = t.where(a, ll2, ll)
            U = t.where(a[:, None], U2, U)
            I = t.where(a[:, None, None], I2, I)
            act = (a & (last >= tol)).to(t.int8)
            if not bool(act.any()):
                break
        cov, info = t.linalg.inv_ex(I)
        se = t.sqrt(cov[:, 0, 0])
        beta = th[:, 0]
        z = beta / se
        conv = ok & (info == 0) & (last < 1e-6) & t.isfinite(z) & t.isfinite(se)
        nan = t.full_like(z, float("nan"))
        self.last_iterations = it
        return (t.where(conv, beta, nan).cpu().numpy(), t.where(conv, se, nan).cpu().numpy(), t.where(conv, z, nan).cpu().numpy(),
                conv.cpu().numpy(), sd.cpu().numpy(), t.where(conv[:, None], th, t.zeros_like(th)))


class D2Cox:
    """Fallback (--cox d2): D's CoxBatch as D2 ran it (12,000 CpGs per batch, start at 0); same interface as FastCox."""

    def __init__(self, duration, event, age, sex, dev, batch=12000):
        self.c = D.CoxBatch(duration, event, age, sex, dev)
        self.order, self.sort, self.weights, self.batch = self.c.order, self.c.sort, self.c.weights, batch

    def fit(self, Xs, w, th0=None):
        import torch
        b, s, z, c, sd = self.c.fit_all(Xs, w, self.batch)
        return b, s, z, c, sd, torch.zeros((Xs.shape[1], 3), dtype=torch.float64)


def make_cox(kind, x, dev):
    if kind == "fast":
        return FastCox(x["duration"], x["event"], x["age"], x["sex"], dev)
    return D2Cox(x["duration"], x["event"], x["age"], x["sex"], dev)


def gpu_setup(device):
    import torch
    from methylgpt_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(device)
    if dev.type == "cuda":
        from numba import cuda
        cuda.select_device(dev.index or 0)
    return torch, dev


# ---------------------------------------------------------------------------------------------------------------------
# stage inputs (CPU): checks 1 and 2
# ---------------------------------------------------------------------------------------------------------------------
def stage_inputs(a):
    require_committed()
    WORK.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    ck = Checks("inputs")
    t0 = time.time()
    while not (NOISE_PRIV / "done_noise.json").exists():
        if time.time() - t0 > a.wait_hours * 3600:
            sys.exit("Cannot proceed - required data file not found: " + str(NOISE_PRIV / "done_noise.json"))
        log("waiting for the noise done marker")
        time.sleep(60)
    rec = {"label": LABEL, "noise": {}, "stored": {}}
    # ---- 1 noise input -----------------------------------------------------------------------------------------------
    same = subprocess.run(["git", "diff", "--quiet", NOISE_COMMIT, "HEAD", "--", NOISE_SCRIPT.name], cwd=HERE).returncode == 0
    clean = git_unchanged(NOISE_SCRIPT)
    ck.add("1a noise script unchanged since 729f99e and clean", same and all(clean), "unchanged", f"{same} {clean}")
    nc = pd.read_csv(require(NOISE_CHECKS))
    ck.add("1b noise checks table all pass", bool(nc["pass"].all()), "all pass", f"{int(nc['pass'].sum())}/{len(nc)}")
    summ = pd.read_csv(require(NOISE_SUMMARY)).set_index("scenario")
    ck.add("1c noise summary covers the five N scenarios with N's label", set(SCENARIOS) <= set(summ.index) and
           all(summ.loc[s, "label"] == LABEL for s in SCENARIOS), LABEL, list(summ.label.unique()))
    g = D.load_gs()
    X = g["X"]
    J = _load("large_gain_J", "large_gain_J_gs_missing_cpgs_knn_2026-10-28.py")
    jids, jX = J.load_gs()
    ck.add("1d noise generator's participant order and matrix equal D.load_gs", np.array_equal(jids, g["ids"]) and
           np.array_equal(np.isfinite(jX), np.isfinite(X)) and np.array_equal(np.nan_to_num(jX, nan=-9), np.nan_to_num(X, nan=-9)),
           "equal", "equal" if np.array_equal(jids, g["ids"]) else "differ")
    del jX
    gsobs = np.isfinite(X)
    for scen in SCENARIOS:
        with np.load(require(NOISE_PRIV / f"private_noisy_{scen}.npz")) as z:
            arrs = {k: z[k] for k in z.files}
        names = noise_arrays(scen)
        ck.add(f"1e {scen}: arrays present", set(names) <= set(arrs), names, sorted(arrs))
        hs = {k: sha_arr(arrs[k]) for k in names}
        ok = all(hs[k] == summ.loc[scen, f"sha256_{k}"] for k in names)
        ck.add(f"1f {scen}: SHA-256 of every array equals the noise summary", ok, {k: summ.loc[scen, f"sha256_{k}"] for k in names}, hs)
        V = arrs["vocab"]
        ck.add(f"1g {scen}: vocab shape and dtype", V.shape == X.shape and V.dtype == np.float32, (X.shape, "float32"), (V.shape, V.dtype))
        fin = np.isfinite(V)
        ok_abs = not fin[~gsobs].any()
        if KIND[scen][0] == "depth":
            R = arrs["reads_vocab"]
            ok_pat = R.dtype == np.uint8 and np.array_equal(fin[gsobs], (R > 0)[gsobs])
            nnr = int(((R == 0) & gsobs).sum())
        else:
            ok_pat = bool(fin[gsobs].all())
            nnr = 0
        ck.add(f"1h {scen}: NaN pattern (absent CpGs NaN; no-read values NaN, others finite)", ok_abs and ok_pat, "pattern", f"{ok_abs} {ok_pat}")
        ck.add(f"1i {scen}: values within [0, 1]", bool(((V[fin] >= 0) & (V[fin] <= 1)).all()), "[0, 1]", "")
        rec["noise"][scen] = {"sha256": hs, "file": str(NOISE_PRIV / f"private_noisy_{scen}.npz"),
                              "file_mtime": time.ctime((NOISE_PRIV / f"private_noisy_{scen}.npz").stat().st_mtime),
                              "n_no_read_values_at_gs_cpgs": nnr}
        log(f"{scen}: noise arrays verified")
        del arrs, V, fin
    # ---- 2 D and D2 inputs -------------------------------------------------------------------------------------------
    for p in STORED:
        tr, cl = git_unchanged(p)
        ck.add(f"2a stored table tracked and unchanged: {p.name}", tr and cl, "tracked, clean", f"{tr}, {cl}")
        rec["stored"][str(p.relative_to(HERE))] = {"sha256": D.sha256_file(p), "mtime": time.ctime(p.stat().st_mtime)}
    dz = pd.read_csv(D_PER_CPG, usecols=["cpg_id", "full_z"])
    d2 = pd.read_csv(D2_PER_CPG, usecols=["endpoint", "cpg_id", "full_z"])
    t2 = d2[d2.endpoint == T2D]
    ck.add("2b D2's type 2 diabetes full-data z equal D's", np.array_equal(t2.cpg_id.to_numpy(str), dz.cpg_id.to_numpy(str)) and
           np.array_equal(t2.full_z.to_numpy(float), dz.full_z.to_numpy(float)), "equal", "")
    ref = stored_reference()
    ck.add("2c stored reference: eight endpoints x 22,322 targets, all z defined", all(len(ref[e]) == 22322 and np.isfinite(ref[e]).all()
                                                                                      for e in ENDPOINTS), "8 x 22322 finite",
           {e: int(np.isfinite(ref[e]).sum()) for e in ENDPOINTS})
    E, cross = D2.load_endpoints(g)
    fr = pd.read_csv(D2_FROZEN).set_index("endpoint")
    order = np.argsort(g["ids"])
    for e in ENDPOINTS:
        x = E[e]
        h = hashlib.sha256(np.ascontiguousarray(np.column_stack([x["event"][order], x["duration"][order]])).tobytes()).hexdigest()
        ck.add(f"2d {e}: events and event/duration SHA-256 equal D2's frozen table", int(x["event"].sum()) == int(fr.loc[e, "n_events"])
               and h == fr.loc[e, "sha256_event_duration_in_sorted_id_order"] and sha_ids(g["ids"]) == fr.loc[e, "sha256_sorted_participant_ids"],
               f"{int(fr.loc[e, 'n_events'])} events", f"{int(x['event'].sum())} events")
    sets, tgt, tidx, _ = target_info()
    dfz = pd.read_csv(D_FROZEN).set_index("setting")
    ck.add("2e D's target set: 22,322 hm27 targets, equal to D's frozen count", int(tgt.sum()) == int(dfz.loc["hm27", "n_target_cpgs"]) == 22322,
           22322, int(tgt.sum()))
    ck.add("2f target CpGs observed in every participant", bool(gsobs[:, tgt].all()), "all observed", "")
    rec["endpoint_checks"] = {e: E[e]["checks"] for e in ENDPOINTS}
    rec["cross_checks"] = cross
    rec["participants_sha256"] = sha_ids(g["ids"])
    atomic_json(WORK / "inputs_record.json", rec)
    mark_done("inputs")
    log("inputs stage passed")


# ---------------------------------------------------------------------------------------------------------------------
# stage coxcheck (GPU): check 3 and timing
# ---------------------------------------------------------------------------------------------------------------------
def stage_coxcheck(a):
    require_committed()
    require_stage("inputs")
    torch, dev = gpu_setup(a.device)
    ck = Checks("coxcheck")
    g = D.load_gs()
    X = g["X"]
    E, _ = D2.load_endpoints(g)
    sets, tgt, tidx, tid = target_info()
    ref = stored_reference()
    dref = pd.read_csv(D_PER_CPG, usecols=["cpg_id", "full_z"]).full_z.to_numpy(float)
    Xt = np.ascontiguousarray(X[:, tidx])
    imp = D2.load_imputations(sets, sha_ids(g["ids"]))["hm27"]
    d2d = pd.read_csv(D2_DRAWS, dtype={"resample_sha256_16": str})
    rec = {"per_endpoint": {}}
    n = len(X)
    for e in ENDPOINTS:
        x = E[e]
        cox = FastCox(x["duration"], x["event"], x["age"], x["sex"], dev)
        M = torch.empty((n, 3 * len(tidx)), dtype=torch.float32, device=dev)
        for k, A in enumerate((Xt, imp["methylgpt"], imp["nearest_neighbours"])):
            M[:, k * len(tidx):(k + 1) * len(tidx)] = cox.sort(A)
        T = len(tidx)
        t1 = time.time()
        b, s, z, c, sd, th = cox.fit(M, cox.weights())
        torch.cuda.synchronize()
        tp = time.time() - t1
        zf = z[:T]
        dzmax = float(np.nanmax(np.abs(zf - ref[e])))
        ck.add(f"3a {e}: FastCox full-data z equal D2's stored z within 1e-4 (22,322 targets, same undefined pattern)",
               dzmax <= Z_TOL and np.array_equal(np.isfinite(zf), np.isfinite(ref[e])), f"<= {Z_TOL}", f"{dzmax:.3e}")
        r = {"max_abs_dz_full_vs_D2_stored": dzmax, "point_fit_seconds_3_versions": round(tp, 2), "iterations": cox.last_iterations,
             "n_not_converged_full": int((~c[:T]).sum())}
        if e == T2D:
            dd = float(np.nanmax(np.abs(zf - dref)))
            ck.add("3b type 2 diabetes: FastCox full-data z equal D's stored z within 1e-4", dd <= Z_TOL and np.isfinite(zf).all(),
                   f"<= {Z_TOL}", f"{dd:.3e}")
            r["max_abs_dz_full_vs_D_stored"] = dd
            cb = D.CoxBatch(x["duration"], x["event"], x["age"], x["sex"], dev)
            t1 = time.time()
            zc = cb.fit_all(cb.sort(Xt), cb.weights(), 12000)[2]
            torch.cuda.synchronize()
            r["d_coxbatch_seconds_22322"] = round(time.time() - t1, 2)
            dm = float(np.nanmax(np.abs(zc - zf)))
            r["max_abs_dz_full_vs_D_CoxBatch_in_memory"] = dm
            ck.add("3c type 2 diabetes: FastCox z equal D's CoxBatch in memory within 1e-4", dm <= Z_TOL, f"<= {Z_TOL}", f"{dm:.3e}")
            del cb
        # D2's bootstrap draws 0..N_CHECK_DRAWS-1 on D's hm27 imputations, warm started
        rngb = np.random.default_rng(SEED)
        dd2 = d2d[d2d.endpoint == e].set_index("draw")
        tb = []
        for bb in range(N_CHECK_DRAWS):
            cnt = np.bincount(rngb.integers(0, n, n), minlength=n)
            h = resample_hash(cnt)
            t1 = time.time()
            zb = cox.fit(M, cox.weights(cnt), th0=th)[2]
            torch.cuda.synchronize()
            tb.append(time.time() - t1)
            rm = D.compare(zb[T:2 * T], zb[:T])[0]
            rk = D.compare(zb[2 * T:], zb[:T])[0]
            dm = abs(rm - float(dd2.loc[bb, "methylgpt_rho"]))
            dk = abs(rk - float(dd2.loc[bb, "knn_rho"]))
            ck.add(f"3d {e}: draw {bb} reproduces D2's draw (resample hash; MethylGPT and kNN rho within 1e-6)",
                   h == dd2.loc[bb, "resample_sha256_16"] and dm <= DRAW_TOL and dk <= DRAW_TOL,
                   f"{dd2.loc[bb, 'resample_sha256_16']}, <= {DRAW_TOL}", f"{h}, {dm:.2e}, {dk:.2e}")
        r["bootstrap_seconds_per_draw_3_versions"] = round(float(np.mean(tb)), 3)
        rec["per_endpoint"][e] = r
        log(f"{e}: full-data z max |dz| {dzmax:.2e}; point {tp:.1f} s, draw {np.mean(tb):.2f} s (3 x 22,322 CpGs)")
        del M, cox
        torch.cuda.empty_cache()
    spd = float(np.mean([v["bootstrap_seconds_per_draw_3_versions"] for v in rec["per_endpoint"].values()]))
    n_ver = {"primary": 1 + 5, "secondary": 1 + 4 * 5}
    rec["projection"] = {grp: {"versions_per_draw": nv, "seconds_per_draw_endpoint": spd * nv / 3,
                               "hours_all_endpoints_one_gpu": spd * nv / 3 * N_BOOT * len(ENDPOINTS) / 3600} for grp, nv in n_ver.items()}
    rec["cox_implementation"] = "fast"
    atomic_json(WORK / "coxcheck_record.json", rec)
    mark_done("coxcheck", cox="fast")
    log(f"coxcheck passed; projection {rec['projection']}")


# ---------------------------------------------------------------------------------------------------------------------
# stage values (GPU): checks 4 and 5; method values of every scenario
# ---------------------------------------------------------------------------------------------------------------------
POSTERIOR_W_LINES = ("xc = torch.where(w > 0, x - self.mu, torch.zeros((), device=x.device)) * w",
                     "G = (self.V * w[i, :, None]).T @ self.V")


def posterior_uses_w_multiplicatively(core):
    """Source check of PPCACore.posterior: w appears only in the two multiplicative-weight lines (and the signature/docstring)."""
    src = inspect.getsource(type(core).posterior)
    body = [l.strip() for l in src.splitlines()]
    code = [l for l in body if l and not l.startswith(("@", "def ", '"""', "#"))]
    w_lines = [l for l in code if re.search(r"\bw\b", l)]
    return sorted(w_lines) == sorted(POSTERIOR_W_LINES), w_lines, hashlib.sha256(src.encode()).hexdigest()


def mgpt_values(model, Xm, tidx, dev, W=None, batch=8, return_scores_rows=0):
    """D's factor-model arithmetic (MethylGPT.impute, method factor) at the target CpGs; W: optional posterior weights."""
    import torch
    core = model.core
    tt = torch.as_tensor(tidx, device=dev)
    out = np.empty((len(Xm), len(tidx)), np.float32)
    scores = []
    with torch.no_grad():
        for i in range(0, len(Xm), batch):
            x = torch.from_numpy(Xm[i:i + batch]).to(dev)
            w = torch.isfinite(x).float() if W is None else torch.from_numpy(W[i:i + batch]).to(dev)
            s = core.posterior(x, w)
            if i < return_scores_rows:
                scores.append(s.clone())
            rec = core.predict(s).clamp(0, 1)
            out[i:i + batch] = rec[:, tt].float().cpu().numpy()
    return (out, torch.cat(scores) if scores else None)


def stage_values(a):
    require_committed()
    require_stage("coxcheck")
    torch, dev = gpu_setup(a.device)
    ck = Checks("values")
    VALS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    want = dict(reversed(l.split()) for l in require(D.CHECKSUMS).read_text().splitlines() if l.strip())
    bad = sorted(f for f in want if D.sha256_file(D.BUNDLE / f) != want[f])
    ck.add("5a release bundle checksums", not bad, "all match", bad or "all match")
    g = D.load_gs()
    X = g["X"]
    n = len(X)
    sets, tgt, tidx, _ = target_info()
    keep = sets["kept_hm27"].to_numpy(bool)
    sha = sha_ids(g["ids"])
    dimp = D2.load_imputations(sets, sha)["hm27"]
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt import MethylGPT
    model = MethylGPT.from_pretrained(str(D.BUNDLE), device=str(dev))
    if model.vocab != list(sets.cpg_id.astype(str)):
        sys.exit("bundle vocabulary differs from D's CpG sets")
    sigma2 = float(model.core.sigma2)
    mu = model.core.mu.cpu().numpy().astype(np.float32)
    rec = {"sigma2": sigma2, "scenarios": {}, "label": LABEL}
    # ---- 5 MethylGPT code path ----------------------------------------------------------------------------------------
    Xr = np.where(keep[None, :], X[:64], np.nan).astype(np.float32)
    Xm = model._matrix(pd.DataFrame(Xr, columns=model.vocab))
    mg64, _ = mgpt_values(model, Xm, tidx, dev)
    dmax = float(np.abs(mg64 - dimp["methylgpt"][:64]).max())
    ck.add("5b methylgpt function reproduces D's stored hm27 MethylGPT imputations (first 64 participants)", dmax <= VALUE_TOL,
           f"<= {VALUE_TOL}", f"{dmax:.3e}")
    w_ok, w_lines, src_sha = posterior_uses_w_multiplicatively(model.core)
    rec["posterior_source_sha256"] = src_sha
    rec["posterior_w_lines"] = w_lines
    # ---- 4 kNN reference ----------------------------------------------------------------------------------------------
    rd = _load("recon_diag", RECON.name)
    from methylgpt_lib import Corpus
    free, total = torch.cuda.mem_get_info(dev)
    log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
    corpus = Corpus()
    train_rows = np.sort(corpus.rows("train"))                                  # D's reference rows (D.stage_impute)
    rows_sha = hashlib.sha256(train_rows.astype(np.int64).tobytes()).hexdigest()
    jrec = json.loads(require(J_KNN_RECORD).read_text())
    ck.add("4a kNN reference rows: D's code, 145,700 rows (J's record of D's reference)", len(train_rows) == D_REF_ROWS ==
           int(jrec["D_reference_rows"]), D_REF_ROWS, len(train_rows))
    Rraw = torch.empty((len(train_rows), X.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, len(train_rows), 8192):
        Rraw[i:i + 8192] = torch.from_numpy(corpus.beta[train_rows[i:i + 8192]]).to(dev)
    mu_h = torch.as_tensor(mu, device=dev).half()
    Xt = torch.from_numpy(X[:1024]).to(dev)
    obs = torch.isfinite(Xt)
    keep_t = torch.from_numpy(keep).to(dev)
    pr = rd.knn_predict_lowmem(Rraw, mu_h, Xt, obs & keep_t[None, :], obs & ~keep_t[None, :], k=10, batch=1024, chunk=8192)
    kd = float(np.abs(np.stack(pr).astype(np.float32) - dimp["nearest_neighbours"][:1024]).max())
    ck.add("4b kNN reference is D's: row SHA-256 recorded; D's stored hm27 fills of the first query batch (1,024) reproduced",
           kd <= VALUE_TOL, f"<= {VALUE_TOL}", f"{kd:.3e}", note=f"row SHA-256 {rows_sha}")
    rec["knn_reference"] = {"rows": int(len(train_rows)), "rows_sha256": rows_sha, "max_abs_diff_D_fills_first_1024": kd,
                            "k": 10, "query_batch": 1024, "reference_chunk": 8192}
    del Xt, obs, pr
    torch.cuda.empty_cache()
    truth = np.ascontiguousarray(X[:, tidx])
    for scen in SCENARIOS:
        t1 = time.time()
        sdir = VALS / scen
        sdir.mkdir(parents=True, exist_ok=True)
        if (sdir / "record.json").exists():
            srec = json.loads((sdir / "record.json").read_text())
            for meth, h in srec["sha256"].items():
                ck.add(f"6a {scen} {meth}: saved values present with their recorded SHA-256 (resumed run)",
                       sha_arr(np.load(sdir / f"{meth}.npy")) == h, h, "")
            log(f"{scen}: values present and verified; skipped")
            rec["scenarios"][scen] = srec
            continue
        V, R = load_noisy(scen)
        acc = []
        vals, srec = {}, {"label": LABEL, "kind": KIND[scen][0], "parameter": KIND[scen][1]}
        # raw_noisy
        xr = V[:, tidx].astype(np.float64)
        nr = ~np.isfinite(xr)
        cm = np.nanmean(xr, axis=0)
        if not np.isfinite(cm).all():
            sys.exit(f"{scen}: a target CpG has no read in any participant")
        vals["raw_noisy"] = np.where(nr, cm[None, :], xr).astype(np.float32)
        srec["n_no_read_values_filled_raw_noisy"] = int(nr.sum())
        # methylgpt
        Xm = model._matrix(pd.DataFrame(V, columns=model.vocab))
        mg, s_locked = mgpt_values(model, Xm, tidx, dev, return_scores_rows=64)
        vals["methylgpt"] = mg
        log(f"{scen}: methylgpt done ({time.time() - t1:.0f} s)")
        # depth-aware (conditions checked first)
        v = depth_variance(scen, V, R)
        obsm = np.isfinite(Xm)
        W1 = np.where(obsm, np.float32(sigma2 / (sigma2 + 0.0)), np.float32(0)).astype(np.float32)
        _, s_w1 = mgpt_values(model, Xm[:64], tidx, dev, W=W1[:64], return_scores_rows=64)
        w1_ok = bool(torch.equal(s_w1, s_locked))
        run_da = w_ok and w1_ok
        reason = None if run_da else ("posterior uses w other than as a multiplicative weight" if not w_ok else
                                      "w = 1 does not reproduce the locked scores")
        srec["depthaware_conditions"] = {"w_multiplicative_in_source": w_ok, "w1_reproduces_locked_scores_first_64": w1_ok,
                                         "run": run_da, "reason_not_run": reason}
        ck.add(f"5c {scen}: depth-aware conditions recorded (run only if both hold)", True, "recorded",
               f"multiplicative {w_ok}, w=1 equal {w1_ok}", note="not a gating check; 'not run' is recorded if a condition fails")
        del W1
        if run_da:
            Wd = np.where(obsm & np.isfinite(v), sigma2 / (sigma2 + np.where(np.isfinite(v), v, 1.0)), 0.0).astype(np.float32)
            vals["methylgpt_depthaware"] = mgpt_values(model, Xm, tidx, dev, W=Wd)[0]
            srec["depthaware_weight_mean_at_observed"] = float(Wd[obsm].mean())
            del Wd
            log(f"{scen}: methylgpt_depthaware done ({time.time() - t1:.0f} s)")
        # blend
        vt = v[:, tidx]
        wt = np.where(np.isfinite(vt), sigma2 / (sigma2 + np.where(np.isfinite(vt), vt, 1.0)), 0.0)
        r = mg.astype(np.float64)
        vals["methylgpt_blend"] = np.where(nr, r, r + wt * (np.where(nr, r, xr) - r)).astype(np.float32)
        srec["blend_weight_mean_at_read_target_values"] = float(wt[~nr].mean())
        del Xm, v, vt, wt, r, xr
        # knn_denoised
        Xq = torch.from_numpy(V).to(dev)
        ctx = torch.isfinite(Xq)
        masked = torch.zeros_like(ctx)
        masked[:, torch.as_tensor(tidx, device=dev)] = True
        kn = rd.knn_predict_lowmem(Rraw, mu_h, Xq, ctx, masked, k=10, batch=1024, chunk=8192)
        vals["knn_denoised"] = np.stack(kn).astype(np.float32)
        del Xq, ctx, masked, kn
        torch.cuda.empty_cache()
        log(f"{scen}: knn_denoised done ({time.time() - t1:.0f} s)")
        hashes = {}
        for meth in METHODS:
            if meth not in vals:
                srec[f"{meth}_status"] = "not run"
                continue
            A = vals[meth]
            ck.add(f"6a {scen} {meth}: shape, finite, within [0, 1]", A.shape == (n, len(tidx)) and bool(np.isfinite(A).all()) and
                   bool(((A >= 0) & (A <= 1)).all()), (n, len(tidx)), A.shape)
            np.save(sdir / f"{meth}.npy", A)
            hashes[meth] = sha_arr(A)
            acc.append({"scenario": scen, "method": meth, "label": LABEL, "n_participants": n, "n_target_cpgs": len(tidx),
                        **D.accuracy(A, truth)})
        srec["sha256"] = hashes
        srec["participants_sha256"] = sha
        srec["target_idx_sha256"] = sha_arr(tidx.astype(np.int64))
        srec["seconds"] = round(time.time() - t1, 1)
        atomic_csv(pd.DataFrame(acc), sdir / "accuracy.csv")
        atomic_json(sdir / "record.json", srec)
        rec["scenarios"][scen] = srec
        log(f"{scen}: values written ({srec['seconds']} s)")
        del vals, V, R
    atomic_csv(pd.concat([pd.read_csv(require(VALS / s_ / "accuracy.csv")) for s_ in SCENARIOS], ignore_index=True),
               WORK / "value_accuracy.csv")
    rec["elapsed_s"] = round(time.time() - t0, 1)
    atomic_json(WORK / "values_record.json", rec)
    mark_done("values")
    log("values stage passed")


# ---------------------------------------------------------------------------------------------------------------------
# stage fit (GPU workers)
# ---------------------------------------------------------------------------------------------------------------------
def versions(group):
    out = []
    for scen in GROUPS[group]:
        r = json.loads(require(VALS / scen / "record.json").read_text())
        for meth in METHODS:
            if meth in r["sha256"]:
                out.append((scen, meth, r["sha256"][meth]))
    return out


def fit_one(e, group, cox, X, tidx, ref, a, torch, dev):
    n, T = len(X), len(tidx)
    ver = versions(group)
    tag = f"{group}_{e}"
    M = torch.empty((n, (1 + len(ver)) * T), dtype=torch.float32, device=dev)
    M[:, :T] = cox.sort(np.ascontiguousarray(X[:, tidx]))
    for k, (scen, meth, h) in enumerate(ver):
        A = np.load(VALS / scen / f"{meth}.npy")
        if sha_arr(A) != h:
            sys.exit(f"{scen} {meth}: values differ from their recorded SHA-256")
        M[:, (k + 1) * T:(k + 2) * T] = cox.sort(A)
        del A
    ppath = WORK / f"point_{tag}.npz"
    if ppath.exists():
        P = np.load(ppath)
        th = torch.as_tensor(P["theta"], device=dev)
        log(f"{e} {group}: point estimates present")
    else:
        t1 = time.time()
        b, s, z, c, sd, th = cox.fit(M, cox.weights())
        dz = float(np.nanmax(np.abs(z[:T] - ref[e])))
        if not (dz <= Z_TOL and np.isfinite(z[:T]).all()):
            sys.exit(f"{e} {group}: refitted full-data z differ from the stored z ({dz:.2e})")
        np.savez(ppath, beta=b, se=s, z=z, converged=c, sd=sd, theta=th.cpu().numpy(), max_abs_dz_full_vs_stored=dz,
                 versions=np.array([f"{s_}|{m_}" for s_, m_, _ in ver]))
        log(f"{e} {group}: point fits of {M.shape[1]:,} CpG columns in {time.time() - t1:.0f} s; full-data refit |dz| {dz:.1e}")
    part = WORK / f"draws_{tag}.csv"
    rows = pd.read_csv(part, dtype={"resample_sha256_16": str}).to_dict("records") if part.exists() else []
    done = sorted({int(r["draw"]) for r in rows})
    k_done = len(done)
    if done != list(range(k_done)) or len(rows) != k_done * len(ver):
        sys.exit(f"{e} {group}: checkpoint is not a contiguous set of complete draws")
    stored = {int(r["draw"]): str(r["resample_sha256_16"]) for r in rows}
    rngb = np.random.default_rng(SEED)
    tb, n_new = time.time(), 0
    for bb in range(N_BOOT):
        cnt = np.bincount(rngb.integers(0, n, n), minlength=n)
        h = resample_hash(cnt)
        if bb < k_done:
            if stored[bb] != h:
                sys.exit(f"{e} {group}: stored resample hash of draw {bb} differs")
            continue
        zb = cox.fit(M, cox.weights(cnt), th0=th)[2]
        zf = zb[:T]
        nf = int((~np.isfinite(zf)).sum())
        for k, (scen, meth, _) in enumerate(ver):
            zv = zb[(k + 1) * T:(k + 2) * T]
            rows.append({"endpoint": e, "group": group, "scenario": scen, "method": meth, "draw": bb, "resample_sha256_16": h,
                         "rho": D.compare(zv, zf)[0], "n_not_converged": int((~np.isfinite(zv)).sum()), "full_not_converged": nf})
        n_new += 1
        if (bb + 1) % CKPT_EVERY == 0 or bb == N_BOOT - 1:
            atomic_csv(pd.DataFrame(rows), part)
        if (bb + 1) % 50 == 0 or n_new == 1:
            log(f"{e} {group}: draw {bb + 1}/{N_BOOT} ({(time.time() - tb) / n_new:.2f} s per draw)")
    atomic_json(WORK / f"done_fit_{tag}.json", {"endpoint": e, "group": group, "n_boot": N_BOOT, "draws_this_run": n_new,
                                                "seconds_this_run": round(time.time() - tb, 1), "columns": int(M.shape[1]),
                                                "device": os.environ.get("CUDA_VISIBLE_DEVICES", a.device), "cox": a.cox,
                                                "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    del M
    torch.cuda.empty_cache()


def stage_fit_worker(a):
    require_committed()
    for s in ("inputs", "coxcheck", "values"):
        require_stage(s)
    torch, dev = gpu_setup(a.device)
    g = D.load_gs()
    X = g["X"]
    E, _ = D2.load_endpoints(g)
    _, _, tidx, _ = target_info()
    ref = stored_reference()
    endpoints = [e for e in a.endpoints.split(",") if e]
    for group in [x for x in a.groups.split(",") if x]:
        for e in endpoints:
            if (WORK / f"done_fit_{group}_{e}.json").exists():
                log(f"{e} {group}: done-marker present; skipped")
                continue
            D2.wait_for_gpu(torch, dev, a.min_free_gb)
            cox = make_cox(a.cox, E[e], dev)
            fit_one(e, group, cox, X, tidx, ref, a, torch, dev)
            del cox
    log(f"worker {os.environ.get('CUDA_VISIBLE_DEVICES')} finished")


def stage_fit(a):
    """One worker per device (CUDA_VISIBLE_DEVICES), fixed contiguous shares of the endpoint order; primary group first."""
    require_committed()
    LOGS.mkdir(parents=True, exist_ok=True)
    cox = json.loads(require(WORK / "done_coxcheck.json").read_text()).get("cox", "fast") if a.cox == "auto" else a.cox
    devs = [d for d in a.devices.split(",") if d]
    shares = np.array_split(np.array(ENDPOINTS), len(devs))
    procs = []
    for d, sh in zip(devs, shares):
        lp = LOGS / f"fit_gpu{d}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": d}
        cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "fit-worker", "--device", "cuda:0", "--endpoints", ",".join(sh),
               "--groups", a.groups, "--cox", cox, "--min-free-gb", str(a.min_free_gb)]
        fh = open(lp, "a")
        procs.append((d, list(sh), subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(HERE), env=env), fh))
        log(f"started worker on GPU {d} for {list(sh)} (log {lp})")
    failed = False
    while True:
        st = [p.poll() for _, _, p, _ in procs]
        if any(rc not in (None, 0) for rc in st):
            failed = True
            for _, _, p, _ in procs:
                if p.poll() is None:
                    p.terminate()
            for _, _, p, _ in procs:
                p.wait()
            break
        if all(rc == 0 for rc in st):
            break
        time.sleep(30)
    for d, sh, p, fh in procs:
        fh.close()
        log(f"worker GPU {d} ({sh}) exit code {p.returncode}")
    if failed:
        sys.exit("a fit worker failed; see the logs")
    log("all fit workers finished")


# ---------------------------------------------------------------------------------------------------------------------
# stage analyze (CPU)
# ---------------------------------------------------------------------------------------------------------------------
def rule_row(rm, rc, dm, dc):
    """Rule arithmetic: rm, dm MethylGPT (point, draws); rc, dc dicts of comparator point values and draws."""
    best = max(rc, key=lambda k: rc[k])
    rb, db = rc[best], dc[best]
    diff = dm - db
    dci = D.pct(diff)
    if rb > 0:
        rel = (rm - rb) / rb
        rel_ci = D.pct(diff / db)
        passed = bool(rel >= 0.10 and dci[0] is not None and dci[0] > 0)
        outcome = "PASS" if passed else "FAIL"
    else:
        rel, rel_ci, passed, outcome = None, [None, None], False, "not applicable (rho_best <= 0; fail)"
    return {"best_comparator": best, "best_comparator_value": rb, "best_comparator_ci95_low": D.pct(db)[0],
            "best_comparator_ci95_high": D.pct(db)[1], "difference": rm - rb, "difference_ci95_low": dci[0],
            "difference_ci95_high": dci[1], "bootstrap_p_difference": D.boot_p(diff), "relative_gain": rel,
            "relative_gain_ci95_low": rel_ci[0], "relative_gain_ci95_high": rel_ci[1], "outcome": outcome, "pass": passed}


def suppress_small(df, cols):
    """Event cells below 5 are suppressed (written as '<5'); returns the table and the number suppressed."""
    k = 0
    df = df.copy()
    for c in cols:
        if c in df:
            small = pd.to_numeric(df[c], errors="coerce") < 5
            k += int(small.sum())
            df[c] = df[c].astype(object).where(~small, "<5")
    return df, k


def stage_analyze(a):
    require_committed()
    for s in ("inputs", "coxcheck", "values"):
        require_stage(s)
    OUT.mkdir(parents=True, exist_ok=True)
    ck = Checks("analyze")
    for grp in GROUPS:
        for e in ENDPOINTS:
            require(WORK / f"done_fit_{grp}_{e}.json")
    _, _, tidx, tid = target_info()
    ref = stored_reference()
    fr = pd.read_csv(D2_FROZEN).set_index("endpoint")
    vrec = json.loads(require(WORK / "values_record.json").read_text())
    # ---- per-CpG point estimates and point rho ---------------------------------------------------------------------
    T = len(tidx)
    pz, pb = {}, {}
    full_refit_dz = {}
    for grp in GROUPS:
        for e in ENDPOINTS:
            P = np.load(WORK / f"point_{grp}_{e}.npz")
            full_refit_dz[f"{grp}:{e}"] = float(P["max_abs_dz_full_vs_stored"])
            for k, sv in enumerate(P["versions"].astype(str)):
                scen, meth = sv.split("|")
                pz[(scen, meth, e)] = P["z"][(k + 1) * T:(k + 2) * T]
                pb[(scen, meth, e)] = P["beta"][(k + 1) * T:(k + 2) * T]
    ck.add("6b fit stage's full-data refit equals the stored z within 1e-4 (every endpoint and group)",
           max(full_refit_dz.values()) <= Z_TOL, f"<= {Z_TOL}", f"{max(full_refit_dz.values()):.2e}")
    avail = {scen: [m for m in METHODS if (scen, m, ENDPOINTS[0]) in pz] for scen in SCENARIOS}
    pr = {(scen, m, e): D.compare(pz[(scen, m, e)], ref[e])[0] for scen in SCENARIOS for m in avail[scen] for e in ENDPOINTS}
    # ---- draws -------------------------------------------------------------------------------------------------------
    dr = pd.concat([pd.read_csv(WORK / f"draws_{grp}_{e}.csv", dtype={"resample_sha256_16": str}) for grp in GROUPS for e in ENDPOINTS],
                   ignore_index=True)
    hh = dr.groupby("draw").resample_sha256_16.nunique()
    d2h = pd.read_csv(D2_DRAWS, dtype={"resample_sha256_16": str}).groupby("draw").resample_sha256_16.first()
    first = dr.groupby("draw").resample_sha256_16.first()
    ck.add("7a one resample per draw shared by every endpoint, scenario and method, equal to D2's draw", bool((hh == 1).all()) and
           len(hh) == N_BOOT and bool((first.loc[range(N_BOOT)].to_numpy() == d2h.loc[range(N_BOOT)].to_numpy()).all()),
           "1 per draw, = D2", f"{int(hh.max())} per draw")
    cnt = dr.groupby(["endpoint", "scenario", "method"]).size()
    ck.add("7b 1,000 draws per endpoint, scenario and method", bool((cnt == N_BOOT).all()) and
           len(cnt) == len(ENDPOINTS) * sum(len(avail[s]) for s in SCENARIOS), N_BOOT, f"{int(cnt.min())}-{int(cnt.max())}")
    mat = {}
    for (scen, meth), d in dr.groupby(["scenario", "method"]):
        mat[(scen, meth)] = d.pivot(index="draw", columns="endpoint", values="rho").loc[range(N_BOOT), list(ENDPOINTS)]
    # ---- rule outcome per scenario (eight endpoints) and secondary (seven endpoints) -------------------------------------
    rule_rows, seven_rows, app_rows = [], [], []
    for scen in SCENARIOS:
        for label, eps, out in (("eight endpoints", ENDPOINTS, rule_rows), ("seven endpoints other than type 2 diabetes", SEVEN, seven_rows)):
            pt = {m: float(np.mean([pr[(scen, m, e)] for e in eps])) for m in avail[scen]}
            dd = {m: mat[(scen, m)][list(eps)].mean(1, skipna=False).to_numpy(float) for m in avail[scen]}
            q = rule_row(pt[DECISION], {c: pt[c] for c in COMPARATORS}, dd[DECISION], {c: dd[c] for c in COMPARATORS})
            row = {"analysis": "N", "scenario": scen, "role": "PRIMARY" if scen == PRIMARY else "secondary (pre-specified)",
                   "decision_row": bool(scen == PRIMARY and len(eps) == 8), "label": LABEL, "endpoint_set": label,
                   "n_endpoints": len(eps), "n_participants": int(fr.n_participants.iloc[0]),
                   "n_events_total": int(sum(int(fr.loc[e, "n_events"]) for e in eps)), "n_target_cpgs": T,
                   "primary_metric": "mean over endpoints of the Spearman rho of per-CpG Cox z (method vs full data) at the target CpGs",
                   "methylgpt_value": pt[DECISION], "methylgpt_ci95_low": D.pct(dd[DECISION])[0], "methylgpt_ci95_high": D.pct(dd[DECISION])[1]}
            for c in COMPARATORS:
                row.update({f"{c}_value": pt[c], f"{c}_ci95_low": D.pct(dd[c])[0], f"{c}_ci95_high": D.pct(dd[c])[1]})
            row.update(q)
            row.update({"n_endpoints_methylgpt_above_best": int(sum(pr[(scen, DECISION, e)] > pr[(scen, q["best_comparator"], e)] for e in eps)),
                        "n_bootstrap": N_BOOT, "interval": INTERVAL, "interval_caveat": CAVEAT_INTERVAL,
                        "rule": RULE if len(eps) == 8 else RULE.replace("eight endpoints", label)})
            out.append(row)
            for app in APPLICATIONS:
                st = vrec["scenarios"][scen].get(f"{app}_status")
                if app not in avail[scen]:
                    app_rows.append({"scenario": scen, "application": app, "endpoint_set": label, "status": st or "not run",
                                     "reason": vrec["scenarios"][scen].get("depthaware_conditions", {}).get("reason_not_run")})
                    continue
                qa = rule_row(pt[app], {c: pt[c] for c in COMPARATORS}, dd[app], {c: dd[c] for c in COMPARATORS})
                dmg = dd[app] - dd[DECISION]
                app_rows.append({"scenario": scen, "application": app, "endpoint_set": label, "status": "run", "label": LABEL,
                                 "value": pt[app], "ci95_low": D.pct(dd[app])[0], "ci95_high": D.pct(dd[app])[1],
                                 **{f"vs_{k}": v for k, v in qa.items()},
                                 "difference_vs_methylgpt": pt[app] - pt[DECISION], "difference_vs_methylgpt_ci95_low": D.pct(dmg)[0],
                                 "difference_vs_methylgpt_ci95_high": D.pct(dmg)[1], "bootstrap_p_vs_methylgpt": D.boot_p(dmg),
                                 "note": "application of MethylGPT, outside the decision; rule arithmetic for reference only"})
    rt = pd.DataFrame(rule_rows)
    # ---- per endpoint -----------------------------------------------------------------------------------------------
    per = []
    for scen in SCENARIOS:
        for e in ENDPOINTS:
            dd = {m: mat[(scen, m)][e].to_numpy(float) for m in avail[scen]}
            q = rule_row(pr[(scen, DECISION, e)], {c: pr[(scen, c, e)] for c in COMPARATORS}, dd[DECISION], {c: dd[c] for c in COMPARATORS})
            row = {"scenario": scen, "endpoint": e, "in_seven_endpoint_mean": e in SEVEN, "label": LABEL,
                   "n_events": int(fr.loc[e, "n_events"]), "n_target_cpgs": T}
            for m in avail[scen]:
                row.update({f"{m}_rho": pr[(scen, m, e)], f"{m}_ci95_low": D.pct(dd[m])[0], f"{m}_ci95_high": D.pct(dd[m])[1],
                            f"{m}_rho_bootstrap_median": float(np.median(dd[m])),
                            f"{m}_n_not_converged": int((~np.isfinite(pz[(scen, m, e)])).sum())})
            row.update(q)
            row["rule_note"] = "rule applied to this endpoint alone (secondary, descriptive)"
            per.append(row)
    perdf = pd.DataFrame(per)
    # ---- hit overlap (descriptive) -----------------------------------------------------------------------------------
    hit = []
    for scen in SCENARIOS:
        for m in avail[scen]:
            for e in ENDPOINTS:
                rp = D.recall_precision(pz[(scen, m, e)], ref[e])
                hit.append({"scenario": scen, "method": m, "endpoint": e, "label": LABEL, **rp, "fdr": FDR, "caveat": CAVEAT_FDR})
    hitdf = pd.DataFrame(hit)
    # ---- per-CpG tables ----------------------------------------------------------------------------------------------
    for scen in SCENARIOS:
        frames = []
        for e in ENDPOINTS:
            df = pd.DataFrame({"endpoint": e, "cpg_id": tid, "full_z_stored": ref[e]})
            for m in avail[scen]:
                df[f"{m}_beta"] = pb[(scen, m, e)]
                df[f"{m}_z"] = pz[(scen, m, e)]
            frames.append(df)
        atomic_csv(pd.concat(frames, ignore_index=True), OUT / f"N_ewas_per_cpg_{scen}.csv.gz", float_format="%.6g")
    # ---- write tables ------------------------------------------------------------------------------------------------
    rt, k1 = suppress_small(rt, ["n_events_total"])
    st7, k2 = suppress_small(pd.DataFrame(seven_rows), ["n_events_total"])
    perdf, k3 = suppress_small(perdf, ["n_events"])
    atomic_csv(rt, OUT / "N_rule_outcome.csv")
    atomic_csv(st7, OUT / "N_secondary_seven_endpoints.csv")
    atomic_csv(perdf, OUT / "N_per_endpoint.csv")
    atomic_csv(pd.DataFrame(app_rows), OUT / "N_applications.csv")
    atomic_csv(hitdf, OUT / "N_hit_overlap.csv")
    atomic_csv(dr.drop(columns=["group"]), OUT / "N_bootstrap_draws.csv.gz")
    va = pd.read_csv(require(WORK / "value_accuracy.csv"))
    atomic_csv(va, OUT / "N_value_accuracy.csv")
    nsum = pd.read_csv(NOISE_SUMMARY)
    nsum = nsum[nsum.scenario.isin(SCENARIOS)][["scenario", "role", "kind", "parameter", "label", "participants",
                                                 "fraction_measured_vocabulary_values_with_no_read", "mean_reads_measured_vocabulary",
                                                 "rmse_noisy_vs_measured", "mean_within_participant_pearson_r"]]
    atomic_csv(nsum.assign(role_in_N=lambda d: np.where(d.scenario == PRIMARY, "PRIMARY", "secondary")), OUT / "N_noise_scenarios.csv")
    ck.add("8b event cells below 5 suppressed (endpoint totals only; number suppressed)", True, "suppressed if < 5", k1 + k2 + k3)
    # ---- records -----------------------------------------------------------------------------------------------------
    crec = json.loads(require(WORK / "coxcheck_record.json").read_text())
    fits = {f"{grp}:{e}": json.loads((WORK / f"done_fit_{grp}_{e}.json").read_text()) for grp in GROUPS for e in ENDPOINTS}
    res = {"script": Path(__file__).name, "label": LABEL, "rule": RULE, "interval": INTERVAL, "interval_caveat": CAVEAT_INTERVAL,
           "fdr_caveat": CAVEAT_FDR, "setting_chosen_after": "analyses J, K and L (2 Nov 2026)", "decision": rule_rows[0],
           "rule_rows": rule_rows, "seven_endpoint_rows": seven_rows, "applications": app_rows, "values": vrec,
           "coxcheck": crec, "fits": fits, "full_refit_max_abs_dz": full_refit_dz,
           "inputs": json.loads(require(WORK / "inputs_record.json").read_text())}
    atomic_json(OUT / "N_results.json", res)
    # ---- checks table, disclosure ------------------------------------------------------------------------------------
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.suffix in (".csv", ".gz", ".json") and not p.name.endswith(".tmp"))
    gz_ok = all(is_gzip(p) for p in files if p.name.endswith(".gz"))
    ck.add("8c every *.csv.gz is gzip-compressed", gz_ok, "gzip magic", gz_ok)
    write_checks_table()
    write_readme(rt, perdf)
    g_ids = pd.read_parquet(require(next(D.CACHE.glob(f"COMMON_SAMPLE_SPLIT_{D.SPLIT}_train_all_gs_batches_all_probes_*.labels.parquet"))),
                            columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    files = sorted(p for p in OUT.iterdir() if p.is_file() and not p.name.endswith(".tmp") and p.name != "N_privacy_check.json")
    pc = D2.privacy_check(files, g_ids)
    atomic_json(OUT / "N_privacy_check.json", pc)
    ck.add("8a no participant identifier in any public file", pc["passed"], "0 IDs, 0 barcodes, no ID column",
           f"{sum(r['n_participant_ids_found'] for r in pc['files'])} IDs, {sum(r['n_barcode_pattern_matches'] for r in pc['files'])} barcodes")
    write_checks_table()
    write_readme(rt, perdf)
    for r in rule_rows:
        log(f"{r['scenario']} ({r['role']}): MethylGPT {r['methylgpt_value']:.4f} vs {r['best_comparator']} {r['best_comparator_value']:.4f}; "
            f"difference {r['difference']:.4f} [{r['difference_ci95_low']:.4f}, {r['difference_ci95_high']:.4f}]; gain "
            f"{r['relative_gain'] if r['relative_gain'] is None else round(r['relative_gain'], 4)}; {r['outcome']}")


def write_checks_table():
    rows = []
    for s in ("inputs", "coxcheck", "values", "analyze"):
        p = WORK / f"checks_{s}.json"
        if p.exists():
            rows += json.loads(p.read_text())
    atomic_csv(pd.DataFrame(rows), OUT / "N_checks.csv")


def write_readme(rt, perdf):
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "README.md" and not p.name.endswith(".tmp"))

    def nrows(p):
        return "JSON" if p.suffix == ".json" else f"{len(pd.read_csv(p)):,}"

    def cols(p):
        if p.suffix == ".json":
            return ""
        return ", ".join(pd.read_csv(p, nrows=0).columns)

    desc = {
        "N_rule_outcome.csv": "Rule outcome, one row per scenario (eight-endpoint mean); decision_row = depth_10. MethylGPT, raw_noisy and knn_denoised values with percentile intervals, best comparator, difference with 95% interval and bootstrap P, relative gain with interval, outcome.",
        "N_secondary_seven_endpoints.csv": "Secondary (b): the same quantities over the seven endpoints other than type 2 diabetes (best comparator re-chosen on that mean).",
        "N_per_endpoint.csv": "Secondary (a, f): per scenario and endpoint, rho of every method with intervals and bootstrap median, undefined-z counts, difference against the endpoint's best comparator, rule arithmetic for that endpoint alone.",
        "N_applications.csv": "Secondary (d): methylgpt_depthaware and methylgpt_blend per scenario and endpoint set, value with interval, rule arithmetic against the best comparator (vs_*), difference against methylgpt with interval; status and reason if not run.",
        "N_hit_overlap.csv": "Secondary (c): per scenario, method and endpoint, BH q < 0.05 hits over the 22,322 targets: full-data hits, method hits, shared same-sign hits, recall, precision (descriptive; FDR on reconstructed values not calibrated).",
        "N_value_accuracy.csv": "Secondary (e): value accuracy at the target CpGs against the measured values (D.accuracy) per scenario and method.",
        "N_noise_scenarios.csv": "The five noise scenarios of N as summarised by the shared noise script (no-read fraction, mean reads, RMSE and within-participant r of noisy vs measured values).",
        "N_bootstrap_draws.csv.gz": "One row per draw, endpoint, scenario and method: resample hash (D2's), rho, undefined-z counts of the method and of the draw's full-data refit (aggregate statistics only).",
        "N_checks.csv": "Every check of the inputs, coxcheck, values and analyze stages, with expected and observed values and pass.",
        "N_results.json": "All results and records (rule rows, applications, value records, Cox check, fit records, inputs and hashes).",
        "N_privacy_check.json": "Identifier scan of every public file (D2.privacy_check).",
    }
    for scen in SCENARIOS:
        desc[f"N_ewas_per_cpg_{scen}.csv.gz"] = (f"Per endpoint and target CpG ({scen}): stored full-data z (reference) and, per method, beta "
                                                 "(log HR per SD) and z (6 significant digits; empty = undefined).")
    p0 = rt[rt.decision_row].iloc[0]
    lines = ["# N. Disease EWAS from noisy measurements of every CpG (Generation Scotland)", "",
             f"**{LABEL.capitalize()}.** Source script: `{Path(__file__).name}` (stages inputs, coxcheck, values, fit, analyze); "
             "design, methods, estimator, rule, interval and checks are in its docstring. The setting was chosen after analyses J, K and L.", "",
             "Inputs (Chronos only): D's frame-cache matrix and participants, D2's endpoints, D's 22,322 hm27 target CpGs, the stored "
             "full-data z of D and D2 (`../../large_gain_2026-10-24/D/D_ewas_per_cpg_hm27.csv.gz`, `../../large_gain_2026-10-28/D2/"
             "D2_ewas_per_cpg_hm27.csv.gz`), the shared noisy profiles of `large_gain_noise_profiles_2026-11-02.py` (commit 729f99e; "
             "`artifacts/large_gain_2026-11-02_private/noise/`, hashes in `../noise/noise_summary_2026-11-02.csv`), the released bundle "
             "methylgpt-1.1 and the corpus training split (kNN reference).", "",
             "Participant-level values (method values, working files, logs) stay in `artifacts/large_gain_2026-11-02_private/N/` and are "
             "not committed.", "",
             f"Decision (depth_10, eight endpoints): MethylGPT {p0.methylgpt_value:.4f}, best comparator {p0.best_comparator} "
             f"{p0.best_comparator_value:.4f}; difference {p0.difference:.4f} [{p0.difference_ci95_low:.4f}, {p0.difference_ci95_high:.4f}]; "
             f"relative gain {p0.relative_gain if pd.isna(p0.relative_gain) else round(float(p0.relative_gain), 4)}; {p0.outcome}.", "",
             "Notes", "",
             f"- Interval caveat: {CAVEAT_INTERVAL}.",
             f"- Hit overlap: {CAVEAT_FDR}.",
             "- The noise realisation is fixed (one draw of the shared generator); intervals cover participant resampling only.",
             "- Only the decision_row of `N_rule_outcome.csv` carries the pre-specified decision; secondary scenarios, single endpoints, "
             "the seven-endpoint mean and the applications are reported with the same arithmetic for reference.",
             "",
             "| File | Contents | Columns | Rows |", "|---|---|---|---:|"]
    for p in files:
        lines.append(f"| `{p.name}` | {desc.get(p.name, '')} | {cols(p)} | {nrows(p)} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["inputs", "coxcheck", "values", "fit", "fit-worker", "analyze"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="0,1", help="fit: physical GPU indices, one worker each (speed only)")
    ap.add_argument("--endpoints", default="", help="fit-worker: endpoints")
    ap.add_argument("--groups", default="primary,secondary", help="fit: groups in order (primary first)")
    ap.add_argument("--cox", default="auto", choices=["auto", "fast", "d2"], help="fit: estimator implementation")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--wait-hours", type=float, default=8.0)
    a = ap.parse_args()
    {"inputs": stage_inputs, "coxcheck": stage_coxcheck, "values": stage_values, "fit": stage_fit, "fit-worker": stage_fit_worker,
     "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    main()
