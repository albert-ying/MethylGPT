#!/usr/bin/env python3
"""Large-gain analysis J: Generation Scotland (GS) incident disease when the deployed input misses CpGs, against
nearest-neighbour (kNN) imputation and the existing comparators. SIMULATED MISSINGNESS ON EMPIRICAL PROFILES: every input
is a measured GS development profile from which CpGs are removed by a fixed mask (27K-array CpGs, seeded random masks, or
the observed-CpG pattern of a public sequencing sample); no methylation value is simulated. The sequencing scenarios use
empirical public coverage patterns applied to the GS profiles.

Written on 28 Oct 2026 for the large-gain analysis plan of 23 Oct 2026 (Decision rule) and the analysis brief for
J, BEFORE any result of this analysis is computed. No disease model is refitted: every fold model is reused as the
templates apply it (scenario_gs_combined_2026-10-01.py, scenario_gs_raw_factor_imputed_2026-10-09.py,
scenario_clocks_cox_2026-10-01.py, score_gs_clocks_missingness_2026-10-01.py, bootstrap_scenarios_gs_2026-10-01.py). The
clock Cox models are refitted by scenario_clocks_cox's own deterministic code on the same full-profile scores, as that
template does on every run (checked against the stored risks). Every result is reported, whichever model it favours.

WHY: Fig. 6 (GS incident disease). Existing deployment-with-missing-CpGs results (mean over 8 endpoints of C-index minus
the age+sex C-index): hm27 MethylGPT combined 0.0186 vs GrimAge2 0.0110; random_90 0.0224 vs raw_cpg_gpu (fold-mean
fill) 0.0170. Those comparators fill missing CpGs with means only; in analyses G and D nearest-neighbour imputation
against corpus training profiles was the strongest comparator, so GS needs it, plus an interval for the
mean-over-endpoints difference.

DATA
  GS development participants only: the 9,866 training-subset participants of the locked incident-disease protocol
  (frame-cache matrix COMMON_SAMPLE_SPLIT_full18859_common_task_a_incident_positive_duration_20260629_train_all_gs_
  batches_all_probes_*.matrix.npz, beta values, 49,156 vocabulary columns: 43,899 observed in every participant, 5,257
  absent from the release). Validation and test participants are not read.
  Endpoints, follow-up, age, sex, the age+sex risk (baseline_risk) and the locked outer folds: gs_cox/mgpt2_embed_
  linear_2026-10-01/private_nested_predictions.parquet (age_adjustment "linear"; the order every template uses): 8
  endpoints, 5 outer folds each (1,973-1,974 participants). The folds were assigned separately per endpoint (about 20%
  agreement between endpoints; 9,724 distinct joint fold patterns), which matters for the bootstrap (below).
  Clock CpGs: the matrix written by score_gs_clocks_missingness_2026-10-01.py (gs_clocks/gs_dev_clock_scores_scenarios_
  2026-10-01.clock_cpg_matrix.npz; 1,195 cg CpGs of GrimAgeV2 and DunedinPACE from gs.tar.1.gz, same participant order
  as the frame cache; 1,102 present in GS). 255 are vocabulary CpGs (237 observed in GS, values identical to the frame
  cache), 940 are not (865 observed in GS).

SCENARIOS (kept CpGs = GS-observed CpGs the mask keeps)
  full           complete profiles (existing; reference row and reproduction check).
  hm27           PRIMARY. Existing mask: the vocabulary CpGs of the 27K array (platform_vocab_masks_2026-10-01.npz
                 "hm27": 22,231; 21,577 observed in GS). Clock CpGs: kept iff Methyl27_Loci is TRUE in the EPIC v1.0 B5
                 manifest (score_gs_clocks_missingness; 71 clock CpGs; identical to the vocabulary mask at all 255
                 vocabulary clock CpGs).
  random_90      Existing: Xs[default_rng(20261001 + int(p * 100)).random((9866, 49156)) < p] = NaN, p = 0.90 (seed
                 20261091). Clock CpGs: the same generator on the clock matrix (shape (9866, 1195), seed 20261091), as
                 score_gs_clocks_missingness does; this realisation is independent of the vocabulary mask, as in the
                 existing analysis.
  random_95      New: the same generator and seed scheme, p = 0.95 (seed 20261096), on both matrices. The scheme draws a
                 new uniform matrix for every p, so random_95 is NOT nested in random_90 (the fraction of random_95's
                 kept CpGs that random_90 also keeps is recorded).
  seq_GSE194154, seq_GSE282512, seq_GSE140730   New, empirical public coverage patterns: analysis I's vocabulary
                 matrices (artifacts/large_gain_2026-10-25/work_I/<GSE>/X.npy; I's analysis samples after its sample QC;
                 NaN = fewer than 10 reads, checked against coverage.npy >= 10): GSE194154 RRBS whole blood (110 samples),
                 GSE282512 RRBS plasma cfDNA (279), GSE140730 WGBS umbilical cord blood (128). GS participant i
                 (frame-cache order) is assigned sample a_i = default_rng([20261028, accession number]).integers(0,
                 n_samples, 9866)[i] (with replacement) and keeps the vocabulary CpGs observed in that sample. Clock CpGs:
                 vocabulary clock CpGs follow the same pattern; non-vocabulary clock CpGs are unobserved (the vocabulary
                 matrices hold no coverage for them), so every model receives the same observed CpGs. Sensitivity
                 (reported, outside the decision set): the same clocks with every GS-observed non-vocabulary clock CpG
                 observed (suffix _nvobs), an upper bound for what the clocks could use.
  Masks are hashed: SHA-256 of np.packbits(observed, axis=1) for the vocabulary and clock masks, and of the sequencing
  assignment vector (int64); per-participant kept counts in J_masks.

MODELS (scored on identical participants, endpoints, folds and vocabulary masks)
  MethylGPT combined (mgpt2_combined): scenario_gs_combined unchanged. The age-stage encoder (runs/aging_heads_clean_
  2026-10-01/best.pt) embedding, the multitask biomarker head (runs/multitask_bio_2026-10-01/final.pt) and the factor
  scores and residuals of its core are recomputed from the kept CpGs (batch 32, TF32); the locked fold models (mgpt2_embed_
  linear, mt_bio_linear, factor_residual) give the risks; mgpt2_combined = equal-weight sum of the within-fold
  standardised embedding, biomarker and factor+residual risks. The stored risks of full, hm27 and random_90
  (scenario_combined_2026-10-01/private_scenario_risks.parquet) must be reproduced (CHECKS); every scenario then uses the
  risks computed by this script (one code path).
  Decision comparators (what a user would otherwise do)
    raw_cpg_v12_mean, raw_cpg_gpu_mean   the v12 raw-CpG fold models (locked_risk) and the dense GPU fold models
                 (gpu_risk), missing CpGs = fitting-fold means (existing; scenario_gs_combined).
    raw_cpg_v12_knn, raw_cpg_gpu_knn   the same fold models with the same functions after the masked GS-observed
                 vocabulary CpGs are filled by nearest neighbours: knn_predict_lowmem of si_reconstruction_diagnostics_
                 2026-10-18.py as analyses D, G and I call it; k = 10; squared Euclidean distance over the participant's
                 kept CpGs (TF32 for the search only); reference values unobserved -> mu; prediction = mean of the
                 neighbours' observed values (mu when none); mu = ppca_core_meanfilled_2026-09-30 "mu" (G's and I's).
                 Reference: G's held-out reference (large_gain_G.knn_reference(F, "heldout"): corpus training-split
                 profiles minus F's flagged duplicates, 145,316 rows, float16); the SHA-256 of its row indices (int64) is
                 checked against G/knn_reference_heldout_2026-10-25.json before any query (abort on mismatch). Query
                 batch 512 and reference chunk 8,192 change memory use only. The 5,257 vocabulary CpGs absent from GS stay
                 unobserved and take the fold mean, as in the existing comparator (their coefficients are 0 in every raw
                 fold model). D's private kNN fills (large_gain_2026-10-24_private/D/imputed_hm27.npz) are NOT reused:
                 D's reference is all 145,700 training-split profiles (not G's 145,316) and its mu is the release
                 bundle's; the fills are recomputed (recorded in the check table).
    grimage2_{mean,knn}, dunedinpace_{mean,knn}, grimage2_dunedinpace_{mean,knn}   clock Cox models (below) on clock
                 scores from mean-filled profiles (existing) and from corpus-kNN-filled profiles (new).
  Secondary comparator (reported, outside the decision set): in-cohort kNN (suffix _cohortknn): the same function with
    the reference = the complete profiles (frame cache, float16) of the GS development participants in the training
    folds of the participant's outer fold for that endpoint (never the held-out fold; checked equal to the v12 models'
    fit_ids), applied to both raw fold models and to the three clock models. Because folds are endpoint-specific, each
    participant has one in-cohort fill per endpoint.
  MethylGPT applications (reported, not comparators): raw_cpg_v12_factor_imputed, raw_cpg_gpu_factor_imputed:
    scenario_gs_raw_factor_imputed unchanged (posterior factor scores from the kept CpGs -> mu + V s clipped to [0, 1],
    batch 64, with the core of load_checkpoint(aging checkpoint's base ckpt) loaded separately as that template does),
    every scenario.
  Covariate model: age_sex = the stored age+sex Cox risk (baseline_risk).
  MethylGPT components (embedding, biomarkers, factor_residual, ensemble_embed_bio): per-endpoint table only.

CLOCK SCORING: score_gs_clocks_missingness exactly, except the fill values of masked vocabulary clock CpGs
  Input: the clock matrix M and the clock mask of the scenario (above). Masked and natively missing clock CpGs = the
  development-set mean of the full data (np.nanmean(M, 0)); CpGs absent from GS have a NaN mean and are left to
  biolearn's "averaging" imputation. biolearn 0.7.1 in /home/kying/anaconda3/envs/integrative (the template's
  environment): GrimAgeV2 (column DNAmGrimAge) with age 47 and female = 1, DunedinPACE.
  How non-vocabulary clock CpGs are handled: exactly as score_gs_clocks_missingness, in every fill variant: kept if the
  scenario's clock mask keeps them (hm27: on the 27K array; random_p: the template's random clock mask; sequencing:
  never, except in the _nvobs sensitivity), otherwise the development-set mean. Corpus or in-cohort kNN cannot fill them
  (the references hold vocabulary CpGs only).
  knn fill: a clock CpG that the scenario's clock mask removes, that is a vocabulary CpG and that is observed in GS takes
  its nearest-neighbour value instead of the mean; every other value is as in the template. The neighbours are found with
  the participant's kept vocabulary CpGs as context minus any vocabulary clock CpG that the clock mask removes. For hm27
  and the sequencing scenarios this is the vocabulary context itself (the masks agree) and the fills are the vocabulary
  kNN values; for random_90 and random_95 (separate clock realisation) a second kNN query with that context gives the
  clock fills, so no masked clock CpG informs its own fill.
  Cox models (scenario_clocks_cox exactly): per endpoint and outer fold, unpenalised Cox (CoxPHSurvivalAnalysis,
  ties="breslow", n_iter=200) on (age, sex, score[s]) standardised on the fitting participants, fitted on their stored
  full-profile scores and applied to the held-out participants' scenario scores. Models: grimage2 (GrimAgeV2),
  dunedinpace, grimage2_dunedinpace (both scores).

PRIMARY METRIC AND DECISION (plan, Decision rule for discrimination)
  C(model, e) = pooled within-fold C-index of endpoint e (concordant + tied/2 over comparable pairs summed over e's outer
  folds; sksurv concordance_index_censored arithmetic, tied_tol 1e-8). M(model) = mean over the 8 endpoints of
  (C(model, e) - C(age_sex, e)).
  Decision set (the 4 raw-CpG and 6 clock models above): C* = the decision comparator with the largest M in the primary scenario
  hm27. Relative gain = (M(mgpt2_combined) - M(C*)) / M(C*). PASS iff relative gain >= 0.10 and the lower end of the 95%
  interval of M(mgpt2_combined) - M(C*) is > 0; otherwise FAIL. If M(C*) <= 0 the relative gain is undefined and the rule
  is reported as not applicable. The same quantities are reported for every secondary scenario (with that scenario's
  own C*), against every comparator separately (decision, in-cohort kNN, _nvobs sensitivity, factor-imputed), and per
  endpoint. The complete-profile row (full; comparators raw mean and clocks) is a reference, not a decision.

INTERVALS: paired bootstrap, 2,000 draws, seed 20261028, percentile 95% intervals; statistic recomputed in every draw as
  the mean over endpoints. Bootstrap P = min(1, 2 (k + 1) / (B + 1)), k = min(#{d_b <= 0}, #{d_b >= 0}).
  DEVIATION FROM THE BRIEF, stated before running: the brief asks for resampling within outer folds with one participant
  resample shared by every model and endpoint. Because the outer folds were assigned separately per endpoint, a resample
  stratified by outer fold cannot be shared across endpoints.
  PRIMARY interval (shared resample): draw b resamples the 9,866 participants with replacement, multiplicities
    w_b = bincount(I[b], minlength=9866) with I = default_rng(20261028).integers(0, 9866, size=(2000, 9866)); w_b is
    shared by every model, endpoint and scenario. Per endpoint the pooled within-fold C-index is recomputed over that
    endpoint's outer folds with pair weights w_i w_j (gs_stacked_2026-10-04.PairCounter's arithmetic; a participant and
    its copies are never comparable; per-fold sizes vary between draws).
  SENSITIVITY interval (within-fold, as bootstrap_scenarios_gs and scenario_gs_raw_factor_imputed resample): per endpoint
    k (sorted order) and outer fold f (sorted order), multiplicities from default_rng([20261028, 1, k]).integers(0, n_f,
    size=(2000, n_f)), independent across endpoints, shared by every model and scenario. Reported for every contrast;
    the decision uses the primary interval.

CHECKS (written to J_checks before any new number is used; the bootstrap stage refuses to run unless all pass)
  (1) Recomputed per-endpoint C-indices of full, hm27 and random_90 equal the stored ones within 1e-6: MethylGPT
      combined and components and both raw mean-fill models (scenario_combined_2026-10-01/scenario_cindex.json), both
      factor-imputed raw models (scenario_raw_factor_imputed_2026-10-09_2000draws.csv), the three clock models
      (scenario_clocks_cox_2026-10-01.json) and age_sex (bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09.csv);
      maximum absolute risk differences recorded. (2) Recomputed clock scores of full, hm27 and random_90 equal the
      stored scores within 1e-6. (3) Observed fractions of full, hm27 and random_90 equal the stored ones exactly. (4) kNN
      reference: 145,316 rows, row SHA-256 equal to G's. (5) D's fills not reused (reference differs). (6) Mask hashes
      recorded. (7) Sequencing patterns: finite X equals coverage >= 10. (8) kNN fills leave kept values unchanged and fill
      every masked GS-observed CpG. (9) In-cohort references equal the v12 fit_ids and exclude the held-out fold. (10) The
      bootstrap's pair counting equals sksurv's pooled C-index for every endpoint and model of full, and
      stk.PairCounter's weighted counts on one endpoint-fold for 5 draws.

STAGES (resumable; done-markers artifacts/logs/large_gain_J_<stage>_2026-10-28.done)
  masks (CPU) -> mgpt (GPU) -> knn (GPU) -> cohort (GPU) -> clockin (CPU) -> clocks (biolearn interpreter, no torch) ->
  cox (CPU) -> checks (CPU) -> boot (CPU) -> analyse (CPU). GPU = cuda:0 or cuda:1, whichever has more free memory
  (execution only). Every stage after masks refuses to run unless this script is committed unchanged.
OUTPUTS
  Aggregate tables: artifacts/large_gain_2026-10-28/J/ (README.md, J_masks, J_checks, J_knn_record, J_cindex_per_endpoint,
  J_summary, J_rule_outcome, J_bootstrap_draws, J_results). Participant-level masks, imputed matrices, clock inputs and
  scores, risks and bootstrap weights: artifacts/large_gain_2026-10-28_private/J/ (inside the git-ignored artifacts/,
  never committed, stay on Chronos), files named private_* as the existing private_risks convention.
Run: python3 large_gain_J_gs_missing_cpgs_knn_2026-10-28.py --stage <stage> [--device cuda:N]
     (stage clocks: /home/kying/anaconda3/envs/integrative/bin/python ... --stage clocks [--workers 4])
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATE = "2026-10-28"
OUTJ = ART / "large_gain_2026-10-28/J"
PRIV = ART / "large_gain_2026-10-28_private/J"
LOGS = ART / "logs"
GSCOX = ART / "gs_cox"
CACHE = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/task_a_private_training_20260629/_frame_cache")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
V12 = Path("/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results")
RAW_DIRS = [V12 / f"nested_raw_cpg_{p}_2026-09-28" for p in ("recovered_part1", "linear_part2", "recovered_part3", "linear_part4")]
PROJECT = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT")
EPIC_MANIFEST = PROJECT / "infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip"
MASKS = ART / "platform_vocab_masks_2026-10-01.npz"
CORE_MU = ART / "ppca_core_meanfilled_2026-09-30.npz"
PREDS = GSCOX / "mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet"
MGPT2_RUN = GSCOX / "mgpt2_embed_linear_2026-10-01"
BIO_RUN = GSCOX / "mt_bio_linear_2026-10-01"
FR_DIR = GSCOX / "factor_residual_2026-10-01"
AGING = ART / "runs/aging_heads_clean_2026-10-01/best.pt"
MULTITASK = ART / "runs/multitask_bio_2026-10-01/final.pt"
CLOCK_MATRIX = ART / "gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.clock_cpg_matrix.npz"
CLOCK_SCORES = ART / "gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.csv"
STORED_RISKS = GSCOX / "scenario_combined_2026-10-01/private_scenario_risks.parquet"
STORED_CINDEX = GSCOX / "scenario_combined_2026-10-01/scenario_cindex.json"
STORED_CLOCK_RISKS = GSCOX / "private_clock_scenario_risks_2026-10-01.parquet"
STORED_CLOCK_JSON = GSCOX / "scenario_clocks_cox_2026-10-01.json"
STORED_FI = GSCOX / "scenario_raw_factor_imputed_2026-10-09/private_risks.parquet"
STORED_FI_CSV = GSCOX / "scenario_raw_factor_imputed_2026-10-09_2000draws.csv"
STORED_BOOT_CSV = GSCOX / "bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09.csv"
WORKI = ART / "large_gain_2026-10-25/work_I"
G_SCRIPT = HERE / "large_gain_G_sparse_inputs_2026-10-25.py"
G_REF_JSON = ART / "large_gain_2026-10-25/G/knn_reference_heldout_2026-10-25.json"
D_IMPUTED = ART / "large_gain_2026-10-24_private/D/imputed_hm27.npz"
RECON = HERE / "si_reconstruction_diagnostics_2026-10-18.py"
STK = HERE / "gs_stacked_2026-10-04.py"
BIOLEARN_PY = "/home/kying/anaconda3/envs/integrative/bin/python"

SEED = 20261028
N_BOOT = 2000
KNN_K = 10
KNN_BATCH = 512
KNN_CHUNK = 8192
TOL_C = 1e-6
SEQ = {"GSE194154": "RRBS, whole blood", "GSE282512": "RRBS, plasma cfDNA", "GSE140730": "WGBS, umbilical cord blood"}
SCENARIOS = ["hm27", "random_90", "random_95"] + [f"seq_{a}" for a in SEQ]
ALL_SCEN = ["full"] + SCENARIOS
PRIMARY = "hm27"
STORED_SCEN = ["full", "hm27", "random_90"]
ENDPOINTS = ["Autoimmune_Diseases", "Cancers", "Cardiovascular_Diseases", "Endocrine_and_Metabolic_Diseases",
             "Kidney_Diseases", "Neurological_Diseases", "Respiratory_Diseases", "date_of_death"]
CLOCK_MODELS = {"grimage2": ["GrimAgeV2"], "dunedinpace": ["DunedinPACE"], "grimage2_dunedinpace": ["GrimAgeV2", "DunedinPACE"]}
COMBOS = {"mgpt2_combined": ["embedding", "biomarkers", "factor_residual"], "ensemble_embed_bio": ["embedding", "biomarkers"]}
RAW = ["raw_cpg_v12", "raw_cpg_gpu"]
DECISION = [f"{r}_{f}" for f in ("mean", "knn") for r in RAW] + [f"{c}_{f}" for f in ("mean", "knn") for c in CLOCK_MODELS]
COHORT = [f"{r}_cohortknn" for r in RAW] + [f"{c}_cohortknn" for c in CLOCK_MODELS]
FACTOR = [f"{r}_factor_imputed" for r in RAW]
NVOBS = [f"{c}_{f}_nvobs" for f in ("mean", "knn") for c in CLOCK_MODELS]
COMPONENTS = ["embedding", "biomarkers", "factor_residual", "ensemble_embed_bio"]
OURS = "mgpt2_combined"
LABEL = "simulated missingness on empirical profiles"
RULE = ("PASS iff (M_MethylGPT - M_best) / M_best >= 0.10 and the lower end of the 95% interval of M_MethylGPT - M_best "
        "is > 0; M = mean over the 8 endpoints of (pooled within-fold C-index - age+sex C-index); best = decision "
        "comparator with the largest M in the scenario")


# ------------------------------------------------------------------ helpers (no torch at import: stage clocks)
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def need_glob(d, pattern):
    hits = sorted(Path(d).glob(pattern))
    if len(hits) != 1:
        sys.exit(f"Cannot proceed - required data file not found: {Path(d) / pattern} ({len(hits)} matches)")
    return hits[0]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def atomic_npy(path, arr):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_parquet(path, df):
    tmp = path.with_name(path.stem + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_csv(path, df, **kw):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def done_path(stage):
    return LOGS / f"large_gain_J_{stage}_{DATE}.done"


def done_marker(stage, info=None):
    write_json(done_path(stage), {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})})


def require_done(*stages):
    for s in stages:
        if not done_path(s).exists():
            sys.exit(f"stage {s} has not completed ({done_path(s)} missing)")


def script_committed():
    me = Path(__file__).resolve()
    r = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(me)], capture_output=True)
    if r.returncode:
        return False
    return subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(me)]).returncode == 0


def require_committed():
    if not script_committed():
        sys.exit("Cannot proceed - the script must be committed unchanged before any result-producing stage runs")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, need(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def pick_device(device=None):
    import torch
    if device:
        return torch.device(device)
    free = [torch.cuda.mem_get_info(i)[0] for i in range(min(2, torch.cuda.device_count()))]
    i = int(np.argmax(free))
    log(f"GPU free memory {[round(f / 2**30, 1) for f in free]} GiB; using cuda:{i}")
    return torch.device(f"cuda:{i}")


def load_gs():
    mp = need_glob(CACHE, f"COMMON_SAMPLE_SPLIT_{SPLIT}_train_all_gs_batches_all_probes_*.matrix.npz")
    ids = pd.read_parquet(need(mp.with_name(mp.name.replace(".matrix.npz", ".labels.parquet"))),
                          columns=["dnam_id"]).dnam_id.astype(str).to_numpy()
    with np.load(mp, allow_pickle=False) as z:
        X = z["matrix"].astype(np.float32)
    if X.shape != (9866, 49156) or len(ids) != len(X):
        sys.exit(f"unexpected GS matrix shape {X.shape}")
    return ids, X


def load_preds():
    p = pd.read_parquet(need(PREDS))
    p = p[p.age_adjustment == "linear"]
    out = {}
    for ep in sorted(p.endpoint.unique()):
        pe = p[p.endpoint == ep].reset_index(drop=True)
        pe["dnam_id"] = pe.dnam_id.astype(str)
        out[ep] = pe
    if list(out) != ENDPOINTS:
        sys.exit(f"endpoints differ from the pre-specified list: {list(out)}")
    return out


def load_clock_matrix():
    z = np.load(need(CLOCK_MATRIX), allow_pickle=False)
    return z["ids"].astype(str), z["cpgs"].astype(str), z["beta"].astype(np.float32)


def epic_hm27():
    """score_gs_clocks_missingness.platform_sets()['hm27'] (EPIC v1.0 B5 manifest, Methyl27_Loci)."""
    with zipfile.ZipFile(need(EPIC_MANIFEST)) as z:
        raw = z.read("infinium-methylationepic-v-1-0-b5-manifest-file.csv").decode()
    epic = pd.read_csv(io.StringIO(raw[raw.index("IlmnID,Name,"):].split("[Controls]")[0]), low_memory=False)
    epic["Name"] = epic["Name"].astype(str)
    return set(epic.Name[epic["Methyl27_Loci"].astype(str).str.upper().eq("TRUE")])


def clock_vocab_index(cpgs, vocab):
    vpos = {c: i for i, c in enumerate(vocab)}
    invoc = np.array([c in vpos for c in cpgs])
    ic = np.flatnonzero(invoc)                       # clock-matrix columns of the vocabulary clock CpGs
    vj = np.array([vpos[c] for c in cpgs[invoc]])    # their vocabulary columns
    return invoc, ic, vj


def load_mask(scen, kind="obs"):
    z = np.load(need(PRIV / "masks" / f"private_mask_{scen}.npz"), allow_pickle=False)
    n, v = z[f"{kind}_shape"]
    return np.unpackbits(z[kind], axis=1, count=int(v)).astype(bool)[:int(n)]


def clock_variants():
    v = ["full__mean"]
    for s in SCENARIOS:
        v += [f"{s}__mean", f"{s}__knn"] + [f"{s}__cohortknn__{e}" for e in ENDPOINTS]
        if s.startswith("seq_"):
            v += [f"{s}__mean_nvobs", f"{s}__knn_nvobs"]
    return v


def models_for(scen):
    if scen == "full":
        return ["age_sex", OURS] + COMPONENTS + [f"{r}_mean" for r in RAW] + FACTOR + [f"{c}_mean" for c in CLOCK_MODELS]
    return ["age_sex", OURS] + COMPONENTS + DECISION + COHORT + FACTOR + (NVOBS if scen.startswith("seq_") else [])


def role(m):
    if m == OURS:
        return "MethylGPT"
    if m == "age_sex":
        return "covariate model"
    if m in COMPONENTS:
        return "MethylGPT component"
    if m in DECISION:
        return "decision comparator"
    if m in COHORT:
        return "secondary comparator (in-cohort kNN)"
    if m in FACTOR:
        return "MethylGPT application (factor imputation)"
    if m in NVOBS:
        return "sensitivity (clocks, non-vocabulary clock CpGs observed)"
    return "other"


# ------------------------------------------------------------------ stage masks
def stage_masks():
    t0 = time.time()
    ids, X = load_gs()
    gsobs = np.isfinite(X)
    plat = np.load(need(MASKS))
    vocab = plat["vocab"].astype(str)
    cids, cpgs, M = load_clock_matrix()
    if not np.array_equal(cids, ids):
        sys.exit("clock matrix participants differ from the frame cache")
    invoc, ic, vj = clock_vocab_index(cpgs, vocab)
    if not np.array_equal(np.isfinite(M[:, ic]), gsobs[:, vj]) or \
            np.nanmax(np.abs(M[:, ic] - X[:, vj])) != 0:
        sys.exit("vocabulary clock CpGs differ between the clock matrix and the frame cache")
    hm27 = epic_hm27()
    cfin = np.isfinite(M)
    stored_obs = json.loads(need(STORED_CINDEX).read_text())["observed_fraction"]
    (PRIV / "masks").mkdir(parents=True, exist_ok=True)
    rows, keep_r90 = [], None
    for scen in ALL_SCEN:
        info = {"scenario": scen, "label": "complete profiles" if scen == "full" else LABEL}
        ck_b = None
        if scen == "full":
            keep = np.ones(X.shape, bool)
            ck = np.ones(M.shape, bool)
            info.update(source="none (existing)", seed="")
        elif scen.startswith("random_"):
            p = int(scen.split("_")[1]) / 100
            seed = 20261001 + int(p * 100)
            keep = ~(np.random.default_rng(seed).random(X.shape) < p)
            ck = ~(np.random.default_rng(seed).random(M.shape) < p)
            info.update(source=f"random_p generator of scenario_gs_combined / score_gs_clocks_missingness, p = {p}"
                        + (" (existing)" if scen == "random_90" else " (new)"), seed=seed)
            if scen == "random_90":
                keep_r90 = keep
        elif scen == "hm27":
            keep = np.broadcast_to(plat["hm27"].astype(bool)[None, :], X.shape).copy()
            ck = np.broadcast_to(np.array([c in hm27 for c in cpgs])[None, :], M.shape).copy()
            info.update(source="platform_vocab_masks hm27; clock CpGs Methyl27_Loci (existing)", seed="")
        else:
            acc = scen.split("_", 1)[1]
            Xi = np.load(need(WORKI / acc / "X.npy"))
            C = np.load(need(WORKI / acc / "coverage.npy"))
            st = pd.read_csv(need(WORKI / acc / "analysis_samples.csv"))
            if Xi.shape != (len(st), X.shape[1]) or not np.array_equal(np.isfinite(Xi), C >= 10):
                sys.exit(f"{acc}: vocabulary matrix and coverage disagree")
            P = np.isfinite(Xi)
            a = np.random.default_rng([SEED, int(acc[3:])]).integers(0, len(P), len(X))
            keep = P[a]
            ck = np.zeros(M.shape, bool)
            ck[:, ic] = keep[:, vj]
            ck_b = ck.copy()
            ck_b[:, ~invoc] = True
            cnt = np.bincount(a, minlength=len(P))
            np.save(PRIV / "masks" / f"private_assignment_{scen}.npy", a.astype(np.int64))
            info.update(source=f"analysis I vocabulary matrix {acc} ({SEQ[acc]}), observed = >= 10 reads (new)",
                        seed=f"[{SEED}, {int(acc[3:])}]", source_samples=len(P), source_samples_used=int((cnt > 0).sum()),
                        max_participants_per_source_sample=int(cnt.max()),
                        source_sample_vocab_cpgs_median=float(np.median(P.sum(1))),
                        sha256_assignment=sha256_bytes(a.astype(np.int64).tobytes()))
        obs = gsobs & keep
        cobs = cfin & ck
        arrays = {"obs": np.packbits(obs, axis=1), "obs_shape": np.array(obs.shape),
                  "cobs": np.packbits(cobs, axis=1), "cobs_shape": np.array(cobs.shape)}
        if ck_b is not None:
            cobs_b = cfin & ck_b
            arrays.update(cobs_nvobs=np.packbits(cobs_b, axis=1), cobs_nvobs_shape=np.array(cobs_b.shape))
            info["sha256_clock_mask_nvobs"] = sha256_bytes(arrays["cobs_nvobs"].tobytes())
            info["clock_kept_nonvocab_mean_nvobs"] = float(cobs_b[:, ~invoc].sum(1).mean())
        atomic_npz(PRIV / "masks" / f"private_mask_{scen}.npz", **arrays)
        k = obs.sum(1)
        info.update(n_participants=len(X), gs_observed_vocab_cpgs=int(gsobs.all(0).sum()), kept_mean=float(k.mean()),
                    kept_median=float(np.median(k)), kept_min=int(k.min()), kept_max=int(k.max()),
                    observed_fraction=float(np.isfinite(np.where(keep, X, np.nan)).mean()),
                    stored_observed_fraction=stored_obs.get(scen, ""),
                    clock_kept_vocab_mean=float(cobs[:, ic].sum(1).mean()),
                    clock_kept_nonvocab_mean=float(cobs[:, ~invoc].sum(1).mean()),
                    clock_vs_vocab_mask_disagreements_at_vocab_clock_cpgs=int((cobs[:, ic] != obs[:, vj]).sum()),
                    sha256_vocab_mask=sha256_bytes(arrays["obs"].tobytes()),
                    sha256_clock_mask=sha256_bytes(arrays["cobs"].tobytes()))
        if scen == "random_95":
            info["fraction_of_kept_also_kept_by_random_90"] = float((obs & keep_r90).sum() / obs.sum())
        rows.append(info)
        log(f"{scen}: kept vocabulary CpGs per participant mean {k.mean():.1f} (min {k.min()}, max {k.max()})")
        del keep, ck, obs, cobs
    OUTJ.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    atomic_csv(OUTJ / f"J_masks_{DATE}.csv", tab)
    done_marker("masks", {"scenarios": ALL_SCEN, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ risk functions (the templates')
def locked_risk(model, feats, cov):
    """scenario_gs_combined.locked_risk (v12 raw-CpG, embedding and biomarker fold models)."""
    z = np.where(np.isfinite(feats), feats, model["mean"])
    z = (z - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return c @ model["coefficient"][:2] + z @ model["coefficient"][2:]


def gpu_risk(model, feats, cov):
    """scenario_gs_combined.gpu_risk (GPU raw-CpG and factor+residual fold models)."""
    z = np.where(np.isfinite(feats), feats, model["mean"])
    z = (z - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return np.hstack([c, z]) @ model["coefficient"]


def fi_locked_risk(model, feats, cov):
    """scenario_gs_raw_factor_imputed.locked_risk."""
    z = (np.where(np.isfinite(feats), feats, model["mean"]) - model["mean"]) / model["scale"]
    c = (cov - model["cov_mean"]) / model["cov_scale"]
    return c @ model["coefficient"][:2] + z @ model["coefficient"][2:]


def raw_v12_model(ep, f):
    d = [d for d in RAW_DIRS if (d / f"{ep}_linear_outer{f}_model.npz").exists()]
    if not d:
        sys.exit(f"Cannot proceed - required data file not found: v12 raw-CpG model {ep} outer{f}")
    return dict(np.load(d[0] / f"{ep}_linear_outer{f}_model.npz", allow_pickle=False))


def gpu_model(name, ep, f):
    return dict(np.load(need(FR_DIR / "models" / f"{name}_{ep}_outer{f}.npz"), allow_pickle=False))


def raw_risks_filled(Xf, preds, pos, ep, held_only=None):
    """Both raw fold models applied (scenario_gs_combined functions) to a filled matrix Xf (rows = frame cache)."""
    pe = preds[ep]
    idx = np.array([pos[s] for s in pe.dnam_id])
    cov = pe[["age", "sex"]].to_numpy(float)
    r_v12, r_gpu = np.full(len(pe), np.nan), np.full(len(pe), np.nan)
    for f in sorted(pe.outer_fold.unique()):
        if held_only is not None and f != held_only:
            continue
        held = np.flatnonzero(pe.outer_fold.to_numpy() == f)
        r_v12[held] = locked_risk(raw_v12_model(ep, f), Xf[idx[held]].astype(np.float64), cov[held])
        r_gpu[held] = gpu_risk(gpu_model("raw_cpg_gpu", ep, f), Xf[idx[held]], cov[held].astype(np.float32))
    return r_v12, r_gpu


# ------------------------------------------------------------------ stage mgpt
def stage_mgpt(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    from methylgpt2_model import BioHeads, embed_batch, load_checkpoint
    require_committed()
    require_done("masks")
    cuda_preinit()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = pick_device(device)
    ids, X = load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    preds = load_preds()
    ag = torch.load(need(AGING), map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)
    model.load_state_dict(ag["model"]); model.eval()
    mt = torch.load(need(MULTITASK), map_location=dev, weights_only=False)
    mt_model, mt_core, _ = load_checkpoint(Path(mt["ckpt"]), None, dev)
    mt_model.load_state_dict(mt["model"]); mt_model.eval()
    mt_heads = BioHeads(mt_core.rank + mt_model.latents.shape[1], len(mt["tissues"]) + 1, len(mt["bio_names"])).to(dev)
    mt_heads.load_state_dict(mt["heads"]); mt_heads.eval()
    _, fi_core, _ = load_checkpoint(Path(ag["ckpt"]), None, dev)   # scenario_gs_raw_factor_imputed's core

    def features(Xs):   # scenario_gs_combined.main.features
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

    def factor_impute(Xs):   # scenario_gs_raw_factor_imputed loop
        Xf = Xs.copy()
        with torch.no_grad():
            for i in range(0, len(Xf), 64):
                x = torch.from_numpy(Xf[i:i + 64]).to(dev)
                obs = torch.isfinite(x)
                rec = fi_core.predict(fi_core.posterior(x, obs.float())).clamp(0, 1)
                Xf[i:i + 64] = torch.where(obs, x, rec).float().cpu().numpy()
        return Xf

    bio_stored = pd.read_parquet(need(BIO_RUN / "private_nested_predictions.parquet"))
    bio_stored = bio_stored[bio_stored.age_adjustment == "linear"]
    bio_stored = bio_stored.assign(dnam_id=bio_stored.dnam_id.astype(str))
    fr_stored = {n: pd.read_parquet(need(FR_DIR / n / "private_nested_predictions.parquet")) for n in ("factor_residual", "raw_cpg_gpu")}
    rec = {}
    for scen in ALL_SCEN:
        fo = PRIV / f"private_risks_mgpt_{scen}.parquet"
        if fo.exists():
            log(f"{scen}: MethylGPT risks exist, skipped")
            continue
        t1 = time.time()
        obs = load_mask(scen)
        Xs = np.where(obs, X, np.nan).astype(np.float32)
        del obs
        E, B, R = features(Xs)
        FR = np.hstack([E, B, R]).astype(np.float32)
        del R
        Xf = factor_impute(Xs)
        if not np.isfinite(Xf).all():
            sys.exit(f"{scen}: factor imputation left missing values")
        frames = []
        for ep, pe in preds.items():
            idx = np.array([pos[s] for s in pe.dnam_id])
            cov = pe[["age", "sex"]].to_numpy(float)
            folds = sorted(pe.outer_fold.unique())
            held_sets = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in folds]
            risk = {k: np.empty(len(pe)) for k in ("embedding", "biomarkers", "factor_residual", "raw_cpg_v12_mean",
                                                   "raw_cpg_gpu_mean", "raw_cpg_v12_factor_imputed", "raw_cpg_gpu_factor_imputed")}
            for f, held in zip(folds, held_sets):
                mm = dict(np.load(need(MGPT2_RUN / f"{ep}_linear_outer{f}_model.npz"), allow_pickle=False))
                bm = dict(np.load(need(BIO_RUN / f"{ep}_linear_outer{f}_model.npz"), allow_pickle=False))
                rm = raw_v12_model(ep, f)
                risk["embedding"][held] = locked_risk(mm, E[idx[held]].astype(np.float64), cov[held])
                risk["biomarkers"][held] = locked_risk(bm, B[idx[held]].astype(np.float64), cov[held])
                risk["raw_cpg_v12_mean"][held] = locked_risk(rm, Xs[idx[held]].astype(np.float64), cov[held])
                gfr = gpu_model("factor_residual", ep, f)
                risk["factor_residual"][held] = gpu_risk(gfr, FR[idx[held]], cov[held].astype(np.float32))
                gm = gpu_model("raw_cpg_gpu", ep, f)
                risk["raw_cpg_gpu_mean"][held] = gpu_risk(gm, Xs[idx[held]], cov[held].astype(np.float32))
                # scenario_gs_raw_factor_imputed application
                zf = (Xf[idx[held]] - gm["mean"]) / gm["scale"]
                risk["raw_cpg_gpu_factor_imputed"][held] = np.hstack(
                    [((cov[held] - gm["cov_mean"]) / gm["cov_scale"]).astype(np.float32), zf.astype(np.float32)]) @ gm["coefficient"]
                risk["raw_cpg_v12_factor_imputed"][held] = fi_locked_risk(rm, Xf[idx[held]].astype(np.float64), cov[held])
            if scen == "full":   # scenario_gs_combined's own check against the stored out-of-fold risks
                checks = {"embedding": pe.risk.to_numpy(float),
                          "biomarkers": bio_stored[bio_stored.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)}
                for name, col in (("factor_residual", "factor_residual"), ("raw_cpg_gpu", "raw_cpg_gpu_mean")):
                    s_ = fr_stored[name]
                    s_ = s_.assign(dnam_id=s_.dnam_id.astype(str))
                    checks[col] = s_[s_.endpoint == ep].set_index("dnam_id").loc[pe.dnam_id, "risk"].to_numpy(float)
                for name, ref in checks.items():
                    if np.max(np.abs(ref - risk[name])) > 1e-3 * (1 + np.abs(ref).max()):
                        sys.exit(f"recomputed {name} risks differ from the stored nested predictions for {ep}")
            for cname, parts in COMBOS.items():
                v = np.zeros(len(pe))
                for held in held_sets:
                    for k in parts:
                        x_ = risk[k][held]; v[held] += (x_ - x_.mean()) / x_.std()
                risk[cname] = v
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            for k, v in risk.items():
                fr_[k] = v
            frames.append(fr_)
        PRIV.mkdir(parents=True, exist_ok=True)
        atomic_parquet(fo, pd.concat(frames, ignore_index=True))
        rec[scen] = {"seconds": round(time.time() - t1, 1)}
        log(f"{scen}: MethylGPT, raw mean-fill and factor-imputed risks written ({time.time() - t1:.0f} s)")
        del Xs, E, B, FR, Xf
    done_marker("mgpt", {"scenarios": ALL_SCEN, "record": rec, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ kNN helpers
def knn_fill(rd, R, mu_h, Xs, ctx, masked, dev, batch=KNN_BATCH):
    """knn_predict_lowmem (G's implementation, k = 10) at the masked columns; returns the filled copy of Xs."""
    import torch
    Xt = torch.from_numpy(Xs).to(dev)
    ct = torch.from_numpy(ctx).to(dev)
    mt = torch.from_numpy(masked).to(dev)
    pr = rd.knn_predict_lowmem(R, mu_h, Xt, ct, mt, k=KNN_K, batch=batch, chunk=KNN_CHUNK)
    del Xt, ct, mt
    out = Xs.copy()
    for i, p in enumerate(pr):
        out[i, masked[i]] = p
    return out


def clock_masked_vocab(scen, ic):
    """(n, 255): vocabulary clock CpGs that the scenario's clock mask removes and that GS observes."""
    cobs = load_mask(scen, "cobs")
    _, _, M = load_clock_matrix()
    return np.isfinite(M[:, ic]) & ~cobs[:, ic]


# ------------------------------------------------------------------ stage knn (corpus reference)
def stage_knn(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    require_committed()
    require_done("masks")
    cuda_preinit()
    t0 = time.time()
    dev = pick_device(device)
    ids, X = load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    gsobs = np.isfinite(X)
    preds = load_preds()
    vocab = np.load(need(MASKS))["vocab"].astype(str)
    _, cpgs, _ = load_clock_matrix()
    invoc, ic, vj = clock_vocab_index(cpgs, vocab)
    G = load_module("large_gain_G", G_SCRIPT)
    F = G.load_f()
    rd = load_module("recon_diag", RECON)
    mu = np.load(need(CORE_MU))["mu"].astype(np.float32)
    gref = json.loads(need(G_REF_JSON).read_text())
    free, total = torch.cuda.mem_get_info(dev)
    log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
    R, rrows = G.knn_reference(F, "heldout", dev)
    sha = sha256_bytes(rrows.astype(np.int64).tobytes())
    ref_ok = sha == gref["rows_sha256"] and len(rrows) == gref["rows"]
    prof = pd.read_parquet(need(F.CORPUS_DIR / "profiles.parquet"), columns=["split"])
    d_rows = int((prof.split == "train").sum())
    record = {"reference_rows": int(len(rrows)), "reference_rows_sha256": sha, "G_rows": gref["rows"],
              "G_rows_sha256": gref["rows_sha256"], "identical_to_G_heldout_reference": bool(ref_ok),
              "D_reference_rows": d_rows, "D_fills_reused": False,
              "D_reason": f"D's reference is all {d_rows} training-split profiles (not G's {len(rrows)}) with the bundle mu; recomputed",
              "mu": str(CORE_MU.relative_to(HERE)), "k": KNN_K, "query_batch": KNN_BATCH, "reference_chunk": KNN_CHUNK,
              "scenarios": {}}
    write_json(OUTJ / f"J_knn_record_{DATE}.json", record)
    if not ref_ok:
        sys.exit("kNN reference differs from G's held-out reference; stopped before any query")
    mu_h = torch.as_tensor(mu, device=dev).half()
    for scen in SCENARIOS:
        fo = PRIV / f"private_risks_knn_{scen}.parquet"
        if fo.exists():
            log(f"{scen}: corpus kNN exists, skipped")
            rec_old = json.loads((PRIV / f"private_knn_record_{scen}.json").read_text())
            record["scenarios"][scen] = rec_old
            continue
        t1 = time.time()
        obs = load_mask(scen)
        Xs = np.where(obs, X, np.nan).astype(np.float32)
        masked = gsobs & ~obs
        Xk = knn_fill(rd, R, mu_h, Xs, obs, masked, dev)
        ok_kept = bool(np.array_equal(Xk[obs], X[obs]))
        ok_fill = bool(np.isfinite(Xk[masked]).all() and not np.isfinite(Xk[~gsobs]).any())
        if not (ok_kept and ok_fill):
            sys.exit(f"{scen}: kNN fill check failed (kept unchanged {ok_kept}, masked filled {ok_fill})")
        atomic_npy(PRIV / f"private_knn_corpus_{scen}.npy", Xk)
        # clock fills (vocabulary clock CpGs masked by the clock mask)
        cm = clock_masked_vocab(scen, ic)
        ctx_c = obs.copy()
        ctx_c[:, vj] &= ~cm
        same_ctx = bool(np.array_equal(ctx_c, obs))
        fill = np.full(cm.shape, np.nan, np.float32)
        if same_ctx:
            fill[cm] = Xk[:, vj][cm]
        else:
            mc = np.zeros(X.shape, bool)
            mc[:, vj] = cm
            Xc = knn_fill(rd, R, mu_h, np.where(ctx_c, X, np.nan).astype(np.float32), ctx_c, mc, dev)
            fill[cm] = Xc[:, vj][cm]
            del Xc, mc
        if not np.isfinite(fill[cm]).all():
            sys.exit(f"{scen}: clock kNN fills incomplete")
        atomic_npy(PRIV / f"private_knn_corpus_clockfill_{scen}.npy", fill)
        frames = []
        for ep, pe in preds.items():
            r_v12, r_gpu = raw_risks_filled(Xk, preds, pos, ep)
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            fr_["raw_cpg_v12_knn"], fr_["raw_cpg_gpu_knn"] = r_v12, r_gpu
            frames.append(fr_)
        rec = {"masked_values_filled": int(masked.sum()), "kept_values_unchanged": ok_kept, "all_masked_filled": ok_fill,
               "clock_masked_vocab_values": int(cm.sum()), "clock_context_equals_vocabulary_context": same_ctx,
               "seconds": round(time.time() - t1, 1)}
        write_json(PRIV / f"private_knn_record_{scen}.json", rec)
        atomic_parquet(fo, pd.concat(frames, ignore_index=True))
        record["scenarios"][scen] = rec
        write_json(OUTJ / f"J_knn_record_{DATE}.json", record)
        log(f"{scen}: corpus kNN fills and raw kNN risks written ({time.time() - t1:.0f} s)")
        del Xs, Xk, masked, obs
    write_json(OUTJ / f"J_knn_record_{DATE}.json", record)
    done_marker("knn", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage cohort (in-cohort kNN)
def stage_cohort(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    require_committed()
    require_done("masks")
    cuda_preinit()
    t0 = time.time()
    dev = pick_device(device)
    ids, X = load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    gsobs = np.isfinite(X)
    preds = load_preds()
    vocab = np.load(need(MASKS))["vocab"].astype(str)
    _, cpgs, _ = load_clock_matrix()
    invoc, ic, vj = clock_vocab_index(cpgs, vocab)
    rd = load_module("recon_diag", RECON)
    mu = np.load(need(CORE_MU))["mu"].astype(np.float32)
    mu_h = torch.as_tensor(mu, device=dev).half()
    obs_all = {s: load_mask(s) for s in SCENARIOS}
    cm_all = {s: clock_masked_vocab(s, ic) for s in SCENARIOS}
    (PRIV / "cohort").mkdir(parents=True, exist_ok=True)
    prev = PRIV / "cohort" / "private_reference_check.csv"
    for ep, pe in preds.items():
        outs = {s: PRIV / "cohort" / f"private_cohortknn_{s}__{ep}.parquet" for s in SCENARIOS}
        if all(o.exists() and o.with_name(o.stem + "_clockfill.npy").exists() for o in outs.values()):
            log(f"{ep}: in-cohort kNN exists, skipped")
            continue
        t1 = time.time()
        idx = np.array([pos[s] for s in pe.dnam_id])
        fold = pe.outer_fold.to_numpy()
        res = {s: {"v12": np.full(len(pe), np.nan), "gpu": np.full(len(pe), np.nan),
                   "fill": np.full((len(X), len(ic)), np.nan, np.float32)} for s in SCENARIOS}
        refcheck = []
        for f in sorted(np.unique(fold)):
            held, fit = idx[fold == f], idx[fold != f]
            rm, gm = raw_v12_model(ep, f), gpu_model("raw_cpg_gpu", ep, f)
            fit_ids = set(rm["fit_ids"].astype(str))
            refcheck.append({"endpoint": ep, "outer_fold": int(f), "reference_rows": int(len(fit)),
                             "equals_v12_fit_ids": set(ids[fit]) == fit_ids,
                             "held_out_in_reference": int(len(np.intersect1d(held, fit)))})
            Rh = torch.from_numpy(X[fit]).to(dev).half()
            Xh = X[held]
            hrows = np.flatnonzero(fold == f)
            cov = pe[["age", "sex"]].to_numpy(float)[hrows]
            for s in SCENARIOS:
                obs = obs_all[s][held]
                Xs = np.where(obs, Xh, np.nan).astype(np.float32)
                masked = gsobs[held] & ~obs
                Xk = knn_fill(rd, Rh, mu_h, Xs, obs, masked, dev)
                if not (np.array_equal(Xk[obs], Xh[obs]) and np.isfinite(Xk[masked]).all()):
                    sys.exit(f"{s} {ep} fold {f}: in-cohort kNN fill check failed")
                # raw fold models of this endpoint and fold on the held-out participants
                res[s]["v12"][hrows] = locked_risk(rm, Xk.astype(np.float64), cov)
                res[s]["gpu"][hrows] = gpu_risk(gm, Xk, cov.astype(np.float32))
                cm = cm_all[s][held]
                ctx_c = obs.copy()
                ctx_c[:, vj] &= ~cm
                fill = np.full(cm.shape, np.nan, np.float32)
                if np.array_equal(ctx_c, obs):
                    fill[cm] = Xk[:, vj][cm]
                else:
                    mc = np.zeros(obs.shape, bool)
                    mc[:, vj] = cm
                    Xc = knn_fill(rd, Rh, mu_h, np.where(ctx_c, Xh, np.nan).astype(np.float32), ctx_c, mc, dev)
                    fill[cm] = Xc[:, vj][cm]
                if not np.isfinite(fill[cm]).all():
                    sys.exit(f"{s} {ep} fold {f}: in-cohort clock fills incomplete")
                res[s]["fill"][held] = fill
            del Rh
            torch.cuda.empty_cache()
        for s in SCENARIOS:
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            fr_["raw_cpg_v12_cohortknn"], fr_["raw_cpg_gpu_cohortknn"] = res[s]["v12"], res[s]["gpu"]
            if not (np.isfinite(fr_.raw_cpg_v12_cohortknn).all() and np.isfinite(fr_.raw_cpg_gpu_cohortknn).all()):
                sys.exit(f"{s} {ep}: in-cohort risks incomplete")
            atomic_npy(outs[s].with_name(outs[s].stem + "_clockfill.npy"), res[s]["fill"])
            atomic_parquet(outs[s], fr_)
        old = pd.read_csv(prev) if prev.exists() else pd.DataFrame()
        tab = pd.concat([old, pd.DataFrame(refcheck)], ignore_index=True).drop_duplicates(["endpoint", "outer_fold"], keep="last")
        atomic_csv(prev, tab)
        log(f"{ep}: in-cohort kNN for {len(SCENARIOS)} scenarios written ({time.time() - t1:.0f} s)")
    done_marker("cohort", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage clockin
def stage_clockin():
    require_committed()
    require_done("masks", "knn", "cohort")
    t0 = time.time()
    cids, cpgs, M = load_clock_matrix()
    vocab = np.load(need(MASKS))["vocab"].astype(str)
    invoc, ic, vj = clock_vocab_index(cpgs, vocab)
    means = np.nanmean(M, 0)
    d = PRIV / "clockin"
    d.mkdir(parents=True, exist_ok=True)
    atomic_csv(d / "private_ids.csv", pd.DataFrame({"sample_id": cids}))
    atomic_csv(d / "private_cpgs.csv", pd.DataFrame({"cpg": cpgs}))

    def filled(cobs, fill=None):
        Z = np.where(cobs, M, means[None, :]).astype(np.float32)     # template: masked -> nan -> development mean
        if fill is not None:
            sub = Z[:, ic]
            ok = np.isfinite(fill)
            sub[ok] = fill[ok]
            Z[:, ic] = sub
        return Z

    for v in clock_variants():
        fo = d / f"private_clockin_{v}.npy"
        if fo.exists():
            continue
        parts = v.split("__")
        scen, kind = parts[0], parts[1]
        if kind.endswith("_nvobs"):
            cobs = load_mask(scen, "cobs_nvobs")
        else:
            cobs = load_mask(scen, "cobs")
        fill = None
        if kind.startswith("knn"):
            fill = np.load(need(PRIV / f"private_knn_corpus_clockfill_{scen}.npy"))
        elif kind == "cohortknn":
            fill = np.load(need(PRIV / "cohort" / f"private_cohortknn_{scen}__{parts[2]}_clockfill.npy"))
        if fill is not None:
            cm = np.isfinite(M[:, ic]) & ~load_mask(scen, "cobs")[:, ic]
            if not np.array_equal(np.isfinite(fill), cm):
                sys.exit(f"{v}: clock fills do not match the clock-masked vocabulary CpGs")
        atomic_npy(fo, filled(cobs, fill))
    done_marker("clockin", {"variants": len(clock_variants()), "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage clocks (biolearn interpreter)
def score_variant(v):
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
    fo = PRIV / "clock_scores" / f"private_scores_{v}.csv"
    atomic_csv(fo, out)
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
    done_marker("clocks", {"variants": len(clock_variants()), "biolearn": getattr(biolearn, "__version__", "?"),
                           "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage cox (clock Cox models)
def stage_cox():
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    require_committed()
    require_done("clocks")
    t0 = time.time()
    preds = load_preds()
    sc = pd.read_csv(need(CLOCK_SCORES))
    sc["sample_id"] = sc.sample_id.astype(str)
    full = sc[sc.scenario == "full"].set_index("sample_id")
    fits = {}
    for ep, pe in preds.items():     # scenario_clocks_cox: fit on the fitting participants' full-profile scores
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
    scores = {}

    def get_scores(v):
        if v not in scores:
            t = pd.read_csv(need(PRIV / "clock_scores" / f"private_scores_{v}.csv"))
            t["sample_id"] = t.sample_id.astype(str)
            scores[v] = t.set_index("sample_id")
        return scores[v]

    for scen in ALL_SCEN:
        fills = ["mean"] if scen == "full" else ["mean", "knn", "cohortknn"] + (["mean_nvobs", "knn_nvobs"] if scen.startswith("seq_") else [])
        frames = []
        for ep, pe in preds.items():
            ids = pe.dnam_id.to_numpy()
            cov = pe[["age", "sex"]].to_numpy(float)
            folds = [np.flatnonzero(pe.outer_fold.to_numpy() == f) for f in sorted(pe.outer_fold.unique())]
            fr_ = pe[["dnam_id", "endpoint", "outer_fold"]].copy()
            for fl in fills:
                v = f"{scen}__cohortknn__{ep}" if fl == "cohortknn" else f"{scen}__{fl}"
                S = get_scores(v)
                for name, cols in CLOCK_MODELS.items():
                    Ms = np.column_stack([cov] + [S.loc[ids, c].to_numpy(float) for c in cols])
                    r = np.empty(len(pe))
                    for k, held in enumerate(folds):
                        m, s, coef = fits[(ep, k, name)]
                        r[held] = ((Ms[held] - m) / s) @ coef
                    fr_[f"{name}_{fl}"] = r
            frames.append(fr_)
        atomic_parquet(PRIV / f"private_risks_clocks_{scen}.parquet", pd.concat(frames, ignore_index=True))
        log(f"{scen}: clock Cox risks written")
    done_marker("cox", {"seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ assembly and pair counting
def assemble(scen, preds):
    """One table per scenario: rows = endpoint x participant in the template order, columns = every model's risk."""
    base = pd.concat([pe[["dnam_id", "endpoint", "outer_fold", "event", "duration", "baseline_risk"]] for pe in preds.values()],
                     ignore_index=True).rename(columns={"baseline_risk": "age_sex"})
    keys = ["dnam_id", "endpoint", "outer_fold"]
    parts = [PRIV / f"private_risks_mgpt_{scen}.parquet", PRIV / f"private_risks_clocks_{scen}.parquet"]
    if scen != "full":
        parts.append(PRIV / f"private_risks_knn_{scen}.parquet")
    for p in parts:
        t = pd.read_parquet(need(p))
        t["dnam_id"] = t.dnam_id.astype(str)
        base = base.merge(t, on=keys, how="left", validate="one_to_one")
    if scen != "full":
        t = pd.concat([pd.read_parquet(need(PRIV / "cohort" / f"private_cohortknn_{scen}__{ep}.parquet")) for ep in ENDPOINTS], ignore_index=True)
        t["dnam_id"] = t.dnam_id.astype(str)
        base = base.merge(t, on=keys, how="left", validate="one_to_one")
    miss = [m for m in models_for(scen) if m not in base.columns or not np.isfinite(base[m].to_numpy(float)).all()]
    if miss:
        sys.exit(f"{scen}: risks missing or not finite for {miss}")
    return base


def fold_counts(ev, du, R, Ws):
    """sksurv concordance counts (concordant, discordant, tied risk; tied_tol 1e-8) of each risk vector in R (m, n) over
    the comparable pairs of one fold, with pair weights w_i w_j for each weight matrix in Ws ((B, n) or None)."""
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
        tied = (comp & tie).astype(np.float64)
        for (tot, arr), W in zip(out, Ws):
            if W is None:
                arr[k] = [con.sum(), tied.sum()]
            else:
                We = W[:, e]
                arr[k, :, 0] = ((We @ con) * W).sum(1)
                arr[k, :, 1] = ((We @ tied) * W).sum(1)
    return out


def cidx_from(con, tied, tot):
    return (con + 0.5 * tied) / tot


def sksurv_pooled(ev, du, r, folds):
    from sksurv.metrics import concordance_index_censored
    t = np.zeros(3)
    for s in folds:
        _, c, d, tr, _ = concordance_index_censored(ev[s], du[s], r[s])
        t += [c, d, tr]
    return (t[0] + 0.5 * t[2]) / t.sum()


# ------------------------------------------------------------------ stage checks
def stage_checks():
    require_committed()
    require_done("mgpt", "knn", "cohort", "cox")
    preds = load_preds()
    rows = []

    def add(check, scenario, item, stored, recomputed, tol, passed=None, note=""):
        diff = abs(float(stored) - float(recomputed)) if isinstance(stored, (int, float, np.floating)) and \
            isinstance(recomputed, (int, float, np.floating)) else ""
        ok = passed if passed is not None else (diff != "" and diff <= tol)
        rows.append({"check": check, "scenario": scenario, "item": item, "stored": stored, "recomputed": recomputed,
                     "abs_diff": diff, "tolerance": tol, "pass": bool(ok), "note": note})

    cj = json.loads(need(STORED_CINDEX).read_text())
    kj = json.loads(need(STORED_CLOCK_JSON).read_text())
    fi = pd.read_csv(need(STORED_FI_CSV))
    bt = pd.read_csv(need(STORED_BOOT_CSV))
    srisk = pd.read_parquet(need(STORED_RISKS)); srisk["dnam_id"] = srisk.dnam_id.astype(str)
    sfi = pd.read_parquet(need(STORED_FI)); sfi["dnam_id"] = sfi.dnam_id.astype(str)
    sck = pd.read_parquet(need(STORED_CLOCK_RISKS)); sck["dnam_id"] = sck.dnam_id.astype(str)
    name_map = {"embedding": "embedding", "biomarkers": "biomarkers", "factor_residual": "factor_residual",
                "raw_cpg_v12_mean": "raw_cpg_v12", "raw_cpg_gpu_mean": "raw_cpg_gpu", "mgpt2_combined": "mgpt2_combined",
                "ensemble_embed_bio": "ensemble_embed_bio"}
    # (1) C-indices and risks of the stored scenarios
    for scen in STORED_SCEN:
        A = assemble(scen, preds)
        for ep, pe in preds.items():
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            folds = [np.flatnonzero(g.outer_fold.to_numpy() == f) for f in sorted(g.outer_fold.unique())]
            ss = srisk[(srisk.scenario == scen) & (srisk.endpoint == ep)].set_index("dnam_id").loc[g.dnam_id]
            for mine, st in name_map.items():
                c = sksurv_pooled(ev, du, g[mine].to_numpy(float), folds)
                add("1 C-index reproduces stored", scen, f"{ep}:{mine}", cj["scenarios"][scen][ep][st], c, TOL_C,
                    note=f"max abs risk diff {np.max(np.abs(ss[st].to_numpy(float) - g[mine].to_numpy(float))):.3g}")
            sf = sfi[(sfi.scenario == scen) & (sfi.endpoint == ep)].set_index("dnam_id").loc[g.dnam_id]
            for m in FACTOR:
                c = sksurv_pooled(ev, du, g[m].to_numpy(float), folds)
                ref = fi[(fi.scenario == scen) & (fi.endpoint == ep) & (fi.comparator == m)].c_comparator
                add("1 C-index reproduces stored", scen, f"{ep}:{m}", float(ref.iloc[0]), c, TOL_C,
                    note=f"max abs risk diff {np.max(np.abs(sf[m].to_numpy(float) - g[m].to_numpy(float))):.3g}")
            sk = sck[(sck.scenario == scen) & (sck.endpoint == ep)]
            for name in CLOCK_MODELS:
                c = sksurv_pooled(ev, du, g[f"{name}_mean"].to_numpy(float), folds)
                rr = sk[sk.model == name].set_index("dnam_id").loc[g.dnam_id, "risk"].to_numpy(float)
                add("1 C-index reproduces stored", scen, f"{ep}:{name}_mean", kj[scen][ep][name], c, TOL_C,
                    note=f"max abs risk diff {np.max(np.abs(rr - g[f'{name}_mean'].to_numpy(float))):.3g}")
            c = sksurv_pooled(ev, du, g["age_sex"].to_numpy(float), folds)
            ref = bt[(bt.scenario == scen) & (bt.endpoint == ep) & (bt.comparator == "age_sex")].c_comparator
            add("1 C-index reproduces stored", scen, f"{ep}:age_sex", float(ref.iloc[0]), c, TOL_C)
    # (2) clock scores
    sc = pd.read_csv(need(CLOCK_SCORES)); sc["sample_id"] = sc.sample_id.astype(str)
    for scen in STORED_SCEN:
        mine = pd.read_csv(need(PRIV / "clock_scores" / f"private_scores_{scen}__mean.csv"))
        mine["sample_id"] = mine.sample_id.astype(str)
        ref = sc[sc.scenario == scen].set_index("sample_id").loc[mine.sample_id]
        for c in ("GrimAgeV2", "DunedinPACE"):
            d = float(np.max(np.abs(ref[c].to_numpy(float) - mine[c].to_numpy(float))))
            add("2 clock scores reproduce stored", scen, c, 0.0, d, TOL_C, passed=d <= TOL_C, note="max abs difference")
    # (3) masks: observed fractions; (6) hashes
    mk = pd.read_csv(need(OUTJ / f"J_masks_{DATE}.csv"))
    for _, r in mk.iterrows():
        if r.scenario in STORED_SCEN:
            add("3 observed fraction equals stored", r.scenario, "observed_fraction", float(r.stored_observed_fraction),
                float(r.observed_fraction), 0.0, passed=float(r.stored_observed_fraction) == float(r.observed_fraction))
        add("6 mask hashed", r.scenario, "sha256 vocabulary / clock mask", "", f"{r.sha256_vocab_mask} / {r.sha256_clock_mask}",
            "", passed=isinstance(r.sha256_vocab_mask, str) and len(r.sha256_vocab_mask) == 64)
    # (4), (5), (8) kNN
    kr = json.loads(need(OUTJ / f"J_knn_record_{DATE}.json").read_text())
    add("4 kNN reference equals G's held-out reference", "all", "rows / row SHA-256", f"{kr['G_rows']} / {kr['G_rows_sha256']}",
        f"{kr['reference_rows']} / {kr['reference_rows_sha256']}", "", passed=kr["identical_to_G_heldout_reference"])
    add("5 D's private hm27 kNN fills reused", "hm27", "reference rows (D vs G)", kr["D_reference_rows"], kr["reference_rows"], "",
        passed=True, note="not reused; " + kr["D_reason"])
    for scen, r in kr["scenarios"].items():
        add("8 kNN fill integrity", scen, "corpus kNN: kept values unchanged and every masked GS-observed CpG filled", "",
            f"{r['masked_values_filled']} values filled", "", passed=r["kept_values_unchanged"] and r["all_masked_filled"])
    # (7) sequencing patterns (checked again here)
    for acc in SEQ:
        Xi = np.load(need(WORKI / acc / "X.npy")); C = np.load(need(WORKI / acc / "coverage.npy"))
        add("7 sequencing pattern = coverage >= 10", f"seq_{acc}", "finite X equals coverage >= 10", "", f"{Xi.shape[0]} samples",
            "", passed=bool(np.array_equal(np.isfinite(Xi), C >= 10)))
    # (9) in-cohort references
    rc = pd.read_csv(need(PRIV / "cohort" / "private_reference_check.csv"))
    add("9 in-cohort reference = training folds (v12 fit_ids), held-out fold excluded", "all", "endpoint-folds", 40, len(rc), "",
        passed=len(rc) == 40 and bool(rc.equals_v12_fit_ids.all()) and int(rc.held_out_in_reference.sum()) == 0)
    # (10) pair counting
    stk = load_module("stk", STK)
    A = assemble("full", preds)
    worst = 0.0
    for ep in ENDPOINTS:
        g = A[A.endpoint == ep]
        ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
        folds = [np.flatnonzero(g.outer_fold.to_numpy() == f) for f in sorted(g.outer_fold.unique())]
        R = np.stack([g[m].to_numpy(float) for m in models_for("full")])
        con = np.zeros(len(R)); tied = np.zeros(len(R)); tot = 0.0
        for s in folds:
            (t_, arr), = fold_counts(ev[s], du[s], R[:, s], [None])
            con += arr[:, 0]; tied += arr[:, 1]; tot += t_
        for k, m in enumerate(models_for("full")):
            worst = max(worst, abs(cidx_from(con[k], tied[k], tot) - sksurv_pooled(ev, du, R[k], folds)))
    add("10 pair counting equals sksurv (point)", "full", "max abs C difference over endpoints and models", 0.0, worst, 1e-12,
        passed=worst <= 1e-12)
    import torch
    g = A[A.endpoint == ENDPOINTS[0]]
    s = np.flatnonzero(g.outer_fold.to_numpy() == 1)
    ev, du = g.event.astype(bool).to_numpy()[s], g.duration.to_numpy(float)[s]
    R = np.stack([g[m].to_numpy(float)[s] for m in (OURS, "age_sex")])
    W = np.stack([np.bincount(np.random.default_rng(b).integers(0, len(s), len(s)), minlength=len(s)) for b in range(5)]).astype(np.float64)
    (tot, arr), = fold_counts(ev, du, R, [W])
    ref = stk.PairCounter(ev, du, torch.device("cpu")).counts(R, W)
    d = float(max(np.abs(arr[..., 0] - ref[..., 0]).max(), np.abs(arr[..., 1] - ref[..., 2]).max(), np.abs(tot[None, :] - ref.sum(-1)).max()))
    add("10 pair counting equals stk.PairCounter (weighted)", "full", f"{ENDPOINTS[0]} fold 1, 5 draws", 0.0, d, 1e-9, passed=d <= 1e-9)
    tab = pd.DataFrame(rows)
    atomic_csv(OUTJ / f"J_checks_{DATE}.csv", tab)
    nf = int((~tab["pass"]).sum())
    log(f"checks: {len(tab)} rows, {nf} failed")
    if nf:
        print(tab[~tab["pass"]].to_string())
        sys.exit("checks failed; no new number is used")
    done_marker("checks", {"rows": len(tab), "failed": 0})


# ------------------------------------------------------------------ stage boot
def stage_boot():
    require_committed()
    require_done("checks")
    t0 = time.time()
    preds = load_preds()
    ids, _ = load_gs()
    pos = {s: i for i, s in enumerate(ids)}
    N = len(ids)
    I = np.random.default_rng(SEED).integers(0, N, size=(N_BOOT, N))
    Wp = np.stack([np.bincount(I[b], minlength=N) for b in range(N_BOOT)]).astype(np.float64)
    del I
    sens = {}
    for k, (ep, pe) in enumerate(preds.items()):
        rng = np.random.default_rng([SEED, 1, k])
        sens[ep] = {}
        for f in sorted(pe.outer_fold.unique()):
            n_f = int((pe.outer_fold == f).sum())
            J = rng.integers(0, n_f, size=(N_BOOT, n_f))
            sens[ep][f] = np.stack([np.bincount(J[b], minlength=n_f) for b in range(N_BOOT)]).astype(np.float64)
    out = {}
    for scen in ALL_SCEN:
        A = assemble(scen, preds)
        models = models_for(scen)
        point = np.zeros((len(models), len(ENDPOINTS)))
        bprim = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        bsens = np.zeros((len(models), len(ENDPOINTS), N_BOOT))
        for j, ep in enumerate(ENDPOINTS):
            g = A[A.endpoint == ep]
            ev, du = g.event.astype(bool).to_numpy(), g.duration.to_numpy(float)
            xr = np.array([pos[s] for s in g.dnam_id])
            R = np.stack([g[m].to_numpy(float) for m in models])
            acc = {"p": [np.zeros(len(models)), np.zeros(len(models)), 0.0],
                   "b": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)],
                   "s": [np.zeros((len(models), N_BOOT)), np.zeros((len(models), N_BOOT)), np.zeros(N_BOOT)]}
            for f in sorted(g.outer_fold.unique()):
                s = np.flatnonzero(g.outer_fold.to_numpy() == f)
                res = fold_counts(ev[s], du[s], R[:, s], [None, Wp[:, xr[s]], sens[ep][f]])
                for key, (tot, arr) in zip(("p", "b", "s"), res):
                    acc[key][0] += arr[..., 0]; acc[key][1] += arr[..., 1]; acc[key][2] = acc[key][2] + tot
            point[:, j] = cidx_from(*acc["p"])
            bprim[:, j] = cidx_from(acc["b"][0], acc["b"][1], acc["b"][2][None, :])
            bsens[:, j] = cidx_from(acc["s"][0], acc["s"][1], acc["s"][2][None, :])
        out[scen] = {"models": models, "point": point, "boot": bprim, "boot_within_fold": bsens}
        log(f"{scen}: point and bootstrap C-indices ({time.time() - t0:.0f} s)")
    PRIV.mkdir(parents=True, exist_ok=True)
    np.savez(PRIV / "private_bootstrap_cindex.npz", **{f"{s}__{k}": (np.array(v[k]) if k == "models" else v[k])
                                                       for s, v in out.items() for k in v})
    done_marker("boot", {"draws": N_BOOT, "seconds": round(time.time() - t0)})


# ------------------------------------------------------------------ stage analyse
def pboot(d):
    k = min(int((d <= 0).sum()), int((d >= 0).sum()))
    return min(1.0, 2 * (k + 1) / (len(d) + 1))


def stage_analyse():
    require_committed()
    require_done("boot")
    preds = load_preds()
    z = np.load(need(PRIV / "private_bootstrap_cindex.npz"), allow_pickle=False)
    mk = pd.read_csv(need(OUTJ / f"J_masks_{DATE}.csv")).set_index("scenario")
    per_ep, summ, rules, draws, res = [], [], [], [], {"settings": {"seed": SEED, "draws": N_BOOT, "k": KNN_K, "rule": RULE,
                                                                    "label": LABEL, "primary": PRIMARY}, "scenarios": {}}
    for scen in ALL_SCEN:
        models = list(z[f"{scen}__models"].astype(str))
        P, B, S = z[f"{scen}__point"], z[f"{scen}__boot"], z[f"{scen}__boot_within_fold"]
        ia = models.index("age_sex")
        io = models.index(OURS)
        dP, dB, dS = P - P[ia], B - B[ia], S - S[ia]             # gain above age+sex, per endpoint
        M, MB, MS = dP.mean(1), dB.mean(1), dS.mean(1)           # mean over endpoints
        for k, m in enumerate(models):
            draws.append(pd.DataFrame({"scenario": scen, "model": m, "draw": np.arange(N_BOOT), "M_shared": MB[k], "M_within_fold": MS[k]}))
            for j, ep in enumerate(ENDPOINTS):
                pe = preds[ep]
                d = dB[io, j] - dB[k, j]
                per_ep.append({"scenario": scen, "endpoint": ep, "model": m, "role": role(m), "n": len(pe),
                               "events": int(pe.event.sum()), "c_index": P[k, j], "c_age_sex": P[ia, j], "delta_c": dP[k, j],
                               "mgpt2_combined_minus_model": dP[io, j] - dP[k, j], "ci95_low": float(np.percentile(d, 2.5)),
                               "ci95_high": float(np.percentile(d, 97.5))})
        comps = [m for m in models if m not in (OURS, "age_sex")]
        dec = [m for m in comps if m in (DECISION if scen != "full" else [f"{r}_mean" for r in RAW] + [f"{c}_mean" for c in CLOCK_MODELS])]
        best = max(dec, key=lambda m: M[models.index(m)])
        for m in comps:
            k = models.index(m)
            d, ds = MB[io] - MB[k], MS[io] - MS[k]
            rel = (M[io] - M[k]) / M[k] if M[k] > 0 else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                relb = d / MB[k]
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            outcome = "not applicable (M <= 0)" if not M[k] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
            summ.append({"scenario": scen, "model": m, "role": role(m), "is_best_decision_comparator": m == best,
                         "M_model": M[k], "M_model_ci95_low": float(np.percentile(MB[k], 2.5)), "M_model_ci95_high": float(np.percentile(MB[k], 97.5)),
                         "M_mgpt2_combined": M[io], "difference": M[io] - M[k], "ci95_low": lo, "ci95_high": hi,
                         "ci95_low_within_fold": float(np.percentile(ds, 2.5)), "ci95_high_within_fold": float(np.percentile(ds, 97.5)),
                         "p_boot": pboot(d), "relative_gain": rel,
                         "relative_gain_ci95_low": float(np.nanpercentile(relb, 2.5)) if M[k] > 0 else np.nan,
                         "relative_gain_ci95_high": float(np.nanpercentile(relb, 97.5)) if M[k] > 0 else np.nan,
                         "rule_against_this_model": outcome})
        kb = models.index(best)
        d, ds = MB[io] - MB[kb], MS[io] - MS[kb]
        rel = (M[io] - M[kb]) / M[kb] if M[kb] > 0 else np.nan
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        outcome = "not applicable (M_best <= 0)" if not M[kb] > 0 else ("PASS" if rel >= 0.10 and lo > 0 else "FAIL")
        lab = "complete profiles (reference, not a decision)" if scen == "full" else (
            LABEL + (" (empirical public sequencing coverage pattern)" if scen.startswith("seq_") else ""))
        rules.append({"analysis": "J", "scenario": scen, "primary": scen == PRIMARY, "label": lab,
                      "kept_vocab_cpgs_per_participant_mean": float(mk.loc[scen, "kept_mean"]),
                      "metric": "mean over 8 endpoints of (pooled within-fold C-index - age+sex C-index)",
                      "methylgpt": OURS, "M_methylgpt": M[io], "M_methylgpt_ci95_low": float(np.percentile(MB[io], 2.5)),
                      "M_methylgpt_ci95_high": float(np.percentile(MB[io], 97.5)), "best_comparator": best, "M_best": M[kb],
                      "relative_gain": rel, "difference": M[io] - M[kb], "ci95_low": lo, "ci95_high": hi,
                      "interval": f"paired bootstrap, one shared participant resample per draw, {N_BOOT} draws, seed {SEED}",
                      "ci95_low_within_fold": float(np.percentile(ds, 2.5)), "ci95_high_within_fold": float(np.percentile(ds, 97.5)),
                      "p_boot": pboot(d), "rule": RULE, "outcome": outcome})
        res["scenarios"][scen] = {"M": {m: float(M[models.index(m)]) for m in models if m != "age_sex"}, "best": best,
                                  "relative_gain": rel, "difference": float(M[io] - M[kb]), "ci95": [lo, hi], "outcome": outcome}
        log(f"{scen}: M {OURS} {M[io]:.4f}; best {best} {M[kb]:.4f}; gain {rel:+.1%}; diff {M[io] - M[kb]:+.4f} [{lo:.4f}, {hi:.4f}] -> {outcome}")
    OUTJ.mkdir(parents=True, exist_ok=True)
    files = {f"J_cindex_per_endpoint_{DATE}.csv": pd.DataFrame(per_ep), f"J_summary_{DATE}.csv": pd.DataFrame(summ),
             f"J_rule_outcome_{DATE}.csv": pd.DataFrame(rules)}
    for fn, df in files.items():
        atomic_csv(OUTJ / fn, df)
    atomic_csv(OUTJ / f"J_bootstrap_draws_{DATE}.csv.gz", pd.concat(draws, ignore_index=True), compression="gzip")
    write_json(OUTJ / f"J_results_{DATE}.json", res)
    write_readme(res)
    done_marker("analyse")


def write_readme(res):
    def rows(fn):
        p = OUTJ / fn
        if fn.endswith(".json"):
            return "-"
        return str(len(pd.read_csv(p)))

    def cols(fn):
        if fn.endswith(".json"):
            return "-"
        return ", ".join(pd.read_csv(OUTJ / fn, nrows=1).columns)

    desc = {
        f"J_masks_{DATE}.csv": "Per scenario: label, mask source and seed, kept vocabulary CpGs per participant (mean, median, min, max), observed fraction (template definition) and the stored one, clock CpGs kept, mask SHA-256 (vocabulary, clock, _nvobs clock), sequencing source-sample counts and assignment SHA-256; random_95: fraction of its kept CpGs also kept by random_90.",
        f"J_checks_{DATE}.csv": "Every pre-specified check (reproduction of stored C-indices, risks, clock scores and observed fractions; kNN reference; D reuse; mask hashes; sequencing patterns; kNN fill integrity; in-cohort references; pair counting), with stored and recomputed values and pass.",
        f"J_knn_record_{DATE}.json": "kNN reference rows and SHA-256 against G's, D reuse decision, per-scenario fill counts, checks and timing.",
        f"J_cindex_per_endpoint_{DATE}.csv": "Per scenario, endpoint and model: role, n, events, pooled within-fold C-index, age+sex C-index, gain above age+sex (delta_c), MethylGPT combined minus model with 95% interval (shared bootstrap).",
        f"J_summary_{DATE}.csv": "Per scenario and comparator: M (mean gain above age+sex) with interval, MethylGPT's M, difference with shared and within-fold intervals, bootstrap P, relative gain with interval, rule outcome against this comparator, best-decision-comparator flag.",
        f"J_rule_outcome_{DATE}.csv": "One row per scenario: primary flag, label, M of MethylGPT, best decision comparator and its M, relative gain, difference, 95% interval (primary and within-fold), bootstrap P, rule, outcome.",
        f"J_bootstrap_draws_{DATE}.csv.gz": "Per scenario, model and draw: M under the shared resample and under the within-fold resample (2,000 draws).",
        f"J_results_{DATE}.json": "Settings and, per scenario, M of every model, best comparator, gain, difference, interval and outcome.",
    }
    pr = res["scenarios"][PRIMARY]
    lines = ["# Large-gain analysis J: GS incident disease with missing CpGs, nearest-neighbour comparators", "",
             f"Script: `large_gain_J_gs_missing_cpgs_knn_2026-10-28.py` (pre-specified design in its docstring). {LABEL.capitalize()}: "
             "measured GS development profiles (9,866 participants) with CpGs removed by fixed masks; the sequencing scenarios apply "
             "empirical public coverage patterns (analysis I's vocabulary matrices) to the GS profiles. No methylation value is simulated.", "",
             "## Primary result (hm27)", "",
             f"- MethylGPT combined M = {pr['M'][OURS]:.4f}; best decision comparator {pr['best']} M = {pr['M'][pr['best']]:.4f}.",
             f"- Relative gain {pr['relative_gain']:+.1%}; difference {pr['difference']:+.4f}, 95% interval [{pr['ci95'][0]:.4f}, {pr['ci95'][1]:.4f}]; outcome {pr['outcome']}.",
             "", "## Files", "", "| File | Rows | Description | Columns |", "|---|---:|---|---|"]
    for fn, ds in desc.items():
        if (OUTJ / fn).exists():
            lines.append(f"| `{fn}` | {rows(fn)} | {ds} | {cols(fn)} |")
    lines += ["", "## Inputs", "",
              "- GS development frame-cache matrix (`COMMON_SAMPLE_SPLIT_full18859_common_task_a_incident_positive_duration_20260629_train_all_gs_batches_all_probes_*.matrix.npz`) and endpoint table `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01/private_nested_predictions.parquet` (locked folds).",
              "- Fold models: `artifacts/gs_cox/mgpt2_embed_linear_2026-10-01`, `mt_bio_linear_2026-10-01`, `factor_residual_2026-10-01/models`, v12 raw-CpG runs under `/Entropy/kying/tmp/methylgpt_publication_2026-09-28/results`.",
              "- Clock CpG matrix and stored scores: `artifacts/gs_clocks/gs_dev_clock_scores_scenarios_2026-10-01.*`; stored risks and C-indices under `artifacts/gs_cox/` (checks).",
              "- kNN reference: G's held-out reference (`large_gain_G_sparse_inputs_2026-10-25.py`, `artifacts/large_gain_2026-10-25/G/knn_reference_heldout_2026-10-25.json`).",
              "- Sequencing coverage patterns: `artifacts/large_gain_2026-10-25/work_I/{GSE194154,GSE282512,GSE140730}/X.npy` (analysis I).",
              "", "## Notes", "",
              "- Interval: one participant resample per draw shared by every model, endpoint and scenario (the locked outer folds differ between endpoints, so a within-fold resample cannot be shared across endpoints); the within-fold resample per endpoint is reported as a sensitivity.",
              "- Decision comparators: raw-CpG fold models (v12, GPU) with fold-mean or corpus-kNN fill, and GrimAge2, DunedinPACE and both with mean- or corpus-kNN-filled clock CpGs. In-cohort kNN is a secondary comparator; raw models with MethylGPT factor imputation are MethylGPT applications; `_nvobs` clocks (sequencing scenarios, non-vocabulary clock CpGs observed) are a sensitivity.",
              "- Participant-level masks, fills, clock scores and risks stay in `artifacts/large_gain_2026-10-28_private/J/` on Chronos and are never committed."]
    (OUTJ / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["masks", "mgpt", "knn", "cohort", "clockin", "clocks", "cox", "checks", "boot", "analyse"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.chdir(HERE)          # checkpoint configs hold paths relative to the repository directory
    LOGS.mkdir(parents=True, exist_ok=True)
    {"masks": stage_masks, "mgpt": lambda: stage_mgpt(a.device), "knn": lambda: stage_knn(a.device),
     "cohort": lambda: stage_cohort(a.device), "clockin": stage_clockin, "clocks": lambda: stage_clocks(a.workers),
     "cox": stage_cox, "checks": stage_checks, "boot": stage_boot, "analyse": stage_analyse}[a.stage]()


if __name__ == "__main__":
    main()
