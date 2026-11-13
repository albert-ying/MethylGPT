#!/usr/bin/env python3
"""Large-gain analysis A: disease classification from few labelled samples across public cohorts.

Written on 24 Oct 2026, before any model is fitted, for the large-gain analysis plan of 23 Oct 2026 (Decision rule,
analysis A, Outputs). The data preparation here is shared with analysis B (large_gain_B_disease_cross_array_2026-10-24.py).
Every result is reported, whichever representation it favours.

DATA AND INCLUSION
  Labels      /Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/compiled_metadata.csv (SHA-256 in the candidate
              table's JSON). One row per GSM: a GSM whose rows disagree on disease, case_control, tissue or vivo_vitro is
              dropped; agreeing duplicates are collapsed. Disease labels are lower-cased, stripped, typographic apostrophes
              replaced by "'", then SYNONYMS applied.
  Control     healthy control: disease in CONTROL_LABELS (healthy, control, normal), case_control not "case",
              vivo_vitro "in_vivo", no cell line.
  Case        disease D as recorded, D not a control label, not a non-specific label (NONSPECIFIC) and not an exposure
              (EXPOSURES: smoker); case_control not "ctrl"; in vivo; no cell line.
  Tissue      the metadata tissue label ("unlabelled" when missing).
  Contrast    one per dataset: within one tissue, one disease against the healthy controls of that tissue. Among the
              (tissue, disease) strata with >= 30 cases and >= 30 controls, the one with the most cases (ties: most
              controls, then alphabetical tissue and disease).
  Arrays      27K, 450K, EPIC (v1), EPIC v2 (accepted platforms ACCEPTED_GPL). A profile's array is the array of its
              series' platform when the series has one array type; for multi-platform series it is inferred from the
              observed vocabulary CpGs (platform_vocab_masks_2026-10-01.npz): fewer than half of the 25,595 450K-not-27K
              CpGs observed -> 27K; else at least half of the 918 EPIC-v2-only CpGs observed -> EPIC v2; else at least half
              of the 136 EPIC-not-450K CpGs observed -> EPIC; else 450K. Only profiles of the contrast's most frequent array
              are kept.
  Profile QC  corpus rule: observed values clipped to [0, 1]; a profile with more than 1% of observed values outside
              [0, 1] is dropped. Corpus profiles already carry this QC (excluded_qc profiles are not used).
  Inclusion   >= 30 cases and >= 30 controls after QC and array restriction (the contrast rule is re-applied then).

  Dataset sources and groups
  (1) Corpus datasets: the corpus `group` (artifacts/corpus_2026-09-30/profiles.parquet), labels joined on
      GSM_ID == profiles.id (non-GSM ids such as TCGA have no labels here). Groups of the test split form the primary
      "test" group, groups of the dev split the primary "val" group. Groups with training-split profiles form the "train"
      group: their profiles were seen in unlabelled pretraining and in the age/sex/tissue and clock-distillation stages.
      The train group is reported separately and never pooled with the primary set.
  (2) External datasets. A metadata-only count on 24 Oct (before any fitting) found 9 corpus test/val datasets that
      qualify (fewer than 15), so GEO datasets absent from the corpus are added by this search, fixed here: every human
      DNA-methylation series of /Entropy/kying/GEO_data/gse_me_hs.csv (ClockBase's GEO index) whose platform field holds
      only accepted Illumina arrays, whose accession is not a corpus group, and none of whose samples (!Sample_geo_accession
      of the series matrix; local copy /Entropy/kying/GEO_data/GEO_hs_dnam/series_matrix, otherwise downloaded) is a
      corpus profile, and which qualifies under the label rule above. Series that share samples: the one with more
      contrast samples is kept (ties: lower accession). Beta values: the series-matrix values when deposited there (local
      ClockBase mirror GEO_hs_dnam/processed_data, used when non-empty), otherwise the deposited processed matrix of the
      GEO supplementary files (downloaded to artifacts/external/<GSE>/; the matrix of processed or normalized beta or M
      values; detection p-value columns, signal-intensity and IDAT files are not used). Value scale: matrices with
      negative values are M values, converted with 2^M / (1 + 2^M); matrices on a 0-100 scale are divided by 100.
      Probe ids: EPIC v2 suffixes are removed and replicate probes averaged. Columns are matched to GSMs through the
      series matrix (GSM id, sample title, source name, description, characteristics, or the Sentrix barcode of the
      sample's IDAT file names); only one-to-one matches are kept. Datasets without a usable deposited matrix, or below
      30 cases or controls after matching and QC, are excluded with the reason recorded. Group "external" (primary).
  The primary set is test + val + external.

  Freezing: stage `select` writes A/candidates_2026-10-24.csv from metadata only; `prepare` reads beta values and applies
  QC and arrays; `clocks` computes clocks that ClockBase lacks; `freeze` writes A/datasets_frozen_2026-10-24.csv (one row
  per dataset: accession, group, disease, disease group, tissue, array, case and control counts, clocks used and their
  source, training sizes evaluated, SHA-256 of the sorted profile list). The frozen table is committed before `fit`.

REPRESENTATIONS (each one the input of the same classifier)
  embedding   MethylGPT, PRIMARY. The 2,048-dimensional sample embedding of the paper's disease models: the age-stage
              encoder runs/aging_heads_clean_2026-10-01/best.pt (bundle weights_age_stage.pt; loaded as in
              extract_gs_features_2026-10-01.py, embed() of pretraining_frozen_probe_2026-10-12.py, TF32 matmul as in the
              analysis code), computed from each profile's observed vocabulary CpGs; unobserved CpGs stay unobserved.
  latent      MethylGPT, secondary. The 1,536 latent scores (input-layer posterior means, ppca_core_meanfilled_2026-09-30,
              rank 1,536): the first 1,536 embedding dimensions, which are the latent scores divided by their prior SD, so
              identical after per-feature standardization.
  raw         comparator. The 49,156 vocabulary CpGs, unobserved values filled with corpus training means (mu of
              ppca_core_meanfilled_2026-09-30.npz, as in si_followup_2026-10-22.py).
  clocks      comparator. GrimAge2, DunedinPACE, PhenoAge and Horvath (multi-tissue, Horvathv1) as one feature set of up
              to four clocks. For each dataset and clock: the ClockBase value (corpus profiles:
              clock_targets_2026-10-01.parquet; external series: ClockBase clock_result read with read_one() of
              build_clock_targets_2026-10-01.py; both keep a value only when at most 10% of the clock's CpGs are missing)
              when ClockBase covers >= 90% of the contrast profiles; otherwise the value computed with biolearn 0.7.1 (the
              library whose model names and outputs ClockBase reports) with the model's default imputation, on all
              GSM-matched profiles of the deposited full-array matrix, kept under the same 10% rule. GrimAge2 follows
              ClockBase's convention: biolearn GrimAgeV2 with the Horvathv1 DNAm age as its age input and every profile
              scored as male (Female = 0). This convention was found before any fitting, on GSE80970 (dev split, both
              sources available): ClockBase's GrimAge "Age" column equals its Horvathv1 value and "Female" is 0 for all
              profiles, and with this convention biolearn reproduces ClockBase's GrimAge2 (r = 0.997; Horvathv1 r = 0.9997,
              DunedinPACE r = 0.994, PhenoAge r = 0.93). Stage `clocks --validate` writes this check for the 450K
              datasets GSE80970, GSE97362 and GSE110607 to A/clock_validation_2026-10-24.csv. Deposited matrices for
              clocks are read for primary datasets, for train-group datasets with a non-empty local ClockBase mirror, and
              for train-group datasets downloaded for analysis B. A clock enters a
              dataset's set when available for >= 90% of its contrast profiles; remaining missing values take the mean of
              the training draw. On 27K data GrimAge2 and DunedinPACE fail the 10% rule, so 27K datasets use the clocks
              that remain. A dataset without any clock is left out of comparisons that involve clocks (listed).
  covariates  descriptive only, not a comparator: metadata age and sex (male = 1), for datasets where both are recorded
              for >= 90% of contrast profiles; missing values take the training-draw mean.
  The comparators of the rule are raw and clocks (plan: mean filling, the same predictors without MethylGPT, published
  clocks). The latent scores are a MethylGPT output and are reported as MethylGPT (secondary), not as a comparator.

CLASSIFIER (identical for every representation)
  Logistic elastic net with an unpenalized intercept; features standardized with the mean and SD of the fitting samples
  (features with SD < 1e-6 dropped); objective mean log loss + lambda * (a |b|_1 + (1 - a) |b|^2 / 2). Fitted on GPU by
  batched proximal gradient (FISTA, gradient-based adaptive restart, step 1/L from a power-iteration bound), 300
  iterations per lambda (2,000 when the representation has <= 100 features), warm starts along the path. Grid:
  a in {0.1, 0.5, 0.9} x 10 lambdas log-spaced from lambda_max(a) to 1e-3 lambda_max(a), lambda_max computed on the whole
  training draw and used for its inner folds. Selection: stratified 5-fold CV within the training draw (StratifiedKFold,
  shuffle, random_state = draw index), mean held-out binomial deviance; ties go to the first setting in the order
  a ascending, lambda descending. Refit on the whole draw with the selected setting. Stage check_solver compares the
  objective with sklearn's saga solver on a sample of problems before `fit` (A/solver_check_2026-10-24.json).

TRAINING DRAWS AND SCORING
  Sizes n = 20, 50, 100, balanced: n/2 cases and n/2 controls drawn without replacement (class-stratified with equal
  allocation, so every dataset has the same number of labelled cases at a given size whatever its case fraction). 50
  draws per size and dataset, rng = numpy default_rng([20261024, accession number, n, draw]); the same draws for every
  representation (paired). A size is evaluated only when the dataset keeps >= 10 cases and >= 10 controls for scoring
  (n/2 + 10 of each); skipped sizes are recorded. Scoring on all remaining contrast profiles: AUC per draw; the dataset
  value is the mean AUC over the 50 draws.

PRIMARY METRIC AND DECISION (plan, analysis A and Decision rule)
  M(rep) = mean over the primary datasets evaluable at n = 50 of (AUC - 0.5). MethylGPT = embedding. The best comparator
  C* is the comparator (raw or clocks) with the larger M on the same datasets. If clocks are unavailable for some primary
  dataset, the primary set is restricted to datasets where every representation is available, and the others are
  reported. Relative gain = (M(embedding) - M(C*)) / M(C*). 95% interval of M(embedding) - M(C*): paired bootstrap over
  datasets, 2,000 draws, seed 20261024, one resample of datasets per draw shared by all representations, C* fixed as on
  the full set. PASS if and only if the relative gain is >= 10% and the lower end of the interval is > 0; otherwise FAIL
  (then Fig. 6d stays the 27K comparator heatmap).

SECONDARY (all reported): the same quantities at n = 20 and n = 100; latent scores in place of the embedding; MethylGPT
minus each comparator separately; the train group (same metrics, separately); subgroups of the primary set (test, val,
external; 27K against the other arrays; disease groups); per-dataset AUC per representation and size with the SD over
draws and the number of draws whose selected model has no nonzero coefficient; the covariate model.

CODE FIXES AFTER THE FIRST COMMIT (f95b72a), 24 Oct, all before the dataset list was frozen and before any model was
fitted; the design above is unchanged:
  (1) select: the profile table is joined to the labels on id only (column clash). (2) Deposited matrices: the header
  search skips series-matrix "!" lines, GenomeStudio "[...]" blocks and lines of empty fields; signal-intensity and
  probe-id columns are excluded; space-separated matrices and headers without a probe-id column name are read; column
  names such as "Beta value for MTG 08--88" are matched without the value prefix; sentinel values (|v| > 1e30, e.g. -3.4e38) are missing. Series whose values are in the
  series matrix but have no file in the ClockBase mirror (GSE201016, GSE26126, GSE37988, GSE49996) are read from the
  local series matrix (the series-matrix values of the rule above). (3) GSM matching also uses the parts of a sample field
  before ":" and inside brackets, and the field without its bracketed parts (titles such as "CTRL1 [methylation]").
  (4) A dataset without any GSM-matched labelled profile is recorded as excluded instead of written empty. (5) The
  solver returns coefficient paths only for the whole-draw problems (memory). (6) The clocks stage skips datasets whose
  deposited matrix has no GSM-matched profile. (7) Secondary rows added to the summary (before fitting): MethylGPT
  against raw alone on every dataset of a set, including datasets without clocks (part of "MethylGPT minus each
  comparator separately"). (8) Solver speed, before the first complete fit (an earlier partial run was discarded and
  refitted): the FISTA update is written with fewer passes over memory (same algorithm and step sizes; soft
  thresholding as z - clip(z, -t, t)); columns constant over all rows of a dataset (dropped by the SD rule in every
  problem anyway) are removed before fitting; for representations without missing values all draws of a size share the
  dataset matrix in one batch (problems differ only in their 0/1 row weights; arithmetic unchanged); representations
  with missing values (clocks, covariates) keep one batch element per draw; data handling and scoring of complete
  representations run on the GPU; the iterations of a single batch element use fused numba CUDA kernels
  (large_gain_enet_cuda_2026-10-24.py: same arithmetic, checked against the torch loop: same selected settings, scores
  within 3e-4). Stage fit takes
  --shard k/K to split datasets across processes (execution only).
  (9) Data validity, found when the first fit of GSE49996 failed on non-finite scores (no result had been looked at; all
  fits were then discarded and rerun): ClockBase's clock values for some series were computed from inputs that are
  not beta values (M values: GSE102468, GSE50759, GSE52635, GSE89251; 0-100 scale: GSE156994) or that contain the
  GEO missing-value sentinel -3.4e38 (GSE37988, GSE49996), giving values such as Horvath -1 for every profile or
  1e40 years. ClockBase values are therefore used only when ClockBase's input matrix is on the beta scale without
  sentinels (clockbase_input_ok); otherwise the clocks are computed with biolearn from that matrix after scale
  conversion. The scale rule is made explicit: M values when > 1% of values are negative and > 1% exceed 1; 0-100
  scale when the median exceeds 1 and < 1% are negative; otherwise beta (the earlier rule called GSE26126, a beta
  matrix with 4% slightly negative values, M values; its profiles are re-read). Clock inputs follow the profile QC
  rule (clipped to [0, 1]).

Stages: select, prepare, clocks (run with /home/kying/anaconda3/bin/python, which has biolearn; no torch import),
freeze, embed (GPU), check_solver (GPU), fit (GPU), analyse.
Outputs: artifacts/large_gain_2026-10-24/A/ (README.md, candidates, frozen datasets, clock validation, solver check,
per-draw and per-dataset AUC, summary, rule outcome). Working files (beta matrices, embeddings) in
artifacts/large_gain_2026-10-24/work_AB/, not committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
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
CORPUS_DIR = ART / "corpus_2026-09-30"
OUT = ART / "large_gain_2026-10-24"
OUTA = OUT / "A"
WORK = OUT / "work_AB"
EXT = ART / "external"
GEO = Path("/Entropy/kying/GEO_data/GEO_hs_dnam")
GSE_INDEX = Path("/Entropy/kying/GEO_data/gse_me_hs.csv")
META = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/compiled_metadata.csv")
CORE = ART / "ppca_core_meanfilled_2026-09-30.npz"
AGE_STAGE = ART / "runs/aging_heads_clean_2026-10-01/best.pt"
MASKS = ART / "platform_vocab_masks_2026-10-01.npz"
BIOLEARN_DATA = Path("/home/kying/anaconda3/lib/python3.12/site-packages/biolearn/data")
BIOLEARN_PY = "/home/kying/anaconda3/bin/python"
DATE = "2026-10-24"

ACCEPTED_GPL = {"GPL8490": "27K", "GPL13534": "450K", "GPL16304": "450K", "GPL18809": "450K",
                "GPL21145": "EPIC", "GPL23976": "EPIC", "GPL33022": "EPIC v2"}
CONTROL_LABELS = {"healthy", "control", "normal"}
NONSPECIFIC = {"nan", "", "na", "n/a", "none", "unknown", "disease", "other", "case", "negative", "positive"}
EXPOSURES = {"smoker"}
SYNONYMS = {"tumour": "tumor", "papillary thyroid carcinoma": "papillary thyroid cancer",
            "gestational diabetes": "gestational diabetes mellitus", "ptsd": "post-traumatic stress disorder"}
CLOCKS = ["GrimAge2", "DunedinPACE", "PhenoAge", "Horvathv1"]
BIOLEARN_NAME = {"GrimAge2": "GrimAgeV2", "DunedinPACE": "DunedinPACE", "PhenoAge": "PhenoAge", "Horvathv1": "Horvathv1"}
REPS = ["embedding", "latent", "raw", "clocks", "covariates"]
MGPT, COMPARATORS = "embedding", ["raw", "clocks"]
SIZES = [20, 50, 100]
PRIMARY_SIZE = 50
N_DRAWS = 50
ALPHAS = [0.1, 0.5, 0.9]
N_LAMBDA, LAMBDA_MIN_RATIO = 10, 1e-3
FOLDS = 5
MIN_CASE = MIN_CTRL = 30
MIN_SCORE = 10
AVAIL = 0.90
BOOT, SEED = 2000, 20261024
PRIMARY_GROUPS = ["test", "val", "external"]
FORCE_PER_DRAW = False      # layout check only
FUSED = True                # fused CUDA kernels for single-batch-element GPU fits


_FUSED_MOD = []


def _fused():
    if not _FUSED_MOD:
        _FUSED_MOD.append(_module("large_gain_enet_cuda", "large_gain_enet_cuda_2026-10-24.py"))
    return _FUSED_MOD[0]

# Disease groups, assigned from the disease label alone (written before any fitting).
DISEASE_GROUP = {
    "cancer and neoplasia": [
        "acute lymphoblastic leukemia", "aml/mds", "barrett's esophagus", "bladder cancer", "breast cancer", "cancer",
        "colorectal cancer", "esophageal squamous cell carcinoma", "gastric cancer", "head and neck squamous cell carcinoma",
        "hepatocellular carcinoma", "intestinal metaplasia", "intraductal papillary neoplasm of the bile duct",
        "lung adenocarcinoma", "lung cancer", "lung squamous cell carcinoma", "oral pharyngeal carcinoma",
        "oral squamous cell carcinoma", "pancreatic ductal adenocarcinoma", "papillary thyroid cancer", "pre-b all",
        "prostate cancer", "retinoblastoma", "serous carcinoma", "serrated adenocarcinoma", "squamous cell carcinoma",
        "tumor", "uterine leiomyoma", "wilms' tumor", "colon cancer", "non-small cell lung cancer"],
    "neurological and psychiatric": [
        "alzheimer's disease", "anxiety disorder", "autism spectrum disorder", "bipolar disorder", "creutzfeldt-jakob disease",
        "depression", "first episode psychosis", "frontotemporal dementia", "huntington's disease", "major depressive disorder",
        "mild cognitive impairment", "multiple system atrophy", "myalgic encephalomyelitis/chronic fatigue syndrome",
        "obsessive-compulsive disorder", "panic disorder", "parkinson's disease", "post-traumatic stress disorder",
        "progressive supranuclear palsy", "schizophrenia", "social anxiety disorder", "intracerebral hemorrhage"],
    "autoimmune and inflammatory": [
        "allergy", "asthma", "atopic asthma", "chronic obstructive pulmonary disease", "crohn's disease", "flare",
        "multiple sclerosis", "psoriasis", "rheumatoid arthritis", "systemic lupus erythematosus", "recurrent wheezing",
        "idiopathic pulmonary fibrosis", "ards"],
    "metabolic and cardiovascular": [
        "diabetic nephropathy", "gestational diabetes mellitus", "hypertension", "nonalcoholic fatty liver disease",
        "polycystic ovary syndrome", "prediabetes", "systemic insulin resistance", "severe acute malnutrition"],
    "infection": [
        "acute chorioamnionitis", "chronic chagasic cardiomyopathy", "chronic hiv infection", "hepatitis c, hiv, drug use",
        "hiv", "multisystem inflammatory syndrome in children"],
    "genetic and developmental": [
        "chd7_variant", "down syndrome", "fasd", "lynch syndrome", "malformations of cortical development",
        "non-syndromic cleft lip and palate", "oculo-auriculo-vertebral spectrum", "sotos syndrome", "williams syndrome",
        "preterm"],
}
DISEASE_GROUP_OF = {d: g for g, ds in DISEASE_GROUP.items() for d in ds}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ labels and contrasts
def norm_disease(s):
    s = s.astype(str).str.strip().str.lower().str.replace("’", "'", regex=False).str.replace(r"\s+", " ", regex=True)
    return s.map(lambda x: SYNONYMS.get(x, x))


def labels_table():
    m = pd.read_csv(need(META), low_memory=False, dtype=str)
    m["disease_n"] = norm_disease(m.disease)
    m["tissue_n"] = m.tissue.fillna("unlabelled").astype(str).str.strip()
    for c in ("case_control", "vivo_vitro", "cell_line"):
        m[c] = m[c].fillna("")
    key = ["disease_n", "case_control", "tissue_n", "vivo_vitro"]
    agree = m.groupby("GSM_ID")[key].nunique(dropna=False).max(axis=1) == 1
    n_conflict = int((~agree).sum())
    m = m[m.GSM_ID.isin(agree.index[agree])].drop_duplicates("GSM_ID").copy()
    invivo = (m.vivo_vitro == "in_vivo") & (m.cell_line == "")
    ctrl = m.disease_n.isin(CONTROL_LABELS) & (m.case_control != "case") & invivo
    case = ~m.disease_n.isin(CONTROL_LABELS | NONSPECIFIC | EXPOSURES) & (m.case_control != "ctrl") & invivo
    m["role"] = np.where(ctrl, "control", np.where(case, "case", "none"))
    m["age_n"] = pd.to_numeric(m.age, errors="coerce")
    m["sex_n"] = m.sex.map({"M": 1.0, "F": 0.0, "male": 1.0, "female": 0.0, "Male": 1.0, "Female": 0.0})
    lab = m.set_index("GSM_ID")[["disease_n", "tissue_n", "role", "age_n", "sex_n"]]
    lab.columns = ["disease", "tissue", "role", "age", "sex"]
    return lab, n_conflict


def choose_contrast(lab):
    """lab: profiles with columns role, disease, tissue. Returns the contrast dict or None, plus the other qualifying strata."""
    strata = []
    for t, b in lab.groupby("tissue"):
        nc = int((b.role == "control").sum())
        if nc < MIN_CTRL:
            continue
        for d, k in b[b.role == "case"].disease.value_counts().items():
            if k >= MIN_CASE:
                strata.append(((-int(k), -nc, str(t), str(d)), dict(tissue=t, disease=d, n_case=int(k), n_ctrl=nc)))
    if not strata:
        return None, ""
    strata.sort(key=lambda z: z[0])
    other = ";".join(f"{s['disease']}|{s['tissue']}:{s['n_case']}/{s['n_ctrl']}" for _, s in strata[1:])
    return strata[0][1], other


def contrast_mask(lab, c):
    return (lab.tissue == c["tissue"]) & (((lab.role == "case") & (lab.disease == c["disease"])) | (lab.role == "control"))


# ------------------------------------------------------------------ series matrices
def series_matrix_path(gse):
    p = GEO / "series_matrix" / f"{gse}_series_matrix.txt.gz"
    if p.exists():
        return p
    q = EXT / gse / f"{gse}_series_matrix.txt.gz"
    if not q.exists():
        q.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/{gse}_series_matrix.txt.gz"
        r = subprocess.run(["curl", "-sS", "--fail", "-o", str(q), url], capture_output=True)
        if r.returncode != 0:
            q.unlink(missing_ok=True)
            return None
    return q


def read_series_samples(path):
    """!Sample_ fields of a series matrix: dict field -> list (per sample) of lists of values."""
    fields = {}
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0][1:]
            vals = [v.strip().strip('"') for v in parts[1:]]
            fields.setdefault(key, []).append(vals)
    return fields


def series_gsms(path):
    f = read_series_samples(path)
    return f.get("Sample_geo_accession", [[]])[0]


# ------------------------------------------------------------------ stage select
def stage_select():
    OUTA.mkdir(parents=True, exist_ok=True)
    lab, n_conflict = labels_table()
    prof = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"))
    man = json.loads(need(CORPUS_DIR / "split_manifest.json").read_text())
    test_g, dev_g = set(man["test_groups"]), set(man["dev_groups"])
    prof = prof[prof.split != "excluded_qc"]
    corpus_ids, corpus_groups = set(prof.id.astype(str)), set(prof.group.astype(str))
    rows = []
    j = prof[["id", "group", "gpl", "split"]].join(lab, on="id", how="inner")
    for g, b in j.groupby("group"):
        c, other = choose_contrast(b)
        if c is None:
            continue
        grp = "test" if g in test_g else "val" if g in dev_g else "train"
        gpl = " ".join(sorted(set(" ".join(b.gpl.dropna().astype(str)).split())))
        rows.append(dict(accession=g, source="corpus", group=grp, platform=gpl, **c, other_strata=other))
    log(f"corpus: {len(rows)} candidate datasets ({sum(r['group'] != 'train' for r in rows)} test/val)")
    # external search
    idx = pd.read_csv(need(GSE_INDEX))
    idx = idx[idx.accession.astype(str).str.startswith("GSE")]
    ext_rows, searched = [], {"index_series": int(len(idx)), "accepted_platform": 0, "not_corpus_group": 0,
                              "series_matrix_found": 0, "no_corpus_samples": 0, "qualifying": 0}
    for _, r in idx.iterrows():
        gse, plat = str(r.accession), str(r.platform)
        gpls = plat.split()
        if not gpls or any(p not in ACCEPTED_GPL for p in gpls):
            continue
        searched["accepted_platform"] += 1
        if gse in corpus_groups:
            continue
        searched["not_corpus_group"] += 1
        sm = series_matrix_path(gse)
        if sm is None:
            continue
        searched["series_matrix_found"] += 1
        gsms = series_gsms(sm)
        if any(s in corpus_ids for s in gsms):
            continue
        searched["no_corpus_samples"] += 1
        b = lab.reindex([s for s in gsms if s in lab.index])
        c, other = choose_contrast(b)
        if c is None:
            continue
        searched["qualifying"] += 1
        ext_rows.append(dict(accession=gse, source="external", group="external", platform=" ".join(gpls), **c,
                             other_strata=other, _gsms=set(gsms)))
    # series sharing samples: keep the one with more contrast samples (ties: lower accession)
    ext_rows.sort(key=lambda r: (-(r["n_case"] + r["n_ctrl"]), r["accession"]))
    kept, seen = [], set()
    for r in ext_rows:
        if r["_gsms"] & seen:
            r["excluded"] = "shares samples with a larger external series"
        seen |= r["_gsms"]
        kept.append(r)
    for r in kept:
        r.pop("_gsms")
    cand = pd.DataFrame(rows + kept)
    cand["disease_group"] = cand.disease.map(DISEASE_GROUP_OF).fillna("other")
    if "excluded" not in cand:
        cand["excluded"] = np.nan
    cand = cand.sort_values(["group", "accession"]).reset_index(drop=True)
    cand.to_csv(OUTA / f"candidates_{DATE}.csv", index=False)
    write_json(OUTA / f"candidates_{DATE}.json", {"metadata": str(META), "metadata_sha256": sha256(META),
                                                  "metadata_gsm_conflicts_dropped": n_conflict, "external_search": searched,
                                                  "gse_index": str(GSE_INDEX), "gse_index_sha256": sha256(GSE_INDEX),
                                                  "counts": cand.groupby("group").size().to_dict()})
    log(f"candidates written: {cand.groupby('group').size().to_dict()}; external search {searched}")


# ------------------------------------------------------------------ deposited matrices (external and clocks)
VALUE_SUFFIX = re.compile(r"[\s._-]*(avg[\s._-]*beta|average[\s._-]*beta|beta[\s._-]*values?|b[\s._-]*values?|beta|"
                          r"m[\s._-]*values?|mvalue|signal|unmethylated|methylated)$", re.I)
DETECTION = re.compile(r"det(ection)?[\s._-]*p|p[\s._-]*val|pval|intensit|signal|probeid", re.I)   # non-value columns
SENTRIX = re.compile(r"(\d{9,12}_R\d{2}C\d{2})")


def deposited_file(acc):
    """The deposited matrix used for a dataset: the local series-matrix mirror if non-empty, else the downloaded
    supplementary matrix (artifacts/external/<GSE>/, files listed in DEPOSITED)."""
    local = GEO / "processed_data" / f"{acc}.csv.gz"
    if local.exists() and local.stat().st_size > 100_000:
        return [local]
    sm = GEO / "series_matrix" / f"{acc}_series_matrix.txt.gz"
    if sm.exists() and series_matrix_has_table(sm):
        return [sm]
    names = list(DEPOSITED.get(acc, []))
    reg = EXT / acc / "DEPOSITED.txt"            # files downloaded for analysis B
    if not names and reg.exists():
        names = [x.strip() for x in reg.read_text().splitlines() if x.strip()]
    files = [EXT / acc / f for f in names]
    return [f for f in files if f.exists()] or None


# supplementary processed matrices (names from the GEO suppl/ listings, 24 Oct 2026)
DEPOSITED = {
    "GSE175458": ["GSE175458_FinalBetas_sysbio_547samps.csv.gz"],
    "GSE179759": ["GSE179759_Matrix_processed.csv.gz"],
    "GSE180474": ["GSE180474_MatrixProcessed.csv.gz"],
    "GSE185090": ["GSE185090_processed.txt.gz"],
    "GSE191082": ["GSE191082_Normalized_data_Blood.txt.gz"],
    "GSE192353": ["GSE192353_Methyl_RAW.csv.gz"],
    "GSE193879": ["GSE193879_Matrix_processed.csv.gz"],
    "GSE195834": ["GSE195834_2-PD-processed.csv.gz"],
    "GSE199057": ["GSE199057_processed_methlyation_matrix.txt.gz"],
    "GSE199334": ["GSE199334_MatrixProcessed.txt.gz"],
    "GSE199700": ["GSE199700_Matrix_processed.txt.gz"],
    "GSE200659": ["GSE200659_avg_beta.txt.gz"],
    "GSE201287": ["GSE201287_matrix_norm.txt.gz"],
    "GSE203332": ["GSE203332_Matrix_processed_dasen_betas.txt.gz"],
    "GSE74414": ["GSE74414_all_beta_values.txt.gz"],
    "GSE75704": ["GSE75704_normalized.txt.gz"],
    "GSE107351": ["GSE107351_Matrix_Processed_blood_Bvalues.txt.gz"],
    "GSE115508": ["GSE115508_aCA_averagebetas_processed_amnion.csv.gz", "GSE115508_aCA_averagebetas_processed_chorion.csv.gz",
                  "GSE115508_aCA_averagebetas_processed_villi.csv.gz"],
    "GSE145361": ["GSE145361_Vallerga2020_NCOMMS_AvgBeta_Matrix-file.txt.gz"],
}


def clock_cpg_set():
    s = set()
    for f, col in (("Horvath1.csv", 0), ("PhenoAge.csv", 0), ("DunedinPACE.csv", 0)):
        s |= set(pd.read_csv(need(BIOLEARN_DATA / f)).iloc[:, col].astype(str))
    s |= set(pd.read_csv(need(BIOLEARN_DATA / "GrimAgeV2.csv"))["var"].astype(str))
    s |= set(pd.read_csv(need(BIOLEARN_DATA / "DunedinPACE_Gold_Means.csv"), index_col=0).index.astype(str))
    return {x for x in s if x.startswith(("cg", "ch"))}


def series_matrix_has_table(path):
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                next(fh, None)
                row = next(fh, "")
                return bool(row) and not row.startswith("!series_matrix_table_end")
    return False


def _sniff(path):
    """Lines before the table header (comments, blank lines, series-matrix '!' lines, GenomeStudio [Header] blocks)
    and the separator."""
    with gzip.open(path, "rt", errors="replace") as fh:
        skip = 0
        for line in fh:
            sep = max(["\t", ",", ";"], key=line.count)
            if line.count(sep) < 2 and line.count(" ") >= 2 and not line.startswith(("#", '"#', "[", "!")):
                sep = " "                                  # space-separated matrices (e.g. GSE203332)
            if line.startswith(("#", '"#', "[", "!")) or not line.replace(sep, "").strip() or line.count(sep) < 2:
                skip += 1
                continue
            break
    return skip, sep


def read_matrix(path, keep_rows):
    """Deposited matrix restricted to keep_rows (probe ids, suffixes removed, replicates averaged); value columns only."""
    skip, sep = _sniff(path)
    head = pd.read_csv(path, sep=sep, skiprows=skip, nrows=0)
    cols = list(head.columns)
    first = pd.read_csv(path, sep=sep, skiprows=skip + 1, nrows=1, header=None)
    kw = {}
    if first.shape[1] == len(cols) + 1:          # header without a name for the probe-id column (e.g. GSE134379)
        cols = ["__probe_id__"] + cols
        kw = {"header": None, "names": cols, "skiprows": skip + 1}
    idcol = cols[0]
    val = [c for c in cols[1:] if not DETECTION.search(str(c)) and not str(c).startswith("Unnamed")]
    parts = []
    for ch in pd.read_csv(path, sep=sep, usecols=[idcol] + val, chunksize=50_000, low_memory=False,
                          na_values=["null", "NULL", "NA", "NaN", ""], **({"skiprows": skip} | kw)):
        ids = ch[idcol].astype(str).str.strip().str.strip('"').str.replace(r"_[A-Z]{2}\d+$", "", regex=True)
        ch = ch.drop(columns=[idcol])
        ch.index = ids
        ch = ch[ch.index.isin(keep_rows)]
        if len(ch):
            ch = ch.apply(pd.to_numeric, errors="coerce")
            parts.append(ch.where(ch.abs() < 1e30).astype(np.float32))     # sentinels such as -3.4e38 are missing
    df = pd.concat(parts) if parts else pd.DataFrame(columns=val, dtype=np.float32)
    if df.index.duplicated().any():
        df = df.groupby(level=0).mean()
    return df


def value_scale(v):
    """M values: > 1% of values negative and > 1% above 1; 0-100 scale: median above 1 and < 1% negative; else beta."""
    fin = v[np.isfinite(v)]
    if not fin.size:
        return "beta"
    neg, gt1 = (fin < 0).mean(), (fin > 1).mean()
    if neg > 0.01 and gt1 > 0.01:
        return "M"
    if np.median(fin) > 1 and neg < 0.01:
        return "percent"
    return "beta"


def to_beta(df):
    scale = value_scale(df.to_numpy(np.float64))
    if scale == "M":
        df = 2.0 ** df / (1.0 + 2.0 ** df)
    elif scale == "percent":
        df = df / 100.0
    return df.astype(np.float32), scale


def clockbase_input_ok(acc):
    """ClockBase's clocks are used only when its input matrix (the local mirror, else the local series matrix) is on the
    beta scale and has no missing-value sentinels; first 20,000 probes checked. No local input: not checkable, used."""
    f = GEO / "processed_data" / f"{acc}.csv.gz"
    if not (f.exists() and f.stat().st_size > 100_000):
        f = GEO / "series_matrix" / f"{acc}_series_matrix.txt.gz"
        if not (f.exists() and series_matrix_has_table(f)):
            return True, "no local ClockBase input"
    skip, sep = _sniff(f)
    d = pd.read_csv(f, sep=sep, skiprows=skip, index_col=0, nrows=20000).apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if (np.abs(d[np.isfinite(d)]) >= 1e30).any():
        return False, "sentinel values in ClockBase input"
    sc = value_scale(d)
    return (sc == "beta"), f"ClockBase input scale {sc}"


def match_columns(columns, sm_path):
    """Map matrix columns to GSMs (one-to-one) through the series matrix."""
    f = read_series_samples(sm_path)
    gsm = f["Sample_geo_accession"][0]
    keys = {g: set() for g in gsm}
    for field, lists in f.items():
        if field in ("Sample_geo_accession",):
            continue
        for vals in lists:
            for g, v in zip(gsm, vals):
                if not v:
                    continue
                keys[g].add(v)
                if ":" in v:
                    keys[g].update(x.strip() for x in v.split(":", 1))
                for s in SENTRIX.findall(v):
                    keys[g].add(s)
                for s in re.findall(r"\[([^\]]+)\]|\(([^)]+)\)", v):     # e.g. title "Placenta_GDM_Yes_Sample1 [CM5_167]"
                    keys[g].update(x.strip() for x in s if x)
                keys[g].add(re.sub(r"\s*(\[[^\]]*\]|\([^)]*\))\s*", "", v).strip())   # title without bracketed parts
    for g in gsm:
        keys[g].add(g)
    inv, inv_ci = {}, {}
    for g, ks in keys.items():
        for k in ks:
            inv.setdefault(k, set()).add(g)
            inv_ci.setdefault(k.lower(), set()).add(g)
    out, method = {}, {}
    for c in columns:
        c0 = str(c).strip()
        c1 = re.sub(r"^(avg[\s._-]*)?(beta|m)[\s._-]*values?[\s._-]*(for[\s._-]+)?", "", c0, flags=re.I).strip()
        cands = [c0, VALUE_SUFFIX.sub("", c0).strip(), c1, *SENTRIX.findall(c0)]
        hit = None
        for k in cands:
            for table, kk in ((inv, k), (inv_ci, k.lower())):
                gs = table.get(kk)
                if gs and len(gs) == 1:
                    hit = next(iter(gs))
                    break
            if hit:
                break
        if hit:
            out[c] = hit
    vc = pd.Series(list(out.values())).value_counts()
    dup = set(vc.index[vc > 1])
    out = {c: g for c, g in out.items() if g not in dup}
    return out


def load_deposited(acc, keep_rows):
    """(GSM-indexed DataFrame probes x samples, info dict) from the deposited matrix files of a dataset."""
    files = deposited_file(acc)
    if not files:
        return None, {"deposited": "none"}
    sm = series_matrix_path(acc)
    frames, info = [], {"files": [str(f) for f in files]}
    for f in files:
        df = read_matrix(f, keep_rows)
        if f.parent == GEO / "processed_data" or f.name.endswith("_series_matrix.txt.gz"):
            mapping = {c: c for c in df.columns if str(c).startswith("GSM")}
        else:
            mapping = match_columns(df.columns, sm)
        info[f"{f.name}|columns"] = int(df.shape[1])
        info[f"{f.name}|matched"] = int(len(mapping))
        df = df[list(mapping)].rename(columns=mapping)
        frames.append(df)
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    df, scale = to_beta(df)
    info["scale"] = scale
    info["probes"] = int(df.shape[0])
    return df, info


# ------------------------------------------------------------------ stage prepare
def platform_array(platform):
    arrs = {ACCEPTED_GPL[p] for p in str(platform).split() if p in ACCEPTED_GPL}
    return next(iter(arrs)) if len(arrs) == 1 else None


def infer_array(X, masks):
    obs = np.isfinite(X)
    f450 = obs[:, masks["hm450"] & ~masks["hm27"]].mean(1)
    fv2 = obs[:, masks["epic_v2"] & ~masks["epic_v1"]].mean(1)
    fe1 = obs[:, masks["epic_v1"] & ~masks["hm450"]].mean(1)
    return np.where(f450 < 0.5, "27K", np.where(fv2 >= 0.5, "EPIC v2", np.where(fe1 >= 0.5, "EPIC", "450K")))


def stage_prepare(only=None):
    WORK.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(need(OUTA / f"candidates_{DATE}.csv"))
    cand = cand[cand.excluded.isna()]
    if only:
        cand = cand[cand.accession.isin(only)]
    lab, _ = labels_table()
    z = np.load(need(MASKS))
    vocab = z["vocab"].astype(str)
    masks = {k: z[k] for k in ("hm27", "hm450", "epic_v1", "epic_v2")}
    vpos = {c: i for i, c in enumerate(vocab)}
    ccpg = clock_cpg_set()
    keep_rows = set(vocab) | ccpg
    prof = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"))
    prof = prof[prof.split != "excluded_qc"]
    beta = np.load(need(CORPUS_DIR / "beta_f16.npy"), mmap_mode="r")
    ct = pd.read_parquet(need(CORPUS_DIR / "clock_targets_2026-10-01.parquet"), columns=CLOCKS)
    status = []
    for _, c in cand.iterrows():
        acc = c.accession
        f_x = WORK / f"X_{acc}.npy"
        f_c = WORK / f"clockcpg_{acc}.parquet"
        prepared = f_x.exists() and (WORK / f"profiles_{acc}.csv").exists()
        if prepared and (c.source == "external" or f_c.exists() or not deposited_file(acc)):
            continue
        t0 = time.time()
        info = {"accession": acc}
        dep = None
        if c.source == "external":
            dep, dinfo = load_deposited(acc, keep_rows)
            info.update(dinfo)
        else:
            # corpus datasets: the deposited full-array matrix is read only for clocks ClockBase lacks
            b = prof[(prof.group == acc) & prof.id.isin(lab.index)]
            b = b[lab.reindex(b.id).role.isin(["case", "control"]).to_numpy()]
            lacking = [k for k in CLOCKS if ct.loc[b.index, k].notna().mean() < AVAIL]
            if not clockbase_input_ok(acc)[0]:
                lacking = list(CLOCKS)
            if lacking and deposited_file(acc):
                dep, dinfo = load_deposited(acc, ccpg)
                info.update(dinfo)
            info["clockbase_lacking"] = ";".join(lacking)
            if prepared:
                if dep is not None:
                    dep.to_parquet(f_c)
                    log(f"{acc}: clock inputs written ({dep.shape[1]} deposited profiles)")
                continue
        if c.source == "corpus":
            b = prof[prof.group == acc]
            b = b[b.id.isin(lab.index)]
            rows = np.sort(b.index.to_numpy())
            ids = prof.loc[rows, "id"].astype(str).to_numpy()
            X = np.asarray(beta[rows], dtype=np.float32)
            frac_out = np.zeros(len(ids))
            corpus_rows = rows
        else:
            if dep is None:
                status.append({**info, "status": "excluded: no usable deposited matrix"})
                continue
            ids = np.array([g for g in dep.columns if g in lab.index])
            V = dep.reindex(index=vocab, columns=ids).to_numpy(np.float32).T
            fin = np.isfinite(V)
            frac_out = ((V < 0) | (V > 1)).sum(1) / np.maximum(fin.sum(1), 1)
            X = np.clip(V, 0, 1)
            corpus_rows = np.full(len(ids), -1)
        if len(ids) == 0:
            status.append({**info, "status": "excluded: no GSM-matched labelled profiles"})
            log(f"{acc}: no GSM-matched labelled profiles")
            continue
        L = lab.reindex(ids)
        arr = platform_array(c.platform)
        arrays = np.full(len(ids), arr) if arr else infer_array(X, masks)
        pt = pd.DataFrame({"id": ids, "corpus_row": corpus_rows, "role": L.role.to_numpy(), "disease": L.disease.to_numpy(),
                           "tissue": L.tissue.to_numpy(), "age": L.age.to_numpy(), "sex": L.sex.to_numpy(),
                           "array": arrays, "frac_out_of_range": frac_out,
                           "n_obs": np.isfinite(X).sum(1)})
        pt["qc_pass"] = (pt.frac_out_of_range <= 0.01) & (pt.n_obs > 0)
        np.save(f_x, X.astype(np.float16))
        pt.to_csv(WORK / f"profiles_{acc}.csv", index=False)
        if dep is not None:
            cl = dep.reindex(index=sorted(clock_cpg_set() & set(dep.index)))
            cl.to_parquet(WORK / f"clockcpg_{acc}.parquet")
        status.append({**info, "status": "prepared", "profiles": int(len(ids)), "seconds": round(time.time() - t0, 1)})
        log(f"{acc}: {len(ids)} labelled profiles, arrays {pd.Series(arrays).value_counts().to_dict()}, {time.time() - t0:.0f} s")
    if status:
        old = WORK / "prepare_status.csv"
        st = pd.DataFrame(status)
        if old.exists():
            st = pd.concat([pd.read_csv(old), st]).drop_duplicates("accession", keep="last")
        st.to_csv(old, index=False)


# ------------------------------------------------------------------ stage clocks (biolearn; anaconda python, no torch)
def stage_clocks(validate):
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery
    gal = ModelGallery()
    todo = sorted(p.name[len("clockcpg_"):-len(".parquet")] for p in WORK.glob("clockcpg_*.parquet"))

    def run(dnam):
        """Four clocks with biolearn defaults; GrimAge2 with ClockBase's convention (age = Horvathv1, all male)."""
        out = pd.DataFrame(index=dnam.columns)
        meta0 = pd.DataFrame({"age": np.nan, "sex": 2}, index=dnam.columns)
        horvath = gal.get("Horvathv1").predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
        for clock, name in BIOLEARN_NAME.items():
            model = gal.get(name)
            sites = [s for s in model.methylation_sites() if s.startswith(("cg", "ch"))]
            miss = 1 - dnam.reindex(sites).notna().sum(0) / len(sites)
            if clock == "GrimAge2":
                meta = pd.DataFrame({"age": horvath.to_numpy(float), "sex": 2}, index=dnam.columns)
                val = model.predict(GeoData(meta, dnam.copy()))["DNAmGrimAge"].reindex(dnam.columns)
            elif clock == "Horvathv1":
                val = horvath
            else:
                val = model.predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
            out[clock] = val.where(miss <= 0.10)
            out[f"{clock}_missing"] = miss
        return out

    for acc in todo:
        f = WORK / f"clocks_biolearn_{acc}.csv"
        if f.exists():
            continue
        dnam = pd.read_parquet(WORK / f"clockcpg_{acc}.parquet").astype(np.float64).clip(0, 1)   # profile QC rule
        if dnam.shape[1] == 0 or dnam.shape[0] == 0:
            log(f"{acc}: no GSM-matched deposited profiles for clocks")
            continue
        t0 = time.time()
        run(dnam).to_csv(f)
        log(f"{acc}: biolearn clocks for {dnam.shape[1]} profiles ({time.time() - t0:.0f} s)")
    if validate:
        try:
            import torch  # noqa: F401
        except ImportError:   # biolearn's interpreter has no torch: read_one() needs only pandas
            import types
            stub = types.ModuleType("methylgpt_lib")
            stub.CORPUS_DIR, stub.Corpus, stub.log, stub.write_json = CORPUS_DIR, None, log, write_json
            sys.modules["methylgpt_lib"] = stub
        bct = _module("build_clock_targets", "build_clock_targets_2026-10-01.py")
        rows = []
        for acc in validate:
            src = GEO / "processed_data" / f"{acc}.csv.gz"
            cb = bct.read_one(str(GEO / "clock_result" / f"{acc}.csv"))
            dnam = read_matrix(src, clock_cpg_set()).astype(np.float64).clip(0, 1)
            ours = run(dnam)
            for clock in CLOCKS:
                a, b = ours[clock], cb[clock].reindex(ours.index) if clock in cb else pd.Series(np.nan, index=ours.index)
                ok = a.notna() & b.notna()
                rows.append({"accession": acc, "clock": clock, "profiles_both": int(ok.sum()),
                             "pearson_r": float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                             "max_abs_diff": float((a[ok] - b[ok]).abs().max()) if ok.any() else np.nan,
                             "median_abs_diff": float((a[ok] - b[ok]).abs().median()) if ok.any() else np.nan})
            log(f"validation {acc} done")
        pd.DataFrame(rows).to_csv(OUTA / f"clock_validation_{DATE}.csv", index=False)


# ------------------------------------------------------------------ stage freeze
def dataset_clocks(acc, pt):
    """Clock values for the profiles of pt (rows aligned) and the source of each clock."""
    out = pd.DataFrame(index=range(len(pt)), columns=CLOCKS, dtype=float)
    src = {}
    cb = None
    if (pt.corpus_row >= 0).all():
        ct = pd.read_parquet(need(CORPUS_DIR / "clock_targets_2026-10-01.parquet"), columns=CLOCKS)
        cb = ct.loc[pt.corpus_row.to_numpy()].reset_index(drop=True)
    else:
        f = GEO / "clock_result" / f"{acc}.csv"
        if f.exists():
            bct = _module("build_clock_targets", "build_clock_targets_2026-10-01.py")
            d = bct.read_one(str(f))
            if d is not None:
                cb = d.reindex(columns=CLOCKS).reindex(pt.id.astype(str).to_numpy()).reset_index(drop=True)
    if cb is not None and not clockbase_input_ok(acc)[0]:
        cb = None
    bl = None
    fb = WORK / f"clocks_biolearn_{acc}.csv"
    if fb.exists():
        bl = pd.read_csv(fb, index_col=0).reindex(pt.id.astype(str).to_numpy()).reset_index(drop=True)
    for k in CLOCKS:
        if cb is not None and cb[k].notna().mean() >= AVAIL:
            out[k], src[k] = cb[k].to_numpy(float), "ClockBase"
        elif bl is not None and bl[k].notna().mean() >= AVAIL:
            out[k], src[k] = bl[k].to_numpy(float), "biolearn"
    return out, src


def accession_number(acc):
    d = re.sub(r"\D", "", acc)
    return int(d) if d else int(hashlib.sha256(acc.encode()).hexdigest()[:8], 16)


def stage_freeze():
    cand = pd.read_csv(need(OUTA / f"candidates_{DATE}.csv"))
    st = pd.read_csv(WORK / "prepare_status.csv") if (WORK / "prepare_status.csv").exists() else pd.DataFrame(columns=["accession"])
    rows = []
    for _, c in cand.iterrows():
        acc = c.accession
        base = dict(accession=acc, source=c.source, group=c.group, platform=c.platform)
        if isinstance(c.excluded, str):
            rows.append({**base, "included": False, "reason": c.excluded})
            continue
        fp = WORK / f"profiles_{acc}.csv"
        if not fp.exists():
            s = st[st.accession == acc]
            rows.append({**base, "included": False, "reason": s.status.iloc[0] if len(s) else "not prepared"})
            continue
        pt = pd.read_csv(fp)
        pt = pt[pt.qc_pass & pt.role.isin(["case", "control"])]
        cc, _ = choose_contrast(pt)
        if cc is None:
            rows.append({**base, "included": False, "reason": "below 30 cases or 30 controls after matching and QC"})
            continue
        sub = pt[contrast_mask(pt, cc)]
        arr = sub.array.value_counts().index[0]
        sub = sub[sub.array == arr]
        cc2, other = choose_contrast(sub)
        if cc2 is None or cc2["disease"] != cc["disease"] or cc2["tissue"] != cc["tissue"]:
            rows.append({**base, "included": False, "reason": "below 30 cases or 30 controls on the main array"})
            continue
        sub = sub[contrast_mask(sub, cc2)]
        clocks, src = dataset_clocks(acc, sub.reset_index(drop=True))
        nca, nco = int((sub.role == "case").sum()), int((sub.role == "control").sum())
        cov_ok = sub.age.notna().mean() >= AVAIL and sub.sex.notna().mean() >= AVAIL
        sizes = [n for n in SIZES if nca >= n // 2 + MIN_SCORE and nco >= n // 2 + MIN_SCORE]
        ids = sorted(sub.id.astype(str))
        rows.append({**base, "included": True, "reason": "", "disease": cc2["disease"],
                     "disease_group": DISEASE_GROUP_OF.get(cc2["disease"], "other"), "tissue": cc2["tissue"], "array": arr,
                     "n_case": nca, "n_ctrl": nco, "clocks_used": ";".join(k for k in CLOCKS if k in src),
                     "clock_source": ";".join(f"{k}:{v}" for k, v in src.items()), "covariates_available": bool(cov_ok),
                     "sizes_evaluated": ";".join(map(str, sizes)),
                     "sizes_skipped": ";".join(str(n) for n in SIZES if n not in sizes),
                     "profiles_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                     "accession_number": accession_number(acc)})
        keep = sub.index.to_numpy()
        np.save(WORK / f"contrast_index_{acc}.npy", keep)
        sub.assign(**{f"clock_{k}": clocks[k].to_numpy() for k in CLOCKS}).to_csv(WORK / f"contrast_{acc}.csv", index=False)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUTA / f"datasets_frozen_{DATE}.csv", index=False)
    inc = fr[fr.included]
    log(f"frozen: {inc.groupby('group').size().to_dict()} included; {int((~fr.included).sum())} excluded")


# ------------------------------------------------------------------ features
def load_contrast(acc):
    pt = pd.read_csv(need(WORK / f"contrast_{acc}.csv"))
    keep = np.load(need(WORK / f"contrast_index_{acc}.npy"))
    return pt, keep


def features(acc, rep, mu=None):
    """(N x p float32 array with NaN only for clocks/covariates, labels y) for the contrast profiles of a dataset."""
    pt, keep = load_contrast(acc)
    y = (pt.role == "case").to_numpy(np.float32)
    if rep in ("embedding", "latent"):
        E = np.load(need(WORK / f"emb_{acc}.npy"))[keep]
        Z = E if rep == "embedding" else E[:, :1536]
    elif rep == "raw":
        X = np.load(need(WORK / f"X_{acc}.npy"), mmap_mode="r")[keep].astype(np.float32)
        Z = np.where(np.isfinite(X), X, mu[None, :]).astype(np.float32)
    elif rep == "clocks":
        cols = [f"clock_{k}" for k in CLOCKS if pt[f"clock_{k}"].notna().mean() >= AVAIL]
        Z = pt[cols].to_numpy(np.float32) if cols else None
    elif rep == "covariates":
        Z = pt[["age", "sex"]].to_numpy(np.float32)
    return Z, y


# ------------------------------------------------------------------ stage embed (GPU)
def stage_embed(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    from methylgpt_model import load_checkpoint
    fp = _module("frozen_probe", "pretraining_frozen_probe_2026-10-12.py")
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    fr = pd.read_csv(need(OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included]
    ag = torch.load(need(AGE_STAGE), map_location=dev, weights_only=False)
    model, core, _ = load_checkpoint(HERE / ag["ckpt"], None, dev)
    model.load_state_dict(ag["model"])
    model.eval()
    del ag
    t0 = time.time()
    for acc in fr.accession:
        f = WORK / f"emb_{acc}.npy"
        if f.exists():
            continue
        X = np.load(need(WORK / f"X_{acc}.npy")).astype(np.float32)
        np.save(f, fp.embed(model, core, X, dev, batch=32))
        log(f"{acc}: {len(X)} embeddings ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ batched logistic elastic net (GPU)
class ENet:
    """Batched logistic elastic net. Problems share a feature matrix within a batch element (draw):
    Z (D, n, p), y (D, n), W (D, Q, n) 0/1 fitting weights, alpha per problem (Q,). The lambda path of problem q is
    lambda_max of its reference problem (ref[q], the whole-draw problem with the same alpha) times RATIOS."""

    def __init__(self, dev, iters):
        self.dev, self.iters = dev, iters

    def fit_path(self, Z, y, W, alpha, ref, store=None):
        """store: indices of the problems whose coefficient paths are returned (default all)."""
        import torch
        D, n, p = Z.shape
        Q = W.shape[1]
        nq = W.sum(-1).clamp_min(1)                                    # (D, Q)
        m = torch.bmm(W, Z) / nq[..., None]                            # (D, Q, p)
        v = torch.bmm(W, Z * Z) / nq[..., None] - m * m
        s = v.clamp_min(0).sqrt()
        inv = torch.where(s > 1e-6, 1.0 / s.clamp_min(1e-6), torch.zeros_like(s))
        ybar = (torch.bmm(W, y[..., None])[..., 0] / nq).clamp(1e-6, 1 - 1e-6)
        a = torch.as_tensor(alpha, device=self.dev, dtype=Z.dtype)[None, :]          # (1, Q)

        def grad_parts(u, c):
            # logits for all n rows of each draw: (D, n, Q)
            eta = torch.bmm(Z, u.transpose(1, 2)) + (c - (m * u).sum(-1))[:, None, :]
            return eta

        def grad(beta, b0):
            u = beta * inv
            eta = grad_parts(u, b0)
            r = (torch.sigmoid(eta) - y[..., None]).transpose(1, 2) * W   # (D, Q, n)
            g = (torch.bmm(r, Z) - m * r.sum(-1, keepdim=True)) * inv / nq[..., None]
            return g, r.sum(-1) / nq, eta

        # lambda_max from the gradient at beta = 0, intercept = logit(ybar)
        b00 = torch.log(ybar / (1 - ybar))
        g0, _, _ = grad(torch.zeros(D, Q, p, device=self.dev, dtype=Z.dtype), b00)
        lmax_own = g0.abs().amax(-1) / a                                    # (D, Q)
        lmax = lmax_own[:, ref]
        ratios = torch.logspace(0, math.log10(LAMBDA_MIN_RATIO), N_LAMBDA, device=self.dev, dtype=Z.dtype)
        # Lipschitz bound: power iteration on X_std^T diag(w) X_std / n
        gen = torch.Generator(device=self.dev).manual_seed(SEED)
        vv = torch.randn(D, Q, p, device=self.dev, dtype=Z.dtype, generator=gen) * (inv > 0)
        for _ in range(30):
            vv = vv / vv.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            u = vv * inv
            t = (torch.bmm(Z, u.transpose(1, 2)) - (m * u).sum(-1)[:, None, :]).transpose(1, 2) * W
            vv = (torch.bmm(t, Z) - m * t.sum(-1, keepdim=True)) * inv / nq[..., None]
        eig = vv.norm(dim=-1)
        Lsm = 0.25 * torch.maximum(eig * 1.05, torch.ones_like(eig))
        inv_nq = inv / nq[..., None]
        beta = torch.zeros(D, Q, p, device=self.dev, dtype=Z.dtype)
        b0 = b00.clone()
        yv = y[..., None]
        path_beta, path_b0, path_eta = [], [], []
        for k in range(N_LAMBDA):
            lam = lmax * ratios[k]                                          # (D, Q)
            L = Lsm + lam * (1 - a)
            step = 1.0 / L
            stp, thr, ridge = step[..., None], (step * lam * a)[..., None], (lam * (1 - a))[..., None]
            if FUSED and Z.is_cuda and D == 1:      # same iterations with fused CUDA kernels (large_gain_enet_cuda_2026-10-24.py)
                nbeta, nb0 = _fused().fista_lambda(Z[0].contiguous(), y[0].contiguous(), W[0].contiguous(), m[0].contiguous(),
                                                   inv[0].contiguous(), inv_nq[0].contiguous(), nq[0].contiguous(),
                                                   beta[0].contiguous(), b0[0].contiguous(), lam[0].contiguous(),
                                                   a[0].contiguous(), Lsm[0].contiguous(), self.iters)
                beta, b0 = nbeta[None], nb0[None]
                _, _, eta = grad(beta, b0)
                sq = slice(None) if store is None else store
                path_beta.append((beta * inv)[:, sq].clone())
                path_b0.append((b0 - (m * beta * inv).sum(-1))[:, sq])
                path_eta.append(eta)
                continue
            yb, yc = beta.clone(), b0.clone()
            tk = torch.ones(D, Q, device=self.dev, dtype=Z.dtype)
            for _ in range(self.iters):
                # gradient at the extrapolated point (same algebra as grad(), fewer passes over (D, Q, p))
                u = yb * inv
                eta = torch.bmm(Z, u.transpose(1, 2))
                eta += (yc - torch.linalg.vecdot(m, u))[:, None, :]
                r = (torch.sigmoid(eta) - yv).transpose(1, 2) * W          # (D, Q, n)
                rs = r.sum(-1)
                G = torch.bmm(r, Z)
                G.addcmul_(m, rs[..., None], value=-1.0)
                G.mul_(inv_nq)
                G.addcmul_(yb, ridge)                                     # + ridge gradient
                G.mul_(-stp).add_(yb)                                     # z = y - step * gradient
                G.sub_(torch.clamp(G, -thr, thr))                         # soft threshold: new beta
                nb = G
                nc = yc - step * (rs / nq)
                d1 = nb - beta
                restart = (torch.linalg.vecdot(yb, d1) - torch.linalg.vecdot(nb, d1)) + (yc - nc) * (nc - b0) > 0
                tk = torch.where(restart, torch.ones_like(tk), tk)
                tk1 = (1 + torch.sqrt(1 + 4 * tk * tk)) / 2
                mom = (tk - 1) / tk1
                yb = torch.addcmul(nb, d1, mom[..., None])
                yc = nc + mom * (nc - b0)
                beta, b0, tk = nb, nc, tk1
            _, _, eta = grad(beta, b0)
            sq = slice(None) if store is None else store
            path_beta.append((beta * inv)[:, sq].clone())        # coefficients on the unstandardized scale
            path_b0.append((b0 - (m * beta * inv).sum(-1))[:, sq])
            path_eta.append(eta)
        return torch.stack(path_beta, 2), torch.stack(path_b0, 2), torch.stack(path_eta, 2), lmax, ratios


def draws_for(acc_num, n, y, n_draws=N_DRAWS):
    cases, ctrls = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for r in range(n_draws):
        rng = np.random.default_rng([SEED, acc_num, n, r])
        tr = np.sort(np.concatenate([rng.choice(cases, n // 2, replace=False), rng.choice(ctrls, n // 2, replace=False)]))
        out.append(tr)
    return out


def fit_select_score(Zall, y, train_sets, dev, fold_seeds, score_sets=None, Zscore=None):
    """For each training set (rows of Zall): inner stratified 5-fold CV over (alpha, lambda), refit, scores.
    Returns list of dicts with scores for score rows (default: all rows not in the training set) and selection info."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    iters = 2000 if Zall.shape[1] <= 100 else 300
    A = len(ALPHAS)
    S = FOLDS + 1
    folds_all = []
    for i, tr in enumerate(train_sets):
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=fold_seeds[i])
        folds = np.zeros(len(tr), int)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(tr)), y[tr])):
            folds[te] = k
        folds_all.append(folds)
    complete = bool(np.isfinite(Zall).all()) and (Zscore is None or bool(np.isfinite(Zscore).all())) and Zall.shape[1] > 100
    if complete and not FORCE_PER_DRAW:
        return _fit_shared(Zall, y, train_sets, folds_all, dev, iters, score_sets, Zscore)
    # columns constant over all rows of the dataset are dropped by the SD rule in every problem; remove them up front
    with np.errstate(all="ignore"):
        keep = np.nanvar(Zall, axis=0) > 0
    if not keep.any():
        keep[0] = True
    Zall = Zall[:, keep]
    if Zscore is not None:
        Zscore = Zscore[:, keep]
    p = Zall.shape[1]
    solver = ENet(dev, iters)
    n = len(train_sets[0])
    per = max(1, min(len(train_sets), int(1.5e8 // max(1, n * p))))
    results = [None] * len(train_sets)
    for b_ in range(0, len(train_sets), per):
        idx = list(range(b_, min(len(train_sets), b_ + per)))
        Zd, yd, Wd = [], [], []
        for i in idx:
            tr = train_sets[i]
            Zt = Zall[tr].copy()
            col_mean = np.nanmean(np.where(np.isfinite(Zt), Zt, np.nan), axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            Zd.append(np.where(np.isfinite(Zt), Zt, col_mean[None, :]))
            yd.append(y[tr])
            Wm = np.ones((S, len(tr)), np.float32)
            for k in range(FOLDS):
                Wm[k, folds_all[i] == k] = 0
            Wd.append(np.repeat(Wm, A, axis=0))          # problem q = s * A + a
        Zt_ = torch.from_numpy(np.stack(Zd).astype(np.float32)).to(dev)
        center = Zt_.mean(1, keepdim=True)
        Zt_ = Zt_ - center
        yt_ = torch.from_numpy(np.stack(yd).astype(np.float32)).to(dev)
        Wt_ = torch.from_numpy(np.stack(Wd)).to(dev)
        alpha = np.tile(ALPHAS, S)
        ref = np.array([FOLDS * A + q % A for q in range(S * A)])
        with torch.no_grad():
            pb, pc, peta, lmax, ratios = solver.fit_path(Zt_, yt_, Wt_, alpha, ref, store=list(range(FOLDS * A, S * A)))
        peta, pb, pc = peta.cpu().numpy(), pb.cpu().numpy(), pc.cpu().numpy()
        ctr = center.cpu().numpy()[:, 0, :]
        for j, i in enumerate(idx):
            tr = train_sets[i]
            a_sel, l_sel, dsel = _select(peta[j], np.arange(len(tr)), folds_all[i], y[tr], 0)
            coef = pb[j, a_sel, l_sel]
            icpt = pc[j, a_sel, l_sel] - float(ctr[j] @ coef)           # undo the draw centring
            if score_sets is None:
                mask = np.ones(len(Zall), bool)
                mask[tr] = False
                Zs = Zall[mask]
            else:
                Zs = (Zscore if Zscore is not None else Zall)[score_sets[i]]
            # missing values in scoring rows take the training-draw mean (clocks, covariates)
            Zt0 = Zall[tr]
            cm = np.nanmean(np.where(np.isfinite(Zt0), Zt0, np.nan), axis=0)
            cm = np.where(np.isfinite(cm), cm, 0.0)
            Zs = np.where(np.isfinite(Zs), Zs, cm[None, :])
            score = Zs.astype(np.float64) @ coef.astype(np.float64) + icpt
            results[i] = {"score": score, "alpha": ALPHAS[a_sel], "lambda_index": int(l_sel),
                          "nonzero": int((coef != 0).sum()), "cv_deviance": dsel}
        del Zt_, Wt_, yt_
    return results


def _select(peta_j, rows, folds, yt, q0):
    """(alpha index, lambda index, deviance) minimizing mean held-out deviance over the inner folds."""
    A = len(ALPHAS)
    dev_ = np.zeros((A, N_LAMBDA))
    for k in range(FOLDS):
        te = folds == k
        for a_ in range(A):
            e = peta_j[rows[te]][:, :, q0 + k * A + a_]                    # (n_te, L)
            ll = np.logaddexp(0, e) - yt[te][:, None] * e
            dev_[a_] += 2 * ll.mean(0) / FOLDS
    flat = int(np.argmin(dev_.reshape(-1)))
    a_sel, l_sel = divmod(flat, N_LAMBDA)
    return a_sel, l_sel, float(dev_[a_sel, l_sel])


def _fit_shared(Zall, y, train_sets, folds_all, dev, iters, score_sets, Zscore):
    """Complete representations: all draws' problems share the dataset matrix on the GPU (one batch element, problems
    differ only in their 0/1 row weights); scoring on the GPU."""
    import torch
    A, S = len(ALPHAS), FOLDS + 1
    Zg = torch.from_numpy(np.ascontiguousarray(Zall, dtype=np.float32)).to(dev)
    keep = Zg.var(0, unbiased=False) > 0          # constant columns: dropped by the SD rule in every problem
    if not bool(keep.any()):
        keep[0] = True
    Zg = Zg[:, keep].contiguous()
    Zsg = None
    if Zscore is not None:
        Zsg = torch.from_numpy(np.ascontiguousarray(Zscore, dtype=np.float32)).to(dev)[:, keep].contiguous()
    N, p = Zg.shape
    solver = ENet(dev, iters)
    D = len(train_sets)
    Wb = np.zeros((1, D * S * A, N), np.float32)
    for jj, tr in enumerate(train_sets):
        for s_ in range(S):
            rows = tr if s_ == FOLDS else tr[folds_all[jj] != s_]
            Wb[0, (jj * S + s_) * A:(jj * S + s_ + 1) * A][:, rows] = 1
    Q = Wb.shape[1]
    center = Zg.mean(0, keepdim=True)
    Zt_ = (Zg - center)[None]
    yt_ = torch.from_numpy(y.astype(np.float32)).to(dev)[None]
    Wt_ = torch.from_numpy(Wb).to(dev)
    alpha = np.tile(ALPHAS, Q // A)
    ref = np.array([(q // (S * A)) * S * A + FOLDS * A + q % A for q in range(Q)])   # whole-draw problem, same alpha
    store = [blk * S * A + FOLDS * A + a_ for blk in range(D) for a_ in range(A)]
    with torch.no_grad():
        pb, pc, peta, lmax, ratios = solver.fit_path(Zt_, yt_, Wt_, alpha, ref, store=store)
    peta = peta[0].cpu().numpy()                                          # (N, L, Q)
    pc = pc[0].cpu().numpy()                                              # (D * A, L)
    sel = [_select(peta, train_sets[jj], folds_all[jj], y[train_sets[jj]], jj * S * A) for jj in range(D)]
    C = torch.stack([pb[0, jj * A + a_, l_] for jj, (a_, l_, _) in enumerate(sel)], 1)        # (p, D)
    icpt = torch.tensor([float(pc[jj * A + a_, l_]) for jj, (a_, l_, _) in enumerate(sel)], device=dev)
    icpt = icpt - (center @ C)[0]                                          # undo the centring
    nz = (C != 0).sum(0).cpu().numpy()
    with torch.no_grad():
        SC = ((Zsg if Zsg is not None else Zg).double() @ C.double() + icpt.double()).cpu().numpy()   # (rows, D)
    results = []
    for jj, tr in enumerate(train_sets):
        if score_sets is None:
            mask = np.ones(N, bool)
            mask[tr] = False
            score = SC[mask, jj]
        else:
            score = SC[score_sets[jj], jj]
        a_, l_, dsel = sel[jj]
        results.append({"score": score, "alpha": ALPHAS[a_], "lambda_index": int(l_), "nonzero": int(nz[jj]), "cv_deviance": dsel})
    del Zt_, Wt_, yt_, pb, Zg
    torch.cuda.empty_cache()
    return results


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


# ------------------------------------------------------------------ stage check_solver
def stage_check_solver(device):
    import torch
    from sklearn.linear_model import LogisticRegression
    dev = torch.device(device)
    fr = pd.read_csv(need(OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included & fr.group.isin(PRIMARY_GROUPS)].sort_values("accession")
    fr = fr[fr.sizes_evaluated.astype(str).str.contains("50")]
    acc = fr.accession.iloc[0]
    mu = np.load(need(CORE))["mu"].astype(np.float32)
    rows = []
    for rep in ("clocks", "embedding", "raw"):
        Z, y = features(acc, rep, mu)
        if Z is None:
            continue
        tr = draws_for(int(fr.accession_number.iloc[0]), 50, y, 1)[0]
        Zt = Z[tr].astype(np.float64)
        Zt = np.where(np.isfinite(Zt), Zt, np.nanmean(Zt, 0))
        yt = y[tr].astype(np.float64)
        p = Zt.shape[1]
        solver = ENet(dev, 2000 if p <= 100 else 300)
        Zc = torch.from_numpy(Zt[None].astype(np.float32)).to(dev)
        cen = Zc.mean(1, keepdim=True)
        W = torch.ones(1, len(ALPHAS), len(tr), device=dev)
        with torch.no_grad():
            pb, pc, _, lmax, ratios = solver.fit_path(Zc - cen, torch.from_numpy(yt[None].astype(np.float32)).to(dev), W,
                                                     np.array(ALPHAS), np.arange(len(ALPHAS)))
        pb, pc, lmax, ratios = pb.cpu().numpy()[0], pc.cpu().numpy()[0], lmax.cpu().numpy()[0], ratios.cpu().numpy()
        m, s = Zt.mean(0), Zt.std(0)
        keep = s > 1e-6
        Xs = (Zt[:, keep] - m[keep]) / s[keep]
        for a_i, a_ in enumerate(ALPHAS):
            for l_i in (0, 4, 9):
                lam = float(lmax[a_i] * ratios[l_i])
                b_std = pb[a_i, l_i][keep] * s[keep]
                c_std = float(pc[a_i, l_i] - cen.cpu().numpy()[0, 0] @ pb[a_i, l_i] + m @ pb[a_i, l_i])

                def obj(b, c):
                    e = Xs @ b + c
                    return float(np.mean(np.logaddexp(0, e) - yt * e) + lam * (a_ * np.abs(b).sum() + (1 - a_) / 2 * (b @ b)))
                ours = obj(b_std, c_std)
                try:
                    sk = LogisticRegression(l1_ratio=a_, C=1.0 / (len(yt) * lam), solver="saga", tol=1e-10,
                                            max_iter=20000 if p <= 3000 else 3000)
                    sk.fit(Xs, yt)
                    theirs = obj(sk.coef_[0], float(sk.intercept_[0]))
                except Exception as e:  # report, do not hide
                    theirs = float("nan")
                    log(f"sklearn failed: {e}")
                rows.append({"accession": acc, "representation": rep, "features": int(keep.sum()), "alpha": a_,
                             "lambda_index": l_i, "objective_ours": ours, "objective_sklearn_saga": theirs,
                             "relative_gap": (ours - theirs) / abs(theirs) if np.isfinite(theirs) else np.nan,
                             "nonzero_ours": int((b_std != 0).sum())})
                log(f"check {rep} a={a_} l={l_i}: ours {ours:.6f} sklearn {theirs:.6f}")
    write_json(OUTA / f"solver_check_{DATE}.json", rows)


# ------------------------------------------------------------------ stage fit
def stage_fit(device, groups=None, shard=None):
    import torch
    dev = torch.device(device)
    fr = pd.read_csv(need(OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included]
    if groups:
        fr = fr[fr.group.isin(groups)]
    mu = np.load(need(CORE))["mu"].astype(np.float32)
    outdir = WORK / "fit_A"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    order = fr.assign(pri=~fr.group.isin(PRIMARY_GROUPS)).sort_values(["pri", "accession"]).reset_index(drop=True)
    if shard:                                   # execution only: k/K runs every K-th dataset
        k, K = map(int, shard.split("/"))
        order = order[order.index % K == k]
    for _, d in order.iterrows():
        acc = d.accession
        f = outdir / f"{acc}.csv"
        if f.exists():
            continue
        sizes = [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
        rows = []
        for rep in REPS:
            if rep == "covariates" and not d.covariates_available:
                continue
            Z, y = features(acc, rep, mu)
            if Z is None:
                continue
            for n in sizes:
                trs = draws_for(int(d.accession_number), n, y)
                res = fit_select_score(Z, y, trs, dev, fold_seeds=list(range(N_DRAWS)))
                for r_i, (tr, res_) in enumerate(zip(trs, res)):
                    mask = np.ones(len(y), bool)
                    mask[tr] = False
                    ys = y[mask]
                    rows.append({"accession": acc, "representation": rep, "n_train": n, "draw": r_i,
                                 "auc": auc(res_["score"], ys), "alpha": res_["alpha"], "lambda_index": res_["lambda_index"],
                                 "nonzero": res_["nonzero"], "cv_deviance": res_["cv_deviance"],
                                 "n_score_case": int(ys.sum()), "n_score_ctrl": int((1 - ys).sum()), "features": int(Z.shape[1])})
        pd.DataFrame(rows).to_csv(f, index=False)
        log(f"{acc} ({d.group}): fitted ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ stage analyse
def boot_means(per, reps, draws=BOOT, seed=SEED):
    """per: DataFrame index dataset, columns representations (AUC - 0.5). Bootstrap mean per representation."""
    rng = np.random.default_rng(seed)
    V = per[reps].to_numpy(float)
    k = len(V)
    idx = rng.integers(0, k, size=(draws, k))
    return {r: V[idx, j].mean(1) for j, r in enumerate(reps)}


def compare(per, reps_all, mg, comps, label):
    """Mean (AUC - 0.5) per representation over datasets of `per` and paired bootstrap intervals."""
    per = per.dropna(subset=[mg] + comps)
    if len(per) == 0:
        return None
    reps = [r for r in reps_all if r in per and per[r].notna().all()]
    M = {r: float(per[r].mean()) for r in reps}
    B = boot_means(per, reps)
    best = max(comps, key=lambda c: M[c])
    diff = B[mg] - B[best]
    out = {"set": label, "datasets": int(len(per)), "methylgpt": mg, "M": M,
           "M_ci95": {r: [float(np.percentile(B[r], 2.5)), float(np.percentile(B[r], 97.5))] for r in reps},
           "best_comparator": best, "best_comparator_M": M[best], "methylgpt_M": M[mg],
           "difference": M[mg] - M[best], "difference_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
           "relative_gain": (M[mg] - M[best]) / M[best] if M[best] > 0 else float("nan"),
           "relative_gain_ci95": [float(np.percentile((B[mg] - B[best]) / B[best], 2.5)),
                                  float(np.percentile((B[mg] - B[best]) / B[best], 97.5))] if M[best] > 0 else None,
           "vs_each": {}}
    for c in [r for r in reps if r != mg]:
        dd = B[mg] - B[c]
        out["vs_each"][c] = {"difference": M[mg] - M[c], "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                             "relative_gain": (M[mg] - M[c]) / M[c] if M[c] > 0 else float("nan"),
                             "datasets_methylgpt_higher": int((per[mg] > per[c]).sum())}
    out["pass"] = bool(np.isfinite(out["relative_gain"]) and out["relative_gain"] >= 0.10 and out["difference_ci95"][0] > 0)
    return out


def stage_analyse():
    fr = pd.read_csv(need(OUTA / f"datasets_frozen_{DATE}.csv"))
    fr = fr[fr.included].set_index("accession")
    parts = [pd.read_csv(f) for f in sorted((WORK / "fit_A").glob("*.csv"))]
    per_draw = pd.concat(parts, ignore_index=True)
    per_draw.to_csv(OUTA / f"per_draw_auc_{DATE}.csv.gz", index=False)
    g = per_draw.groupby(["accession", "representation", "n_train"])
    pdx = g.auc.agg(["mean", "std", "median", "count"]).rename(columns={"mean": "auc_mean", "std": "auc_sd", "median": "auc_median",
                                                                       "count": "draws"})
    pdx["null_model_draws"] = g.nonzero.apply(lambda s: int((s == 0).sum()))
    pdx["mean_nonzero"] = g.nonzero.mean()
    pdx = pdx.reset_index().join(fr[["group", "disease", "disease_group", "tissue", "array", "n_case", "n_ctrl", "clocks_used"]],
                                 on="accession")
    pdx.to_csv(OUTA / f"per_dataset_auc_{DATE}.csv", index=False)
    wide = {n: pdx[pdx.n_train == n].pivot(index="accession", columns="representation", values="auc_mean") - 0.5 for n in SIZES}
    res = {"primary": None, "secondary": {}, "datasets": {}, "missing_clocks": {}}
    sets = {"primary": fr.index[fr.group.isin(PRIMARY_GROUPS)], "train": fr.index[fr.group == "train"]}
    for g_ in PRIMARY_GROUPS:
        sets[f"primary|{g_}"] = fr.index[fr.group == g_]
    sets["primary|27K"] = fr.index[fr.group.isin(PRIMARY_GROUPS) & (fr.array == "27K")]
    sets["primary|450K/EPIC/EPIC v2"] = fr.index[fr.group.isin(PRIMARY_GROUPS) & (fr.array != "27K")]
    for dg in sorted(fr.disease_group.unique()):
        sets[f"primary|{dg}"] = fr.index[fr.group.isin(PRIMARY_GROUPS) & (fr.disease_group == dg)]
        sets[f"train|{dg}"] = fr.index[(fr.group == "train") & (fr.disease_group == dg)]
    rows = []
    for name, ids in sets.items():
        for n in SIZES:
            w = wide[n].reindex([i for i in ids if i in wide[n].index])
            if not len(w):
                continue
            res["missing_clocks"][f"{name}|{n}"] = sorted(w.index[w.get("clocks", pd.Series(np.nan, index=w.index)).isna()])
            for mg, comps, tag in (("embedding", COMPARATORS, "raw and clocks"), ("latent", COMPARATORS, "raw and clocks"),
                                   ("embedding", ["raw"], "raw only, all datasets"), ("latent", ["raw"], "raw only, all datasets")):
                r = compare(w, ["embedding", "latent", "raw", "clocks"], mg, comps, f"{name}|n={n}|{mg}|{tag}")
                if r is None:
                    continue
                key = f"{name}|n={n}|{mg}|{tag}"
                decision = name == "primary" and n == PRIMARY_SIZE and mg == MGPT and tag == "raw and clocks"
                if decision:
                    res["primary"] = r
                else:
                    res["secondary"][key] = r
                rows.append({"set": name, "n_train": n, "methylgpt_representation": mg, "comparators": tag, "datasets": r["datasets"],
                             "methylgpt_M": r["methylgpt_M"], "M_ci95_low": r["M_ci95"][mg][0], "M_ci95_high": r["M_ci95"][mg][1],
                             "M_raw": r["M"].get("raw"), "M_clocks": r["M"].get("clocks"),
                             "M_latent": r["M"].get("latent"), "M_embedding": r["M"].get("embedding"),
                             "best_comparator": r["best_comparator"], "best_comparator_M": r["best_comparator_M"],
                             "difference": r["difference"], "difference_ci95_low": r["difference_ci95"][0],
                             "difference_ci95_high": r["difference_ci95"][1], "relative_gain": r["relative_gain"],
                             "meets_rule": r["pass"], "decision_set": decision})
            # covariates (descriptive): datasets with covariates
            if "covariates" in w:
                wc = w.dropna(subset=["covariates"])
                if len(wc):
                    rows.append({"set": name, "n_train": n, "methylgpt_representation": "covariates (age and sex, descriptive)",
                                 "datasets": int(len(wc)), "methylgpt_M": float(wc.covariates.mean()),
                                 "M_raw": float(wc.raw.mean()), "M_clocks": float(wc.clocks.mean()) if wc.clocks.notna().all() else np.nan,
                                 "M_embedding": float(wc.embedding.mean()), "M_latent": float(wc.latent.mean()),
                                 "decision_set": False})
    pd.DataFrame(rows).to_csv(OUTA / f"summary_{DATE}.csv", index=False)
    p = res["primary"]
    rule = pd.DataFrame([{
        "analysis": "A: disease classification from 50 labelled samples (25 cases, 25 controls), primary datasets",
        "primary_metric": "mean over datasets of (AUC - 0.5), n = 50",
        "datasets": p["datasets"], "methylgpt_representation": "embedding", "methylgpt_value": p["methylgpt_M"],
        "best_comparator": p["best_comparator"], "best_comparator_value": p["best_comparator_M"],
        "relative_gain": p["relative_gain"], "difference": p["difference"],
        "difference_ci95_low": p["difference_ci95"][0], "difference_ci95_high": p["difference_ci95"][1],
        "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
        "outcome": "PASS" if p["pass"] else "FAIL"}])
    rule.to_csv(OUTA / f"rule_outcome_{DATE}.csv", index=False)
    write_json(OUTA / f"results_{DATE}.json", res)
    log(f"PRIMARY: {rule.iloc[0].to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "prepare", "clocks", "freeze", "embed", "check_solver", "fit", "analyse"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--validate", nargs="*", default=[])
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--shard")
    a = ap.parse_args()
    if a.stage == "select":
        stage_select()
    elif a.stage == "prepare":
        stage_prepare(a.only)
    elif a.stage == "clocks":
        stage_clocks(a.validate)
    elif a.stage == "freeze":
        stage_freeze()
    elif a.stage == "embed":
        stage_embed(a.device)
    elif a.stage == "check_solver":
        stage_check_solver(a.device)
    elif a.stage == "fit":
        stage_fit(a.device, a.groups, a.shard)
    else:
        stage_analyse()


if __name__ == "__main__":
    main()
