#!/usr/bin/env python3
"""Large-gain analysis F: a disease-trained MethylGPT representation for disease classification from few labels.

Written on 24 Oct 2026, before any model is fitted, for the large-gain analysis plan of 23 Oct 2026 (Decision rule,
analysis A, Outputs); analysis F of the MethylGPT revision. Question: when MethylGPT is further trained on the
case-control labels of corpus training-split datasets, does logistic elastic net on its output classify disease in a
new public cohort from a few labelled samples much better than the baselines a user would otherwise use (elastic net
on raw CpGs with mean filling, published clocks)? Analysis A's datasets, few-label protocol and comparators are reused
unchanged (large_gain_A_disease_few_labels_2026-10-24.py, imported as a module; its comparator AUCs are read from A's
committed per-draw table). Every result is reported, whichever representation it favours.

DATA
  Labels, contrasts and arrays: exactly analysis A's rules (A.labels_table, A.choose_contrast, A.contrast_mask,
  A.platform_array, A.infer_array), on the corpus (artifacts/corpus_2026-09-30).
  Held-out set (final evaluation only, touched once per locked evaluation): A's primary set (groups test, val and
  external) exactly as A froze it: included rows of A/datasets_frozen_2026-10-24.csv once committed, their contrast
  profiles in A's order (work_AB/contrast_<acc>.csv, contrast_index_<acc>.npy) and beta values (work_AB/X_<acc>.npy).
  Train-split datasets: corpus groups outside split_manifest test_groups and dev_groups. Only profiles with
  split == "train" are used for training (AltumAge validation and test profiles, split "excluded_downstream", and
  QC exclusions are not).
  Development-validation (dev-val) datasets: about 20% of A's train-group datasets, stratified by A's disease group.
  Within each disease group g (sorted by name) with n_g datasets in A's candidate table (train group, not excluded),
  k_g = max(1, round(0.2 n_g)) datasets are drawn without replacement from those evaluable at n = 50 (>= 35 cases and
  >= 35 controls, A's rule n/2 + 10), with one generator numpy default_rng(20261024) used over the groups in order on
  the accessions sorted alphabetically. Contrast and profiles as A defines them for corpus datasets. Their whole corpus
  group is excluded from training during development. When A commits its frozen table, the dev-val list is checked
  against it (same contrast, same sorted-profile SHA-256); a dev-val dataset that A excluded is dropped, not replaced.
  Training strata: every (group, tissue, disease) stratum of a train-split group with >= 10 cases of disease D in tissue
  t and >= 10 healthy controls of tissue t (A's role rules; split == "train" profiles only). Two pools (design option):
    A30  the stratum of A's contrast for each of A's train-group datasets (one stratum per dataset);
    S15  every stratum with >= 15 cases and >= 15 controls (includes A30's strata; one head per stratum).
  Development trains on the strata of groups other than the dev-val groups; the final model trains on the strata of
  all train-split groups (dev-val groups included). Strata left with < 10 cases or < 10 controls after duplicate
  removal are dropped. Stage `split` writes F/datasets_F_2026-10-24.csv (dev-val list and training strata), committed
  before any fitting.

DUPLICATE CHECK (stage dedup, before any fitting; betas and sample ids only, no held-out labels)
  Held-out side: every profile A prepares for a held-out dataset (corpus: the group's profiles with metadata labels;
  external: rows of work_AB/X_<acc>.npy), for the dev-val datasets and A's primary datasets. Training side: every
  profile of every training stratum (all train-split groups). For each (training group, held-out dataset) pair:
  Pearson correlation of every training profile with every held-out profile over the CpGs observed in >= 90% of the
  profiles of both (other missing values take that dataset's CpG mean), on raw beta values ("raw") and on beta values
  minus the corpus training mean mu ("centred", which removes the CpG-level profile shared by all samples).
  A training profile is a duplicate of a held-out dataset's profile if (a) they share a GSM id, or (b) its highest
  centred r with that dataset's profiles is >= 0.98, or (c) that highest centred r is >= 0.95 and exceeds the
  profile's second-highest centred r with the same dataset by >= 0.01 (a match that stands out from every other
  profile of the dataset). Calibration, done on betas only before any fitting: raw r > 0.99 (the example criterion
  of the brief) does not separate duplicates on 450K and EPIC data: 216 of 286 GSE80970 profiles (val, Mount Sinai
  brain) have raw r > 0.99 with a GSE59685 profile (train, London brain, other individuals), and 105 of 675 GSE80417
  profiles with GSE84727 (another schizophrenia cohort); within one dataset the nearest other profile has centred r
  0.92-0.97 (median of maxima). GSE97362 (test) re-uses about 35 GSE74432 (train) control profiles under new GSM ids:
  these matches have centred r 0.958-0.9998 and stand 0.03-0.06 above the second-best match, while non-matches
  near 0.95 stand < 0.006 above it. Raw r > 0.99 counts are reported for every pair. Profiles with a highest centred
  r >= 0.94 that are not flagged are listed in F/duplicates_near_not_removed_2026-10-24.csv. (Amendment 24 Oct:
  development keeps the removal set it started with, duplicates of dev-val and corpus primary profiles, when the
  external datasets are added to the check; the final training removes duplicates of all primary profiles.)
  Development removes duplicates of dev-val and primary
  profiles; the final training removes duplicates of primary profiles (dev-val groups become training data).
  Outputs: F/duplicates_pairs_2026-10-24.csv (per pair: profiles, shared CpGs, max correlations, counts flagged by
  each criterion), F/duplicates_removed_2026-10-24.csv (removed training profile ids, the held-out dataset and
  the reason). Pairs with external held-out datasets are checked when A's external beta matrices exist, and always
  before the final training (stage final refuses to run otherwise).

REPRESENTATION DESIGNS (candidate set fixed here; all start from released weights)
  Encoder input: each profile's observed vocabulary CpGs; unobserved CpGs stay unobserved (as A's embedding).
  D1  Frozen age-stage encoder (bundle weights_age_stage.pt = runs/aging_heads_clean_2026-10-01/best.pt, whose
      2,048-dimensional sample embedding [latent scores / prior SD, transformer summary] is A's "embedding") plus a
      trainable adapter on that embedding. Adapter input is standardized with the mean and SD of the training
      profiles' embeddings (fixed). Adapter "linear": dropout(p) then Linear(2048, K); "mlp": Linear(2048, H), GELU,
      dropout(p), Linear(H, K). Output representation: the K adapter outputs.
  D2  D1 with the last encoder self-attention block (enc_self[7]) and the summary LayerNorm (sum_norm) trainable
      (learning rate lr_enc); the latent-score part of the embedding is fixed by construction. Output: adapter
      outputs ("adapter") or the fine-tuned 2,048-dimensional embedding ("encoder").
  D3  Full fine-tuning from the released multitask weights (bundle weights.pt = runs/multitask_bio_2026-10-01/final.pt,
      encoder and BioHeads), encoder parameters as in train_multitask_2026-10-01.py (enc_params), with the adapter
      and disease heads, plus the multitask losses as auxiliary losses so that age, sex, tissue and clocks are kept:
      loss_age + 0.2 loss_sex + 0.2 loss_tissue + 1.0 loss_clocks exactly as in train_multitask, weight w_aux, on
      batches of 32 labelled training-split profiles (a fixed seeded subset of 20,000 of train_multitask's labelled
      profiles, duplicates removed) with its masking augmentation (0-50%). Output: "adapter" or "encoder".
  Disease objective (the per-stratum heads use balanced sampling across datasets: groups drawn uniformly, then one
  of the group's strata uniformly, then per_class cases and per_class controls, with replacement only when a class
  is smaller):
    heads     one binary case-control head per training stratum (logit = w_s . r + b_s), mean binary cross-entropy.
    episodic  few-label episodes that imitate the downstream use: a stratum, a balanced support set of
              k_s = min(25, m - 5) per class and a query set of k_q = min(25, m - k_s) per class (m = smaller class
              size); support features standardized with the support mean and SD (as the downstream classifier does);
              ridge classifier on +-1 labels in closed form (dual form, penalty exp(l) k_s, l trainable, initial 0);
              query loss binary cross-entropy of tau * score (tau trainable, initial 1).
    both      heads + episodic.
  27K view augmentation (option aug27k): with probability 0.25 a training profile (heads) or a whole episode
  (episodic) is replaced by its 27K-array view (vocabulary CpGs outside the 27K array set unobserved, mask
  platform_vocab_masks hm27), so that the representation also serves 27K cohorts.
  Optimizer AdamW (adapter and heads lr, weight decay wd; encoder lr_enc, weight decay 0.01; D3 BioHeads 10 x lr_enc,
  the multitask ratio), 300-step warm-up, cosine decay to 5% (as train_multitask), gradient clipping at 1.0, seed
  20261024 unless stated.
  Default configuration (design D1): pool S15, objective heads, adapter mlp, H 1024, K 64, dropout 0.1, aug27k off, lr 1e-3,
  wd 1e-2, 32 strata per step x 8 per class (D3: 16 x 4), 8 episodes per step (D3: 2), 4,000 steps with checkpoints
  evaluated at steps 500, 1,000, 2,000 and 4,000 (D3: 1,000, 2,000, 4,000; lr_enc 1e-5; w_aux 1).
  Development search (sequential; every configuration and checkpoint logged; selection always by the rule below):
    S1  D1, pool {A30, S15} x objective {heads, episodic, both} x adapter {linear, mlp} (12 runs).
    S2  D1 around the best S1 run: K in {16, 32, 128, 256}, then aug27k on, dropout in {0, 0.3}, wd in {1e-3, 1e-1},
        lr in {3e-4, 3e-3} (one factor at a time from the best so far; at most 13 runs).
    S3  D2 from the best D1 configuration: lr_enc in {1e-5, 5e-5} x output {adapter, encoder} (at most 4 runs).
    S4  D3 from the best configuration: lr_enc in {1e-5, 3e-5} x output {adapter, encoder} (at most 4 runs).
    S5  the selected configuration with seeds 20261025 and 20261026 (variability, reported; not used to select).
  Selection rule: the logged (configuration, checkpoint step) with the highest dev-val metric; ties (equal to 1e-4)
  go to the simpler design (D1 < D2 < D3), then fewer steps, then smaller K. Development never reads held-out labels.

DEV-VAL METRIC AND THE DOWNSTREAM CLASSIFIER (A's protocol, A's code)
  For a dataset, representation R of its contrast profiles; A.draws_for(A.accession_number(acc), n, y) (50 balanced
  draws of n/2 cases and n/2 controls, rng default_rng([20261024, accession number, n, draw])); A.fit_select_score
  with fold_seeds = draw index (logistic elastic net, alpha in {0.1, 0.5, 0.9} x 10 lambdas, 5-fold CV deviance,
  refit); AUC on the remaining contrast profiles; dataset value = mean AUC over the 50 draws. Datasets are stacked in
  one call of A.fit_select_score (score_sets given), which fits each draw independently, so results equal per-dataset
  calls (checked once, F/stacking_check_2026-10-24.json). Dev-val metric: mean over dev-val datasets of (mean AUC - 0.5)
  at n = 50. Development references on the same draws (not used to select): A's embedding, latent, raw and clocks
  (F/dev_reference_2026-10-24.csv).

LOCK AND RETRAINING
  Stage lock applies the selection rule to F/dev_log_2026-10-24.csv and writes F/locked_design_2026-10-24.json
  (configuration, step, seed 20261024, checkpoint path, A's protocol fingerprint: SHA-256 of the source of A's
  protocol functions and its constants, A's frozen-table SHA-256). The selected configuration is then retrained once
  on the strata of all train-split groups (final pool, primary-set duplicates removed), seed 20261024, with the
  development run's schedule (warm-up and cosine over its total steps) stopped at the selected step (stage final;
  checkpoint F/checkpoints/final_<k>.pt). The script, lock file and development log are
  committed before stage heldout runs. Stage heldout refuses to run if A's protocol fingerprint differs from the lock.

LOCKED EVALUATION (stage heldout, once per lock; numbered k = 1, 2)
  Representation of the final model for each primary dataset (contrast profiles in A's order); A's protocol at every
  size A evaluates for the dataset (n = 20, 50, 100; sizes_evaluated). Comparators and the released embedding: A's
  per-draw AUCs (A/per_draw_auc_2026-10-24.csv.gz, committed), paired by (accession, n, draw); the scoring-set sizes
  are checked to be identical.
  PRIMARY: M(R) = mean over A's primary datasets evaluable at n = 50 of (AUC - 0.5), R = F's representation. Best
  comparator C* = raw or clocks, whichever has the larger M on the same datasets (datasets lacking any of these are
  left out, listed, as in A). Relative gain = (M(F) - M(C*)) / M(C*). 95% interval of M(F) - M(C*): paired bootstrap
  over datasets, 2,000 resamples, seed 20261024, A.boot_means/A.compare (one resample of datasets per draw shared by
  all representations, C* fixed on the full set). PASS if and only if the relative gain is >= 10% and the lower end of
  the interval is > 0; otherwise FAIL.
  SECONDARY (all reported): the same at n = 20 and n = 100; F minus the released embedding (A's "embedding", the gain
  from disease training) and minus the latent scores, each with its interval; held-out datasets whose disease label
  equals the disease label of a final training stratum ("disease seen") against the others ("disease unseen");
  external against corpus (test and val) held-out datasets; 27K against the other arrays; per-dataset values.
  If a locked evaluation fails, the held-out set is not used to iterate. Development may resume on train-split data
  only; a new design is locked (locked_design_2026-10-24_2.json) and evaluated once more; every locked evaluation is
  reported in order.

AMENDMENT 24 Oct 2026 (after commit 87ec923, before any D2 or D3 run; computation only): the PPCA posterior means
  (latent scores) do not depend on any trainable weight, so D2 and D3 take them from the cached age-stage embeddings
  (first 1,536 dimensions times the prior SD) instead of recomputing them for every batch, and D2 caches the
  dev-val inputs of the last block once. Results are unchanged up to floating-point rounding. D3's auxiliary batches
  use one fixed random mask per auxiliary profile (drawn once, seed 20261024 + 13, with train_multitask's masking
  distribution: a fraction u * 0.5 of observed CpGs, u uniform) with its posterior computed once, instead of a new
  mask at every step (the per-step posterior over 49,156 CpGs made D3 about 25 times slower on the shared GPU).
AMENDMENT 24 Oct 2026, after S1 (12 runs) and before any S2 run: on dev-val every S1 adapter scored below the
  released embedding (best 0.2393, S15 / episodic / linear / K 64 at step 500, against 0.2561 for A's embedding on
  the same draws), losing most on datasets whose signal lies outside the directions of the training strata (brain
  GSE41826, GSE72778, GSE74486; 27K blood GSE41037). Added to the search: adapter "linear_id", a full-rank
  Linear(2048, 2048) initialized to the identity (bias 0, input standardized as before, dropout p on the input), so
  that training starts exactly at the released embedding (the downstream classifier standardizes each feature) and
  disease training moves it from there; run with the S1-best pool and objective at lr 1e-4 and 1e-3 (S2b, at most 2
  runs, then the S2 factors that were best for the linear adapter, at most 4 runs). From S2 on, step 250 is added to
  the evaluated checkpoints because every S1 run peaked at its first checkpoint (500). The selection rule is
  unchanged (all logged checkpoints, highest dev-val metric).
AMENDMENT 24 Oct 2026, after S3 started and before any S4 (D3) run: every D1 and D2 run peaked at or before step
  500 (S1, S2, S2b, first S3 run), and a D3 step costs about 50 times a D1 step on the shared GPU, so S4 runs use a
  2,000-step schedule evaluated at steps 250, 500, 1,000 and 2,000 (instead of 4,000 steps evaluated at 1,000, 2,000
  and 4,000). Checkpoints, rebuilt encoders, rows and auxiliary data are kept in memory between the runs of one
  process (computation only; the rebuilt encoder equals load_checkpoint exactly, checked on 64 profiles).
AMENDMENT 24 Oct 2026, after locked evaluation 1 was first run (design, checkpoint and draws unchanged): the
  pre-planned pairing check (A's released embedding passed through this script's evaluation) did not reproduce A's
  per-draw AUCs exactly (mean absolute difference 0.0046, largest 0.39, 1.8% of n = 50 draws with a different
  selected penalty; mostly draws at lambda_max where one side returns the null model). Cause: this script enabled
  TF32 matrix multiplication for the whole held-out stage, so A's elastic net ran at reduced precision, while A fits
  in full FP32 (A enables TF32 only in its embedding stage). The held-out stage now computes the representation with
  TF32 (as A's embedding) and runs A.fit_select_score with TF32 off (as A's fit). The as-run TF32 outputs are kept
  with the suffix _tf32_asrun and reported; the FP32 run, whose pairing check must match A exactly, is the paired
  locked evaluation 1. Development runs all used TF32 in the classifier (consistent between designs); the dev-val
  references (stage devref) ran in FP32.

Stages: split, xcache, dedup, embed, stacking_check, devref, train, lock, final, heldout, analyse.
Outputs: artifacts/large_gain_2026-10-24/F/ (README.md, datasets_F, duplicates, dev_log, dev_reference, locked design,
per-draw and per-dataset AUC, summary, rule outcome). Working files in artifacts/large_gain_2026-10-24/work_F/ and
checkpoints in F/checkpoints/, not committed. GPU: cuda:1, shared; batch sizes modest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
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
OUTF = OUT / "F"
WORKAB = OUT / "work_AB"
WORKF = OUT / "work_F"
CKPT_DIR = OUTF / "checkpoints"
A_SCRIPT = HERE / "large_gain_A_disease_few_labels_2026-10-24.py"
AGE_STAGE = ART / "runs/aging_heads_clean_2026-10-01/best.pt"
MULTITASK = ART / "runs/multitask_bio_2026-10-01/final.pt"
CORE = ART / "ppca_core_meanfilled_2026-09-30.npz"
MASKS = ART / "platform_vocab_masks_2026-10-01.npz"
DATE = "2026-10-24"
SEED = 20261024
DEV_FRACTION = 0.2
DEV_N = 50
MIN_STRATUM, S15 = 10, 15
DUP_RAW, DUP_CENTRED_ABS, DUP_CENTRED_MARGIN_MIN, DUP_MARGIN, DUP_SHARED_OBS = 0.99, 0.98, 0.95, 0.01, 0.90
AUX_SUBSET = 20000
DEV_LOG = OUTF / f"dev_log_{DATE}.csv"

DEFAULT = dict(design="D1", pool="S15", objective="heads", adapter="mlp", hidden=1024, K=64, dropout=0.1, aug27k=False, lr=1e-3,
               wd=1e-2, lr_enc=1e-5, w_aux=1.0, output="adapter", strata_per_step=32, per_class=8,
               episodes_per_step=8, steps=4000, eval_steps=[500, 1000, 2000, 4000], seed=SEED)
D3_OVERRIDES = dict(strata_per_step=16, per_class=4, episodes_per_step=2, eval_steps=[1000, 2000, 4000])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def need(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    return path


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def git_committed(path):
    """True when the file is tracked and identical to HEAD."""
    r = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(path)], capture_output=True)
    if r.returncode:
        return False
    return subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0


def load_a():
    spec = importlib.util.spec_from_file_location("large_gain_A", A_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["large_gain_A"] = m          # needed by inspect.getsource for A's classes (protocol fingerprint)
    spec.loader.exec_module(m)
    return m


A = load_a()
PROTOCOL_OBJECTS = ["draws_for", "fit_select_score", "ENet", "auc", "boot_means", "compare", "accession_number",
                    "features", "load_contrast"]
PROTOCOL_CONSTANTS = ["SEED", "SIZES", "PRIMARY_SIZE", "N_DRAWS", "ALPHAS", "N_LAMBDA", "LAMBDA_MIN_RATIO", "FOLDS",
                      "MIN_SCORE", "BOOT", "PRIMARY_GROUPS", "COMPARATORS", "MGPT"]


def protocol_fingerprint():
    fp = {k: hashlib.sha256(inspect.getsource(getattr(A, k)).encode()).hexdigest() for k in PROTOCOL_OBJECTS}
    fp["constants"] = {k: getattr(A, k) for k in PROTOCOL_CONSTANTS}
    return fp


# ------------------------------------------------------------------ data tables
_CACHE = {}


def labels():
    if "lab" not in _CACHE:
        f = WORKF / "labels.pkl"
        if f.exists() and f.stat().st_mtime > A.META.stat().st_mtime:
            _CACHE["lab"] = pd.read_pickle(f)
        else:
            lab, _ = A.labels_table()
            WORKF.mkdir(parents=True, exist_ok=True)
            lab.to_pickle(f)
            _CACHE["lab"] = lab
    return _CACHE["lab"]


def profiles():
    if "prof" not in _CACHE:
        p = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"))
        _CACHE["prof"] = p[p.split != "excluded_qc"]
    return _CACHE["prof"]


def manifest():
    return json.loads(need(CORPUS_DIR / "split_manifest.json").read_text())


def a_candidates():
    c = pd.read_csv(need(OUTA / f"candidates_{DATE}.csv"))
    return c[c.excluded.isna()].copy()


def a_frozen(required=True):
    """A's committed frozen table (included rows) or None."""
    f = OUTA / f"datasets_frozen_{DATE}.csv"
    if f.exists() and git_committed(f):
        fr = pd.read_csv(f)
        return fr[fr.included].copy()
    if required:
        sys.exit(f"Cannot proceed - A's frozen dataset table is not committed: {f}")
    return None


def strata_table():
    """Training strata (>= MIN_STRATUM cases and controls) of all train-split groups, split == 'train' profiles."""
    lab, prof, man = labels(), profiles(), manifest()
    held = set(man["test_groups"]) | set(man["dev_groups"])
    p = prof[(prof.split == "train") & ~prof.group.isin(held)]
    j = p[["id", "group"]].join(lab, on="id", how="inner")
    j = j[j.role.isin(["case", "control"])]
    rows = []
    for g, b in j.groupby("group"):
        for t, bt in b.groupby("tissue"):
            nc = int((bt.role == "control").sum())
            if nc < MIN_STRATUM:
                continue
            for d, k in bt[bt.role == "case"].disease.value_counts().items():
                if k >= MIN_STRATUM:
                    rows.append(dict(stratum=f"{g}|{t}|{d}", group=g, tissue=t, disease=d, n_case=int(k), n_ctrl=nc))
    return pd.DataFrame(rows)


def stratum_rows(st_row):
    """Corpus rows (cases, controls) of a training stratum (split == 'train' profiles)."""
    lab, prof = labels(), profiles()
    p = prof[(prof.group == st_row.group) & (prof.split == "train")]
    L = lab.reindex(p.id.astype(str))
    tis = (L.tissue == st_row.tissue).to_numpy()
    case = tis & (L.role == "case").to_numpy() & (L.disease == st_row.disease).to_numpy()
    ctrl = tis & (L.role == "control").to_numpy()
    idx = p.index.to_numpy()
    return np.sort(idx[case]), np.sort(idx[ctrl])


def corpus_contrast(acc, platform, X=None, rows=None):
    """A's contrast for a corpus dataset (as A's prepare + freeze): DataFrame in sorted corpus-row order."""
    lab, prof = labels(), profiles()
    b = prof[(prof.group == acc) & prof.id.isin(lab.index)]
    rows = np.sort(b.index.to_numpy()) if rows is None else rows
    ids = prof.loc[rows, "id"].astype(str).to_numpy()
    L = lab.reindex(ids)
    arr = A.platform_array(platform)
    if arr:
        arrays = np.full(len(ids), arr)
    else:
        z = np.load(need(MASKS))
        arrays = A.infer_array(X, {k: z[k] for k in ("hm27", "hm450", "epic_v1", "epic_v2")})
    pt = pd.DataFrame({"id": ids, "corpus_row": rows, "role": L.role.to_numpy(), "disease": L.disease.to_numpy(),
                       "tissue": L.tissue.to_numpy(), "array": arrays})
    pt = pt[pt.role.isin(["case", "control"])]
    cc, _ = A.choose_contrast(pt)
    if cc is None:
        return None, None
    sub = pt[A.contrast_mask(pt, cc)]
    arr = sub.array.value_counts().index[0]
    sub = sub[sub.array == arr]
    cc2, _ = A.choose_contrast(sub)
    if cc2 is None or cc2["disease"] != cc["disease"] or cc2["tissue"] != cc["tissue"]:
        return None, None
    sub = sub[A.contrast_mask(sub, cc2)].reset_index(drop=True)
    return sub, dict(cc2, array=arr)


# ------------------------------------------------------------------ stage split
def stage_split():
    OUTF.mkdir(parents=True, exist_ok=True)
    cand = a_candidates()
    tr = cand[cand.group == "train"].sort_values("accession")
    ok50 = (tr.n_case >= DEV_N // 2 + A.MIN_SCORE) & (tr.n_ctrl >= DEV_N // 2 + A.MIN_SCORE)
    rng = np.random.default_rng(SEED)
    devval = []
    for dg in sorted(tr.disease_group.unique()):
        n_g = int((tr.disease_group == dg).sum())
        k = max(1, int(round(DEV_FRACTION * n_g)))
        pool = tr[(tr.disease_group == dg) & ok50].accession.to_numpy()
        devval += list(rng.permutation(pool)[:k])
    devval = sorted(devval)
    st = strata_table()
    a30 = {f"{r.accession}|{r.tissue}|{r.disease}" for r in tr.itertuples()}
    st["pool_S15"] = (st.n_case >= S15) & (st.n_ctrl >= S15)
    st["pool_A30"] = st.stratum.isin(a30)
    st["devval_group"] = st.group.isin(devval)
    rows = []
    for r in tr.itertuples():
        if r.accession not in devval:
            continue
        rows.append(dict(kind="devval", accession=r.accession, group=r.accession, disease=r.disease, tissue=r.tissue,
                         disease_group=r.disease_group, platform=r.platform, n_case=r.n_case, n_ctrl=r.n_ctrl))
    for r in st.itertuples():
        rows.append(dict(kind="training_stratum", accession=r.stratum, group=r.group, disease=r.disease, tissue=r.tissue,
                         disease_group=A.DISEASE_GROUP_OF.get(r.disease, "other"), n_case=r.n_case, n_ctrl=r.n_ctrl,
                         pool_S15=r.pool_S15, pool_A30=r.pool_A30, devval_group=r.devval_group))
    missing_a30 = sorted(a30 - set(st.stratum))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUTF / f"datasets_F_{DATE}.csv", index=False)
    info = {"a_candidates": str(OUTA / f"candidates_{DATE}.csv"),
            "a_candidates_sha256": sha256_file(OUTA / f"candidates_{DATE}.csv"),
            "a_candidates_committed": git_committed(OUTA / f"candidates_{DATE}.csv"),
            "devval": devval, "devval_per_disease_group": tab[tab.kind == "devval"].disease_group.value_counts().to_dict(),
            "a_train_datasets": int(len(tr)), "a_train_evaluable_n50": int(ok50.sum()),
            "strata_min10": int(len(st)), "strata_S15": int(st.pool_S15.sum()), "strata_A30": int(st.pool_A30.sum()),
            "a30_strata_below_10_after_split_train_filter": missing_a30,
            "strata_S15_dev": int((st.pool_S15 & ~st.devval_group).sum()),
            "strata_A30_dev": int((st.pool_A30 & ~st.devval_group).sum()),
            "diseases_S15": int(st[st.pool_S15].disease.nunique()), "groups_S15": int(st[st.pool_S15].group.nunique())}
    write_json(OUTF / f"datasets_F_{DATE}.json", info)
    log(json.dumps({k: v for k, v in info.items() if k != "devval"}, default=str))
    log(f"dev-val: {devval}")


def split_table():
    return pd.read_csv(need(OUTF / f"datasets_F_{DATE}.csv"))


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]).to_numpy()


# ------------------------------------------------------------------ stage xcache (corpus rows needed by F)
def xcache_plan():
    tab = split_table()
    cand = a_candidates()
    prof, lab = profiles(), labels()
    st = tab[tab.kind == "training_stratum"]
    need_rows = set()
    for r in st.itertuples():
        ca, co = stratum_rows(r)
        need_rows |= set(ca) | set(co)
    held = list(tab[tab.kind == "devval"].accession) + list(cand[cand.group.isin(["test", "val"])].accession)
    for acc in held:
        b = prof[(prof.group == acc) & prof.id.isin(lab.index)]
        need_rows |= set(b.index)
    return np.array(sorted(need_rows))


def stage_xcache():
    WORKF.mkdir(parents=True, exist_ok=True)
    rows = xcache_plan()
    f_rows, f_x = WORKF / "xcache_rows.npy", WORKF / "xcache_f16.npy"
    if f_rows.exists() and np.array_equal(np.load(f_rows), rows) and f_x.exists():
        log(f"xcache up to date ({len(rows)} rows)")
        return
    beta = np.load(need(CORPUS_DIR / "beta_f16.npy"), mmap_mode="r")
    out = np.lib.format.open_memmap(f_x, mode="w+", dtype=np.float16, shape=(len(rows), beta.shape[1]))
    t0 = time.time()
    # contiguous runs read in one slice each
    starts = np.flatnonzero(np.r_[True, np.diff(rows) != 1])
    ends = np.r_[starts[1:], len(rows)]
    for k, (s, e) in enumerate(zip(starts, ends)):
        out[s:e] = beta[rows[s]:rows[e - 1] + 1]
        if k % 200 == 0:
            log(f"xcache {e}/{len(rows)} rows ({time.time() - t0:.0f} s)")
    out.flush()
    np.save(f_rows, rows)
    log(f"xcache written: {len(rows)} rows ({time.time() - t0:.0f} s)")


class XCache:
    """Cached corpus rows (memory-mapped; contiguous runs are read with one slice each)."""

    def __new__(cls):
        if "xcache" not in _CACHE:
            o = super().__new__(cls)
            o.rows = np.load(need(WORKF / "xcache_rows.npy"))
            o.X = np.load(need(WORKF / "xcache_f16.npy"), mmap_mode="r")
            o.pos = pd.Series(np.arange(len(o.rows)), index=o.rows)
            _CACHE["xcache"] = o
        return _CACHE["xcache"]

    def get(self, corpus_rows, dtype=np.float32):
        pos = self.pos.loc[np.asarray(corpus_rows)].to_numpy()
        out = np.empty((len(pos), self.X.shape[1]), dtype)
        order = np.argsort(pos, kind="stable")
        sp = pos[order]
        starts = np.flatnonzero(np.r_[True, np.diff(sp) != 1])
        ends = np.r_[starts[1:], len(sp)]
        for a, b in zip(starts, ends):
            out[order[a:b]] = self.X[sp[a]:sp[b - 1] + 1]
        return out


# ------------------------------------------------------------------ held-out profiles (betas and ids only)
def heldout_profiles(scope):
    """{dataset: (ids, X)} for the duplicate check. scope 'dev' = dev-val + corpus primary; 'all' adds external."""
    tab, cand = split_table(), a_candidates()
    prof, lab, xc = profiles(), labels(), XCache()
    out = {}
    for acc in list(tab[tab.kind == "devval"].accession) + list(cand[cand.group.isin(["test", "val"])].accession):
        b = prof[(prof.group == acc) & prof.id.isin(lab.index)]
        rows = np.sort(b.index.to_numpy())
        out[acc] = (prof.loc[rows, "id"].astype(str).to_numpy(), xc.get(rows))
    missing = []
    if scope == "all":
        for acc in cand[cand.group == "external"].accession:
            fx, fp = WORKAB / f"X_{acc}.npy", WORKAB / f"profiles_{acc}.csv"
            if not (fx.exists() and fp.exists()):
                missing.append(acc)
                continue
            out[acc] = (pd.read_csv(fp).id.astype(str).to_numpy(), np.load(fx).astype(np.float32))
    return out, missing


# ------------------------------------------------------------------ stage dedup
def prep_dup(X, sh, centred, mu):
    import torch
    X = X[:, sh].float()
    m = torch.nanmean(X, 0)
    X = torch.where(torch.isfinite(X), X, m)
    if centred:
        X = X - mu[sh]
    X = X - X.mean(1, keepdim=True)
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)


def stage_dedup(device, scope):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    dev = torch.device(device)
    tab = split_table()
    prof = profiles()
    mu = torch.from_numpy(np.load(need(CORE))["mu"].astype(np.float32)).to(dev)
    xc = XCache()
    held, missing = heldout_profiles(scope)
    devset = set(tab[tab.kind == "devval"].accession)
    kinds = {a: ("devval" if a in devset else "primary") for a in held}
    H = {}
    for acc, (ids, X) in held.items():
        T = torch.from_numpy(X).to(dev, torch.float16)
        H[acc] = (ids, T, torch.isfinite(T).float().mean(0))
    st = tab[tab.kind == "training_stratum"]
    pairs, removed, near = [], [], []
    t0 = time.time()
    for gi, (g, sg) in enumerate(st.groupby("group")):
        rows = set()
        for r in sg.itertuples():
            ca, co = stratum_rows(r)
            rows |= set(ca) | set(co)
        rows = np.array(sorted(rows))
        ids_t = prof.loc[rows, "id"].astype(str).to_numpy()
        Xt = torch.from_numpy(xc.get(rows)).to(dev)
        obs_t = torch.isfinite(Xt).float().mean(0)
        for acc, (ids_h, Th, obs_h) in H.items():
            if acc == g:
                continue
            sh = (obs_t >= DUP_SHARED_OBS) & (obs_h >= DUP_SHARED_OBS)
            nsh = int(sh.sum())
            id_dup = np.isin(ids_t, ids_h)
            base = dict(training_group=g, heldout=acc, heldout_kind=kinds[acc], n_train=len(rows), n_heldout=len(ids_h),
                        shared_cpgs=nsh, id_shared=int(id_dup.sum()))
            if nsh < 1000:
                pairs.append({**base, "n_flagged": int(id_dup.sum())})
                for i in np.flatnonzero(id_dup):
                    removed.append(dict(training_group=g, training_id=ids_t[i], corpus_row=int(rows[i]), heldout=acc,
                                        heldout_kind=kinds[acc], heldout_id=ids_t[i], reason="shared GSM id"))
                continue
            P = {name: prep_dup(Xt, sh, name == "centred", mu) @ prep_dup(Th, sh, name == "centred", mu).T
                 for name in ("raw", "centred")}
            k2 = min(2, P["centred"].shape[1])
            top = P["centred"].topk(k2, dim=1)
            best = top.values[:, 0].cpu().numpy()
            second = top.values[:, 1].cpu().numpy() if k2 > 1 else np.full(len(best), -1.0)
            jbest = top.indices[:, 0].cpu().numpy()
            raw_at = P["raw"].gather(1, top.indices[:, :1])[:, 0].cpu().numpy()
            raw_max = P["raw"].max(1).values.cpu().numpy()
            crit_abs = best >= DUP_CENTRED_ABS
            crit_margin = (best >= DUP_CENTRED_MARGIN_MIN) & (best - second >= DUP_MARGIN)
            flag = id_dup | crit_abs | crit_margin
            pairs.append({**base, "max_raw": float(raw_max.max()), "max_centred": float(best.max()),
                          "median_best_centred": float(np.median(best)), "q99_best_centred": float(np.quantile(best, 0.99)),
                          "n_raw_gt_099": int((raw_max > DUP_RAW).sum()), "n_centred_ge_098": int(crit_abs.sum()),
                          "n_centred_margin": int(crit_margin.sum()), "n_flagged": int(flag.sum())})
            for i in np.flatnonzero(flag | (best >= 0.94)):
                reason = ";".join(r_ for r_, c_ in (("shared GSM id", id_dup[i]), ("centred r >= 0.98", crit_abs[i]),
                                                     ("centred r >= 0.95 and margin >= 0.01", crit_margin[i])) if c_)
                (removed if flag[i] else near).append(dict(
                    training_group=g, training_id=ids_t[i], corpus_row=int(rows[i]), heldout=acc, heldout_kind=kinds[acc],
                    heldout_id=ids_h[jbest[i]], reason=reason, centred_r=float(best[i]), centred_second=float(second[i]),
                    raw_r=float(raw_at[i])))
            del P, top
        del Xt
        if gi % 20 == 0:
            log(f"dedup group {gi + 1}/{st.group.nunique()} ({time.time() - t0:.0f} s)")
    cols = ["training_group", "training_id", "corpus_row", "heldout", "heldout_kind", "heldout_id", "reason", "centred_r",
            "centred_second", "raw_r"]
    pd.DataFrame(pairs).to_csv(OUTF / f"duplicates_pairs_{DATE}.csv", index=False)
    pd.DataFrame(removed, columns=cols).to_csv(OUTF / f"duplicates_removed_{DATE}.csv", index=False)
    pd.DataFrame(near, columns=cols).to_csv(OUTF / f"duplicates_near_not_removed_{DATE}.csv", index=False)
    rem = pd.DataFrame(removed, columns=cols)
    write_json(OUTF / f"duplicates_{DATE}.json", {
        "scope": scope, "external_missing": missing, "raw_reported": DUP_RAW, "centred_abs": DUP_CENTRED_ABS,
        "centred_margin_min": DUP_CENTRED_MARGIN_MIN, "margin": DUP_MARGIN, "shared_obs": DUP_SHARED_OBS,
        "heldout_datasets": sorted(held), "pairs": len(pairs),
        "removed_profiles_any": int(rem.corpus_row.nunique()),
        "removed_profiles_primary": int(rem[rem.heldout_kind == "primary"].corpus_row.nunique()),
        "removed_profiles_devval": int(rem[rem.heldout_kind == "devval"].corpus_row.nunique()),
        "removed_by_heldout": rem.groupby("heldout").corpus_row.nunique().to_dict(),
        "removed_by_training_group": rem.groupby("training_group").corpus_row.nunique().to_dict()})
    log(f"dedup done: {len(pairs)} pairs, {rem.corpus_row.nunique()} training profiles flagged; external missing {missing}")


def removed_rows(phase):
    """Corpus rows removed from training: phase 'dev' (dev-val + primary duplicates) or 'final' (primary only)."""
    f = OUTF / f"duplicates_removed_{DATE}.csv"
    meta = json.loads(need(OUTF / f"duplicates_{DATE}.json").read_text())
    if phase == "final" and (meta["scope"] != "all" or meta["external_missing"]):
        sys.exit("Cannot proceed - the duplicate check against the external held-out datasets has not been run (dedup --scope all)")
    r = pd.read_csv(need(f))
    if phase == "final":
        r = r[r.heldout_kind == "primary"]
    else:
        # development keeps the removal set it started with (dev-val and corpus primary datasets); duplicates of
        # external held-out profiles matter for the final training only (amendment of 24 Oct)
        ext = set(a_candidates().query("group == 'external'").accession)
        r = r[~r.heldout.isin(ext)]
    return set(r.corpus_row.astype(int))


# ------------------------------------------------------------------ models
def _cached_load(path):
    """torch.load of a checkpoint, kept in memory for later runs of the same process (CPU tensors)."""
    import torch
    key = ("ckpt", str(path))
    if key not in _CACHE:
        _CACHE[key] = torch.load(need(path), map_location="cpu", weights_only=False)
    return _CACHE[key]


def load_encoder(which, dev):
    """Released encoder (as load_checkpoint of methylgpt2_model, then the stage weights), rebuilt from cached
    checkpoints."""
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import PPCACore
    from methylgpt2_model import BioHeads, MethylGPT2
    ck = _cached_load(AGE_STAGE if which == "age_stage" else MULTITASK)
    base = _cached_load(HERE / ck["ckpt"])
    c = base["config"]
    if ("core", c["core"]) not in _CACHE:
        _CACHE[("core", c["core"])] = PPCACore.load(c["core"], rank=c["rank"])
    core = PPCACore(_CACHE[("core", c["core"])].mu, _CACHE[("core", c["core"])].V, _CACHE[("core", c["core"])].lam,
                    _CACHE[("core", c["core"])].sigma2).to(dev)
    model = MethylGPT2(core, d_tok=c["d_tok"], d=c["d"], heads=c["heads"], n_latent=c["latents"],
                       n_score_tokens=c["score_tokens"], n_self=c["self_layers"], d_read=c.get("d_read", 64),
                       encoder=c.get("encoder", "tokens"), d_dense=c.get("d_dense", 2048),
                       n_dense_tokens=c.get("dense_tokens", 32)).to(dev)
    model.load_state_dict(base["model"])
    model.load_state_dict(ck["model"])
    heads = None
    if which == "multitask":
        heads = BioHeads(core.rank + model.latents.shape[1], len(ck["tissues"]) + 1, len(ck["bio_names"])).to(dev)
        heads.load_state_dict(ck["heads"])
    meta = {k: ck[k] for k in ("fm", "fs", "tissues", "bio_names", "bio_median", "bio_scale") if k in ck}
    model.eval()
    return model, core, heads, meta


def embed_profiles(model, core, X, dev, batch=32, z7=False, S=None):
    """A's embedding (embed_batch, TF32 as in A); optionally the input of the last self-attention block. S: optional
    precomputed core scores (n x 1,536) for the rows of X."""
    import torch
    from methylgpt2_model import embed_batch
    out = np.empty((len(X), core.rank + model.latents.shape[1]), np.float32)
    zs = [] if z7 else None
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch], dtype=np.float32)).to(dev)
            ctx = torch.isfinite(x)
            s = None if S is None else torch.from_numpy(np.ascontiguousarray(S[i:i + batch], dtype=np.float32)).to(dev)
            if not z7:
                e = embed_batch(model, core, x, ctx) if s is None else embed_with_s(model, x, ctx, s)
                out[i:i + batch] = e.float().cpu().numpy()
                continue
            z, s = encode_upto(model, core, x, ctx, len(model.enc_self) - 1, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zf = model.enc_self[-1](z)
            out[i:i + batch] = model.embedding(zf, s).float().cpu().numpy()
            zs.append(z.to(torch.bfloat16).cpu())
    return out, (torch.cat(zs) if z7 else None)


def encode_upto(model, core, x, ctx, n_blocks, s=None):
    """encode_dense (methylgpt2_model) up to n_blocks self-attention blocks; returns z and core scores s
    (posterior computed unless given)."""
    import torch
    if s is None:
        s = core.posterior(x, ctx.float())
    with torch.autocast("cuda", enabled=False):
        xs, c, e = model.inputs(x, ctx, s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        w = ctx.to(c.dtype)
        scale = (model.P / w.sum(1, keepdim=True).clamp_min(1.0)).to(c.dtype)
        h = model.dense_c(c * w * scale) + model.dense_e(e * scale) + model.dense_w(w)
        dense_tokens = model.dense_mlp(h).view(c.shape[0], model.n_dense_tokens, -1)
        sz = s / torch.sqrt(model.core.lam)
        score_tokens = model.score_proj(sz).view(c.shape[0], model.n_score_tokens, -1)
        z = torch.cat([model.latents.expand(c.shape[0], -1, -1).to(score_tokens.dtype), score_tokens,
                       dense_tokens.to(score_tokens.dtype)], 1)
        for blk in model.enc_self[:n_blocks]:
            z = blk(z)
    return z, s


def embed_with_s(model, x, ctx, s):
    """embed_batch (methylgpt2_model) with the core scores s given (gradients flow through the encoder)."""
    import torch
    with torch.autocast("cuda", enabled=False):
        xs, c, e = model.inputs(x, ctx, s)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode(xs, c, e, ctx, s)
    return model.embedding(z, s)


def scores_from_emb(E, core):
    """Core scores s from cached embeddings (first 1,536 dimensions are s / sqrt(lam))."""
    return E[:, :1536] * np.sqrt(core.lam.cpu().numpy())[None, :]


def make_adapter(cfg, d_in, mean, sd):
    import torch
    import torch.nn as nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("m", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("s", torch.as_tensor(sd, dtype=torch.float32))
            if cfg["adapter"] == "linear":
                self.net = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(d_in, cfg["K"]))
            elif cfg["adapter"] == "linear_id":
                assert cfg["K"] == d_in
                lin = nn.Linear(d_in, d_in)
                with torch.no_grad():
                    lin.weight.copy_(torch.eye(d_in))
                    lin.bias.zero_()
                self.net = nn.Sequential(nn.Dropout(cfg["dropout"]), lin)
            else:
                self.net = nn.Sequential(nn.Linear(d_in, cfg["hidden"]), nn.GELU(), nn.Dropout(cfg["dropout"]),
                                         nn.Linear(cfg["hidden"], cfg["K"]))

        def forward(self, e):
            return self.net((e.float() - self.m) / self.s)

    return Adapter()


def episodic_loss(R, ys, sup, qry, log_lam, log_tau):
    """R (E, m, K) features of E episodes (padded), ys (E, m) 0/1, sup/qry (E, m) 0/1 masks. Ridge in dual form on
    support features standardized with the support mean and SD; BCE of tau * score on the query."""
    import torch
    import torch.nn.functional as Fn
    ns = sup.sum(1, keepdim=True)
    m = (R * sup[..., None]).sum(1, keepdim=True) / ns[..., None]
    v = (((R - m) ** 2) * sup[..., None]).sum(1, keepdim=True) / ns[..., None]
    Rz = (R - m) / (v.sqrt() + 1e-4)
    Rs = Rz * sup[..., None]
    t = (2 * ys - 1) * sup
    Kmat = Rs @ Rs.transpose(1, 2)
    lam = torch.exp(log_lam) * ns
    eye = torch.eye(R.shape[1], device=R.device)[None]
    alpha = torch.linalg.solve(Kmat + lam[..., None] * eye, t[..., None])          # padded rows: alpha = 0
    score = (Rz @ Rs.transpose(1, 2) @ alpha)[..., 0]
    logit = torch.exp(log_tau) * score
    ll = Fn.binary_cross_entropy_with_logits(logit, ys, reduction="none")
    return torch.where(qry > 0, ll, torch.zeros_like(ll)).sum() / qry.sum()


class StrataSampler:
    def __init__(self, strata, seed):
        """strata: list of dicts with group, case (array of item indices), ctrl (array)."""
        self.strata = strata
        self.rng = np.random.default_rng(seed)
        self.groups = sorted({s["group"] for s in strata})
        self.by_group = {g: [i for i, s in enumerate(strata) if s["group"] == g] for g in self.groups}

    def pick(self, k):
        gs = self.rng.choice(len(self.groups), k, replace=True)
        return [int(self.rng.choice(self.by_group[self.groups[g]])) for g in gs]

    def heads_batch(self, k, per_class):
        idx, y, sid = [], [], []
        for s_i in self.pick(k):
            s = self.strata[s_i]
            for cls, pool in ((1.0, s["case"]), (0.0, s["ctrl"])):
                take = self.rng.choice(pool, per_class, replace=len(pool) < per_class)
                idx += list(take)
                y += [cls] * per_class
                sid += [s_i] * per_class
        return np.array(idx), np.array(y, np.float32), np.array(sid)

    def episodes(self, k):
        """Padded episodes: item index (E, m), labels, support mask, query mask; m = 100."""
        E, M = k, 100
        idx = np.zeros((E, M), int)
        ys, sup, qry = np.zeros((E, M), np.float32), np.zeros((E, M), np.float32), np.zeros((E, M), np.float32)
        for e, s_i in enumerate(self.pick(k)):
            s = self.strata[s_i]
            mm = min(len(s["case"]), len(s["ctrl"]))
            ks = min(25, mm - 5)
            kq = min(25, mm - ks)
            pos = 0
            for cls, pool in ((1.0, s["case"]), (0.0, s["ctrl"])):
                perm = self.rng.permutation(pool)
                for part, kk, mask in ((perm[:ks], ks, sup), (perm[ks:ks + kq], kq, qry)):
                    idx[e, pos:pos + kk] = part
                    ys[e, pos:pos + kk] = cls
                    mask[e, pos:pos + kk] = 1
                    pos += kk
        return idx, ys, sup, qry


def training_strata(cfg, phase):
    """List of strata dicts (case/ctrl corpus rows) for a pool and phase, duplicates removed."""
    tab = split_table()
    st = tab[tab.kind == "training_stratum"].copy()
    st = st[truthy(st[f"pool_{cfg['pool']}"])]
    if phase == "dev":
        st = st[~truthy(st.devval_group)]
    rem = removed_rows(phase)
    out = []
    for r in st.itertuples():
        ca, co = stratum_rows(r)
        ca, co = np.array([x for x in ca if x not in rem]), np.array([x for x in co if x not in rem])
        if len(ca) >= MIN_STRATUM and len(co) >= MIN_STRATUM:
            out.append(dict(stratum=r.accession, group=r.group, disease=r.disease, tissue=r.tissue, case_rows=ca, ctrl_rows=co))
    return out


def devval_sets(with_x=True):
    """{acc: (contrast DataFrame, X)} for the dev-val datasets (A's contrast for corpus datasets); X is None when
    with_x is False (D1 uses cached embeddings)."""
    key = f"devval_{with_x}"
    if key not in _CACHE:
        _CACHE[key] = _devval_sets(with_x)
    return _CACHE[key]


def _devval_sets(with_x):
    tab, xc = split_table(), XCache()
    prof, lab = profiles(), labels()
    out = {}
    for r in tab[tab.kind == "devval"].itertuples():
        b = prof[(prof.group == r.accession) & prof.id.isin(lab.index)]
        rows = np.sort(b.index.to_numpy())
        X = None if A.platform_array(r.platform) else xc.get(rows)      # array inference needs the betas
        sub, cc = corpus_contrast(r.accession, r.platform, X, rows)
        if sub is None:
            continue
        out[r.accession] = (sub, xc.get(sub.corpus_row.to_numpy()) if with_x else None)
    return out


# ------------------------------------------------------------------ protocol on stacked datasets
def protocol_aucs(Z, ys, n, dev, accs=None):
    """A's protocol (draws_for, fit_select_score, auc) for datasets {acc: Z}, {acc: y} at size n, stacked in one call.
    Returns per-draw DataFrame."""
    accs = sorted(Z) if accs is None else accs
    blocks, yall, train_sets, score_sets, owners, folds = [], [], [], [], [], []
    off = 0
    for acc in accs:
        y = ys[acc]
        trs = A.draws_for(int(A.accession_number(acc)), n, y)
        for r, tr in enumerate(trs):
            mask = np.ones(len(y), bool)
            mask[tr] = False
            train_sets.append(tr + off)
            score_sets.append(np.flatnonzero(mask) + off)
            owners.append((acc, r))
            folds.append(r)
        blocks.append(np.asarray(Z[acc], np.float32))
        yall.append(y)
        off += len(y)
    Zall, yall = np.concatenate(blocks), np.concatenate(yall)
    res = A.fit_select_score(Zall, yall, train_sets, dev, folds, score_sets=score_sets)
    rows = []
    for (acc, r), ss, rr in zip(owners, score_sets, res):
        ysc = yall[ss]
        rows.append({"accession": acc, "n_train": n, "draw": r, "auc": A.auc(rr["score"], ysc), "alpha": rr["alpha"],
                     "lambda_index": rr["lambda_index"], "nonzero": rr["nonzero"], "cv_deviance": rr["cv_deviance"],
                     "n_score_case": int(ysc.sum()), "n_score_ctrl": int((1 - ysc).sum()), "features": int(Zall.shape[1])})
    return pd.DataFrame(rows)


def evaluable(y, n):
    return (y == 1).sum() >= n // 2 + A.MIN_SCORE and (y == 0).sum() >= n // 2 + A.MIN_SCORE


# ------------------------------------------------------------------ stage embed (frozen age-stage embeddings)
def stage_embed(device):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    xc = XCache()
    hm27 = np.load(need(MASKS))["hm27"]
    model = core = None
    for view, f in (("full", WORKF / "emb_age_stage.npy"), ("hm27", WORKF / "emb_age_stage_hm27.npy")):
        if f.exists() and len(np.load(f, mmap_mode="r")) == len(xc.rows):
            log(f"embeddings ({view}) up to date")
            continue
        if model is None:
            model, core, _, _ = load_encoder("age_stage", dev)
        t0 = time.time()
        out = np.empty((len(xc.rows), 2048), np.float32)
        for i in range(0, len(xc.rows), 2048):
            X = np.asarray(xc.X[i:i + 2048], np.float32)
            if view == "hm27":
                X[:, ~hm27] = np.nan
            out[i:i + 2048] = embed_profiles(model, core, X, dev)[0]
            log(f"embedded ({view}) {min(i + 2048, len(xc.rows))}/{len(xc.rows)} ({time.time() - t0:.0f} s)")
        np.save(f, out)


def emb_cache():
    if "emb" not in _CACHE:
        _CACHE["emb"] = np.load(need(WORKF / "emb_age_stage.npy"))
    return XCache(), _CACHE["emb"]


# ------------------------------------------------------------------ stage stacking_check and devref
def stage_stacking_check(device):
    import torch
    dev = torch.device(device)
    xc, E = emb_cache()
    dv = devval_sets()
    accs = sorted(dv)[:2]
    Z = {a: E[xc.pos.loc[dv[a][0].corpus_row.to_numpy()].to_numpy()] for a in accs}
    ys = {a: (dv[a][0].role == "case").to_numpy(np.float32) for a in accs}
    stacked = protocol_aucs(Z, ys, DEV_N, dev, accs)
    single = pd.concat([protocol_aucs({a: Z[a]}, {a: ys[a]}, DEV_N, dev, [a]) for a in accs])
    m = stacked.merge(single, on=["accession", "draw"], suffixes=("_stacked", "_single"))
    out = {"datasets": accs, "draws": int(len(m)), "max_abs_auc_diff": float((m.auc_stacked - m.auc_single).abs().max()),
           "mean_auc_stacked": float(m.auc_stacked.mean()), "mean_auc_single": float(m.auc_single.mean()),
           "same_selected_setting": float(((m.alpha_stacked == m.alpha_single) & (m.lambda_index_stacked == m.lambda_index_single)).mean())}
    write_json(OUTF / f"stacking_check_{DATE}.json", out)
    log(f"stacking check: {out}")


def stage_devref(device):
    import torch
    dev = torch.device(device)
    xc, E = emb_cache()
    dv = devval_sets()
    mu = np.load(need(CORE))["mu"].astype(np.float32)
    ys = {a: (s.role == "case").to_numpy(np.float32) for a, (s, _) in dv.items()}
    accs = [a for a in sorted(dv) if evaluable(ys[a], DEV_N)]
    parts = []
    for rep in ("embedding", "latent", "raw", "clocks"):
        Z = {}
        for a in accs:
            sub, X = dv[a]
            if rep in ("embedding", "latent"):
                e = E[xc.pos.loc[sub.corpus_row.to_numpy()].to_numpy()]
                Z[a] = e if rep == "embedding" else e[:, :1536]
            elif rep == "raw":
                Z[a] = np.where(np.isfinite(X), X, mu[None, :]).astype(np.float32)
            else:
                cl, _ = A.dataset_clocks(a, sub.assign(corpus_row=sub.corpus_row.astype(int)))
                cols = [k for k in A.CLOCKS if cl[k].notna().mean() >= A.AVAIL]
                if cols:
                    Z[a] = cl[cols].to_numpy(np.float32)
        if rep == "clocks":
            # clock sets differ in size between datasets: one call per feature count
            by_p = {}
            for a, z in Z.items():
                by_p.setdefault(z.shape[1], []).append(a)
            for p_, aa in by_p.items():
                parts.append(protocol_aucs({a: Z[a] for a in aa}, ys, DEV_N, dev, aa).assign(representation=rep))
        else:
            parts.append(protocol_aucs(Z, ys, DEV_N, dev, accs).assign(representation=rep))
        log(f"devref {rep} done")
    d = pd.concat(parts)
    d.to_csv(WORKF / "dev_reference_per_draw.csv.gz", index=False)
    per = d.groupby(["representation", "accession"]).auc.mean().unstack(0) - 0.5
    per.to_csv(OUTF / f"dev_reference_{DATE}.csv")
    log("dev-val references (mean AUC - 0.5, n = 50):\n" + per.mean().to_string())


# ------------------------------------------------------------------ stage train (development) and final
def config_from(args_json):
    cfg = dict(DEFAULT)
    over = json.loads(args_json) if args_json else {}
    if over.get("design") == "D3":
        cfg.update(D3_OVERRIDES)
    cfg.update(over)
    return cfg


def run_id(cfg):
    keys = sorted(k for k in cfg if k not in ("eval_steps",))
    h = hashlib.sha256(json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:10]
    return f"{cfg['design']}_{cfg['pool']}_{cfg['objective']}_{cfg['adapter']}_K{cfg['K']}_{h}"


class Trainer:
    """Trains one configuration; representation() maps beta matrices to the output representation."""

    def __init__(self, cfg, phase, dev):
        import torch
        import torch.nn as nn
        self.cfg, self.phase, self.dev = cfg, phase, dev
        torch.manual_seed(cfg["seed"])
        self.xc, self.E = emb_cache()
        strata = training_strata(cfg, phase)
        rows = np.array(sorted({x for s in strata for x in np.r_[s["case_rows"], s["ctrl_rows"]]}))
        self.item_rows = rows
        self.item_pos = pd.Series(np.arange(len(rows)), index=rows)
        self.strata = [dict(s, case=self.item_pos.loc[s["case_rows"]].to_numpy(), ctrl=self.item_pos.loc[s["ctrl_rows"]].to_numpy())
                       for s in strata]
        self.sampler = StrataSampler(self.strata, cfg["seed"])
        cache_pos = self.xc.pos.loc[rows].to_numpy()
        design = cfg["design"]
        self.model = self.core = self.bioheads = None
        self.hm27 = np.load(need(MASKS))["hm27"]
        self.hm27_t = torch.from_numpy(self.hm27).to(dev)
        if design in ("D1", "D2"):
            Etr = self.E[cache_pos]
            self.Etr = torch.from_numpy(Etr).to(dev)
            mean, sd = Etr.mean(0), Etr.std(0) + 1e-6
            if cfg.get("aug27k"):
                self.Etr27 = torch.from_numpy(np.load(need(WORKF / "emb_age_stage_hm27.npy"), mmap_mode="r")[cache_pos]).to(dev)
            if design == "D2":
                self.model, self.core, _, _ = load_encoder("age_stage", dev)
                self.z7 = self._z7_cache(rows, False)
                self.z7_27 = self._z7_cache(rows, True) if cfg.get("aug27k") else None
                self.blk = self.model.enc_self[-1]
                self.sum_norm = self.model.sum_norm
                for p_ in self.model.parameters():
                    p_.requires_grad_(False)
                for p_ in list(self.blk.parameters()) + list(self.sum_norm.parameters()):
                    p_.requires_grad_(True)
        else:
            self.model, self.core, self.bioheads, self.mt_meta = load_encoder("multitask", dev)
            key = ("Xtr", hashlib.sha256(np.asarray(rows).tobytes()).hexdigest())
            if key not in _CACHE:
                _CACHE[key] = self.xc.get(rows, np.float16)
            self.Xtr = _CACHE[key]
            self.Str = torch.from_numpy(scores_from_emb(self.E[cache_pos], self.core)).to(dev)
            if cfg.get("aug27k"):
                E27 = np.load(need(WORKF / "emb_age_stage_hm27.npy"), mmap_mode="r")
                self.Str27 = torch.from_numpy(scores_from_emb(np.asarray(E27[cache_pos]), self.core)).to(dev)
            sub = np.sort(np.random.default_rng(cfg["seed"]).choice(len(rows), min(4000, len(rows)), replace=False))
            e0 = embed_profiles(self.model, self.core, self.Xtr[sub].astype(np.float32), dev,
                                S=self.Str[torch.as_tensor(sub, device=dev)].cpu().numpy())[0]
            mean, sd = e0.mean(0), e0.std(0) + 1e-6
            self._aux_setup(phase)
        self.adapter = make_adapter(cfg, 2048, mean, sd).to(dev)
        S = len(self.strata)
        self.Wh = nn.Parameter(torch.zeros(S, cfg["K"], device=dev))
        self.bh = nn.Parameter(torch.zeros(S, device=dev))
        self.log_lam = nn.Parameter(torch.zeros((), device=dev))
        self.log_tau = nn.Parameter(torch.zeros((), device=dev))
        groups = [{"params": list(self.adapter.parameters()) + [self.Wh, self.bh, self.log_lam, self.log_tau],
                   "lr": cfg["lr"], "weight_decay": cfg["wd"]}]
        if design == "D2":
            groups.append({"params": list(self.blk.parameters()) + list(self.sum_norm.parameters()), "lr": cfg["lr_enc"],
                           "weight_decay": 0.01})
        if design == "D3":
            enc = [p_ for n_, p_ in self.model.named_parameters() if not n_.startswith(
                ("u1", "u2", "q_emb", "out", "dec", "h_norm", "ds_head", "q_load", "sum_proj", "h_proj"))]
            groups.append({"params": enc, "lr": cfg["lr_enc"], "weight_decay": 0.01})
            groups.append({"params": list(self.bioheads.parameters()), "lr": cfg["lr_enc"] * 10, "weight_decay": 0.01})
        self.opt = torch.optim.AdamW(groups, eps=1e-10)
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.n_strata, self.n_items, self.n_groups = S, len(rows), len(self.sampler.groups)
        log(f"trainer {design} {phase}: {S} strata, {self.n_groups} groups, {len(rows)} profiles")

    def _z7_cache(self, rows, view27):
        import torch
        key = ("z7", view27, hashlib.sha256(np.asarray(rows).tobytes()).hexdigest())
        if key not in _CACHE:
            _CACHE[key] = self._z7_compute(rows, view27)
        return _CACHE[key]

    def _z7_compute(self, rows, view27):
        import torch
        zs = []
        Ev = np.load(need(WORKF / ("emb_age_stage_hm27.npy" if view27 else "emb_age_stage.npy")), mmap_mode="r")
        for i in range(0, len(rows), 2048):
            X = self.xc.get(rows[i:i + 2048])
            if view27:
                X[:, ~self.hm27] = np.nan
            S = scores_from_emb(np.asarray(Ev[self.xc.pos.loc[rows[i:i + 2048]].to_numpy()]), self.core)
            zs.append(embed_profiles(self.model, self.core, X, self.dev, z7=True, S=S)[1])
            log(f"z7 cache ({'27K view' if view27 else 'full'}) {min(i + 2048, len(rows))}/{len(rows)}")
        return torch.cat(zs)

    def _aux_setup(self, phase):
        """Auxiliary multitask data as train_multitask: clean ages, sex, tissue, standardized clock targets."""
        import torch
        key = ("aux", self.cfg["seed"], phase)
        if key in _CACHE:
            self.aux_X, S, self.aux = _CACHE[key]
            self.aux_S = torch.from_numpy(S).to(self.dev)
            self.aux_rng = np.random.default_rng(self.cfg["seed"] + 11)
            return
        tm = A._module("train_multitask", "train_multitask_2026-10-01.py")
        sys.path.insert(0, str(HERE))
        from methylgpt2_lib import F
        prof = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"))
        age, sex, tissue, top = tm.clean_labels(prof)
        assert top == self.mt_meta["tissues"]
        lab = pd.read_parquet(need(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
        age = lab.age_clean.to_numpy(float)
        is_train = (prof.split == "train").to_numpy()
        clocks = pd.read_parquet(need(CORPUS_DIR / "clock_targets_2026-10-01.parquet"))
        assert list(c for c in clocks.columns if c not in ("id", "split")) == list(self.mt_meta["bio_names"])
        B = clocks[self.mt_meta["bio_names"]].to_numpy(float)
        Bz = np.clip((B - self.mt_meta["bio_median"]) / self.mt_meta["bio_scale"], -6, 6)
        labelled = np.flatnonzero(is_train & (np.isfinite(age) | np.isfinite(sex) | np.isfinite(tissue) | np.isfinite(Bz).any(1)))
        rem = removed_rows(phase)
        labelled = np.array([r for r in labelled if r not in rem])
        sub = np.sort(np.random.default_rng(self.cfg["seed"] + 7).choice(labelled, min(AUX_SUBSET, len(labelled)), replace=False))
        f = WORKF / f"aux_{self.cfg['seed']}_{phase}_{len(sub)}.npy"
        fr = WORKF / f"aux_{self.cfg['seed']}_{phase}_{len(sub)}_rows.npy"
        if f.exists() and fr.exists() and np.array_equal(np.load(fr), sub):
            self.aux_X = np.load(f, mmap_mode="r")
        else:
            beta = np.load(need(CORPUS_DIR / "beta_f16.npy"), mmap_mode="r")
            t0 = time.time()
            self.aux_X = np.lib.format.open_memmap(f, mode="w+", dtype=np.float16, shape=(len(sub), beta.shape[1]))
            for i in range(0, len(sub), 1000):
                self.aux_X[i:i + 1000] = beta[sub[i:i + 1000]]
                log(f"aux profiles {min(i + 1000, len(sub))}/{len(sub)} ({time.time() - t0:.0f} s)")
            self.aux_X.flush()
            np.save(fr, sub)
            self.aux_X = np.load(f, mmap_mode="r")
        self.aux_X = np.asarray(self.aux_X).copy()
        # one fixed mask per auxiliary profile (amendment): fraction u * 0.5 of observed CpGs, u ~ U(0, 1)
        mrng = np.random.default_rng(self.cfg["seed"] + 13)
        fs_ = WORKF / f"aux_{self.cfg['seed']}_{phase}_{len(sub)}_scores.npy"
        for i in range(0, len(sub), 1000):
            blk = self.aux_X[i:i + 1000].astype(np.float32)
            u = mrng.random((len(blk), 1)) * 0.5
            blk[mrng.random(blk.shape) < u] = np.nan
            self.aux_X[i:i + 1000] = blk.astype(np.float16)
        if fs_.exists():
            S = np.load(fs_)
        else:
            S = np.empty((len(sub), self.core.rank), np.float32)
            t0 = time.time()
            with torch.no_grad():
                for i in range(0, len(sub), 32):
                    x = torch.from_numpy(self.aux_X[i:i + 32].astype(np.float32)).to(self.dev)
                    S[i:i + 32] = self.core.posterior(x, torch.isfinite(x).float()).cpu().numpy()
                    if i % 3200 == 0:
                        log(f"aux posteriors {i + 32}/{len(sub)} ({time.time() - t0:.0f} s)")
            np.save(fs_, S)
        self.aux_S = torch.from_numpy(S).to(self.dev)
        self.aux = dict(age=age[sub], sex=sex[sub], tissue=tissue[sub], Bz=Bz[sub].astype(np.float32),
                        F=F(np.nan_to_num(age[sub], nan=30.0)))
        self.aux_rng = np.random.default_rng(self.cfg["seed"] + 11)
        _CACHE[key] = (self.aux_X, S, self.aux)

    def features(self, items, view27=None):
        """2,048-dimensional embeddings of training items (positions into item_rows); view27 (bool per item) selects
        the 27K-array view."""
        import torch
        d = self.cfg["design"]
        it = torch.as_tensor(items, device=self.dev)
        v27 = None if view27 is None or not view27.any() else torch.as_tensor(view27, device=self.dev)
        if d == "D1":
            e = self.Etr[it]
            return e if v27 is None else torch.where(v27[:, None], self.Etr27[it], e)
        if d == "D2":
            z = self.z7[torch.as_tensor(items)]
            s_part = self.Etr[it, :1536]
            if v27 is not None:
                z = torch.where(torch.as_tensor(view27)[:, None, None], self.z7_27[torch.as_tensor(items)], z)
                s_part = torch.where(v27[:, None], self.Etr27[it, :1536], s_part)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = self.blk(z.to(self.dev, non_blocking=True))
            return torch.cat([s_part, self.sum_norm(z.float().mean(1))], -1)
        x = torch.from_numpy(self.Xtr[items].astype(np.float32)).to(self.dev)
        ctx = torch.isfinite(x)
        s = self.Str[it]
        if v27 is not None:
            ctx = ctx & (self.hm27_t[None, :] | ~v27[:, None])
            s = torch.where(v27[:, None], self.Str27[it], s)
        return embed_with_s(self.model, x, ctx, s)

    def view_mask(self, n, per_episode=None):
        if not self.cfg.get("aug27k"):
            return None
        if per_episode is None:
            return self.sampler.rng.random(n) < 0.25
        return np.repeat(self.sampler.rng.random(per_episode[0]) < 0.25, per_episode[1])

    def aux_loss(self):
        import torch
        import torch.nn.functional as Fn
        from methylgpt2_model import embed_batch
        a = self.aux
        idx = self.aux_rng.choice(len(self.aux_X), 32, replace=False)
        x = torch.from_numpy(self.aux_X[idx].astype(np.float32)).to(self.dev)
        e = embed_with_s(self.model, x, torch.isfinite(x), self.aux_S[torch.as_tensor(idx, device=self.dev)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pa, ps, pt, pb = self.bioheads(e)
        fm, fs = self.mt_meta["fm"], self.mt_meta["fs"]
        yb = torch.from_numpy(a["Bz"][idx]).to(self.dev)
        mb = torch.isfinite(yb)
        lb = ((pb.float() - torch.nan_to_num(yb)) ** 2)[mb].mean() if mb.any() else torch.zeros((), device=self.dev)
        ya, ys_, yt = (torch.from_numpy(a[k][idx]).to(self.dev) for k in ("age", "sex", "tissue"))
        ma, ms, mt = torch.isfinite(ya), torch.isfinite(ys_), torch.isfinite(yt)
        fa = torch.from_numpy(((a["F"][idx] - fm) / fs).astype(np.float32)).to(self.dev)
        la = ((pa.float() - fa) ** 2)[ma].mean() if ma.any() else torch.zeros((), device=self.dev)
        ls = Fn.binary_cross_entropy_with_logits(ps.float()[ms], ys_[ms].float()) if ms.any() else torch.zeros((), device=self.dev)
        lt = Fn.cross_entropy(pt.float()[mt], yt[mt].long()) if mt.any() else torch.zeros((), device=self.dev)
        return la + 0.2 * ls + 0.2 * lt + 1.0 * lb, (float(la), float(ls), float(lt), float(lb))

    def train_modules(self, on):
        mods = [self.adapter] + ([self.blk] if self.cfg["design"] == "D2" else []) + \
               ([self.model, self.bioheads] if self.cfg["design"] == "D3" else [])
        for m in mods:
            m.train(on)
        if self.cfg["design"] == "D2":
            self.model.eval()
            self.blk.train(on)

    def step(self, k):
        import torch
        import torch.nn.functional as Fn
        cfg = self.cfg
        warm = min(1.0, k / 300)
        cos = 0.5 * (1 + math.cos(math.pi * k / cfg["steps"]))
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * warm * (0.05 + 0.95 * cos)
        self.train_modules(True)
        loss = torch.zeros((), device=self.dev)
        parts = {}
        if cfg["objective"] in ("heads", "both"):
            idx, y, sid = self.sampler.heads_batch(cfg["strata_per_step"], cfg["per_class"])
            r = self.adapter(self.features(idx, self.view_mask(len(idx))))
            sid_t = torch.as_tensor(sid, device=self.dev)
            logit = (r * self.Wh[sid_t]).sum(-1) + self.bh[sid_t]
            lh = Fn.binary_cross_entropy_with_logits(logit, torch.as_tensor(y, device=self.dev))
            loss = loss + lh
            parts["heads"] = float(lh)
        if cfg["objective"] in ("episodic", "both"):
            idx, ys, sup, qry = self.sampler.episodes(cfg["episodes_per_step"])
            flat = idx.reshape(-1)
            r = self.adapter(self.features(flat, self.view_mask(len(flat), idx.shape))).view(idx.shape[0], idx.shape[1], -1)
            le = episodic_loss(r, *(torch.as_tensor(v, device=self.dev) for v in (ys, sup, qry)), self.log_lam, self.log_tau)
            loss = loss + le
            parts["episodic"] = float(le)
        if cfg["design"] == "D3" and cfg["w_aux"] > 0:
            la, aux_parts = self.aux_loss()
            loss = loss + cfg["w_aux"] * la
            parts["aux"] = float(la)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        params = [p_ for g in self.opt.param_groups for p_ in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.opt.step()
        return parts

    def representation(self, X=None, cache_rows=None):
        """Output representation for beta matrix X (n x 49,156, NaN unobserved). D1 may use cached embeddings."""
        import torch
        self.train_modules(False)
        d, out_kind = self.cfg["design"], self.cfg["output"]
        with torch.no_grad():
            if d == "D1":
                e = self.E[self.xc.pos.loc[cache_rows].to_numpy()] if cache_rows is not None else \
                    embed_profiles(*self._age_stage(), X, self.dev)[0]
                E_ = torch.from_numpy(e).to(self.dev)
            elif d == "D2":
                S = None if cache_rows is None else scores_from_emb(self.E[self.xc.pos.loc[cache_rows].to_numpy()], self.core)
                e, z7 = embed_profiles(self.model, self.core, X, self.dev, z7=True, S=S)
                outs = []
                for i in range(0, len(z7), 256):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = self.blk(z7[i:i + 256].to(self.dev))
                    outs.append(torch.cat([torch.from_numpy(e[i:i + 256, :1536]).to(self.dev), self.sum_norm(z.float().mean(1))], -1))
                E_ = torch.cat(outs)
            else:
                S = None if cache_rows is None else scores_from_emb(self.E[self.xc.pos.loc[cache_rows].to_numpy()], self.core)
                E_ = torch.from_numpy(embed_profiles(self.model, self.core, X, self.dev, S=S)[0]).to(self.dev)
            if out_kind == "encoder" and d != "D1":
                return E_.float().cpu().numpy()
            return torch.cat([self.adapter(E_[i:i + 4096]) for i in range(0, len(E_), 4096)]).float().cpu().numpy()

    def _age_stage(self):
        if not hasattr(self, "_ag"):
            m, c, _, _ = load_encoder("age_stage", self.dev)
            self._ag = (m, c)
        return self._ag

    def state(self):
        st = {"cfg": self.cfg, "phase": self.phase, "adapter": self.adapter.state_dict(), "Wh": self.Wh.detach().cpu(),
              "bh": self.bh.detach().cpu(), "log_lam": float(self.log_lam), "log_tau": float(self.log_tau),
              "strata": [s["stratum"] for s in self.strata]}
        if self.cfg["design"] == "D2":
            st["blk"] = self.blk.state_dict()
            st["sum_norm"] = self.sum_norm.state_dict()
        if self.cfg["design"] == "D3":
            st["model"] = self.model.state_dict()
            st["bioheads"] = self.bioheads.state_dict()
        return st


def dev_metric(trainer, dv, dev):
    ys = {a: (s.role == "case").to_numpy(np.float32) for a, (s, _) in dv.items()}
    accs = [a for a in sorted(dv) if evaluable(ys[a], DEV_N)]
    Z = {}
    for a in accs:
        sub, X = dv[a]
        Z[a] = trainer.representation(X=X if trainer.cfg["design"] != "D1" else None, cache_rows=sub.corpus_row.to_numpy())
    d = protocol_aucs(Z, ys, DEV_N, dev, accs)
    per = d.groupby("accession").auc.mean() - 0.5
    return float(per.mean()), per, d


def stage_train(device, cfg_json, tag):
    """cfg_json: one configuration (JSON object) or a list of them, run in order in this process; configurations
    whose every evaluation step is already in the development log are skipped."""
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    over = json.loads(cfg_json) if cfg_json else {}
    for o in (over if isinstance(over, list) else [over]):
        cfg = config_from(json.dumps(o))
        rid = run_id(cfg)
        done = set(pd.read_csv(DEV_LOG).query("run_id == @rid").step) if DEV_LOG.exists() else set()
        if set(cfg["eval_steps"]) <= done:
            log(f"{rid}: already in the development log, skipped")
            continue
        train_one(cfg, rid, tag, dev)


def train_one(cfg, rid, tag, dev):
    import torch
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    dv = devval_sets(with_x=cfg["design"] != "D1")
    tr = Trainer(cfg, "dev", dev)
    t0 = time.time()
    hist = []
    for k in range(1, cfg.get("stop_at", cfg["steps"]) + 1):     # stop_at: S5 seed runs stop at the selected step
        hist.append(tr.step(k))
        if k % 100 == 0:
            h = pd.DataFrame(hist[-100:]).mean().to_dict()
            log(f"{rid} step {k}: " + " ".join(f"{a} {b:.4f}" for a, b in h.items()) + f" ({time.time() - t0:.0f} s)")
        if k in cfg["eval_steps"]:
            te = time.time()
            M, per, d = dev_metric(tr, dv, dev)
            ck = CKPT_DIR / f"{rid}_step{k}.pt"
            torch.save(tr.state(), ck)
            d.to_csv(WORKF / f"devdraws_{rid}_step{k}.csv.gz", index=False)
            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "run_id": rid, "step": k,
                   "devval_metric": M, "devval_datasets": int(len(per)), "train_strata": tr.n_strata,
                   "train_groups": tr.n_groups, "train_profiles": tr.n_items,
                   "train_loss_last100": json.dumps(pd.DataFrame(hist[-100:]).mean().round(5).to_dict()),
                   "per_dataset": json.dumps(per.round(5).to_dict()), "checkpoint": str(ck.relative_to(HERE)),
                   "config": json.dumps(cfg, sort_keys=True), "train_seconds": round(te - t0, 1),
                   "eval_seconds": round(time.time() - te, 1)}
            old = pd.read_csv(DEV_LOG) if DEV_LOG.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(DEV_LOG, index=False)
            log(f"DEV {rid} step {k}: dev-val mean AUC - 0.5 = {M:.4f} over {len(per)} datasets")
    del tr
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ lock, final training, held-out evaluation
DESIGN_ORDER = {"D1": 0, "D2": 1, "D3": 2}


def select_from_log():
    d = pd.read_csv(need(DEV_LOG))
    d = d[d.tag != "S5_seed"]
    cfgs = d.config.map(json.loads)
    d = d.assign(design_rank=cfgs.map(lambda c: DESIGN_ORDER[c["design"]]), K=cfgs.map(lambda c: c["K"]),
                 metric_r=d.devval_metric.round(4))
    d = d.sort_values(["metric_r", "design_rank", "step", "K"], ascending=[False, True, True, True])
    return d.iloc[0], d


def lock_path(k):
    return OUTF / (f"locked_design_{DATE}.json" if k == 1 else f"locked_design_{DATE}_{k}.json")


def dev_summary():
    """One row per development run: configuration, dev-val metric at each evaluated step, best step."""
    d = pd.read_csv(need(DEV_LOG))
    rows = []
    for rid, g in d.groupby("run_id", sort=False):
        c = json.loads(g.config.iloc[0])
        r = {"tag": g.tag.iloc[0], "run_id": rid, **{k: c.get(k) for k in ("design", "pool", "objective", "adapter", "K", "hidden",
             "dropout", "aug27k", "lr", "wd", "lr_enc", "w_aux", "output", "steps", "seed")}}
        for st_, m in zip(g.step, g.devval_metric):
            r[f"devval_step{int(st_)}"] = m
        r["best_step"] = int(g.step.iloc[int(np.argmax(g.devval_metric.to_numpy()))])
        r["best_devval_metric"] = float(g.devval_metric.max())
        r["train_strata"], r["train_groups"], r["train_profiles"] = int(g.train_strata.iloc[0]), int(g.train_groups.iloc[0]), int(g.train_profiles.iloc[0])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUTF / f"dev_summary_{DATE}.csv", index=False)
    return out


def stage_lock(k):
    dev_summary()
    best, d = select_from_log()
    cfg = json.loads(best.config)
    cfg["steps_selected"] = int(best.step)
    fr = OUTA / f"datasets_frozen_{DATE}.csv"
    lock = {"lock_number": k, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "selected_run": best.run_id, "selected_step": int(best.step), "devval_metric": float(best.devval_metric),
            "devval_datasets": int(best.devval_datasets), "development_runs_logged": int(len(d)),
            "selection_rule": "highest dev-val mean (AUC - 0.5) at n = 50; ties (1e-4) to simpler design, fewer steps, smaller K",
            "config": cfg, "final_training": {"pool": cfg["pool"], "phase": "final (all train-split groups, primary duplicates removed)",
                                              "seed": SEED, "steps": int(best.step), "schedule_steps": int(cfg["steps"]),
                                              "checkpoint": str((CKPT_DIR / f"final_{k}.pt").relative_to(HERE))},
            "development_checkpoint": best.checkpoint,
            "evaluation_protocol": {"heldout": "A's primary set (test, val, external), included rows of A's frozen table",
                                    "sizes": A.SIZES, "primary_size": A.PRIMARY_SIZE, "draws": A.N_DRAWS,
                                    "classifier": "A.fit_select_score (logistic elastic net, A's grid and CV)",
                                    "metric": "mean over datasets of (AUC - 0.5)", "comparators": A.COMPARATORS,
                                    "also_reported": ["embedding (released MethylGPT)", "latent"],
                                    "bootstrap": {"draws": A.BOOT, "seed": A.SEED},
                                    "rule": "PASS iff relative gain over the best of raw and clocks >= 10% and lower 95% bound of the difference > 0"},
            "a_protocol_fingerprint": protocol_fingerprint(),
            "a_frozen_table_sha256": sha256_file(fr) if fr.exists() else None,
            "a_frozen_table_committed": git_committed(fr) if fr.exists() else False,
            "script_sha256": sha256_file(Path(__file__))}
    write_json(lock_path(k), lock)
    log(f"locked: {best.run_id} step {best.step} dev-val {best.devval_metric:.4f}")


def read_lock(k):
    return json.loads(need(lock_path(k)).read_text())


def stage_final(device, k):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    lock = read_lock(k)
    cfg = dict(lock["config"])          # cfg["steps"] keeps the development schedule length (warm-up and cosine)
    stop = int(lock["selected_step"])  # training stops at the selected checkpoint step, as in development
    cfg["seed"] = SEED
    cfg["eval_steps"] = []
    tr = Trainer(cfg, "final", dev)
    t0 = time.time()
    hist = []
    for s in range(1, stop + 1):
        hist.append(tr.step(s))
        if s % 100 == 0:
            log(f"final step {s}: " + " ".join(f"{a} {b:.4f}" for a, b in pd.DataFrame(hist[-100:]).mean().items()) +
                f" ({time.time() - t0:.0f} s)")
    st = tr.state()
    st["train_strata_diseases"] = sorted({s["disease"] for s in tr.strata})
    torch.save(st, HERE / lock["final_training"]["checkpoint"])
    write_json(OUTF / f"final_training_{k}_{DATE}.json", {"lock": k, "strata": tr.n_strata, "groups": tr.n_groups,
                                                         "profiles": tr.n_items, "steps": stop, "schedule_steps": cfg["steps"],
                                                         "seconds": round(time.time() - t0, 1),
                                                         "diseases": st["train_strata_diseases"],
                                                         "strata_list": st["strata"]})
    log(f"final model saved ({tr.n_strata} strata)")


def trainer_from_state(st, dev):
    import torch
    tr = Trainer.__new__(Trainer)
    tr.cfg, tr.phase, tr.dev = st["cfg"], st["phase"], dev
    d = tr.cfg["design"]
    tr.xc = None
    if d == "D1":
        tr.model = None
    elif d == "D2":
        tr.model, tr.core, _, _ = load_encoder("age_stage", dev)
        tr.blk, tr.sum_norm = tr.model.enc_self[-1], tr.model.sum_norm
        tr.blk.load_state_dict(st["blk"])
        tr.sum_norm.load_state_dict(st["sum_norm"])
    else:
        tr.model, tr.core, tr.bioheads, tr.mt_meta = load_encoder("multitask", dev)
        tr.model.load_state_dict(st["model"])
        tr.bioheads.load_state_dict(st["bioheads"])
    a = st["adapter"]
    tr.adapter = make_adapter(tr.cfg, 2048, a["m"].cpu().numpy(), a["s"].cpu().numpy()).to(dev)
    tr.adapter.load_state_dict(a)
    return tr


def stage_heldout(device, k):
    import torch
    sys.path.insert(0, str(HERE))
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(device)
    lock = read_lock(k)
    for p_ in (Path(__file__), lock_path(k), DEV_LOG):
        if not git_committed(p_):
            sys.exit(f"Cannot proceed - {p_} must be committed before the locked evaluation")
    if protocol_fingerprint() != lock["a_protocol_fingerprint"]:
        sys.exit("Cannot proceed - A's protocol functions or constants differ from the lock")
    fr = a_frozen()
    if lock.get("a_frozen_table_sha256") and lock["a_frozen_table_sha256"] != sha256_file(OUTA / f"datasets_frozen_{DATE}.csv"):
        sys.exit("Cannot proceed - A's frozen table changed after the lock")
    out = OUTF / f"per_draw_auc_F_eval{k}_{DATE}.csv.gz"
    if out.exists():
        sys.exit(f"locked evaluation {k} already run: {out}")
    st = torch.load(need(HERE / lock["final_training"]["checkpoint"]), map_location="cpu", weights_only=False)
    tr = trainer_from_state(st, dev)
    prim = fr[fr.group.isin(A.PRIMARY_GROUPS)].sort_values("accession")
    reps, ys, sizes = {}, {}, {}
    for d in prim.itertuples():
        pt, keep = A.load_contrast(d.accession)
        X = np.load(need(WORKAB / f"X_{d.accession}.npy"))[keep].astype(np.float32)
        reps[d.accession] = tr.representation(X=X)
        ys[d.accession] = (pt.role == "case").to_numpy(np.float32)
        sizes[d.accession] = [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
        log(f"{d.accession}: representation {reps[d.accession].shape}")
    torch.backends.cuda.matmul.allow_tf32 = False      # A's fit stage runs the classifier in full FP32 (amendment)
    parts = []
    for n in A.SIZES:
        accs = [a for a in sorted(reps) if n in sizes[a]]
        if accs:
            parts.append(protocol_aucs(reps, ys, n, dev, accs))
            log(f"n = {n}: {len(accs)} datasets")
    res = pd.concat(parts).assign(representation="F")
    res.to_csv(out, index=False)
    log(f"locked evaluation {k} written: {out}")
    # pairing check: A's released embedding (A's own feature files) through this pipeline, compared with A's
    # per-draw table in stage analyse
    emb = {a: A.features(a, "embedding")[0] for a in sorted(reps)}
    chk = [protocol_aucs(emb, ys, n, dev, [a for a in sorted(emb) if n in sizes[a]]) for n in A.SIZES]
    pd.concat(chk).assign(representation="embedding_via_F_pipeline").to_csv(
        OUTF / f"pairing_check_embedding_eval{k}_{DATE}.csv.gz", index=False)
    log("pairing check written")


def stage_analyse(k):
    fr = a_frozen().set_index("accession")
    fa = OUTA / f"per_draw_auc_{DATE}.csv.gz"
    if not (fa.exists() and git_committed(fa)):
        sys.exit(f"Cannot proceed - A's per-draw table is not committed: {fa}")
    a = pd.read_csv(fa)
    f = pd.read_csv(need(OUTF / f"per_draw_auc_F_eval{k}_{DATE}.csv.gz"))
    lock = read_lock(k)
    fin = json.loads(need(OUTF / f"final_training_{k}_{DATE}.json").read_text())
    seen_dis = set(fin["diseases"])
    prim = fr.index[fr.group.isin(A.PRIMARY_GROUPS)]
    # pairing check: identical scoring sets
    chk = f.merge(a[a.representation == "raw"], on=["accession", "n_train", "draw"], suffixes=("", "_a"))
    pairing = {"draws_F": int(len(f)), "draws_matched_to_A_raw": int(len(chk)),
               "scoring_sizes_identical": bool(((chk.n_score_case == chk.n_score_case_a) & (chk.n_score_ctrl == chk.n_score_ctrl_a)).all())}
    fe = OUTF / f"pairing_check_embedding_eval{k}_{DATE}.csv.gz"
    if fe.exists():
        e = pd.read_csv(fe).merge(a[a.representation == "embedding"], on=["accession", "n_train", "draw"], suffixes=("", "_a"))
        pairing.update({"embedding_draws_compared": int(len(e)), "embedding_auc_max_abs_diff": float((e.auc - e.auc_a).abs().max()),
                        "embedding_auc_mean_abs_diff": float((e.auc - e.auc_a).abs().mean()),
                        "embedding_same_selected_setting": float(((e.alpha == e.alpha_a) & (e.lambda_index == e.lambda_index_a)).mean())})
    allr = pd.concat([a, f], ignore_index=True)
    g = allr.groupby(["accession", "representation", "n_train"]).auc
    per = g.mean().rename("auc_mean").to_frame().join(g.std().rename("auc_sd")).reset_index()
    per = per[per.accession.isin(prim)]
    per = per.join(fr[["group", "disease", "disease_group", "tissue", "array", "n_case", "n_ctrl"]], on="accession")
    per["disease_seen_in_training"] = per.disease.isin(seen_dis)
    per.to_csv(OUTF / f"per_dataset_auc_eval{k}_{DATE}.csv", index=False)
    wide = {n: per[per.n_train == n].pivot(index="accession", columns="representation", values="auc_mean") - 0.5 for n in A.SIZES}
    seen = fr.disease.isin(seen_dis)
    sets = {"primary": prim,
            "primary|disease seen": fr.index[fr.group.isin(A.PRIMARY_GROUPS) & seen],
            "primary|disease unseen": fr.index[fr.group.isin(A.PRIMARY_GROUPS) & ~seen],
            "primary|external": fr.index[fr.group == "external"],
            "primary|corpus (test, val)": fr.index[fr.group.isin(["test", "val"])],
            "primary|27K": fr.index[fr.group.isin(A.PRIMARY_GROUPS) & (fr.array == "27K")],
            "primary|450K/EPIC/EPIC v2": fr.index[fr.group.isin(A.PRIMARY_GROUPS) & (fr.array != "27K")]}
    reps_all = ["F", "embedding", "latent", "raw", "clocks"]
    rows, res = [], {"lock": k, "pairing_check": pairing, "primary": None, "secondary": {}}
    for name, ids in sets.items():
        for n in A.SIZES:
            w = wide[n].reindex([i for i in ids if i in wide[n].index])
            if not len(w) or "F" not in w:
                continue
            r = A.compare(w, reps_all, "F", A.COMPARATORS, f"{name}|n={n}")
            if r is None:
                continue
            key = f"{name}|n={n}"
            if name == "primary" and n == A.PRIMARY_SIZE:
                res["primary"] = r
            else:
                res["secondary"][key] = r
            ve = r["vs_each"]
            rows.append({"set": name, "n_train": n, "datasets": r["datasets"], "M_F": r["methylgpt_M"],
                         "M_F_ci95_low": r["M_ci95"]["F"][0], "M_F_ci95_high": r["M_ci95"]["F"][1],
                         "M_embedding": r["M"].get("embedding"), "M_latent": r["M"].get("latent"), "M_raw": r["M"].get("raw"),
                         "M_clocks": r["M"].get("clocks"), "best_comparator": r["best_comparator"],
                         "best_comparator_M": r["best_comparator_M"], "difference": r["difference"],
                         "difference_ci95_low": r["difference_ci95"][0], "difference_ci95_high": r["difference_ci95"][1],
                         "relative_gain": r["relative_gain"], "meets_rule": r["pass"],
                         "F_minus_embedding": ve.get("embedding", {}).get("difference"),
                         "F_minus_embedding_ci95_low": (ve.get("embedding", {}).get("ci95") or [None, None])[0],
                         "F_minus_embedding_ci95_high": (ve.get("embedding", {}).get("ci95") or [None, None])[1],
                         "F_relative_gain_over_embedding": ve.get("embedding", {}).get("relative_gain"),
                         "F_minus_latent": ve.get("latent", {}).get("difference"),
                         "datasets_F_higher_than_embedding": ve.get("embedding", {}).get("datasets_methylgpt_higher"),
                         "decision_set": name == "primary" and n == A.PRIMARY_SIZE})
    pd.DataFrame(rows).to_csv(OUTF / f"summary_eval{k}_{DATE}.csv", index=False)
    p = res["primary"]
    left_out = sorted(set(wide[A.PRIMARY_SIZE].index) - set(wide[A.PRIMARY_SIZE].dropna(subset=["F", "raw", "clocks"]).index))
    rule = pd.DataFrame([{"analysis": f"F (locked evaluation {k}): disease-trained MethylGPT, 50 labelled samples, A's primary datasets",
                          "primary_metric": "mean over datasets of (AUC - 0.5), n = 50", "datasets": p["datasets"],
                          "datasets_left_out_missing_representation": ";".join(left_out),
                          "methylgpt_representation": f"F ({lock['selected_run']}, step {lock['selected_step']})",
                          "methylgpt_value": p["methylgpt_M"], "best_comparator": p["best_comparator"],
                          "best_comparator_value": p["best_comparator_M"], "relative_gain": p["relative_gain"],
                          "difference": p["difference"], "difference_ci95_low": p["difference_ci95"][0],
                          "difference_ci95_high": p["difference_ci95"][1],
                          "released_embedding_value": p["M"].get("embedding"),
                          "F_minus_released_embedding": p["vs_each"]["embedding"]["difference"],
                          "F_minus_released_embedding_ci95_low": p["vs_each"]["embedding"]["ci95"][0],
                          "F_minus_released_embedding_ci95_high": p["vs_each"]["embedding"]["ci95"][1],
                          "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
                          "outcome": "PASS" if p["pass"] else "FAIL"}])
    rule.to_csv(OUTF / f"rule_outcome_eval{k}_{DATE}.csv", index=False)
    write_json(OUTF / f"results_eval{k}_{DATE}.json", res)
    log(f"PRIMARY (lock {k}): {rule.iloc[0].to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["split", "xcache", "dedup", "embed", "stacking_check", "devref",
                                                        "train", "lock", "final", "heldout", "analyse"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--scope", default="dev", choices=["dev", "all"])
    ap.add_argument("--config", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lock", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "split":
        stage_split()
    elif a.stage == "xcache":
        stage_xcache()
    elif a.stage == "dedup":
        stage_dedup(a.device, a.scope)
    elif a.stage == "embed":
        stage_embed(a.device)
    elif a.stage == "stacking_check":
        stage_stacking_check(a.device)
    elif a.stage == "devref":
        stage_devref(a.device)
    elif a.stage == "train":
        stage_train(a.device, a.config, a.tag)
    elif a.stage == "lock":
        stage_lock(a.lock)
    elif a.stage == "final":
        stage_final(a.device, a.lock)
    elif a.stage == "heldout":
        stage_heldout(a.device, a.lock)
    else:
        stage_analyse(a.lock)


if __name__ == "__main__":
    main()
