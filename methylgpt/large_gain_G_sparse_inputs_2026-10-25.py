#!/usr/bin/env python3
"""Large-gain analysis G: few-label disease classification when each sample measures a sparse, sample-specific subset
of CpGs (as with sequencing). SIMULATED MISSINGNESS ON EMPIRICAL PROFILES: every input of this analysis is an empirical
beta profile from which CpGs were removed by a seeded random mask (defined below); no values are simulated.

Written on 25 Oct 2026, before any model is fitted under this analysis, for the large-gain analysis plan of 23 Oct 2026
(Decision rule; analysis A) and the analysis brief for G. Reuses analysis A's protocol, frozen held-out primary
set and draws (large_gain_A_disease_few_labels_2026-10-24.py, imported) and analysis F's 24 development-validation
datasets and duplicate removal (large_gain_F_disease_trained_representation_2026-10-24.py, imported). The released
MethylGPT is used as is (no training). Every result is reported, whichever representation it favours.

DATA (frozen by stage freeze in G/datasets_G_2026-10-25.csv before any fit)
  Development: F's 24 dev-val datasets (F/datasets_F_2026-10-24.csv, kind devval), contrast profiles as A defines them
  (F.devval_sets; identical sorted-profile SHA-256 to A's frozen table, checked by F).
  Held-out (touched once, after the lock): A's primary set (included test, val and external rows of
  A/datasets_frozen_2026-10-24.csv, commit 9c6a247), contrast profiles in A's order (A.load_contrast, work_AB/X_<acc>.npy),
  A's sizes per dataset (sizes_evaluated) and A's draws (A.draws_for).

MASKING (simulated missingness; fixed, shared by every representation and comparator)
  For profile id p (GSM id) and keep fraction f: h = first 8 bytes of SHA-256(p) (little-endian integer); one uniform
  value per vocabulary CpG u = numpy default_rng([20261025, h]).random(49156), CpGs in vocabulary order; among the
  profile's observed vocabulary CpGs, the k = max(1, round(f * n_observed)) with the smallest u are kept, all others are
  unobserved. Masks are nested in f (a CpG kept at 0.05 is kept at 0.10 and 0.30). f in {0.05, 0.10, 0.30}. Per
  dataset and f, the mean number of kept CpGs and the SHA-256 of the packed mask are written (G/masks_2026-10-25.csv).

REPRESENTATIONS (all computed from the masked profile only)
  MethylGPT
    embedding  A's embedding: the age-stage encoder (runs/aging_heads_clean_2026-10-01/best.pt = bundle
               weights_age_stage.pt), 2,048 dimensions, computed from the kept CpGs (F.embed_profiles, TF32 as A's
               embed stage).
    latent     its first 1,536 dimensions (latent scores / prior SD).
    imputed    MethylGPT application: the released bundle's MethylGPT.impute (artifacts/release/methylgpt2-1.1, default
               method "factor": kept values unchanged, every other vocabulary CpG = mu + V s clipped to [0, 1], s the
               posterior latent scores from the kept CpGs), then the classifier on the 49,156 CpGs (FP32 imputation, as
               analysis D).
  Comparators (what a user would otherwise do)
    raw_mean   the 49,156 CpGs, unkept values = corpus training means (mu of ppca_core_meanfilled_2026-09-30).
    raw_knn    the 49,156 CpGs, unkept values imputed by nearest neighbours: k = 10, analysis D's and the reconstruction
               benchmark's function (knn_predict_lowmem of si_reconstruction_diagnostics_2026-10-18.py; squared Euclidean
               distance over the query's kept CpGs, reference values unobserved -> mu, prediction = mean of the
               neighbours' observed values, mu when none). Reference: corpus training-split profiles (split "train"),
               float16, minus every training profile F flagged as a duplicate of a held-out or dev-val profile; for the
               development stage the 24 dev-val groups are also removed from the reference (their profiles are
               training-split profiles; a query must not find itself). Note: the released MethylGPT was trained on all
               training-split profiles without labels, dev-val profiles included, so on dev-val MethylGPT has seen the
               query profiles and kNN has not; on the held-out set neither has.
    clocks     GrimAge2, DunedinPACE, PhenoAge and Horvath (Horvathv1) computed with biolearn 0.7.1 (A's convention:
               GrimAge2 with the Horvathv1 age and every profile scored as male; A's run() reproduced) from the
               mean-filled masked profile (kept values, other vocabulary CpGs = mu); clock CpGs outside the vocabulary
               are absent from such a profile and take biolearn's default imputation. A's 10% missing-CpG rule is not
               applied: 79% of GrimAge2 and 70% of DunedinPACE CpGs are outside the vocabulary, so under the rule only
               PhenoAge and Horvath would remain at every f; all four clocks enter as one feature set.
  Complete-profile reference (same draws): A's per-draw AUCs on complete profiles (embedding, latent, raw, clocks).

CLASSIFIER, DRAWS, METRIC: A's (A.draws_for, A.fit_select_score with fold seeds = draw index, AUC on the remaining
  contrast profiles, FP32 as A's fit stage), stacked over datasets (F.protocol_aucs; identical to per-dataset calls up to
  the lambda_max null-model boundary, F's pairing check). Metric M = mean over datasets of (mean AUC over draws - 0.5).

DEVELOPMENT AND LOCK (stage dev, stage lock)
  On the 24 dev-val datasets at n = 50 (A's draws): M of every representation at every f. For each f the best comparator
  C*(f) is the comparator with the largest M at that f. Relative gain g(f, r) = (M(r, f) - M(C*(f), f)) / M(C*(f), f)
  for r in {embedding, latent, imputed}. The primary setting (f, r) is the one with the largest g (ties: smaller f,
  then the order embedding, latent, imputed). The choice is made on development data and written to
  G/locked_setting_2026-10-25.json, which is committed with the development table before the held-out stage runs.

HELD-OUT EVALUATION (stage heldout, once): every representation and comparator at every f and every size A evaluates
  (n = 20, 50, 100). PRIMARY: M(r*, f*) at n = 50 over A's primary datasets evaluable at n = 50 (every representation is
  available for every dataset here, clocks included, so 24 datasets; A's 21-dataset subset reported as secondary),
  best comparator C*(f*) = the best of raw_mean, raw_knn and clocks at f* on the same datasets, relative gain, 95%
  interval of M(r*) - M(C*) from the paired dataset bootstrap (A.compare: 2,000 resamples, seed 20261024, C* fixed on the
  full set). PASS iff relative gain >= 10% and the lower end of the interval > 0 (plan's Decision rule).
  SECONDARY: the other f and representations; n = 20 and 100; MethylGPT minus each comparator; external against corpus
  (test, val) datasets; 27K against other arrays; A's 21 datasets; complete-profile reference on the same draws.

AMENDMENT 25 Oct 2026, before any G result was examined: the classifier is called once per dataset and size, as A's
  fit stage calls it (protocol_per_dataset below), instead of stacking datasets (F.protocol_aucs). A's solver now
  treats its input as one shared matrix (commit 67f9edc), so stacking 24 datasets used 72 GB of GPU memory; per-dataset
  calls also reproduce A's numerics exactly. The partial development output written with stacking (f = 0.05,
  embedding and latent) was deleted unread and refitted.

Stages: freeze, represent (--set dev|heldout), clocks (--set; anaconda python with biolearn), fit (--set), lock, analyse.
Outputs: artifacts/large_gain_2026-10-25/G/ (README.md, datasets, masks, per-draw and per-dataset AUC, dev summary,
locked setting, summary, rule outcome). Working files (masked representations) in artifacts/large_gain_2026-10-25/work_G/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = ART / "large_gain_2026-10-25"
OUTG = OUT / "G"
WORKG = OUT / "work_G"
CORPUS_DIR = ART / "corpus_2026-09-30"
BUNDLE = ART / "release/methylgpt2-1.1"
CHECKSUMS = HERE / "package/methylgpt-1.1.checksums.sha256"
F_SCRIPT = HERE / "large_gain_F_disease_trained_representation_2026-10-24.py"
BIOLEARN_PY = "/home/kying/anaconda3/bin/python"
BIOLEARN_DATA = Path("/home/kying/anaconda3/lib/python3.12/site-packages/biolearn/data")
DATE = "2026-10-25"
SEED_G = 20261025
FRACTIONS = [0.05, 0.10, 0.30]
MGPT_REPS = ["embedding", "latent", "imputed"]
COMP_REPS = ["raw_mean", "raw_knn", "clocks"]
REPS = MGPT_REPS + COMP_REPS
CLOCKS = ["GrimAge2", "DunedinPACE", "PhenoAge", "Horvathv1"]
KNN_K = 10
LABEL = "simulated missingness on empirical profiles"


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


def load_f():
    spec = importlib.util.spec_from_file_location("large_gain_F", F_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["large_gain_F"] = m
    spec.loader.exec_module(m)
    return m


def git_committed(path):
    r = subprocess.run(["git", "-C", str(HERE), "ls-files", "--error-unmatch", str(path)], capture_output=True)
    if r.returncode:
        return False
    return subprocess.run(["git", "-C", str(HERE), "diff", "--quiet", "HEAD", "--", str(path)]).returncode == 0


def ftag(f):
    return f"f{int(round(f * 100)):02d}"


# ------------------------------------------------------------------ masks
def keep_mask(ids, X, f):
    """Kept CpGs (bool, n x 49,156) of the masking rule in the docstring."""
    out = np.zeros(X.shape, bool)
    for i, pid in enumerate(ids):
        h = int.from_bytes(hashlib.sha256(str(pid).encode()).digest()[:8], "little")
        u = np.random.default_rng([SEED_G, h]).random(X.shape[1])
        obs = np.flatnonzero(np.isfinite(X[i]))
        k = max(1, int(round(f * len(obs))))
        out[i, obs[np.argsort(u[obs], kind="stable")[:k]]] = True
    return out


# ------------------------------------------------------------------ datasets
def dev_sets(F):
    """{acc: (ids, y, X)} for the dev-val datasets (complete profiles)."""
    out = {}
    for a, (sub, X) in sorted(F.devval_sets(with_x=True).items()):
        out[a] = (sub.id.astype(str).to_numpy(), (sub.role == "case").to_numpy(np.float32), X)
    return out


def heldout_table(F):
    fr = F.a_frozen()
    return fr[fr.group.isin(F.A.PRIMARY_GROUPS)].sort_values("accession")


def heldout_sets(F):
    out = {}
    for d in heldout_table(F).itertuples():
        pt, keep = F.A.load_contrast(d.accession)
        X = np.load(need(F.WORKAB / f"X_{d.accession}.npy"))[keep].astype(np.float32)
        out[d.accession] = (pt.id.astype(str).to_numpy(), (pt.role == "case").to_numpy(np.float32), X)
    return out


def stage_freeze():
    F = load_f()
    OUTG.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, (ids, y, X) in dev_sets(F).items():
        rows.append(dict(set="dev", accession=a, n_case=int(y.sum()), n_ctrl=int((1 - y).sum()), sizes="50",
                         profiles_sha256=hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()))
    for d in heldout_table(F).itertuples():
        rows.append(dict(set="heldout", accession=d.accession, group=d.group, disease=d.disease, array=d.array,
                         n_case=int(d.n_case), n_ctrl=int(d.n_ctrl), sizes=d.sizes_evaluated, profiles_sha256=d.profiles_sha256,
                         clocks_used_in_A=d.clocks_used))
    pd.DataFrame(rows).to_csv(OUTG / f"datasets_G_{DATE}.csv", index=False)
    fa = F.OUTA / f"datasets_frozen_{F.DATE}.csv"
    write_json(OUTG / f"datasets_G_{DATE}.json", {
        "label": LABEL, "fractions": FRACTIONS, "mask_seed": SEED_G,
        "a_frozen_table": str(fa), "a_frozen_table_sha256": sha256_file(fa), "a_frozen_table_committed": git_committed(fa),
        "f_datasets_table_sha256": sha256_file(F.OUTF / f"datasets_F_{F.DATE}.csv"),
        "f_duplicates_removed_sha256": sha256_file(F.OUTF / f"duplicates_removed_{F.DATE}.csv"),
        "dev_datasets": int(sum(r["set"] == "dev" for r in rows)), "heldout_datasets": int(sum(r["set"] == "heldout" for r in rows))})
    log(f"frozen: {pd.DataFrame(rows).groupby('set').size().to_dict()}")


# ------------------------------------------------------------------ stage represent
def knn_reference(F, which, dev):
    """Training-split reference rows: duplicates F flagged removed; dev-val groups removed for 'dev'."""
    import torch
    prof = pd.read_parquet(need(CORPUS_DIR / "profiles.parquet"))
    rows = prof.index[(prof.split == "train").to_numpy()].to_numpy()
    dup = pd.read_csv(need(F.OUTF / f"duplicates_removed_{F.DATE}.csv"))
    drop = set(dup.corpus_row.astype(int))
    if which == "dev":
        dv = set(F.split_table().query("kind == 'devval'").accession)
        drop |= set(prof.index[prof.group.isin(dv)])
    rows = np.sort(np.array([r for r in rows if r not in drop]))
    beta = np.load(need(CORPUS_DIR / "beta_f16.npy"), mmap_mode="r")
    R = torch.empty((len(rows), beta.shape[1]), dtype=torch.float16, device=dev)
    starts = np.flatnonzero(np.r_[True, np.diff(rows) != 1])
    ends = np.r_[starts[1:], len(rows)]
    t0 = time.time()
    for k, (s, e) in enumerate(zip(starts, ends)):
        for c in range(s, e, 8192):
            c2 = min(e, c + 8192)
            R[c:c2] = torch.from_numpy(np.asarray(beta[rows[c]:rows[c2 - 1] + 1])).to(dev)
        if k % 500 == 0:
            log(f"kNN reference {e}/{len(rows)} rows ({time.time() - t0:.0f} s)")
    return R, rows


def stage_represent(which, device):
    import torch
    F = load_f()
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    cuda_preinit()
    dev = torch.device(device)
    if which == "heldout":
        lock = OUTG / f"locked_setting_{DATE}.json"
        if not (lock.exists() and git_committed(lock)):
            sys.exit("Cannot proceed - the locked setting must be committed before held-out representations are computed")
    sets = dev_sets(F) if which == "dev" else heldout_sets(F)
    mu = np.load(need(F.CORE))["mu"].astype(np.float32)
    clock_rows = clock_cpg_rows()
    wd = WORKG / which
    wd.mkdir(parents=True, exist_ok=True)
    mrows = []
    # MethylGPT embedding (age-stage encoder, TF32 as A's embed stage)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, core, _, _ = F.load_encoder("age_stage", dev)
    for f in FRACTIONS:
        for a, (ids, y, X) in sets.items():
            fe = wd / f"{a}_{ftag(f)}_embedding.npy"
            keep = keep_mask(ids, X, f)
            mrows.append(dict(set=which, accession=a, f=f, profiles=len(ids), mean_kept=float(keep.sum(1).mean()),
                              min_kept=int(keep.sum(1).min()), mean_observed=float(np.isfinite(X).sum(1).mean()),
                              mask_sha256=hashlib.sha256(np.packbits(keep).tobytes()).hexdigest(), label=LABEL))
            Xm = np.where(keep, X, np.nan).astype(np.float32)
            if not fe.exists():
                np.save(fe, F.embed_profiles(model, core, Xm, dev)[0])
            fc = wd / f"{a}_{ftag(f)}_clockin.parquet"
            if not fc.exists():
                Z = np.where(keep, X, mu[None, :])[:, clock_rows[0]]
                pd.DataFrame(Z.T.astype(np.float64), index=clock_rows[1], columns=ids).to_parquet(fc)
        log(f"{which} {ftag(f)}: embeddings and clock inputs written")
    pd.DataFrame(mrows).to_csv(OUTG / f"masks_{which}_{DATE}.csv", index=False)
    del model
    torch.cuda.empty_cache()
    # MethylGPT imputation (release bundle API, factor method, FP32 as analysis D)
    torch.backends.cuda.matmul.allow_tf32 = False
    want = dict(reversed(l.split()) for l in need(CHECKSUMS).read_text().splitlines() if l.strip())
    bad = sorted(fn for fn in want if sha256_file(BUNDLE / fn) != want[fn])
    if bad:
        sys.exit(f"release bundle checksum mismatch: {bad}")
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt import MethylGPT
    mg = MethylGPT.from_pretrained(str(BUNDLE), device=str(dev))
    vocab = list(np.load(need(F.MASKS))["vocab"].astype(str))
    if mg.vocab != vocab:
        sys.exit("bundle vocabulary differs from the platform-mask vocabulary")
    for f in FRACTIONS:
        for a, (ids, y, X) in sets.items():
            fi = wd / f"{a}_{ftag(f)}_imputed.npy"
            if fi.exists():
                continue
            keep = keep_mask(ids, X, f)
            Xm = np.where(keep, X, np.nan).astype(np.float32)
            imp = mg.impute(pd.DataFrame(Xm, columns=vocab), batch_size=8).to_numpy(np.float32)
            if not np.array_equal(imp[keep], X[keep]):
                sys.exit(f"{a}: impute() changed kept values")
            np.save(fi, imp)
        log(f"{which} {ftag(f)}: MethylGPT imputation written")
    del mg
    torch.cuda.empty_cache()
    # nearest neighbours
    spec = importlib.util.spec_from_file_location("recon_diag", HERE / "si_reconstruction_diagnostics_2026-10-18.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    todo = [(f, a) for f in FRACTIONS for a in sets if not (wd / f"{a}_{ftag(f)}_knn.npy").exists()]
    if todo:
        free, total = torch.cuda.mem_get_info(dev)
        log(f"GPU free {free / 2**30:.1f} of {total / 2**30:.1f} GiB before loading the kNN reference")
        R, rrows = knn_reference(F, which, dev)
        mu_h = torch.as_tensor(mu, device=dev).half()
        write_json(OUTG / f"knn_reference_{which}_{DATE}.json", {"rows": int(len(rrows)),
                   "rows_sha256": hashlib.sha256(rrows.astype(np.int64).tobytes()).hexdigest()})
        for f, a in todo:
            ids, y, X = sets[a]
            keep = keep_mask(ids, X, f)
            Xt = torch.from_numpy(np.where(keep, X, np.nan).astype(np.float32)).to(dev)
            kt = torch.from_numpy(keep).to(dev)
            pr = rd.knn_predict_lowmem(R, mu_h, Xt, kt, ~kt, k=KNN_K, batch=512, chunk=8192)
            out = np.where(keep, X, np.nan).astype(np.float32)
            for i, p in enumerate(pr):
                out[i, ~keep[i]] = p
            if not np.isfinite(out).all():
                sys.exit(f"{a}: kNN output has missing values")
            np.save(wd / f"{a}_{ftag(f)}_knn.npy", out)
            del Xt, kt
            log(f"{which} {ftag(f)} {a}: kNN done")
        del R
        torch.cuda.empty_cache()


def clock_cpg_rows():
    """Vocabulary positions and ids of the clock CpGs (four clocks and DunedinPACE's gold-standard CpGs)."""
    s = set()
    for fn, col in (("Horvath1.csv", 0), ("PhenoAge.csv", 0), ("DunedinPACE.csv", 0)):
        s |= set(pd.read_csv(need(BIOLEARN_DATA / fn)).iloc[:, col].astype(str))
    s |= set(pd.read_csv(need(BIOLEARN_DATA / "GrimAgeV2.csv"))["var"].astype(str))
    s |= set(pd.read_csv(need(BIOLEARN_DATA / "DunedinPACE_Gold_Means.csv"), index_col=0).index.astype(str))
    vocab = np.load(need(ART / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    pos = np.array([i for i, c in enumerate(vocab) if c in s])
    return pos, list(vocab[pos])


# ------------------------------------------------------------------ stage clocks (biolearn interpreter, no torch)
def stage_clocks(which):
    from biolearn.data_library import GeoData
    from biolearn.model_gallery import ModelGallery
    gal = ModelGallery()

    def run(dnam):
        """A's run(): four clocks with biolearn defaults; GrimAge2 with ClockBase's convention (age = Horvathv1, male)."""
        out = pd.DataFrame(index=dnam.columns)
        meta0 = pd.DataFrame({"age": np.nan, "sex": 2}, index=dnam.columns)
        horvath = gal.get("Horvathv1").predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
        for clock, name in {"GrimAge2": "GrimAgeV2", "DunedinPACE": "DunedinPACE", "PhenoAge": "PhenoAge", "Horvathv1": "Horvathv1"}.items():
            model = gal.get(name)
            sites = [s for s in model.methylation_sites() if s.startswith(("cg", "ch"))]
            out[f"{clock}_cpgs_absent"] = 1 - dnam.reindex(sites).notna().sum(0) / len(sites)
            if clock == "GrimAge2":
                meta = pd.DataFrame({"age": horvath.to_numpy(float), "sex": 2}, index=dnam.columns)
                val = model.predict(GeoData(meta, dnam.copy()))["DNAmGrimAge"].reindex(dnam.columns)
            elif clock == "Horvathv1":
                val = horvath
            else:
                val = model.predict(GeoData(meta0.copy(), dnam.copy()))["Predicted"].reindex(dnam.columns)
            out[clock] = val
        return out

    wd = WORKG / which
    for fc in sorted(wd.glob("*_clockin.parquet")):
        fo = fc.with_name(fc.name.replace("_clockin.parquet", "_clocks.csv"))
        if fo.exists():
            continue
        t0 = time.time()
        run(pd.read_parquet(fc)).to_csv(fo)
        log(f"{fc.name}: clocks ({time.time() - t0:.0f} s)")


# ------------------------------------------------------------------ stage fit
def protocol_per_dataset(A, Z, ys, n, dev, accs):
    """A's stage_fit loop for one size: A.draws_for, A.fit_select_score (fold seeds = draw index, scoring on all
    remaining contrast profiles), A.auc; one call per dataset."""
    rows = []
    for acc in accs:
        y = ys[acc]
        trs = A.draws_for(int(A.accession_number(acc)), n, y)
        res = A.fit_select_score(np.asarray(Z[acc], np.float32), y, trs, dev, fold_seeds=list(range(A.N_DRAWS)))
        for r, (tr, rr) in enumerate(zip(trs, res)):
            mask = np.ones(len(y), bool)
            mask[tr] = False
            ysc = y[mask]
            rows.append({"accession": acc, "n_train": n, "draw": r, "auc": A.auc(rr["score"], ysc), "alpha": rr["alpha"],
                         "lambda_index": rr["lambda_index"], "nonzero": rr["nonzero"], "cv_deviance": rr["cv_deviance"],
                         "n_score_case": int(ysc.sum()), "n_score_ctrl": int((1 - ysc).sum()), "features": int(Z[acc].shape[1])})
    return pd.DataFrame(rows)


def features(F, which, a, f, rep, X, ids, mu):
    wd = WORKG / which
    if rep in ("embedding", "latent"):
        E = np.load(need(wd / f"{a}_{ftag(f)}_embedding.npy"))
        return E if rep == "embedding" else E[:, :1536]
    if rep == "imputed":
        return np.load(need(wd / f"{a}_{ftag(f)}_imputed.npy"))
    if rep == "raw_knn":
        return np.load(need(wd / f"{a}_{ftag(f)}_knn.npy"))
    if rep == "raw_mean":
        return np.where(keep_mask(ids, X, f), X, mu[None, :]).astype(np.float32)
    c = pd.read_csv(need(wd / f"{a}_{ftag(f)}_clocks.csv"), index_col=0).reindex(ids)
    return c[CLOCKS].to_numpy(np.float32)


def stage_fit(which, device):
    import torch
    F = load_f()
    sys.path.insert(0, str(HERE))
    from methylgpt_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False          # A's fit stage precision
    dev = torch.device(device)
    if which == "heldout":
        lock = OUTG / f"locked_setting_{DATE}.json"
        if not (lock.exists() and git_committed(lock)):
            sys.exit("Cannot proceed - the locked setting must be committed before the held-out evaluation")
        sizes = {d.accession: [int(x) for x in str(d.sizes_evaluated).split(";") if x and x != "nan"]
                 for d in heldout_table(F).itertuples()}
        sets = heldout_sets(F)
    else:
        sets = dev_sets(F)
        sizes = {a: [50] for a in sets}
    mu = np.load(need(F.CORE))["mu"].astype(np.float32)
    outf = OUTG / f"per_draw_auc_{which}_{DATE}.csv.gz"
    done = pd.read_csv(outf) if outf.exists() else pd.DataFrame(columns=["f", "representation"])
    parts = [done] if len(done) else []
    ys = {a: v[1] for a, v in sets.items()}
    for f in FRACTIONS:
        for rep in REPS:
            if ((done.f == f) & (done.representation == rep)).any():
                continue
            t0 = time.time()
            Z = {a: features(F, which, a, f, rep, X, ids, mu) for a, (ids, y, X) in sets.items()}
            for n in sorted({n for v in sizes.values() for n in v}):
                accs = [a for a in sorted(Z) if n in sizes[a]]
                if not accs:
                    continue
                parts.append(protocol_per_dataset(F.A, Z, ys, n, dev, accs).assign(f=f, representation=rep, masking=LABEL))
            pd.concat(parts, ignore_index=True).to_csv(outf, index=False)
            log(f"{which} {ftag(f)} {rep}: fitted ({time.time() - t0:.0f} s)")
            del Z


# ------------------------------------------------------------------ stage lock and analyse
def stage_lock():
    d = pd.read_csv(need(OUTG / f"per_draw_auc_dev_{DATE}.csv.gz"))
    per = d[d.n_train == 50].groupby(["f", "representation", "accession"]).auc.mean() - 0.5
    M = per.groupby(["f", "representation"]).mean().unstack()
    rows = []
    for f in FRACTIONS:
        best = max(COMP_REPS, key=lambda c: M.loc[f, c])
        for r in MGPT_REPS:
            rows.append(dict(f=f, representation=r, M=M.loc[f, r], best_comparator=best, best_comparator_M=M.loc[f, best],
                             relative_gain=(M.loc[f, r] - M.loc[f, best]) / M.loc[f, best],
                             **{f"M_{c}": M.loc[f, c] for c in COMP_REPS}))
    tab = pd.DataFrame(rows)
    tab["datasets"] = int(per.index.get_level_values("accession").nunique())
    tab["label"] = LABEL
    tab.to_csv(OUTG / f"dev_summary_{DATE}.csv", index=False)
    order = {r: i for i, r in enumerate(MGPT_REPS)}
    best = sorted(rows, key=lambda r: (-round(r["relative_gain"], 6), r["f"], order[r["representation"]]))[0]
    write_json(OUTG / f"locked_setting_{DATE}.json", {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"), "chosen_on": "development data: F's 24 dev-val datasets, n = 50",
        "rule": "largest relative gain of a MethylGPT representation over the best comparator at the same f",
        "f": best["f"], "representation": best["representation"], "devval_M": best["M"],
        "devval_best_comparator": best["best_comparator"], "devval_best_comparator_M": best["best_comparator_M"],
        "devval_relative_gain": best["relative_gain"], "label": LABEL,
        "a_frozen_table_sha256": sha256_file(load_f().OUTA / "datasets_frozen_2026-10-24.csv"),
        "script_sha256": sha256_file(Path(__file__))})
    log(f"locked: f = {best['f']}, {best['representation']} (dev-val gain {best['relative_gain']:.3f} over {best['best_comparator']})")
    log("\n" + tab.round(4).to_string())


def stage_analyse():
    F = load_f()
    A = F.A
    lock = json.loads(need(OUTG / f"locked_setting_{DATE}.json").read_text())
    fr = heldout_table(F).set_index("accession")
    d = pd.read_csv(need(OUTG / f"per_draw_auc_heldout_{DATE}.csv.gz"))
    a = pd.read_csv(need(F.OUTA / f"per_draw_auc_{F.DATE}.csv.gz"))
    a = a[a.accession.isin(fr.index) & a.representation.isin(["embedding", "latent", "raw", "clocks"])]
    g = d.groupby(["f", "accession", "representation", "n_train"]).auc
    per = g.mean().rename("auc_mean").to_frame().join(g.std().rename("auc_sd")).reset_index()
    ga = a.groupby(["accession", "representation", "n_train"]).auc
    pera = ga.mean().rename("auc_mean").to_frame().join(ga.std().rename("auc_sd")).reset_index().assign(f=1.0)
    pera["representation"] = "complete_" + pera.representation
    allp = pd.concat([per, pera], ignore_index=True).join(fr[["group", "disease", "array"]], on="accession")
    allp["masking"] = np.where(allp.f < 1, LABEL, "complete profiles (A)")
    allp.to_csv(OUTG / f"per_dataset_auc_heldout_{DATE}.csv", index=False)
    a21 = [x for x in fr.index if x in set(a[(a.representation == "clocks") & (a.n_train == 50)].accession)]
    sets = {"primary": list(fr.index), "A's 21 datasets": a21, "external": list(fr.index[fr.group == "external"]),
            "corpus (test, val)": list(fr.index[fr.group.isin(["test", "val"])]), "27K": list(fr.index[fr.array == "27K"]),
            "450K/EPIC/EPIC v2": list(fr.index[fr.array != "27K"])}
    rows, res = [], {"lock": lock, "primary": None, "secondary": {}}
    for f in FRACTIONS:
        for n in A.SIZES:
            w = per[(per.f == f) & (per.n_train == n)].pivot(index="accession", columns="representation", values="auc_mean") - 0.5
            wc = pera[pera.n_train == n].pivot(index="accession", columns="representation", values="auc_mean") - 0.5
            for name, ids in sets.items():
                ww = w.reindex([i for i in ids if i in w.index])
                if not len(ww):
                    continue
                for mg in MGPT_REPS:
                    r = A.compare(ww, REPS, mg, COMP_REPS, f"{name}|f={f}|n={n}|{mg}")
                    if r is None:
                        continue
                    key = f"{name}|f={f}|n={n}|{mg}"
                    decision = name == "primary" and n == A.PRIMARY_SIZE and f == lock["f"] and mg == lock["representation"]
                    if decision:
                        res["primary"] = r
                    else:
                        res["secondary"][key] = r
                    cref = wc.reindex(ww.index)
                    rows.append({"set": name, "f": f, "n_train": n, "methylgpt_representation": mg, "datasets": r["datasets"],
                                 "M_methylgpt": r["methylgpt_M"], "M_ci95_low": r["M_ci95"][mg][0], "M_ci95_high": r["M_ci95"][mg][1],
                                 **{f"M_{x}": r["M"].get(x) for x in REPS}, "best_comparator": r["best_comparator"],
                                 "best_comparator_M": r["best_comparator_M"], "difference": r["difference"],
                                 "difference_ci95_low": r["difference_ci95"][0], "difference_ci95_high": r["difference_ci95"][1],
                                 "relative_gain": r["relative_gain"], "meets_rule": r["pass"],
                                 **{f"complete_M_{x}": float(cref[f"complete_{x}"].mean())
                                    if f"complete_{x}" in cref and cref[f"complete_{x}"].notna().all() else np.nan
                                    for x in ("embedding", "latent", "raw", "clocks")},
                                 "decision_set": decision, "masking": LABEL})
    pd.DataFrame(rows).to_csv(OUTG / f"summary_heldout_{DATE}.csv", index=False)
    p = res["primary"]
    pd.DataFrame([{"analysis": f"G: sparse sample-specific inputs ({LABEL}), keep fraction f = {lock['f']} (chosen on dev-val), "
                               f"50 labelled samples, A's primary datasets",
                   "primary_metric": "mean over datasets of (AUC - 0.5), n = 50", "datasets": p["datasets"],
                   "methylgpt_representation": lock["representation"], "methylgpt_value": p["methylgpt_M"],
                   "best_comparator": p["best_comparator"], "best_comparator_value": p["best_comparator_M"],
                   "relative_gain": p["relative_gain"], "difference": p["difference"],
                   "difference_ci95_low": p["difference_ci95"][0], "difference_ci95_high": p["difference_ci95"][1],
                   "rule": "relative gain >= 10% and interval of the difference excludes zero (lower end > 0)",
                   "outcome": "PASS" if p["pass"] else "FAIL", "masking": LABEL, "mask_seed": SEED_G}]).to_csv(
        OUTG / f"rule_outcome_{DATE}.csv", index=False)
    write_json(OUTG / f"results_{DATE}.json", res)
    log(f"PRIMARY: f = {lock['f']} {lock['representation']}: {p['methylgpt_M']:.4f} vs {p['best_comparator']} "
        f"{p['best_comparator_M']:.4f}, gain {p['relative_gain']:.3f}, CI {p['difference_ci95']}, {'PASS' if p['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["freeze", "represent", "clocks", "fit", "lock", "analyse"])
    ap.add_argument("--set", default="dev", choices=["dev", "heldout"])
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    if a.stage == "freeze":
        stage_freeze()
    elif a.stage == "represent":
        stage_represent(a.set, a.device)
    elif a.stage == "clocks":
        stage_clocks(a.set)
    elif a.stage == "fit":
        stage_fit(a.set, a.device)
    elif a.stage == "lock":
        stage_lock()
    else:
        stage_analyse()


if __name__ == "__main__":
    main()
