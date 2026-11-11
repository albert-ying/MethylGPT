#!/usr/bin/env python3
"""Age-stable CpGs with high MethylGPT age attribution (Fig. 5), a masking test of them, and two Fig. 3 options.

Written on 10 Nov 2026 before any computation below was run; the definitions of part B are those of the
analysis specification and are fixed here and in
the README before computing. Every result is reported. Aggregate outputs only, except the UMAP coordinates of public GEO
profiles already exported for Fig. 3 (parts D and E). No Generation Scotland file is read; no model is trained.

PROFILES (parts A to C): the 5,811 test-split profiles with metadata ages (34 GEO datasets) of Fig. 5. Young: the 1,000
profiles younger than 30 years drawn by si_followup_2026-10-22 (rows|young of si_attribution_old_young_2026-10-23/
group_means.npz); old: all 724 profiles aged 60 or older (rows|old). Tissue groups: the tissue labels with >= 100 of the
5,811 profiles (Whole Blood, Brain, Muscle, Other, Colon, Cells, Adipose, Liver), the rest pooled (Lung, Pancreas, Skin;
108 profiles). Beta values from the corpus, clipped to [0, 1] as the released package does.
ATTRIBUTION: the saved per-profile absolute gradient x input of the released age output, averaged over the three
released runs (si_attribution_old_young_2026-10-23/per_profile_abs_attribution.npy, 5,811 x 49,156; NaN where the CpG is
not observed). Check: its group means equal group_means.npz (young, old, < 20, 20 to < 60, >= 60) within 1e-6 relative.

PART A (per-CpG table, 49,156 rows; every statistic over the profiles in which the CpG is observed)
  n_observed; mean and SD (ddof 1) of beta; mean absolute attribution over all 5,811 profiles; Spearman rho of beta with
  age over all profiles, within Whole Blood and within Brain (average ranks; NaN with fewer than 10 observed profiles or a
  constant value); tissue-adjusted rho: Spearman partial correlation of beta and age given the tissue-group indicators
  (ranks over the observed profiles, each rank vector centred within tissue group, Pearson correlation of the
  residuals); delta beta = mean beta old minus young, with a two-sided Mann-Whitney P (scipy asymptotic, continuity
  correction; CpGs observed in >= 10 young and >= 10 old profiles); eta squared = between-tissue-group sum of squares /
  total sum of squares of beta; flags: the four clocks of Fig. 5c (Horvath multi-tissue, Horvath skin and blood,
  Hannum, PhenoAge; table5 of the 2026-10-23 export) and the annotations of Supplementary Note 10 (CpG-island relation,
  enhancer, DNase hypersensitivity: cpg_annotation_2026-10-01.parquet; PRC2 target: RefGene gene in
  BENPORATH_PRC2_TARGETS; TFBS: any ENCODE TFBS_NAME in the EPIC v1.0 B5 manifest; all defined for EPIC v1 CpGs).

PART B (pre-specified sets)
  high attribution        the 1,000 CpGs with the largest mean absolute attribution (over the 49,156)
  age-stable              |rho all| < 0.1 and |delta beta| < 0.02 (both defined)
  age-stable, sensitivity |tissue-adjusted rho| < 0.1 and |delta beta| < 0.02
  age-correlated reference the 1,000 CpGs with the largest |rho all|
  Reported: high-attribution CpGs that are age-stable, the number expected (1,000 x the age-stable fraction of CpGs with
  both statistics defined), Fisher's exact test (two-sided; high attribution against the rest, age-stable against not;
  CpGs with both statistics defined); the same for the sensitivity definition; overlap of high attribution with the
  age-correlated reference (the earlier 148 came from other rankings: 2,000-profile attribution and within-dataset
  training-profile correlation, EPIC v1 CpGs; check 3 reproduces it from those arrays).
  For the high-attribution age-stable set against all other vocabulary CpGs: odds ratio (Woolf 95% interval, 0.5 added
  to every cell when one is 0, as Supplementary Note 10) and Fisher P for each annotation (background: EPIC v1
  vocabulary CpGs, where the annotations are defined), clock CpGs and the top decile of eta squared (background: all
  vocabulary CpGs with the statistic defined); gometh (missMethyl, GO and KEGG, array.type EPIC, prior.prob TRUE, all
  vocabulary CpGs as all.cpg; gometh keeps the EPIC-annotated ones), FDR as reported by gometh.
  Distributions (n, mean, 10th, 25th, 50th, 75th, 90th percentiles) of eta squared and SD of beta in four groups: high
  attribution age-stable; high attribution age-correlated (|rho all| >= 0.3); age-correlated reference; all CpGs.
  Old-versus-young attribution (table6 log2 ratio, CpGs tested there) of the high-attribution age-stable CpGs against
  the other high-attribution CpGs: the same summaries and a two-sided Mann-Whitney P.

PART C (masking test; released bundle, age = mean of the three runs; masked CpGs set to missing, the model's own
  missing-value input; the 5,811 profiles)
  (1) the high-attribution age-stable CpGs (k CpGs); (2) k CpGs drawn without replacement from the high-attribution CpGs
  with |rho all| >= 0.3, 10 draws; (3) k CpGs matched to set (1) one for one on mean-beta decile and observation-rate
  decile (deciles over the vocabulary; tied rate edges merged), drawn without replacement from CpGs outside set (1),
  20 draws; (4) the age-correlated reference set; (5) nothing (baseline). Draws: numpy default_rng(20261110), set (2)
  first. If the pool of (2) holds fewer than k CpGs, every draw takes the whole pool (reported).
  MedAE and mean signed error (predicted minus metadata age), overall and per tissue group with >= 100 profiles; change
  from baseline with a 95% interval from 2,000 draws resampling the 34 datasets (seed 20261110; predictions fixed; paired,
  both statistics recomputed on each resampled set). Drawn sets: every draw reported, and the mean and range over draws.
  Check: baseline predictions equal the saved held-out ensemble predictions (age_ensemble_2026-10-04) within 1e-3 years.

PART D (Fig. 3 option 1) Set D (fig3_setD_umap_2026-11-10, same profiles and order): UMAP with umap-learn defaults as in
  the submitted figure's code (15 neighbours, min_dist 0.1, Euclidean, random_state 42) for raw beta values and the
  encoder output, (i) on the 50 principal components of fig3 (work file of export_figure_tables_2026-11-10) and (ii) on
  the full features (49,156 mean-filled CpGs; 512 encoder dimensions).

PART E (Fig. 3 option 2, cross-array consistency) Set C profiles with >= 90% of the EPIC v1-mask vocabulary CpGs
  observed; each profile twice, complete and reduced to the 27K-mask CpGs (others set missing). Raw beta values
  (training-mean filled), latent scores and the encoder output (dimensions 1,537 to 2,048 of the released run-1
  embedding), each computed from the copy itself. Per representation: centred exact PCA of the complete and reduced
  copies together, 50 components, one UMAP of both copies (30 neighbours, min_dist 0.3, seed 0). Metrics on the 50
  components, cosine: fraction of reduced copies whose nearest complete profile is their own complete copy; median
  cosine distance between the two copies divided by the median distance of each complete profile to its nearest other
  complete profile; 95% intervals from 2,000 draws resampling datasets (seed 20261110; per-profile values fixed).

AMENDMENT 1 (10 Nov 2026, after part C's predictions; no prediction is redone): some resampled sets contain no profile
  of a tissue group, which made every per-tissue interval NaN. A tissue group's interval now uses the draws that contain
  at least one of its profiles (the rule of the earlier intervals, si_sample_representations ci()); the number of draws
  used is in column bootstrap_draws_with_profiles. Summaries are recomputed from the saved predictions (--stage c_summary).

CHECKS (abort on failure): 1 profile counts (5,811, 34 datasets; young 1,000, old 724) and GEO-only exported rows;
  2 attribution group means; 3 the earlier overlap of 148; 4 baseline predictions; 5 part D profiles and order equal the
  set D table; 6 part E complete-copy latent and encoder from the cache equal a recomputation for 64 profiles (max abs
  diff <= 1e-3); 7 per-CpG n_observed equals the observed count of the beta matrix.

STAGES  --stage a | b | c | d | e | write | all (c and e use the GPU; run with CUDA_VISIBLE_DEVICES set to one GPU).
OUTPUTS artifacts/attribution_stability_2026-11-10/ (parts A to C) and artifacts/figure_tables_2026-11-10/ (parts D and E,
  README_options_2026-11-10.md); work files in artifacts/attribution_stability_2026-11-10_work/ (not exported).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt2_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json  # noqa: E402

DATE = "2026-11-10"
OUT = ARTIFACTS / f"attribution_stability_{DATE}"
WORK = ARTIFACTS / f"attribution_stability_{DATE}_work"
FT = ARTIFACTS / f"figure_tables_{DATE}"
FTW = ARTIFACTS / f"figure_tables_{DATE}_work"
EVAL = ARTIFACTS / "eval"
ATT = ARTIFACTS / "si_attribution_old_young_2026-10-23"
BUNDLE = ARTIFACTS / "release/methylgpt2-1.1"
SEED = 20261110
NBOOT = 2000
RANK = 1536
CHECKS = []


def chk(name, ok, detail=""):
    CHECKS.append({"check": name, "passed": bool(ok), "detail": str(detail)})
    log(f"check {name}: {'pass' if ok else 'FAIL'} {detail}")


def save_checks(stage):
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / f"checks_{stage}.json", CHECKS)
    bad = [c for c in CHECKS if not c["passed"]]
    if bad:
        sys.exit(f"{stage}: {len(bad)} check(s) failed: {[c['check'] for c in bad]}")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FTX = None


def ftx():
    global FTX
    if FTX is None:
        FTX = _load("export_figure_tables", "export_figure_tables_2026-11-10.py")
    return FTX


def heldout():
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    assert (lab.id.to_numpy() == prof.id.to_numpy()).all()
    ho = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(lab.age_meta.to_numpy(float)))
    age = lab.age_meta.to_numpy(float)[ho]
    tis = prof.tissue.fillna("no tissue label").to_numpy(str)[ho]
    vc = pd.Series(tis).value_counts()
    big = sorted(vc[vc >= 100].index)
    grp = np.where(np.isin(tis, big), tis, "pooled (< 100 profiles)")
    ds = prof.group.astype(str).to_numpy()[ho]
    gm = np.load(require(ATT / "group_means.npz"))
    young, old = np.searchsorted(ho, np.sort(gm["rows|young"])), np.searchsorted(ho, np.sort(gm["rows|old"]))
    assert (ho[young] == np.sort(gm["rows|young"])).all() and (ho[old] == np.sort(gm["rows|old"])).all()
    return prof, lab, ho, age, tis, grp, ds, young, old, big


def corpus_beta(rows):
    from methylgpt2_lib import Corpus
    X = Corpus().get(rows)
    np.clip(X, 0, 1, out=X)
    return X


# ----------------------------------------------------------------------------------------------------------------------
# part A
# ----------------------------------------------------------------------------------------------------------------------
def masked_pearson(x, y):
    """Column-wise Pearson correlation of x and y (n x c, NaN where unobserved; same pattern)."""
    m = np.isfinite(x) & np.isfinite(y)
    n = m.sum(0)
    x0, y0 = np.where(m, x, 0.0), np.where(m, y, 0.0)
    mx, my = x0.sum(0) / np.maximum(n, 1), y0.sum(0) / np.maximum(n, 1)
    dx, dy = np.where(m, x - mx, 0.0), np.where(m, y - my, 0.0)
    den = np.sqrt((dx * dx).sum(0) * (dy * dy).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dx * dy).sum(0) / den
    return np.where((n >= 10) & (den > 0), r, np.nan)


def ranks(B, a):
    from scipy.stats import rankdata
    A = np.where(np.isfinite(B), a[:, None], np.nan)
    return rankdata(B, axis=0, nan_policy="omit"), rankdata(A, axis=0, nan_policy="omit")


def centre_within(R, G):
    """Centre each column of R (NaN where unobserved) within the groups of the one-hot matrix G (n x K)."""
    M = np.isfinite(R)
    S = G.T @ np.where(M, R, 0.0)
    C = G.T @ M.astype(float)
    mu = S / np.maximum(C, 1)
    return np.where(M, R - G @ mu, np.nan)


def write_definitions():
    """README section with the definitions, written before any statistic is computed (specification)."""
    OUT.mkdir(parents=True, exist_ok=True)
    doc = __doc__.split("CHECKS (abort on failure)")[0]
    (OUT / "README.md").write_text("# Age-stable CpGs with high MethylGPT age attribution (Fig. 5) and Fig. 3 options\n\n"
                                   f"Definitions written to this README at {time.strftime('%Y-%m-%d %H:%M')} before part A computed any statistic; "
                                   f"the same text is the docstring of `attribution_stability_fig3_options_{DATE}.py`, committed before the run.\n\n```text\n" + doc + "```\n")


def stage_a():
    write_definitions()
    t0 = time.time()
    prof, lab, ho, age, tis, grp, ds, young, old, big = heldout()
    chk("1 profiles", len(ho) == 5811 and len(set(ds)) == 34 and len(young) == 1000 and len(old) == 724,
        f"{len(ho)} profiles, {len(set(ds))} datasets, young {len(young)}, old {len(old)}; tissue groups {big} + pooled {(grp == 'pooled (< 100 profiles)').sum()}")
    X = corpus_beta(ho)
    O = np.isfinite(X)
    # attribution
    att = np.load(require(ATT / "per_profile_abs_attribution.npy"), mmap_mode="r")
    prow = np.load(require(ATT / "per_profile_rows.npy"))
    assert sorted(prow.tolist()) == ho.tolist()
    att = np.asarray(att[np.argsort(prow)], np.float64)       # rows in corpus order (= ho)
    chk("2 attribution observed pattern equals beta observed pattern", (np.isfinite(att) == O).all())
    gm = np.load(require(ATT / "group_means.npz"))
    for g in ("young", "old", "lt20", "20to60", "ge60"):
        idx = np.searchsorted(ho, np.sort(gm[f"rows|{g}"]))
        with np.errstate(invalid="ignore"):
            mg = np.nanmean(att[idx], 0)
        ref = gm[f"mean|{g}"]
        ok = np.isfinite(ref) & np.isfinite(mg)
        rel = float(np.max(np.abs(mg[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-30)))
        chk(f"2 attribution group mean equals group_means ({g})", rel <= 1e-6 and (np.isfinite(mg) == np.isfinite(ref)).all(), f"max rel diff {rel:.2e}")
    n_obs = O.sum(0)
    with np.errstate(invalid="ignore"):
        mean_attr = np.nanmean(att, 0)
        mb, sdb = np.nanmean(X, 0), np.nanstd(X, 0, ddof=1)
    del att
    chk("7 n_observed equals observed count", (n_obs == np.isfinite(X).sum(0)).all())
    # Spearman rho (all, whole blood, brain) and tissue-adjusted rho, eta squared, in column chunks
    groups = sorted(set(grp))
    G = (grp[:, None] == np.array(groups)[None, :]).astype(float)
    wb, br = tis == "Whole Blood", tis == "Brain"
    P = X.shape[1]
    rho, rho_wb, rho_br, rho_adj, eta = (np.full(P, np.nan) for _ in range(5))
    for j in range(0, P, 2048):
        B = X[:, j:j + 2048].astype(np.float64)
        rb, ra = ranks(B, age)
        rho[j:j + 2048] = masked_pearson(rb, ra)
        rho_adj[j:j + 2048] = masked_pearson(centre_within(rb, G), centre_within(ra, G))
        for m, dst in ((wb, rho_wb), (br, rho_br)):
            rb2, ra2 = ranks(B[m], age[m])
            dst[j:j + 2048] = masked_pearson(rb2, ra2)
        M = np.isfinite(B)
        tot = np.where(M, B - np.where(M, B, 0).sum(0) / np.maximum(M.sum(0), 1), 0.0)
        sst = (tot ** 2).sum(0)
        resid = np.where(M, centre_within(B, G), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            eta[j:j + 2048] = np.where((sst > 0) & (M.sum(0) >= 10), 1 - (resid ** 2).sum(0) / sst, np.nan)
        if j % 10240 == 0:
            log(f"part A columns {j}/{P} ({time.time() - t0:.0f} s)")
    # delta beta and Mann-Whitney, old against young
    from scipy.stats import mannwhitneyu
    with np.errstate(invalid="ignore"):
        dbeta = np.nanmean(X[old], 0) - np.nanmean(X[young], 0)
    ny, no = O[young].sum(0), O[old].sum(0)
    pmw = np.full(P, np.nan)
    Xy, Xo = X[young], X[old]
    for j in np.flatnonzero((ny >= 10) & (no >= 10)):
        a, b = Xo[:, j], Xy[:, j]
        pmw[j] = mannwhitneyu(a[np.isfinite(a)], b[np.isfinite(b)], alternative="two-sided", method="asymptotic").pvalue
    dbeta = np.where((ny >= 10) & (no >= 10), dbeta, np.nan)
    # annotations and clocks
    se = _load("si_enrichment", "si_enrichment_attribution_2026-10-21.py")
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    vocab = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))["vocab"].astype(str)
    assert (ann.cpg.to_numpy() == vocab).all()
    prc2 = set(l.strip() for l in require(se.PRC2).read_text().split("\n") if l.strip())
    genes = ann.genes.fillna("").astype(str).str.split(";")
    t5 = pd.read_csv(require(ARTIFACTS / "figure_tables_2026-10-23/table5_fig5bd_cpg_attribution.csv.gz"),
                     usecols=["cpg_id", "in_horvath_multitissue", "in_horvath_skin_blood", "in_hannum", "in_phenoage", "in_any_of_four_clocks"])
    assert (t5.cpg_id.to_numpy() == vocab).all()
    t6 = pd.read_csv(require(ARTIFACTS / "figure_tables_2026-10-23/table6_fig5c_old_young_attribution.csv.gz"), usecols=["cpg_id", "log2_ratio_old_young", "tested"])
    assert (t6.cpg_id.to_numpy() == vocab).all()
    epic = ann.in_epic_v1.to_numpy(bool)
    A = pd.DataFrame({"cpg_id": vocab, "n_observed": n_obs, "mean_beta": mb, "sd_beta": sdb, "mean_abs_attribution": mean_attr,
                      "rho_age_all": rho, "rho_age_whole_blood": rho_wb, "n_observed_whole_blood": O[wb].sum(0),
                      "rho_age_brain": rho_br, "n_observed_brain": O[br].sum(0), "rho_age_tissue_adjusted": rho_adj,
                      "delta_beta_old_minus_young": dbeta, "p_mannwhitney_old_young": pmw, "n_young_observed": ny, "n_old_observed": no,
                      "eta_squared_tissue": eta, "in_epic_v1": epic,
                      "island_relation": np.where(epic, ann.island_relation.fillna("").astype(str), ""),
                      "enhancer": np.where(epic, ann.enhancer.fillna(False).astype(bool), False),
                      "dnase_hs": np.where(epic, ann.dnase_hs.fillna(False).astype(bool), False),
                      "prc2_target": genes.map(lambda gs: any(g in prc2 for g in gs)).to_numpy() & epic,
                      "tfbs": se.manifest_tfbs(vocab) & epic,
                      "in_horvath_multitissue": t5.in_horvath_multitissue.to_numpy(bool), "in_horvath_skin_blood": t5.in_horvath_skin_blood.to_numpy(bool),
                      "in_hannum": t5.in_hannum.to_numpy(bool), "in_phenoage": t5.in_phenoage.to_numpy(bool),
                      "in_any_of_four_clocks": t5.in_any_of_four_clocks.to_numpy(bool),
                      "log2_ratio_attribution_old_young": np.where(t6.tested.to_numpy(bool), t6.log2_ratio_old_young.to_numpy(float), np.nan)})
    WORK.mkdir(parents=True, exist_ok=True)
    A.to_parquet(WORK / "part_a.parquet")
    log(f"part A done ({time.time() - t0:.0f} s)")
    save_checks("a")


# ----------------------------------------------------------------------------------------------------------------------
# part B
# ----------------------------------------------------------------------------------------------------------------------
def define_sets(A):
    o = np.argsort(-np.where(np.isfinite(A.mean_abs_attribution), A.mean_abs_attribution, -np.inf), kind="stable")
    high = np.zeros(len(A), bool); high[o[:1000]] = True
    ar = np.abs(A.rho_age_all.to_numpy())
    o2 = np.argsort(-np.where(np.isfinite(ar), ar, -np.inf), kind="stable")
    ref = np.zeros(len(A), bool); ref[o2[:1000]] = True
    db = np.abs(A.delta_beta_old_minus_young.to_numpy())
    defined = np.isfinite(ar) & np.isfinite(db)
    stable = defined & (ar < 0.1) & (db < 0.02)
    adj = np.abs(A.rho_age_tissue_adjusted.to_numpy())
    defined_s = np.isfinite(adj) & np.isfinite(db)
    stable_s = defined_s & (adj < 0.1) & (db < 0.02)
    return high, ref, stable, defined, stable_s, defined_s


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [10, 25, 50, 75, 90]) if len(v) else [np.nan] * 5
    return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else np.nan, **{f"p{p}": float(x) for p, x in zip((10, 25, 50, 75, 90), q)}}


def stage_b():
    from scipy.stats import fisher_exact, mannwhitneyu
    se = _load("si_enrichment", "si_enrichment_attribution_2026-10-21.py")
    A = pd.read_parquet(require(WORK / "part_a.parquet"))
    high, ref, stable, defined, stable_s, defined_s = define_sets(A)
    res = {}
    for name, st, dfn in (("age-stable", stable, defined), ("age-stable, tissue-adjusted (sensitivity)", stable_s, defined_s)):
        a = int((high & st & dfn).sum()); b = int((high & ~st & dfn).sum()); c = int((~high & st & dfn).sum()); d = int((~high & ~st & dfn).sum())
        f = se.fisher(high[dfn], st[dfn])
        res[name] = {"high_attribution_age_stable": a, "high_attribution_with_statistics": a + b, "age_stable_fraction": (a + c) / (a + b + c + d),
                     "expected": 1000 * (a + c) / (a + b + c + d), "cpgs_with_statistics": a + b + c + d, "odds_ratio": f["odds_ratio"],
                     "ci95": f["ci95"], "fisher_p": float(fisher_exact([[a, b], [c, d]])[1])}
        log(f"{name}: {a} of {a + b} high-attribution CpGs; expected {res[name]['expected']:.1f}; OR {f['odds_ratio']:.3f} P {res[name]['fisher_p']:.3g}")
    res["overlap_high_attribution_age_correlated_reference"] = int((high & ref).sum())
    # check 3: the earlier 148 from the 2026-10-21 arrays
    Az = np.load(require(ARTIFACTS / "si_enrichment_2026-10-21/attribution.npz")); dz = np.load(require(ARTIFACTS / "si_enrichment_2026-10-21/direction.npz"))
    runs = ("run 1", "seed 2", "seed 3")
    score = np.mean([Az[f"score|{r}"] for r in runs], 0)
    bg = A.in_epic_v1.to_numpy() & (np.mean([Az[f"n_observed|{r}"] for r in runs], 0) > 0)
    t1 = np.argsort(-np.where(bg & np.isfinite(score), score, -np.inf))[:1000]
    t2 = np.argsort(-np.where(bg & np.isfinite(dz["r"]), np.abs(dz["r"]), -np.inf))[:1000]
    ov = len(set(t1) & set(t2))
    chk("3 earlier overlap reproduced", ov == 148, f"{ov}")
    res["overlap_earlier_rankings_check"] = ov
    S1 = high & stable
    res["high_attribution_age_stable_k"] = int(S1.sum())
    # annotations, clocks, eta squared top decile
    epic = A.in_epic_v1.to_numpy(bool)
    eta = A.eta_squared_tissue.to_numpy()
    top_eta = np.isfinite(eta) & (eta >= np.nanpercentile(eta, 90))
    rows = []
    sets = {"CpG island": (A.island_relation == "Island").to_numpy(), "CpG shore": (A.island_relation == "Shore").to_numpy(),
            "CpG shelf": (A.island_relation == "Shelf").to_numpy(), "open sea": (A.island_relation == "OpenSea").to_numpy(),
            "enhancer": A.enhancer.to_numpy(bool), "DNase hypersensitivity": A.dnase_hs.to_numpy(bool), "PRC2 target": A.prc2_target.to_numpy(bool),
            "TFBS": A.tfbs.to_numpy(bool)}
    for k, s in sets.items():
        rows.append({"annotation": k, "background_definition": "EPIC v1 vocabulary CpGs", **se.fisher(S1[epic], s[epic])})
    for k, s, bgm in (("clock CpG (any of the four)", A.in_any_of_four_clocks.to_numpy(bool), np.ones(len(A), bool)),
                      ("top decile of eta squared (tissue-discriminating)", top_eta, np.isfinite(eta))):
        rows.append({"annotation": k, "background_definition": "all vocabulary CpGs" + (" with eta squared defined" if "eta" in k else ""), **se.fisher(S1[bgm], s[bgm])})
    enr = pd.DataFrame(rows)
    enr["ci95_low"], enr["ci95_high"] = enr.ci95.map(lambda c: c[0]), enr.ci95.map(lambda c: c[1])
    enr = enr.drop(columns="ci95").rename(columns={"top_in_set": "set_annotated", "top": "set_size", "background_in_set": "rest_annotated",
                                                   "background": "rest_size", "fraction_top": "fraction_set", "fraction_background": "fraction_rest", "p": "fisher_p"})
    enr.to_csv(WORK / "part_b_enrichment.csv", index=False)
    # gometh
    vocab = A.cpg_id.to_numpy()
    gd = WORK / "gometh"
    gd.mkdir(parents=True, exist_ok=True)
    pd.Series(vocab[S1]).to_csv(gd / "sig.txt", index=False, header=False)
    pd.Series(vocab).to_csv(gd / "all.txt", index=False, header=False)
    (gd / "run.R").write_text(f"""suppressMessages(library(missMethyl))
sig <- readLines("{gd}/sig.txt"); allc <- readLines("{gd}/all.txt")
for (col in c("GO", "KEGG")) {{
  r <- gometh(sig.cpg = sig, all.cpg = allc, collection = col, array.type = "EPIC", prior.prob = TRUE)
  r <- r[order(r$P.DE), ]
  write.csv(r, file.path("{gd}", paste0(col, ".csv")))
}}
""")
    p = subprocess.run(["Rscript", str(gd / "run.R")], capture_output=True, text=True)
    res["gometh"] = {"returncode": p.returncode, "stderr_tail": p.stderr[-1500:]}
    if p.returncode == 0:
        for col in ("GO", "KEGG"):
            r = pd.read_csv(gd / f"{col}.csv", index_col=0)
            res["gometh"][col] = {"terms_tested": int(len(r)), "terms_fdr_0.05": int((r.FDR < 0.05).sum()), "min_fdr": float(r.FDR.min()),
                                  "top10": [{"id": str(i), "term": str(row.get("TERM", row.get("Description", ""))), "n": int(row["N"]), "de": int(row["DE"]),
                                             "p": float(row["P.DE"]), "fdr": float(row["FDR"])} for i, row in r.head(10).iterrows()]}
            log(f"gometh {col}: {res['gometh'][col]['terms_fdr_0.05']} of {len(r)} terms at FDR < 0.05")
    else:
        log(f"gometh failed: {p.stderr[-800:]}")
    # distributions
    ar = np.abs(A.rho_age_all.to_numpy())
    groups = {"high attribution, age-stable": S1, "high attribution, age-correlated (|rho| >= 0.3)": high & (ar >= 0.3),
              "age-correlated reference": ref, "all CpGs": np.ones(len(A), bool)}
    dist = []
    for g, m in groups.items():
        for var in ("eta_squared_tissue", "sd_beta"):
            dist.append({"group": g, "statistic": var, **summ(A[var].to_numpy()[m])})
    lr = A.log2_ratio_attribution_old_young.to_numpy()
    for g, m in (("high attribution, age-stable", S1), ("other high-attribution CpGs", high & ~S1)):
        dist.append({"group": g, "statistic": "log2_ratio_attribution_old_young", **summ(lr[m])})
    a_, b_ = lr[S1][np.isfinite(lr[S1])], lr[high & ~S1][np.isfinite(lr[high & ~S1])]
    res["log2_ratio_mannwhitney_p"] = float(mannwhitneyu(a_, b_, alternative="two-sided").pvalue) if len(a_) and len(b_) else None
    pd.DataFrame(dist).to_csv(WORK / "part_b_distributions.csv", index=False)
    write_json(WORK / "part_b.json", res)
    save_checks("b")


# ----------------------------------------------------------------------------------------------------------------------
# part C
# ----------------------------------------------------------------------------------------------------------------------
def boot_index(ds, W):
    _, codes = np.unique(ds, return_inverse=True)
    members = [np.flatnonzero(codes == g) for g in range(W.shape[1])]
    return [np.concatenate([np.tile(members[g], int(w)) for g, w in enumerate(wb) if w > 0]) for wb in W]


def stage_c():
    import torch
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    prof, lab, ho, age, tis, grp, ds, young, old, big = heldout()
    A = pd.read_parquet(require(WORK / "part_a.parquet"))
    high, ref, stable, defined, stable_s, defined_s = define_sets(A)
    S1 = np.flatnonzero(high & stable)
    k = len(S1)
    ar = np.abs(A.rho_age_all.to_numpy())
    pool2 = np.flatnonzero(high & (ar >= 0.3))
    rng = np.random.default_rng(SEED)
    if k == 0:
        sys.exit("part C: no high-attribution age-stable CpG; sets (1) to (3) are empty (report this instead of masking)")
    sets = {"(1) high attribution, age-stable": [S1]}
    sets["(2) high attribution, age-correlated (|rho| >= 0.3)"] = [np.sort(rng.choice(pool2, min(k, len(pool2)), replace=False)) for _ in range(10)]
    mb = A.mean_beta.to_numpy()
    rate = A.n_observed.to_numpy() / len(ho)
    bdec = pd.qcut(mb, 10, labels=False)
    rdec = pd.qcut(rate, 10, labels=False, duplicates="drop")
    stratum = bdec * 100 + rdec
    outside = np.setdiff1d(np.arange(len(A)), S1)
    draws3 = []
    for _ in range(20):
        pick, used = [], set()
        for j in S1:
            cand = [c for c in outside[stratum[outside] == stratum[j]] if c not in used]
            c = int(rng.choice(cand))
            used.add(c); pick.append(c)
        draws3.append(np.sort(pick))
    sets["(3) random, matched on mean beta and observation rate"] = draws3
    sets["(4) age-correlated reference"] = [np.flatnonzero(ref)]
    sets["(5) baseline (nothing masked)"] = [np.array([], int)]
    info = {"k": k, "pool2": int(len(pool2)), "pool2_smaller_than_k": bool(len(pool2) < k), "strata": int(len(np.unique(stratum)))}
    log(f"part C: k {k}, pool (2) {len(pool2)}")
    X = corpus_beta(ho)
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device="cuda:0")
    preds = {}
    for name, draws in sets.items():
        for i, cols in enumerate(draws):
            Xm = X.copy()
            Xm[:, cols] = np.nan
            p = model.predict_age_runs(pd.DataFrame(Xm, columns=model.vocab), batch_size=32).mean(1).to_numpy(float)
            preds[(name, i)] = p
            log(f"{name} draw {i}: masked {len(cols)} CpGs, MedAE {np.median(np.abs(p - age)):.3f} ({time.time() - t0:.0f} s)")
    ze = np.load(require(EVAL / "age_ensemble_2026-10-04.predictions.npz"))
    assert (ze["heldout_ids"].astype(str) == prof.id.to_numpy()[ho]).all()
    dmax = float(np.abs(preds[("(5) baseline (nothing masked)", 0)] - ze["ensemble|heldout"]).max())
    chk("4 baseline predictions equal saved held-out ensemble", dmax <= 1e-3, f"max abs diff {dmax:.3g} years")
    np.savez(WORK / "part_c_predictions.npz", age=age, **{f"{n}|{i}": v for (n, i), v in preds.items()},
             **{f"cols|{n}|{i}": c for n, d in sets.items() for i, c in enumerate(d)})
    write_json(WORK / "part_c.json", info)
    save_checks("c")
    summarise_c()


def summarise_c():
    """Part C summaries from the saved predictions (amendment 1: a tissue group's interval uses the draws that contain it)."""
    prof, lab, ho, age, tis, grp, ds, young, old, big = heldout()
    z = np.load(require(WORK / "part_c_predictions.npz"))
    assert np.array_equal(z["age"], age)
    preds, sets = {}, {}
    for key in z.files:
        if key == "age" or key.startswith("cols|"):
            continue
        name, i = key.rsplit("|", 1)
        preds[(name, int(i))] = z[key]
        sets.setdefault(name, {})[int(i)] = z[f"cols|{key}"]
    order = ["(1)", "(2)", "(3)", "(4)", "(5)"]
    preds = dict(sorted(preds.items(), key=lambda kv: (order.index(kv[0][0][:3]), kv[0][1])))
    sets = {n: [d[i] for i in sorted(d)] for n, d in sets.items()}
    sr = _load("si_sample_representations", "si_sample_representations_2026-10-18.py")
    _, W = sr.boot_weights(ds, n=NBOOT, seed=SEED)
    BI = boot_index(ds, W)
    base = preds[("(5) baseline (nothing masked)", 0)] - age
    subsets = {"all": np.ones(len(ho), bool), **{t: tis == t for t in big}}
    rows = []
    for (name, i), p in preds.items():
        e = p - age
        for sub, m in subsets.items():
            dm, ds_ = [], []
            for idx in BI:
                ii = idx[m[idx]]
                if len(ii):
                    dm.append(np.median(np.abs(e[ii])) - np.median(np.abs(base[ii])))
                    ds_.append(np.mean(e[ii]) - np.mean(base[ii]))
            rows.append({"set": name, "draw": i, "profiles": sub, "n_profiles": int(m.sum()), "n_datasets": int(len(set(ds[m]))),
                         "n_cpgs_masked": int(len(sets[name][i])), "bootstrap_draws_with_profiles": len(dm), "medae": float(np.median(np.abs(e[m]))), "mean_signed_error": float(np.mean(e[m])),
                         "delta_medae": float(np.median(np.abs(e[m])) - np.median(np.abs(base[m]))),
                         "delta_medae_ci95_low": float(np.percentile(dm, 2.5)), "delta_medae_ci95_high": float(np.percentile(dm, 97.5)),
                         "delta_mean_signed_error": float(np.mean(e[m]) - np.mean(base[m])),
                         "delta_mean_signed_error_ci95_low": float(np.percentile(ds_, 2.5)), "delta_mean_signed_error_ci95_high": float(np.percentile(ds_, 97.5))})
    C = pd.DataFrame(rows)
    C.to_csv(WORK / "part_c_by_draw.csv", index=False)
    agg = C.groupby(["set", "profiles"], sort=False).agg(n_draws=("draw", "size"), n_cpgs_masked=("n_cpgs_masked", "first"),
                                                          n_profiles=("n_profiles", "first"), medae_mean=("medae", "mean"),
                                                          delta_medae_mean=("delta_medae", "mean"), delta_medae_min=("delta_medae", "min"),
                                                          delta_medae_max=("delta_medae", "max"), delta_mse_mean=("delta_mean_signed_error", "mean"),
                                                          delta_mse_min=("delta_mean_signed_error", "min"), delta_mse_max=("delta_mean_signed_error", "max")).reset_index()
    agg.to_csv(WORK / "part_c_summary.csv", index=False)
    log(f"part C summaries written ({len(C)} rows)")


# ----------------------------------------------------------------------------------------------------------------------
# part D
# ----------------------------------------------------------------------------------------------------------------------
FEATS = {}


def _umap_key(key):
    import umap
    kw = dict(random_state=42) if key.startswith("D|") else dict(n_neighbors=30, min_dist=0.3, random_state=0)
    return key, umap.UMAP(**kw).fit_transform(np.ascontiguousarray(FEATS[key], dtype=np.float32)).astype(np.float32)


def run_umaps(keys, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return dict(ex.map(_umap_key, keys))


def stage_d():
    t0 = time.time()
    z = np.load(require(FTW / "extract.npz"), allow_pickle=True)
    D = z["rows_D"]
    t = pd.read_csv(require(FT / f"fig3_setD_umap_{DATE}.csv.gz"), usecols=["corpus_row", "profile_id", "dataset"])
    chk("5 part D profiles and order equal the set D table", (t.corpus_row.to_numpy() == D).all(), f"{len(D)} profiles")
    zb = np.load(require(FTW / "bundle_pass_D.npz"))
    assert (zb["rows"] == D).all()
    mu = np.load(require(BUNDLE / "core.npz"))["mu"].astype(np.float32)
    X = corpus_beta(D)
    X = np.where(np.isfinite(X), X, mu[None, :]).astype(np.float32)
    FEATS.update({"D|raw|pcs50": z["pcs_D_raw"][:, :50], "D|encoder|pcs50": z["pcs_D_encoder"][:, :50],
                  "D|encoder|full": zb["emb"][:, RANK:], "D|raw|full": X})
    res = run_umaps(list(FEATS), 4)
    out = t[["corpus_row", "profile_id", "dataset"]].copy()
    for key in ("D|raw|pcs50", "D|raw|full", "D|encoder|pcs50", "D|encoder|full"):
        _, rep, inp = key.split("|")
        out[f"umap1_{rep}_{inp}"], out[f"umap2_{rep}_{inp}"] = res[key][:, 0], res[key][:, 1]
    out.to_parquet(WORK / "part_d.parquet")
    log(f"part D done ({time.time() - t0:.0f} s)")
    save_checks("d")


# ----------------------------------------------------------------------------------------------------------------------
# part E
# ----------------------------------------------------------------------------------------------------------------------
def stage_e():
    import torch
    from methylgpt2_lib import cuda_preinit
    cuda_preinit()
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    fx = ftx()
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    z = np.load(require(FTW / "extract.npz"), allow_pickle=True)
    D, C = z["rows_D"], z["rows_C"]
    pm = np.load(require(ARTIFACTS / "platform_vocab_masks_2026-10-01.npz"))
    XC = corpus_beta(C)
    fe = np.isfinite(XC[:, pm["epic_v1"]]).mean(1)
    keep = fe >= 0.9
    rows = C[keep]
    X = XC[keep]
    del XC
    Xr = X.copy()
    Xr[:, ~pm["hm27"]] = np.nan
    n = len(rows)
    log(f"part E: {n} of {len(C)} set C profiles with >= 90% of EPIC v1 CpGs observed")
    zb = np.load(require(FTW / "bundle_pass_D.npz"))
    pos = np.searchsorted(D, rows)
    lat_c, enc_c = zb["lat"][pos], zb["emb"][pos][:, RANK:]
    sys.path.insert(0, str(HERE / "package/src"))
    from methylgpt2 import MethylGPT
    model = MethylGPT.from_pretrained(str(BUNDLE), device="cuda:0")

    def bundle(M):
        L, E = np.empty((len(M), RANK), np.float32), np.empty((len(M), 512), np.float32)
        with torch.no_grad():
            for i in range(0, len(M), 32):
                x = torch.from_numpy(M[i:i + 32]).cuda()
                _, s, e = model._encode(x)
                L[i:i + 32], E[i:i + 32] = s.float().cpu().numpy(), e[:, RANK:].float().cpu().numpy()
        return L, E
    l64, e64 = bundle(X[:64])
    d6 = max(float(np.abs(l64 - lat_c[:64]).max()), float(np.abs(e64 - enc_c[:64]).max()))
    chk("6 complete-copy latent and encoder equal the cache (64 profiles)", d6 <= 1e-3, f"max abs diff {d6:.3g}")
    lat_r, enc_r = bundle(Xr)
    mu = model.core.mu.cpu().numpy()
    del model
    torch.cuda.empty_cache()
    reps = {"raw": (np.where(np.isfinite(X), X, mu), np.where(np.isfinite(Xr), Xr, mu)), "latent": (lat_c, lat_r), "encoder": (enc_c, enc_r)}
    del X, Xr
    ds = prof.group.astype(str).to_numpy()[rows]
    sr = _load("si_sample_representations", "si_sample_representations_2026-10-18.py")
    _, W = sr.boot_weights(ds, n=NBOOT, seed=SEED)
    BI = boot_index(ds, W)
    met, pcs = [], {}
    for rep, (Mc, Mr) in reps.items():
        S, var, tot, _ = fx.pca(torch.from_numpy(np.ascontiguousarray(np.concatenate([Mc, Mr]), dtype=np.float32)).cuda())
        torch.cuda.empty_cache()
        pcs[rep] = S
        T = torch.as_tensor(S, dtype=torch.float64, device="cuda:0")
        T = T / T.norm(dim=1, keepdim=True)
        Tc, Tr = T[:n], T[n:]
        match = np.empty(n, bool); dcopy = np.empty(n); dnn = np.empty(n)
        for i in range(0, n, 2048):
            Sr = Tr[i:i + 2048] @ Tc.T
            match[i:i + 2048] = (Sr.argmax(1).cpu().numpy() == np.arange(i, min(i + 2048, n)))
            dcopy[i:i + 2048] = (1 - (Tr[i:i + 2048] * Tc[i:i + 2048]).sum(1)).clamp_min(0).cpu().numpy()
            Scc = Tc[i:i + 2048] @ Tc.T
            r = torch.arange(Scc.shape[0], device=Scc.device)
            Scc[r, r + i] = -np.inf
            dnn[i:i + 2048] = (1 - Scc.max(1).values).clamp_min(0).cpu().numpy()
        fr = [match[idx].mean() for idx in BI]
        ra = [np.median(dcopy[idx]) / np.median(dnn[idx]) for idx in BI]
        for metric, val, dr in (("fraction of reduced copies nearest to their own complete copy", match.mean(), fr),
                                ("median copy distance / median nearest-neighbour distance among complete profiles", np.median(dcopy) / np.median(dnn), ra)):
            met.append({"representation": rep, "metric": metric, "value": float(val), "ci95_low": float(np.percentile(dr, 2.5)),
                        "ci95_high": float(np.percentile(dr, 97.5)), "n_profiles": n, "n_datasets": int(len(set(ds))),
                        "median_copy_distance": float(np.median(dcopy)), "median_nn_distance_complete": float(np.median(dnn)),
                        "explained_variance_ratio_50": float(var.sum() / tot)})
            log(f"E {rep} {metric}: {val:.4f} [{np.percentile(dr, 2.5):.4f}, {np.percentile(dr, 97.5):.4f}]")
    import umap
    res = {f"E|{rep}": umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(np.asarray(pcs[rep][:, :50], np.float32)) for rep in reps}
    tisv = prof.tissue.fillna("no tissue label").to_numpy(str)[rows]
    arr = z["array"][np.searchsorted(D, rows)]
    out = pd.DataFrame({"corpus_row": np.concatenate([rows, rows]), "profile_id": np.concatenate([prof.id.to_numpy()[rows]] * 2),
                        "dataset": np.concatenate([ds, ds]), "tissue": np.concatenate([tisv, tisv]), "array": np.concatenate([arr, arr]),
                        "copy": np.repeat(["complete", "27K"], n)})
    for rep in reps:
        out[f"umap1_{rep}"], out[f"umap2_{rep}"] = res[f"E|{rep}"][:, 0], res[f"E|{rep}"][:, 1]
    out.to_parquet(WORK / "part_e.parquet")
    pd.DataFrame(met).to_csv(WORK / "part_e_metrics.csv", index=False)
    log(f"part E done ({time.time() - t0:.0f} s)")
    save_checks("e")


# ----------------------------------------------------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------------------------------------------------
def stage_write():
    fx = ftx()
    OUT.mkdir(parents=True, exist_ok=True)
    A = pd.read_parquet(require(WORK / "part_a.parquet"))
    high, ref, stable, defined, stable_s, defined_s = define_sets(A)
    A["high_attribution"], A["age_stable"], A["age_stable_tissue_adjusted"], A["age_correlated_reference"] = high, stable, stable_s, ref
    files = {}

    def put(df, path):
        df.to_csv(path, index=False, float_format="%.6g")
        files[str(path.relative_to(ARTIFACTS))] = {"rows": int(len(df)), "bytes": path.stat().st_size}
    put(A, OUT / f"part_a_cpg_age_attribution_{DATE}.csv.gz")
    put(pd.read_csv(WORK / "part_b_enrichment.csv"), OUT / f"part_b_enrichment_{DATE}.csv")
    put(pd.read_csv(WORK / "part_b_distributions.csv"), OUT / f"part_b_distributions_{DATE}.csv")
    for col in ("GO", "KEGG"):
        g = WORK / "gometh" / f"{col}.csv"
        if g.exists():
            put(pd.read_csv(g), OUT / f"part_b_gometh_{col}_{DATE}.csv")
    put(pd.read_csv(WORK / "part_c_by_draw.csv"), OUT / f"part_c_masking_by_draw_{DATE}.csv")
    put(pd.read_csv(WORK / "part_c_summary.csv"), OUT / f"part_c_masking_summary_{DATE}.csv")
    d = pd.read_parquet(WORK / "part_d.parquet")
    chk("1 GEO only (part D)", fx.is_geo(d.profile_id, d.dataset).all())
    put(d, FT / f"fig3_option1_setD_umap_defaults_{DATE}.csv.gz")
    e = pd.read_parquet(WORK / "part_e.parquet")
    chk("1 GEO only (part E)", fx.is_geo(e.profile_id, e.dataset).all() and (pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"), columns=["split"]).split.to_numpy()[e.corpus_row] == "test").all())
    put(e, FT / f"fig3_option2_cross_array_umap_{DATE}.csv.gz")
    put(pd.read_csv(WORK / "part_e_metrics.csv"), FT / f"fig3_option2_cross_array_metrics_{DATE}.csv")
    checks = []
    for st in ("a", "b", "c", "d", "e"):
        checks += json.loads(require(WORK / f"checks_{st}.json").read_text())
    checks += CHECKS
    pd.DataFrame(checks).to_csv(OUT / f"checks_{DATE}.csv", index=False)
    summary = {"part_b": json.loads((WORK / "part_b.json").read_text()), "part_c": json.loads((WORK / "part_c.json").read_text()),
               "files": files, "checks_passed": all(c["passed"] for c in checks)}
    write_json(OUT / f"summary_{DATE}.json", summary)
    write_json(WORK / "files.json", files)
    save_checks("write")
    write_readmes(files, summary, pd.DataFrame(checks))
    log(f"written {files}")


DESC = {
    f"part_a_cpg_age_attribution_{DATE}.csv.gz": "Part A, one row per vocabulary CpG: the statistics and flags of the docstring, and the part B set flags "
        "(high_attribution, age_stable, age_stable_tissue_adjusted, age_correlated_reference). rho columns are Spearman correlations of beta with "
        "metadata age; delta_beta_old_minus_young and p_mannwhitney_old_young compare old (>= 60, 724) with young (< 30, 1,000) profiles; "
        "eta_squared_tissue is the between-tissue-group fraction of the variance of beta; log2_ratio_attribution_old_young is table6 of the 2026-10-23 export.",
    f"part_b_enrichment_{DATE}.csv": "Part B: high-attribution age-stable CpGs against the rest of the background, per annotation (odds ratio, Woolf 95% interval, Fisher P).",
    f"part_b_distributions_{DATE}.csv": "Part B: eta squared and SD of beta in the four groups, and the old-versus-young attribution log2 ratio of the high-attribution age-stable CpGs against the other high-attribution CpGs.",
    f"part_b_gometh_GO_{DATE}.csv": "Part B: gometh GO result for the high-attribution age-stable CpGs (all terms, sorted by P).",
    f"part_b_gometh_KEGG_{DATE}.csv": "Part B: gometh KEGG result (all pathways, sorted by P).",
    f"part_c_masking_by_draw_{DATE}.csv": "Part C: every masking condition and draw, overall and per tissue group with >= 100 profiles: MedAE, mean signed error, change from baseline with 95% dataset-resampling intervals.",
    f"part_c_masking_summary_{DATE}.csv": "Part C: per condition and profile subset, mean and range over draws of the change from baseline.",
    f"fig3_option1_setD_umap_defaults_{DATE}.csv.gz": "Part D: set D UMAPs with umap-learn defaults (15 neighbours, min_dist 0.1, Euclidean, random_state 42) for raw values and the encoder output, on the 50 principal components (pcs50) and on the full features (full); same profiles and order as fig3_setD_umap_2026-11-10.csv.gz.",
    f"fig3_option2_cross_array_umap_{DATE}.csv.gz": "Part E: set C profiles with >= 90% of EPIC v1 CpGs observed, each as a complete and a 27K copy (column copy); one UMAP per representation over both copies (30 neighbours, min_dist 0.3, seed 0, on 50 principal components).",
    f"fig3_option2_cross_array_metrics_{DATE}.csv": "Part E: per representation, the fraction of 27K copies whose nearest complete profile is their own complete copy, and the median copy distance over the median nearest-neighbour distance among complete profiles; 95% intervals resampling datasets.",
}


def write_readmes(files, summary, checks):
    b, c = summary["part_b"], summary["part_c"]
    L = [(OUT / "README.md").read_text(), "## Files", ""]
    for k, v in files.items():
        if k.startswith(f"attribution_stability_{DATE}"):
            L.append(f"- `{k.split('/', 1)[1]}`: {v['rows']:,} rows, {v['bytes']:,} bytes. {DESC.get(k.split('/', 1)[1], '')}")
    L.append(f"- `checks_{DATE}.csv`, `summary_{DATE}.json`: every check; part B and C summaries.")
    L.extend(["", "Inputs: `artifacts/si_attribution_old_young_2026-10-23/per_profile_abs_attribution.npy`, `per_profile_rows.npy`, `group_means.npz`; "
              "`artifacts/corpus_2026-09-30/` (beta values, profile and age tables); `artifacts/cpg_annotation_2026-10-01.parquet`; "
              "`artifacts/external/BENPORATH_PRC2_TARGETS.txt`; EPIC v1.0 B5 manifest (TFBS); `artifacts/figure_tables_2026-10-23/table5_fig5bd_cpg_attribution.csv.gz`, "
              "`table6_fig5c_old_young_attribution.csv.gz`; `artifacts/release/methylgpt2-1.1`; `artifacts/eval/age_ensemble_2026-10-04.predictions.npz` (check); "
              "`artifacts/si_enrichment_2026-10-21/attribution.npz`, `direction.npz` (check 3).", "", "## Part B results", ""])
    for name in ("age-stable", "age-stable, tissue-adjusted (sensitivity)"):
        r = b[name]
        L.append(f"- {name}: {r['high_attribution_age_stable']} of {r['high_attribution_with_statistics']} high-attribution CpGs "
                 f"(expected {r['expected']:.1f} from the age-stable fraction {r['age_stable_fraction']:.4f} of {r['cpgs_with_statistics']:,} CpGs); "
                 f"odds ratio {r['odds_ratio']:.3f} (95% interval {r['ci95'][0]:.3f} to {r['ci95'][1]:.3f}), Fisher P {r['fisher_p']:.3g}.")
    L.append(f"- Overlap of the 1,000 high-attribution CpGs with the 1,000 most age-correlated CpGs (|rho all|): {b['overlap_high_attribution_age_correlated_reference']}. "
             f"The earlier rankings give {b['overlap_earlier_rankings_check']} (check 3).")
    g = b.get("gometh", {})
    for col in ("GO", "KEGG"):
        if col in g:
            L.append(f"- gometh {col}: {g[col]['terms_fdr_0.05']} of {g[col]['terms_tested']:,} terms at FDR < 0.05 (smallest FDR {g[col]['min_fdr']:.3g}).")
    if g.get("returncode", 0) != 0:
        L.append(f"- gometh did not run: {g.get('stderr_tail', '')[-300:]}")
    L.append(f"- Old-versus-young attribution log2 ratio, high-attribution age-stable against other high-attribution CpGs: Mann-Whitney P {b['log2_ratio_mannwhitney_p']}.")
    L.extend(["", "## Part C", "", f"k = {c['k']} CpGs per masked set (1) to (3); pool of set (2): {c['pool2']} CpGs"
              + ("; smaller than k, so every draw masks the whole pool" if c["pool2_smaller_than_k"] else "") + f"; {c['strata']} matching strata.", "",
              "| set | profiles | draws | CpGs masked | change in MedAE, mean (range) | change in mean signed error, mean (range) |", "|---|---|---|---|---|---|"])
    sm = pd.read_csv(OUT / f"part_c_masking_summary_{DATE}.csv")
    for _, r in sm[sm.profiles == "all"].iterrows():
        L.append(f"| {r.set} | all | {r.n_draws} | {r.n_cpgs_masked} | {r.delta_medae_mean:.3f} ({r.delta_medae_min:.3f} to {r.delta_medae_max:.3f}) | "
                 f"{r.delta_mse_mean:.3f} ({r.delta_mse_min:.3f} to {r.delta_mse_max:.3f}) |")
    A = pd.read_csv(OUT / f"part_a_cpg_age_attribution_{DATE}.csv.gz")
    S = A[A.high_attribution & A.age_stable]
    nadj = int((S.rho_age_tissue_adjusted.abs() >= 0.1).sum())
    L.extend(["", "Per tissue and per draw, with 95% intervals: part_c_masking_by_draw.", "", "## Notes", "",
              f"- {int(S.in_any_of_four_clocks.sum())} of the {len(S)} high-attribution age-stable CpGs are CpGs of the four clocks (part_b_enrichment). "
              f"{nadj} of the {len(S)} have |tissue-adjusted rho| >= 0.1: weak age correlation over all profiles, stronger within tissue groups; "
              "the sensitivity definition (14 CpGs) excludes them.",
              "- Amendment 1 (after part C's predictions): a tissue group's interval uses the resampled sets that contain at least one of its profiles "
              "(column bootstrap_draws_with_profiles). Adipose comes from one dataset, so its interval is a single value. A median recomputed on "
              "resampled datasets can put the point estimate outside the percentile interval (Other, set 1).",
              "- Masking 27 CpGs changes the age error little for every set, including the drawn and matched controls; masking the 1,000 most "
              "age-correlated CpGs raises MedAE by 0.116 years overall.", "", "## Checks", ""])
    L.extend(f"- {r.check}: {'pass' if r.passed else 'FAIL'}{'; ' + str(r.detail) if isinstance(r.detail, str) and r.detail else ''}" for _, r in checks.iterrows())
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    em = pd.read_csv(FT / f"fig3_option2_cross_array_metrics_{DATE}.csv")
    M = ["# Fig. 3 options (parts D and E)", "",
         f"Written by `attribution_stability_fig3_options_{DATE}.py`; design in its docstring (committed before the run). Public GEO profiles only "
         "(the profiles of the set C and D tables of this folder). README.md of this folder describes the set C and D tables.", ""]
    for k, v in files.items():
        if k.startswith(f"figure_tables_{DATE}"):
            M.append(f"- `{k.split('/', 1)[1]}`: {v['rows']:,} rows, {v['bytes']:,} bytes. {DESC.get(k.split('/', 1)[1], '')}")
    M.extend(["", "Columns of the option 1 table: corpus_row, profile_id, dataset, umap1/umap2 for raw and encoder on pcs50 and full. "
              "Option 2 table: corpus_row, profile_id, dataset, tissue, array (as the set C table), copy (complete or 27K), umap1/umap2 for raw, latent and encoder.", "",
              "## Part E metrics", "", "| representation | metric | value [95% interval] | profiles / datasets |", "|---|---|---|---|"])
    for _, r in em.iterrows():
        M.append(f"| {r.representation} | {r.metric} | {r.value:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.n_profiles:,} / {r.n_datasets} |")
    M.extend(["", "Checks: see artifacts/attribution_stability_2026-11-10/checks_2026-11-10.csv (checks 1, 5, 6)."])
    (FT / f"README_options_{DATE}.md").write_text("\n".join(M) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b", "c", "c_summary", "d", "e", "write", "all"], required=True)
    a = ap.parse_args()
    for s in (["a", "b", "c", "d", "e", "write"] if a.stage == "all" else [a.stage]):
        CHECKS.clear()
        {"a": stage_a, "b": stage_b, "c": stage_c, "c_summary": summarise_c, "d": stage_d, "e": stage_e, "write": stage_write}[s]()


if __name__ == "__main__":
    main()
