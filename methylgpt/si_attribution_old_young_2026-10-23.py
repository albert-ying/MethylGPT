#!/usr/bin/env python3
"""Age attribution of old against young held-out profiles, per CpG and by gene set (tables for Fig. 5b, c and e).

Written on 24 Oct 2026, before running (file name dated 23 Oct to match the export specification
of 23 Oct 2026, tables 5-7). Every result is reported, whichever way it falls.

Data and inclusion
  Held-out (test-split) profiles of the MethylGPT corpus with a metadata age: 5,811 profiles from 34 public GEO
  datasets (artifacts/corpus_2026-09-30/profiles.parquet, profiles_age_clean_2026-10-01.parquet). No Generation
  Scotland data are read.
  Young and old: the profiles already drawn by si_followup_2026-10-22.py (rows|young and rows|old of
  artifacts/si_followup_2026-10-22/attribution_age.npz: 1,000 of the 3,200 profiles younger than 30 years, seed
  20261022, and all 724 profiles aged 60 or older), so the young and old columns of tables 5 and 6 describe the
  same profiles. The lists are frozen by that committed file; this script reads them and does not redraw them.
  Three age groups (table 5): all 5,811 held-out profiles, younger than 20 years (2,445), 20 to under 60 (2,642)
  and 60 or older (724). The specification writes ">60"; age 60 is placed in the oldest group so that the groups are
  exhaustive and the oldest group is the old group above. The counts were read from the profile table before this
  script was written; they are inclusion counts, not outcomes.

Model and attribution (stage attribute, GPU; no model is fitted)
  Gradient x input of the released age output, the code of si_followup_2026-10-22.py attribution_age: the three
  released runs (runs/multitask_bio_*/final.pt), gradient of the standardized age output with respect to each
  observed beta value times the beta value minus the input-layer training mean; missing CpGs are filled with the
  training mean and get zero attribution. Profiles are processed in batches of 4 in this order: the young rows
  (sorted), the old rows (sorted), then the other held-out rows (sorted), so the young and old batches are those of
  attribution_age. The per-profile value of a CpG is its absolute attribution averaged over the three runs (missing
  where the CpG is not observed). A group mean is the mean of this value over the group's profiles in which the
  CpG is observed; it equals the mean over runs of the per-run group score in attribution_age.npz.
  Checks (reported): per-run young and old scores against attribution_age.npz (maximum absolute difference), and
  the predicted ages against the saved held-out predictions of each run.

Table 6 (stage test, CPU)
  Tested CpGs: vocabulary CpGs observed in at least 10 young and 10 old profiles. For each: mean_young, mean_old,
  log2(mean_old / mean_young), and a two-sided Mann-Whitney U test of the per-profile values, old against young
  (scipy.stats.mannwhitneyu, method "asymptotic", tie and continuity correction). Multiple testing:
  Benjamini-Hochberg q over all tested CpGs (scipy.stats.false_discovery_control, method "bh").
  PRIMARY (table 6): the number of tested CpGs with q < 0.05, and of these the numbers higher in old (log2 ratio
  > 0) and higher in young.
  Secondary (all reported): the same test within Whole Blood (783 young, 116 old profiles) and within Brain (103,
  256), the only tissues with at least 100 profiles in both groups, each with its own BH correction over the CpGs
  observed in at least 10 profiles of each group: CpGs at q < 0.05 by direction, and the Spearman correlation of the
  within-tissue log2 ratios with the pooled log2 ratios over CpGs tested in both. Reason, stated before running:
  the young group is 78% whole blood and the old group 35% brain and 20% colon, so pooled differences mix age and
  tissue. Also the median pooled log2 ratio, the Spearman correlation of the young and old group means over tested
  CpGs, and the overlap of the 1,000 CpGs with the largest positive log2 ratio with the four published clocks
  (Horvath multi-tissue, Horvath skin and blood, Hannum, PhenoAge; BioLearn files, as si_followup).

Table 7 (stage genesets, R)
  CpG-to-gene mapping of the gene-set stage of si_followup_2026-10-22.py: background CpGs are the EPIC v1.0
  vocabulary CpGs observed in the 2,000 profiles of artifacts/si_enrichment_2026-10-21/attribution.npz, here also
  restricted to the CpGs tested in table 6 (a CpG unobserved in a group has no group mean); each CpG maps to every
  RefGene name in artifacts/cpg_annotation_2026-10-01.parquet; gene score = mean over the gene's CpGs of
  (mean_old - mean_young). fgsea settings of si_followup: fgseaMultilevel, minSize 15, maxSize 500,
  set.seed(20261022), MSigDB Hallmark, GO biological process, KEGG legacy and Reactome through the same msigdbr
  calls. One setting differs, fixed before running: scoreType "std" instead of "pos", because the difference is
  signed (positive NES: genes whose attribution is higher in old profiles; negative: higher in young).
  PRIMARY (table 7): the number of Hallmark gene sets at FDR (fgsea padj) < 0.05.
  Secondary: the same count for each collection, split by the sign of NES.

Limits, stated before running: the Mann-Whitney and fgsea P values treat profiles (and genes) as independent. The
young and old profiles come from 25 and 19 datasets (10 in both) and differ in tissue, so q values describe these
profiles; they are not tests of age adjusted for dataset or tissue.
Seeds and resamples: no resampling; fgsea seed 20261022; young draw seed 20261022 (inherited, not redrawn).

Outputs
  artifacts/si_attribution_old_young_2026-10-23/per_profile_abs_attribution.npy  (5,811 x 49,156 float32; public
      profiles; not committed)  and  per_profile_rows.npy
  artifacts/si_attribution_old_young_2026-10-23/group_means.npz          group means and observed counts per CpG
  artifacts/si_attribution_old_young_2026-10-23/attribution_checks.json
  artifacts/si_attribution_old_young_2026-10-23/old_young_cpg.csv.gz      table 6 rows (all CpGs)
  artifacts/si_attribution_old_young_2026-10-23/genesets/                 gene scores, run.R, fgsea_*.csv, run.log
  artifacts/eval/si_attribution_old_young_2026-10-23.json                 primary and secondary results
  The figure tables are written by export_figure_tables_2026-10-23.py.
Stages: --stage attribute (GPU), --stage test, --stage genesets (R), --stage summarise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, log, require, write_json

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
OUT = ARTIFACTS / "si_attribution_old_young_2026-10-23"
FOLLOW = ARTIFACTS / "si_followup_2026-10-22"
ENR = ARTIFACTS / "si_enrichment_2026-10-21"
FGSEA_SEED = 20261022
MIN_OBS = 10
GROUPS3 = {"lt20": (-np.inf, 20.0), "20to60": (20.0, 60.0), "ge60": (60.0, np.inf)}   # [lo, hi)
TISSUES = ["Whole Blood", "Brain"]


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def heldout():
    prof = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
    lab = pd.read_parquet(require(CORPUS_DIR / "profiles_age_clean_2026-10-01.parquet"))
    age = lab.age_meta.to_numpy(float)
    ho = np.flatnonzero((prof.split == "test").to_numpy() & np.isfinite(age))
    return prof, age, ho


def attribute(device):
    import torch
    from methylgpt_lib import Corpus, cuda_preinit, Finv
    from methylgpt_model import BioHeads, Heads, load_checkpoint, embed_batch
    se = _module("si_enrichment_attribution", "si_enrichment_attribution_2026-10-21.py")
    cuda_preinit()
    dev = torch.device(device)
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = Corpus()
    prof, age, ho = heldout()
    za = np.load(require(FOLLOW / "attribution_age.npz"))
    young, old = za["rows|young"], za["rows|old"]
    rest = np.setdiff1d(ho, np.concatenate([young, old]))
    if not (np.isin(young, ho).all() and np.isin(old, ho).all()):
        raise SystemExit("young/old rows are not all held-out profiles with an age")
    blocks = [np.sort(young), np.sort(old), np.sort(rest)]
    order = np.concatenate(blocks)
    n, p = len(order), za["score|young|run 1"].shape[0]
    log(f"profiles: young {len(young)}, old {len(old)}, other {len(rest)}; total {n}")
    acc = np.zeros((n, p), np.float32)
    zsaved = np.load(require(EVAL / "age_revision_2026-10-01.predictions.npz"))
    zseeds = {"seed 2": np.load(require(EVAL / "age_seed2_scenarios_2026-10-04.predictions.npz")),
              "seed 3": np.load(require(EVAL / "age_seed3_scenarios_2026-10-04.predictions.npz"))}
    saved_key = {"run 1": (zsaved, "MethylGPT|heldout"), "seed 2": (zseeds["seed 2"], "MethylGPT seed 2|heldout"),
                 "seed 3": (zseeds["seed 3"], "MethylGPT seed 3|heldout")}
    checks = {}
    t0 = time.time()
    observed = None
    for name, rel in se.RUNS.items():
        ck = torch.load(require(ARTIFACTS / rel), map_location=dev, weights_only=False)
        model, core, _ = load_checkpoint(Path(ck["ckpt"]), ck.get("config", {}).get("core"), dev)
        model.load_state_dict(ck["model"]); model.eval()
        d_in = core.rank + model.latents.shape[1]
        heads = (BioHeads(d_in, len(ck["tissues"]) + 1, len(ck["bio_names"])) if "bio_names" in ck else Heads(d_in, len(ck["tissues"]) + 1)).to(dev)
        heads.load_state_dict(ck["heads"]); heads.eval()
        for obj in {id(core): core, id(model.core): model.core}.values():
            obj.posterior = types.MethodType(type(obj).posterior.__wrapped__, obj)
            obj.predict = types.MethodType(type(obj).predict.__wrapped__, obj)
        model.inputs = types.MethodType(type(model).inputs.__wrapped__, model)
        for mod in (model, heads, core):
            for prm in mod.parameters():
                prm.requires_grad_(False)
        mu = core.mu.detach()
        pred = np.empty(n)
        obs_run = np.zeros((n, p), bool)
        start = 0
        for bi, b in enumerate(blocks):
            X = corpus.get(b)
            S = np.zeros(p)   # per-run block sum of |a|, as attribution_age
            for i in range(0, len(X), 4):
                xb = torch.from_numpy(X[i:i + 4]).to(dev)
                ctx = torch.isfinite(xb)
                x = torch.where(ctx, xb, mu).requires_grad_(True)
                out = heads(embed_batch(model, core, x, ctx))[0]
                out.sum().backward()
                a = (x.grad * (x.detach() - mu) * ctx).cpu().numpy()
                S += np.abs(a).sum(0)
                acc[start + i:start + i + len(a)] += np.abs(a).astype(np.float32)
                obs_run[start + i:start + i + len(a)] = ctx.cpu().numpy()
                pred[start + i:start + i + len(a)] = Finv(out.detach().float().cpu().numpy() * ck["fs"] + ck["fm"])
            if bi < 2:   # young and old blocks: check against the per-run scores of attribution_age.npz
                gname = ("young", "old")[bi]
                score = S / np.maximum(obs_run[start:start + len(b)].sum(0), 1)
                checks[f"{name}|{gname}|max_abs_diff_vs_attribution_age"] = float(np.max(np.abs(score - za[f"score|{gname}|{name}"])))
            start += len(b)
        if observed is None:
            observed = obs_run
        elif not np.array_equal(observed, obs_run):
            raise SystemExit("observed masks differ between runs")
        zs, key = saved_key[name]
        checks[f"{name}|max_abs_diff_vs_saved_prediction"] = float(np.max(np.abs(pred - zs[key][np.searchsorted(ho, order)])))
        log(f"{name}: done ({time.time() - t0:.0f} s); prediction check {checks[f'{name}|max_abs_diff_vs_saved_prediction']:.2e}")
        del model, core, heads, ck
        torch.cuda.empty_cache()
    acc /= len(se.RUNS)
    acc[~observed] = np.nan
    # run-mean check: mean over runs of the per-run young/old scores equals the group mean of the run-mean values
    for gname, blk in (("young", slice(0, len(young))), ("old", slice(len(young), len(young) + len(old)))):
        ref = np.mean([za[f"score|{gname}|{r}"] for r in se.RUNS], axis=0)
        mine = np.nansum(acc[blk].astype(np.float64), axis=0) / np.maximum(observed[blk].sum(0), 1)
        checks[f"run_mean|{gname}|max_abs_diff_vs_attribution_age"] = float(np.max(np.abs(mine - ref)))
        checks[f"run_mean|{gname}|max_rel_diff_vs_attribution_age"] = float(np.max(np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-12)))
    np.save(OUT / "per_profile_abs_attribution.npy", acc)
    np.save(OUT / "per_profile_rows.npy", order)
    write_json(OUT / "attribution_checks.json", checks)
    log(f"attribution written; checks {checks}")


def _mwu_table(V, ia, ib, min_obs=MIN_OBS):
    """Per-CpG two-sided Mann-Whitney U of V[ib] (old) against V[ia] (young); V has NaN where not observed."""
    from scipy.stats import mannwhitneyu, false_discovery_control
    A, B = V[ia], V[ib]
    na, nb = np.isfinite(A).sum(0), np.isfinite(B).sum(0)
    ma = np.nansum(A.astype(np.float64), 0) / np.maximum(na, 1)
    mb = np.nansum(B.astype(np.float64), 0) / np.maximum(nb, 1)
    tested = (na >= min_obs) & (nb >= min_obs)
    U = np.full(V.shape[1], np.nan); P = np.full(V.shape[1], np.nan)
    for j in np.flatnonzero(tested):
        a = A[:, j]; b = B[:, j]
        r = mannwhitneyu(b[np.isfinite(b)], a[np.isfinite(a)], alternative="two-sided", method="asymptotic", use_continuity=True)
        U[j], P[j] = r.statistic, r.pvalue
    Q = np.full(V.shape[1], np.nan)
    Q[tested] = false_discovery_control(P[tested], method="bh")
    with np.errstate(divide="ignore", invalid="ignore"):
        l2 = np.where(tested & (ma > 0) & (mb > 0), np.log2(mb / ma), np.nan)
    return {"n_young": na, "n_old": nb, "mean_young": ma, "mean_old": mb, "log2_ratio": l2, "U": U, "p": P, "q": Q, "tested": tested}


def _count(t):
    sig = t["tested"] & (t["q"] < 0.05)
    return {"tested": int(t["tested"].sum()), "q_lt_0.05": int(sig.sum()),
            "q_lt_0.05_higher_in_old": int((sig & (t["log2_ratio"] > 0)).sum()),
            "q_lt_0.05_higher_in_young": int((sig & (t["log2_ratio"] < 0)).sum())}


def test():
    from scipy.stats import spearmanr
    se = _module("si_enrichment_attribution", "si_enrichment_attribution_2026-10-21.py")
    sf = _module("si_followup", "si_followup_2026-10-22.py")
    prof, age, ho = heldout()
    V = np.load(require(OUT / "per_profile_abs_attribution.npy"), mmap_mode="r")
    order = np.load(require(OUT / "per_profile_rows.npy"))
    za = np.load(require(FOLLOW / "attribution_age.npz"))
    pos = {r: i for i, r in enumerate(order)}
    iy = np.array([pos[r] for r in za["rows|young"]]); io = np.array([pos[r] for r in za["rows|old"]])
    V = np.asarray(V)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    cpg = ann.cpg.to_numpy()
    # group means for table 5 (three age groups) and young/old
    gm = {"rows|young": za["rows|young"], "rows|old": za["rows|old"]}
    ag = age[order]
    for g, (lo, hi) in GROUPS3.items():
        idx = np.flatnonzero((ag >= lo) & (ag < hi))
        nobs = np.isfinite(V[idx]).sum(0)
        gm[f"mean|{g}"] = np.nansum(V[idx].astype(np.float64), 0) / np.maximum(nobs, 1)
        gm[f"n_observed|{g}"] = nobs
        gm[f"n_profiles|{g}"] = np.array(len(idx))
        gm[f"rows|{g}"] = order[idx]
        log(f"group {g}: {len(idx)} profiles")
    # primary: pooled old vs young
    t = _mwu_table(V, iy, io)
    gm.update({f"mean|young": t["mean_young"], "mean|old": t["mean_old"], "n_observed|young": t["n_young"], "n_observed|old": t["n_old"]})
    np.savez(OUT / "group_means.npz", **gm)
    tab = pd.DataFrame({"cpg_id": cpg, "n_young_observed": t["n_young"], "n_old_observed": t["n_old"],
                        "mean_abs_attribution_young": t["mean_young"], "mean_abs_attribution_old": t["mean_old"],
                        "log2_ratio_old_young": t["log2_ratio"], "mannwhitney_u_old_vs_young": t["U"],
                        "p_mannwhitney": t["p"], "q_bh": t["q"], "tested": t["tested"]})
    res = {"design": {"young": "<30 years, 1,000 drawn (si_followup_2026-10-22, seed 20261022)", "old": ">=60 years, all 724",
                      "min_observed_per_group": MIN_OBS, "test": "two-sided Mann-Whitney U, asymptotic, continuity correction",
                      "correction": "Benjamini-Hochberg over tested CpGs"},
           "n_young": int(len(iy)), "n_old": int(len(io)),
           "datasets_young": int(prof.group.iloc[za["rows|young"]].nunique()), "datasets_old": int(prof.group.iloc[za["rows|old"]].nunique()),
           "datasets_both": int(len(set(prof.group.iloc[za["rows|young"]]) & set(prof.group.iloc[za["rows|old"]]))),
           "tissue_young": prof.tissue.iloc[za["rows|young"]].fillna("no tissue label").value_counts().to_dict(),
           "tissue_old": prof.tissue.iloc[za["rows|old"]].fillna("no tissue label").value_counts().to_dict(),
           "primary_table6": {"definition": "tested CpGs with BH q < 0.05, old (>=60) against young (<30) per-profile absolute attribution, pooled", **_count(t)}}
    tt = t["tested"]
    res["secondary_table6"] = {"median_log2_ratio_tested": float(np.nanmedian(t["log2_ratio"][tt])),
                               "spearman_young_old_means_tested": float(spearmanr(t["mean_young"][tt], t["mean_old"][tt]).statistic),
                               "three_groups_n_profiles": {g: int(gm[f"n_profiles|{g}"]) for g in GROUPS3}}
    # clock overlap of the 1,000 CpGs whose attribution is most higher in old
    sets = {}
    for name, f in sf.CLOCKS.items():
        c = pd.read_csv(require(sf.CLOCK_DIR / f)).CpGmarker.astype(str)
        sets[name] = np.isin(cpg, c.to_numpy())
    sets["union of four clocks"] = np.any(list(sets.values()), axis=0)
    top = np.zeros(len(cpg), bool)
    top[np.argsort(-np.where(tt & np.isfinite(t["log2_ratio"]), t["log2_ratio"], -np.inf))[:1000]] = True
    res["secondary_table6"]["top1000_log2_ratio_clock_overlap"] = {k: {**se.fisher(top[tt], s[tt]), "clock_cpgs_tested": int(s[tt].sum())} for k, s in sets.items()}
    # within-tissue secondaries
    tis = prof.tissue.to_numpy()
    res["secondary_within_tissue"] = {}
    for T in TISSUES:
        ty = iy[tis[za["rows|young"]] == T]; to = io[tis[za["rows|old"]] == T]
        w = _mwu_table(V, ty, to)
        both = tt & w["tested"] & np.isfinite(w["log2_ratio"]) & np.isfinite(t["log2_ratio"])
        res["secondary_within_tissue"][T] = {"n_young": int(len(ty)), "n_old": int(len(to)), **_count(w),
                                             "spearman_log2_ratio_with_pooled": float(spearmanr(w["log2_ratio"][both], t["log2_ratio"][both]).statistic),
                                             "cpgs_in_correlation": int(both.sum())}
        key = T.lower().replace(" ", "_")
        tab[f"{key}_n_young_observed"] = w["n_young"]; tab[f"{key}_n_old_observed"] = w["n_old"]
        tab[f"{key}_log2_ratio_old_young"] = w["log2_ratio"]; tab[f"{key}_p_mannwhitney"] = w["p"]; tab[f"{key}_q_bh"] = w["q"]
        log(f"{T}: {res['secondary_within_tissue'][T]}")
    tab.to_csv(OUT / "old_young_cpg.csv.gz", index=False)
    write_json(OUT / "test_results.json", res)
    log(f"primary table 6: {res['primary_table6']}")


def genesets():
    d = OUT / "genesets"; d.mkdir(parents=True, exist_ok=True)
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    A = np.load(require(ENR / "attribution.npz"))
    runs = [k.split("|", 1)[1] for k in A.files if k.startswith("score|")]
    bg = ann.in_epic_v1.to_numpy() & (np.mean([A[f"n_observed|{r}"] for r in runs], axis=0) > 0)
    tab = pd.read_csv(require(OUT / "old_young_cpg.csv.gz"))
    if not np.array_equal(tab.cpg_id.to_numpy(), ann.cpg.to_numpy()):
        raise SystemExit("CpG order differs from the annotation")
    keep = bg & tab.tested.to_numpy(bool)
    diff = (tab.mean_abs_attribution_old - tab.mean_abs_attribution_young).to_numpy()
    genes = ann.genes.fillna("").astype(str).str.split(";")
    rows = [(g, diff[i]) for i in np.flatnonzero(keep) for g in set(genes[i]) if g]
    gsc = pd.DataFrame(rows, columns=["gene", "score"]).groupby("gene").score.agg(["mean", "size"]).rename(columns={"mean": "score", "size": "n_cpgs"})
    gsc.to_csv(d / "gene_scores.csv")
    write_json(d / "diagnostics.json", {"background_cpgs_si_followup": int(bg.sum()), "background_cpgs_used": int(keep.sum()),
                                        "genes": int(len(gsc)), "genes_score_positive": int((gsc.score > 0).sum())})
    script = f"""
suppressMessages({{library(fgsea); library(msigdbr)}})
d <- "{d}"
gs <- read.csv(file.path(d, "gene_scores.csv"))
stats <- setNames(gs$score, gs$gene)
get_sets <- function(coll, sub = NULL) {{
  m <- tryCatch(if (is.null(sub)) msigdbr(species = "Homo sapiens", collection = coll) else msigdbr(species = "Homo sapiens", collection = coll, subcollection = sub),
                error = function(e) if (is.null(sub)) msigdbr(species = "Homo sapiens", category = coll) else msigdbr(species = "Homo sapiens", category = coll, subcategory = sub))
  split(m$gene_symbol, m$gs_name)
}}
colls <- list(Hallmark = c("H", NA), GO_BP = c("C5", "GO:BP"), KEGG_legacy = c("C2", "CP:KEGG_LEGACY"), Reactome = c("C2", "CP:REACTOME"))
set.seed({FGSEA_SEED})
for (nm in names(colls)) {{
  cc <- colls[[nm]]
  sets <- get_sets(cc[1], if (is.na(cc[2])) NULL else cc[2])
  r <- fgseaMultilevel(pathways = sets, stats = stats, scoreType = "std", minSize = 15, maxSize = 500)
  r <- r[order(r$pval), ]
  r$leadingEdge <- sapply(r$leadingEdge, function(x) paste(x, collapse = ";"))
  write.csv(as.data.frame(r), file.path(d, paste0("fgsea_", nm, ".csv")), row.names = FALSE)
}}
cat("fgsea", as.character(packageVersion("fgsea")), "msigdbr", as.character(packageVersion("msigdbr")), "\\n")
"""
    (d / "run.R").write_text(script)
    p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True)
    (d / "run.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"R failed: {p.stderr[-3000:]}")
    log("genesets written")


def summarise():
    res = json.loads(require(OUT / "test_results.json").read_text())
    res["checks"] = json.loads(require(OUT / "attribution_checks.json").read_text())
    d = OUT / "genesets"
    res["genesets_diagnostics"] = json.loads(require(d / "diagnostics.json").read_text())
    res["genesets_versions"] = (d / "run.log").read_text().strip().splitlines()[0] if (d / "run.log").exists() else ""
    G = {}
    for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome"):
        r = pd.read_csv(require(d / f"fgsea_{nm}.csv"))
        sig = r.padj < 0.05
        G[nm] = {"tested": int(len(r)), "fdr_0.05": int(sig.sum()), "fdr_0.05_NES_positive": int((sig & (r.NES > 0)).sum()),
                 "fdr_0.05_NES_negative": int((sig & (r.NES < 0)).sum()),
                 "top10": [{"pathway": str(x.pathway), "size": int(x["size"]), "NES": float(x.NES), "pval": float(x.pval), "padj": float(x.padj)}
                           for _, x in r.head(10).iterrows()]}
    res["secondary_table7"] = G
    res["primary_table7"] = {"definition": "Hallmark gene sets at FDR < 0.05, fgsea (scoreType std) on gene-level old-minus-young mean absolute attribution",
                             "fdr_0.05": G["Hallmark"]["fdr_0.05"], "tested": G["Hallmark"]["tested"],
                             "NES_positive": G["Hallmark"]["fdr_0.05_NES_positive"], "NES_negative": G["Hallmark"]["fdr_0.05_NES_negative"]}
    write_json(EVAL / "si_attribution_old_young_2026-10-23.json", res)
    log(f"primary table 6 {res['primary_table6']} | primary table 7 {res['primary_table7']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["attribute", "test", "genesets", "summarise"], required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    {"attribute": lambda: attribute(a.device), "test": test, "genesets": genesets, "summarise": summarise}[a.stage]()


if __name__ == "__main__":
    main()
