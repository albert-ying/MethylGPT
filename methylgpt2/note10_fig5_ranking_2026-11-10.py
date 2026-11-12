#!/usr/bin/env python3
"""Supplementary Note 10 and Supplementary Fig. 14 on the attribution ranking of Fig. 5.

Written on 11 Nov 2026 before running. Every result is reported. Aggregate outputs only (public GEO held-out profiles);
no Generation Scotland file is read; nothing is trained.

WHAT CHANGES  Only the ranking. Note 10 and Supplementary Fig. 14 report the earlier run: mean absolute attribution over
  2,000 held-out profiles, from artifacts/si_enrichment_2026-10-21/attribution.npz. Here every analysis uses
  mean_abs_attribution of artifacts/attribution_stability_2026-11-10/part_a_cpg_age_attribution_2026-11-10.csv.gz
  (gradient x input of the released age output, three runs, 5,811 held-out profiles), the ranking of Fig. 5 and the
  Methods. The annotation files, background definitions and code are those of si_enrichment_attribution_2026-10-21.py
  and si_followup_2026-10-22.py, whose functions are imported: fisher() (conditional odds ratio with Woolf interval),
  mantel_haenszel(), manifest_tfbs(), the PRC2 gene list, the island strata, the direction file and the gometh and
  fgsea calls.
BACKGROUND (as before)  EPIC v1.0 vocabulary CpGs observed in the attribution profiles (in_epic_v1 and n_observed > 0;
  47,962 expected). Top n = the n background CpGs with the largest score.
ANALYSES
  - Annotations (TFBS, PRC2 targets, CpG island, CpG shore, enhancer, DNase hypersensitivity) among the top 500, 1,000
    and 2,000 against the rest of the background: percentages, odds ratio, 95% interval, Fisher P.
  - PRC2 targets: unstratified, and the Mantel-Haenszel odds ratio within CpG-island strata, for the three top sets.
  - PRC2 for the 1,000 background CpGs whose attribution rose most from young (< 30 years, 1,000 profiles) to old
    (>= 60, 724), by log(old / young): unstratified and Mantel-Haenszel; TFBS and CpG island as in si_followup A2. The
    young and old means come from si_attribution_old_young_2026-10-23/group_means.npz, the same profiles and values as
    the earlier attribution_age.npz, so this set changes only through the background.
  - Direction: methylation rises with age where the within-dataset correlation with age in 10,000 training profiles
    (si_enrichment_2026-10-21/direction.npz) is positive, as before. Among the top 1,000: counts gaining and losing,
    percentages, odds ratio and 95% interval against the background.
  - gometh: missMethyl, GO and KEGG, array.type "EPIC", prior.prob TRUE, all.cpg = the background. Sets: the top 1,000,
    the top 2,000, and the top 1,000 split into CpGs that gain and that lose methylation with age. For each: terms at
    FDR < 0.05, the lowest FDR, and the top 10 terms.
  - fgsea, rank-based over all genes, as si_followup G4: gene score = mean score of the gene's background CpGs;
    fgseaMultilevel, scoreType "pos", sizes 15 to 500, set.seed(20261022); Hallmark, GO biological process, KEGG legacy
    and Reactome. Panel d of Supplementary Fig. 14 needs it.
  - Clock CpGs, as the analysis specification asks: the 950 vocabulary CpGs of the four clocks (in_any_of_four_clocks of part A) among
    the top 100 and top 1,000 of all 49,156 vocabulary CpGs, with the odds ratio against the vocabulary background
    (39 and 149 expected). Per clock and their union against the EPIC background, as si_followup A1 (BioLearn files);
    this is panel a of Supplementary Fig. 14.
  - Overlap of the top 1,000 by attribution with the top 1,000 by:
    (i) |within-dataset correlation with age| in training profiles (direction.npz; the ranking behind the 148), in the
        background;
    (ii) |tissue-adjusted rho| in the held-out profiles (part A; Fig. 5d);
    (iii) |rho| within whole blood (part A).
    For (ii) and (iii) the universe is the vocabulary CpGs with both statistics defined. Both top-1,000 sets are taken
    within each universe. Expected overlap = 1000^2 / universe size. Clock CpGs (950) among the top 1,000 of each
    ranking are also given.
SUPPLEMENTARY FIG. 14  The existing script, figures/plot_si_attribution_harmonize_2026-10-22.py of the master branch
  (stage attr), run unchanged through figures/run_nofooter_2026-11-10.py in a staging folder. Its inputs are
  replaced by files with the same names and formats computed on the new ranking. Its outputs are copied to new names:
  note10_sf14_fig5_ranking_2026-11-10.pdf and .png. The panel data go to a CSV.
AMENDMENT 1 (11 Nov 2026, after the first run; statistics, gometh and fgsea unchanged): the existing script fixes the
  x axis of panel d at -log10 FDR 1 to 6, and with the new ranking the lowest fgsea FDR is 1.06e-6 (5.98), so the
  figure gate of nature_figure_utils blocks the figure (a point within 2 px of the axes edge). The figure is drawn
  with a copy of the script in which only that limit is 1 to 6.5 (ticks unchanged):
  figures/plot_si_attribution_fig5_ranking_2026-11-10.py, written by stage sf14.

CHECKS (abort on failure)
  1. The same code with the earlier ranking reproduces the stored values of si_enrichment_attribution_2026-10-21.json
     and si_followup_2026-10-22.json within 1e-9: TFBS, PRC2 and MH odds ratios at 500, 1,000 and 2,000; direction
     437 of 1,000; union of four clocks 146 of 943; the rising-set PRC2 odds ratio; the overlap of 148.
  2. The background has 47,962 CpGs.
  3. The part A ranking is complete: a finite score for every background CpG.
STAGES  --stage stats | r | sf14 | write | all
OUTPUTS artifacts/attribution_stability_2026-11-10/
  - note10_fig5_ranking_2026-11-10.json and .csv
  - note10_gometh_GO_2026-11-10.csv and note10_gometh_KEGG_2026-11-10.csv
  - note10_fgsea_2026-11-10.csv
  - note10_sf14_panel_data_2026-11-10.csv
  - note10_sf14_fig5_ranking_2026-11-10.pdf and .png
  Work files go to artifacts/note10_fig5_ranking_2026-11-10_work/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methylgpt2_lib import ARTIFACTS, log, require, write_json  # noqa: E402

DATE = "2026-11-10"
OUT = ARTIFACTS / f"attribution_stability_{DATE}"
WORK = ARTIFACTS / f"note10_fig5_ranking_{DATE}_work"
MASTER = Path("/Entropy/kying/agent-worktrees/methylgpt-master-finalize-2026-11-10")
ENR = ARTIFACTS / "si_enrichment_2026-10-21"
CHECKS = []
SETS_ORDER = ["PRC2 targets", "TFBS", "CpG island", "CpG shore", "enhancer", "DNase hypersensitivity"]


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


def context():
    se = _load("si_enrichment_attribution", "si_enrichment_attribution_2026-10-21.py")
    fu = _load("si_followup", "si_followup_2026-10-22.py")
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    prc2 = set(l.strip() for l in require(se.PRC2).read_text().split("\n") if l.strip())
    genes = ann.genes.fillna("").astype(str).str.split(";")
    cpg = ann.cpg.to_numpy()
    sets = {"PRC2 targets": genes.map(lambda gs: any(g in prc2 for g in gs)).to_numpy(), "TFBS": se.manifest_tfbs(cpg),
            "CpG island": (ann.island_relation == "Island").to_numpy(), "CpG shore": ann.island_relation.astype(str).str.contains("Shore").to_numpy(),
            "enhancer": ann.enhancer.fillna(False).to_numpy(bool), "DNase hypersensitivity": ann.dnase_hs.fillna(False).to_numpy(bool)}
    island = ann.island_relation.fillna("OpenSea").astype(str).str.replace("N_|S_", "", regex=True).to_numpy()
    clocks = {}
    for name, f in fu.CLOCKS.items():
        c = pd.read_csv(require(fu.CLOCK_DIR / f)).CpGmarker.astype(str)
        clocks[name] = np.isin(cpg, c.to_numpy())
    clocks["union of four clocks"] = np.any(list(clocks.values()), axis=0)
    r_age = np.load(require(ENR / "direction.npz"))["r"]
    gm = np.load(require(ARTIFACTS / "si_attribution_old_young_2026-10-23/group_means.npz"))
    return se, ann, genes, cpg, sets, island, clocks, r_age, gm


def note10(score, bg, se, sets, island, clocks, r_age, young, old):
    """Every Note 10 statistic for one ranking (score) within background bg."""
    v = np.where(bg & np.isfinite(score), score, -np.inf)
    order = np.argsort(-v, kind="stable")
    res = {"background_cpgs": int(bg.sum()), "rankings": {}}
    up = r_age > 0
    tops = {}
    for n in (500, 1000, 2000):
        top = np.zeros(len(v), bool); top[order[:n]] = True
        tops[n] = top
        b = {k: se.fisher(top[bg], s[bg]) for k, s in sets.items()}
        b["PRC2 targets, stratified by CpG-island relation"] = se.mantel_haenszel(top[bg], sets["PRC2 targets"][bg], island[bg])
        b["methylation rises with age"] = {"top": float(np.mean(up[top & bg])), "background": float(np.mean(up[bg & ~top])),
                                           "top_prc2": float(np.mean(up[top & bg & sets["PRC2 targets"]])),
                                           "background_prc2": float(np.mean(up[bg & ~top & sets["PRC2 targets"]])),
                                           "fisher": se.fisher(top[bg], up[bg]),
                                           "top_gain": int((top & bg & up).sum()), "top_lose": int((top & bg & ~up).sum()),
                                           "top_direction_undefined": int((top & bg & ~np.isfinite(r_age)).sum())}
        res["rankings"][str(n)] = b
    top = tops[1000]
    res["clock_overlap_epic_background"] = {k: {**se.fisher(top[bg], s[bg]), "clock_cpgs_in_background": int(s[bg].sum())} for k, s in clocks.items()}
    rise = np.where(bg, np.log((old + 1e-12) / (young + 1e-12)), -np.inf)
    trise = np.zeros(len(v), bool); trise[np.argsort(-rise)[:1000]] = True
    res["rising_top1000"] = {**{k: se.fisher(trise[bg], sets[k][bg]) for k in ("PRC2 targets", "TFBS", "CpG island")},
                             "PRC2 targets, stratified by CpG-island relation": se.mantel_haenszel(trise[bg], sets["PRC2 targets"][bg], island[bg]),
                             "overlap_with_top1000_attribution": int((trise & top).sum())}
    t2 = np.argsort(-np.where(bg & np.isfinite(r_age), np.abs(r_age), -np.inf))[:1000]
    res["overlap_top1000_within_dataset_training_correlation"] = int(len(set(order[:1000]) & set(t2)))
    return res, order, tops, trise


def stage_stats():
    t0 = time.time()
    se, ann, genes, cpg, sets, island, clocks, r_age, gm = context()
    A = pd.read_csv(require(OUT / f"part_a_cpg_age_attribution_{DATE}.csv.gz"))
    assert (A.cpg_id.to_numpy() == cpg).all()
    # check 1: the earlier ranking through the same code
    Az = np.load(require(ENR / "attribution.npz"))
    runs = ("run 1", "seed 2", "seed 3")
    old_score = np.mean([Az[f"score|{r}"] for r in runs], 0)
    old_bg = ann.in_epic_v1.to_numpy() & (np.mean([Az[f"n_observed|{r}"] for r in runs], 0) > 0)
    za = np.load(require(ARTIFACTS / "si_followup_2026-10-22/attribution_age.npz"))
    oy = np.mean([za[f"score|young|{r}"] for r in runs], 0); oo = np.mean([za[f"score|old|{r}"] for r in runs], 0)
    R0, _, _, _ = note10(old_score, old_bg, se, sets, island, clocks, r_age, oy, oo)
    E = json.loads(require(ARTIFACTS / "eval/si_enrichment_attribution_2026-10-21.json").read_text())["rankings"]["MethylGPT attribution"]
    F = json.loads(require(ARTIFACTS / "eval/si_followup_2026-10-22.json").read_text())
    pairs = [(R0["rankings"][n][k]["odds_ratio"], E[n][k]["odds_ratio"], f"{k} top {n}") for n in ("500", "1000", "2000")
             for k in ("TFBS", "PRC2 targets", "PRC2 targets, stratified by CpG-island relation")]
    pairs += [(R0["rankings"]["1000"]["methylation rises with age"]["fisher"]["top_in_set"], E["1000"]["methylation rises with age"]["fisher"]["top_in_set"], "direction gain"),
              (R0["clock_overlap_epic_background"]["union of four clocks"]["top_in_set"], F["A1_clock_overlap"]["union of four clocks"]["top_in_set"], "clock union top 1,000"),
              (R0["clock_overlap_epic_background"]["union of four clocks"]["clock_cpgs_in_background"], 943, "clock CpGs in background"),
              (R0["rising_top1000"]["PRC2 targets"]["odds_ratio"], F["A2_age_specific"]["rising_top1000_enrichment"]["PRC2 targets"]["odds_ratio"], "rising PRC2"),
              (R0["overlap_top1000_within_dataset_training_correlation"], 148, "overlap 148")]
    bad = [p[2] for p in pairs if abs(float(p[0]) - float(p[1])) > 1e-9]
    chk("1 earlier ranking reproduces Note 10 through this code", not bad, f"{len(pairs) - len(bad)} of {len(pairs)} values; mismatches {bad}")
    # the Fig. 5 ranking
    score = A.mean_abs_attribution.to_numpy()
    bg = ann.in_epic_v1.to_numpy() & (A.n_observed.to_numpy() > 0)
    chk("2 background 47,962 CpGs", bg.sum() == 47962, f"{bg.sum()}")
    chk("3 finite score for every background CpG", np.isfinite(score[bg]).all())
    young, old = gm["mean|young"], gm["mean|old"]
    R, order, tops, trise = note10(score, bg, se, sets, island, clocks, r_age, young, old)
    # clock CpGs, 950 vocabulary CpGs, vocabulary background
    clk = A.in_any_of_four_clocks.to_numpy(bool)
    vall = np.where(np.isfinite(score), score, -np.inf)
    oall = np.argsort(-vall, kind="stable")
    R["clock_cpgs_vocabulary"] = {}
    for n in (100, 1000):
        t = np.zeros(len(score), bool); t[oall[:n]] = True
        R["clock_cpgs_vocabulary"][str(n)] = {**se.fisher(t, clk), "clock_cpgs_in_vocabulary": int(clk.sum()), "vocabulary": int(len(clk))}
    # overlaps with other rankings
    ov = {}
    ov["(i) within-dataset correlation with age, training profiles (background)"] = {
        "overlap": R["overlap_top1000_within_dataset_training_correlation"], "universe": int((bg & np.isfinite(r_age)).sum()),
        "expected": 1e6 / int((bg & np.isfinite(r_age)).sum())}
    for lab, col in (("(ii) tissue-adjusted correlation with age, held-out profiles (Fig. 5d)", "rho_age_tissue_adjusted"),
                     ("(iii) correlation with age within whole blood", "rho_age_whole_blood"), ("(reference) correlation with age, all held-out profiles", "rho_age_all")):
        x = np.abs(A[col].to_numpy())
        U = np.isfinite(x) & np.isfinite(score)
        ta = np.argsort(-np.where(U, score, -np.inf), kind="stable")[:1000]
        tb = np.argsort(-np.where(U, x, -np.inf), kind="stable")[:1000]
        ov[lab] = {"overlap": int(len(set(ta) & set(tb))), "universe": int(U.sum()), "expected": 1e6 / int(U.sum()),
                   "clock_cpgs_in_top1000_of_this_ranking": int(clk[tb].sum()), "clock_cpgs_in_top1000_attribution_same_universe": int(clk[ta].sum())}
    rt = np.argsort(-np.where(bg & np.isfinite(r_age), np.abs(r_age), -np.inf))[:1000]
    ov["(i) within-dataset correlation with age, training profiles (background)"]["clock_cpgs_in_top1000_of_this_ranking"] = int(clk[rt].sum())
    R["overlaps"] = ov
    R["earlier_ranking_reproduced"] = R0
    WORK.mkdir(parents=True, exist_ok=True)
    write_json(WORK / "stats.json", R)
    # gene lists for R
    d = WORK / "genesets"; d.mkdir(parents=True, exist_ok=True)
    up = r_age > 0
    t1000 = order[:1000]
    lists = {"top1000": cpg[t1000], "top2000": cpg[order[:2000]], "top1000_gain": cpg[t1000][up[t1000]], "top1000_lose": cpg[t1000][~up[t1000]]}
    for k, v_ in lists.items():
        pd.Series(v_).to_csv(d / f"{k}.txt", index=False, header=False)
    pd.Series(cpg[bg]).to_csv(d / "background.txt", index=False, header=False)
    rows = [(g, score[i]) for i in np.flatnonzero(bg) for g in set(genes[i]) if g]
    pd.DataFrame(rows, columns=["gene", "score"]).groupby("gene").score.mean().to_csv(d / "gene_scores.csv")
    np.savez(WORK / "rank_arrays.npz", score=score, bg=bg, trise=trise, young=young, old=old)
    log(f"stats done ({time.time() - t0:.0f} s)")
    save_checks("stats")


def stage_r():
    t0 = time.time()
    d = WORK / "genesets"
    script = f"""
suppressMessages({{library(missMethyl); library(fgsea); library(msigdbr)}})
d <- "{d}"
bg <- readLines(file.path(d, "background.txt"))
for (tag in c("top1000", "top2000", "top1000_gain", "top1000_lose")) {{
  sig <- readLines(file.path(d, paste0(tag, ".txt")))
  for (col in c("GO", "KEGG")) {{
    r <- gometh(sig.cpg = sig, all.cpg = bg, collection = col, array.type = "EPIC", prior.prob = TRUE)
    r <- r[order(r$P.DE), ]
    write.csv(r, file.path(d, paste0("gometh_", tag, "_", col, ".csv")))
  }}
}}
gs <- read.csv(file.path(d, "gene_scores.csv"))
stats <- setNames(gs$score, gs$gene)
get_sets <- function(coll, sub = NULL) {{
  m <- tryCatch(if (is.null(sub)) msigdbr(species = "Homo sapiens", collection = coll) else msigdbr(species = "Homo sapiens", collection = coll, subcollection = sub),
                error = function(e) if (is.null(sub)) msigdbr(species = "Homo sapiens", category = coll) else msigdbr(species = "Homo sapiens", category = coll, subcategory = sub))
  split(m$gene_symbol, m$gs_name)
}}
colls <- list(Hallmark = c("H", NA), GO_BP = c("C5", "GO:BP"), KEGG_legacy = c("C2", "CP:KEGG_LEGACY"), Reactome = c("C2", "CP:REACTOME"))
set.seed(20261022)
for (nm in names(colls)) {{
  cc <- colls[[nm]]
  sets <- get_sets(cc[1], if (is.na(cc[2])) NULL else cc[2])
  r <- fgseaMultilevel(pathways = sets, stats = stats, scoreType = "pos", minSize = 15, maxSize = 500)
  r <- r[order(r$pval), ]
  r$leadingEdge <- sapply(r$leadingEdge, function(x) paste(head(x, 15), collapse = ";"))
  write.csv(as.data.frame(r), file.path(d, paste0("fgsea_", nm, ".csv")), row.names = FALSE)
}}
"""
    (d / "run.R").write_text(script)
    p = subprocess.run(["Rscript", str(d / "run.R")], capture_output=True, text=True)
    (d / "run.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        sys.exit(f"R failed: {p.stderr[-3000:]}")
    log(f"gometh and fgsea done ({time.time() - t0:.0f} s)")


def stage_sf14():
    """Run the existing Supplementary Fig. 14 script unchanged on inputs of the same names computed on the new ranking."""
    R = json.loads(require(WORK / "stats.json").read_text())
    z = np.load(require(WORK / "rank_arrays.npz"))
    st = WORK / "sf14_stage"
    if st.exists():
        shutil.rmtree(st)
    (st / "figures").mkdir(parents=True)
    for f in ("plot_si_attribution_harmonize_2026-10-22.py", "nature_figure_utils.py", "run_nofooter_2026-11-10.py"):
        shutil.copy2(require(MASTER / "figures" / f), st / "figures" / f)
    # amendment 1: the one changed line (panel d x limit), saved as a new script beside the original
    src = (st / "figures/plot_si_attribution_harmonize_2026-10-22.py").read_text()
    old_line = 'set_domain(axd["d"], xlim=(1.0, 6.0))'
    assert src.count(old_line) == 1
    src = src.replace(old_line, 'set_domain(axd["d"], xlim=(1.0, 6.5))   # 11 Nov 2026: lowest FDR 1.06e-6 on the Fig. 5 ranking')
    (st / "figures/plot_si_attribution_harmonize_2026-10-22.py").write_text(src)
    (HERE / "figures/plot_si_attribution_fig5_ranking_2026-11-10.py").write_text(src)
    a = st / "artifacts"
    (a / "eval").mkdir(parents=True); (a / "external").mkdir(); (a / "si_followup_2026-10-22/genesets").mkdir(parents=True)
    os.symlink(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"), a / "cpg_annotation_2026-10-01.parquet")
    os.symlink(require(ARTIFACTS / "external/BENPORATH_PRC2_TARGETS.txt"), a / "external/BENPORATH_PRC2_TARGETS.txt")
    write_json(a / "eval/si_enrichment_attribution_2026-10-21.json", {"rankings": {"MethylGPT attribution": R["rankings"]}})
    write_json(a / "eval/si_followup_2026-10-22.json", {"A1_clock_overlap": R["clock_overlap_epic_background"],
                                                         "A2_age_specific": {"n_young": 1000, "n_old": 724}})
    np.savez(a / "si_followup_2026-10-22/attribution_age.npz", **{"score|young|fig5": z["young"], "score|old|fig5": z["old"]})
    for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome"):
        shutil.copy2(require(WORK / f"genesets/fgsea_{nm}.csv"), a / f"si_followup_2026-10-22/genesets/fgsea_{nm}.csv")
    env = dict(os.environ, MPLBACKEND="Agg")
    p = subprocess.run([sys.executable, "run_nofooter_2026-11-10.py", "plot_si_attribution_harmonize_2026-10-22.py", "attr"],
                       cwd=st / "figures", capture_output=True, text=True, env=env)
    (WORK / "sf14_run.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        sys.exit(f"Supplementary Fig. 14 script failed: {p.stderr[-3000:]}")
    log("Supplementary Fig. 14 rendered")


def gometh_tables():
    d = WORK / "genesets"
    out, summ = {}, {}
    for col in ("GO", "KEGG"):
        frames = []
        for tag in ("top1000", "top2000", "top1000_gain", "top1000_lose"):
            r = pd.read_csv(d / f"gometh_{tag}_{col}.csv", index_col=0)
            r.insert(0, "term_id", r.index)
            r.insert(0, "set", tag)
            frames.append(r)
            term = "TERM" if "TERM" in r else ("Description" if "Description" in r else r.columns[2])
            summ[f"{tag}|{col}"] = {"cpgs": int(sum(1 for _ in open(d / f"{tag}.txt"))), "terms_tested": int(len(r)), "terms_fdr_0.05": int((r.FDR < 0.05).sum()),
                                    "lowest_fdr": float(r.FDR.min()),
                                    "top10": [{"id": str(x.term_id), "term": str(x[term]), "n": int(x["N"]), "de": int(x["DE"]), "p": float(x["P.DE"]), "fdr": float(x["FDR"])}
                                              for _, x in r.head(10).iterrows()]}
        out[col] = pd.concat(frames, ignore_index=True)
    fg = []
    for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome"):
        r = pd.read_csv(d / f"fgsea_{nm}.csv")
        r.insert(0, "collection", nm)
        fg.append(r)
        summ[f"fgsea|{nm}"] = {"sets_tested": int(len(r)), "sets_fdr_0.05": int((r.padj < 0.05).sum()), "lowest_fdr": float(r.padj.min()),
                               "top10": [{"pathway": str(x.pathway), "size": int(x["size"]), "NES": float(x.NES), "padj": float(x.padj)} for _, x in r.head(10).iterrows()]}
    return out, pd.concat(fg, ignore_index=True), summ


def flat_rows(R, summ):
    rows = []

    def add(analysis, set_, n, f):
        rows.append({"analysis": analysis, "set": set_, "top_n": n, "in_set": f.get("top_in_set"), "set_size": f.get("top"),
                     "background_in_set": f.get("background_in_set"), "background_size": f.get("background"),
                     "percent_set": None if f.get("fraction_top") is None else 100 * f["fraction_top"],
                     "percent_background": None if f.get("fraction_background") is None else 100 * f["fraction_background"],
                     "odds_ratio": f.get("odds_ratio"), "ci95_low": f["ci95"][0] if "ci95" in f else None,
                     "ci95_high": f["ci95"][1] if "ci95" in f else None, "p": f.get("p")})
    for n, b in R["rankings"].items():
        for k in SETS_ORDER + ["PRC2 targets, stratified by CpG-island relation"]:
            add("annotation", k, int(n), b[k])
        add("direction: methylation rises with age", f"gain {b['methylation rises with age']['top_gain']}, lose {b['methylation rises with age']['top_lose']}",
            int(n), b["methylation rises with age"]["fisher"])
    for k, f in R["clock_overlap_epic_background"].items():
        add("clock CpGs, EPIC background (Supplementary Fig. 14a)", k, 1000, f)
    for n, f in R["clock_cpgs_vocabulary"].items():
        add("clock CpGs (950), vocabulary background", "any of the four clocks", int(n), f)
    for k in ("PRC2 targets", "PRC2 targets, stratified by CpG-island relation", "TFBS", "CpG island"):
        add("1,000 CpGs whose attribution rose most from < 30 to >= 60 years", k, 1000, R["rising_top1000"][k])
    for k, o in R["overlaps"].items():
        rows.append({"analysis": "overlap of the top 1,000 by attribution with the top 1,000 by", "set": k, "top_n": 1000, "in_set": o["overlap"],
                     "background_size": o["universe"], "odds_ratio": None, "expected_by_chance": o["expected"],
                     "clock_cpgs_in_top1000_of_ranking": o.get("clock_cpgs_in_top1000_of_this_ranking")})
    for k, s in summ.items():
        rows.append({"analysis": "gometh" if not k.startswith("fgsea") else "fgsea", "set": k, "in_set": s.get("terms_fdr_0.05", s.get("sets_fdr_0.05")),
                     "set_size": s.get("cpgs"), "background_size": s.get("terms_tested", s.get("sets_tested")), "p": s["lowest_fdr"],
                     "note": "in_set = terms at FDR < 0.05; background_size = terms tested; p = lowest FDR"})
    return pd.DataFrame(rows)


def panel_data(R):
    z = np.load(require(WORK / "rank_arrays.npz"))
    ann = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
    rows = []
    for k, f in R["clock_overlap_epic_background"].items():
        rows.append({"panel": "a", "item": k, "top_n": 1000, "in_top": f["top_in_set"], "in_background": f["clock_cpgs_in_background"],
                     "odds_ratio": f["odds_ratio"], "ci95_low": f["ci95"][0], "ci95_high": f["ci95"][1]})
    for n, b in R["rankings"].items():
        for k in ("TFBS", "PRC2 targets, stratified by CpG-island relation", "CpG shore", "enhancer"):
            rows.append({"panel": "b", "item": k, "top_n": int(n), "odds_ratio": b[k]["odds_ratio"], "ci95_low": b[k]["ci95"][0], "ci95_high": b[k]["ci95"][1]})
    young, old = z["young"], z["old"]
    bgc = ann.in_epic_v1.to_numpy() & (young > 0) & (old > 0)
    rows.append({"panel": "c", "item": "CpGs in the hexbin (EPIC v1, attribution > 0 in both groups)", "in_background": int(bgc.sum())})
    d = WORK / "genesets"
    for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome"):
        f = pd.read_csv(d / f"fgsea_{nm}.csv")
        f = f[f.padj < 0.05].sort_values("padj").head(6 if nm == "GO_BP" else 4)
        for _, r in f.iterrows():
            rows.append({"panel": "d", "item": f"{nm}: {r.pathway}", "size": int(r["size"]), "NES": float(r.NES), "padj": float(r.padj)})
    return pd.DataFrame(rows)


def stage_write():
    R = json.loads(require(WORK / "stats.json").read_text())
    tabs, fg, summ = gometh_tables()
    R["gometh_fgsea"] = summ
    files = {}

    def put(df, name):
        df.to_csv(OUT / name, index=False)
        files[name] = {"rows": int(len(df)), "bytes": (OUT / name).stat().st_size}
    put(flat_rows(R, summ), f"note10_fig5_ranking_{DATE}.csv")
    put(tabs["GO"], f"note10_gometh_GO_{DATE}.csv")
    put(tabs["KEGG"], f"note10_gometh_KEGG_{DATE}.csv")
    put(fg, f"note10_fgsea_{DATE}.csv")
    put(panel_data(R), f"note10_sf14_panel_data_{DATE}.csv")
    st = WORK / "sf14_stage/figures"
    for src, dst in (("supplementary_attribution_2026-10-22_nofooter.pdf", f"note10_sf14_fig5_ranking_{DATE}.pdf"),
                     ("supplementary_attribution_2026-10-22_nofooter.png", f"note10_sf14_fig5_ranking_{DATE}.png")):
        shutil.copy2(require(st / src), OUT / dst)
        files[dst] = {"bytes": (OUT / dst).stat().st_size}
    checks = json.loads((WORK / "checks_stats.json").read_text())
    R["checks"] = checks
    R["files"] = files
    R["definitions"] = __doc__
    write_json(OUT / f"note10_fig5_ranking_{DATE}.json", R)
    log(f"written {files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stats", "r", "sf14", "write", "all"], required=True)
    a = ap.parse_args()
    for s in (["stats", "r", "sf14", "write"] if a.stage == "all" else [a.stage]):
        CHECKS.clear()
        {"stats": stage_stats, "r": stage_r, "sf14": stage_sf14, "write": stage_write}[s]()


if __name__ == "__main__":
    main()
