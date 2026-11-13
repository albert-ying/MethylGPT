#!/usr/bin/env python3
"""Export the tables behind the original-figure panels.

The Mac builds the figures; this script only writes tables. No model is run and no existing result file is changed:
tables 1-5 copy existing caches and JSON values and add profile or CpG identifiers; tables 6 and 7 copy the outputs
of si_attribution_old_young_2026-10-23.py (pre-specified, committed before it was run). Table 8 (Generation Scotland
Cox tables) is not exported: the three files are already in the master repository (checked 24 Oct 2026).

Public profiles only: every profile-level table is checked to hold only test-split corpus profiles with GEO sample
(GSM) identifiers from GEO series (GSE) datasets; the script stops otherwise. No Generation Scotland file is read.

Output: artifacts/figure_tables_2026-10-23/*.csv.gz and README.md (columns, row counts, sizes, sources, inputs).
Run: python export_figure_tables_2026-10-23.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from methylgpt_lib import ARTIFACTS, CORPUS_DIR, log, require

HERE = Path(__file__).resolve().parent
EVAL = ARTIFACTS / "eval"
FIG = HERE / "figures"
OUT = ARTIFACTS / "figure_tables_2026-10-23"
ATT = ARTIFACTS / "si_attribution_old_young_2026-10-23"
CLOCKS = {"horvath_multitissue": "Horvath1.csv", "horvath_skin_blood": "Horvath2.csv", "hannum": "Hannum.csv", "phenoage": "PhenoAge.csv"}
CLOCK_DIR = Path("/home/kying/anaconda3/lib/python3.12/site-packages/biolearn/data")
MASTER = Path("/Entropy/kying/agent-worktrees/methylgpt-master-2026-10-11")
GS_FILES = ["artifacts/gs_cox/full_coverage_cindex_table_2026-10-01.csv", "artifacts/gs_cox/stacked_2026-10-04.json",
            "artifacts/gs_cox/paired_bootstrap_mgpt2_combined_2026-10-01_finite_p_2026-10-09.csv"]
SCRIPT = "export_figure_tables_2026-10-23.py"

PROF = pd.read_parquet(require(CORPUS_DIR / "profiles.parquet"))
ANN = pd.read_parquet(require(ARTIFACTS / "cpg_annotation_2026-10-01.parquet"))
RECORD = []   # (file, description, source, inputs, {column: description})


def public_only(rows, what):
    q = PROF.iloc[np.asarray(rows)]
    ok = (q.split == "test").all() and q.id.astype(str).str.match(r"^GSM\d+$").all() and q.group.astype(str).str.match(r"^GSE\d+$").all()
    if not ok:
        raise SystemExit(f"{what}: rows that are not public test-split GEO profiles; not written")
    log(f"{what}: {len(q)} public test-split GEO profiles from {q.group.nunique()} datasets")


def write(df, name, desc, source, inputs, cols):
    missing = set(df.columns) ^ set(cols)
    if missing:
        raise SystemExit(f"{name}: column descriptions do not match columns: {missing}")
    path = OUT / name
    df.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
    RECORD.append((name, desc, source, inputs, cols, len(df), path.stat().st_size))
    log(f"{name}: {len(df)} rows, {path.stat().st_size:,} bytes")


def t1_cpg_umap():
    z = np.load(require(FIG / "figure2_umap_coords_2026-10-02.npz"))
    idx, xy = z["idx"], z["xy"]
    a = ANN.iloc[idx]
    epic = a.in_epic_v1.to_numpy(bool)
    df = pd.DataFrame({"vocab_index": idx, "cpg_id": a.cpg.to_numpy(), "umap1": xy[:, 0], "umap2": xy[:, 1],
                       "island_relation": a.island_relation.to_numpy(), "enhancer": pd.array(np.where(epic, a.enhancer.to_numpy(), None), dtype="boolean"),
                       "chr": a.chr.to_numpy(), "chromosome_x": (a.chr == "X").to_numpy(), "in_epic_v1": epic})
    write(df, "table1_fig2bcd_cpg_umap.csv.gz",
          "Fig. 2b-d. UMAP (umap-learn, cosine, 30 neighbours, min_dist 0.3, seed 0) of the input-layer loadings scaled by the factor standard deviations, for the 15,000 vocabulary CpGs drawn at random (seed 20261002) and plotted in Fig. 2; coordinates copied from the figure cache.",
          "figures/plot_fig2_2026-10-02.py (cache) and this script", ["figures/figure2_umap_coords_2026-10-02.npz", "artifacts/cpg_annotation_2026-10-01.parquet"],
          {"vocab_index": "row of the CpG in the 49,156-CpG vocabulary (0-based)", "cpg_id": "Illumina probe ID",
           "umap1": "UMAP coordinate 1", "umap2": "UMAP coordinate 2",
           "island_relation": "relation to CpG island, EPIC v1.0 B5 manifest (Island, Shore, Shelf, OpenSea); empty for CpGs not on EPIC v1 (Fig. 2 shows them as unannotated)",
           "enhancer": "enhancer flag (450K or FANTOM5), EPIC v1.0 B5 manifest; empty for CpGs not on EPIC v1 (Fig. 2: unannotated)",
           "chr": "chromosome (EPIC v1 manifest, else 450K manifest)", "chromosome_x": "chr == X",
           "in_epic_v1": "CpG on the EPIC v1.0 B5 manifest"})


def t2_setA():
    z = np.load(require(FIG / "figure3_umap_coords_2026-10-02.npz"))
    rows = z["rows"]
    public_only(rows, "set A UMAP")
    if not np.array_equal(rows, np.load(require(EVAL / "representations_setA_2026-10-01.npz"))["rows"]):
        raise SystemExit("figure 3 cache rows differ from the set A representation rows")
    q = PROF.iloc[rows]
    df = pd.DataFrame({"corpus_row": rows, "profile_id": q.id.to_numpy(), "dataset": q.group.to_numpy(), "tissue": q.tissue.to_numpy(),
                       "sex": q.sex.to_numpy(), "umap1_latent": z["fs"][:, 0], "umap2_latent": z["fs"][:, 1],
                       "umap1_raw": z["raw"][:, 0], "umap2_raw": z["raw"][:, 1]})
    write(df, "table2_fig3acdf_setA_umap.csv.gz",
          "Fig. 3a, c, d, f. UMAP coordinates of set A (2,371 test-split profiles, six tissues, 43 GEO datasets): latent scores (input-layer posterior means) and raw beta values, each reduced to 50 principal components then UMAP (30 neighbours, min_dist 0.3, seed 0); coordinates copied from the figure cache.",
          "figures/plot_fig3_2026-10-02.py (cache) and this script",
          ["figures/figure3_umap_coords_2026-10-02.npz", "artifacts/eval/representations_setA_2026-10-01.npz (row check)", "artifacts/corpus_2026-09-30/profiles.parquet"],
          {"corpus_row": "row in the corpus profile table", "profile_id": "GEO sample (GSM) ID", "dataset": "GEO series (GSE)",
           "tissue": "tissue label", "sex": "sex label (F, M; empty if unlabelled)",
           "umap1_latent": "UMAP 1, latent scores", "umap2_latent": "UMAP 2, latent scores",
           "umap1_raw": "UMAP 1, raw beta values", "umap2_raw": "UMAP 2, raw beta values"})
    res = json.loads(require(EVAL / "representation_benchmark_2026-10-01.json").read_text())
    rec = []
    for geom in ("natural", "standardized"):
        for rep, v in res[geom]["set_a"].items():
            rec.append({"geometry": geom, "representation": rep, **{k: v.get(k) for k in ("tissue_knn_bacc", "sex_knn_bacc", "tissue_silhouette", "tissue_energy_distance")}})
    write(pd.DataFrame(rec), "table2_fig3acdf_setA_knn_metrics.csv.gz",
          "Fig. 3c. Set A metrics per representation, copied from the 2026-10-01 representation benchmark (stratified 5-fold 10-NN, cosine).",
          "benchmark_representations_2026-10-01.py (values copied by this script)", ["artifacts/eval/representation_benchmark_2026-10-01.json"],
          {"geometry": "natural (centred; primary) or standardized (per-feature z-scores; sensitivity)",
           "representation": "raw (beta values), raw_pca50, factor_scores (latent scores), mgpt2_selfsupervised, mgpt2_multitask (embedding), untrained_transformer, random_projection",
           "tissue_knn_bacc": "tissue 10-NN balanced accuracy", "sex_knn_bacc": "sex 10-NN balanced accuracy",
           "tissue_silhouette": "tissue silhouette (cosine, 50 PCs)", "tissue_energy_distance": "energy distance between tissues"})


def t3_setB():
    z = np.load(require(EVAL / "si_sample_representations_umap_setB_2026-10-18.npz"))
    rows = z["rows"]
    public_only(rows, "set B UMAP")
    q = PROF.iloc[rows]
    if not (q.group.to_numpy().astype(str) == z["dataset"].astype(str)).all():
        raise SystemExit("set B cache datasets differ from the profile table")
    df = pd.DataFrame({"corpus_row": rows, "profile_id": q.id.to_numpy(), "dataset": z["dataset"], "age": z["age"], "sex": z["sex"],
                       "umap1_latent": z["latent"][:, 0], "umap2_latent": z["latent"][:, 1], "umap1_raw": z["raw"][:, 0], "umap2_raw": z["raw"][:, 1]})
    write(df, "table3_fig3be_setB_umap.csv.gz",
          "Fig. 3b, e. UMAP coordinates of set B (949 whole-blood test-split profiles with sex and metadata age, 10 GEO datasets): latent scores and raw beta values (PCA-50 then UMAP, 30 neighbours, min_dist 0.3, seed 0); copied from the cache.",
          "si_sample_representations_2026-10-18.py (cache) and this script",
          ["artifacts/eval/si_sample_representations_umap_setB_2026-10-18.npz", "artifacts/corpus_2026-09-30/profiles.parquet"],
          {"corpus_row": "row in the corpus profile table", "profile_id": "GEO sample (GSM) ID", "dataset": "GEO series (GSE)",
           "age": "metadata age (years)", "sex": "sex label (F, M)",
           "umap1_latent": "UMAP 1, latent scores", "umap2_latent": "UMAP 2, latent scores",
           "umap1_raw": "UMAP 1, raw beta values", "umap2_raw": "UMAP 2, raw beta values"})
    c = pd.read_csv(require(EVAL / "si_sample_representations_2026-10-18.csv"))
    c = c[(c.set == "B") & c.metric.isin(["dataset 10-NN balanced accuracy", "dataset iLISI (median)"])].drop(columns="set")
    write(c.reset_index(drop=True), "table3_fig3be_setB_dataset_mixing.csv.gz",
          "Fig. 3b, e. Set B dataset 10-NN balanced accuracy (stratified 5-fold) and median dataset iLISI per representation, copied from Supplementary Table 3 rows (intervals from resampling the 10 datasets, 2,000 draws).",
          "si_sample_representations_2026-10-18.py (rows copied by this script)", ["artifacts/eval/si_sample_representations_2026-10-18.csv"],
          {"geometry": "natural (primary) or standardized", "representation": "representation (factor_scores = latent scores; mgpt2_multitask = embedding; combat*, harmony_pca50 = batch-corrected references)",
           "metric": "dataset 10-NN balanced accuracy, or dataset iLISI (median)", "cv": "cross-validation scheme (none for iLISI)",
           "value": "metric value", "ci95_low": "2.5th percentile, dataset resampling", "ci95_high": "97.5th percentile, dataset resampling",
           "n_profiles": "profiles", "n_datasets": "datasets", "note": "note from the source table"})


def t4_pca():
    z = np.load(require(EVAL / "si_embedding_pretraining_pca_2026-10-18.npz"), allow_pickle=True)
    rows = z["rows"]
    public_only(rows, "held-out PCA")
    q = PROF.iloc[rows]
    if not (q.group.to_numpy().astype(str) == z["dataset"].astype(str)).all():
        raise SystemExit("PCA cache datasets differ from the profile table")
    enc = {"ssl": "selfsupervised", "age_stage": "age_stage", "nopretrain": "nopretrain"}
    df = pd.DataFrame({"corpus_row": rows, "profile_id": q.id.to_numpy(), "dataset": z["dataset"].astype(str), "age": z["age"], "tissue": z["tissue"].astype(str)})
    for k, lab in enc.items():
        df[f"pc1_{lab}"] = z[k][:, 0]; df[f"pc2_{lab}"] = z[k][:, 1]
    cols = {"corpus_row": "row in the corpus profile table", "profile_id": "GEO sample (GSM) ID", "dataset": "GEO series (GSE)",
            "age": "metadata age (years)", "tissue": "tissue label ('no tissue label' if missing)"}
    for lab, txt in (("selfsupervised", "self-supervised encoder (runs/stopped_cf_dense_2026-09-30/best.pt)"),
                     ("age_stage", "age-stage encoder, seed 20261001 (runs/aging_heads_clean_2026-10-01/best.pt)"),
                     ("nopretrain", "model without pretraining: random input-layer loadings and transformer, same age, sex and tissue stage (runs/aging_heads_nopretrain_lr5e-5/best.pt)")):
        cols[f"pc1_{lab}"] = f"PC1 of the 2,048-dimensional embedding, {txt}"; cols[f"pc2_{lab}"] = f"PC2, {txt}"
    write(df, "table4_fig4bcd_heldout_pca.csv.gz",
          "Fig. 4b-d. First two principal components of the sample embedding for the 5,811 test-split profiles with metadata age (34 GEO datasets); PCA fitted separately on each encoder's embeddings of these profiles; coordinates copied from the cache.",
          "si_embedding_pretraining_2026-10-18.py (cache) and this script",
          ["artifacts/eval/si_embedding_pretraining_pca_2026-10-18.npz", "artifacts/corpus_2026-09-30/profiles.parquet"], cols)
    j = json.loads(require(EVAL / "si_embedding_pretraining_2026-10-18.json").read_text())["heldout_pca"]
    rec = [{"encoder": lab, "pc": i + 1, "explained_variance_ratio": j["explained_variance_ratio"][k][i],
            "spearman_with_age": float(z[f"{k}_pc_age_spearman"][i])} for k, lab in enc.items() for i in range(2)]
    write(pd.DataFrame(rec), "table4_fig4bcd_pca_variance.csv.gz",
          "Fig. 4b-d. Variance explained by PC1 and PC2 of each encoder and the Spearman correlation of each PC with age (5,811 profiles), copied.",
          "si_embedding_pretraining_2026-10-18.py (values copied by this script)",
          ["artifacts/eval/si_embedding_pretraining_2026-10-18.json (heldout_pca)", "artifacts/eval/si_embedding_pretraining_pca_2026-10-18.npz"],
          {"encoder": "selfsupervised, age_stage or nopretrain (as table4_fig4bcd_heldout_pca)", "pc": "principal component (1 or 2)",
           "explained_variance_ratio": "fraction of the embedding's total variance explained by the PC",
           "spearman_with_age": "Spearman correlation of the PC with metadata age"})


def clock_membership():
    cpg = ANN.cpg.to_numpy()
    out = {}
    for k, f in CLOCKS.items():
        out[f"in_{k}"] = np.isin(cpg, pd.read_csv(require(CLOCK_DIR / f)).CpGmarker.astype(str).to_numpy())
    out["in_any_of_four_clocks"] = np.any(list(out.values()), axis=0)
    return out


def t5_attribution():
    za = np.load(require(ARTIFACTS / "si_followup_2026-10-22/attribution_age.npz"))
    runs = [k.split("|", 2)[2] for k in za.files if k.startswith("score|young|")]
    gm = np.load(require(ATT / "group_means.npz"))
    A = np.load(require(ARTIFACTS / "si_enrichment_2026-10-21/attribution.npz"))
    runs_all = [k.split("|", 1)[1] for k in A.files if k.startswith("score|")]
    df = pd.DataFrame({"cpg_id": ANN.cpg.to_numpy(), "genes": ANN.genes.fillna("").to_numpy(), "island_relation": ANN.island_relation.to_numpy(),
                       "in_epic_v1": ANN.in_epic_v1.to_numpy(), **clock_membership()})
    df["mean_abs_attribution_heldout2000"] = np.mean([A[f"score|{r}"] for r in runs_all], axis=0)
    df["mean_abs_attribution_young_lt30"] = np.mean([za[f"score|young|{r}"] for r in runs], axis=0)
    df["n_observed_young_lt30"] = gm["n_observed|young"]
    df["mean_abs_attribution_old_ge60"] = np.mean([za[f"score|old|{r}"] for r in runs], axis=0)
    df["n_observed_old_ge60"] = gm["n_observed|old"]
    n3 = {}
    for g, lab in (("lt20", "lt20"), ("20to60", "20to60"), ("ge60", "ge60")):
        df[f"mean_abs_attribution_{lab}"] = gm[f"mean|{g}"]; df[f"n_observed_{lab}"] = gm[f"n_observed|{g}"]
        n3[lab] = int(gm[f"n_profiles|{g}"])
    cols = {"cpg_id": "Illumina probe ID", "genes": "UCSC RefGene names (EPIC v1.0 B5 manifest; ';'-separated, empty if none)",
            "island_relation": "relation to CpG island (EPIC v1 manifest; empty if not on EPIC v1)", "in_epic_v1": "CpG on the EPIC v1.0 B5 manifest",
            "in_horvath_multitissue": "CpG of the Horvath multi-tissue clock (BioLearn Horvath1.csv)", "in_horvath_skin_blood": "CpG of the Horvath skin and blood clock (Horvath2.csv)",
            "in_hannum": "CpG of the Hannum clock (Hannum.csv)", "in_phenoage": "CpG of PhenoAge (PhenoAge.csv)", "in_any_of_four_clocks": "in any of the four clocks",
            "mean_abs_attribution_heldout2000": "mean absolute attribution over 2,000 held-out profiles (si_enrichment_2026-10-21; the ranking used for the top-1,000 CpG analyses), mean of three runs",
            "mean_abs_attribution_young_lt30": "mean absolute attribution, 1,000 held-out profiles younger than 30 years (attribution_age.npz), mean of three runs",
            "n_observed_young_lt30": "young profiles in which the CpG is observed",
            "mean_abs_attribution_old_ge60": "mean absolute attribution, 724 held-out profiles aged 60 or older (attribution_age.npz), mean of three runs",
            "n_observed_old_ge60": "old profiles in which the CpG is observed",
            "mean_abs_attribution_lt20": f"mean absolute attribution, held-out profiles younger than 20 years (n = {n3['lt20']:,}), mean of three runs",
            "n_observed_lt20": "profiles <20 years in which the CpG is observed",
            "mean_abs_attribution_20to60": f"mean absolute attribution, held-out profiles aged 20 to under 60 years (n = {n3['20to60']:,}), mean of three runs",
            "n_observed_20to60": "profiles 20 to under 60 years in which the CpG is observed",
            "mean_abs_attribution_ge60": f"mean absolute attribution, held-out profiles aged 60 or older (n = {n3['ge60']:,}; the same profiles as the old group), mean of three runs",
            "n_observed_ge60": "profiles >=60 years in which the CpG is observed"}
    write(df, "table5_fig5bd_cpg_attribution.csv.gz",
          f"Fig. 5b, d. Per-CpG gradient x input attribution of the released age output (mean absolute attribution over the profiles in which the CpG is observed, averaged over the three released runs), all 49,156 vocabulary CpGs. Young and old columns from si_followup_2026-10-22 attribution_age.npz; the three age groups (<20: {n3['lt20']:,}; 20 to <60: {n3['20to60']:,}; >=60: {n3['ge60']:,} profiles, all 5,811 held-out profiles with metadata age) are a new computation by si_attribution_old_young_2026-10-23.py, which also reproduces the young and old columns (see README checks).",
          "si_followup_2026-10-22.py, si_enrichment_attribution_2026-10-21.py, si_attribution_old_young_2026-10-23.py; assembled by this script",
          ["artifacts/si_followup_2026-10-22/attribution_age.npz", "artifacts/si_enrichment_2026-10-21/attribution.npz",
           "artifacts/si_attribution_old_young_2026-10-23/group_means.npz", "artifacts/cpg_annotation_2026-10-01.parquet", "BioLearn clock coefficient files"], cols)


def t6_t7():
    tab = pd.read_csv(require(ATT / "old_young_cpg.csv.gz"))
    if not np.array_equal(tab.cpg_id.to_numpy(), ANN.cpg.to_numpy()):
        raise SystemExit("table 6 CpG order differs from the annotation")
    cols = {"cpg_id": "Illumina probe ID", "n_young_observed": "young (<30 years) profiles in which the CpG is observed (of 1,000)",
            "n_old_observed": "old (>=60 years) profiles in which the CpG is observed (of 724)",
            "mean_abs_attribution_young": "mean absolute attribution, young, mean of three runs", "mean_abs_attribution_old": "mean absolute attribution, old, mean of three runs",
            "log2_ratio_old_young": "log2(mean old / mean young); empty if not tested",
            "mannwhitney_u_old_vs_young": "Mann-Whitney U statistic of the old profiles' values against the young",
            "p_mannwhitney": "two-sided Mann-Whitney P across profiles (asymptotic, tie and continuity correction)",
            "q_bh": "Benjamini-Hochberg q over the tested CpGs (PRIMARY: count of q < 0.05)",
            "tested": "observed in at least 10 young and 10 old profiles"}
    for T, key in (("Whole Blood", "whole_blood"), ("Brain", "brain")):
        cols[f"{key}_n_young_observed"] = f"secondary, {T} only: young profiles observed"
        cols[f"{key}_n_old_observed"] = f"secondary, {T} only: old profiles observed"
        cols[f"{key}_log2_ratio_old_young"] = f"secondary, {T} only: log2 ratio"
        cols[f"{key}_p_mannwhitney"] = f"secondary, {T} only: Mann-Whitney P"
        cols[f"{key}_q_bh"] = f"secondary, {T} only: BH q over the CpGs tested in {T}"
    write(tab, "table6_fig5c_old_young_attribution.csv.gz",
          "Fig. 5c (new computation, pre-specified). For each vocabulary CpG, old (>=60 years, 724) against young (<30 years, 1,000) held-out profiles: per-profile absolute attribution averaged over the three runs; log2 ratio of the group means, two-sided Mann-Whitney P across profiles, Benjamini-Hochberg q; secondary columns repeat the test within Whole Blood (783 young, 116 old) and Brain (103, 256).",
          "si_attribution_old_young_2026-10-23.py (stage test; copied by this script)",
          ["artifacts/si_attribution_old_young_2026-10-23/old_young_cpg.csv.gz", "artifacts/si_attribution_old_young_2026-10-23/per_profile_abs_attribution.npy (not committed)"], cols)
    d = ATT / "genesets"
    parts = []
    for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome"):
        r = pd.read_csv(require(d / f"fgsea_{nm}.csv"))
        r.insert(1, "collection", nm)
        parts.append(r)
    f = pd.concat(parts, ignore_index=True).rename(columns={"pval": "p", "padj": "fdr", "leadingEdge": "leading_edge_genes"})
    f = f[["pathway", "collection", "NES", "ES", "p", "fdr", "log2err", "size", "leading_edge_genes"]]
    write(f, "table7_fig5e_fgsea_old_minus_young.csv.gz",
          "Fig. 5e (new computation, pre-specified). fgsea (fgseaMultilevel, scoreType std, minSize 15, maxSize 500, seed 20261022) on genes ranked by the mean over their CpGs of the old-minus-young mean absolute attribution; MSigDB Hallmark, GO biological process, KEGG legacy, Reactome (msigdbr). Positive NES: attribution higher in old profiles.",
          "si_attribution_old_young_2026-10-23.py (stage genesets; copied by this script)",
          [f"artifacts/si_attribution_old_young_2026-10-23/genesets/fgsea_{nm}.csv" for nm in ("Hallmark", "GO_BP", "KEGG_legacy", "Reactome")],
          {"pathway": "MSigDB gene set name", "collection": "Hallmark, GO_BP, KEGG_legacy or Reactome", "NES": "normalized enrichment score",
           "ES": "enrichment score", "p": "fgsea P", "fdr": "Benjamini-Hochberg adjusted P within the collection (fgsea padj; PRIMARY: Hallmark sets with fdr < 0.05)",
           "log2err": "fgsea log2 error of the P estimate", "size": "genes of the set among the ranked genes",
           "leading_edge_genes": "leading-edge genes (';'-separated, all)"})
    g = pd.read_csv(require(d / "gene_scores.csv"))
    write(g, "table7_fig5e_gene_scores.csv.gz",
          "Fig. 5e. Gene-level statistic ranked by fgsea: mean over the gene's CpGs of the old-minus-young mean absolute attribution.",
          "si_attribution_old_young_2026-10-23.py (stage genesets; copied by this script)", ["artifacts/si_attribution_old_young_2026-10-23/genesets/gene_scores.csv"],
          {"gene": "RefGene name", "score": "mean old-minus-young mean absolute attribution over the gene's CpGs", "n_cpgs": "CpGs mapped to the gene"})


def gs_check():
    import subprocess
    out = []
    for f in GS_FILES:
        tracked = subprocess.run(["git", "-C", str(MASTER), "ls-files", "--error-unmatch", f], capture_output=True).returncode == 0
        same = (MASTER / f).exists() and (ARTIFACTS.parent / f).exists() and (MASTER / f).read_bytes() == (ARTIFACTS.parent / f).read_bytes()
        out.append((f, (MASTER / f).exists(), tracked, same))
    return out


def readme(res):
    p6, p7 = res["primary_table6"], res["primary_table7"]
    s6 = res["secondary_table6"]; wt = res["secondary_within_tissue"]; s7 = res["secondary_table7"]
    ch = res["checks"]
    L = ["# Tables behind the original-figure panels (MethylGPT revision)", "",
         "Written by `export_figure_tables_2026-10-23.py` on 24 Oct 2026 for the Mac figure builds "
         "(specification of 23 Oct 2026). Tables only: no figure, legend or manuscript text, no model weights.",
         "Profile-level tables (2, 3, 4) hold only test-split corpus profiles from public GEO series (GSM sample IDs in GSE datasets); "
         "the export script checks this before writing. No Generation Scotland participant-level value is in any table.", "",
         "## Files", ""]
    for name, desc, source, inputs, cols, n, size in RECORD:
        L += [f"### {name}", "", desc, "", f"- Rows: {n:,}; size: {size:,} bytes", f"- Source script: {source}",
              "- Inputs: " + "; ".join(f"`{i}`" for i in inputs), "- Columns:"]
        L += [f"  - `{c}`: {t}" for c, t in cols.items()]
        L.append("")
    L += ["## Table 8 (Fig. 6c, d)", "",
          "Not exported. The three Generation Scotland Cox tables are already tracked in the master checkout "
          f"`{MASTER}` and are byte-identical to the working-repo copies:"]
    for f, exists, tracked, same in gs_check():
        L.append(f"- `{f}`: present {exists}, tracked {tracked}, identical {same}")
    L += ["", "## Pre-specified results of tables 6 and 7", "",
          "Design, primary metrics and multiple-testing correction are in the docstring of `si_attribution_old_young_2026-10-23.py`, "
          "committed before it was run. Full results: `artifacts/eval/si_attribution_old_young_2026-10-23.json`.", "",
          f"- Table 6 primary: {p6['q_lt_0.05']:,} of {p6['tested']:,} tested CpGs at BH q < 0.05 (old >= 60 years, n = {res['n_old']}, against young < 30 years, n = {res['n_young']}); "
          f"{p6['q_lt_0.05_higher_in_old']:,} higher in old, {p6['q_lt_0.05_higher_in_young']:,} higher in young.",
          f"- Table 6 secondary: median log2 ratio {s6['median_log2_ratio_tested']:.3f}; Spearman correlation of young and old CpG means {s6['spearman_young_old_means_tested']:.3f}."]
    for T, v in wt.items():
        L.append(f"- Table 6 secondary, {T} only (young {v['n_young']}, old {v['n_old']}): {v['q_lt_0.05']:,} of {v['tested']:,} CpGs at q < 0.05 "
                 f"({v['q_lt_0.05_higher_in_old']:,} higher in old, {v['q_lt_0.05_higher_in_young']:,} higher in young); Spearman correlation of log2 ratios with the pooled ratios {v['spearman_log2_ratio_with_pooled']:.3f} over {v['cpgs_in_correlation']:,} CpGs.")
    u = s6["top1000_log2_ratio_clock_overlap"]["union of four clocks"]
    L.append(f"- Table 6 secondary: the 1,000 CpGs with the largest log2 ratio hold {u['top_in_set']} CpGs of the four clocks (odds ratio {u['odds_ratio']:.2f}, 95% interval {u['ci95'][0]:.2f}-{u['ci95'][1]:.2f}, Fisher P {u['p']:.3g}).")
    L.append(f"- Table 7 primary: {p7['fdr_0.05']} of {p7['tested']} Hallmark gene sets at FDR < 0.05 ({p7['NES_positive']} with positive NES, {p7['NES_negative']} negative).")
    for nm, v in s7.items():
        r = pd.read_csv(require(ATT / "genesets" / f"fgsea_{nm}.csv"))
        top = r.iloc[0]
        L.append(f"- Table 7, {nm}: {v['fdr_0.05']} of {v['tested']} sets at FDR < 0.05 ({v['fdr_0.05_NES_positive']} positive, {v['fdr_0.05_NES_negative']} negative NES); "
                 f"{int(r.padj.isna().sum())} sets without a P value (fgsea: unbalanced positive and negative gene statistics; left empty, counted as not significant); "
                 f"smallest P: {top.pathway} (NES {top.NES:.2f}, P {top.pval:.2g}, FDR {top.padj:.2g}).")
    dg = res["genesets_diagnostics"]
    L.append(f"- Table 7 inputs: {dg['background_cpgs_used']:,} CpGs mapped to {dg['genes']:,} genes; {dg['genes_score_positive']:,} genes have a positive old-minus-young score.")
    L += ["", "Limits stated before running: P values treat profiles (and genes) as independent; the young and old profiles come from "
          f"{res['datasets_young']} and {res['datasets_old']} datasets ({res['datasets_both']} in both) and differ in tissue "
          "(young mostly whole blood; old mostly brain, colon and other tissues), so the pooled q values are not tests of age adjusted for dataset or tissue. "
          "The within-tissue rows are the pre-specified secondary for this.", "",
          "## Checks", ""]
    for k, v in ch.items():
        L.append(f"- {k}: {v:.3g}" + (" years (predicted age against the saved held-out predictions)" if "prediction" in k else ""))
    L += ["", "## Deviations and notes", "",
          "- Table 5: the three age-group columns (<20, 20 to <60, >=60) are not in `attribution_age.npz`; they were computed by the new pre-specified script with the same attribution code. "
          "The specification writes '>60'; age 60 was placed in the oldest group so the groups are exhaustive and the oldest group is the old group (52 profiles are exactly 60).",
          "- Table 6 needs per-profile values for the Mann-Whitney test, which `attribution_age.npz` does not store, so attribution was recomputed on GPU (same profiles, batches and code); the checks above compare it with the stored scores.",
          "- Table 7 uses fgsea scoreType 'std' instead of the 'pos' of si_followup_2026-10-22, because the old-minus-young difference is signed; this was fixed in the docstring before running. "
          "Leading-edge genes are given in full (si_followup kept the first 15).",
          "- Table 4: only the three encoders named in the specification are exported; the npz also holds the random-initialisation control (randominit).",
          "- Table 1: enhancer is left empty for the 1,194 vocabulary CpGs that are not on EPIC v1 (the annotation file stores False for them; Fig. 2 shows them as unannotated).", ""]
    (OUT / "README.md").write_text("\n".join(L))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t1_cpg_umap(); t2_setA(); t3_setB(); t4_pca(); t5_attribution(); t6_t7()
    res = json.loads(require(EVAL / "si_attribution_old_young_2026-10-23.json").read_text())
    readme(res)
    log("README written")


if __name__ == "__main__":
    main()
