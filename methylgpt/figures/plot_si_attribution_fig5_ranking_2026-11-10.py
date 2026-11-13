"""Supplementary Figures for the attribution and array-harmonization analyses (22 October 2026).

attr  What MethylGPT's age output relies on
      ../artifacts/eval/si_enrichment_attribution_2026-10-21.json   annotation enrichment of the top attribution CpGs
      ../artifacts/eval/si_followup_2026-10-22.json                 clock-CpG overlap, age-specific attribution, gene sets
      ../artifacts/si_followup_2026-10-22/attribution_age.npz       attribution in young and old held-out profiles
      ../artifacts/si_followup_2026-10-22/genesets/fgsea_*.csv      rank-based gene-set tests
harm  Representations of the same profile across array generations
      ../artifacts/eval/si_followup_2026-10-22.json
Run: MPLBACKEND=Agg python plot_si_attribution_harmonize_2026-10-22.py [attr] [harm], and again with NATURE_FIGURE_NOFOOTER=1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, save_figure, set_domain,
                                 pad_categorical_axis, ref_line, add_colorbar, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_BLUE)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()


def forest(ax, rows, color, label=None, marker="o", dy=0.0):
    for i, (lab, f) in enumerate(rows):
        ax.plot(f["ci95"], [i + dy] * 2, color=color, lw=0.9, solid_capstyle="butt", zorder=2)
        ax.scatter([f["odds_ratio"]], [i + dy], s=16, marker=marker, color=color, edgecolors="none", zorder=3, clip_on=False,
                   label=label if i == 0 else None)


def attr():
    E = json.loads((A / "eval/si_enrichment_attribution_2026-10-21.json").read_text())["rankings"]["MethylGPT attribution"]
    F = json.loads((A / "eval/si_followup_2026-10-22.json").read_text())
    ann = pd.read_parquet(A / "cpg_annotation_2026-10-01.parquet")
    za = np.load(A / "si_followup_2026-10-22/attribution_age.npz")
    runs = sorted({k.split("|")[2] for k in za.files if k.startswith("score|")})
    young = np.mean([za[f"score|young|{r}"] for r in runs], axis=0); old = np.mean([za[f"score|old|{r}"] for r in runs], axis=0)
    bg = ann.in_epic_v1.to_numpy() & (young > 0) & (old > 0)
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.70))
    # manual positions: the long pathway names of d would otherwise narrow the top row under constrained layout
    axd = {"a": fig.add_axes([0.19, 0.70, 0.22, 0.22]), "b": fig.add_axes([0.53, 0.70, 0.21, 0.22]),
           "c": fig.add_axes([0.81, 0.70, 0.145, 0.22]), "d": fig.add_axes([0.44, 0.08, 0.53, 0.50])}

    # a: clock CpGs among the top 1,000
    ax = axd["a"]
    A1 = F["A1_clock_overlap"]
    order = ["Hannum", "Horvath skin and blood", "Horvath multi-tissue", "PhenoAge", "union of four clocks"]
    rows = [(f"{k} ({A1[k]['top_in_set']}/{A1[k]['clock_cpgs_in_background']})", A1[k]) for k in order]
    forest(ax, rows, RUST["400"])
    ax.set_xscale("log"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0].replace("union of four clocks", "Any of the four") for r in rows])
    ax.set_xlabel("Odds ratio, top 1,000 CpGs")
    ax.set_title("CpGs of published clocks", fontsize=7, fontweight="normal")

    # b: annotations among the top 500 / 1,000 / 2,000
    ax = axd["b"]
    sets = ["TFBS", "PRC2 targets, stratified by CpG-island relation", "CpG shore", "enhancer"]
    labs = {"TFBS": "TF binding sites", "PRC2 targets, stratified by CpG-island relation": "PRC2 (island strata)",
            "CpG shore": "CpG shores", "enhancer": "Enhancers"}
    for n, c, mk, dy in (("500", RUST["200"], "D", -0.22), ("1000", RUST["400"], "o", 0.0), ("2000", RUST["600"], "s", 0.22)):
        forest(ax, [(labs[s], E[n][s]) for s in sets], c, label=f"Top {int(n):,}", marker=mk, dy=dy)
    ax.set_yticks(range(len(sets))); ax.set_yticklabels([labs[s] for s in sets])
    ax.set_xlabel("Odds ratio")
    ax.set_title("Genomic annotation", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="upper right")

    # c: attribution in young and old profiles
    ax = axd["c"]
    a2 = F["A2_age_specific"]
    ly, lo = np.log10(young[bg]), np.log10(old[bg])
    hb = ax.hexbin(ly, lo, gridsize=45, bins="log", cmap=CMAP_BLUE, mincnt=1, linewidths=0, rasterized=True, clip_on=False)
    rise = np.log(old / young); rise = np.where(bg, rise, -np.inf)
    top = np.argsort(-rise)[:1000]
    genes = ann.genes.fillna("").astype(str).str.split(";")
    prc2 = set(l.strip() for l in (A / "external/BENPORATH_PRC2_TARGETS.txt").read_text().split("\n") if l.strip())
    isp = np.array([any(x in prc2 for x in genes[i]) for i in top])
    ax.scatter(np.log10(young[top][isp]), np.log10(old[top][isp]), s=6, color=RUST["400"], edgecolors="none", zorder=3, clip_on=False,
               label=f"PRC2-target CpGs among the 1,000\nwith the largest rise ({isp.sum()})")
    lim = [min(ly.min(), lo.min()), max(ly.max(), lo.max())]
    ax.plot(lim, lim, ls=(0, (2, 2)), color=GRAY["400"], lw=0.7, zorder=2)
    ax.set_xlabel("Age < 30 (log10)"); ax.set_ylabel("Age ≥ 60 (log10)")
    ax.set_title(f"Attribution ({a2['n_young']:,} young, {a2['n_old']:,} old)", fontsize=7, fontweight="normal")
    add_colorbar(hb, ax, label="CpGs", inset=(1.05, 0.08, 0.05, 0.55))

    # d: rank-based gene-set tests over all genes
    ax = axd["d"]
    frames = []
    for nm, lab, c in (("Hallmark", "Hallmark", GOLD["400"]), ("GO_BP", "GO biological process", RUST["400"]), ("KEGG_legacy", "KEGG", BLUE["400"]),
                       ("Reactome", "Reactome", SAGE["400"])):
        f = pd.read_csv(A / f"si_followup_2026-10-22/genesets/fgsea_{nm}.csv")
        f = f[f.padj < 0.05].sort_values("padj").head(6 if nm == "GO_BP" else 4)
        f["collection"], f["color"] = lab, c
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    yy = np.arange(len(f))[::-1]
    for y_, (_, r) in zip(yy, f.iterrows()):
        ax.scatter(-np.log10(r.padj), y_, s=10 + 4 * r["size"] ** 0.5, color=r.color, edgecolors="none", zorder=3, clip_on=False)
    FIX = {"cams": "(CAMs)", "Gpcr": "GPCR", "Class a 1": "Class A/1",
           "Immune response regulating cell surface receptor signaling pathway": "Immune response-regulating receptor signalling",
           "Immunoregulatory interactions between a lymphoid and a non lymphoid cell": "Lymphoid-non-lymphoid immunoregulatory interactions"}

    def name(p):
        t = p.split("_", 1)[1].replace("_", " ").capitalize()
        for a_, b_ in FIX.items():
            t = t.replace(a_, b_)
        return t[:60]
    ax.set_yticks(yy); ax.set_yticklabels([name(p) for p in f.pathway], fontsize=7)
    ax.set_xlabel("−log10 FDR (fgsea, gene-level attribution)")
    for lab, c in (("Hallmark", GOLD["400"]), ("GO biological process", RUST["400"]), ("KEGG", BLUE["400"]), ("Reactome", SAGE["400"])):
        ax.scatter([], [], s=16, color=c, label=lab, clip_on=False)
    nature_legend(ax, loc="lower right", title="Collection")


    trim_axes(fig)
    for k in ("a", "b"):
        pad_categorical_axis(axd[k], "y", pad=0.6)
        ref_line(axd[k], x=1)
    set_domain(axd["b"], xlim=(0.5, 2.0))
    axd["b"].set_xticks([0.5, 1.0, 1.5, 2.0])
    pad_categorical_axis(axd["d"], "y", pad=0.6)
    set_domain(axd["d"], xlim=(1.0, 6.5))   # 11 Nov 2026: lowest FDR 1.06e-6 on the Fig. 5 ranking
    axd["d"].set_xticks([1, 2, 3, 4, 5, 6])
    ref_line(axd["d"], x=-np.log10(0.05))
    add_panel_labels(fig, axd)
    save_figure(fig, str(HERE / "supplementary_attribution_2026-10-22.pdf"), axes_dict=axd,
                review_png=str(HERE / "supplementary_attribution_2026-10-22_review.png"),
                footer="data: held-out test datasets; artifacts/eval/si_enrichment_attribution_2026-10-21.json, si_followup_2026-10-22.json")
    plt.close(fig)


def harm():
    F = json.loads((A / "eval/si_followup_2026-10-22.json").read_text())["H_harmonization"]
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.34), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    axd = {"a": fig.add_subplot(gs[0]), "b": fig.add_subplot(gs[1])}
    sc = [("hm450|to_complete", "450K"), ("epic_v2|to_complete", "EPIC v2"), ("hm27|to_complete", "27K"), ("hm27|to_epic_v2", "27K → EPIC v2"),
          ("hm450|to_epic_v2", "450K → EPIC v2"), ("random_50|to_complete", "50% removed"), ("random_70|to_complete", "70% removed"),
          ("random_90|to_complete", "90% removed")]
    xx = np.arange(len(sc))
    ax = axd["a"]
    for kind, lab, c, mk, off in (("latent", "Latent scores", RUST["400"], "o", -0.2), ("raw", "Raw beta values", BLUE["400"], "s", 0.0),
                                  ("raw_pca50", "Raw beta, 50 PCs", BLUE["200"], "D", 0.2)):
        ax.scatter(xx + off, [100 * F[f"{kind}|{s}"]["top1"] for s, _ in sc], s=16, marker=mk, color=c, edgecolors="none", zorder=3,
                   clip_on=False, label=lab)
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in sc], rotation=35, ha="right")
    ax.set_ylabel("Profiles that retrieve\nthemselves (%)")
    ax.set_title("Six-tissue panel", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="lower left")
    ax = axd["b"]
    sel = [("hm27|to_complete", "27K"), ("hm27|to_epic_v2", "27K → EPIC v2"), ("random_70|to_complete", "70% removed"), ("random_90|to_complete", "90% removed")]
    for j, (comp, lab, c, mk) in enumerate((("raw", "minus raw beta", BLUE["400"], "s"), ("raw_pca50", "minus 50 raw PCs", BLUE["200"], "D"))):
        for i, (s, _) in enumerate(sel):
            v = F[f"latent|{s}"]
            y_ = len(sel) - 1 - i + (j - 0.5) * 0.3
            ax.plot([100 * x for x in v[f"top1_minus_{comp}_ci95"]], [y_] * 2, color=c, lw=0.9, solid_capstyle="butt", zorder=2)
            ax.scatter([100 * v[f"top1_minus_{comp}"]], [y_], s=16, marker=mk, color=c, edgecolors="none", zorder=3, clip_on=False,
                       label=lab if i == 0 else None)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([l for _, l in sel][::-1])
    ax.set_xlabel("Difference in retrieval (percentage points)")
    nature_legend(ax, loc="lower left", title="Latent scores")
    trim_axes(fig)
    set_domain(axd["a"], ylim=(-4, 104))
    axd["a"].set_yticks([0, 25, 50, 75, 100]); axd["a"].spines["left"].set_bounds(0, 100)
    pad_categorical_axis(axd["b"], "y", pad=0.6)
    ref_line(axd["b"], x=0)
    add_panel_labels(fig, axd)
    save_figure(fig, str(HERE / "supplementary_harmonization_2026-10-22.pdf"), axes_dict=axd,
                review_png=str(HERE / "supplementary_harmonization_2026-10-22_review.png"),
                footer="data: test-split profiles of the six-tissue panel; artifacts/eval/si_followup_2026-10-22.json")
    plt.close(fig)


if __name__ == "__main__":
    for w in (sys.argv[1:] or ["attr", "harm"]):
        {"attr": attr, "harm": harm}[w]()
