"""Supplementary Figures for the sample-representation reruns (18 October 2026).

sfa  Sample representations by dataset, age and sex
     ../artifacts/eval/si_sample_representations_2026-10-18.json             metrics (si_sample_representations_2026-10-18.py)
     ../artifacts/eval/si_sample_representations_umap_setB_2026-10-18.npz    set B UMAP coordinates (same script)
     figure3_umap_coords_2026-10-02.npz                                       set A UMAP coordinates of Fig. 3a,b
     ../artifacts/eval/si_sample_representations_cross_tissue_2026-10-18.csv per-profile cross-tissue neighbour fractions
     ../artifacts/eval/representation_benchmark_2026-10-01.json               set A standardized sex 10-NN (Fig. 3)
sfb  Sample embedding structure with and without pretraining
     ../artifacts/eval/si_embedding_pretraining_2026-10-18.json              metrics (si_embedding_pretraining_2026-10-18.py)
     ../artifacts/eval/si_embedding_pretraining_pca_2026-10-18.npz           held-out PCA coordinates
     ../artifacts/eval/si_age_label_composition_2026-10-18.{csv,npz}         age-labelled profiles by split and tissue
     ../artifacts/eval/si_embedding_probe_2026-10-18.{json,csv}              linear age probes fitted on training profiles
                                                                              (si_embedding_probe_2026-10-18.py; panels e, f)
Run: MPLBACKEND=Agg python plot_si_sample_embedding_2026-10-18.py [sfa] [sfb]
     and again with NATURE_FIGURE_NOFOOTER=1 for the no-footer copies.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,
                                 set_domain, allow_dead_margin, ref_line, label_panel_group, add_colorbar, categorical_colors,
                                 pad_categorical_axis, ERRORBAR, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_BLUE)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
TISSUE_COL = {"Whole Blood": RUST["400"], "Brain": BLUE["400"], "Colon": GOLD["400"], "Liver": SAGE["400"],
              "Muscle": PLUM["400"], "Adipose": GRAY["400"]}
SEX_COL = {"F": GOLD["300"], "M": BLUE["300"]}
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
CMAP_AGE = LinearSegmentedColormap.from_list("age", [GOLD["200"], RUST["300"], PLUM["400"], BLUE["600"]])


def umap_axis(ax, title=None, labels=False):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if labels:
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    if title:
        ax.set_title(title, fontsize=7, fontweight="normal")
    ax._no_trim = True


def corner(ax, text, loc="lower right"):
    x, ha = (0.99, "right") if "right" in loc else (0.01, "left")
    y, va = (0.01, "bottom") if "lower" in loc else (0.99, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6, color=INK)


def finish(fig, axd, name, footer, groups=None, after_trim=None):
    trim_axes(fig, exclude=[ax for ax in fig.axes if getattr(ax, "_no_trim", False)])
    if after_trim:
        after_trim()
    fig.canvas.draw()
    if groups:
        for letter, axes in groups.items():
            label_panel_group(fig, axes, letter)
    fig.savefig(str(HERE / f"{name}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{name}.pdf"), axes_dict=axd, review_png=str(HERE / f"{name}_review.png"), footer=footer)
    plt.close(fig)


def pt_interval(ax, y, e, color, marker="o", s=16, filled=True, label=None):
    lo, hi = e["ci95"]
    ax.plot([lo, hi], [y, y], color=color, lw=ERRORBAR["forest"]["elinewidth"], solid_capstyle="butt", zorder=2)
    kw = dict(color=color, edgecolors="none") if filled else dict(facecolors="white", edgecolors=color, linewidths=0.9)
    ax.scatter([e["value"]], [y], s=s, marker=marker, zorder=3, clip_on=False, label=label, **kw)


# =================================================================================================================
def sfa():
    R = json.loads((A / "eval/si_sample_representations_2026-10-18.json").read_text())
    old = json.loads((A / "eval/representation_benchmark_2026-10-01.json").read_text())
    ub = np.load(A / "eval/si_sample_representations_umap_setB_2026-10-18.npz", allow_pickle=False)
    ua = np.load(HERE / "figure3_umap_coords_2026-10-02.npz")
    za = np.load(A / "eval/representations_setA_2026-10-01.npz")
    assert (za["rows"] == ua["rows"]).all()
    prof = pd.read_parquet(A / "corpus_2026-09-30/profiles.parquet", columns=["sex", "tissue", "group"])
    sexA = prof.sex.to_numpy()[ua["rows"]]
    ct = pd.read_csv(A / "eval/si_sample_representations_cross_tissue_2026-10-18.csv")

    fig = plt.figure(figsize=figsize_nature("double", aspect=1.12), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.03)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 0.95, 1.1])
    r1 = outer[0].subgridspec(1, 6, width_ratios=[1, 1, 0.95, 1, 1, 0.3])
    r2 = outer[1].subgridspec(1, 5, width_ratios=[1, 1, 0.75, 1.05, 0.7])
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.55, 1])
    r4 = outer[3].subgridspec(1, 3, width_ratios=[1.15, 1.0, 0.8])
    ax_ad, ax_bd, ax_leg, ax_aa, ax_ba = (fig.add_subplot(r1[i]) for i in range(5))
    ax_cs, ax_cr, ax_sl, ax_f, ax_tl = (fig.add_subplot(r2[i]) for i in range(5))
    ax_dt, ax_ds = fig.add_subplot(r3[0]), fig.add_subplot(r3[1])
    ax_e, ax_g, ax_gl = (fig.add_subplot(r4[i]) for i in range(3))
    for ax in (ax_leg, ax_sl, ax_tl, ax_gl):
        ax.axis("off"); ax._no_trim = True; ax._ying_is_colorbar = True

    def legend_in(ax, h, l, title, **kw):
        lg = ax.legend(h, l, loc="center left", frameon=False, fontsize=7, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.0,
                       title=title, title_fontsize=7, **kw)
        lg._legend_box.align = "left"
        return lg

    def umap_panel(ax, title):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_box_aspect(1)
        ax.set_title(title, fontsize=7, fontweight="normal")
        ax._no_trim = True

    sbn = R["set_b"]["natural"]
    # a: set B by dataset
    ds = ub["dataset"].astype(str)
    lv = sorted(set(ds))
    cols = dict(zip(lv, categorical_colors(len(lv), vivid=True)))
    for ax, key, rep, name in ((ax_ad, "latent", "factor_scores", "Latent scores"), (ax_bd, "raw", "raw", "Raw beta values")):
        xy = ub[key]
        for d in lv:
            m = ds == d
            ax.scatter(xy[m, 0], xy[m, 1], s=2.5, color=cols[d], alpha=0.75, edgecolors="none", rasterized=True, clip_on=False,
                       label=f"{d} ({m.sum()})")
        v = sbn[rep]
        umap_panel(ax, f"{name}\ndataset 10-NN {v['dataset_knn_bacc|stratified5']['value']:.2f}\niLISI {v['batch_ilisi']['value']:.2f}")
    legend_in(ax_leg, *ax_ad.get_legend_handles_labels(), f"Dataset ($n$ = {len(ds)})", markerscale=2.5)

    # b: set B by age
    age = ub["age"]
    o = np.argsort(age)
    for ax, key, rep, name in ((ax_aa, "latent", "factor_scores", "Latent scores"), (ax_ba, "raw", "raw", "Raw beta values")):
        xy = ub[key]
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=age[o], cmap=CMAP_AGE, vmin=0, vmax=90, s=2.5, alpha=0.85, edgecolors="none",
                        rasterized=True, clip_on=False)
        v = sbn[rep]
        umap_panel(ax, f"{name}\nage $R^2$ {v['age_ridge_r2|kfold5']['value']:.2f}\n(leave one dataset out {v['age_ridge_r2|lodo']['value']:.2f})")
    add_colorbar(sc, ax_ba, label="Age (years)", ticks=[0, 45, 90], inset=(1.06, 0.12, 0.06, 0.70))

    # c: set A by sex (coordinates of Fig. 3a,b)
    san, sas = R["set_a"]["natural"], R["set_a"]["standardized"]
    for ax, key, rep, name in ((ax_cs, "fs", "factor_scores", "Latent scores"), (ax_cr, "raw", "raw", "Raw beta values")):
        xy = ua[key]
        for sx, lab_ in ((None, "Not recorded"), ("F", "Female"), ("M", "Male")):
            m = (sexA == sx) if sx else ~np.isin(sexA, ["M", "F"])
            ax.scatter(xy[m, 0], xy[m, 1], s=2, color=SEX_COL.get(sx, GRAY["100"]), alpha=0.75, edgecolors="none", rasterized=True,
                       clip_on=False, label=f"{lab_} ({m.sum():,})")
        umap_panel(ax, f"{name}\nsex 10-NN {san[rep]['sex_knn_bacc|stratified5']['value']:.2f}"
                       f"\n(standardized {sas[rep]['sex_knn_bacc|stratified5']['value']:.2f})")
    h, l = ax_cs.get_legend_handles_labels()
    legend_in(ax_sl, h[1:] + h[:1], l[1:] + l[:1], f"Sex ($n$ = {len(sexA):,})", markerscale=3)

    # f: cross-tissue neighbour fraction, raw vs latent (jittered)
    rng = np.random.default_rng(0)
    jx = ct.cross_tissue_fraction_raw.to_numpy() + rng.uniform(-0.02, 0.02, len(ct))
    jy = ct.cross_tissue_fraction_latent.to_numpy() + rng.uniform(-0.02, 0.02, len(ct))
    for t, c in TISSUE_COL.items():
        m = (ct.tissue == t).to_numpy()
        ax_f.scatter(jx[m], jy[m], s=5, color=c, alpha=0.6, edgecolors="none", clip_on=False, rasterized=True,
                     label=f"{t} ({m.sum()})")
    ax_f.set_box_aspect(1)
    ax_f.set_xlabel("Cross-tissue neighbours,\nraw (of 10)"); ax_f.set_ylabel("Cross-tissue neighbours,\nlatent scores (of 10)")
    xt = R["set_a_cross_tissue_neighbours"]
    ax_f.text(1.0, 0.0, f"$\\rho$ = {xt['spearman_raw_vs_latent']['rho']:.2f}\ndiscordant {xt['discordant_total']}\nof {len(ct):,}",
              transform=ax_f.transAxes, ha="right", va="bottom", fontsize=6, color=INK)
    legend_in(ax_tl, *ax_f.get_legend_handles_labels(), "Tissue", markerscale=2)

    # d: 10-NN balanced accuracy, stratified vs leave-one-dataset-out (set A, natural geometry)
    reps = [("raw", "Raw beta values", BLUE["400"]), ("raw_pca50", "Raw, 50 PCs", BLUE["200"]), ("factor_scores", "Latent scores", RUST["400"]),
            ("mgpt2_selfsupervised", "Self-supervised embedding", RUST["300"]), ("mgpt2_multitask", "Embedding", RUST["200"]),
            ("untrained_transformer", "Untrained transformer", PLUM["300"]), ("random_projection", "Random projection", GRAY["300"])]
    schemes = [("stratified5", "Stratified 5-fold", "o", True), ("lodo", "Leave one dataset out", "s", False)]
    for ax, target, xlabel in ((ax_dt, "tissue", "Tissue 10-NN balanced accuracy"), (ax_ds, "sex", "Sex 10-NN balanced accuracy")):
        for i, (k, lab_, c) in enumerate(reps):
            for j, (sch, slab, mk, filled) in enumerate(schemes):
                pt_interval(ax, i + (j - 0.5) * 0.34, san[k][f"{target}_knn_bacc|{sch}"], c, mk, s=13, filled=filled,
                            label=slab if i == 0 else None)
        ax.set_yticks(range(len(reps)))
        ax.set_yticklabels([r[1] for r in reps] if ax is ax_dt else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    nature_legend(ax_dt, title="Cross-validation", loc="lower left")

    # e: Leiden ARI over the resolution grid (set A)
    cl = R["set_a_clustering"]
    keys_ = sorted((k for k in cl["raw"] if k.startswith("leiden_")), key=lambda k: float(k.split("_")[1]))
    res_ = [float(k.split("_")[1]) for k in keys_]
    for k, lab_, c in [reps[0], reps[2], reps[4], reps[5], reps[6]]:
        for tgt, ls in (("tissue", "-"), ("dataset", (0, (2, 1.5)))):
            ax_e.plot(res_, [cl[k][kk][f"ari_{tgt}"]["value"] for kk in keys_], ls=ls, color=c, lw=1.0, marker="o", ms=2.2, clip_on=False)
        km = cl[k]["kmeans_6"]
        ax_e.scatter([4.0], [km["ari_tissue"]["value"]], s=16, color=c, marker="D", clip_on=False, zorder=3)
        ax_e.scatter([4.0], [km["ari_dataset"]["value"]], s=16, facecolors="white", edgecolors=c, linewidths=0.8, marker="D",
                     clip_on=False, zorder=3)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("Leiden resolution"); ax_e.set_ylabel("Adjusted Rand index")
    ax_e.set_title("Colours as in d. Solid: vs tissue; dashed:\nvs dataset; diamonds: k-means, k = 6", fontsize=7, fontweight="normal")

    # g: ComBat, dataset identifiability vs age R2; within-fold (filled) and fitted on all profiles (open)
    wf, ins = R["set_b_combat_within_fold"], R["set_b"]["natural"]
    adj = "|combat_adjustable_cpgs"
    ncpg = R["combat_corrected_for_fig3d"]["cpg_counts"]["adjusted_by_combat"]
    gp = [("raw", "Raw beta values", BLUE["400"], "s"), ("factor_scores", "Latent scores", RUST["400"], "o"),
          ("mgpt2_multitask", "Embedding", RUST["200"], "o"), ("combat", "ComBat, all CpGs", SAGE["600"], "v"),
          ("raw" + adj, f"Raw, {ncpg:,} adjustable CpGs", BLUE["200"], "s"), ("combat" + adj, "ComBat, adjustable CpGs", SAGE["500"], "v"),
          ("combat_sex" + adj, "  + sex protected", SAGE["400"], "^"), ("combat_sex_age" + adj, "  + sex and age protected", SAGE["300"], "D")]
    for k, lab_, c, mk in gp:
        e = wf[k]
        ax_g.plot(e["dataset_knn_bacc"]["ci95"], [e["age_ridge_r2"]["value"]] * 2, color=c, lw=0.6, zorder=2)
        ax_g.plot([e["dataset_knn_bacc"]["value"]] * 2, e["age_ridge_r2"]["ci95"], color=c, lw=0.6, zorder=2)
        ax_g.scatter(e["dataset_knn_bacc"]["value"], e["age_ridge_r2"]["value"], s=22, marker=mk, color=c, edgecolors="none",
                     clip_on=False, zorder=3, label=lab_)
        if k.startswith("combat"):
            v = ins[k]
            ax_g.scatter(v["dataset_knn_bacc|stratified5"]["value"], v["age_ridge_r2|kfold5"]["value"], s=22, marker=mk,
                         facecolors="white", edgecolors=c, linewidths=0.9, clip_on=False, zorder=3)
    ax_g.set_xlabel("Dataset identifiable\n(10-NN balanced accuracy)"); ax_g.set_ylabel("Age $R^2$ (cross-validated)")
    ax_g.set_title("Whole blood", fontsize=7, fontweight="normal")
    legend_in(ax_gl, *ax_g.get_legend_handles_labels(), f"Filled: fitted within folds\nOpen: fitted on all {len(ds)}\nprofiles")

    axd = {"a": ax_ad, "b": ax_aa, "c": ax_cs, "d": ax_dt, "e": ax_e, "f": ax_f, "g": ax_g}
    groups = {"a": [ax_ad, ax_bd], "b": [ax_aa, ax_ba], "c": [ax_cs, ax_cr], "d": [ax_dt, ax_ds], "e": [ax_e], "f": [ax_f], "g": [ax_g]}

    def after():
        for ax in (ax_dt, ax_ds):
            pad_categorical_axis(ax, "y", pad=0.6)
        set_domain(ax_f, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
        set_domain(ax_dt, xlim=(0.78, 1.0))
        ax_dt.set_xticks([0.8, 0.9, 1.0]); ax_dt.spines["bottom"].set_bounds(0.8, 1.0)
        for ax in (ax_f,):
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.spines["bottom"].set_bounds(0, 1); ax.spines["left"].set_bounds(0, 1)

    finish(fig, axd, "si_sample_representations_2026-10-18", "data: test-split datasets of the MethylGPT corpus; artifacts/eval/si_sample_representations_2026-10-18.json",
           groups, after)


# =================================================================================================================
GROUPS = [("training (metadata ages)", "Training (metadata ages)", GRAY["400"]),
          ("AltumAge validation", "AltumAge validation", SAGE["400"]),
          ("AltumAge test, training-split datasets", "AltumAge test, training-split datasets", GOLD["400"]),
          ("AltumAge test, other datasets", "AltumAge test, other datasets", PLUM["400"]),
          ("held-out test-split datasets", "Held-out test-split datasets", BLUE["400"])]
ENC = [("raw", "Raw beta values", BLUE["400"], "s"), ("latent_scores", "Latent scores (input layer)", RUST["200"], "o"),
       ("latent_scores_scaled", "Latent scores / prior SD", RUST["200"], "D"),
       ("untrained_transformer", "Untrained transformer", PLUM["200"], "o"), ("ssl", "Self-supervised encoder", RUST["300"], "o"),
       ("age_stage", "Age-stage encoder", RUST["500"], "o"), ("randominit", "Random transformer + age stage", PLUM["400"], "^"),
       ("nopretrain", "No pretraining + age stage", PLUM["600"], "v")]


def sfb():
    R = json.loads((A / "eval/si_embedding_pretraining_2026-10-18.json").read_text())
    comp = np.load(A / "eval/si_age_label_composition_2026-10-18.npz", allow_pickle=False)
    pca = np.load(A / "eval/si_embedding_pretraining_pca_2026-10-18.npz", allow_pickle=False)
    cdf = pd.DataFrame({"group": comp["group"], "tissue": comp["tissue"], "age": comp["age"]})
    counts = R["age_label_composition"]["counts"]
    top = cdf.tissue.value_counts()
    top = [t for t in top.index if t != "no tissue label"][:9]
    lab_t = {t: ('"Other" (as labelled)' if t == "Other" else t) for t in top}
    cdf["row"] = np.where(cdf.tissue.isin(top), cdf.tissue.map(lab_t),
                          np.where(cdf.tissue == "no tissue label", "No tissue label", "Remaining tissues"))
    rows = [lab_t[t] for t in top] + ["Remaining tissues", "No tissue label"]

    PJ = json.loads((A / "eval/si_embedding_probe_2026-10-18.json").read_text())
    PC = pd.read_csv(A / "eval/si_embedding_probe_2026-10-18.csv")
    fig = plt.figure(figsize=figsize_nature("double", aspect=1.06), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.03)
    outer = fig.add_gridspec(4, 1, height_ratios=[0.8, 0.6, 0.6, 0.9])
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.25, 1.25, 0.72])
    r2 = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    r3 = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.15])
    ax_n = fig.add_subplot(r1[0]); ax_s = fig.add_subplot(r1[1], sharey=ax_n)
    pb = [fig.add_subplot(r2[i]) for i in range(3)]
    pc = [fig.add_subplot(r3[i]) for i in range(3)]
    r4 = outer[3].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_d, ax_e = fig.add_subplot(r4[0]), fig.add_subplot(r4[1])

    # a: counts (dot plot) and ages (strip plot) by tissue and split group
    off = np.linspace(-0.32, 0.32, len(GROUPS))
    rng = np.random.default_rng(0)
    for (g, lab_, c), o in zip(GROUPS, off):
        sub = cdf[cdf.group == g]
        n = sub.row.value_counts().reindex(rows).fillna(0).to_numpy()
        yy = np.arange(len(rows)) + o
        keep = n > 0
        ax_n.scatter(n[keep], yy[keep], s=14, color=c, edgecolors="none", clip_on=False, zorder=3,
                     label=f"{lab_.replace(', ', ',\n')}\n($n$ = {counts[g]['profiles']:,}; {counts[g]['datasets']} datasets)")
        yi = np.array([rows.index(r) for r in sub.row]) + o + rng.uniform(-0.05, 0.05, len(sub))
        ax_s.scatter(sub.age, yi, s=1.2, color=c, alpha=0.3 if g.startswith("training") else 0.6, edgecolors="none",
                     rasterized=True, clip_on=False)
    ax_n.set_xscale("log")
    ax_n.set_yticks(range(len(rows))); ax_n.set_yticklabels(rows)
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Profiles with age")
    ax_s.set_xlabel("Age (years)")
    ax_s.tick_params(labelleft=False)
    h, l = ax_n.get_legend_handles_labels()
    ax_lg = fig.add_subplot(r1[2]); ax_lg.axis("off"); ax_lg._no_trim = True; ax_lg._ying_is_colorbar = True
    lg = ax_lg.legend(h, l, loc="center left", frameon=False, fontsize=7, title="Age-labelled profiles", title_fontsize=7,
                      handletextpad=0.3, labelspacing=0.6, borderaxespad=0.0)
    lg._legend_box.align = "left"

    # b, c: held-out PCA, coloured by age and by tissue
    arms = [("ssl", "Self-supervised\nencoder"), ("age_stage", "Age-stage\nencoder"), ("nopretrain", "No pretraining\n+ age stage")]
    age = pca["age"]
    tis = pca["tissue"].astype(str)
    tcount = pd.Series(tis).value_counts()
    ttop = [t for t in tcount.index if t not in ("no tissue label", "Other")][:6]
    tcol = dict(zip(ttop, [RUST["400"], BLUE["400"], GOLD["400"], SAGE["400"], PLUM["400"], BLUE["200"]]))
    evr = R["heldout_pca"]["explained_variance_ratio"]
    o = np.argsort(age)
    for ax, (k, title) in zip(pb, arms):
        xy = pca[k]
        sc = ax.scatter(xy[o, 0], xy[o, 1], c=age[o], cmap=CMAP_AGE, vmin=0, vmax=100, s=1.5, alpha=0.8, edgecolors="none",
                        rasterized=True, clip_on=False)
        ax.set_title(title, fontsize=7, fontweight="normal")
    add_colorbar(sc, pb[2], label="Age (years)", ticks=[0, 50, 100], inset=(1.10, 0.10, 0.06, 0.70))
    for ax, (k, title) in zip(pc, arms):
        xy = pca[k]
        m0 = ~np.isin(tis, ttop)
        ax.scatter(xy[m0, 0], xy[m0, 1], s=1.5, color=GRAY["100"], alpha=0.8, edgecolors="none", rasterized=True, clip_on=False,
                   label=f"Other or unlabelled ({m0.sum():,})")
        for t in ttop:
            m = tis == t
            ax.scatter(xy[m, 0], xy[m, 1], s=1.5, color=tcol[t], alpha=0.8, edgecolors="none", rasterized=True, clip_on=False,
                       label=f"{t} ({m.sum():,})")
    for ax, (k, _) in list(zip(pb, arms)) + list(zip(pc, arms)):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"PC1 ({100 * evr[k][0]:.0f}%)", fontsize=7); ax.set_ylabel(f"PC2 ({100 * evr[k][1]:.0f}%)", fontsize=7)
        ax._no_trim = True
    h, l = pc[2].get_legend_handles_labels()
    ax_tl = fig.add_subplot(r3[3]); ax_tl.axis("off"); ax_tl._no_trim = True; ax_tl._ying_is_colorbar = True
    lg2 = ax_tl.legend(h[1:] + h[:1], l[1:] + l[:1], loc="center left", frameon=False, fontsize=7,
                       markerscale=3, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.0, title=f"Tissue ($n$ = {len(tis):,})",
                       title_fontsize=7)
    lg2._legend_box.align = "left"

    # d, e: linear age probe fitted on the 41,845 training profiles, applied unchanged to profiles with CpGs removed at random
    X3 = json.loads((A / "eval/si_favourable_extensions_2026-10-19.json").read_text())["x3_probe"]
    D = pd.DataFrame(PJ["e1_e2_e3_e6"])
    D = D[(D.representation == "embedding") & (D.comparison == "age_stage minus nopretrain") & (D.labels == PC.labels.max())]
    med = PC[(PC.representation == "embedding") & (PC.labels == PC.labels.max())].set_index(["set", "encoder"]).medae
    RATES = [30, 50, 70, 90]

    def probe(dom, rate):
        k = f"{dom}|random_{rate}"
        if rate == 90:
            r = D[D.set == k].iloc[0]
            sd_ = [med[(k, a_)] for a_ in ("age_stage", "age_stage_s2", "age_stage_s3")]
            return dict(a=med[(k, "age_stage")], b=med[(k, "nopretrain")], d=r.medae_difference, ci=list(r.ci95), lo=min(sd_), hi=max(sd_))
        r = X3[k]
        return dict(a=r["a"], b=r["b"], d=r["difference"], ci=r["ci95"], lo=r["age_stage_seed_range"][0], hi=r["age_stage_seed_range"][1])
    nh = int(PC[(PC.set == "heldout|all")].n_profiles.iloc[0]); na = int(PC[(PC.set == "altumage_test|all")].n_profiles.iloc[0])
    for dom, dlab, ls, mk, off, ce in (("heldout", f"Held-out ($n$ = {nh:,}; 34 datasets)", "-", "o", -1.3, RUST["500"]),
                                       ("altumage_test", f"AltumAge test ($n$ = {na:,}; 110 datasets)", (0, (3, 2)), "s", 1.3, RUST["300"])):
        P_ = [probe(dom, q) for q in RATES]
        ax_d.fill_between(RATES, [p_["lo"] for p_ in P_], [p_["hi"] for p_ in P_], color=RUST["100"], alpha=0.6, lw=0, zorder=1)
        ax_d.plot(RATES, [p_["a"] for p_ in P_], ls=ls, marker=mk, ms=3.5, lw=0.9, color=RUST["500"], zorder=3, clip_on=False,
                  label=f"Pretrained, {dom.replace('altumage_test', 'AltumAge test').replace('heldout', 'held-out')}")
        ax_d.plot(RATES, [p_["b"] for p_ in P_], ls=ls, marker=mk, ms=3.5, lw=0.9, color=PLUM["600"], mfc="white", zorder=3, clip_on=False,
                  label=f"No pretraining, {dom.replace('altumage_test', 'AltumAge test').replace('heldout', 'held-out')}")
        for q, p_ in zip(RATES, P_):
            ax_e.plot([q + off] * 2, p_["ci"], color=ce, lw=0.8, solid_capstyle="butt", zorder=2)
        ax_e.scatter([q + off for q in RATES], [p_["d"] for p_ in P_], s=14, marker=mk, color=ce, edgecolors="none", zorder=3, clip_on=False, label=dlab)
    ax_d.set_xlabel("CpGs removed at random (%)"); ax_d.set_ylabel("Probe MedAE (years)")
    ax_d.set_title("Linear age probe fitted on training profiles", fontsize=7, fontweight="normal")
    nature_legend(ax_d, loc="upper left")
    ax_e.set_xlabel("CpGs removed at random (%)"); ax_e.set_ylabel("Pretrained minus no\npretraining (years)")
    ax_e.set_title("Paired difference with 95% interval", fontsize=7, fontweight="normal")
    nature_legend(ax_e, loc="lower left")

    axd = {"a": ax_n, "b": pb[0], "c": pc[0], "d": ax_d, "e": ax_e}
    groups = {"a": [ax_n, ax_s], "b": pb, "c": pc, "d": [ax_d], "e": [ax_e]}

    def after():
        pad_categorical_axis(ax_n, "y", pad=0.6)
        for ax in (ax_d, ax_e):
            set_domain(ax, xlim=(25, 95))
            ax.set_xticks(RATES)
            ax.spines["bottom"].set_bounds(30, 90)
        ref_line(ax_e, y=0)
        set_domain(ax_s, xlim=(-4, 120))
        ax_s.set_xticks([0, 20, 40, 60, 80, 100, 120])
        ax_s.spines["bottom"].set_bounds(0, 120)

    finish(fig, axd, "si_embedding_pretraining_2026-10-18", "data: MethylGPT corpus and AltumAge splits; artifacts/eval/si_embedding_pretraining_2026-10-18.json",
           groups, after)

def si2_fig1():
    """Supplementary Information 2, Fig. 1: embedding structure within held-out datasets and the full probe grid."""
    R = json.loads((A / "eval/si_embedding_pretraining_2026-10-18.json").read_text())
    PJ = json.loads((A / "eval/si_embedding_probe_2026-10-18.json").read_text())
    ENC = [e for e in globals()["ENC"] if e[0] not in ("untrained_transformer", "randominit")]   # pretraining is not split into parts
    PC = pd.read_csv(A / "eval/si_embedding_probe_2026-10-18.csv")
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.92), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.03)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 0.85])
    r4 = outer[0].subgridspec(2, 3, width_ratios=[1.75, 1, 1])
    qd = [fig.add_subplot(r4[i, j]) for i in range(2) for j in range(3)]
    r5 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_e, ax_f = fig.add_subplot(r5[0]), fig.add_subplot(r5[1])
    # a: metrics with dataset-resampled 95% intervals
    ta, ba, wa = R["set_a_tissue"], R["set_b_age"], R["set_a_age_within_tissue"]
    two = lambda e1, e2: [(e1, "o", True), (e2, "s", False)]
    panels = [(qd[0], "Tissue silhouette\n(six-tissue panel)", lambda k: [(ta[k]["tissue_silhouette"], "o", True)]),
              (qd[1], "Tissue 10-NN balanced accuracy", lambda k: two(ta[k]["tissue_knn_bacc|stratified5"], ta[k]["tissue_knn_bacc|lodo"])),
              (qd[2], "Age $R^2$ within tissue\n(six-tissue panel)", lambda k: two(wa[k]["age_ridge_r2_within_tissue|kfold5"], wa[k]["age_ridge_r2_within_tissue|lodo"])),
              (qd[3], "Age $R^2$\n(whole-blood panel)", lambda k: two(ba[k]["age_ridge_r2|kfold5"], ba[k]["age_ridge_r2|lodo"])),
              (qd[4], "Age MedAE (years)\n(whole-blood panel)", lambda k: two(ba[k]["age_knn_medae|kfold5"], ba[k]["age_knn_medae|lodo"])),
              (qd[5], "Age MedAE within tissue (years)\n(six-tissue panel)", lambda k: two(wa[k]["age_knn_medae_within_tissue|kfold5"], wa[k]["age_knn_medae_within_tissue|lodo"]))]
    for ax, xlabel, get in panels:
        for i, (k, lab_, c, mk) in enumerate(ENC):
            items = get(k)
            for j, (e, mk2, filled) in enumerate(items):
                dy = 0 if len(items) == 1 else (j - 0.5) * 0.34
                pt_interval(ax, i + dy, e, c, mk if (filled and len(items) == 1) else mk2, s=12, filled=filled,
                            label=("Random 5-fold" if filled else "Leave one dataset out") if i == 0 and len(items) > 1 else None)
        ax.set_yticks(range(len(ENC)))
        ax.set_yticklabels([e[1] for e in ENC] if ax in (qd[0], qd[3]) else [])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
    nature_legend(qd[3], title="Cross-validation", loc="lower left")

    # b: linear age probe fitted on the 41,845 training profiles, applied unchanged; MedAE by set and encoder
    SETS = [("heldout|all", "Held-out, all CpGs"), ("heldout|hm27", "Held-out, 27K CpGs"), ("heldout|random_90", "Held-out, 90% removed"),
            ("altumage_test|all", "AltumAge test, all CpGs"), ("altumage_test|hm27", "AltumAge test, 27K CpGs"),
            ("altumage_test|random_90", "AltumAge test, 90% removed")]
    PE = [("ssl", "Self-supervised", RUST["300"], "o"), ("age_stage", "Age stage (seed range)", RUST["500"], "o"),
          ("nopretrain", "No pretraining", PLUM["600"], "v")]
    full = PC[(PC.representation == "embedding") & (PC.labels == PC.labels.max())]
    med = full.set_index(["set", "encoder"]).medae
    for j, (k, lab_, c, mk) in enumerate(PE):
        dy = (j - 1.0) * 0.2
        for i, (sk, _) in enumerate(SETS):
            if k == "age_stage":
                v = [med[(sk, a_)] for a_ in ("age_stage", "age_stage_s2", "age_stage_s3")]
                ax_e.plot([min(v), max(v)], [i + dy] * 2, color=c, lw=0.8, solid_capstyle="butt", zorder=2)
            ax_e.scatter([med[(sk, k)]], [i + dy], s=13, marker=mk, color=c, edgecolors="none", zorder=3, clip_on=False,
                         label=lab_ if i == 0 else None)
    ax_e.set_xscale("log")
    ax_e.set_yticks(range(len(SETS))); ax_e.set_yticklabels([l_ for _, l_ in SETS]); ax_e.invert_yaxis()
    ax_e.set_xlabel("Probe MedAE (years, log scale)")
    nh = full[full.set == "heldout|all"].n_profiles.iloc[0]; na = full[full.set == "altumage_test|all"].n_profiles.iloc[0]
    ax_e.set_title("Probe fitted on training profiles", fontsize=7, fontweight="normal")
    ax_e.legend(loc="center right", frameon=False, fontsize=7, title=f"$n$ = {nh:,} held-out;\n{na:,} AltumAge test", title_fontsize=7,
                handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)._legend_box.align = "left"

    # c: paired difference, age-stage minus no pretraining, by number of labelled training profiles
    D = pd.DataFrame(PJ["e1_e2_e3_e6"])
    D = D[(D.representation == "embedding") & (D.comparison == "age_stage minus nopretrain")]
    LAB = [(int(PC.labels.max()), "All 41,845 labels", RUST["500"], "o"), (10000, "10,000 labels", RUST["300"], "s"), (1000, "1,000 labels", RUST["200"], "D")]
    for j, (n, lab_, c, mk) in enumerate(LAB):
        dy = (j - 1) * 0.22
        for i, (sk, _) in enumerate(SETS):
            r = D[(D.labels == n) & (D.set == sk)].iloc[0]
            ax_f.plot(r.ci95, [i + dy] * 2, color=c, lw=0.8, solid_capstyle="butt", zorder=2)
            ax_f.scatter([r.medae_difference], [i + dy], s=13, marker=mk, color=c, edgecolors="none", zorder=3, clip_on=False,
                         label=lab_ if i == 0 else None)
    ax_f.set_yticks(range(len(SETS))); ax_f.set_yticklabels([]); ax_f.invert_yaxis()
    ax_f.set_xlabel("MedAE difference, age stage minus\nno pretraining (years; negative favours pretraining)")
    ax_f.set_title("Paired difference with 95% interval", fontsize=7, fontweight="normal")
    ax_f.legend(loc="center right", frameon=False, fontsize=7, handletextpad=0.3, labelspacing=0.3, borderaxespad=0.2)

    axd = {"a": qd[0], "b": ax_e, "c": ax_f}

    def after():
        for ax in qd:
            pad_categorical_axis(ax, "y", pad=0.6)
        for ax in (ax_e, ax_f):
            ax.set_ylim(len(SETS) - 0.4, -0.6)
        set_domain(ax_e, xlim=(1.6, 500))
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        ax_e.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
        ax_e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax_e.xaxis.set_minor_locator(NullLocator())
        ax_e.spines["bottom"].set_bounds(2, 20)
        allow_dead_margin(ax_e, "x", reason="right end holds the encoder legend")
        set_domain(ax_f, xlim=(-15.5, 19))
        ax_f.set_xticks([-15, -10, -5, 0, 5, 10])
        ax_f.spines["bottom"].set_bounds(-15, 10)
        allow_dead_margin(ax_f, "x", reason="right end holds the legend")
        ref_line(ax_f, x=0)

    finish(fig, axd, "si2_embedding_designs_2026-10-19", "data: artifacts/eval/si_embedding_pretraining_2026-10-18.json, si_embedding_probe_2026-10-18.{json,csv}",
           {"a": qd, "b": [ax_e], "c": [ax_f]}, after)


if __name__ == "__main__":
    which = sys.argv[1:] or ["sfa", "sfb", "si2"]
    for w in which:
        {"sfa": sfa, "sfb": sfb, "si2": si2_fig1}[w]()
