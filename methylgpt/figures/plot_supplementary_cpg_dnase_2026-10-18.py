"""Supplementary Figure: CpG representations, DNase hypersensitivity and baselines (extension of Fig. 2).

Inputs (aggregate files only, run from this directory):
  ../artifacts/eval/cpg_representation_si_2026-10-18.json          kNN, AUC and interval results
  ../artifacts/eval/cpg_representation_si_figdata_2026-10-18.npz   island confusion matrices and vote histograms
  ../artifacts/eval/cpg_umap_si_2026-10-18.npz                     UMAP coordinates (illustration only)
  ../artifacts/cpg_annotation_2026-10-01.parquet                   manifest annotation (UMAP colours)
All numbers come from benchmark_cpg_representations_si_2026-10-18.py.
Run: MPLBACKEND=Agg python plot_supplementary_cpg_dnase_2026-10-18.py
     NATURE_FIGURE_NOFOOTER=1 MPLBACKEND=Agg python plot_supplementary_cpg_dnase_2026-10-18.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,
                                 pad_categorical_axis, ref_line, label_panel_group, confusion_matrix_panel, mark_pictorial,
                                 ERRORBAR, TYPE, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
NAME = "supplementary_figure_cpg_dnase_2026-10-18"
retro_style()

# rows of panel b, top to bottom: (key, label, colour); further representations are in Supplementary Table 3
REPS = [("loadings", "Input-layer loadings", RUST["400"]),
        ("encoder_input", "Encoder, mask", RUST["200"]),
        ("dense_c", "Encoder, centred values", RUST["200"]),
        ("dense_e", "Encoder, residuals", RUST["200"]),
        ("q_emb", "Query embedding", RUST["200"]),
        ("h_full", "h, full query", RUST["300"]),
        ("h_qemb", "h, query embedding", RUST["200"]),
        ("h_random", "h, randomly initialized", PLUM["400"]),
        ("raw_profiles", "Raw beta", BLUE["400"]),
        ("raw_pca64", "Raw beta, 64 PCs", BLUE["200"]),
        ("baseline", "Mean, SD, probe type", GRAY["400"]),
        ("random", "Random vectors", GRAY["200"])]
LABELS = [("island_relation", "CpG island relation", 0.25), ("enhancer", "Enhancer", 0.5), ("dnase_hs", "DNase HS", 0.5),
          ("x_chromosome", "Chromosome X", 0.5)]
ISL = [("Island", "Island", RUST["400"]), ("Shore", "Shore", GOLD["400"]), ("Shelf", "Shelf", SAGE["400"]), ("OpenSea", "Open sea", BLUE["300"])]
UMAP_REPS = [("loadings", "Input-layer\nloadings"), ("h_full", "Decoder state h"), ("raw_profiles", "Raw beta\nvalues")]


def interval(ax, x, y, lo, hi, col, filled=True, vertical=False, ms=3.2, marker="o", label=None):
    kw = dict(mfc=col, mec=INK, mew=0.3) if filled else dict(mfc="white", mec=col, mew=0.8)
    err = [[x - lo], [hi - x]] if not vertical else [[y - lo], [hi - y]]
    if vertical:
        ax.errorbar(x, y, yerr=err, fmt=marker, ms=ms, ecolor=col, **ERRORBAR["forest"], clip_on=False, zorder=3, label=label, **kw)
    else:
        ax.errorbar(x, y, xerr=err, fmt=marker, ms=ms, ecolor=col, **ERRORBAR["forest"], clip_on=False, zorder=3, label=label, **kw)


def main():
    res = json.loads((A / "eval/cpg_representation_si_2026-10-18.json").read_text())
    R = res["results"]
    fd = np.load(A / "eval/cpg_representation_si_figdata_2026-10-18.npz")
    um = np.load(A / "eval/cpg_umap_si_2026-10-18.npz")
    ann = pd.read_parquet(A / "cpg_annotation_2026-10-01.parquet")
    isl_all = ann.island_relation.fillna("NA").to_numpy()[um["idx"]]
    grid = [(int(a), float(b)) for a, b in um["grid"]]
    nE, nX = R["loadings"]["island_relation"]["grouped_cv"]["n"], R["loadings"]["x_chromosome"]["grouped_cv"]["n"]

    fig = plt.figure(figsize=figsize_nature("double", aspect=1.30), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.03)
    outer = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.6, 1.28, 1.45])
    r0 = outer[0].subgridspec(3, 6)
    rb = outer[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    r2 = outer[2].subgridspec(1, 3, width_ratios=[1.0, 1.3, 1.0])
    r3 = outer[3].subgridspec(1, 3, width_ratios=[0.8, 1.0, 1.3])

    # ---- a: UMAP illustration (no metrics on the panels)
    ua = []
    for i, (rk, rl) in enumerate(UMAP_REPS):
        for j, (nn_, md) in enumerate(grid):
            ax = fig.add_subplot(r0[i, j])
            xy = um[f"{rk}__{nn_}__{md}"]
            m0 = isl_all == "NA"
            ax.scatter(xy[m0, 0], xy[m0, 1], s=0.35, color=GRAY["100"], alpha=0.5, edgecolors="none", rasterized=True, clip_on=False)
            for key, lab, col in ISL:
                m = isl_all == key
                ax.scatter(xy[m, 0], xy[m, 1], s=0.35, color=col, alpha=0.5, edgecolors="none", rasterized=True, clip_on=False)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            mark_pictorial(ax)
            if i == 0:
                ax.set_title(f"{nn_} neighbours\nmin. dist. {md}", fontsize=TYPE["tick"], fontweight="normal", loc="center")
            if j == 0:
                ax.set_ylabel(rl, fontsize=TYPE["tick"])
            ua.append(ax)
    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, color=c, label=f"{l} ($n$ = {(isl_all == k).sum():,})") for k, l, c in ISL]
    handles.append(Line2D([], [], ls="none", marker="o", ms=3.5, color=GRAY["100"],
                          label=f"Not on EPIC v1 ($n$ = {(isl_all == 'NA').sum():,})"))
    fig.legend(handles=handles, title="CpG island relation, 15,000 CpGs", loc="outside upper center", ncol=5, frameon=False,
               fontsize=TYPE["legend"], title_fontsize=TYPE["legend"], handletextpad=0.2, columnspacing=1.0)

    # ---- b: kNN balanced accuracy, every representation; filled all CpGs, open within mean-beta decile x probe type
    bx = [fig.add_subplot(rb[0, j]) for j in range(4)]
    yy = np.arange(len(REPS))[::-1].astype(float)
    for ax, (lab, title, chance), short in zip(bx, LABELS, ["island relation", "enhancer", "DNase HS", "chromosome X"]):
        for (rk, rl, col), y in zip(REPS, yy):
            g = R[rk][lab]["grouped_cv"]
            s = R[rk][lab]["within_mean_decile_probe_type"]
            interval(ax, s["balanced_accuracy"], y, *s["ci95"], col, False, ms=3.0)
            interval(ax, g["balanced_accuracy"], y, *g["ci95"], col, True, ms=3.0)
        ax.set_yticks(yy)
        ax.set_yticklabels([r[1] for r in REPS] if ax is bx[0] else [])
        n = nX if lab == "x_chromosome" else nE
        ax.set_title(f"{title} ($n$ = {n:,})", fontsize=TYPE["tick"], fontweight="normal", loc="center")
        ax.set_xlabel(f"Balanced accuracy,\n{short}")
        ax._chance = chance

    # ---- c: primary comparisons and decoder contrasts
    cx = fig.add_subplot(r2[0])
    D = res["differences"]
    CON = [("raw_pca64 - loadings", "Loadings \u2212 raw, 64 PCs", BLUE["300"], -1),
           ("baseline - loadings", "Loadings \u2212 baseline", GRAY["400"], -1),
           ("h_full - loadings", "h \u2212 loadings", RUST["300"], 1),
           ("h_full - h_random", "h \u2212 randomly initialized h", PLUM["400"], 1)]
    xx = np.arange(len(LABELS))
    offs = np.linspace(-0.3, 0.3, len(CON))
    for (cname, cl, col, sign), off in zip(CON, offs):
        for an, dx_, filled in (("grouped", -0.04, True), ("adjusted", 0.04, False)):
            for k, (lab, _, _) in enumerate(LABELS):
                d = D[an][cname][lab]
                lo, hi = sorted(sign * np.array(d["ci95"]))
                interval(cx, k + off + dx_, sign * d["difference"], lo, hi, col, filled, vertical=True, ms=2.8,
                         label=cl if (an == "grouped" and k == 0) else None)
    cx.set_xticks(xx)
    cx.set_xticklabels(["Island\nrelation", "Enhancer", "DNase\nHS", "Chromo-\nsome X"])
    cx.set_ylabel("Difference in\nbalanced accuracy")

    # ---- d: DNase within CpG-island strata (AUC)
    dx = fig.add_subplot(r2[1])
    DREPS = [("loadings", "Input-layer loadings", RUST["400"], "o"), ("h_full", "Decoder state h", RUST["300"], "^"),
             ("h_random", "h, randomly initialized", PLUM["400"], "v"), ("raw_profiles", "Raw beta", BLUE["400"], "s"),
             ("raw_pca64", "Raw beta, 64 PCs", BLUE["200"], "s"), ("baseline", "Mean, SD, probe type", GRAY["400"], "D"),
             ("random", "Random vectors", GRAY["200"], "D")]
    offs = np.linspace(-0.32, 0.32, len(DREPS))
    for (rk, rl, col, mk), off in zip(DREPS, offs):
        per = R[rk]["dnase_hs"]["within_island_stratum"]["per_stratum"]
        for k, (key, _, _) in enumerate(ISL):
            a_ = per[key]["auc"]["True"]
            interval(dx, k + off, a_["auc"], *a_["ci95"], col, True, vertical=True, ms=2.8, marker=mk, label=rl if k == 0 else None)
    prev, nI = res["dnase_prevalence_by_island"], res["dnase_prevalence_by_island_n"]
    dx.set_xticks(xx)
    dx.set_xticklabels([f"{l}\n$n$ = {nI[k]:,}\n{100 * prev[k]:.0f}% HS" for k, l, _ in ISL])
    dx.set_ylabel("AUC for DNase HS,\nkNN within stratum")

    # ---- f: confusion matrix, loadings, island relation (row-normalized, values printed)
    ex = fig.add_subplot(r3[0])
    lx = fig.add_subplot(r3[2])
    cls = [str(c) for c in fd["island_classes"]]
    order = [cls.index(k) for k, _, _ in ISL]
    cm = fd["loadings__island_confusion"][np.ix_(order, order)]
    confusion_matrix_panel(ex, cm, [l for _, l, _ in ISL], normalize=True, aspect="auto")
    ex.set_xlabel("Predicted, loadings 10-NN")
    ex.set_ylabel("Annotated")

    # ---- g: island vote fractions by annotated class (loadings)
    fx = fig.add_subplot(r3[1])
    hist = fd["loadings__island_vote_hist"]  # (annotated class, voted class, votes 0..10)
    ii = cls.index("Island")
    for k, l, col in ISL:
        t = cls.index(k)
        h = hist[t, ii] / hist[t, ii].sum()
        fx.plot(np.arange(11) / 10, h, color=col, lw=1.0, marker="o", ms=2.6, mec=INK, mew=0.3, clip_on=False,
                label=f"{l} ($n$ = {int(hist[t, ii].sum()):,})")
    fx.set_xlabel("Island CpGs among 10 neighbours (fraction)")
    fx.set_ylabel("Fraction of CpGs")

    # ---- e: one-vs-rest AUC by island class
    gx = fig.add_subplot(r2[2])
    GREPS = [r for r in DREPS if r[0] in ("loadings", "h_full", "raw_profiles", "raw_pca64", "baseline")]
    offs = np.linspace(-0.24, 0.24, len(GREPS))
    for (rk, rl, col, mk), off in zip(GREPS, offs):
        au = R[rk]["island_relation"]["auc_one_vs_rest"]
        for k, (key, _, _) in enumerate(ISL):
            interval(gx, k + off, au[key]["auc"], *au[key]["ci95"], col, True, vertical=True, ms=2.8, marker=mk,
                     label=rl if k == 0 else None)
    gx.set_xticks(xx)
    gx.set_xticklabels([l for _, l, _ in ISL])
    gx.set_ylabel("One-vs-rest AUC")

    trim_axes(fig, exclude=ua + [ex])
    for ax in bx:
        ref_line(ax, x=ax._chance)
        pad_categorical_axis(ax, "y", pad=0.6)
    ref_line(cx, y=0)
    ref_line(dx, y=0.5)
    ref_line(gx, y=0.5)
    for ax in (cx, dx, gx):
        pad_categorical_axis(ax, "x", pad=0.5)
    hc, lc = cx.get_legend_handles_labels()
    hc += [Line2D([], [], ls="none", marker="o", ms=3.0, mfc=GRAY["300"], mec=INK, mew=0.3),
           Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=GRAY["400"], mew=0.8)]
    lc += ["Filled: all CpGs", "Open: within mean-beta\ndecile and probe type"]
    hd, ld = dx.get_legend_handles_labels()
    lx.axis("off")
    lopt = dict(frameon=False, fontsize=TYPE["legend"], labelspacing=0.3, handletextpad=0.45, handlelength=1.0, borderaxespad=0.0)
    lg = lx.legend(hc, lc, loc="upper left", bbox_to_anchor=(0.0, 1.0), **lopt)
    lg.set_title("Contrast (c) and kNN (b, c)", prop={"size": TYPE["legend"]}); lg._legend_box.align = "left"
    lx.add_artist(lg)
    lg2 = lx.legend(hd, ld, loc="upper right", bbox_to_anchor=(1.0, 1.0), **lopt)
    lg2.set_title("Representation (d, e)", prop={"size": TYPE["legend"]}); lg2._legend_box.align = "left"
    nature_legend(fx, title="Annotated class", loc="upper center")
    label_panel_group(fig, ua, "a")
    label_panel_group(fig, bx, "b")
    add_panel_labels(fig, {"c": cx, "d": dx, "e": gx, "f": ex, "g": fx})
    fig.savefig(str(HERE / f"{NAME}_draft.png"), dpi=150)
    axd = {**{f"a{i}": a for i, a in enumerate(ua)}, **{f"b{i}": a for i, a in enumerate(bx)},
           "c": cx, "d": dx, "e": gx, "f": ex, "g": fx}
    save_figure(fig, str(HERE / f"{NAME}.pdf"), axes_dict=axd, review_png=str(HERE / f"{NAME}_review.png"),
                footer="data: methylgpt 1.1; EPIC v1.0 B5 manifest; benchmark_cpg_representations_si_2026-10-18.py; artifacts/eval")
    plt.close(fig)


if __name__ == "__main__":
    main()
