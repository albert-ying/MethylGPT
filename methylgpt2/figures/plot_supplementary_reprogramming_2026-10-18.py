#!/usr/bin/env python3
"""Supplementary Figure: reprogramming time course (GSE54848, 30 profiles).

Inputs (written by ../reprogramming_time_course_si_2026-10-18.py)
  ../artifacts/eval/reprogramming_time_course_si_2026-10-18.csv       predicted age per profile and method
  ../artifacts/eval/reprogramming_time_course_si_2026-10-18.pca.csv   PC1/PC2 and distance to the ESC/iPSC centroid
  ../artifacts/eval/reprogramming_time_course_si_2026-10-18.json      PC variance explained, n per day, donor-age search
Panels
  a  MethylGPT predicted age per run and the mean of the three runs, every profile; ESC/iPSC references at the right
  b  elastic net (same labels), Horvath 2013, Horvath skin and blood and the MethylGPT mean, every profile
  c-e  PCA fitted on the 30 profiles: raw beta values, latent scores, full embedding of runs 1-3; coloured by day
  f  distance to the centroid of the three ESC/iPSC references, as a fraction of the mean day-0 distance
  g  pre-specified primary quantity: day 0, days 20 to 49 and the references for every method and run
Donor age is not reported in the local GEO metadata, so no reference line is drawn.
Run: MPLBACKEND=Agg python plot_supplementary_reprogramming_2026-10-18.py   (and again with NATURE_FIGURE_NOFOOTER=1)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, label_panel_group, nature_legend,
                                 save_figure, set_domain, ref_line, add_colorbar, pad_categorical_axis, allow_dead_margin,
                                 RUST, GRAY, BLUE, GOLD, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts" / "eval"
STEM = "supplementary_reprogramming_2026-10-18"
retro_style()

X_ESC, X_IPS, X_SEP = 58.5, 65.0, 54.0
RUNS = ["MethylGPT run 1", "MethylGPT run 2", "MethylGPT run 3"]
MEAN, EN, H13, HSB = "MethylGPT mean of runs", "Elastic net (same labels)", "Horvath 2013", "Horvath skin & blood"
STYLE = {RUNS[0]: ("Run 1", RUST["200"], "o"), RUNS[1]: ("Run 2", RUST["300"], "s"), RUNS[2]: ("Run 3", RUST["500"], "^"),
         MEAN: ("Mean of runs", RUST["400"], "D"), EN: ("Elastic net (same labels)", BLUE["400"], "o"),
         H13: ("Horvath 2013", GOLD["300"], "v"), HSB: ("Horvath skin and blood", GOLD["500"], "s")}
DAY_CMAP = LinearSegmentedColormap.from_list("day_gray", [GRAY["100"], GRAY["300"], GRAY["600"]])
REF_MARK = {"ESC": ("*", 46), "iPSC reference": ("X", 22)}


def xpos(df):
    return np.where(df.group == "ESC", X_ESC, np.where(df.group == "iPSC reference", X_IPS, df.day))


def time_course(ax, df, cols, dodge, title, lw_hero=None):
    x = xpos(df)
    tc = (df.group == "time course").to_numpy()
    for c, dx in zip(cols, dodge):
        lab, col, mk = STYLE[c]
        dm = df[tc].groupby("day")[c].mean()
        hero = c == lw_hero
        ax.plot(dm.index + dx, dm.values, "-", lw=1.2 if hero else 0.8, color=col, zorder=2 + hero, clip_on=False)
        ax.scatter(x + dx, df[c], s=13 if hero else 11, marker=mk, color=col, edgecolors=INK, linewidths=0.3,
                   label=lab, clip_on=False, zorder=4 + hero)
    ax.set_xlabel("Days of reprogramming")
    ax.set_ylabel("Predicted age (years)")
    nature_legend(ax, title=title, loc="upper left", bbox_to_anchor=(0.27, 1.0))   # left of the reference separator


def finish_time_axis(ax, ylim, yticks):
    ax.set_xticks([0, 10, 20, 30, 40, 50, X_ESC, X_IPS])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50", "ESC", "iPSC"])
    ax.set_yticks(yticks)
    set_domain(ax, xlim=(-1.5, X_IPS + 1.5), ylim=ylim)
    ax.spines["bottom"].set_bounds(0, X_IPS)
    ax.spines["left"].set_bounds(yticks[0], yticks[-1])
    ref_line(ax, x=X_SEP)


def main():
    ages = pd.read_csv(A / "reprogramming_time_course_si_2026-10-18.csv")
    pca = pd.read_csv(A / "reprogramming_time_course_si_2026-10-18.pca.csv")
    res = json.loads((A / "reprogramming_time_course_si_2026-10-18.json").read_text())
    assert res["donor_age"]["result"] == "donor age not reported in the local GEO metadata"   # so no reference line
    nday = res["n_per_day"]
    assert set(nday.values()) == {2, 3} and len(ages) == 30

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.97), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.04)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.95, 0.58, 0.80])
    r1 = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    r2 = outer[1].subgridspec(1, 5)
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.12, 1.0])
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1])}
    pax = [fig.add_subplot(r2[k]) for k in range(5)]
    axd.update({"c": pax[0], "d": pax[1], "e": pax[2], "f": fig.add_subplot(r3[0]), "g": fig.add_subplot(r3[1])})

    # a, b: predicted age, every profile (2 or 3 per day), references at the right
    time_course(axd["a"], ages, RUNS + [MEAN], [-0.9, -0.3, 0.3, 0.9], "MethylGPT ($n$ = 2 or 3 per day)", lw_hero=MEAN)
    time_course(axd["b"], ages, [EN, H13, HSB, MEAN], [-0.9, -0.3, 0.3, 0.9], "Method ($n$ = 2 or 3 per day)", lw_hero=MEAN)

    # c-e: PCA of each representation, coloured by day; references with their own symbols
    reps = ["Raw beta values", "Latent scores", "Embedding, run 1", "Embedding, run 2", "Embedding, run 3"]
    titles = [f"Raw beta values\n({res['raw_beta_cpgs_observed_in_all_30_profiles']:,} CpGs)", "Latent scores\n(1,536 dimensions)",
              "Embedding, run 1\n(2,048 dimensions)", "Embedding, run 2\n(2,048 dimensions)", "Embedding, run 3\n(2,048 dimensions)"]
    norm = Normalize(0, 49)
    sc = None
    for ax, rep, ttl in zip(pax, reps, titles):
        p = pca[pca.representation == rep]
        tc = p[p.group == "time course"]
        cen = tc.groupby("day")[["PC1", "PC2"]].mean().sort_index()
        ax.plot(cen.PC1, cen.PC2, "-", lw=0.6, color=GRAY["300"], zorder=1)
        sc = ax.scatter(tc.PC1, tc.PC2, c=tc.day, cmap=DAY_CMAP, norm=norm, s=13, edgecolors=INK, linewidths=0.3,
                        clip_on=False, zorder=3)
        for g, (mk, sz) in REF_MARK.items():
            q = p[p.group == g]
            ax.scatter(q.PC1, q.PC2, marker=mk, s=sz, facecolors="white", edgecolors=INK, linewidths=0.6, clip_on=False, zorder=4,
                       label="ESC, H9 ($n$ = 2)" if g == "ESC" else "iPSC, 201B7 ($n$ = 1)")
        v = res["pca"][rep]
        ax.set_xlabel(f"PC1 ({100 * v['pc1_variance_explained']:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * v['pc2_variance_explained']:.1f}%)")
        ax.set_title(ttl, fontsize=7, linespacing=1.05)
        ax.set_xticks([]); ax.set_yticks([])
        for g, lab in (("ESC", "ESC"), ("iPSC reference", "iPSC")):
            q = p[p.group == g]
            ax.annotate(lab, (q.PC1.min(), q.PC2.mean()), xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=6, color=INK)

    # f: distance to the ESC/iPSC centroid in the full representation space
    ax = axd["f"]
    fstyle = {"Raw beta values": (BLUE["400"], "s", 1.0), "Latent scores": (RUST["400"], "o", 1.0),
              "Embedding, run 1": (RUST["200"], "o", 0.8), "Embedding, run 2": (RUST["200"], "s", 0.8),
              "Embedding, run 3": (RUST["200"], "^", 0.8)}
    for (rep, (col, mk, lw)), dx in zip(fstyle.items(), [-0.8, -0.4, 0.0, 0.4, 0.8]):
        p = pca[pca.representation == rep]
        tc = p[p.group == "time course"]
        dm = tc.groupby("day").distance_fraction_of_day0.mean()
        ax.plot(dm.index + dx, dm.values, "-", lw=lw, color=col, zorder=2, clip_on=False)
        ax.scatter(xpos(p) + dx, p.distance_fraction_of_day0, s=10, marker=mk, color=col, edgecolors=INK, linewidths=0.3,
                   clip_on=False, zorder=3)
    ax.set_xlabel("Days of reprogramming")
    ax.set_ylabel("Distance to ESC/iPSC centroid\n(fraction of day 0)")

    # g: pre-specified primary quantity per method and run
    ax = axd["g"]
    order = RUNS + [MEAN, EN, H13, HSB]
    tc = ages[ages.group == "time course"]
    d0, late = tc[tc.day == 0], tc[tc.day >= 20]
    for j, c in enumerate(order):
        i = len(order) - 1 - j
        lab, col, _ = STYLE[c]
        ax.plot([d0[c].mean(), late[c].mean()], [i, i], "-", lw=0.6, color=col, zorder=1)
        ax.scatter(d0[c], np.full(len(d0), i - 0.2), s=13, marker="o", facecolors="white", edgecolors=col, linewidths=0.8,
                   clip_on=False, zorder=3)
        ax.scatter(late[c], np.full(len(late), i), s=9, marker="o", color=col, edgecolors=INK, linewidths=0.3, alpha=0.9,
                   clip_on=False, zorder=3)
        for g, (mk, sz) in REF_MARK.items():
            q = ages[ages.group == g]
            ax.scatter(q[c], np.full(len(q), i + 0.22), marker=mk, s=sz * 0.8, color=col, edgecolors=INK, linewidths=0.3,
                       clip_on=False, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([("MethylGPT, " + STYLE[c][0].lower()) if c in RUNS + [MEAN] else STYLE[c][0] for c in order][::-1])
    ax.set_xlabel("Predicted age (years)")
    hd = [Line2D([], [], ls="none", marker="o", mfc="white", mec=GRAY["400"], mew=0.8, ms=4, label=f"Day 0 ($n$ = {nday['0']})"),
          Line2D([], [], ls="none", marker="o", mfc=GRAY["400"], mec=INK, mew=0.3, ms=3.5,
                 label=f"Days 20 to 49 ($n$ = {int(late.shape[0])})"),
          Line2D([], [], ls="none", marker="*", mfc=GRAY["400"], mec=INK, mew=0.3, ms=6, label="ESC, H9 ($n$ = 2)"),
          Line2D([], [], ls="none", marker="X", mfc=GRAY["400"], mec=INK, mew=0.3, ms=4.5, label="iPSC, 201B7 ($n$ = 1)")]
    lg = ax.legend(handles=hd, loc="upper right", frameon=False, fontsize=7, labelspacing=0.3, handletextpad=0.45,
                   handlelength=1.0, borderpad=0.25, borderaxespad=0.35, title="Profiles")
    lg.get_title().set_fontsize(7); lg.get_title().set_color(INK); lg._legend_box.align = "left"

    trim_axes(fig, exclude=pax)
    finish_time_axis(axd["a"], (-2, 61.5), [0, 10, 20, 30, 40, 50, 60])
    finish_time_axis(axd["b"], (-2, 61.5), [0, 10, 20, 30, 40, 50, 60])
    allow_dead_margin(axd["a"], "y", reason="domain shared with panel b so MethylGPT and the clocks read on one scale")
    finish_time_axis(axd["f"], (0, 1.08), [0, 0.25, 0.5, 0.75, 1.0])
    axd["f"].set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    # direct labels instead of a legend (three curves; the three embedding runs coincide)
    for x_, y_, t_, c_, va in ((34, 0.785, "Embedding, runs 1 to 3", RUST["300"], "bottom"),
                               (40, 0.35, "Raw beta values", BLUE["400"], "bottom"),
                               (40, 0.175, "Latent scores", RUST["500"], "top")):
        axd["f"].text(x_, y_, t_, color=c_, fontsize=6.5, ha="center", va=va)
    axd["g"].set_xticks([0, 10, 20, 30, 40, 50, 60])
    set_domain(axd["g"], xlim=(-2, 61.5))
    pad_categorical_axis(axd["g"], "y", pad=0.5)
    for ax in pax:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    add_colorbar(sc, pax[-1], label="Day", ticks=[0, 20, 49], inset=(1.06, 0.18, 0.06, 0.64))

    add_panel_labels(fig, {k: v for k, v in axd.items() if k != "e"})
    label_panel_group(fig, pax[2:], "e")
    fig.savefig(str(HERE / f"{STEM}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{STEM}.pdf"), axes_dict=axd, review_png=str(HERE / f"{STEM}_review.png"),
                footer="data: GSE54848 (450K, 30 profiles); artifacts/eval/reprogramming_time_course_si_2026-10-18")
    plt.close(fig)


if __name__ == "__main__":
    main()
