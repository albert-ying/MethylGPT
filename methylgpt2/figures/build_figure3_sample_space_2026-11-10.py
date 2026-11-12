"""Figure 3. Sample representations of 43,191 public profiles from 1,897 datasets.

Layout of the submitted figure: a MethylGPT row and a raw-methylation row, each with a large UMAP coloured by
tissue and two small UMAPs coloured by array and by dataset.

MethylGPT row: the 512 transformer dimensions of the released embedding (run 1, after the age, sex, tissue and
clock stage), called the encoder output. Raw row: beta values of the 49,156 vocabulary CpGs, unmeasured CpGs
filled with training means. Each: first 50 principal components, UMAP (30 neighbours, min_dist 0.3, seed 0),
computed on Chronos (export_figure_tables_2026-11-10.py).

Numbers in the panels are 10-nearest-neighbour balanced accuracies (fig3_metrics, set D): tissue (stratified
fivefold, tissues with at least 30 profiles in each of at least 2 datasets), array and dataset (whole blood;
array with each dataset held out, dataset stratified fivefold). Higher tissue accuracy keeps biology; lower array
and dataset accuracy means more mixing. The encoder output mixes arrays and datasets and keeps less tissue
resolution than raw values; both are shown.

Sources: ../artifacts/figure_tables_2026-11-10/fig3_setD_umap_2026-11-10.csv.gz, fig3_metrics_2026-11-10.csv;
original_submission/fig3.pdf (row icons).
Run: MPLBACKEND=Agg python build_figure3_sample_space_2026-11-10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from nature_figure_utils import (retro_style, save_figure, mark_pictorial, categorical_colors, allow_dead_margin,
                                 GRAY, BLUE, RUST, GOLD, SAGE, PLUM, INK, TYPE)

HERE = Path(__file__).resolve().parent
D = HERE.parent / "artifacts/figure_tables_2026-11-10"
PROV = HERE / "figure3_provenance_2026-11-10"
retro_style()
TXT = TYPE["tick"]


def read(name, **kw):
    p = D / name
    if not p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return pd.read_csv(p, **kw)


S = read("fig3_setD_umap_2026-11-10.csv.gz")
M = read("fig3_metrics_2026-11-10.csv").query("set == 'D'")
REPS = [("encoder", "MethylGPT encoder"), ("raw", "Raw methylation")]
N_TISSUES = 12


def metric(rep, metric, cv, profiles):
    r = M[(M.representation == rep) & (M.metric == metric) & (M.cv == cv) & (M.profiles == profiles)]
    if len(r) != 1:
        sys.exit(f"Cannot proceed - metric row not found: {rep} {metric} {cv} {profiles}")
    return float(r.value.iloc[0])


def tissue_classes():
    named = [t for t in S.tissue.value_counts().index if t not in ("Other", "no tissue label")][:N_TISSUES]
    cols = dict(zip(named, categorical_colors(N_TISSUES, vivid=True)))
    lab = S.tissue.where(S.tissue.isin(named), np.where(S.tissue == "no tissue label", "No label", "Other tissues"))
    cols["Other tissues"] = GRAY["300"]
    cols["No label"] = GRAY["100"]
    return lab, cols, named


def array_colors():
    return {"27K": SAGE["400"], "450K": BLUE["300"], "EPIC": RUST["300"], "other": GRAY["200"]}


def dataset_colors():
    """One colour per dataset from a fixed eight-hue cycle in a seeded random order: neighbouring points of the
    same colour mostly share a dataset, so islands read as single colours and mixed regions as speckle."""
    hues = [BLUE["300"], RUST["300"], SAGE["400"], GOLD["300"], PLUM["300"], BLUE["500"], RUST["500"], GRAY["400"]]
    ds = S.dataset.unique()
    order = np.random.default_rng(20261110).permutation(len(ds))
    return S.dataset.map({d: hues[i % len(hues)] for d, i in zip(ds, order)})


def scatter(ax, rep, colours, order, s):
    x, y = S[f"umap1_{rep}"].values, S[f"umap2_{rep}"].values
    ax.scatter(x[order], y[order], s=s, c=np.asarray(colours, dtype=object)[order].tolist(), linewidths=0, alpha=0.85,
               rasterized=True, clip_on=False)
    pad = 0.03 * max(np.ptp(x), np.ptp(y))
    ax.set_xlim(x.min() - pad, x.max() + pad); ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    allow_dead_margin(ax, "both", reason="UMAP coordinates carry no scale")


def axis_arrows(ax):
    """Short UMAP axis stubs in the lower-left corner instead of a frame."""
    for (dx, dy), lab, rot in (((0.16, 0), "UMAP 1", 0), ((0, 0.16), "UMAP 2", 90)):
        ax.annotate("", xy=(dx, dy), xytext=(0, 0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0.0, -0.035, "UMAP 1", transform=ax.transAxes, fontsize=TXT, ha="left", va="top", color=INK)
    ax.text(-0.035, 0.0, "UMAP 2", transform=ax.transAxes, fontsize=TXT, ha="right", va="bottom", rotation=90, color=INK)


def icon(fig, xref, rect):
    doc = fitz.open(HERE / "original_submission/fig3.pdf")
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    ax = fig.add_axes(rect)
    ax.imshow(arr, interpolation="antialiased"); ax.set_axis_off(); mark_pictorial(ax)
    return ax


FW, FH = 7.2, 6.1


def build():
    fig = plt.figure(figsize=(FW, FH))

    def A(x, y, w, h):
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])

    tis, tcol, named = tissue_classes()
    acol = array_colors()
    dcol = dataset_colors()
    rng = np.random.default_rng(0)
    back = np.where(tis.isin(["No label", "Other tissues"]))[0]
    front = np.where(~tis.isin(["No label", "Other tissues"]))[0]
    order_t = np.r_[rng.permutation(back), rng.permutation(front)]
    order = rng.permutation(len(S))
    axd, rows = {}, []
    for k, (rep, title) in enumerate(REPS):
        y0 = 0.08 + k * 3.02
        big = A(0.42, y0 + 0.36, 3.0, 2.5)
        small1 = A(5.15, y0 + 0.36, 1.55, 1.18)
        small2 = A(5.15, y0 + 1.72, 1.55, 1.18)
        scatter(big, rep, tis.map(tcol).values, order_t, 1.1)
        scatter(small1, rep, S.array.map(acol).values, order, 0.5)
        scatter(small2, rep, dcol.values, order, 0.5)
        axis_arrows(big)
        acc_t = metric(rep, "tissue 10-NN balanced accuracy", "stratified5", "tissue rule 1")
        acc_a = metric(rep, "array 10-NN balanced accuracy", "lodo", "whole blood, arrays")
        acc_d = metric(rep, "dataset 10-NN balanced accuracy", "stratified5", "whole blood, datasets")
        big.text(0.0, 1.0, f"Tissue accuracy {acc_t:.2f}", transform=big.transAxes, fontsize=TXT, ha="left", va="top", color=INK)
        small1.set_title(f"Array, accuracy {acc_a:.2f}", loc="left", fontsize=TXT, fontweight="normal", pad=2)
        small2.set_title(f"Dataset, accuracy {acc_d:.2f}", loc="left", fontsize=TXT, fontweight="normal", pad=2)
        fig.text(0.3 / FW, 1 - (y0 + 0.14) / FH, title, fontsize=TYPE["panel_title"], fontweight="bold", ha="left", va="center",
                 color=INK)
        icon(fig, 101 if rep == "encoder" else 97, [1.85 / FW, 1 - (y0 + 0.42) / FH, 0.9 / FW, 0.42 / FH])
        L = "abc" if k == 0 else "def"
        axd.update({L[0]: big, L[1]: small1, L[2]: small2})
        rows.append(dict(representation=rep, tissue_accuracy=acc_t, array_accuracy_dataset_held_out=acc_a,
                         dataset_accuracy=acc_d))
        if k == 0:                                   # one tissue legend for both rows
            n = tis.value_counts()
            h = [Line2D([], [], ls="", marker="o", ms=4, mfc=tcol[t], mec="none", label=f"{t.replace('Whole Blood', 'Whole blood')} ({n[t]:,})")
                 for t in named + ["Other tissues", "No label"]]
            fig.legend(handles=h, title="Tissue", loc="center left", bbox_to_anchor=(3.62 / FW, 1 - 3.05 / FH), frameon=False,
                       fontsize=TXT, title_fontsize=TXT, handletextpad=0.3, labelspacing=0.32, alignment="left", borderaxespad=0)
            na = S.array.value_counts()
            ha = [Line2D([], [], ls="", marker="o", ms=4, mfc=acol[a], mec="none", label=f"{a} ({na[a]:,})") for a in ("27K", "450K", "EPIC")]
            small1.legend(handles=ha, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=TXT, handletextpad=0.2,
                          labelspacing=0.3, borderaxespad=0)
    for letter, ax in axd.items():
        bb = ax.get_position()
        if letter in "ad":                          # letter on the line of the row header
            y = 1 - (0.08 + (0 if letter == "a" else 3.02) + 0.14) / FH
            fig.text(0.04 / FW, y, letter, fontsize=TYPE["panel_letter"], fontweight="bold", color="black", ha="left", va="center")
        else:
            fig.text(bb.x0 - 0.22 / FW, bb.y1 + 0.16 / FH, letter, fontsize=TYPE["panel_letter"], fontweight="bold", color="black",
                     ha="left", va="bottom")
    PROV.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(PROV / "panel_metrics.csv", index=False)
    pd.DataFrame(dict(class_=tis.value_counts().index, n=tis.value_counts().values)).to_csv(PROV / "tissue_classes.csv", index=False)
    return fig, axd


if __name__ == "__main__":
    fig, axd = build()
    if "--draft" in sys.argv:
        fig.savefig(HERE / "figure3_draft.png", dpi=170, facecolor="white")
        sys.exit(0)
    save_figure(fig, HERE / "figure3_sample_space_2026-11-10.pdf", axes_dict=axd, dpi=600,
                review_png=HERE / "figure3_sample_space_2026-11-10_review.png")
    fig.savefig(HERE / "figure3_sample_space_2026-11-10.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("saved")
