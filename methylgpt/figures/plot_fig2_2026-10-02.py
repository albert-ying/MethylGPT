"""Figure 2: CpG representations and genomic annotation.

Inputs (run from this directory):
  ../artifacts/ppca_core_meanfilled_2026-09-30.npz                 factor loadings and variances
  ../artifacts/cpg_annotation_2026-10-01.parquet                   manifest annotation of vocabulary CpGs
  ../artifacts/eval/cpg_representation_benchmark_2026-10-01.json   classification and energy-distance metrics
UMAP (umap-learn, cosine, 30 neighbours, seed 0) of the loadings scaled by factor standard
deviations, on 15,000 CpGs drawn at random (seed 20261002); coordinates cached next to this script.
Run: MPLBACKEND=Agg python plot_fig2_2026-10-02.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,
                                 pad_categorical_axis, set_domain, ref_line, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
REPS = [("factor_loadings", "Input-layer loadings", RUST["400"], "o"), ("encoder_input", "Encoder input weights", RUST["200"], "^"),
        ("raw_profiles", "Raw beta values", BLUE["400"], "s"), ("raw_pca64", "Raw, 64 PCs", BLUE["200"], "s"),
        ("random", "Random vectors", GRAY["300"], "D")]
LABELS = [("island_relation", "CpG island relation"), ("enhancer", "Enhancer"), ("x_chromosome", "Chromosome X")]


def umap_coords():
    cache = HERE / "figure2_umap_coords_2026-10-02.npz"
    if cache.exists():
        z = np.load(cache)
        return z["idx"], z["xy"]
    import umap
    z = np.load(A / "ppca_core_meanfilled_2026-09-30.npz")
    L = z["V"][:, :1536] * np.sqrt(z["lam"][:1536])[None, :]
    idx = np.sort(np.random.default_rng(20261002).choice(len(L), 15000, replace=False))
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(L[idx])
    np.savez(cache, idx=idx, xy=xy)
    return idx, xy


def main():
    ann = pd.read_parquet(A / "cpg_annotation_2026-10-01.parquet")
    idx, xy = umap_coords()
    a = ann.iloc[idx].reset_index(drop=True)
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.62), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.05)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.85])
    r1 = outer[0].subgridspec(1, 3)
    r2 = outer[1].subgridspec(1, 2)
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1]), "c": fig.add_subplot(r1[2]),
           "d": fig.add_subplot(r2[0]), "e": fig.add_subplot(r2[1])}

    def scatter(ax, cats, colors, title):
        for cat, col in colors.items():
            m = cats == cat
            ax.scatter(xy[m, 0], xy[m, 1], s=1.2, color=col, alpha=0.6, edgecolors="none", label=f"{cat} ({m.sum():,})", clip_on=False,
                       rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        nature_legend(ax, title=title, loc="lower left", markerscale=5)

    isl = a.island_relation.fillna("unannotated").replace({"OpenSea": "Open sea"}).to_numpy()
    scatter(axd["a"], isl, {"Island": RUST["400"], "Shore": GOLD["400"], "Shelf": SAGE["400"], "Open sea": BLUE["300"]}, "CpG island")
    enh = np.where(a.in_epic_v1, np.where(a.enhancer, "Enhancer", "Other"), "unannotated")
    scatter(axd["b"], enh, {"Other": GRAY["200"], "Enhancer": RUST["400"]}, "Annotation")
    chx = np.where(a.chr == "X", "Chromosome X", "Autosome")
    scatter(axd["c"], chx, {"Autosome": GRAY["200"], "Chromosome X": RUST["400"]}, "Location")

    res = json.loads((A / "eval/cpg_representation_benchmark_2026-10-01.json").read_text())["results"]
    reps = [r for r in REPS if r[0] in res]
    chance = {"island_relation": 0.25, "enhancer": 0.5, "x_chromosome": 0.5}
    for key, metric, ax, lab in (("knn_balanced_accuracy", "Balanced accuracy, 10-NN", axd["d"], "d"),
                                 ("energy_distance", "Energy distance between classes", axd["e"], "e")):
        xx = np.arange(len(LABELS))
        offs = np.linspace(-0.28, 0.28, len(reps))
        for (rk, rl, c, mk), off in zip(reps, offs):
            ax.scatter(xx + off, [res[rk][l][key] for l, _ in LABELS], s=16, marker=mk, color=c, label=rl,
                       edgecolors="none", clip_on=False, zorder=3)
        if key == "knn_balanced_accuracy":
            for x_, (l, _) in zip(xx, LABELS):
                ax.hlines(chance[l], x_ - 0.38, x_ + 0.38, color=GRAY["400"], lw=0.6, ls=(0, (2, 2)))
        ax.set_xticks(xx); ax.set_xticklabels([n for _, n in LABELS])
        ax.set_ylabel(metric)
        if lab == "e":
            ax.set_yscale("log")
    nature_legend(axd["d"], title="Representation", loc="upper left")
    trim_axes(fig, exclude=[axd["a"], axd["b"], axd["c"]])
    pad_categorical_axis(axd["d"], "x", pad=0.5)
    pad_categorical_axis(axd["e"], "x", pad=0.5)
    add_panel_labels(fig, axd)
    fig.savefig(str(HERE / "figure2_2026-10-02_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "figure2_2026-10-02.pdf"), axes_dict=axd, review_png=str(HERE / "figure2_2026-10-02_review.png"),
                footer="data: MethylGPT input-layer loadings; EPIC v1 / HM450 manifest annotation; artifacts/eval")
    plt.close(fig)


if __name__ == "__main__":
    main()
