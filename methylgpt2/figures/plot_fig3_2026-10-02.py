"""Figure 3: sample representations of held-out datasets.

Inputs (run from this directory):
  ../artifacts/eval/representations_setA_2026-10-01.npz         cached set A representations (row ids)
  ../artifacts/eval/representation_benchmark_2026-10-01.json     metrics, natural geometry
  ../artifacts/eval/representation_benchmark_missing_2026-10-02.json   incomplete profiles; Harmony on factor scores
  ../artifacts/corpus_2026-09-30/profiles.parquet                tissue labels
UMAPs (umap-learn, 30 neighbours, seed 0) of the first 50 principal components of MethylGPT
factor scores and of raw beta values; coordinates cached next to this script.
Run: MPLBACKEND=Agg python plot_fig3_2026-10-02.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,
                                 pad_categorical_axis, set_domain, allow_dead_margin,
                                 RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CATEGORICAL_VIVID)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
TISSUE_COL = {"Whole Blood": RUST["400"], "Brain": BLUE["400"], "Colon": GOLD["400"], "Liver": SAGE["400"],
              "Muscle": PLUM["400"], "Adipose": GRAY["400"]}


def umaps():
    cache = HERE / "figure3_umap_coords_2026-10-02.npz"
    if cache.exists():
        z = np.load(cache)
        return z["rows"], z["fs"], z["raw"]
    import umap
    z = np.load(A / "eval/representations_setA_2026-10-01.npz")
    out = {}
    for k, X in (("fs", z["rep_factor_scores"]), ("raw", z["raw_full"])):
        P = PCA(50, random_state=0).fit_transform(X - X.mean(0))
        out[k] = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(P)
    np.savez(cache, rows=z["rows"], **out)
    return z["rows"], out["fs"], out["raw"]


def main():
    prof = pd.read_parquet(A / "corpus_2026-09-30/profiles.parquet", columns=["tissue"])
    rows, fs, raw = umaps()
    tissue = prof.tissue.to_numpy()[rows]
    res = json.loads((A / "eval/representation_benchmark_2026-10-01.json").read_text())["natural"]
    mis = json.loads((A / "eval/representation_benchmark_missing_2026-10-02.json").read_text())

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.66), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.05)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95])
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    r2 = outer[1].subgridspec(1, 2, width_ratios=[1.15, 1.0])
    re_ = r2[1].subgridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.12)
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1]), "c": fig.add_subplot(r1[2]),
           "d": fig.add_subplot(r2[0]), "e": fig.add_subplot(re_[0])}
    e2 = fig.add_subplot(re_[1], sharex=axd["e"])

    for key, xy, title in (("a", fs, "MethylGPT latent scores"), ("b", raw, "Raw beta values")):
        ax = axd[key]
        for t, col in TISSUE_COL.items():
            m = tissue == t
            ax.scatter(xy[m, 0], xy[m, 1], s=3, color=col, alpha=0.7, edgecolors="none", label=f"{t} ({m.sum()})",
                       clip_on=False, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if key == "a":   # the two UMAPs are separate embeddings; axes labelled once
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(title, fontsize=7, fontweight="normal")
    nature_legend(axd["a"], title="Tissue", loc="lower left", markerscale=3)

    # c: tissue silhouette vs sex balanced accuracy, set A
    ax = axd["c"]
    reps = [("raw", "Raw beta", BLUE["400"], "s"), ("raw_pca50", "Raw, 50 PCs", BLUE["200"], "s"),
            ("factor_scores", "Latent scores", RUST["400"], "o"), ("mgpt2_multitask", "Embedding", RUST["200"], "o"),
            ("untrained_transformer", "Untrained transformer", PLUM["400"], "^"), ("random_projection", "Random projection", GRAY["300"], "D")]
    for k, lab, c, mk in reps:
        v = res["set_a"][k]
        ax.scatter(v["tissue_silhouette"], v["sex_knn_bacc"], s=22, marker=mk, color=c, label=lab, edgecolors="none", clip_on=False, zorder=3)
    ax.set_xlabel("Tissue silhouette"); ax.set_ylabel("Sex balanced accuracy, 10-NN")
    nature_legend(ax, title="Representation", loc="upper left")

    # d: mixing vs retained biology, blood batches
    ax = axd["d"]
    pts = [("Raw beta", res["set_b"]["raw"], BLUE["400"], "s"), ("Latent scores", res["set_b"]["factor_scores"], RUST["400"], "o"),
           ("Embedding", res["set_b"]["mgpt2_multitask"], RUST["200"], "o"), ("ComBat, adjusted CpGs", res["set_b"]["combat_adjusted_cpgs"], SAGE["500"], "v"),
           ("Harmony on raw PCs", res["set_b"]["harmony_pca50"], SAGE["400"], "^")]
    for lab, v, c, mk in pts:
        open_ = lab.startswith("Harmony")   # drawn open so the factor-score Harmony point stays visible on top
        ax.scatter(v["batch_knn_bacc"], v["age_r2_ridge"], s=44 if open_ else 26, marker=mk, label=lab, clip_on=False, zorder=3,
                   **({"facecolors": "white", "edgecolors": c, "linewidths": 1.0} if open_ else {"color": c, "edgecolors": "none"}))
    hf = mis["set_b"]["MethylGPT factor scores | Harmony"]
    ax.scatter(hf["batch_knn_bacc"], hf["age_r2_ridge"], s=16, marker="^", color=SAGE["600"], edgecolors="none",
               label="Harmony on latent-score PCs", clip_on=False, zorder=4)
    ax.set_xlabel("Dataset identifiable (balanced accuracy, 10-NN)"); ax.set_ylabel("Age $R^2$ (cross-validated)")
    nature_legend(ax, title="Whole blood, 10 datasets", loc="upper left")

    # e: incomplete profiles; x = mean fraction of vocabulary CpGs observed (all CpGs, 27K-array CpGs, 70% removed)
    scen = ["random_70", "hm27", "all"]
    obs = [100 * mis["set_a"][s]["raw (mean-filled)"]["observed_fraction"] for s in scen]
    for name, lab, c, mk in (("raw (mean-filled)", "Raw beta, missing filled with means", BLUE["400"], "s"),
                             ("MethylGPT factor scores", "Latent scores", RUST["400"], "o")):
        for ax, metric in ((axd["e"], "sex_knn_bacc"), (e2, "tissue_silhouette")):
            ax.plot(obs, [mis["set_a"][s][name][metric] for s in scen], "-", marker=mk, ms=3.5, lw=0.9, color=c, label=lab, clip_on=False)
    axd["e"].set_ylabel("Sex, 10-NN\nbalanced accuracy"); e2.set_ylabel("Tissue\nsilhouette")
    e2.set_xlabel("CpGs observed (% of vocabulary)")
    axd["e"].tick_params(labelbottom=False)
    for x_, s in zip(obs, ("70% removed", "27K array", "All")):
        axd["e"].text(x_, 0.955, s, fontsize=6, ha="center", va="bottom", color=INK)
    fs_mid = np.interp(65, obs, [mis["set_a"][s]["MethylGPT factor scores"]["sex_knn_bacc"] for s in scen])
    raw_mid = np.interp(65, obs, [mis["set_a"][s]["raw (mean-filled)"]["sex_knn_bacc"] for s in scen])
    axd["e"].text(65, fs_mid - 0.006, "Latent scores", fontsize=6, ha="center", va="top", color=RUST["500"])
    axd["e"].text(65, raw_mid - 0.006, "Raw beta, missing values\nfilled with training means", fontsize=6, ha="center", va="top", color=BLUE["500"])

    trim_axes(fig, exclude=[axd["a"], axd["b"]])
    set_domain(axd["e"], ylim=(0.85, 0.95)); axd["e"].set_yticks([0.86, 0.90, 0.94]); axd["e"].set_yticklabels(["0.86", "0.90", "0.94"])
    set_domain(e2, ylim=(0.43, 0.48)); e2.set_yticks([0.44, 0.46, 0.48]); e2.set_yticklabels(["0.44", "0.46", "0.48"])
    add_panel_labels(fig, axd)
    fig.savefig(str(HERE / "figure3_2026-10-02_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "figure3_2026-10-02.pdf"), axes_dict=axd, review_png=str(HERE / "figure3_2026-10-02_review.png"),
                footer="data: test-split datasets of the MethylGPT corpus; artifacts/eval")
    plt.close(fig)


if __name__ == "__main__":
    main()
