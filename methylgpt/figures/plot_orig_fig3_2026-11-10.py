"""Original Figure 3 in the submitted layout and look: the same profiles from complete and 27K-array measurements.

All 39,099 450K and EPIC GEO profiles of the 43,191-profile representation set (11,907 test-split from 104 datasets,
27,192 training-split), each represented twice: from all its measured CpGs (complete copy) and from its 27K-array CpGs
only (27K copy). One PCA (50 components) and one UMAP per representation over both copies of all profiles (Chronos,
fig3_all_cross_array_2026-11-10.py, commit 5d72547). The printed percentage is for the test-split profiles; the
legend also gives all profiles.

Rows: MethylGPT latent scores and raw beta values (vocabulary CpGs, unmeasured CpGs filled with training means).
a, d  both copies coloured by copy (large panels); the printed percentage is the fraction of profiles whose 27K copy
      has its own complete copy as nearest complete profile (cosine distance on 50 principal components)
b, e  both copies coloured by tissue
c, f  lines joining the two copies of 400 randomly drawn profiles

The encoder output of the same analysis and the set D figure (tissue, dataset, array) are in the supplement
(plot_supp_fig26_setD_encoder_2026-11-10.py; Supplementary Table 3).

Data: ../artifacts/figure_tables_2026-11-10/fig3_option2_cross_array_umap_2026-11-10.csv.gz,
      fig3_option2_cross_array_metrics_2026-11-10.csv
Run: MPLBACKEND=Agg python plot_orig_fig3_2026-11-10.py
"""
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from orig_rebuild_common_2026_11_10 import ART, Page, dot_legend, original_images, outlined_cloud, read_csv, style, trim

D = ART / "figure_tables_2026-11-10"
X = read_csv(D / "fig3_all_cross_array_umap_2026-11-10.csv.gz")
M = read_csv(D / "fig3_all_cross_array_metrics_2026-11-10.csv")
N_LINKS = 400

TISSUE = {"Whole Blood": "#4a8bc2", "Cells": "#e15759", "Brain": "#f28e2b", "Colon": "#b6a6d3", "Liver": "#f6c08a",
          "Uterus": "#76b7b2", "Lung": "#edc948", "Muscle": "#59a14f", "Breast": "#9c755f", "Ovary": "#b07aa1"}
OTHER, NOLAB = "#bab0ac", "#dcdcdc"
COPY = {"complete": "#1f6f7a", "27K": "#ef8a3c"}      # deep teal and warm orange (dark teal with a warm accent)
OUTLINE = "#5c5c5c"


def own_copy_fraction(rep, subset="test (primary)"):
    r = M[(M.representation == rep) & (M.profiles == subset) & M.metric.str.startswith("share of profiles whose own complete copy is nearest")]
    if len(r) != 1:
        raise SystemExit(f"Cannot proceed - metric row not found: {rep}")
    return float(r.value.iloc[0])


def two_copies(ax, x, y, s_complete, s_27k):
    """Outlined cloud of both copies: thin grey outline and white body around all points, then complete copies
    (larger) and 27K copies (smaller) on top."""
    k = (X["copy"] == "complete").to_numpy()
    ax.scatter(x, y, s=s_complete * 3.4, c=OUTLINE, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(x, y, s=s_complete * 2.3, c="white", linewidths=0, rasterized=True, zorder=2)
    ax.scatter(x[k], y[k], s=s_complete, c=COPY["complete"], linewidths=0, rasterized=True, zorder=3)
    ax.scatter(x[~k], y[~k], s=s_27k, c=COPY["27K"], linewidths=0, rasterized=True, zorder=4)


def frame(ax, x, y, fs):
    pad = 1.0
    xl = (np.floor(np.min(x) - pad), np.ceil(np.max(x) + pad)); yl = (np.floor(np.min(y) - pad), np.ceil(np.max(y) + pad))
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xticks([t for t in range(-50, 51, 10) if xl[0] <= t <= xl[1]])
    ax.set_yticks([t for t in range(-50, 51, 10) if yl[0] <= t <= yl[1]])
    ax.set_xlabel("UMAP component 1", labelpad=2)
    ax.set_ylabel("UMAP component 2", labelpad=2)
    style(ax, *fs)
    trim(ax)


def block(pg, y0, rep, letters, header):
    x, y = X[f"umap1_{rep}"].values, X[f"umap2_{rep}"].values
    lab = X.tissue.where(X.tissue.isin(list(TISSUE)), np.where(X.tissue == "no tissue label", "No label", "Other tissues"))
    tcol = lab.map({**TISSUE, "Other tissues": OTHER, "No label": NOLAB}).values
    rng = np.random.default_rng(0)
    back = np.where(lab.isin(["No label", "Other tissues"]))[0]
    front = np.where(~lab.isin(["No label", "Other tissues"]))[0]
    order_t = np.r_[rng.permutation(back), rng.permutation(front)]
    order = rng.permutation(len(X))
    pg.text(60, y0 + 16.4, header, 10, bold=True)

    # a, d: both copies coloured by copy (the comparison the figure is about); complete copies large underneath,
    # 27K copies small on top, so a copy that sits on its complete copy shows as an orange core in a teal rim
    ax = pg.axes(38, y0 + 52, 268, y0 + 270)
    two_copies(ax, x, y, s_complete=2.6, s_27k=0.45)   # 78,198 points: smaller marks than for the 10,020-profile version
    frame(ax, x, y, (8, 9))
    pg.text(60, y0 + 38, f"Nearest complete profile is its own, {100 * own_copy_fraction(rep):.0f}%", 7.5)
    dot_legend(pg, 284, y0 + 82, "Measured CpGs", [("Complete profile", COPY["complete"]), ("27K-array CpGs only", COPY["27K"])],
               size=7.5, dy=10.6)

    # b, e: tissue
    ax = pg.axes(424, y0 + 54, 522, y0 + 140)
    outlined_cloud(ax, x, y, tcol, s=0.5, order=order_t)
    frame(ax, x, y, (6.5, 7.5))
    entries = [(t.replace("Whole Blood", "Whole blood"), c) for t, c in TISSUE.items()]
    entries += [("Other tissues", OTHER), ("No label", NOLAB)]
    dot_legend(pg, 533, y0 + 44, "Tissue", entries, size=6, dy=7.3, title_gap=8.5)

    # c, f: the two copies of the same profile joined by a line
    ax = pg.axes(424, y0 + 187, 522, y0 + 273)
    W = X[X.split == "test"].pivot(index="profile_id", columns="copy", values=[f"umap1_{rep}", f"umap2_{rep}"])
    pick = np.random.default_rng(20261110).choice(len(W), N_LINKS, replace=False)
    a = W[[(f"umap1_{rep}", "complete"), (f"umap2_{rep}", "complete")]].values[pick]
    b = W[[(f"umap1_{rep}", "27K"), (f"umap2_{rep}", "27K")]].values[pick]
    ax.scatter(x, y, s=0.2, c="#e6e6e6", linewidths=0, rasterized=True, zorder=1)         # all points as a faint map
    ax.add_collection(LineCollection(np.stack([a, b], axis=1), colors="#8c8c8c", linewidths=0.3, alpha=0.55, zorder=2))
    ax.scatter(a[:, 0], a[:, 1], s=6.5, c=COPY["complete"], linewidths=0, zorder=3)
    ax.scatter(b[:, 0], b[:, 1], s=1.8, c=COPY["27K"], linewidths=0, zorder=4)
    frame(ax, x, y, (6.5, 7.5))
    pg.text(428, y0 + 171, f"Both copies of {N_LINKS} profiles, joined", 7)
    for s, (lx, ly) in zip(letters, ((13.6, 25.4), (385, 36), (385, 169))):
        pg.letter(lx, y0 + ly, s)
    seg = np.hypot(*(a - b).T)
    return dict(representation=rep, own_complete_copy_nearest=own_copy_fraction(rep), profiles=int(len(W)),
                median_link_length_umap=float(np.median(seg)))


def main():
    pg = Page(606)
    imgs = {xr: (a, b) for xr, a, b in original_images(3)}
    pg.place(*imgs[101])                       # DNA -> MethylGPT -> clusters icon
    pg.place(*imgs[97])                        # DNA -> grey clusters icon
    rows = [block(pg, 0, "latent", "abc", "MethylGPT latent-score sample space"),
            block(pg, 297, "raw", "def", "Raw DNA methylation sample space")]
    pd.DataFrame(rows).to_csv(ART.parent / "figures" / "figure3_provenance_2026-11-10" / "panel_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    pg.save("fig3", orig_n=3)


if __name__ == "__main__":
    main()
