"""Supplementary Figure 26: encoder output and raw beta values of set D (43,191 GEO profiles, 1,897 datasets), submitted layout.

Rows: MethylGPT encoder output (the 512 transformer dimensions of the released embedding, run 1) and raw beta
values (vocabulary CpGs, unmeasured CpGs filled with training means). Each row: a large UMAP coloured by tissue
and two small UMAPs coloured by dataset and by array, as the submitted figure had tissue, batch and sex.

Numbers on the small panels are 10-nearest-neighbour balanced accuracies within whole blood from fig3_metrics:
dataset (stratified fivefold) and array (each dataset held out). Lower values mean more mixing. Tissue accuracy
and the other representations are reported in Supplementary Table 3 and the text, not on the canvas.

The set A version of this script (six tissues, latent scores) is briefs_2026-11-10/plot_orig_fig3_setA_backup_2026-11-10.py.

Data: ../artifacts/figure_tables_2026-11-10/fig3_setD_umap_2026-11-10.csv.gz, fig3_metrics_2026-11-10.csv
Run: MPLBACKEND=Agg python plot_orig_fig3_2026-11-10.py
"""
import numpy as np
import pandas as pd

from orig_rebuild_common_2026_11_10 import ART, Page, dot_legend, original_images, outlined_cloud, read_csv, style, trim

D = ART / "figure_tables_2026-11-10"
S = read_csv(D / "fig3_setD_umap_2026-11-10.csv.gz")
M = read_csv(D / "fig3_metrics_2026-11-10.csv").query("set == 'D'")

# tissue colours: the submitted figure's hues first, then further distinct hues; greys for the rest
TISSUE = {"Whole Blood": "#4a8bc2", "Brain": "#f28e2b", "Cells": "#e15759", "Colon": "#b6a6d3", "Breast": "#76b7b2",
          "Liver": "#f6c08a", "Lung": "#edc948", "Prostate": "#9c755f", "Skin": "#ff9da7", "Muscle": "#59a14f",
          "Ovary": "#b07aa1", "Adipose": "#8cd17d"}
OTHER, NOLAB = "#bab0ac", "#dcdcdc"
ARRAY = {"27K": "#59a14f", "450K": "#4a8bc2", "EPIC": "#f28e2b", "other": "#c8c8c8"}
DS_HUES = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#e15759", "#76b7b2", "#edc948", "#9c755f"]


def metric(rep, name, cv, profiles):
    r = M[(M.representation == rep) & (M.metric == name) & (M.cv == cv) & (M.profiles == profiles)]
    if len(r) != 1:
        raise SystemExit(f"Cannot proceed - metric row not found: {rep} {name} {cv} {profiles}")
    return float(r.value.iloc[0])


def tissue_colours():
    lab = S.tissue.where(S.tissue.isin(list(TISSUE)), np.where(S.tissue == "no tissue label", "No label", "Other tissues"))
    cols = lab.map({**TISSUE, "Other tissues": OTHER, "No label": NOLAB}).values
    return lab, cols


def dataset_colours():
    ds = S.dataset.unique()
    order = np.random.default_rng(20261110).permutation(len(ds))
    return S.dataset.map({d: DS_HUES[i % len(DS_HUES)] for d, i in zip(ds, order)}).values


def cloud(pg, box, x, y, colours, s, order, fs):
    ax = pg.axes(*box)
    outlined_cloud(ax, x, y, colours, s=s, order=order)
    pad = 1.0
    xl = (np.floor(np.min(x) - pad), np.ceil(np.max(x) + pad)); yl = (np.floor(np.min(y) - pad), np.ceil(np.max(y) + pad))
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    step = 5 if max(xl[1] - xl[0], yl[1] - yl[0]) <= 30 else 10
    ax.set_xticks([t for t in range(-50, 51, step) if xl[0] <= t <= xl[1]])
    ax.set_yticks([t for t in range(-50, 51, step) if yl[0] <= t <= yl[1]])
    ax.set_xlabel("UMAP component 1", labelpad=2)
    ax.set_ylabel("UMAP component 2", labelpad=2)
    style(ax, *fs)
    trim(ax)
    return ax


def block(pg, y0, rep, letters, header, legends):
    x, y = S[f"umap1_{rep}"].values, S[f"umap2_{rep}"].values
    lab, tcol = tissue_colours()
    back = np.where(lab.isin(["No label", "Other tissues"]))[0]
    front = np.where(~lab.isin(["No label", "Other tissues"]))[0]
    rng = np.random.default_rng(0)
    order_t = np.r_[rng.permutation(back), rng.permutation(front)]
    order = rng.permutation(len(S))
    pg.text(60, y0 + 16.4, header, 10, bold=True)
    cloud(pg, (38, y0 + 52, 268, y0 + 270), x, y, tcol, 1.0, order_t, (8, 9))
    if legends:
        n = lab.value_counts()
        entries = [(f"{t.replace('Whole Blood', 'Whole blood')} ({n[t]:,})", c) for t, c in TISSUE.items()]
        entries += [(f"Other tissues ({n['Other tissues']:,})", OTHER), (f"No label ({n['No label']:,})", NOLAB)]
        dot_legend(pg, 284, y0 + 70, "Tissue", entries, size=7.5, dy=10.6)
    acc_d = metric(rep, "dataset 10-NN balanced accuracy", "stratified5", "whole blood, datasets")
    acc_a = metric(rep, "array 10-NN balanced accuracy", "lodo", "whole blood, arrays")
    cloud(pg, (424, y0 + 54, 522, y0 + 140), x, y, dataset_colours(), 0.5, order, (6.5, 7.5))
    pg.text(428, y0 + 38, f"Dataset 10-NN accuracy {acc_d:.2f}", 7)
    cloud(pg, (424, y0 + 187, 522, y0 + 273), x, y, S.array.map(ARRAY).values, 0.5, order, (6.5, 7.5))
    pg.text(428, y0 + 171, f"Array 10-NN accuracy {acc_a:.2f}", 7)
    if legends:
        na = S.array.value_counts()
        entries = [(f"{a} ({na[a]:,})", ARRAY[a]) for a in ("27K", "450K", "EPIC")]
        if na.get("other", 0):                 # profiles on other platforms are drawn in grey and must be in the key
            entries.append((f"Other ({na['other']:,})", ARRAY["other"]))
        dot_legend(pg, 533, y0 + 190, "Array", entries, size=6.5, dy=9, title_gap=10)
    for s, (lx, ly) in zip(letters, ((13.6, 25.4), (385, 36), (385, 169))):
        pg.letter(lx, y0 + ly, s)
    return dict(representation=rep, dataset_accuracy=acc_d, array_accuracy_dataset_held_out=acc_a)


def main():
    pg = Page(606)
    imgs = {xr: (a, b) for xr, a, b in original_images(3)}
    pg.place(*imgs[101])                       # DNA -> MethylGPT -> clusters icon
    pg.place(*imgs[97])                        # DNA -> grey clusters icon
    rows = [block(pg, 0, "encoder", "abc", "MethylGPT encoder sample space", True),
            block(pg, 297, "raw", "def", "Raw DNA methylation sample space", True)]
    pd.DataFrame(rows).to_csv(ART.parent / "figures" / "figure3_provenance_2026-11-10" / "panel_metrics.csv", index=False)
    pg.save("supp_fig26_setD_encoder")


if __name__ == "__main__":
    main()
