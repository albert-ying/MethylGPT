"""Original Figure 2 rebuilt on the submitted artwork with the rebuilt model's CpG loadings.

a  submitted schematic with its labels changed to the rebuilt model (training profiles, probabilistic
   input layer, CpG loadings of 1,536 dimensions for 49,156 CpGs); the return arrow to the genome is labelled
   as the comparison with genomic annotation
b-d UMAP of the input-layer loadings of 15,000 vocabulary CpGs, coloured by relation to CpG islands,
   enhancer annotation and chromosome X (same plot type and position as submitted)

Data: ../artifacts/figure_tables_2026-10-23/table1_fig2bcd_cpg_umap.csv.gz
Run: MPLBACKEND=Agg python plot_orig_fig2_2026-11-10.py
"""
import numpy as np
from orig_rebuild_common_2026_11_10 import ART, Page, dot_legend, original_images, outlined_cloud, read_csv, style, trim

T = read_csv(ART / "figure_tables_2026-10-23/table1_fig2bcd_cpg_umap.csv.gz")
BLUE, RED, YEL, OLV, GREY = "#6baed6", "#c9556e", "#e2b340", "#8a8f5c", "#b8b8b8"


def patch(arr, box, colour=None):
    """Cover a pixel box of the schematic with its surrounding colour (labels are redrawn as text)."""
    x0, y0, x1, y1 = box
    if colour is None:
        ring = np.concatenate([arr[y0 - 3:y0, x0:x1].reshape(-1, 3), arr[y1:y1 + 3, x0:x1].reshape(-1, 3)])
        colour = np.median(ring, axis=0)
    arr[y0:y1, x0:x1] = colour


def panel_a(pg):
    xref, arr, bb = [i for i in original_images(2) if i[0] == 86][0]
    arr = arr.copy()
    sx = arr.shape[1] / (bb[2] - bb[0])
    sy = arr.shape[0] / (bb[3] - bb[1])

    def px(x0, y0, x1, y1):  # page points -> pixel box
        return (int((x0 - bb[0]) * sx), int((y0 - bb[1]) * sy), int((x1 - bb[0]) * sx), int((y1 - bb[1]) * sy))

    white = np.array([255, 255, 255])
    patch(arr, px(16.5, 133, 103, 146), white)        # 'Raw CpG embedding'
    patch(arr, px(170, 133, 284, 146), white)        # 'Contextualized CpG embedding'
    patch(arr, px(274, 152, 290, 210), white)        # '64 dimension'
    patch(arr, px(200, 216, 258, 230), white)        # '49K CpG sites'
    patch(arr, px(112, 175.5, 164, 197))               # 'Transformer blocks'
    patch(arr, px(205, 97.5, 241, 116), white)        # 'Biological context' (return arrow)
    pg.place(arr, bb)
    # the return arrow marks the comparison of the loadings with genomic annotation (Fig. 2b-d); nothing is fed back
    pg.text(236, 98.0, "Compared with", 7, color="#9c5c97", ha="center")
    pg.text(236, 106.5, "genomic annotation", 7, color="#9c5c97", ha="center")
    pg.text(58, 135.5, "145,700 training profiles", 7.5, color="#8c8c8c", ha="center")
    pg.text(227, 135.5, "CpG loadings", 7.5, color="#a0509b", ha="center")
    pg.text(137, 178.5, "Probabilistic", 6.3, bold=True, ha="center")
    pg.text(137, 186.5, "input layer", 6.3, bold=True, ha="center")
    pg.fig.text(281.5 / 595.276, 1 - 181 / pg.ph, "1,536 dimensions", fontsize=7, rotation=-90, ha="center",
                va="center", color="#231f20")
    pg.text(228.5, 218.5, "49,156 CpGs", 7.5, ha="center")


def cloud_panel(pg, box, colours, order):
    ax = pg.axes(*box)
    outlined_cloud(ax, T.umap1.values, T.umap2.values, colours, s=1.6, order=order)
    ax.set_xlim(-5, 15)
    ax.set_ylim(-3, 17)
    ax.set_xticks(range(-5, 16, 5))
    ax.set_yticks(range(0, 16, 5))
    ax.set_xlabel("UMAP component 1", labelpad=2)
    ax.set_ylabel("UMAP component 2", labelpad=2)
    style(ax)
    trim(ax)
    return ax


def main():
    pg = Page(422)
    panel_a(pg)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(T))
    # b: relation to CpG island
    isl = T.island_relation.fillna("NA")
    cmap = {"Island": BLUE, "OpenSea": RED, "Shore": YEL, "Shelf": OLV, "NA": GREY}
    cloud_panel(pg, (345, 58, 493, 206), isl.map(cmap).values, order)
    n = isl.value_counts()
    dot_legend(pg, 512, 70, "Relation to CpG island",
               [(f"Island ({n['Island']:,})", BLUE), (f"Open sea ({n['OpenSea']:,})", RED),
                (f"Shore ({n['Shore']:,})", YEL), (f"Shelf ({n['Shelf']:,})", OLV),
                (f"Not annotated ({n['NA']:,})", GREY)], size=7)
    pg.text(434, 18.4, "CpG loadings of the input layer", 10, bold=True, ha="center")
    pg.text(434, 33, "15,000 of 49,156 CpGs", 7.5, ha="center")
    # c: enhancer; annotated enhancers drawn last so that they are not hidden
    enh = T.enhancer.map({True: "Yes", False: "No"}).fillna("NA")
    o2 = np.concatenate([order[(enh.values[order] != "Yes")], order[(enh.values[order] == "Yes")]])
    cloud_panel(pg, (62, 243, 210, 391), enh.map({"Yes": RED, "No": BLUE, "NA": GREY}).values, o2)
    n = enh.value_counts()
    dot_legend(pg, 229, 252, "Enhancer",
               [(f"No ({n['No']:,})", BLUE), (f"Yes ({n['Yes']:,})", RED), (f"Not annotated ({n['NA']:,})", GREY)], size=7)
    # d: chromosome X
    cx = T.chromosome_x.astype(bool).values
    o3 = np.concatenate([order[~cx[order]], order[cx[order]]])
    cloud_panel(pg, (345, 243, 493, 391), np.where(cx, RED, BLUE), o3)
    dot_legend(pg, 512, 252, "Chromosome",
               [(f"Autosome ({(~cx).sum():,})", BLUE), (f"X ({cx.sum():,})", RED)], size=7)
    for s, x, y in (("a", 13.1, 9.1), ("b", 297.5, 9.1), ("c", 15.9, 222.6), ("d", 297.5, 222.6)):
        pg.letter(x, y, s)
    pg.save("fig2", orig_n=2)


if __name__ == "__main__":
    main()
