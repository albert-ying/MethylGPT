"""Original Figure 5 rebuilt in the submitted layout, with attribution of the age output in place of attention.

a  submitted schematic; the CpG-by-CpG attention matrices are replaced by a per-CpG attribution readout
   (illustration, no data)
b  per-CpG mean absolute gradient x input attribution of the age output compared between three age groups
   of the 5,811 held-out profiles (<20, 20 to <60, >=60 years)
c  old (>=60 years) against young (<30 years) attribution within whole blood: log2 ratio and Mann-Whitney P
d  the ten CpGs with the largest fall and the ten with the largest rise in attribution from <20 to >=60 years
   (same direction and q < 0.05 within whole blood), scaled per CpG
e  gene sets ranked by old-minus-young attribution (fgsea): network of the sets with P < 0.01 linked by shared
   leading-edge genes, and the twelve sets with the smallest P

Data: ../artifacts/figure_tables_2026-10-23/table5_*, table6_*, table7_fig5e_fgsea_old_minus_young.csv.gz
Run: MPLBACKEND=Agg python plot_orig_fig5_2026-11-10.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from orig_rebuild_common_2026_11_10 import ART, PW, SPINE, Page, original_images, read_csv, style, trim

D = ART / "figure_tables_2026-10-23"
T5 = read_csv(D / "table5_fig5bd_cpg_attribution.csv.gz")
T6 = read_csv(D / "table6_fig5c_old_young_attribution.csv.gz")
T7 = read_csv(D / "table7_fig5e_fgsea_old_minus_young.csv.gz")
GS = read_csv(D / "table7_fig5e_gene_scores.csv.gz")
G = [("mean_abs_attribution_lt20", "<20 years"), ("mean_abs_attribution_20to60", "20 to <60 years"),
     ("mean_abs_attribution_ge60", "≥60 years")]
RDBU = plt.get_cmap("RdBu_r")
HEAT = LinearSegmentedColormap.from_list("ylorrd", ["#fff7c0", "#fdd266", "#f89540", "#e2452c", "#b3151b"])


def panel_a(pg):
    xref, arr, bb = [i for i in original_images(5) if i[0] == 96][0]
    arr = arr.copy()
    sx, sy = arr.shape[1] / (bb[2] - bb[0]), arr.shape[0] / (bb[3] - bb[1])

    def wipe(x0, y0, x1, y1):
        arr[int((y0 - bb[1]) * sy):int((y1 - bb[1]) * sy), int((x0 - bb[0]) * sx):int((x1 - bb[0]) * sx)] = 255

    wipe(3, 117, 44, 137)        # 'Attention Matrices'
    wipe(30, 139, 226, 247)      # matrices, their axis labels and the caption
    pg.place(arr, bb)
    pg.text(24, 119, "CpG", 6.5, bold=True, ha="center")
    pg.text(24, 127, "profile", 6.5, bold=True, ha="center")
    rng = np.random.default_rng(5)
    ax = pg.axes(0, 139, 240, 247, z=4)
    ax.set_xlim(0, 240); ax.set_ylim(247, 139); ax.set_axis_off()
    for x0, vals in ((47, [.15, .9, .35, .75, .2, .1, .3, .2]), (144, [.2, .25, .15, .4, .3, .85, .95, .7])):
        ax.annotate("", xy=(x0 + 36, 163), xytext=(x0 + 36, 146), arrowprops=dict(arrowstyle="-|>", color="#8c8c8c", lw=1))
        for i, v in enumerate(vals):
            ax.add_patch(Rectangle((x0 + i * 9, 168), 9, 22, fc=HEAT(v), ec="white", lw=0.6))
        ax.text(x0 + 36, 199, "49,156 CpGs", fontsize=6.5, ha="center", va="top")
    ax.text(40, 179, "Attribution\nper CpG", fontsize=6.5, ha="right", va="center", fontweight="bold")
    ax.text(121, 222, "Attribution of the age output by age group", fontsize=7.5, ha="center", va="top", fontweight="bold")


def panel_b(pg):
    pairs = [(0, 1), (1, 2), (0, 2)]
    lim = (2e-5, 3e-2)
    hb = None
    for k, (i, j) in enumerate(pairs):
        ax = pg.axes(278 + k * 95, 30, 278 + k * 95 + 72, 102)
        x, y = T5[G[i][0]].values, T5[G[j][0]].values
        ok = (x > 0) & (y > 0)
        hb = ax.hexbin(x[ok], y[ok], gridsize=55, xscale="log", yscale="log", norm=LogNorm(1, 400), cmap="RdBu_r",
                       linewidths=0, extent=(np.log10(lim[0]), np.log10(lim[1]), np.log10(lim[0]), np.log10(lim[1])),
                       mincnt=1, rasterized=True)
        ax.plot(lim, lim, color="#333333", lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xticks([1e-4, 1e-3, 1e-2]); ax.set_yticks([1e-4, 1e-3, 1e-2])
        ax.set_xlabel(G[i][1], labelpad=1); ax.set_ylabel(G[j][1], labelpad=1)
        style(ax, 6.5, 7, offset=3, minor=False)
        ax.spines["bottom"].set_bounds(1e-4, 1e-2); ax.spines["left"].set_bounds(1e-4, 1e-2)
        rho = spearmanr(x, y)[0]
        ax.text(0.04, 0.97, f"$r_s$ = {rho:.2f}", transform=ax.transAxes, fontsize=6.5, va="top")
    cax = pg.axes(566, 36, 571, 96)
    cb = pg.fig.colorbar(hb, cax=cax)
    cb.set_ticks([1, 10, 100]); cb.ax.tick_params(labelsize=6, length=2, width=0.5); cb.outline.set_visible(False)
    pg.text(568, 24, "CpGs", 6.5, ha="center")
    pg.text(420, 8.5, "Attribution per CpG (mean absolute gradient × input)", 8.5, ha="center")


def panel_c(pg):
    w = T6.dropna(subset=["whole_blood_p_mannwhitney", "whole_blood_log2_ratio_old_young"])
    x, y = w.whole_blood_log2_ratio_old_young.values, -np.log10(w.whole_blood_p_mannwhitney.values)
    ax = pg.axes(283, 146, 535, 222)
    o = np.argsort(np.abs(x))
    sc = ax.scatter(x[o], y[o], c=x[o], cmap=RDBU, norm=TwoSlopeNorm(0, -3, 3), s=2.5, linewidths=0, rasterized=True)
    thr = -np.log10(w.loc[w.whole_blood_q_bh < 0.05, "whole_blood_p_mannwhitney"].max())
    ax.axhline(thr, color="#8c8c8c", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(-4.4, 6.2); ax.set_ylim(0, 70)
    ax.set_xticks(range(-4, 5, 2)); ax.set_yticks(range(0, 61, 20))
    ax.set_xlabel("log$_2$ (attribution in old / young), whole blood", labelpad=1)
    ax.set_ylabel("−log$_{10}$ $P$", labelpad=1)
    style(ax, 7, 7.5, offset=3)
    trim(ax)
    n_up = int(((w.whole_blood_q_bh < 0.05) & (x > 0)).sum()); n_dn = int(((w.whole_blood_q_bh < 0.05) & (x < 0)).sum())
    ax.text(0.995, 0.97, f"higher in old\n{n_up:,} CpGs", transform=ax.transAxes, fontsize=6.5, ha="right", va="top")
    ax.text(0.0, 0.97, f"higher in young\n{n_dn:,} CpGs", transform=ax.transAxes, fontsize=6.5, ha="left", va="top")
    cax = pg.axes(566, 152, 571, 212)
    cb = pg.fig.colorbar(sc, cax=cax); cb.set_ticks([-3, 0, 3]); cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_visible(False)
    pg.text(568, 141, "log$_2$ ratio", 6.5, ha="center")
    lab = label_volcano(ax, w)
    print("volcano labels:", lab)
    return n_up, n_dn, len(w)


def panel_d(pg):
    t = T5.merge(T6[["cpg_id", "whole_blood_log2_ratio_old_young", "whole_blood_q_bh"]], on="cpg_id")
    r = np.log2(t[G[2][0]] / t[G[0][0]])
    ok = (t.whole_blood_q_bh < 0.05) & (np.sign(t.whole_blood_log2_ratio_old_young) == np.sign(r)) & np.isfinite(r)
    t = t.assign(r=r)[ok]
    sel = list(t.nsmallest(10, "r").index) + list(t.nlargest(10, "r").sort_values("r").index)
    s = t.loc[sel]
    m = s[[g for g, _ in G]].values.T
    m = m / m.max(axis=0, keepdims=True)
    ax = pg.axes(60, 276, 520, 340)
    im = ax.imshow(m[::-1], cmap=RDBU, vmin=0, vmax=1, aspect="auto", interpolation="none")
    ax.set_yticks(range(3)); ax.set_yticklabels([g for _, g in G][::-1], fontsize=7.5)
    lab = [f"{c} ({g.split(';')[0]})" if isinstance(g, str) and g else f"{c} (intergenic)" for c, g in zip(s.cpg_id, s.genes)]
    ax.set_xticks(range(20)); ax.set_xticklabels(lab, rotation=90, fontsize=6.5)
    ax.axvline(9.5, color="white", lw=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=2.5, width=0.6, color=SPINE)
    cax = pg.axes(532, 280, 537, 336)
    cb = pg.fig.colorbar(im, cax=cax); cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_visible(False)
    pg.fig.text(562 / PW, 1 - 308 / pg.ph, "Attribution scaled per CpG", fontsize=6.5, rotation=90, ha="center", va="center")
    pg.text(290, 262, "CpGs with the largest fall (left) and rise (right) in attribution with age", 8.5, ha="center")
    return s


def pretty(p):
    for pre in ("GOBP_", "REACTOME_", "KEGG_", "HALLMARK_"):
        if p.startswith(pre):
            p = p[len(pre):]
    p = p.replace("_", " ").capitalize()
    for a, b in (("gpcr", "GPCR"), ("Gpcr", "GPCR"), ("Class a 1", "Class A/1"), ("mrna", "mRNA"), ("V d j", "V(D)J"), ("apc c", "APC/C"),
                 ("tp53", "TP53"), ("dubs", "DUBs"), ("Class i ", "Class I "), ("cdc20", "CDC20"), ("cdh1", "CDH1")):
        p = p.replace(a, b)
    import re
    p = re.sub(r"\batp\b", "ATP", p)
    return p


def panel_e_bars_two_sided(pg, y0):
    """Gene sets ranked by old-minus-young attribution: the ten with the smallest P on each side, as in the
    submitted panel (bars coloured by -log10 P); asterisks, FDR < 0.05 within the collection."""
    t = T7.dropna(subset=["p"])
    t = (t[t.NES > 0].nsmallest(10, "p").sort_values("NES", ascending=False)._append(
         t[t.NES < 0].nsmallest(10, "p").sort_values("NES", ascending=False)))
    ax = pg.axes(300, y0 + 16, 520, y0 + 232)
    yy = np.arange(len(t))[::-1]
    lp = -np.log10(t.p.values)
    cmap = LinearSegmentedColormap.from_list("p", ["#a9d3e0", "#b9b3b8", "#b8605a", "#8b1a1a"])
    norm = plt.Normalize(1, 5)
    ax.barh(yy, t.NES.values, height=0.82, color=cmap(norm(lp)), edgecolor="white", linewidth=0.5, zorder=2)
    for y, nes, fdr in zip(yy, t.NES.values, t.fdr.values):
        if fdr < 0.05:
            ax.text(nes + 0.04, y - 0.12, "*", fontsize=9, fontweight="bold", va="center", ha="left")
    ax.axvline(0, color="#333333", lw=0.7, ls=(0, (2, 2)), zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels([pretty(p) for p in t.pathway], fontsize=7.5)
    ax.set_xlim(-2, 2); ax.set_ylim(-0.7, len(t) - 0.3)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Normalized enrichment score (NES)", labelpad=2)
    style(ax, 7.5, 8.3, offset=3, minor=False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    trim(ax, y=False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = pg.axes(536, y0 + 100, 542, y0 + 150)
    cb = pg.fig.colorbar(sm, cax=cax, extend="max"); cb.set_ticks([1, 3, 5]); cb.ax.tick_params(labelsize=6, length=2, width=0.5)
    cb.outline.set_visible(False)
    pg.text(539, y0 + 86, "−log$_{10}$ $P$", 6.5, ha="center")
    pg.text(530, y0 + 164, "* FDR < 0.05", 6.5)
    pg.text(410, y0 + 2, "Higher attribution in young profiles < > Higher attribution in old profiles", 8, bold=True, ha="center")
    return t


# ---------------------------------------------------------------------------------- panel e: gene-set network
NET_P = 0.01          # gene sets shown: nominal fgsea P below this value
NET_J = 0.15          # edge: Jaccard index of the leading-edge genes of two sets at or above this value
MOD_COL = ["#c0504d", "#4f81bd", "#8064a2", "#e39b3b", "#6aa84f", "#3aa6b9", "#b5739d"]
GREY = "#b9b9b9"


def gene_set_network():
    """Nodes: gene sets with P < NET_P. Edges: shared leading-edge genes (Jaccard >= NET_J).
    Modules: connected components with at least three sets, ordered by size."""
    import itertools
    import networkx as nx
    t = T7.dropna(subset=["p"])
    t = t[t.p < NET_P].reset_index(drop=True)
    L = [set(x.split(";")) for x in t.leading_edge_genes]
    G = nx.Graph()
    G.add_nodes_from(range(len(t)))
    for i, j in itertools.combinations(range(len(t)), 2):
        jac = len(L[i] & L[j]) / len(L[i] | L[j])
        if jac >= NET_J:
            G.add_edge(i, j, w=jac)
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(t.p[list(c)])))
    return t, L, G, comps


REGIONS = [(0.2, 1.2, 5.6, 6.2), (6.6, 1.2, 10.3, 3.1), (11.3, 1.2, 15.2, 3.1),      # x0, y0, x1, y1 per module, largest first
           (6.6, 4.85, 8.9, 6.2), (9.8, 4.85, 12.1, 6.2), (13.0, 4.85, 15.2, 6.2)]


def region_layout(G, comps):
    """Kamada-Kawai layout of each module, scaled into its region; pairs and single sets in rows underneath."""
    import networkx as nx
    big = [c for c in comps if len(c) >= 3]
    small = [c for c in comps if len(c) < 3]
    if len(big) > len(REGIONS):
        raise SystemExit(f"{len(big)} modules but {len(REGIONS)} regions: extend REGIONS")
    placed, centres = {}, []
    for c, (x0, y0, x1, y1) in zip(big, REGIONS):
        sub = G.subgraph(c)
        pos = nx.kamada_kawai_layout(sub, weight=None)
        keys = list(pos); xy = np.array([pos[k] for k in keys])
        if (x1 - x0) > (y1 - y0) and np.ptp(xy[:, 1]) > np.ptp(xy[:, 0]):
            xy = xy[:, ::-1]                               # long axis of the module along the long side of its region
        xy = (xy - xy.min(0)) / np.maximum(np.ptp(xy, axis=0), 1e-9)
        xy = xy * [x1 - x0, y1 - y0] + [x0, y0]
        placed.update(dict(zip(keys, map(tuple, xy))))
        centres.append((c, (x0 + x1) / 2, y0, y1))
    x, y = 0.3, 7.55
    for c in small:
        w = 0.5 * len(c)
        if x + w > 15.3:
            x = 0.3; y += 0.55
        for k, n in enumerate(sorted(c)):
            placed[n] = (x + 0.45 * k, y)
        x += w + 0.36
    return placed, centres, y + 0.45


def panel_e(pg, y0):
    t, L, G, comps = gene_set_network()
    module = {}
    for k, c in enumerate([c for c in comps if len(c) >= 3]):
        for n in c:
            module[n] = k
    pos, centres, H = region_layout(G, comps)
    ax = pg.axes(12, y0 + 16, 356, y0 + 252)
    ax.set_xlim(-0.2, 15.6); ax.set_ylim(H, 0.0); ax.set_axis_off()
    for i, j, d in G.edges(data=True):
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#8a8a8a", lw=0.3 + 1.8 * d["w"], alpha=0.4, zorder=1,
                solid_capstyle="round")
    lp = -np.log10(t.p.values)
    size = 13 + 15 * (lp - 2)
    col = [MOD_COL[module[n] % len(MOD_COL)] if n in module else GREY for n in range(len(t))]
    sig = t.fdr.values < 0.05
    xy = np.array([pos[n] for n in range(len(t))])
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=col, edgecolors="white", linewidths=0.5, zorder=3, clip_on=False)
    ax.scatter(xy[sig, 0], xy[sig, 1], s=size[sig] + 18, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4, clip_on=False)
    young = t.NES.values < 0
    ax.scatter(xy[young, 0], xy[young, 1], s=size[young] * 0.25, c="white", linewidths=0, zorder=5, clip_on=False)
    info = []
    for c, cx, ytop, ybot in centres:                 # module label: its set with the smallest P, its size, shared genes
        k = module[next(iter(c))]
        lead = t.loc[sorted(c)].sort_values("p").iloc[0]
        genes = pd.Series([g for n in c for g in L[n]]).value_counts()
        top = genes[genes >= max(2, int(0.6 * len(c)))].index
        gs = GS.set_index("gene").score.reindex(top).sort_values(ascending=False).index[:3].tolist()
        import textwrap
        name = textwrap.wrap(pretty(lead.pathway), 24)
        name[-1] += f" ($n$ = {len(c)})"
        ax.text(cx, ytop - 0.3, "\n".join(name), fontsize=6.2, fontweight="bold", ha="center", va="bottom", linespacing=1.0,
                color=MOD_COL[k % len(MOD_COL)])
        ax.text(cx, ybot + 0.42, ", ".join(gs), fontsize=5.6, fontstyle="italic", ha="center", va="center", color="#333333")
        info.append((pretty(lead.pathway), len(c), gs))
    for n in np.where(sig)[0]:                        # FDR-significant sets outside a module, named
        if n not in module:
            ax.text(pos[n][0], pos[n][1] + 0.3, pretty(t.pathway[n]), fontsize=5.6, ha="center", va="top")
    n_out = int(sum(len(c) for c in comps if len(c) < 3))
    ax.text(0.1, 7.12, f"Sets outside the modules ($n$ = {n_out})", fontsize=5.8, ha="left", va="center", color="#555555")
    pg.text(184, y0 + 0, "Gene sets with $P$ < 0.01, linked by shared leading-edge genes", 7.8, bold=True, ha="center")
    # ranked strip: the twelve sets with the smallest P, in the colours of the network
    r = t.nsmallest(12, "p").sort_values("p")
    bx = pg.axes(490, y0 + 22, 566, y0 + 190)
    yy = np.arange(len(r))[::-1]
    bx.barh(yy, r.NES.values, height=0.72, color=[col[n] for n in r.index], edgecolor="white", linewidth=0.4, zorder=2)
    for y, nes, fdr in zip(yy, r.NES.values, r.fdr.values):
        if fdr < 0.05:
            bx.text(nes + 0.03, y - 0.12, "*", fontsize=9, fontweight="bold", va="center", ha="left")
    bx.set_yticks(yy); bx.set_yticklabels([pretty(p) for p in r.pathway], fontsize=6.6)
    bx.set_xlim(0, 1.8); bx.set_ylim(-0.7, len(r) - 0.3); bx.set_xticks([0, 0.5, 1.0, 1.5])
    bx.set_xlabel("Normalized enrichment score", labelpad=2)
    style(bx, 6.8, 7.6, offset=3, minor=False)
    bx.spines["left"].set_visible(False); bx.tick_params(axis="y", length=0)
    trim(bx, y=False)
    pg.text(470, y0 + 4, "Gene sets with the smallest $P$", 7.5, bold=True, ha="center")
    # key: node size, ring, open centre
    kx = pg.axes(372, y0 + 226, 584, y0 + 254); kx.set_xlim(0, 10); kx.set_ylim(0, 1); kx.set_axis_off()
    for v, x in ((2, 0.3), (4, 1.2), (6, 2.2)):
        kx.scatter([x], [0.66], s=13 + 15 * (v - 2), c="#8a8a8a", edgecolors="white", linewidths=0.5, clip_on=False)
        kx.text(x, 0.1, f"{v}", fontsize=5.8, ha="center")
    kx.text(2.85, 0.1, "−log$_{10}$ $P$", fontsize=5.8, ha="left")
    kx.scatter([5.0], [0.7], s=42, facecolors="none", edgecolors="black", linewidths=1.0, clip_on=False)
    kx.text(5.4, 0.7, "FDR < 0.05 (asterisk in bars)", fontsize=5.8, va="center")
    kx.scatter([5.0], [0.16], s=34, c="#8a8a8a", edgecolors="white", linewidths=0.5, clip_on=False)
    kx.scatter([5.0], [0.16], s=8, c="white", linewidths=0, clip_on=False)
    kx.text(5.4, 0.16, "higher in young profiles", fontsize=5.8, va="center")
    print("network:", len(t), "sets,", G.number_of_edges(), "edges,", len(centres), "modules,", n_out, "sets outside modules,",
          int((t.NES > 0).sum()), "with positive NES")
    print("modules:", info)
    return r


def label_volcano(pg_ax, w):
    """Name the most extreme CpGs of each side (largest |log2 ratio| among q < 0.05, gene-annotated)."""
    g = T5.set_index("cpg_id").genes
    w = w[(w.whole_blood_q_bh < 0.05)].assign(gene=lambda d: d.cpg_id.map(g)).dropna(subset=["gene"])
    out = []
    for side, sub, ha, tx, y_top in (("old", w.nlargest(4, "whole_blood_log2_ratio_old_young"), "left", 4.95, 46),
                                     ("young", w.nsmallest(4, "whole_blood_log2_ratio_old_young"), "right", -3.05, 50)):
        sub = sub.assign(y=-np.log10(sub.whole_blood_p_mannwhitney)).sort_values("y", ascending=False)
        for k, r in enumerate(sub.itertuples()):      # text rows in the order of the points, so leaders do not cross
            x = r.whole_blood_log2_ratio_old_young
            pg_ax.annotate(r.gene.split(";")[0], xy=(x, r.y), xytext=(tx, y_top - 9.5 * k), fontsize=6, fontstyle="italic",
                           ha=ha, va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color="#444444", shrinkA=1, shrinkB=1.5))
            out.append((side, r.cpg_id, r.gene.split(";")[0], round(x, 2)))
    return out


def main():
    pg = Page(700)
    panel_a(pg)
    panel_b(pg)
    info = panel_c(pg)
    s = panel_d(pg)
    t = panel_e(pg, 436)
    for l, x, y in (("a", 10.2, 6.9), ("b", 240.4, 7.1), ("c", 241.2, 132.7), ("d", 10.7, 244.7), ("e", 11.5, 430)):
        pg.letter(x, y, l)
    pg.save("fig5", orig_n=5)
    print("panel c (whole blood): higher in old, higher in young, tested:", info)
    print(s[["cpg_id", "genes", "r"]].to_string())
    print(t[["pathway", "collection", "NES", "p", "fdr"]].to_string())


if __name__ == "__main__":
    main()
