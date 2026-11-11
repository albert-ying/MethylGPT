"""Figure 5. CpGs and gene sets that drive MethylGPT's age output in young and old profiles.

Claim: attribution of the age output shifts with age. Most CpGs carry more attribution in old than in young
blood, the most significant ones sit at immune genes and at known age loci, and the gene sets that gain
attribution group into an immune module and receptor and ion-signalling modules.

Hue key: rust = higher attribution in older profiles; blue = higher in younger profiles; grey = not significant;
sage = CpG of a published clock; ink ring = gene set with FDR < 0.05. Attribution strips and the density
panels use colour ramps. Definitions, caveats and thresholds are in the legend, not in the figure.

Panels
  a  schematic: mean absolute gradient x input attribution of the age output, per CpG and age group
  b  per-CpG attribution compared between consecutive age groups (pooled tissues)
  c  old against young within whole blood: log2 ratio and Mann-Whitney P per CpG
  d  the eight CpGs with the largest whole-blood ratio in each direction, with CpG, genomic context,
     P and the profiles observed
  e  exploratory: gene sets with P < 0.01 joined by shared leading-edge genes, and the eleven sets with the smallest P

Sources (all local, aggregate):
  ../artifacts/figure_tables_2026-10-23/table5_fig5bd_cpg_attribution.csv.gz
  ../artifacts/figure_tables_2026-10-23/table6_fig5c_old_young_attribution.csv.gz
  ../artifacts/figure_tables_2026-10-23/table7_fig5e_fgsea_old_minus_young.csv.gz
  ../artifacts/figure_tables_2026-10-23/table7_fig5e_gene_scores.csv.gz
  original_submission/fig5.pdf (panel a artwork)
Outputs: figure5_attribution_2026-11-10.pdf/.png and figure5_provenance_2026-11-10/*.csv
Run: MPLBACKEND=Agg python build_figure5_attribution_2026-11-10.py
"""
from __future__ import annotations

import itertools
import re
import sys
import textwrap
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from scipy.stats import spearmanr
from shapely.geometry import MultiPoint

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, set_domain, add_panel_labels, label_panel_group,
                                 nature_legend, smart_annotate, place_stat_text, ref_line, add_colorbar, save_figure,
                                 mark_pictorial, allow_dead_margin, RUST, GRAY, BLUE, SAGE, GOLD, PLUM, INK, BG, TYPE)

HERE = Path(__file__).resolve().parent
D = HERE.parent / "artifacts/figure_tables_2026-10-23"
PROV = HERE / "figure5_provenance_2026-11-10"
retro_style()
TXT = TYPE["tick"]                    # one text size in the figure (no small grey subtitle-level text)
HEAT = LinearSegmentedColormap.from_list("ying_heat", ["#FFF8E1", GOLD["100"], GOLD["200"], RUST["200"], RUST["300"], RUST["500"]])
PROFILE = LinearSegmentedColormap.from_list("ying_profile", ["#FFFFFF", PLUM["100"], PLUM["200"], PLUM["300"]])
DENSITY = LinearSegmentedColormap.from_list("ying_density", [BLUE["300"], BLUE["200"], "#EFE9C9", GOLD["200"], RUST["300"], RUST["500"]])


def read(name):
    p = D / name
    if not p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return pd.read_csv(p)


T5 = read("table5_fig5bd_cpg_attribution.csv.gz")
T6 = read("table6_fig5c_old_young_attribution.csv.gz")
T7 = read("table7_fig5e_fgsea_old_minus_young.csv.gz").dropna(subset=["p"])
GS = read("table7_fig5e_gene_scores.csv.gz")
AGE = [("mean_abs_attribution_lt20", "<20 years", 2445), ("mean_abs_attribution_20to60", "20 to <60 years", 2642),
       ("mean_abs_attribution_ge60", "≥60 years", 724)]
W = T6.merge(T5, on="cpg_id").dropna(subset=["whole_blood_p_mannwhitney", "whole_blood_log2_ratio_old_young"]).copy()
W["x"] = W.whole_blood_log2_ratio_old_young
W["lp"] = -np.log10(W.whole_blood_p_mannwhitney)
W["gene"] = W.genes.fillna("").str.split(";").str[0]
W["sig"] = W.whole_blood_q_bh < 0.05
N_OLD_WB, N_YOUNG_WB = 116, 783       # whole-blood profiles >=60 and <30 years (table README, secondary within-tissue test)
N_OLD_ALL, N_YOUNG_ALL = 724, 1000    # pooled held-out profiles >=60 and <30 years (table README)
N_HALLMARK = int((read("table7_fig5e_fgsea_old_minus_young.csv.gz").collection == "Hallmark").sum())


def it(sym):
    return rf"$\it{{{sym}}}$"


def pretty(p):
    for pre in ("GOBP_", "REACTOME_", "KEGG_", "HALLMARK_"):
        if p.startswith(pre):
            p = p[len(pre):]
    p = p.replace("_", " ").capitalize()
    for a, b in (("gpcr", "GPCR"), ("Gpcr", "GPCR"), ("Class a 1", "Class A/1"), ("mrna", "mRNA"), ("V d j", "V(D)J"),
                 ("apc c", "APC/C"), ("tp53", "TP53"), ("Class i ", "Class I "), ("G alpha i", "Gαi"), ("ca2", "Ca²⁺")):
        p = p.replace(a, b)
    return re.sub(r"\batp\b", "ATP", p)


# ------------------------------------------------------------------------------------------------- panel a
def panel_a(ax):
    doc = fitz.open(HERE / "original_submission/fig5.pdf")
    info = [i for i in doc[0].get_image_info(xrefs=True) if i["xref"] == 96][0]
    pix = fitz.Pixmap(doc, 96)
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    x0, y0, x1, y1 = info["bbox"]
    sy = arr.shape[0] / (y1 - y0); sx = arr.shape[1] / (x1 - x0)
    arr[int((52.3 - y0) * sy):int((61.8 - y0) * sy), :] = 255          # raster labels of the submitted panel
    cut = 88.0                                                         # faces and DNA; the attention arrows are not used
    ax.imshow(arr[:int((cut - y0) * sy)], extent=(x0, x1, cut, y0), interpolation="antialiased", zorder=1)
    ax.set_xlim(-8, 242); ax.set_ylim(190, 2); ax.set_aspect("equal"); ax.set_axis_off()
    mark_pictorial(ax)
    cols = {84: ("Young profiles", [.2, .8, .4, .6, .3, .7, .5, .25], [.15, .7, .3, .55, .2, .1, .3, .2]),
            182: ("Old profiles", [.3, .7, .5, .5, .4, .8, .6, .35], [.25, .3, .2, .5, .4, .9, 1.0, .75])}      # illustration only
    for cx, (name, beta, attr) in cols.items():
        ax.text(cx, 57, name, fontsize=TYPE["tick"], ha="center", va="center", color=INK)
        for i, q in enumerate(beta):                                   # CpG profile: methylation values
            ax.add_patch(Rectangle((cx - 36 + i * 9, 96), 9, 11, fc=PROFILE(0.15 + 0.75 * q), ec="white", lw=0.5, zorder=2))
        ax.add_patch(FancyBboxPatch((cx - 30, 121), 60, 15, boxstyle="round,pad=0,rounding_size=3", fc="white", ec=GRAY["300"],
                                    lw=0.7, zorder=2))
        ax.text(cx, 128.5, "MethylGPT", fontsize=TXT, ha="center", va="center", color=INK)
        ax.annotate("", xy=(cx, 120.5), xytext=(cx, 108.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7,
                                                                           shrinkA=0, shrinkB=0, mutation_scale=6))
        ax.annotate("", xy=(cx, 149.5), xytext=(cx, 136.5), arrowprops=dict(arrowstyle="-|>", color=GRAY["400"], lw=0.7,
                                                                           shrinkA=0, shrinkB=0, mutation_scale=6))
        ax.text(cx, 155, "Predicted age", fontsize=TXT, ha="center", va="center", color=INK)
        for i, q in enumerate(attr):                                   # attribution per CpG
            ax.add_patch(Rectangle((cx - 36 + i * 9, 172), 9, 13, fc=HEAT(0.05 + 0.9 * q), ec="white", lw=0.5, zorder=2))
        ax.annotate("", xy=(cx, 171), xytext=(cx, 160), arrowprops=dict(arrowstyle="-|>", color=GRAY["400"], lw=0.7, shrinkA=0,
                                                                        shrinkB=0, mutation_scale=6, linestyle=(0, (2, 1.5))))
    ax.text(40, 101.5, "CpG\nprofile", fontsize=TXT, ha="right", va="center", color=INK, linespacing=1.0)
    ax.text(40, 178.5, "Attribution\nper CpG", fontsize=TXT, ha="right", va="center", color=INK, linespacing=1.0)


# ------------------------------------------------------------------------------------------------- panel b
def panel_b(axes):
    pairs = [(0, 1), (1, 2)]
    lim = (2e-5, 3e-2)
    rows, hb = [], None
    for ax, (i, j) in zip(axes, pairs):
        x, y = T5[AGE[i][0]].values, T5[AGE[j][0]].values
        ok = (x > 0) & (y > 0)
        hb = ax.hexbin(x[ok], y[ok], gridsize=50, xscale="log", yscale="log", norm=LogNorm(1, 1000), cmap=DENSITY,
                       linewidths=0, mincnt=1, rasterized=True,
                       extent=(np.log10(lim[0]), np.log10(lim[1]), np.log10(lim[0]), np.log10(lim[1])))
        ax.set_xlim(*lim); ax.set_ylim(*lim)
        ax.set_xticks([1e-4, 1e-3, 1e-2]); ax.set_yticks([1e-4, 1e-3, 1e-2])
        ax.set_xlabel(f"Attribution, {AGE[i][1]}")
        ax.set_ylabel(f"Attribution, {AGE[j][1]}")
        rho = spearmanr(x, y)[0]; above = float((y > x).mean()); ratio = float(np.median(y[ok] / x[ok]))
        rows.append(dict(x_group=AGE[i][1], y_group=AGE[j][1], spearman_rho=rho, fraction_above_identity=above,
                         median_ratio_y_over_x=ratio, n_cpgs=int(len(x)), max_cpgs_per_hex=int(hb.get_array().max())))
    return hb, rows, lim


def finish_b(axes, rows, lim, hb, cax):
    for ax, r in zip(axes, rows):
        set_domain(ax, xlim=lim, ylim=lim)
        ax.set_xticks([1e-4, 1e-3, 1e-2]); ax.set_yticks([1e-4, 1e-3, 1e-2])
        ax.set_xticklabels(["0.0001", "0.001", "0.01"]); ax.set_yticklabels(["0.0001", "0.001", "0.01"])
        ax.minorticks_off()
        allow_dead_margin(ax, "both", reason="both panels share one attribution domain")
        ax.plot(lim, lim, color=GRAY["400"], lw=0.55, ls=(0, (2, 2)), zorder=3)
        ax.text(0.05, 0.97, f"$\\rho$ = {r['spearman_rho']:.2f}", transform=ax.transAxes, fontsize=TXT, va="top",
                ha="left", color=INK)
        ax.text(0.97, 0.04, f"{100 * r['fraction_above_identity']:.0f}% above\nthe diagonal", transform=ax.transAxes,
                fontsize=TXT, va="bottom", ha="right", color=INK, linespacing=1.05)
    cb = plt.colorbar(hb, cax=cax)
    cb.set_ticks([1, 10, 100, 1000]); cb.set_ticklabels(["1", "10", "100", "1,000"]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=TYPE["tick"], length=2, width=0.5)
    cb.ax.minorticks_off()
    cb.ax.set_title("CpGs", fontsize=TYPE["tick"], fontweight="normal", loc="left", pad=7)


# ------------------------------------------------------------------------------------------------- panel c
CLOCK_LOCI = ["ELOVL2", "FHL2", "KLF14"]


def panel_c(ax):
    ns, up, dn = ~W.sig, W.sig & (W.x > 0), W.sig & (W.x < 0)
    kw = dict(linewidths=0, rasterized=True, clip_on=False)
    ax.scatter(W.x[ns], W.lp[ns], s=2.2, c=GRAY["100"], alpha=0.6, zorder=1, label=f"Not significant ({int(ns.sum()):,})", **kw)
    ax.scatter(W.x[dn], W.lp[dn], s=2.2, c=BLUE["300"], alpha=0.45, zorder=2, label=f"Higher in young ({int(dn.sum()):,})", **kw)
    ax.scatter(W.x[up], W.lp[up], s=2.2, c=RUST["300"], alpha=0.35, zorder=2, label=f"Higher in old ({int(up.sum()):,})", **kw)
    ck = W[W.in_any_of_four_clocks.astype(bool)]
    ax.scatter(ck.x, ck.lp, s=1.5, c=SAGE["400"], alpha=0.85, zorder=3, label=f"Clock CpG ({len(ck):,})", **kw)
    ax.set_xlabel("log$_2$ attribution ratio, old / young (whole blood)")
    ax.set_ylabel("−log$_{10}$ $P$")
    s = W[W.sig & (W.gene != "")]
    top_up = s[s.x > 0].sort_values("lp", ascending=False).drop_duplicates("gene").head(6)
    top_dn = s[s.x < 0].sort_values("lp", ascending=False).drop_duplicates("gene").head(3)
    loci = s[s.gene.isin(CLOCK_LOCI) & s.in_any_of_four_clocks.astype(bool)].sort_values("lp", ascending=False).drop_duplicates("gene")
    lab = pd.concat([top_up.assign(kind="smallest P"), top_dn.assign(kind="smallest P"), loci.assign(kind="clock locus")])
    lab = lab.drop_duplicates("cpg_id")
    counts = dict(n_tested=len(W), higher_in_old=int(up.sum()), higher_in_young=int(dn.sum()), not_significant=int(ns.sum()),
                  clock_cpgs=len(ck), clock_higher_in_old=int((ck.sig & (ck.x > 0)).sum()),
                  clock_higher_in_young=int((ck.sig & (ck.x < 0)).sum()), clock_not_significant=int((~ck.sig).sum()))
    return lab, counts


def finish_c(ax, lab, counts):
    thr = W.loc[W.sig, "lp"].min()
    ref_line(ax, y=thr)
    col = [SAGE["500"] if k == "clock locus" else (RUST["500"] if v > 0 else BLUE["500"]) for k, v in zip(lab.kind, lab.x)]
    ax.scatter(lab.x, lab.lp, s=9, c=col, linewidths=0, zorder=5, clip_on=False)
    nature_legend(ax, title="CpGs", loc="upper left", markerscale=3.0)
    old = lab[lab.x > 0]
    smart_annotate(ax, [it(g) for g in old.gene], list(zip(old.x, old.lp)), fontsize=TXT, color=INK, lw=0.4,
                   mode="column", direction="right")
    young = lab[lab.x < 0].sort_values("lp", ascending=False)      # three labels in a short column below the legend
    for k, r in enumerate(young.itertuples()):
        ty = 44 - 5.5 * k
        ax.text(-3.0, ty, it(r.gene), fontsize=TXT, ha="right", va="center", color=INK)
        ax.plot([-2.93, r.x], [ty, r.lp], color=INK, lw=0.4, zorder=4)


# ------------------------------------------------------------------------------------------------- panel d
STAB = HERE.parent / "artifacts/attribution_stability_2026-11-10/part_a_cpg_age_attribution_2026-11-10.csv.gz"
RANKINGS = [("mean_abs_attribution", "Attribution", RUST["400"], 1.7),
            ("rho_age_tissue_adjusted", "Correlation with age", BLUE["400"], 1.1),
            ("delta_beta_old_minus_young", "Old minus young methylation", SAGE["400"], 1.1),
            ("sd_beta", "Variance", GOLD["300"], 1.1)]
K_MAX = 10000


def panel_d(ax):
    """Clock CpGs among the top-ranked CpGs, for four rankings of the 49,156 vocabulary CpGs (5,811 held-out profiles).
    The four clocks' outputs were training targets of MethylGPT; rankings by age association or variance do not use them."""
    if not STAB.exists():
        sys.exit(f"Cannot proceed - required data file not found: {STAB}")
    S = pd.read_csv(STAB)
    clock = S.in_any_of_four_clocks.astype(bool).to_numpy()
    n, n_clock = len(S), int(clock.sum())
    k = np.arange(1, K_MAX + 1)
    rows = []
    for col, lab, colour, lw in RANKINGS:
        v = S[col].abs().to_numpy()
        if np.isnan(v).any():
            sys.exit(f"Cannot proceed - missing values in {col}")
        hits = np.cumsum(clock[np.argsort(-v, kind="stable")])[:K_MAX]
        ax.plot(k[9:], hits[9:], color=colour, lw=lw, label=lab, zorder=4 if col == "mean_abs_attribution" else 3,
                solid_capstyle="round")
        rows.append(dict(ranking=lab, column=col, **{f"top_{j}": int(hits[j - 1]) for j in (100, 500, 1000, 2000, 5000, 10000)}))
        if col == "mean_abs_attribution":
            for j, dx in ((100, 0.80), (1000, 0.80)):
                ax.plot(j, hits[j - 1], marker="o", ms=3.6, mfc=RUST["500"], mec="white", mew=0.5, zorder=6)
                ax.text(j * dx, hits[j - 1] + 14, f"{hits[j - 1]}", fontsize=TXT, ha="right", va="bottom", color=INK, zorder=6)
    ax.plot(k[9:], k[9:] * n_clock / n, color=GRAY["500"], lw=0.9, ls=(0, (3, 2)), label="Random ranking", zorder=2)
    rows.append(dict(ranking="Random ranking (expected)", column="", **{f"top_{j}": round(j * n_clock / n, 1) for j in (100, 500, 1000, 2000, 5000, 10000)}))
    ax.set_xscale("log")
    ax.set_xlim(10, K_MAX); ax.set_ylim(0, 400)
    ax.set_xticks([10, 100, 1000, 10000]); ax.set_xticklabels(["10", "100", "1,000", "10,000"])
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Top-ranked CpGs"); ax.set_ylabel(f"Clock CpGs among them")
    ax.legend(loc="upper left", frameon=False, fontsize=TXT, handlelength=1.5, handletextpad=0.5, labelspacing=0.32, borderaxespad=0.2)
    return pd.DataFrame(rows), dict(cpgs=n, clock_cpgs=n_clock)


# ------------------------------------------------------------------------------------------------- panel e
NET_P, NET_J = 0.01, 0.15                # display thresholds, chosen after the enrichment results were known
EW, EH = 4.7, 3.17                       # network axes in inches; its data coordinates are inches from the top left
HERO = (1.12, 0.56, 2.02, 2.24)          # x0, y0, x1, y1 of the largest group
CARD_X, CARD_TXT, CARD_Y0, CARD_DY = 3.08, 3.68, 0.03, 0.487
N_HUB = 8
N_TOP = 11


def network():
    t = T7[T7.p < NET_P].reset_index(drop=True)
    L = [set(x.split(";")) for x in t.leading_edge_genes]
    G = nx.Graph(); G.add_nodes_from(range(len(t)))
    for i, j in itertools.combinations(range(len(t)), 2):
        jac = len(L[i] & L[j]) / len(L[i] | L[j])
        if jac >= NET_J:
            G.add_edge(i, j, w=jac)
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(t.p[list(c)])))
    return t, L, G, [c for c in comps if len(c) >= 3], [c for c in comps if len(c) < 3]


def blob(points, pad):
    """Smooth outline around a group: buffered convex hull."""
    g = MultiPoint([tuple(p) for p in points]).convex_hull.buffer(pad, quad_segs=24)
    return PathPatch(MPath(np.asarray(g.exterior.coords), closed=True), fc=BG, ec="none", zorder=0.5)


def fit(G, c, box):
    x0, y0, x1, y1 = box
    p = nx.kamada_kawai_layout(G.subgraph(c), weight=None)
    keys = list(p); xy = np.array([p[q] for q in keys])
    if ((x1 - x0) < (y1 - y0)) == (np.ptp(xy[:, 0]) > np.ptp(xy[:, 1])):
        xy = xy[:, ::-1]
    xy = (xy - xy.min(0)) / np.maximum(np.ptp(xy, axis=0), 1e-9) * [x1 - x0, y1 - y0] + [x0, y0]
    return dict(zip(keys, map(tuple, xy)))


def column(ax, items, x, ha, ytop, ybot):
    """Stack labels (node y, text, node xy) in one column without overlap; straight hairline leaders."""
    items = sorted(items, key=lambda q: q[0])
    h = [0.088 * (txt.count("\n") + 1) + 0.05 for _, txt, _ in items]
    y = [q[0] for q in items]
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + (h[i - 1] + h[i]) / 2)
    over = y[-1] + h[-1] / 2 - ybot
    if over > 0:
        y = [v - over for v in y]
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - (h[i] + h[i + 1]) / 2)
    y = [max(v, ytop + h[i] / 2) for i, v in enumerate(y)]
    for (ny, txt, (px, py)), yy in zip(items, y):
        ax.text(x, yy, txt, fontsize=TXT, ha=ha, va="center", color=INK, linespacing=1.0, zorder=6)
        ax.plot([px, x + (0.03 if ha == "right" else -0.03)], [py, yy], color=INK, lw=0.4, zorder=2.5, solid_capstyle="round")


def panel_e(ax, bx):
    t, L, G, mods, rest = network()
    lp = -np.log10(t.p.values)
    size = 9 + 12 * (lp - 2)
    sig = t.fdr.values < 0.05
    young = t.NES.values < 0
    pos, group = {}, {}
    pos.update(fit(G, mods[0], HERO))
    for k, c in enumerate(mods[1:], start=1):
        y0 = CARD_Y0 + (k - 1) * CARD_DY
        pos.update(fit(G, c, (CARD_X, y0 + 0.06, CARD_X + 0.46, y0 + 0.36)))
    for k, c in enumerate(mods):
        for n in c:
            group[n] = k
    rest = sorted(rest, key=lambda c: (-len(c), min(t.p[list(c)])))
    rest_nodes = [n for c in rest for n in sorted(c, key=lambda n: t.p[n])]
    for i, n in enumerate(rest_nodes):
        pos[n] = (0.16 + (i % 16) * 0.148, 2.69 + (i // 16) * 0.17)
    ax.set_xlim(0, EW); ax.set_ylim(EH, 0); ax.set_axis_off(); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(blob([pos[n] for n in mods[0]], 0.12))
    for c in mods[1:]:
        ax.add_patch(blob([pos[n] for n in c], 0.07))
    for i, j, d in G.edges(data=True):
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=GRAY["300"], lw=0.25 + 1.5 * d["w"], alpha=0.6, zorder=1,
                solid_capstyle="round")
    xy = np.array([pos[n] for n in range(len(t))])
    ingroup = np.array([n in group for n in range(len(t))])
    fill = np.where(young, BLUE["300"], np.where(sig, RUST["400"], np.where(ingroup, RUST["200"], RUST["100"])))
    edge = np.where(sig, INK, np.where(young, BLUE["400"], np.where(ingroup, RUST["400"], RUST["200"])))
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=list(fill), edgecolors=list(edge), linewidths=np.where(sig, 0.9, 0.4), zorder=3, clip_on=False)
    info = []

    def shared_genes(c, k):
        genes = pd.Series([g for n in c for g in L[n]]).value_counts()
        shared = genes[genes >= max(2, int(np.ceil(0.6 * len(c))))].index
        return GS.set_index("gene").score.reindex(shared).sort_values(ascending=False).index[:k].tolist()

    def fdr_note(c):
        k = int(sum(sig[n] for n in c))
        return f"{k} at FDR < 0.05" if k else "none at FDR < 0.05"

    # largest group: name, counts, shared genes, eight sets named on both sides
    c = mods[0]
    members = t.loc[sorted(c)].sort_values("p")
    top = shared_genes(c, 5)
    ax.text(0.04, 0.08, f"1  {pretty(members.pathway.iloc[0])}", fontsize=TYPE["tick"], fontweight="bold", ha="left", va="center", color=INK)
    ax.text(0.04, 0.23, f"{len(c)} gene sets · " + ", ".join(it(g) for g in top), fontsize=TXT, ha="left", va="center",
            color=INK)
    cx = np.mean([pos[n][0] for n in c])
    left, right = [], []
    for n in members.index[:N_HUB]:
        txt = "\n".join(textwrap.wrap(pretty(t.pathway[n]), 16))
        (left if pos[n][0] < cx else right).append((pos[n][1], txt, pos[n]))
    column(ax, left, HERO[0] - 0.2, "right", 0.36, 2.42)
    column(ax, right, HERO[2] + 0.2, "left", 0.36, 2.42)
    info.append(dict(group=1, lead_set=members.pathway.iloc[0], n_sets=len(c), n_fdr_005=int(sum(sig[n] for n in c)),
                     shared_genes=";".join(top), named_sets=";".join(t.pathway[n] for n in members.index[:N_HUB])))
    # the other groups: mini network, lead set, counts, shared genes, further member sets
    for k, c in enumerate(mods[1:], start=1):
        y0 = CARD_Y0 + (k - 1) * CARD_DY
        members = t.loc[sorted(c)].sort_values("p")
        top = shared_genes(c, 2)
        names = []
        name = textwrap.wrap(f"{k + 1}  {pretty(members.pathway.iloc[0])}", 24)
        ax.text(CARD_TXT, y0 - 0.02, "\n".join(name), fontsize=TYPE["tick"], fontweight="bold", ha="left", va="top", color=INK,
                linespacing=1.0)
        yb = y0 - 0.02 + 0.115 * len(name) + 0.055
        ax.text(CARD_TXT, yb, f"{len(c)} sets · " + ", ".join(it(g) for g in top), fontsize=TXT, ha="left", va="center",
                color=INK)
        info.append(dict(group=k + 1, lead_set=members.pathway.iloc[0], n_sets=len(c), n_fdr_005=int(sum(sig[n] for n in c)),
                         shared_genes=";".join(top), named_sets=";".join(names)))
    # sets outside the groups
    ax.text(0.04, 2.53, f"Other gene sets ({len(rest_nodes)})", fontsize=TXT, ha="left", va="center", color=INK)
    for n in rest_nodes:
        if sig[n]:
            ax.text(pos[n][0] - 0.06, pos[n][1] + 0.15, pretty(t.pathway[n]), fontsize=TXT, ha="left", va="center", color=INK)
    # key
    kx, ky = 3.08, 2.66
    for v, dx in ((2, 0.0), (4, 0.2), (6, 0.44)):
        ax.scatter([kx + 0.07 + dx], [ky], s=9 + 12 * (v - 2), c=RUST["200"], edgecolors=RUST["400"], linewidths=0.4, clip_on=False, zorder=3)
        ax.text(kx + 0.07 + dx, ky + 0.17, f"{v}", fontsize=TXT, ha="center", va="center")
    ax.text(kx + 0.68, ky + 0.17, "−log$_{10}$ $P$", fontsize=TXT, ha="left", va="center")
    ax.scatter([kx + 0.95], [ky], s=30, c=RUST["400"], edgecolors=INK, linewidths=0.9, clip_on=False, zorder=3)
    ax.text(kx + 1.04, ky, "FDR < 0.05", fontsize=TXT, ha="left", va="center")
    ax.scatter([kx + 1.2], [ky + 0.17], s=20, c=BLUE["300"], edgecolors=BLUE["400"], linewidths=0.4, clip_on=False, zorder=3)
    ax.text(kx + 1.29, ky + 0.17, "Higher in young", fontsize=TXT, ha="left", va="center")
    net = dict(t=t, pos=pos, group=group, G=G, info=info, n_rest=len(rest_nodes), n_sig=int(sig.sum()), n_groups=len(mods))
    # the sets with the smallest P
    r = t.nsmallest(N_TOP, "p").sort_values("p")
    yy = np.arange(len(r))
    rs = r.fdr.values < 0.05
    bx.hlines(yy, 2.8, -np.log10(r.p.values), color=GRAY["100"], lw=0.6, zorder=1)
    bx.scatter(-np.log10(r.p.values), yy, s=26, c=list(np.where(rs, RUST["400"], RUST["200"])),
               edgecolors=list(np.where(rs, INK, RUST["400"])), linewidths=np.where(rs, 0.9, 0.4), zorder=3, clip_on=False)
    bx.set_yticks(yy); bx.set_yticklabels(["\n".join(textwrap.wrap(pretty(p), 21)) for p in r.pathway], linespacing=0.95)
    bx.set_xlabel("Enrichment, −log$_{10}$ $P$")
    net["companion"] = (r, yy)
    return net


def finish_e(ax, bx, net):
    r, yy = net["companion"]
    group = net["group"]
    set_domain(bx, xlim=(2.8, 6.2), ylim=(len(r) - 0.45, -0.55))
    bx.set_xticks([3, 4, 5, 6])
    bx.spines["left"].set_visible(False); bx.tick_params(axis="y", length=0)
    tr = bx.get_yaxis_transform()
    for y, n in zip(yy, r.index):
        bx.text(1.2, y, f"{group[n] + 1}" if n in group else "–", transform=tr, fontsize=TXT, ha="center", va="center", color=INK)
        q = r.fdr[n]
        bx.text(2.08, y, f"{q:.3f}" if q < 0.1 else f"{q:.2f}", transform=tr, fontsize=TXT, ha="right", va="center",
                color=INK, fontweight="bold" if q < 0.05 else "normal")
    bx.text(1.2, -0.95, "Group", transform=tr, fontsize=TXT, ha="center", va="center")
    bx.text(2.08, -0.95, "FDR", transform=tr, fontsize=TXT, ha="right", va="center")


# --------------------------------------------------------------------------------------------------- build
FW, FH = 7.2, 8.34                      # inches


def build():
    fig = plt.figure(figsize=(FW, FH))

    def A(x, y, w, h):                  # inches from the top left
        return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH])

    axd = {"a": A(0.1, 0.1, 2.3, 1.787), "b1": A(3.25, 0.24, 1.38, 1.38), "b2": A(5.3, 0.24, 1.38, 1.38),
           "c": A(0.5, 2.42, 2.95, 2.05), "d": A(4.42, 2.5, 2.5, 1.86),
           "e": A(0.05, 5.08, EW, EH), "e2": A(6.15, 5.3, 0.48, 2.3)}
    cax = A(6.78, 0.68, 0.07, 0.6)
    panel_a(axd["a"])
    b_axes = [axd["b1"], axd["b2"]]
    hb, brows, lim = panel_b(b_axes)
    lab, counts = panel_c(axd["c"])
    dtab, dmeta = panel_d(axd["d"])
    net = panel_e(axd["e"], axd["e2"])
    trim_axes(fig, exclude=[axd["a"], axd["e"], cax])
    set_domain(axd["d"], xlim=(10, K_MAX), ylim=(0, 400))
    finish_b(b_axes, brows, lim, hb, cax)
    axd["c"].set_xticks(range(-4, 7, 2)); axd["c"].set_yticks(range(0, 61, 20))
    set_domain(axd["c"], xlim=(-5.7, 7.7), ylim=(-2.5, 72))
    finish_c(axd["c"], lab, counts)
    finish_e(axd["e"], axd["e2"], net)
    add_panel_labels(fig, {"a": axd["a"], "c": axd["c"]})
    for letter, (x, y) in {"b": (2.62, 0.2), "d": (3.82, 2.36), "e": (0.05, 4.98)}.items():
        fig.text(x / FW, 1 - y / FH, letter, fontsize=TYPE["panel_letter"], fontweight="bold", color="black", ha="left", va="center")
    PROV.mkdir(exist_ok=True)
    pd.DataFrame(brows).to_csv(PROV / "panel_b_age_group_comparisons.csv", index=False)
    lab[["cpg_id", "gene", "kind", "x", "lp", "whole_blood_q_bh", "in_any_of_four_clocks"]].to_csv(PROV / "panel_c_labelled_cpgs.csv", index=False)
    pd.DataFrame([counts]).to_csv(PROV / "panel_c_counts.csv", index=False)
    dtab.assign(**dmeta).to_csv(PROV / "panel_d_clock_cpgs_by_ranking.csv", index=False)
    t = net["t"]
    t.assign(group=[net["group"].get(n, -1) + 1 for n in range(len(t))])[["pathway", "collection", "NES", "p", "fdr", "size", "group"]].to_csv(
        PROV / "panel_e_gene_sets.csv", index=False)
    pd.DataFrame([dict(a=t.pathway[i], b=t.pathway[j], jaccard=d["w"]) for i, j, d in net["G"].edges(data=True)]).to_csv(
        PROV / "panel_e_edges.csv", index=False)
    pd.DataFrame(net["info"]).to_csv(PROV / "panel_e_groups.csv", index=False)
    return fig, axd


if __name__ == "__main__":
    fig, axd = build()
    if "--draft" in sys.argv:            # layout iteration only; the deliverable always goes through the preflight below
        fig.savefig(HERE / "figure5_draft.png", dpi=170, facecolor="white")
        sys.exit(0)
    save_figure(fig, HERE / "figure5_attribution_2026-11-10.pdf", axes_dict=axd, dpi=600,
                review_png=HERE / "figure5_attribution_2026-11-10_review.png")
    fig.savefig(HERE / "figure5_attribution_2026-11-10.png", dpi=300, bbox_inches="tight")
    print("saved")
