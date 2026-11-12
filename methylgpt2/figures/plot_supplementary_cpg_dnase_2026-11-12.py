"""Supplementary Figure 9: CpG representations, DNase hypersensitivity and baselines, without the model-part series.

Redraw of plot_supplementary_cpg_dnase_2026-10-18.py keeping MethylGPT's input-layer loadings and the comparators (raw
beta values, their first 64 principal components, the mean, SD and probe-type baseline and random vectors). The series
built from parts of the model are removed: the encoder's input weights (mask, centred values, residuals), the query
embedding, the decoder state h (full query, query embedding, randomly initialized), the contrasts h minus loadings and h
minus randomly initialized h, and the decoder-state UMAP row.

Panels b-g are drawn from the local aggregate files:
  ../artifacts/eval/cpg_representation_si_2026-10-18.json          kNN, AUC and interval results
  ../artifacts/eval/cpg_representation_si_figdata_2026-10-18.npz   island confusion matrices and vote histograms
(benchmark_cpg_representations_si_2026-10-18.py). Panel a cannot be redrawn: its UMAP coordinates,
../artifacts/eval/cpg_umap_si_2026-10-18.npz, are not on this machine. Panel a is therefore taken from the installed
figure (supplementary_figure9_installed_2026-11-10.pdf/.png, a copy of manuscript_v13/figures/supplementary_figure9.*):
its top band (island-relation legend, column titles and the loadings row) and its raw-beta row are kept unchanged, the
decoder-state row between them is cut out, and the raw-beta row moves up into its place at the same row spacing. The cut
lines lie in blank bands of the installed PNG (checked below) and are converted to PDF points at 300 dpi, so the PDF and
the PNG are cut at the same places. Panels b-g are gated by the nature-figure preflight and placed below panel a.

Outputs (this folder): supplementary_figure_cpg_dnase_2026-11-12_bg.pdf/.png (panels b-g) and
supplementary_figure_cpg_dnase_2026-11-12.pdf/.png (the assembled figure).
Run: MPLBACKEND=Agg /opt/homebrew/bin/python3 plot_supplementary_cpg_dnase_2026-11-12.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "nature-figure")   # the skill's current preflight
from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,  # noqa: E402
                                 pad_categorical_axis, ref_line, label_panel_group, confusion_matrix_panel,
                                 ERRORBAR, TYPE, RUST, GRAY, BLUE, GOLD, SAGE, INK)

A = HERE.parent / "artifacts"
SRC_JSON = A / "eval/cpg_representation_si_2026-10-18.json"
SRC_NPZ = A / "eval/cpg_representation_si_figdata_2026-10-18.npz"
OLD_PDF = HERE / "supplementary_figure9_installed_2026-11-10.pdf"
OLD_PNG = HERE / "supplementary_figure9_installed_2026-11-10.png"
NAME = "supplementary_figure_cpg_dnase_2026-11-12"
S = 300 / 72
# cut lines in the installed 300-dpi PNG (pixels): top band [0, A_END) holds the legend, column titles and loadings row;
# the raw-beta row is [B_START, B_END); B_START is chosen so that the row's first ink sits where the decoder-state row's
# first ink sat (pixel 545), keeping the row spacing
A_END, B_START, B_END = 508, 786, 1016
retro_style()

REPS = [("loadings", "Input-layer loadings", RUST["400"]),
        ("raw_profiles", "Raw beta", BLUE["400"]),
        ("raw_pca64", "Raw beta, 64 PCs", BLUE["200"]),
        ("baseline", "Mean, SD, probe type", GRAY["400"]),
        ("random", "Random vectors", GRAY["200"])]
LABELS = [("island_relation", "CpG island relation", 0.25), ("enhancer", "Enhancer", 0.5), ("dnase_hs", "DNase HS", 0.5),
          ("x_chromosome", "Chromosome X", 0.5)]
ISL = [("Island", "Island", RUST["400"]), ("Shore", "Shore", GOLD["400"]), ("Shelf", "Shelf", SAGE["400"]), ("OpenSea", "Open sea", BLUE["300"])]


def dodge(n, step):
    """Offsets for n series at the same spacing as the earlier figure."""
    return (np.arange(n) - (n - 1) / 2) * step


def interval(ax, x, y, lo, hi, col, filled=True, vertical=False, ms=3.2, marker="o", label=None):
    kw = dict(mfc=col, mec=INK, mew=0.3) if filled else dict(mfc="white", mec=col, mew=0.8)
    err = [[x - lo], [hi - x]] if not vertical else [[y - lo], [hi - y]]
    if vertical:
        ax.errorbar(x, y, yerr=err, fmt=marker, ms=ms, ecolor=col, **ERRORBAR["forest"], clip_on=False, zorder=3, label=label, **kw)
    else:
        ax.errorbar(x, y, xerr=err, fmt=marker, ms=ms, ecolor=col, **ERRORBAR["forest"], clip_on=False, zorder=3, label=label, **kw)


def draw_bg():
    res = json.loads(SRC_JSON.read_text())
    R = res["results"]
    fd = np.load(SRC_NPZ)
    nE, nX = R["loadings"]["island_relation"]["grouped_cv"]["n"], R["loadings"]["x_chromosome"]["grouped_cv"]["n"]

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.76), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.03)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.28, 1.45])
    rb = outer[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.0, 1.3, 1.0])
    r3 = outer[2].subgridspec(1, 3, width_ratios=[0.8, 1.0, 1.3])

    # ---- b: kNN balanced accuracy; filled all CpGs, open within mean-beta decile x probe type
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

    # ---- c: primary comparisons (loadings against the comparators)
    cx = fig.add_subplot(r2[0])
    D = res["differences"]
    CON = [("raw_pca64 - loadings", "Loadings − raw, 64 PCs", BLUE["300"], -1),
           ("baseline - loadings", "Loadings − baseline", GRAY["400"], -1)]
    xx = np.arange(len(LABELS))
    for (cname, cl, col, sign), off in zip(CON, dodge(len(CON), 0.2)):
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
    DREPS = [("loadings", "Input-layer loadings", RUST["400"], "o"), ("raw_profiles", "Raw beta", BLUE["400"], "s"),
             ("raw_pca64", "Raw beta, 64 PCs", BLUE["200"], "s"), ("baseline", "Mean, SD, probe type", GRAY["400"], "D"),
             ("random", "Random vectors", GRAY["200"], "D")]
    for (rk, rl, col, mk), off in zip(DREPS, dodge(len(DREPS), 0.64 / 6)):
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
    GREPS = [r for r in DREPS if r[0] in ("loadings", "raw_profiles", "raw_pca64", "baseline")]
    for (rk, rl, col, mk), off in zip(GREPS, dodge(len(GREPS), 0.12)):
        au = R[rk]["island_relation"]["auc_one_vs_rest"]
        for k, (key, _, _) in enumerate(ISL):
            interval(gx, k + off, au[key]["auc"], *au[key]["ci95"], col, True, vertical=True, ms=2.8, marker=mk,
                     label=rl if k == 0 else None)
    gx.set_xticks(xx)
    gx.set_xticklabels([l for _, l, _ in ISL])
    gx.set_ylabel("One-vs-rest AUC")

    trim_axes(fig, exclude=[ex])
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
    label_panel_group(fig, bx, "b")
    add_panel_labels(fig, {"c": cx, "d": dx, "e": gx, "f": ex, "g": fx})
    axd = {**{f"b{i}": a for i, a in enumerate(bx)}, "c": cx, "d": dx, "e": gx, "f": ex, "g": fx}
    save_figure(fig, str(HERE / f"{NAME}_bg.pdf"), axes_dict=axd, review_png=str(HERE / f"{NAME}_bg_review.png"))
    fig.savefig(str(HERE / f"{NAME}_bg.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def assemble():
    old = np.asarray(Image.open(OLD_PNG).convert("RGB"))
    gray = old.mean(2)
    for r in (A_END, B_START, B_END):
        if gray[r].min() < 250:
            sys.exit(f"cut row {r} of {OLD_PNG.name} is not blank; stop")
    bg = np.asarray(Image.open(HERE / f"{NAME}_bg.png").convert("RGB"))
    top = np.concatenate([old[:A_END], old[B_START:B_END]], axis=0)
    W = max(top.shape[1], bg.shape[1])
    canvas = np.full((top.shape[0] + bg.shape[0], W, 3), 255, np.uint8)
    canvas[:top.shape[0], :top.shape[1]] = top
    canvas[top.shape[0]:, :bg.shape[1]] = bg
    Image.fromarray(canvas).save(HERE / f"{NAME}.png", dpi=(300, 300))

    src = fitz.open(OLD_PDF)
    sp = src[0]
    if abs(sp.rect.height * S - old.shape[0]) > 2:
        sys.exit("installed PDF and PNG do not match at 300 dpi; stop")
    bgdoc = fitz.open(HERE / f"{NAME}_bg.pdf")
    bw, bh = bgdoc[0].rect.width, bgdoc[0].rect.height
    a_end, b0, b1 = A_END / S, B_START / S, B_END / S
    W0, H0 = sp.rect.width, sp.rect.height

    def keep_only(y0, y1):
        """A copy of the installed page with everything outside [y0, y1) deleted, so the clipped placement carries no
        hidden content (in particular no decoder-state images)."""
        d = fitz.open(OLD_PDF)
        pg = d[0]
        for r in (fitz.Rect(0, 0, W0, y0), fitz.Rect(0, y1, W0, H0)):
            if r.height > 0:
                pg.add_redact_annot(r, fill=False)
        pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                            text=fitz.PDF_REDACT_TEXT_REMOVE)
        return d

    part_a, part_b = keep_only(0, a_end), keep_only(b0, b1)
    if [len(p[0].get_image_info()) for p in (part_a, part_b)] != [6, 6]:
        sys.exit("expected six UMAP images in each kept row; stop")
    top_h = a_end + (b1 - b0)
    out = fitz.open()
    page = out.new_page(width=max(W0, bw), height=top_h + bh)
    page.show_pdf_page(fitz.Rect(0, 0, W0, a_end), part_a, 0, clip=fitz.Rect(0, 0, W0, a_end))
    page.show_pdf_page(fitz.Rect(0, a_end, W0, top_h), part_b, 0, clip=fitz.Rect(0, b0, W0, b1))
    page.show_pdf_page(fitz.Rect(0, top_h, bw, top_h + bh), bgdoc, 0)
    out.save(HERE / f"{NAME}.pdf", garbage=3, deflate=True)
    print(f"assembled {NAME}.pdf ({page.rect.width:.1f} x {page.rect.height:.1f} pt) and {NAME}.png "
          f"({canvas.shape[1]} x {canvas.shape[0]} px); panel b-g part {bw:.1f} x {bh:.1f} pt, {bg.shape[1]} x {bg.shape[0]} px")


if __name__ == "__main__":
    for path in (SRC_JSON, SRC_NPZ, OLD_PDF, OLD_PNG):
        if not path.exists():
            sys.exit(f"Cannot proceed - required data file not found: {path}")
    draw_bg()
    assemble()
