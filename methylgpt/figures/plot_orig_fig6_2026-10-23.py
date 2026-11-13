"""Original Figure 6 rebuilt for the revision on the submitted artwork (fig6.ai).

a  original panel-a artwork with relabelled boxes (rebuilt MethylGPT, three inputs, elastic-net Cox),
   outcome text, cohort numbers and comparator row; the Apply arrow and intervention strip are removed
b  Sankey kept unchanged (original raster, as placed in fig6.ai)
c  C-index per endpoint with only 27K-array CpGs: MethylGPT point against a horizontal line for the linear
   model on raw CpGs (Cox regression, original protocol, refitted), as in the original panel; * q < 0.05
d  heatmap 8 endpoints x 6 comparators with 27K-array CpGs, colour = MethylGPT combined minus comparator
   C-index, asterisks for Benjamini-Hochberg q < 0.05 (as computed over the scenario contrasts)

Data (aggregate tables, local):
  ../artifacts/gs_cox/full_coverage_cindex_table_2026-10-01.csv
  ../artifacts/gs_cox/stacked_2026-10-04.json
  ../artifacts/gs_cox/bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09_finite_p_2026-10-09.csv (scenario hm27)
  ../artifacts/gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json
Run: MPLBACKEND=Agg python plot_orig_fig6_2026-10-23.py
"""
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts/gs_cox"
ORIG = HERE / "original_submission"
OUT = HERE / "original_rebuild"

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": True, "svg.fonttype": "none"})

PAGE_W = 595.276                    # pt, width of the original PDF page
PAGE_H = 4706 / 600 * 72            # pt, height of the original figure (fig6.png at 600 dpi)
TOP_BAND = 346.0                    # pt, panels a and b taken from the artwork above this line
DPI_ART = 600

C_TXT = "#050707"                   # panel c text colour in the original
D_TXT = "#4c4d4f"                   # panel d text colour in the original
BOX_EDGE = "#5c5ea6"                # outline of the original output boxes


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Cannot proceed - required data file not readable: {p} ({e})")


def load_csv(p, **kw):
    try:
        return pd.read_csv(p, **kw)
    except OSError as e:
        sys.exit(f"Cannot proceed - required data file not found: {p} ({e})")


def render_artwork():
    """Render fig6.ai above TOP_BAND at 600 dpi with its live text removed (labels are reset below)."""
    src = ORIG / "fig6.ai"
    if not src.exists():
        sys.exit(f"Cannot proceed - required data file not found: {src}")
    doc = fitz.open(src)
    page = doc[0]
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                page.add_redact_annot(fitz.Rect(s["bbox"]))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    pix = page.get_pixmap(dpi=DPI_ART, clip=fitz.Rect(0, 0, PAGE_W, TOP_BAND), alpha=False)
    return np.asarray(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))


def cover(ax, x0, y0, x1, y1, color="white"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="none", zorder=5))


def txt(ax, x, y, s, size, weight="bold", color="#231f20", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=7,
            linespacing=1.15, **kw)


def panel_a(cv, audit):
    """Edits on the original panel-a artwork, coordinates in pt from the top-left of the page."""
    common = audit["common_intersection"]["common_sample_count"]
    split = audit["split_counts_by_any_endpoint_event"]
    max_days = max(v["duration_max"] for v in audit["label_summaries"].values())
    n_cat = sum(1 for k in audit["label_summaries"] if k != "date_of_death")

    # model box
    cover(cv, 211.5, 42.5, 249.5, 74.0)
    txt(cv, 230.4, 54.0, "MethylGPT", 6.5)
    txt(cv, 230.4, 66.5, "PPCA input +\ndense encoder", 5, weight="normal")

    # three inputs replace the single embedding stack
    cover(cv, 246.0, 18.5, 299.0, 30.0)
    txt(cv, 272.2, 25.0, "Three inputs", 6.5, weight="normal")
    cover(cv, 261.0, 30.5, 284.0, 89.0)
    cols = [("#e3dcef", "Embedding"), ("#dbe3f1", f"25 distilled outputs"), ("#d6ebf2", "Residuals")]
    for i, (fc, lab) in enumerate(cols):
        x0 = 261.8 + i * 7.5
        cv.add_patch(FancyBboxPatch((x0, 31.5), 6.3, 56.0, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=fc, edgecolor="#4a4a5a", lw=0.6, zorder=6))
        txt(cv, x0 + 3.15, 59.5, lab, 5, weight="normal", rotation=90)

    # elastic-net Cox replaces ridge regression (live text removed from the artwork)
    txt(cv, 312.6, 58.2, "Elastic-net\nCox\nnested CV", 6)
    txt(cv, 315.0, 84.6, "5 outer × 3 inner folds\nage + sex in every model", 5, weight="normal")

    # output boxes
    cover(cv, 161.0, 125.0, 212.0, 137.5)
    txt(cv, 186.5, 131.2, "Mortality", 6.3)
    cover(cv, 225.0, 124.5, 277.5, 146.5)
    txt(cv, 251.0, 135.5, f"{n_cat} disease\ncategories", 6.3)

    # remove Apply arrow, intervention strip and the 10-year text
    cover(cv, 0.0, 190.0, 348.0, TOP_BAND)

    # endpoint and cohort statements
    txt(cv, 251.0, 214.0, f"Mortality + {n_cat} disease categories\n(incident endpoints)", 8.5)
    txt(cv, 68.0, 210.0, f"{common:,} eligible", 8.5)
    txt(cv, 68.0, 236.0, f"{split['train']['size']:,} development\n{split['val']['size']:,} validation\n"
                          f"{split['test']['size']:,} test", 6.5, weight="normal")
    txt(cv, 251.0, 241.0, f"Follow-up up to {max_days / 365.25:.1f} years ({max_days:,.0f} days)", 6.5,
        weight="normal")

    # comparator row
    txt(cv, 174.0, 271.0, "Compared with", 6.5)
    boxes = [("Age + sex", 52.0), ("Raw CpGs", 52.0), ("GrimAge2 + DunedinPACE", 100.0)]
    gap = 10.0
    total = sum(w for _, w in boxes) + gap * (len(boxes) - 1)
    x = 174.0 - total / 2
    for lab, w in boxes:
        cv.add_patch(FancyBboxPatch((x, 280.0), w, 17.0, boxstyle="round,pad=0,rounding_size=3",
                                    facecolor="white", edgecolor=BOX_EDGE, lw=1.0, zorder=6))
        txt(cv, x + w / 2, 288.5, lab, 6.3)
        x += w + gap


EP_C = [("Neurological_Diseases", "Neurological Diseases", "#4b98b7"),
        ("Kidney_Diseases", "Kidney Diseases", "#b05888"),
        ("Cancers", "Cancers", "#de952c"),
        ("Cardiovascular_Diseases", "Cardiovascular Diseases", "#c05b43"),
        ("Autoimmune_Diseases", "Autoimmune Diseases", "#926f59"),
        ("date_of_death", "Mortality", "#80a867"),
        ("Respiratory_Diseases", "Respiratory Diseases", "#6f89a0"),
        ("Endocrine_and_Metabolic_Diseases", "Endocrine and Metabolic Diseases", "#f7a540")]


def panel_c(ax, sc):
    """Original design: one MethylGPT value per endpoint against the linear model on raw CpGs
    (Cox regression, original protocol, refitted), here with only 27K-array CpGs observed."""
    for i, (key, _, col) in enumerate(EP_C):
        r = sc[(sc.endpoint == key) & (sc.comparator == "raw_cpg_v12")].iloc[0]
        m, base, q = r["c_mgpt2_combined"], r["c_comparator"], r["q_bh"]
        ax.hlines(base, i - 0.32, i + 0.32, color="#5a5a5a", lw=1.4, zorder=2)
        ax.vlines(i, base, m, color=col, lw=1.0, alpha=0.6, zorder=2)
        ax.plot(i, m, ls="", marker="o", ms=6.0, mfc=col, mec="black", mew=0.5, zorder=4, clip_on=False)
        ax.text(i, max(m, base) + 0.012, f"{m:.3f}" + ("*" if q < 0.05 else ""), ha="center", va="bottom",
                fontsize=5.5, fontweight="bold", color=C_TXT)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):              # set before tick labels: set_position resets the ticks
        ax.spines[s_].set_linewidth(0.9)
        ax.spines[s_].set_position(("outward", 4))
    ax.axhline(0.5, color="#b0b0b0", lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax.text(len(EP_C) - 0.45, 0.503, "Random (0.5)", fontsize=5, style="italic", color="#808080",
            ha="right", va="bottom")
    ax.set_xlim(-0.55, len(EP_C) - 0.45)
    ax.set_ylim(0.45, 0.92)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticks(np.arange(0.45, 0.921, 0.01), minor=True)
    ax.set_xticks(range(len(EP_C)))
    ax.set_xticklabels([l for _, l, _ in EP_C], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("C-index", fontsize=8.5, color=C_TXT, labelpad=2)
    ax.tick_params(axis="both", labelsize=8.5, colors=C_TXT, length=3.5, width=0.8)
    ax.tick_params(axis="y", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    ax.set_title("MethylGPT vs linear model on raw CpGs, 27K-array CpGs", fontsize=6.5, fontweight="bold",
                 color=C_TXT, pad=3)
    handles = [Line2D([], [], ls="", marker="o", ms=5.0, mfc="#bdbdbd", mec="black", mew=0.5, label="MethylGPT"),
               Line2D([], [], color="#5a5a5a", lw=1.4, label="Linear model, raw CpGs (baseline)")]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=5.5,
                    frameon=True, framealpha=1.0, edgecolor="none", handletextpad=0.4, borderaxespad=0.15,
                    borderpad=0.2)
    leg.set_zorder(5)


EP_D = [("Respiratory_Diseases", "Respiratory Diseases"), ("date_of_death", "Date of Death"),
        ("Autoimmune_Diseases", "Autoimmune Diseases"), ("Cancers", "Cancers"),
        ("Cardiovascular_Diseases", "Cardiovascular Diseases"),
        ("Endocrine_and_Metabolic_Diseases", "Endocrine And Metabolic Diseases"),
        ("Kidney_Diseases", "Kidney Diseases"), ("Neurological_Diseases", "Neurological Diseases")]
CMP_D = [("age_sex", "Age + sex"), ("raw_cpg_v12", "Raw CpGs,\nconverged fit"), ("raw_cpg_gpu", "Raw CpGs,\ndense fit"),
         ("grimage2", "GrimAge2"), ("dunedinpace", "DunedinPACE"), ("grimage2_dunedinpace", "GrimAge2 +\nDunedinPACE")]
D_LIM = 0.05


def panel_d(ax, cax, pb):
    assert len(pb) == 48, f"expected 48 contrasts, found {len(pb)}"
    piv = pb.pivot(index="endpoint", columns="comparator", values="difference").loc[[e for e, _ in EP_D], [c for c, _ in CMP_D]]
    q = pb.pivot(index="endpoint", columns="comparator", values="q_bh").loc[piv.index, piv.columns]
    norm = TwoSlopeNorm(vmin=-D_LIM, vcenter=0.0, vmax=D_LIM)
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for k in range(piv.shape[1] + 1):
        ax.axvline(k - 0.5, color="white", lw=0.8)
    for k in range(piv.shape[0] + 1):
        ax.axhline(k - 0.5, color="white", lw=0.8)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            if q.values[r, c] < 0.05:
                v = piv.values[r, c]
                col = "white" if abs(v) > 0.6 * D_LIM else "#1a1a1a"
                ax.text(c, r + 0.12, "*", ha="center", va="center", fontsize=9, fontweight="bold", color=col)
    ax.set_xticks(range(len(CMP_D)))
    ax.set_xticklabels([l for _, l in CMP_D], rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5,
                       linespacing=1.0)
    ax.set_yticks(range(len(EP_D)))
    ax.set_yticklabels([l for _, l in EP_D], fontsize=7.5)
    ax.tick_params(length=0, colors=D_TXT, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("Disease", fontsize=8, color=D_TXT, labelpad=4)
    ax.set_xlabel("Comparator", fontsize=8, color=D_TXT, labelpad=2)
    cb = plt.colorbar(im, cax=cax, extend="both", extendfrac=0.04)
    cb.set_ticks([-0.05, -0.025, 0, 0.025, 0.05])
    cb.set_ticklabels(["−0.05", "−0.025", "0", "0.025", "0.05"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=D_TXT, length=2, width=0.5)
    cb.set_label("ΔC-index, MethylGPT − comparator", fontsize=6.5, color=D_TXT, labelpad=3)
    return piv, q


def build(review=False):
    tab = load_csv(A / "full_coverage_cindex_table_2026-10-01.csv", index_col=0)
    stacked = load_json(A / "stacked_2026-10-04.json")["endpoints"]
    sc_all = load_csv(A / "bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09_finite_p_2026-10-09.csv")
    pb = sc_all[sc_all.scenario == "hm27"].reset_index(drop=True)
    audit = load_json(A / "gs_endpoint_definitions_common_split_2026-06-29_audit.json")

    W, H = PAGE_W / 72, PAGE_H / 72
    fig = plt.figure(figsize=(W, H))

    def rect(x0, y0, x1, y1):            # pt from top-left -> figure fraction
        return [x0 / PAGE_W, 1 - y1 / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H]

    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, PAGE_W); cv.set_ylim(PAGE_H, 0); cv.axis("off")
    cv.imshow(render_artwork(), extent=(0, PAGE_W, TOP_BAND, 0), interpolation="none", zorder=1)
    panel_a(cv, audit)

    axc = fig.add_axes(rect(52, 368, 254, 456), zorder=2)
    panel_c(axc, pb)
    axd = fig.add_axes(rect(396, 374, 542, 487), zorder=2)
    cax = fig.add_axes(rect(551.5, 374, 557, 487), zorder=2)
    piv, q = panel_d(axd, cax, pb)
    txt(cv, 469.0, 358.0, "C-index Difference vs Comparators, 27K CpGs", 10.5, color=D_TXT)

    for lab, x, y in (("a", 10, 16.5), ("b", 340, 18.5), ("c", 10, 351.5), ("d", 252, 342.5)):
        cv.text(x, y, lab, fontsize=12, fontweight="bold", color="#231f20", ha="left", va="baseline", zorder=8)

    return fig, tab, stacked, piv, q


def side_by_side(new_png, out_png):
    a = Image.open(ORIG / "fig6.png").convert("RGB")
    b = Image.open(new_png).convert("RGB")
    h = max(a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h), Image.LANCZOS)
    b = b.resize((round(b.width * h / b.height), h), Image.LANCZOS)
    gap = 60
    canvas = Image.new("RGB", (a.width + b.width + gap, h), "white")
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + gap, 0))
    canvas.save(out_png, dpi=(300, 300))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, tab, stacked, piv, q = build(review=False)
    fig.savefig(OUT / "fig6.pdf")
    fig.savefig(OUT / "fig6.png", dpi=300)
    plt.close(fig)
    fig, *_ = build(review=True)
    fig.savefig(OUT / "fig6_review.png", dpi=300)
    plt.close(fig)
    side_by_side(OUT / "fig6.png", OUT / "fig6_side_by_side.png")

    print("panel c (27K): MethylGPT vs linear model on raw CpGs, original protocol")
    print(piv["raw_cpg_v12"].round(4).to_string())
    print("panel d differences:"); print(piv.round(4).to_string())
    print("q<0.05 cells:", int((q.values < 0.05).sum()), "of", q.size,
          "| cells beyond +/-", D_LIM, ":", int((np.abs(piv.values) > D_LIM).sum()))


if __name__ == "__main__":
    main()
