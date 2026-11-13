"""Original Figure 6 rebuilt on the submitted artwork with the single-clock comparisons (10 November 2026).

a  submitted schematic with relabelled boxes (as plot_orig_fig6_2026-10-23.py); the comparator row is removed;
   the death endpoint is labelled "Death" and the 9,866 participants "training" (Methods)
b  Sankey redrawn in the submitted style with the seven disease categories analysed and their 55 constituent
   diseases (Supplementary Table 7; endpoint definitions in the Generation Scotland audit file below); the
   submitted panel also showed two groups that are not endpoints (liver and digestive, other)
c  C-index per endpoint on complete profiles: MethylGPT's held-out risk added to GrimAge2 or DunedinPACE
   (points) against the same clock alone (horizontal lines), weights fitted within held-out folds;
   * q < 0.05 across endpoints
d  heatmap, 8 endpoints by 7 reduced or noisy measurement settings: C-index of the MethylGPT combined model
   minus the best single comparator of the setting (fold models fitted on complete profiles, applied unchanged);
   * q < 0.05 across endpoints. Noise settings are simulated on the measured profiles.
e  Kaplan-Meier curves of the top decile, middle 80% and bottom decile of risk residualized on age and sex,
   MethylGPT + DunedinPACE against DunedinPACE alone, death and type 2 diabetes (released rows only)

Data: ../artifacts/large_gain_2026-11-08/Q/Q_endpoint_contrasts_2026-11-08.csv
      ../artifacts/large_gain_2026-11-08/S/S_cindex_endpoint_contrasts_2026-11-08.csv
      ../artifacts/large_gain_2026-11-09/T/T_km_curves_2026-11-09.csv, T_endpoint_contrasts_2026-11-09.csv
      ../artifacts/gs_cox/gs_endpoint_definitions_common_split_2026-06-29_audit.json
Run: MPLBACKEND=Agg python plot_orig_fig6_2026-11-10.py
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("orig6", HERE / "plot_orig_fig6_2026-10-23.py")
o6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o6)
LG = HERE.parent / "artifacts"
PAGE_W, PAGE_H = o6.PAGE_W, o6.PAGE_H
C_TXT, D_TXT = o6.C_TXT, o6.D_TXT


# endpoint names as in Methods and Supplementary Table 7 (one name per endpoint in panels a-e)
EP_NAME = {"Neurological_Diseases": "Neurological disease", "Kidney_Diseases": "Kidney disease", "Cancers": "Cancer",
           "Cardiovascular_Diseases": "Cardiovascular disease", "Autoimmune_Diseases": "Autoimmune disease",
           "date_of_death": "Death", "Respiratory_Diseases": "Respiratory disease",
           "Endocrine_and_Metabolic_Diseases": "Endocrine and metabolic disease"}
EP_C = [(k, EP_NAME[k], c) for k, _, c in o6.EP_C]
EP_D = [(k, EP_NAME[k]) for k, _ in o6.EP_D]


def panel_a(cv, audit):
    o6.panel_a(cv, audit)
    for art in list(cv.patches) + list(cv.texts):     # drop the comparator row of the 23 October build
        y = art.get_position()[1] if hasattr(art, "get_position") else art.get_y()
        if 262.0 <= y <= 300.0:
            art.remove()
    for t in list(cv.texts):
        s = t.get_text()
        if s == "PPCA input +\ndense encoder":          # the model is named once; its parts are in Fig. 1a
            t.remove()
        elif s == "MethylGPT":
            t.set_y(58.2)                                # centre of the model box
        elif s == "Mortality":
            t.set_text("Death")
        elif s.startswith("Mortality + "):
            t.set_text(s.replace("Mortality + ", "Death + ", 1))
        elif " development\n" in s:
            t.set_text(s.replace(" development\n", " training\n", 1))


# panel b: categories top to bottom as in the submitted Sankey, colours sampled from it
SANKEY = [("Respiratory_Diseases", "#46b7d0"), ("Neurological_Diseases", "#feefad"), ("Autoimmune_Diseases", "#97ceb3"),
          ("Cardiovascular_Diseases", "#5bc4be"), ("Cancers", "#f26a6b"), ("Kidney_Diseases", "#975aa4"),
          ("Endocrine_and_Metabolic_Diseases", "#fbb147")]
DISEASE_NAME = {"Alzheimers": "Alzheimer's disease", "Parkinsons": "Parkinson's disease", "COPD": "COPD",
                "Celiac_Disease": "Coeliac disease", "Graves_Disease": "Graves' disease",
                "Guillain_Barre_Syndrome": "Guillain-Barré syndrome", "Hashimotos_Thyroiditis": "Hashimoto's thyroiditis",
                "Sjogrens_Syndrome": "Sjögren's syndrome", "Autoimmune_Hemolytic_Anemia": "Autoimmune haemolytic anaemia",
                "Pernicious_Anemia": "Pernicious anaemia", "Colon_Rectum_cancer": "Colorectal cancer", "Leukemia": "Leukaemia",
                "Non_Hodgkin_Lymphoma": "Non-Hodgkin lymphoma", "Ovary_cancer": "Ovarian cancer",
                "Pancrease_cancer": "Pancreatic cancer", "Aortic_aneurysm_dissection": "Aortic aneurysm and dissection",
                "Arterial_embolism_thrombosis": "Arterial embolism and thrombosis"}


def disease_label(col):
    return DISEASE_NAME.get(col, col.replace("_", " ").capitalize())


def panel_b(cv, audit):
    """Diseases (left) joined to the seven disease categories analysed (right)."""
    from matplotlib.colors import to_rgb
    o6.cover(cv, 346.5, 0.0, PAGE_W, o6.TOP_BAND)       # the submitted raster, which also showed two non-endpoint groups
    groups = []
    for key, col in SANKEY:
        cols = audit["label_summaries"][key]["mapped_disease_columns"].split(";")
        groups.append((key, col, [disease_label(c) for c in cols]))
    n_rows = sum(len(g[2]) for g in groups)
    assert n_rows == 55 and [len(g[2]) for g in groups] == [7, 4, 19, 10, 13, 1, 1], [len(g[2]) for g in groups]
    Y0, Y1 = 28.0, 338.0
    pitch = (Y1 - Y0) / (n_rows - 1)
    XB0, XB1, XR0, XR1 = 350.0, 359.0, 549.0, 558.0     # left bars, right boxes (pt)
    SLOT, HL = 1.6, 0.9                                   # ribbon height at the box, half height at the bar
    centres, half, row = [], [], 0
    for _, _, ds in groups:
        centres.append(Y0 + pitch * (row + (len(ds) - 1) / 2))
        half.append(max(len(ds) * SLOT, 2.4) / 2)
        row += len(ds)
    for i in range(len(groups) - 2, -1, -1):            # keep boxes and their labels apart (bottom up)
        need = max(half[i] + half[i + 1] + 3.0, 8.0)
        centres[i] = min(centres[i], centres[i + 1] - need)
    s_ = np.linspace(0, 1, 40)
    ease = 3 * s_ ** 2 - 2 * s_ ** 3
    row = 0
    for (key, col, ds), yc, h in zip(groups, centres, half):
        light = tuple(0.55 * np.array(to_rgb(col)) + 0.45)
        top = yc - len(ds) * SLOT / 2
        for j, name in enumerate(ds):
            yl = Y0 + pitch * row
            yr = top + SLOT * (j + 0.5)
            xs = XB1 + (XR0 - XB1) * s_
            upper = (yl - HL) + ((yr - SLOT / 2) - (yl - HL)) * ease
            lower = (yl + HL) + ((yr + SLOT / 2) - (yl + HL)) * ease
            cv.add_patch(Polygon(np.c_[np.r_[xs, xs[::-1]], np.r_[upper, lower[::-1]]], closed=True, facecolor=col,
                                 edgecolor="none", alpha=0.28, zorder=6))
            cv.add_patch(Rectangle((XB0, yl - HL), XB1 - XB0, 2 * HL, facecolor=light, edgecolor="#4d4d4d", lw=0.4,
                                   zorder=6.5))
            cv.text(XB1 + 2.5, yl, name, fontsize=6, color="#231f20", ha="left",
                    va="center", zorder=7)
            row += 1
        cv.add_patch(Rectangle((XR0, yc - h), XR1 - XR0, 2 * h, facecolor=col, edgecolor="#4d4d4d", lw=0.5, zorder=6.5))
        cv.text(XR0 - 2.5, yc, EP_NAME[key], fontsize=6, color="#231f20", ha="right", va="center", zorder=7)
    return {EP_NAME[k]: len(ds) for k, _, ds in groups}


def panel_c(ax, q):
    q = q[q.scenario == "full"]
    stars = []
    for i, (key, _, col) in enumerate(EP_C):
        for dx, tag, mk in ((-0.19, "D_G", "o"), (0.19, "D_D", "D")):
            r = q[(q.endpoint == key) & q.contrast.str.startswith(tag)].iloc[0]
            ax.hlines(r.c_index_b, i + dx - 0.15, i + dx + 0.15, color="#5a5a5a", lw=1.3, zorder=2)
            ax.vlines(i + dx, r.c_index_b, r.c_index_a, color=col, lw=0.9, alpha=0.6, zorder=2)
            ax.plot(i + dx, r.c_index_a, ls="", marker=mk, ms=5.0 if mk == "o" else 4.3, mfc=col, mec="black", mew=0.5,
                    zorder=4, clip_on=False)
            if r.q_bh_across_endpoints < 0.05:
                ax.text(i + dx, max(r.c_index_a, r.c_index_b) + 0.004, "*", ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color=C_TXT)
                stars.append((key, tag))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_linewidth(0.9)
        ax.spines[s_].set_position(("outward", 4))
    ax.set_xlim(-0.55, len(EP_C) - 0.45)
    ax.set_ylim(0.60, 0.90)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9])
    ax.set_yticks(np.arange(0.60, 0.901, 0.01), minor=True)
    ax.set_xticks(range(len(EP_C)))
    ax.set_xticklabels([l for _, l, _ in EP_C], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("C-index", fontsize=8.5, color=C_TXT, labelpad=2)
    ax.tick_params(axis="both", labelsize=8.5, colors=C_TXT, length=3.5, width=0.8)
    ax.tick_params(axis="y", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)
    ax.set_title("MethylGPT added to each clock, complete profiles", fontsize=6.5, fontweight="bold", color=C_TXT, pad=12)
    handles = [Line2D([], [], ls="", marker="o", ms=4.6, mfc="#bdbdbd", mec="black", mew=0.5, label="MethylGPT + GrimAge2"),
               Line2D([], [], ls="", marker="D", ms=4.0, mfc="#bdbdbd", mec="black", mew=0.5, label="MethylGPT + DunedinPACE"),
               Line2D([], [], color="#5a5a5a", lw=1.3, label="Clock alone")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=3, fontsize=5.5, frameon=False,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0)
    return stars


SET_D = [("J", "hm27", "27K-array\nCpGs"), ("J", "random_90", "Random 10%\nof CpGs"),
         ("M", "gauss_0.05", "Gaussian noise,\nSD 0.05"), ("M", "gauss_0.10", "Gaussian noise,\nSD 0.10"),
         ("M", "depth_30", "Read depth 30"), ("M", "depth_10", "Read depth 10"), ("M", "depth_5", "Read depth 5")]
D_LIM = 0.05


def panel_d(ax, cax, s):
    s = s[s.contrast == "best_single"]
    val = np.full((len(EP_D), len(SET_D)), np.nan)
    qv = np.full_like(val, np.nan)
    comps = []
    for c, (an, sc, _) in enumerate(SET_D):
        t = s[(s.analysis == an) & (s.scenario == sc)].set_index("endpoint")
        assert len(t) == 8, (an, sc, len(t))
        comps.append(t.comparator.iloc[0])
        for r, (ep, _) in enumerate(EP_D):
            val[r, c] = t.loc[ep, "difference"]
            qv[r, c] = t.loc[ep, "q_bh_across_endpoints"]
    im = ax.imshow(val, cmap=plt.get_cmap("RdBu_r"), norm=TwoSlopeNorm(vmin=-D_LIM, vcenter=0, vmax=D_LIM), aspect="auto",
                   interpolation="nearest")
    for k in range(val.shape[1] + 1):
        ax.axvline(k - 0.5, color="white", lw=0.8)
    for k in range(val.shape[0] + 1):
        ax.axhline(k - 0.5, color="white", lw=0.8)
    for r in range(val.shape[0]):
        for c in range(val.shape[1]):
            if qv[r, c] < 0.05:
                ax.text(c, r + 0.12, "*", ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if abs(val[r, c]) > 0.6 * D_LIM else "#1a1a1a")
    ax.set_xticks(range(len(SET_D)))
    ax.set_xticklabels([l for *_, l in SET_D], rotation=45, ha="right", rotation_mode="anchor", fontsize=6.5, linespacing=1.0)
    ax.set_yticks(range(len(EP_D)))
    ax.set_yticklabels([l for _, l in EP_D], fontsize=7.5)
    ax.tick_params(length=0, colors=D_TXT, pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("Endpoint", fontsize=8, color=D_TXT, labelpad=4)
    ax.set_xlabel("Measurement available at prediction", fontsize=8, color=D_TXT, labelpad=2)
    cb = plt.colorbar(im, cax=cax, extend="both", extendfrac=0.04)
    cb.set_ticks([-0.05, -0.025, 0, 0.025, 0.05])
    cb.set_ticklabels(["−0.05", "−0.025", "0", "0.025", "0.05"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, colors=D_TXT, length=2, width=0.5)
    cb.set_label("ΔC-index, MethylGPT − best comparator", fontsize=6.5, color=D_TXT, labelpad=3)
    return val, qv, comps


KM_COL = {"top_decile": "#c0392b", "middle_80pct": "#8c8c8c", "bottom_decile": "#3b6fb6"}
KM_LAB = {"top_decile": "Top 10% of risk", "middle_80pct": "Middle 80%", "bottom_decile": "Bottom 10%"}


def panel_e(fig, rect, km, tc):
    km = km[(km.scenario == "full") & (~km.suppressed.astype(bool))]
    eps = [("date_of_death", "Death"), ("Endocrine_and_Metabolic_Diseases", "Type 2 diabetes")]
    out = {}
    for k, (ep, name) in enumerate(eps):
        ax = fig.add_axes(rect(52 + k * 150, 266, 52 + k * 150 + 108, 322), zorder=3)
        for model, ls in (("C2_mean", (0, (3, 1.6))), ("QD_mean", "-")):
            for g in ("bottom_decile", "middle_80pct", "top_decile"):
                d = km[(km.endpoint == ep) & (km.model == model) & (km.group == g)].sort_values("year")
                if len(d):                      # yearly values; years with fewer than 5 events in a cell are withheld
                    solid = ls == "-"
                    ax.plot(d.year, 1 - d.survival, color=KM_COL[g], lw=1.1 if solid else 0.9, ls=ls if len(d) > 2 else "",
                            marker="o", ms=2.2, mfc=KM_COL[g] if solid else "white", mec=KM_COL[g], mew=0.6, clip_on=False)
        r = tc[(tc.scenario == "full") & tc.contrast.str.startswith("T_D") & (tc.endpoint == ep) &
               (tc.metric == "event_capture_top10")].iloc[0]
        out[ep] = (r.value_a, r.value_b)
        ax.text(0.03, 0.97, f"{name}\nEvents in top 10%:\n{100 * r.value_a:.1f}% against {100 * r.value_b:.1f}%",
                transform=ax.transAxes, fontsize=5.5, va="top", color=C_TXT, linespacing=1.15)
        ax.set_xlim(0, 17)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_ylim(0, None)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_linewidth(0.8)
            ax.spines[s_].set_position(("outward", 3))
        ax.tick_params(labelsize=6.5, length=3, width=0.7, colors=C_TXT)
        ax.set_xlabel("Years from baseline", fontsize=7, labelpad=1.5, color=C_TXT)
        if k == 0:
            ax.set_ylabel("Cumulative incidence", fontsize=7, labelpad=2, color=C_TXT)
    h = [Line2D([], [], color=KM_COL[g], lw=1.1, label=KM_LAB[g]) for g in ("top_decile", "middle_80pct", "bottom_decile")]
    h += [Line2D([], [], color="#333333", lw=1.1, label="MethylGPT + DunedinPACE"),
          Line2D([], [], color="#333333", lw=0.9, ls=(0, (3, 1.6)), label="DunedinPACE alone")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(46 / PAGE_W, 1 - 246 / PAGE_H), ncol=5, fontsize=5.5,
               frameon=False, handlelength=1.6, handletextpad=0.4, columnspacing=0.9)
    return out


def build():
    q = o6.load_csv(LG / "large_gain_2026-11-08/Q/Q_endpoint_contrasts_2026-11-08.csv")
    s = o6.load_csv(LG / "large_gain_2026-11-08/S/S_cindex_endpoint_contrasts_2026-11-08.csv")
    km = o6.load_csv(LG / "large_gain_2026-11-09/T/T_km_curves_2026-11-09.csv")
    tc = o6.load_csv(LG / "large_gain_2026-11-09/T/T_endpoint_contrasts_2026-11-09.csv")
    audit = o6.load_json(o6.A / "gs_endpoint_definitions_common_split_2026-06-29_audit.json")
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))

    def rect(x0, y0, x1, y1):
        return [x0 / PAGE_W, 1 - y1 / PAGE_H, (x1 - x0) / PAGE_W, (y1 - y0) / PAGE_H]

    cv = fig.add_axes([0, 0, 1, 1], zorder=0)
    cv.set_xlim(0, PAGE_W); cv.set_ylim(PAGE_H, 0); cv.axis("off")
    cv.imshow(o6.render_artwork(), extent=(0, PAGE_W, o6.TOP_BAND, 0), interpolation="none", zorder=1)
    panel_a(cv, audit)
    panel_b(cv, audit)
    cap = panel_e(fig, rect, km, tc)
    stars = panel_c(fig.add_axes(rect(52, 386, 254, 462), zorder=2), q)
    val, qv, comps = panel_d(fig.add_axes(rect(396, 374, 542, 487), zorder=2), fig.add_axes(rect(551.5, 374, 557, 487), zorder=2), s)
    o6.txt(cv, 469.0, 358.0, "C-index difference, reduced or noisy measurements", 8.5, color=D_TXT)
    for lab, x, y in (("a", 10, 16.5), ("b", 340, 18.5), ("e", 10, 262), ("c", 10, 366), ("d", 336, 361)):
        cv.text(x, y, lab, fontsize=12, fontweight="bold", color="#231f20", ha="left", va="baseline", zorder=8)
    return fig, stars, val, qv, comps, cap


def main():
    o6.OUT.mkdir(parents=True, exist_ok=True)
    fig, stars, val, qv, comps, cap = build()
    fig.savefig(o6.OUT / "fig6.pdf")
    fig.savefig(o6.OUT / "fig6.png", dpi=300)
    plt.close(fig)
    o6.side_by_side(o6.OUT / "fig6.png", o6.OUT / "fig6_side_by_side.png")
    print("panel c stars:", stars)
    print("panel d comparators per setting:", list(zip([s for _, s, _ in SET_D], comps)))
    print(np.round(val, 4)); print("cells below zero:", int((val < 0).sum()), "q<0.05:", int((qv < 0.05).sum()),
                                   "beyond colour limits:", int((np.abs(val) > D_LIM).sum()))
    print("panel e top-decile capture (MethylGPT + DunedinPACE, DunedinPACE):", cap)


if __name__ == "__main__":
    main()
