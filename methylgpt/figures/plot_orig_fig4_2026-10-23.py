#!/usr/bin/env python3
"""Original Fig. 4 of the MethylGPT submission, rebuilt on the original artwork with the rebuilt-model results.

Layout, panel letters, panel positions and plot types follow figures/original_submission/fig4.pdf (A4 page,
eight embedded panel images plus text letters). The original page is the template: every retained region is
the original embedded image placed at its original page position; every replaced panel is redrawn on the same
axes rectangle measured from the original image.

Panels
  a  pie (tissue composition) recomputed; icons, methylGPT logo, arrow and the age density copied from the
     original (the per-profile training ages for the density are on Chronos only:
     si_age_label_composition_2026-10-18.npz)
  b-d original crops (needs si_embedding_pretraining_pca_2026-10-18.npz, Chronos only)
  e  MedAE per method on the AltumAge test set as points with 95% dataset-bootstrap intervals (no bars)
  f  MedAE versus input missingness, three models, 95% dataset-bootstrap intervals, array subsets as open symbols
  g  PCA of the rebuilt MethylGPT embedding (run 1) of GSE54848, coloured by predicted age (mean of three runs)
  h  predicted age (mean +- SD over replicates) versus days after sorting

Inputs (all local)
  ../artifacts/eval/si_age_label_composition_2026-10-18.csv        a  (group 'training (metadata ages)')
  ../artifacts/eval/age_ensemble_2026-10-04.json                   e  (a_altumage_test, e_clocks_altumage_test)
  ../artifacts/eval/missing_cpgs_comparators_2026-10-18_table.csv  e, f (MLP and per-scenario MedAE with CIs)
  ../artifacts/eval/missing_cpgs_comparators_2026-10-18.json       f  (observed fractions per scenario)
  ../artifacts/eval/reprogramming_time_course_si_2026-10-18.pca.csv  g
  ../artifacts/eval/reprogramming_time_course_si_2026-10-18.csv      g, h
Outputs
  original_rebuild/fig4.pdf, fig4.png (300 dpi), fig4_review.pdf/.png (with 'awaiting Chronos table' markers),
  fig4_side_by_side.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator
from PIL import Image

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts" / "eval"
ORIG = HERE / "original_submission" / "fig4.pdf"
ORIG_PNG = HERE / "original_submission" / "fig4.png"
OUT = HERE / "original_rebuild"

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
                     "svg.fonttype": "none"})
INK = "#231f20"          # colour of the original panel letters and titles
SPINE = "#717171"        # original axis colour (seaborn ticks style)
PW, PH = 595.276, 841.89  # original page size in points (A4)


def log(msg):
    print(f"[fig4] {msg}", file=sys.stderr)


def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"Cannot proceed - required data file not readable: {p} ({e})")


def load_csv(p):
    if not Path(p).exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    return pd.read_csv(p)


# ----------------------------------------------------------------------------------------- original artwork
def original_images():
    """Embedded panel images of the original page with their page bboxes (points, top-left origin)."""
    doc = fitz.open(ORIG)
    page = doc[0]
    out = {}
    for info in page.get_image_info(xrefs=True):
        pix = fitz.Pixmap(doc, info["xref"])
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        out[info["xref"]] = (arr.copy(), tuple(info["bbox"]))
    names = {88: "b", 89: "e", 90: "f", 91: "c", 92: "d", 93: "a", 94: "h", 95: "g"}
    return {names[k]: v for k, v in out.items()}


def fig_rect(x0, y0, x1, y1):
    """Page points (top-left origin) -> figure fraction rect."""
    return [x0 / PW, 1 - y1 / PH, (x1 - x0) / PW, (y1 - y0) / PH]


def place(fig, arr, bbox, z=1, alpha=None):
    ax = fig.add_axes(fig_rect(*bbox), zorder=z)
    if alpha is not None:
        arr = np.dstack([arr, alpha])
    ax.imshow(arr, interpolation="none", aspect="auto", extent=(0, 1, 0, 1))
    ax.set_axis_off()
    return ax


def crop(img, bbox, px):
    """Crop pixel box px=(x0,y0,x1,y1) of an embedded image; return the array and its page bbox."""
    arr, (bx0, by0, bx1, by1) = img
    sx, sy = arr.shape[1] / (bx1 - bx0), arr.shape[0] / (by1 - by0)
    x0, y0, x1, y1 = px
    return arr[y0:y1, x0:x1], (bx0 + x0 / sx, by0 + y0 / sy, bx0 + x1 / sx, by0 + y1 / sy)


def data_axes(fig, xcal, ycal, xlim, ylim):
    """Axes whose data limits land on the page positions measured from the original.
    xcal=(v0, pt0, v1, pt1) along x (page points), ycal likewise along y (page points, downwards)."""
    def lin(cal, v):
        v0, p0, v1, p1 = cal
        return p0 + (v - v0) * (p1 - p0) / (v1 - v0)
    x0, x1 = lin(xcal, xlim[0]), lin(xcal, xlim[1])
    yb, yt = lin(ycal, ylim[0]), lin(ycal, ylim[1])
    ax = fig.add_axes(fig_rect(x0, yt, x1, yb), zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.patch.set_alpha(0)
    return ax


def style(ax, fs_tick, fs_label, offset=5, minor_x=True, minor_y=True):
    """seaborn 'ticks' look of the original: grey offset spines trimmed to the tick range, minor ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", offset))
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(which="major", length=4, width=0.9, color=SPINE, labelsize=fs_tick, labelcolor="black", pad=2)
    ax.tick_params(which="minor", length=2, width=0.6, color=SPINE)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.label.set_size(fs_label)
    ax.yaxis.label.set_size(fs_label)


def trim(ax, x=True, y=True):
    if x:
        t = [v for v in ax.get_xticks() if ax.get_xlim()[0] - 1e-9 <= v <= ax.get_xlim()[1] + 1e-9]
        ax.spines["bottom"].set_bounds(min(t), max(t))
    if y:
        t = [v for v in ax.get_yticks() if min(ax.get_ylim()) - 1e-9 <= v <= max(ax.get_ylim()) + 1e-9]
        ax.spines["left"].set_bounds(min(t), max(t))
    # keep minor ticks within the trimmed spines, as in the original seaborn 'ticks' panels
    for axis, on, sp in ((ax.xaxis, x, "bottom"), (ax.yaxis, y, "left")):
        if on:
            lo, hi = ax.spines[sp].get_bounds()
            locs = [v for v in axis.get_minorticklocs() if lo - 1e-9 <= v <= hi + 1e-9]
            axis.set_minor_locator(FixedLocator(locs))


def text_pt(fig, x, y_top, s, size, bold=True, color=INK, ha="left"):
    """Place text whose glyph box top is at y_top (page points), as read from the original PDF spans."""
    fig.text(x / PW, 1 - (y_top + 0.905 * size) / PH, s, fontsize=size, fontweight="bold" if bold else "normal",
             color=color, ha=ha, va="baseline")


# ----------------------------------------------------------------------------------------------- panel a
PIE_COLORS = {"Whole blood": "#8dd1c6", "Other": "#f47e71", "Brain": "#b4d66c", "Skin": "#bb80b7",
              "Muscle": "#f7ee6f", "Cells": "#80b1d3"}


def panel_a(fig, imgs, review):
    tab = load_csv(A / "si_age_label_composition_2026-10-18.csv")
    tr = tab[tab.group == "training (metadata ages)"].set_index("tissue").profiles.astype(int)
    total = int(tr.sum())
    named = {"Whole blood": int(tr["Whole Blood"]), "Brain": int(tr["Brain"]), "Cells": int(tr["Cells"]),
             "Skin": int(tr["Skin"]), "Muscle": int(tr["Muscle"])}
    named["Other"] = total - sum(named.values())  # 'Other', 'no tissue label' and tissues below 1%, pooled
    log(f"a: training profiles {total}; slices {named}")
    order = ["Muscle", "Skin", "Cells", "Brain", "Other", "Whole blood"]  # clockwise from 12 o'clock; Brain kept beside the brain icon
    sizes = np.array([named[k] for k in order], float)
    frac = sizes / total

    img = imgs["a"]
    k = img[0].shape[1] / (img[1][2] - img[1][0])     # px per pt (5.72)
    cx, cy, R, r_in = 558, 485, 292, 115              # measured on the original image (px)
    ox, oy = img[1][0], img[1][1]
    pc = (ox + cx / k, oy + cy / k)
    Rp = R / k
    ax = fig.add_axes(fig_rect(pc[0] - Rp, pc[1] - Rp, pc[0] + Rp, pc[1] + Rp), zorder=2)
    wedges, _ = ax.pie(sizes, colors=[PIE_COLORS[k_] for k_ in order], startangle=90, counterclock=False,
                       radius=1.0, wedgeprops=dict(width=1 - r_in / R, linewidth=0))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    # labels: names outside, percentages inside the ring (original style; small slices carry the % with the name)
    lab_pos = {"Muscle": (-0.08, 1.10, "right"), "Skin": (0.12, 1.10, "left")}
    for w, name, f in zip(wedges, order, frac):
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        pct = f"{100 * f:.1f}%"
        if name in lab_pos:
            x, y, ha = lab_pos[name]
            ax.text(x, y, f"{name} {pct}", fontsize=5.3, ha=ha, va="center", color="black")
            ax.plot([np.cos(ang) * 1.01, x + (0.02 if ha == 'left' else -0.02)], [np.sin(ang) * 1.01, y - 0.05],
                    color="#555555", lw=0.4, clip_on=False)
            continue
        rr = 0.70
        ax.text(rr * np.cos(ang), rr * np.sin(ang), pct, fontsize=5.0, ha="center", va="center", color="black")
        lr = 1.13
        ha = "left" if np.cos(ang) > 0.15 else ("right" if np.cos(ang) < -0.15 else "center")
        x, y = lr * np.cos(ang), lr * np.sin(ang)
        if name == "Brain":      # under the brain icon, as the original label sat beside it
            x, y, ha = 1.08, -0.02, "left"
        if name == "Cells":      # clear of the skin icon
            x, y, ha = 0.80, 0.86, "left"
        ax.text(x, y, name, fontsize=5.6, ha=ha, va="center", color="black")
    # retained artwork from the original panel a: people (hole), icons, logo, chip, hourglass, arrow, text
    arr = img[0]
    s = 112
    sub, bb = crop(img, img[1], (cx - s, cy - s, cx + s, cy + s))
    yy, xx = np.mgrid[-s:s, -s:s]
    mask = (np.hypot(xx + 0.5, yy + 0.5) <= r_in - 3).astype(np.uint8) * 255
    place(fig, sub, bb, z=4, alpha=mask)
    for px in ((470, 50, 645, 150),     # muscle
               (735, 35, 850, 210),     # skin
               (842, 300, 990, 428),    # brain (old pie pixels at its left edge are keyed out below)
               (140, 625, 228, 745),    # blood drop
               (1000, 0, arr.shape[1], arr.shape[0])):  # methylGPT logo, chip, hourglass, arrow, 'Finetuning for age prediction'
        sub, bb = crop(img, img[1], px)
        if px[0] == 470:     # grey remnants of the old 'Muscle'/'Skin' labels at the lower edge of the muscle crop
            sub = sub.copy()
            a_ = sub[..., :3].astype(int)
            sub[(a_.max(-1) - a_.min(-1) < 30) & (a_.max(-1) < 225)] = 255
        if px[0] == 842:
            sub = sub.copy()
            d = np.abs(sub.astype(int) - np.array([0xf4, 0x7e, 0x71])).sum(-1)
            sub[d < 90] = 255
        place(fig, sub, bb, z=1)
    # title under the pie
    tx, ty = ox + 590 / k, oy + 838 / k
    fig.text(tx / PW, 1 - ty / PH, f"Diverse sources of samples (n = {total:,})", fontsize=7.0, fontweight="bold",
             ha="center", va="center", color="black")
    # density: per-profile training ages are on Chronos only -> original density crop kept
    sub, bb = crop(img, img[1], (140, 885, 980, arr.shape[0]))
    place(fig, sub, bb, z=1)
    if review:
        marker(fig, bb)
    return named, total


def marker(fig, bb):
    x0, y0, x1, y1 = bb
    fig.patches.append(plt.Rectangle((x0 / PW, 1 - y1 / PH), (x1 - x0) / PW, (y1 - y0) / PH, transform=fig.transFigure,
                                     fill=False, edgecolor="red", lw=0.8, ls=(0, (3, 2)), zorder=20))
    fig.text((x1 - 2) / PW, 1 - (y0 + 2) / PH, "awaiting Chronos table", fontsize=6, color="red", ha="right", va="top",
             fontweight="bold", zorder=21, bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.85))


# ----------------------------------------------------------------------------------------------- panel e
E_TEST = "#c06576"


def panel_e(fig):
    ens = load_json(A / "age_ensemble_2026-10-04.json")
    tb = load_csv(A / "missing_cpgs_comparators_2026-10-18_table.csv")
    ov = tb[(tb.part == "overall") & (tb.scenario == "all")].set_index("model_or_contrast")
    at, ck = ens["a_altumage_test"], ens["e_clocks_altumage_test"]
    z = "Zhang 2019 (>= 90% of its CpGs observed)"
    rows = [("MethylGPT", at["ensemble"]["medae"], at["ensemble"]["medae_ci95_dataset_bootstrap"], at["ensemble"]["n"]),
            ("ElasticNet", at["elastic net"]["medae"], at["elastic net"]["medae_ci95_dataset_bootstrap"], at["elastic net"]["n"]),
            ("MLP (AltumAge)", ov.loc["MLP", "estimate"], [ov.loc["MLP", "ci95_lo"], ov.loc["MLP", "ci95_hi"]], int(ov.loc["MLP", "n_profiles"])),
            ("Horvath", ck["Horvath 2013"]["clock"]["medae"], None, ck["Horvath 2013"]["clock"]["n"]),
            ("Hannum", ck["Hannum"]["clock"]["medae"], None, ck["Hannum"]["clock"]["n"]),
            ("Horvath skin & blood", ck["Horvath skin & blood"]["clock"]["medae"], None, ck["Horvath skin & blood"]["clock"]["n"]),
            ("Lin", ck["Lin"]["clock"]["medae"], None, ck["Lin"]["clock"]["n"]),
            ("Zhang 2019", ck[z]["clock"]["medae"], None, ck[z]["clock"]["n"]),
            ("AltumAge (pyaging)", ck["AltumAge"]["clock"]["medae"], None, ck["AltumAge"]["clock"]["n"])]
    # calibration measured on the original panel e image (page points)
    n = len(rows)
    ax = data_axes(fig, (-0.5, 74.7, n - 0.5, 274.3), (0, 584.2, 14, 438.6), (-0.6, n - 0.4), (0, 14))
    xs = np.arange(n)
    for x, (name, v, ci, _) in zip(xs, rows):
        if ci is not None:
            ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=E_TEST, elinewidth=0.9, capsize=2.2,
                        capthick=0.9, zorder=3)
        ax.scatter(x, v, s=22, color=E_TEST, edgecolors="#333333", linewidths=0.6, zorder=4, clip_on=False)
        ytxt = (ci[1] if ci is not None else v) + 0.35
        ax.text(x, ytxt, f"{v:.2f}", fontsize=6.0, ha="center", va="bottom", color="black")
    ax.axhline(rows[0][1], color="black", lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.set_xticks(xs)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylabel("Median Absolute Error", labelpad=3)
    style(ax, 7.3, 8.2, minor_x=False)
    ax.set_xticklabels([{"Horvath skin & blood": "Horvath skin\n& blood", "AltumAge (pyaging)": "AltumAge\n(pyaging)"}.get(r[0], r[0])
                        for r in rows], fontsize=6.8, linespacing=0.95)
    for t in ax.get_xticklabels():
        t.set_rotation(45); t.set_ha("right"); t.set_rotation_mode("anchor")
    ax.xaxis.set_minor_locator(MultipleLocator(1 / 5))
    ax.tick_params(axis="x", which="major", length=4)
    trim(ax)
    h = [Line2D([], [], marker="o", ls="none", color=E_TEST, mec="#333333", mew=0.6, ms=4.5),
         Line2D([], [], color=E_TEST, lw=0.9, marker="_", ms=5)]
    ax.legend(h, ["Test set", "95% CI (datasets)"], fontsize=6.6, loc="upper left", bbox_to_anchor=(0.03, 1.0),
              frameon=True, edgecolor="#cccccc", fancybox=False, handlelength=1.4, borderpad=0.5)
    log("e: " + "; ".join(f"{r[0]} {r[1]:.2f} (n={r[3]})" for r in rows))
    return rows


# ----------------------------------------------------------------------------------------------- panel f
F_COL = {"MethylGPT": ("#51a1c4", "o"), "Elastic net": ("#b13f54", "s"), "MLP": ("#ddb551", "^")}
F_LAB = {"MethylGPT": "MethylGPT", "Elastic net": "ElasticNet", "MLP": "MLP (AltumAge)"}


def panel_f(fig):
    tb = load_csv(A / "missing_cpgs_comparators_2026-10-18_table.csv")
    js = load_json(A / "missing_cpgs_comparators_2026-10-18.json")
    of = js["observed_fraction"]
    ov = tb[tb.part == "overall"]
    rand = [("all", 0), ("random_10", 10), ("random_30", 30), ("random_50", 50), ("random_70", 70), ("random_90", 90)]
    arrays = [(s, 100 * (1 - of[s] / of["all"])) for s in ("hm450", "epic_v2", "hm27")]

    def get(model, scen):
        r = ov[(ov.model_or_contrast == model) & (ov.scenario == scen)]
        return None if r.empty else r.iloc[0]

    ax = data_axes(fig, (0, 379.0, 90, 539.8), (0, 572.1, 24.5, 411.5), (-4, 94), (0, 24.5))
    ax.grid(True, which="major", color="#a0a0a0", lw=0.4, ls=(0, (1, 1.5)))
    out = {}
    for m, (c, mk) in F_COL.items():
        xs, ys, lo, hi = [], [], [], []
        for s, x in rand:
            r = get(m, s)
            if r is None:
                continue
            xs.append(x); ys.append(r.estimate); lo.append(r.estimate - r.ci95_lo); hi.append(r.ci95_hi - r.estimate)
        out[m] = dict(zip(xs, ys))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=mk, ms=3.6, lw=1.2, elinewidth=0.8, capsize=2.2,
                    capthick=0.8, zorder=4, clip_on=False, label=F_LAB[m])
        for s, x in arrays:
            r = get(m, s)
            ax.errorbar([x], [r.estimate], yerr=[[r.estimate - r.ci95_lo], [r.ci95_hi - r.estimate]], fmt=mk, ms=3.6,
                        mfc="white", mec=c, mew=0.9, ecolor=c, elinewidth=0.6, capsize=1.5, zorder=5, clip_on=False)
    # elastic net applied to MethylGPT-imputed profiles (dashed)
    xs, ys = [], []
    for s, x in rand:
        r = get("Elastic net, MethylGPT imputation", s)
        if r is not None:
            xs.append(x); ys.append(r.estimate)
    ax.plot(xs, ys, ls=(0, (3, 2)), color="#b13f54", lw=1.0, marker="s", ms=2.4, mfc="white", zorder=3,
            label="ElasticNet, MethylGPT imputation")
    hm27 = arrays[2][1]
    top27 = max(get(m, "hm27").ci95_hi for m in F_COL)
    ax.text(hm27, top27 + 0.5, "27K", fontsize=6.2, ha="center", va="bottom")
    ax.annotate("450K, EPIC v2", xy=(4.5, max(get(m, "epic_v2").ci95_hi for m in F_COL) + 0.3), xytext=(13, 7.2),
                fontsize=6.2, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xticks(range(0, 91, 10))
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("Input data missingness (%)", labelpad=3)
    ax.set_ylabel("Median Absolute Error", labelpad=3)
    style(ax, 8.2, 9.4, minor_x=False)
    trim(ax)
    h, l = ax.get_legend_handles_labels()
    h, l = h[1:] + h[:1], l[1:] + l[:1]
    h.append(Line2D([], [], marker="o", ls="none", mfc="white", mec="#555555", ms=3.6))
    l.append("Array CpG subsets (open)")
    ax.legend(h, l, fontsize=6.6, loc="upper left", bbox_to_anchor=(0.0, 1.03), frameon=True, edgecolor="#cccccc",
              fancybox=False, handlelength=2.0, borderpad=0.5, labelspacing=0.35)
    log("f: " + "; ".join(f"{m} " + "/".join(f"{v:.2f}" for v in d.values()) for m, d in out.items())
        + "; arrays x=" + ", ".join(f"{s} {x:.1f}%" for s, x in arrays))
    return out, arrays


# ----------------------------------------------------------------------------------------------- panel g
def panel_g(fig):
    pca = load_csv(A / "reprogramming_time_course_si_2026-10-18.pca.csv")
    ages = load_csv(A / "reprogramming_time_course_si_2026-10-18.csv").set_index("gsm")["MethylGPT mean of runs"]
    e = pca[pca.representation == "Embedding, run 1"].copy()
    e["age"] = e.gsm.map(ages)
    assert e.age.notna().all() and len(e) == 30
    ax = fig.add_axes(fig_rect(88.0, 653.0, 246.0, 790.0), zorder=3)
    ax.patch.set_alpha(0)
    vmin, vmax = 0, float(np.ceil(e.age.max() / 5) * 5)
    tc = e[e.group == "time course"]
    sc = ax.scatter(tc.PC1, tc.PC2, c=tc.age, cmap="viridis", vmin=vmin, vmax=vmax, s=28, edgecolors="black",
                    linewidths=0.9, zorder=3, clip_on=False)
    for g_, mk, lab in (("ESC", "*", "ESC (H9)"), ("iPSC reference", "D", "iPSC (201B7)")):
        r = e[e.group == g_]
        ax.scatter(r.PC1, r.PC2, c=r.age, cmap="viridis", vmin=vmin, vmax=vmax, s=55 if mk == "*" else 22,
                   marker=mk, edgecolors="black", linewidths=0.8, zorder=3, clip_on=False)
        ax.text(r.PC1.mean() + 2.0, r.PC2.mean(), lab, fontsize=5.6, color="#4d4d4d", ha="left", va="center")
    # day labels: one per replicate group, offset like the original (grey, upper right of the point)
    # one label per day at the replicate with the largest PC1 + PC2; labels nudged down when they would collide
    pts = e[["PC1", "PC2"]].to_numpy()
    placed = []

    def free(x, y):  # label box ~ (x, y) to (x + 4, y + 1.9) in data units; avoid points and other labels
        if any(abs(x - px_) < 4.3 and abs(y - py_) < 2.0 for px_, py_ in placed):
            return False
        return not np.any((pts[:, 0] > x - 0.9) & (pts[:, 0] < x + 4.2) & (pts[:, 1] > y - 0.9) & (pts[:, 1] < y + 2.6))

    groups = []
    for d, grp in tc.groupby("day"):
        if grp.PC1.max() - grp.PC1.min() > 6:   # replicates far apart (day 15): label each
            groups += [(d, grp.loc[[i]]) for i in grp.index]
        else:
            groups.append((d, grp))
    for d, grp in groups:
        r = grp.loc[grp.PC1.idxmax()]
        cands = [(r.PC1 + 0.9, r.PC2 + 0.6), (r.PC1 + 0.9, r.PC2 - 2.4)] + \
                [(grp.PC1.max() + 1.2, grp.PC2.mean() + dy - 0.9) for dy in (0, 2.2, -2.2, 4.4, -4.4, 6.6, -6.6)]
        x, y = next((c for c in cands if free(*c)), cands[0])
        placed.append((x, y))
        ax.text(x, y, f"D{int(d)}", fontsize=5.2, color="#4d4d4d", ha="left", va="bottom")
    ax.set_xlim(-28, 25)
    ax.set_ylim(-13, 31)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel("PC1", labelpad=2)
    ax.set_ylabel("PC2", labelpad=2)
    style(ax, 7.0, 8.4, offset=6)
    trim(ax)
    fig.text(165.0 / PW, 1 - 647.0 / PH, "iPSC Reprogramming Trajectory", fontsize=9.0, ha="center", va="center")
    cax = fig.add_axes(fig_rect(255.1, 652.0, 267.0, 797.2), zorder=3)
    cb = fig.colorbar(sc, cax=cax)
    cb.outline.set_edgecolor(SPINE)
    cb.outline.set_linewidth(0.7)
    cb.ax.yaxis.set_major_locator(MultipleLocator(5))
    cb.ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    cb.ax.tick_params(which="major", labelsize=7.0, length=4, width=0.9, color=SPINE)
    cb.ax.tick_params(which="minor", length=2, width=0.6, color=SPINE)
    cb.set_label("Predicted Age (years)", fontsize=8.4, labelpad=3)
    return e


# ----------------------------------------------------------------------------------------------- panel h
def panel_h(fig):
    c = load_csv(A / "reprogramming_time_course_si_2026-10-18.csv")
    tc = c[c.group == "time course"]
    ax = data_axes(fig, (0, 356.3, 50, 567.9), (0, 778.2, 60, 622.5), (-0.5, 50), (-3, 62))
    ax.grid(True, which="major", color="#878787", lw=0.5, ls=(0, (2, 2)))
    out = {}
    for col, lab, colr in (("Horvath skin & blood", "Horvath skin & blood", "blue"), ("Horvath 2013", "Horvath", "green"),
                           ("MethylGPT mean of runs", "MethylGPT", "red")):
        g = tc.groupby("day")[col].agg(["mean", "std"])
        out[lab] = g
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=colr, lw=1.4, elinewidth=0.8, capsize=2.5, capthick=0.8,
                    label=lab, zorder=3)
    ax.set_xticks(range(0, 51, 10))
    ax.set_yticks(range(0, 61, 10))
    ax.set_xlabel("Days after sorting", labelpad=3)
    ax.set_ylabel("Predicted age (years)", labelpad=3)
    style(ax, 8.2, 9.4, offset=5)
    trim(ax)
    ax.legend(fontsize=7.6, loc="upper right", frameon=True, edgecolor="#cccccc", fancybox=False, handlelength=2.0,
              borderpad=0.4, labelspacing=0.35)
    for k, g in out.items():
        log(f"h: {k} day0 {g['mean'].iloc[0]:.2f}, day20-49 means {g['mean'].loc[20:].min():.2f}-{g['mean'].loc[20:].max():.2f}")
    return out


# ------------------------------------------------------------------------------------------------- build
def build(review):
    imgs = original_images()
    fig = plt.figure(figsize=(PW / 72, PH / 72))
    fig.patch.set_facecolor("white")
    # b, c, d kept: embedding PCA coordinates are on Chronos only
    for k in ("b", "c", "d"):
        place(fig, *imgs[k])
        if review:
            marker(fig, imgs[k][1])
    text_pt(fig, 338.91, 4.54, "Sample embedding before finetuning", 10)
    text_pt(fig, 63.65, 231.08, "Sample embedding after finetuning", 10)
    res = {"a": panel_a(fig, imgs, review), "e": panel_e(fig), "f": panel_f(fig), "g": panel_g(fig), "h": panel_h(fig)}
    for letter, x, y in (("a", 11.73, 4.62), ("b", 274.79, 4.77), ("c", 12.17, 221.76), ("d", 316.71, 218.84),
                         ("e", 12.67, 425.01), ("f", 317.21, 394.86), ("g", 13.05, 622.77), ("h", 308.42, 610.15)):
        text_pt(fig, x, y, letter, 12)
    return fig, res


def main():
    OUT.mkdir(exist_ok=True)
    for review, stem in ((False, "fig4"), (True, "fig4_review")):
        fig, _ = build(review)
        try:
            fig.savefig(OUT / f"{stem}.pdf")
            fig.savefig(OUT / f"{stem}.png", dpi=300)
        finally:
            plt.close(fig)
        log(f"wrote {OUT / stem}.pdf/.png")
    # side by side: original (left) and revised review copy (right), same height
    pix = fitz.open(ORIG)[0].get_pixmap(dpi=300)     # original page rendered on the same A4 frame
    left = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    right = Image.open(OUT / "fig4_review.png").convert("RGB")
    h = right.height
    left = left.resize((round(left.width * h / left.height), h), Image.LANCZOS)
    gap = 60
    sbs = Image.new("RGB", (left.width + gap + right.width, h), "white")
    sbs.paste(left, (0, 0))
    sbs.paste(right, (left.width + gap, 0))
    sbs.save(OUT / "fig4_side_by_side.png")
    log(f"wrote {OUT / 'fig4_side_by_side.png'}")


if __name__ == "__main__":
    main()
