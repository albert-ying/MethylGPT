"""Supplementary Figs 22 to 25: disease prediction from reduced and noisy measurements, EWAS at unmeasured
and noisy CpGs, individual diseases, clock agreement between arrays and classification from few labels.

Every plotted value is a row of figures/si_large_gain_tables_2026-11-10/*.csv (built from artifacts/large_gain_*;
see that folder's README and source columns). Noise and missingness are simulated on measured profiles.
Run: MPLBACKEND=Agg python plot_si_large_gain_2026-11-10.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, ref_line,
                                 save_figure, set_domain, ERRORBAR, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, TYPE)

HERE = Path(__file__).resolve().parent
T = HERE / "si_large_gain_tables_2026-11-10"
OUT = HERE
retro_style()

KEY = {"MethylGPT": RUST["300"], "MethylGPT with read-depth weighting": RUST["500"], "GrimAge2 (mean filling)": BLUE["300"],
       "DunedinPACE (mean filling)": SAGE["300"], "Raw CpGs (dense fit, mean filling)": GRAY["300"],
       "Nearest-neighbour denoising": GOLD["300"], "Noisy values": GRAY["300"]}
MK = {"MethylGPT": "o", "MethylGPT with read-depth weighting": "D", "GrimAge2 (mean filling)": "s",
      "DunedinPACE (mean filling)": "^", "Raw CpGs (dense fit, mean filling)": "v", "Nearest-neighbour denoising": "s",
      "Noisy values": "v"}
SHORT = {"Complete profiles": "Complete", "Gaussian noise SD 0.02": "SD 0.02", "Gaussian noise SD 0.05": "SD 0.05",
         "Gaussian noise SD 0.10": "SD 0.10", "Mean read depth 30": "Depth 30", "Mean read depth 10": "Depth 10",
         "Mean read depth 5": "Depth 5", "Mean read depth 2": "Depth 2", "Mean read depth 1": "Depth 1"}


def legend_cell(axl, src, title, loc="center left"):
    """Park a panel's legend in its own blank cell so that it cannot sit on data."""
    h, l = src.get_legend_handles_labels()
    axl.set_axis_off()
    leg = axl.legend(h, l, title=title, loc=loc, frameon=False, fontsize=TYPE["legend"], title_fontsize=TYPE["legend"],
                     handlelength=1.4, handletextpad=0.5, labelspacing=0.45, borderaxespad=0, alignment="left")
    return leg


def row_legend(fig, src, title, ncol):
    h, l = src.get_legend_handles_labels()
    return fig.legend(h, l, title=title, loc="outside lower center", ncol=ncol, frameon=False, fontsize=TYPE["legend"],
                      title_fontsize=TYPE["legend"], handlelength=1.4, handletextpad=0.5, columnspacing=1.2, borderaxespad=0.0, borderpad=0.1)


def read(name):
    p = T / name
    if not p.exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")
    return pd.read_csv(p)


def lines_by_setting(ax, d, metric, order, labels, legend_title, ylabel, bars=True):
    d = d[d.metric == metric]
    xs = {m: i for i, m in enumerate(order)}
    for lab in labels:
        s = d[(d.label == lab) & d.measurement_label.isin(order)].copy()
        s["x"] = s.measurement_label.map(xs)
        s = s.sort_values("x")
        if s.empty:
            continue
        ax.plot(s.x, s.value, color=KEY[lab], lw=1.0, marker=MK[lab], ms=3.6, mec=INK, mew=0.35, label=lab.replace(" (mean filling)", ""),
                clip_on=False, zorder=3 if lab.startswith("MethylGPT") else 2)
        if bars and s.ci95_low.notna().any():
            ax.errorbar(s.x, s.value, yerr=[s.value - s.ci95_low, s.ci95_high - s.value], fmt="none", ecolor=KEY[lab],
                        **ERRORBAR["point"])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[m] for m in order])
    ax.set_ylabel(ylabel)
    return xs


def fig22():
    a = read("sf22a_reduced_inputs.csv"); b = read("sf22b_noise_cindex.csv"); c = read("sf22c_noise_hazard_ratio.csv")
    fig, axd = plt.subplot_mosaic([["a", "a"], ["b", "c"]], figsize=figsize_nature("double", aspect=0.66), layout="constrained",
                                  gridspec_kw=dict(height_ratios=[1.0, 0.9]))
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.03)
    ax = axd["a"]
    g = a[a.metric == "Mean C-index gain above age and sex"]
    order = list(dict.fromkeys(g.measurement_label))
    lab_short = {"Random 10% of CpGs retained": "Random 10% of CpGs", "Random 5% of CpGs retained": "Random 5% of CpGs",
                 "RRBS whole-blood coverage (GSE194154)": "RRBS whole-blood coverage", "RRBS plasma coverage (GSE282512)": "RRBS plasma coverage",
                 "WGBS cord-blood coverage (GSE140730)": "WGBS cord-blood coverage"}
    yt = []
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        r = g[g.measurement_label == m]
        mg = r[r.label == "MethylGPT"].iloc[0]; cp = r[r.label != "MethylGPT"].iloc[0]
        ax.errorbar(cp.value, y - 0.16, xerr=[[cp.value - cp.ci95_low], [cp.ci95_high - cp.value]], fmt="none", ecolor=INK, **ERRORBAR["forest"])
        ax.plot(cp.value, y - 0.16, marker="s", ms=4.2, mfc=GRAY["200"], mec=INK, mew=0.4, ls="none", clip_on=False)
        ax.errorbar(mg.value, y + 0.16, xerr=[[mg.value - mg.ci95_low], [mg.ci95_high - mg.value]], fmt="none", ecolor=INK, **ERRORBAR["forest"])
        ax.plot(mg.value, y + 0.16, marker="o", ms=4.6, mfc=RUST["300"], mec=INK, mew=0.4, ls="none", clip_on=False)
        comp = cp.label.replace(" (dense fit, mean filling)", "").replace("nearest-neighbour filling", "NN filling")
        ax.text(0.0385, y - 0.16, comp, fontsize=TYPE["annotation"], va="center", ha="right", color=INK)
        yt.append(lab_short.get(m, m))
    ax.set_yticks(range(len(order) - 1, -1, -1)); ax.set_yticklabels(yt)
    ax.set_xlabel("C-index gain above age and sex")
    ax.set_title("Reduced inputs (circles, MethylGPT; squares, best single comparator)")
    order_n = list(SHORT)
    labs = ["MethylGPT", "MethylGPT with read-depth weighting", "GrimAge2 (mean filling)", "DunedinPACE (mean filling)",
            "Raw CpGs (dense fit, mean filling)"]
    lines_by_setting(axd["b"], b, "Mean C-index gain above age and sex", order_n, labs, "Model", "C-index gain above\nage and sex")
    axd["b"].set_title("Simulated noise, concordance")
    row_legend(fig, axd["b"], "Model in b and c", 5)
    lines_by_setting(axd["c"], c, "Mean log hazard ratio per SD", order_n, labs, "Model", "Log hazard ratio per SD\nbeyond age and sex")
    axd["c"].set_title("Simulated noise, risk separation")
    trim_axes(fig)
    set_domain(axd["a"], xlim=(0, 0.04), ylim=(-0.6, len(order) - 0.4))
    for k in "bc":
        plt.setp(axd[k].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ref_line(axd[k], y=0)
    add_panel_labels(fig, axd)
    save_figure(fig, OUT / "supplementary_reduced_noisy_2026-11-10.pdf", axes_dict=axd, review_png=OUT / "supplementary_reduced_noisy_2026-11-10.png")


def fig23():
    a = read("sf23a_ewas_unmeasured.csv"); b = read("sf23b_ewas_noise.csv")
    fig, axd = plt.subplot_mosaic([["a", "b"]], figsize=figsize_nature("double", aspect=0.38), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[1.06, 1.1]))
    fig.get_layout_engine().set(w_pad=0.04)
    ax = axd["a"]
    g = a[a.metric.str.startswith("Spearman")]
    eps = [e for e in dict.fromkeys(g.endpoint_label) if not e.startswith("Mean over seven")]
    eps = sorted([e for e in eps if not e.startswith("Mean")], key=lambda e: g[(g.endpoint_label == e) & (g.label == "MethylGPT imputation")].value.iloc[0]) + ["Mean over eight endpoints"]
    for i, e in enumerate(eps):
        m = g[(g.endpoint_label == e) & (g.label == "MethylGPT imputation")].value.iloc[0]
        n = g[(g.endpoint_label == e) & (g.label == "Nearest-neighbour imputation")].value.iloc[0]
        y = i if not e.startswith("Mean") else i + 0.5
        ax.plot([n, m], [y, y], color=GRAY["300"], lw=0.75, zorder=1)
        ax.plot(n, y, marker="s", ms=4.2, mfc=GOLD["200"], mec=INK, mew=0.4, ls="none", clip_on=False, label="Nearest-neighbour imputation" if i == 0 else None)
        ax.plot(m, y, marker="o", ms=4.6, mfc=RUST["300"], mec=INK, mew=0.4, ls="none", clip_on=False, label="MethylGPT imputation" if i == 0 else None)
    ax.set_yticks([i if not e.startswith("Mean") else i + 0.5 for i, e in enumerate(eps)])
    ax.set_yticklabels([e.replace("Mean over eight endpoints", "Mean of eight endpoints") for e in eps])
    ax.set_xlabel("Spearman correlation with full-data association statistics")
    ax.set_title("22,322 CpGs absent from the 27K array")
    ax.text(0.0, 0.93, "squares, nearest-neighbour imputation\ncircles, MethylGPT imputation", transform=ax.transAxes, fontsize=TYPE["annotation"], ha="left", va="top", color=INK)
    order = ["Gaussian noise SD 0.05", "Mean read depth 30", "Mean read depth 10", "Mean read depth 5", "Mean read depth 2"]
    labs = ["MethylGPT", "MethylGPT with read-depth weighting", "Nearest-neighbour denoising", "Noisy values"]
    lines_by_setting(axd["b"], b, "Mean Spearman correlation of per-CpG Cox z with full-data z", order, labs, "Values used",
                     "Mean Spearman correlation\nover eight endpoints", bars=False)
    axd["b"].set_title("Simulated noise")
    row_legend(fig, axd["b"], "Values used (b)", 4)
    trim_axes(fig)
    set_domain(axd["a"], ylim=(-0.6, len(eps)))
    plt.setp(axd["b"].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    add_panel_labels(fig, axd)
    save_figure(fig, OUT / "supplementary_ewas_2026-11-10.pdf", axes_dict=axd, review_png=OUT / "supplementary_ewas_2026-11-10.png")


def fig24():
    d = read("sf24_individual_diseases.csv")
    one = d[d.metric == "Difference in C-index"]
    dd = one[one.label.str.contains("DunedinPACE")].sort_values("value")
    order = list(dd.disease_label)
    fig, ax = plt.subplots(figsize=figsize_nature("double", aspect=0.48), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.004, h_pad=0.004)
    axd = {"a": ax}
    y = np.arange(len(order))
    ev = one[one.label.str.contains("DunedinPACE minus")].set_index("disease_label").loc[order].n_events
    for key, col, dy in (("DunedinPACE", SAGE, 0.17), ("GrimAge2", BLUE, -0.17)):
        s_ = one[one.label.str.contains(key + " minus")].set_index("disease_label").loc[order]
        ax.errorbar(s_.value, y + dy, xerr=[s_.value - s_.ci95_low, s_.ci95_high - s_.value], fmt="none", ecolor=INK, **ERRORBAR["forest"])
        sig = (s_.q_value < 0.05).values
        ax.scatter(s_.value[~sig], (y + dy)[~sig], s=18, facecolors="white", edgecolors=col["400"], linewidths=0.8, zorder=3, clip_on=False)
        ax.scatter(s_.value[sig], (y + dy)[sig], s=18, facecolors=col["300"], edgecolors=INK, linewidths=0.4, zorder=3, clip_on=False,
                   label=f"Added to {key}")
        m = d[(d.metric.str.startswith("Mean difference")) & d.label.str.contains(key + " minus")].iloc[0]
        ax.errorbar(m.value, -1.4 + dy, xerr=[[m.value - m.ci95_low], [m.ci95_high - m.value]], fmt="none", ecolor=INK, **ERRORBAR["forest"])
        ax.scatter([m.value], [-1.4 + dy], s=24, marker="D", facecolors=col["400"], edgecolors=INK, linewidths=0.4, zorder=3, clip_on=False)
    ax.set_yticks(list(y) + [-1.4])
    ax.set_yticklabels([f"{o.replace('Chronic obstructive pulmonary disease', 'COPD')} ($n$ = {int(e):,})" for o, e in zip(order, ev)] + ["Mean of 13 diseases"])
    ax.set_xlabel("Difference in C-index, MethylGPT + clock minus the clock alone")
    ax.set_title("MethylGPT added to a clock, complete profiles ($n$ = 9,866 participants)")
    nature_legend(ax, title="Filled, $q$ < 0.05", loc="upper left")
    trim_axes(fig)
    set_domain(ax, ylim=(-2.1, len(order) - 0.3))
    ref_line(ax, x=0)
    add_panel_labels(fig, axd) if False else None
    save_figure(fig, OUT / "supplementary_individual_diseases_2026-11-10.pdf", axes_dict=axd,
                review_png=OUT / "supplementary_individual_diseases_2026-11-10.png")


def fig25():
    a = read("sf25a_array_agreement.csv"); b = read("sf25b_few_labels_sparse.csv")
    fig, axd = plt.subplot_mosaic([["a", "a"], ["b", "c"]], figsize=figsize_nature("double", aspect=0.72), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[1.35, 1.0], height_ratios=[0.8, 1.0]))
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.1)
    ax = axd["a"]
    g = a[(a.metric == "Mean absolute between-array difference on own SD")]
    qs = ["Primary mean of GrimAge2 and DunedinPACE", "GrimAge2", "DunedinPACE", "PhenoAge", "Horvath age"]
    qlab = ["GrimAge2 and\nDunedinPACE", "GrimAge2", "DunedinPACE", "PhenoAge", "Horvath age"]
    for i, q in enumerate(qs):
        y = len(qs) - 1 - i
        for lab, col, mk, dy in (("Raw clocks (reference filling)", GRAY["200"], "s", -0.14), ("MethylGPT", RUST["300"], "o", 0.14)):
            ind = g[(g.quantity_label == q) & (g.label == lab) & (g.aggregation_label == "Individual comparison")]
            if len(ind):
                ax.scatter(ind.value, np.full(len(ind), y + dy), s=9, facecolors=col, edgecolors="none", alpha=0.75, zorder=2, clip_on=False)
            al = g[(g.quantity_label == q) & (g.label == lab) & (g.comparison_label == "All six comparisons")]
            if len(al):
                ax.plot(al.value.iloc[0], y + dy, marker=mk, ms=5, mfc=col, mec=INK, mew=0.45, ls="none", zorder=3, clip_on=False,
                        label=("MethylGPT distilled output" if lab == "MethylGPT" else "Clock with reference filling") if i == 0 else None)
    ax.set_yticks(range(len(qs) - 1, -1, -1)); ax.set_yticklabels(qlab)
    ax.set_xlabel("Mean absolute difference between arrays (s.d.)")
    ax.set_title("Two arrays, same people ($n$ = 354 pairs)")
    ax = axd["b"]
    g = b[(b.metric == "Mean AUC minus 0.5") & (b.n_train == 50)]
    cols = {"MethylGPT-imputed CpGs": (RUST["300"], "o"), "Nearest-neighbour imputation": (GOLD["300"], "s"),
            "Raw CpGs (mean imputation)": (GRAY["300"], "v"), "Clocks": (BLUE["300"], "^")}
    for lab, (col, mk) in cols.items():
        s = g[g.label == lab].sort_values("keep_fraction")
        ax.plot(range(len(s)), s.value + 0.5, color=col, lw=1.0, marker=mk, ms=3.8, mec=INK, mew=0.35, label=lab, clip_on=False)
        ax.errorbar(range(len(s)), s.value + 0.5, yerr=[s.value - s.ci95_low, s.ci95_high - s.value], fmt="none", ecolor=col, **ERRORBAR["point"])
        kf = [f"{int(100 * k)}%" for k in s.keep_fraction]
    ax.set_xticks(range(len(kf))); ax.set_xticklabels(kf)
    ax.set_xlabel("CpGs kept per sample"); ax.set_ylabel("Mean AUC, 50 labelled samples")
    ax.set_title("Disease classification")
    row_legend(fig, ax, "Classifier input (b)", 4)
    ax = axd["c"]
    p = b[(b.metric == "Per-dataset mean AUC")]
    pv = p.pivot_table(index="dataset_label", columns="label", values="value")
    ax.scatter(pv["Nearest-neighbour imputation"], pv["MethylGPT-imputed CpGs"], s=16, facecolors=RUST["300"], edgecolors=INK, linewidths=0.4, clip_on=False, zorder=3)
    ax.set_xlabel("AUC, nearest-neighbour imputation"); ax.set_ylabel("AUC, MethylGPT-imputed CpGs")
    ax.set_title(f"$n$ = {len(pv)} datasets, 5% of CpGs")
    lo = np.floor(min(pv.min().min(), 0.5) * 20) / 20
    trim_axes(fig)
    set_domain(axd["a"], ylim=(-0.9, len(qs) - 0.5))
    nature_legend(axd["a"], title="Method")   # marker key; small and large symbols are defined in the SI legend
    set_domain(axd["c"], xlim=(lo, 1.02), ylim=(lo, 1.02))
    axd["c"].plot([lo, 1], [lo, 1], color=GRAY["400"], lw=0.55, ls=(0, (2, 2)), zorder=1)
    add_panel_labels(fig, axd)
    save_figure(fig, OUT / "supplementary_arrays_few_labels_2026-11-10.pdf", axes_dict=axd,
                review_png=OUT / "supplementary_arrays_few_labels_2026-11-10.png")
    return int((pv["MethylGPT-imputed CpGs"] > pv["Nearest-neighbour imputation"]).sum()), len(pv)


if __name__ == "__main__":
    for f in (fig22, fig23, fig24, fig25):
        try:
            r = f()
            print(f.__name__, "saved", r if r is not None else "")
        except Exception as e:      # report every figure's gate result in one run
            print(f.__name__, "NOT SAVED:", str(e)[:1500])
