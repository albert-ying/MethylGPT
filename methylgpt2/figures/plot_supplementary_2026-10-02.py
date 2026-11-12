"""Supplementary Figures 1-5 for the v13 revision (aggregate result files only).

SF1 reconstruction error by held-out chromosome and gene fold (eval/heldout_structured_2026-10-01.json)
SF2 GSE308517 mean absolute error by masking rate (eval/gse308517_*.json)
SF3 factor-model reconstruction of endocrine and smoking marker CpGs in Generation Scotland
    (gs_features_2026-10-01/extraction_record_{allcore,blood}_core.json)
SF4 label efficiency by endpoint (gs_cox/scenario_label_efficiency_summary_2026-10-01.csv)
SF5 C-index by endpoint in missing-CpG scenarios (gs_cox/scenario_summary_combined_2026-10-01.csv)
sf6 pretraining control (eval/pretraining_control_2026-10-03.json)
Manuscript numbering follows first citation: sf1 -> SF1, sf2 -> SF2, sf6 -> SF3, sf3 -> SF4, sf5 -> SF5,
sf4 -> SF6 (mapping applied by export_manuscript_pngs_2026-10-02.py).
Run: MPLBACKEND=Agg python plot_supplementary_2026-10-02.py [sf1 ... sf6]
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, add_panel_labels, nature_legend, save_figure, trim_axes,
                                 pad_categorical_axis, set_domain, allow_dead_margin, ref_line, label_panel_group,
                                 add_colorbar, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_DIVERGE_CRISP)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
M = {"methylgpt2": ("MethylGPT", RUST["400"], "o"), "knn10": ("Nearest neighbours", BLUE["400"], "D"), "cpg_mean": ("CpG means", GRAY["400"], "s")}
EP = [("date_of_death", "Death"), ("Respiratory_Diseases", "Respiratory"), ("Cardiovascular_Diseases", "Cardiovascular"),
      ("Endocrine_and_Metabolic_Diseases", "Type 2 diabetes"), ("Autoimmune_Diseases", "Autoimmune"), ("Cancers", "Cancers"),
      ("Kidney_Diseases", "Chronic kidney disease"), ("Neurological_Diseases", "Neurological")]


def save(fig, axd, name, footer, after_trim=None, group=None):
    trim_axes(fig, exclude=[ax for k, ax in axd.items() if getattr(ax, "_no_trim", False)])
    if after_trim:
        after_trim()
    if group:
        label_panel_group(fig, list(axd.values()), group)
    else:
        add_panel_labels(fig, axd)
    fig.savefig(str(HERE / f"{name}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{name}.pdf"), axes_dict=axd, review_png=str(HERE / f"{name}_review.png"), footer=footer)
    plt.close(fig)


def sf1():
    st = json.loads((A / "eval/heldout_structured_2026-10-01.json").read_text())["mask_types"]
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.30), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.6])
    axd = {"a": fig.add_subplot(gs[0]), "b": fig.add_subplot(gs[1])}
    for key, panel in (("chromosome", "a"), ("gene_fold", "b")):
        by = st[key]["by_group"]
        keys = list(by)
        if key == "chromosome":   # numeric order, X last
            keys = sorted(keys, key=lambda k: (k == "X", int(k) if k.isdigit() else 99))
        xx = np.arange(len(keys))
        ax = axd[panel]
        for m, off in (("cpg_mean", -0.2), ("knn10", 0.0), ("methylgpt2", 0.2)):
            lab, c, mk = M[m]
            ax.scatter(xx + off, [by[k][m]["mae"] for k in keys], s=12, marker=mk, color=c, label=lab, edgecolors="none", clip_on=False)
        ax.set_xticks(xx); ax.set_xticklabels([k.replace("fold", "Fold ") for k in keys])
        ax.set_yscale("log"); ax.set_ylabel("Mean absolute error")
        ax.set_xlabel("Held-out chromosome" if key == "chromosome" else "Held-out gene fold")

    def pads():
        pad_categorical_axis(axd["a"], "x", pad=0.6); pad_categorical_axis(axd["b"], "x", pad=0.6)
        for ax in axd.values():
            ax.set_yticks([0.03, 0.1, 0.3]); ax.set_yticklabels(["0.03", "0.1", "0.3"]); set_domain(ax, ylim=(0.025, 0.3))
        nature_legend(axd["a"], title="Method", loc="upper left", ncol=3)
    save(fig, axd, "supplementary_figure1_2026-10-02", "data: 2,000 profiles from 107 test datasets; eval/heldout_structured_2026-10-01.json", pads)


def sf2():
    g = json.loads((A / "eval/gse308517_cf_dense_step2000_2026-09-30.json").read_text())["seed_averaged_mae_by_rate"]
    ml = json.loads((A / "eval/gse308517_methylimp2_2026-10-01.json").read_text())["seed_averaged_mae_by_rate"]
    rates = [15, 30, 50, 70]
    fig, ax = plt.subplots(figsize=figsize_nature("1.5col", aspect=0.5), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005)
    for key, lab, c, mk in (("MLM", "Original MethylGPT", SAGE["500"], "^"), ("methylimp2", "methyLImp2", SAGE["300"], "v"),
                            ("fold_mean", "CpG means", GRAY["400"], "s"), ("donor_knn_k5", "Nearest neighbours", BLUE["400"], "D"),
                            ("methylgpt2", "MethylGPT", RUST["400"], "o"), ("methylgpt2_foldmean", "MethylGPT, fold-mean centre", RUST["600"], "o")):
        vals = [ml[str(r)]["methylimp2"] if key == "methylimp2" else g[str(r)][key] for r in rates]
        ax.plot(rates, vals, "-", marker=mk, ms=3, lw=1.0, color=c, label=lab, clip_on=False)
    ax.set_xticks(rates); ax.set_xlabel("CpGs masked (%)"); ax.set_ylabel("Mean absolute error")
    ax.set_ylim(0.015, 0.055)
    nature_legend(ax, loc="upper left", ncol=3)
    allow_dead_margin(ax, "y", reason="room for the legend")
    trim_axes(fig)
    set_domain(ax, ylim=(0.015, 0.055))
    fig.savefig(str(HERE / "supplementary_figure2_2026-10-02_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "supplementary_figure2_2026-10-02.pdf"), axes_dict={"a": ax},
                review_png=str(HERE / "supplementary_figure2_2026-10-02_review.png"),
                footer="data: GSE308517, fixed protocol (136 profiles, three seeds)")
    plt.close(fig)


def sf3():
    a = json.loads((A / "gs_features_2026-10-01/extraction_record_allcore_core.json").read_text())["marker_r2_gs"]
    b = json.loads((A / "gs_features_2026-10-01/extraction_record_blood_core.json").read_text())["marker_r2_gs"]
    keys = list(a)
    fig, ax = plt.subplots(figsize=figsize_nature("1.5col", aspect=0.5), layout="constrained")
    yy = np.arange(len(keys))[::-1]
    ax.scatter([a[k] for k in keys], yy, s=18, color=RUST["400"], label="All-tissue input layer (MethylGPT)", clip_on=False, zorder=3)
    ax.scatter([b[k] for k in keys], yy, s=18, marker="D", color=BLUE["400"], label="Whole-blood input layer", clip_on=False, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(keys)
    ax.set_xlabel("Reconstruction $R^2$ in Generation Scotland")
    nature_legend(ax, loc="upper right")
    trim_axes(fig)
    pad_categorical_axis(ax, "y", pad=0.6)
    ref_line(ax, x=0)
    fig.savefig(str(HERE / "supplementary_figure3_2026-10-02_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "supplementary_figure3_2026-10-02.pdf"), axes_dict={"a": ax},
                review_png=str(HERE / "supplementary_figure3_2026-10-02_review.png"),
                footer="data: Generation Scotland development subjects (aggregate R2 per CpG)")
    plt.close(fig)


def facet_figure(summary_draw, facet_draw, name, footer, legend_from=None, aspect=0.62, xlabel=None):
    fig = plt.figure(figsize=figsize_nature("double", aspect=aspect), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.05, w_pad=0.04)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.8, 0.8])
    top = fig.add_subplot(gs[0, :])
    facets = [fig.add_subplot(gs[1 + i // 4, i % 4]) for i in range(8)]
    summary_draw(top)
    for (ep, lab), ax in zip(EP, facets):
        facet_draw(ax, ep)
        ax.set_title(lab, fontsize=7, fontweight="normal")
    h, l = facets[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    trim_axes(fig, exclude=[top] if getattr(top, "_no_trim", False) else [])
    for ax in facets[:4]:   # facets share the x categories; label the bottom row only
        ax.tick_params(labelbottom=False)
    if xlabel:
        for ax in facets[4:]:
            ax.set_xlabel(xlabel)
    if getattr(top, "_after", None):
        top._after()
    add_panel_labels(fig, {"a": top})
    label_panel_group(fig, facets, "b")
    fig.savefig(str(HERE / f"{name}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{name}.pdf"), axes_dict={"a": top, "b": facets[0]},
                review_png=str(HERE / f"{name}_review.png"), footer=footer)
    plt.close(fig)


def sf4():
    le = pd.read_csv(A / "gs_cox/scenario_label_efficiency_summary_2026-10-01.csv")
    fr = [(0.05, "5%"), (0.1, "10%"), (0.25, "25%")]

    def summary(ax):
        xx = np.arange(len(EP))
        for (f, lab), c, off in zip(fr, (RUST["200"], RUST["300"], RUST["500"]), (-0.2, 0.0, 0.2)):
            d = le[le.fraction == f].set_index("endpoint")
            ax.scatter(xx + off, [d.loc[e, "biomarkers"] - d.loc[e, "raw_cpg"] for e, _ in EP], s=16, color=c,
                       label=f"{lab} of participants", edgecolors="none", clip_on=False, zorder=3)
        ax.set_xticks(xx); ax.set_xticklabels([l for _, l in EP])
        ax.set_ylabel("Distilled clocks minus\nraw CpG (C-index)")
        nature_legend(ax, title="Fitting set", loc="upper right", ncol=3)
        ax._after = lambda: (ref_line(ax, y=0), pad_categorical_axis(ax, "x", pad=0.5))

    def facet(ax, ep):
        d = le[le.endpoint == ep].sort_values("fraction")
        for col, name, c, mk in (("age_sex", "Age and sex", GRAY["300"], "o"), ("raw_cpg", "Raw CpGs (dense fit)", BLUE["500"], "s"),
                                 ("grimage2_dunedinpace", "GrimAge2 + DunedinPACE", GOLD["400"], "D"),
                                 ("biomarkers", "MethylGPT distilled outputs", RUST["400"], "o")):
            ax.plot(d.mean_fit_subjects, d[col], "-", marker=mk, ms=2.5, lw=0.9, color=c, label=name, clip_on=False)
        ax.set_xscale("log"); ax.set_xticks([400, 2000]); ax.set_xticklabels(["400", "2,000"])
    facet_figure(summary, facet, "supplementary_figure4_2026-10-02", "data: Generation Scotland development subjects (aggregate C-indices)",
                 xlabel="Fitting participants")


def sf5():
    sc = pd.read_csv(A / "gs_cox/scenario_summary_combined_2026-10-01.csv")
    order = ["full", "hm450", "epic_v2", "random_30", "random_50", "random_70", "hm27", "random_90"]
    labels = ["Full", "450K", "EPIC v2", "30%", "50%", "70%", "27K", "90%"]

    def summary(ax):
        D = np.array([[sc[(sc.endpoint == e) & (sc.scenario == s)].eval("mgpt2_combined - best_baseline").iloc[0] for s in order] for e, _ in EP])
        im = ax.imshow(D, cmap=CMAP_DIVERGE_CRISP, vmin=-0.05, vmax=0.05, aspect="auto")
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                ax.text(j, i, f"{D[i, j]:+.3f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
        ax.set_yticks(range(len(EP))); ax.set_yticklabels([l for _, l in EP])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("MethylGPT combined minus best comparator (C-index)", fontsize=7, fontweight="normal")
        ax._no_trim = True
        add_colorbar(im, ax, label="Difference", ticks=[-0.05, 0, 0.05])

    def facet(ax, ep):
        d = sc[sc.endpoint == ep].set_index("scenario").loc[order]
        xx = np.arange(len(order))
        for col, name, c, mk in (("raw_cpg_v12", "Raw CpGs (converged fit)", BLUE["300"], "s"), ("raw_cpg_gpu", "Raw CpGs (dense fit)", BLUE["500"], "s"),
                                 ("grimage2_dunedinpace", "GrimAge2 + DunedinPACE", GOLD["400"], "D"), ("mgpt2_combined", "MethylGPT, equal weights", RUST["400"], "o")):
            ax.plot(xx, d[col], "-", marker=mk, ms=2.5, lw=0.9, color=c, label=name, clip_on=False)
        ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
    facet_figure(summary, facet, "supplementary_figure5_2026-10-02", "data: Generation Scotland development subjects (aggregate C-indices)",
                 aspect=0.56)


def sf6():
    """Pretraining control: same stage and seed, with and without pretraining (eval/pretraining_control_2026-10-03.json)."""
    pc = json.loads((A / "eval/pretraining_control_2026-10-03.json").read_text())
    at, ho = pc["altumage_test_medae"], pc["heldout_34_datasets"]
    rw = pc["transformer_random_weights_vs_pretrained"]
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.32), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
    axd = {"a": fig.add_subplot(gs[0]), "b": fig.add_subplot(gs[1])}
    ax = axd["a"]
    rr = [("all", 0), ("random_30", 30), ("random_50", 50), ("random_70", 70), ("random_90", 90)]
    series = (("Pretrained", RUST["400"], "o", lambda s: at[s]["pretrained"]),
              ("Randomly initialized", PLUM["400"], "s", lambda s: at[s]["no_pretraining"]))
    for lab, c, mk, val in series:
        ax.plot([x for _, x in rr], [val(s) for s, _ in rr], "-", marker=mk, ms=3.5, lw=1.0, color=c, label=lab, clip_on=False)
        ax.scatter([100 * (1 - 22231 / 49156)], [val("hm27")], s=22, marker=mk, facecolors="white", edgecolors=c, linewidths=0.9,
                   zorder=4, clip_on=False)
    ax.text(100 * (1 - 22231 / 49156) + 3, at["hm27"]["no_pretraining"], "27K CpGs (open)", fontsize=6, ha="left", va="center", color=INK)
    ax.set_xlabel("CpGs removed (%)"); ax.set_ylabel("MedAE, AltumAge test (years)")
    nature_legend(ax, loc="upper left")
    ax = axd["b"]
    rows = [("AltumAge test, MedAE", at["all"], rw["altumage_test_medae"]["all"]), ("Held-out, MedAE", ho["medae"], rw["heldout_34_datasets"]["medae"]),
            ("Held-out, MAE", ho["mae"], rw["heldout_34_datasets"]["mae"])]
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, v, w) in zip(yy, rows):
        ax.hlines(y_, v["pretrained"], v["no_pretraining"], color=GRAY["200"], lw=1.0)
        ax.scatter(v["no_pretraining"], y_, s=22, marker="s", color=PLUM["400"], zorder=3, clip_on=False)
        ax.scatter(v["pretrained"], y_, s=20, color=RUST["400"], zorder=4, clip_on=False)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Error (years)")

    def pads():
        pad_categorical_axis(axd["b"], "y", pad=0.5)
    save(fig, axd, "supplementary_figure6_2026-10-03", "data: AltumAge test set and 34 held-out datasets; eval/pretraining_control_2026-10-03.json", pads)


if __name__ == "__main__":
    import sys
    sel = sys.argv[1:]
    for f in (sf1, sf2, sf3, sf4, sf5, sf6):
        if sel and f.__name__ not in sel:
            continue
        try:
            f()
            print(f.__name__, "ok")
        except AssertionError as e:
            print(f.__name__, "PREFLIGHT", str(e)[:1500])
