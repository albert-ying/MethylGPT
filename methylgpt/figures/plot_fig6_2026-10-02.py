"""Figure 6: incident disease in Generation Scotland (9,866 development subjects, locked folds).

Values are read from ../artifacts/gs_cox (aggregate tables only; no participant data):
  full_coverage_cindex_table_2026-10-01.csv             C-index per endpoint and model
  stacked_2026-10-04.json                               C-index of the stacked model (weights learned in inner folds)
  paired_bootstrap_mgpt2_combined_2026-10-01.csv        1,000-draw paired differences
  scenario_summary_combined_2026-10-01.csv              C-index under missing-CpG scenarios
  scenario_raw_factor_imputed_2026-10-09.json           dense raw-CpG fit with factor-model imputation
  bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09_finite_p_2026-10-09.csv    scenario wins/losses (2,000 draws)
Run: MPLBACKEND=Agg python plot_fig6_2026-10-02.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain,
                                 save_figure, ref_line, pad_categorical_axis, add_colorbar, allow_dead_margin, ERRORBAR,
                                 RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_DIVERGE_CRISP)

A = Path(__file__).resolve().parent.parent / "artifacts/gs_cox"
retro_style()
EP = [("date_of_death", "Death (575)"), ("Respiratory_Diseases", "Respiratory (463)"),
      ("Cardiovascular_Diseases", "Cardiovascular (687)"), ("Endocrine_and_Metabolic_Diseases", "Type 2 diabetes (253)"),
      ("Autoimmune_Diseases", "Autoimmune (247)"), ("Cancers", "Cancers (611)"), ("Kidney_Diseases", "Chronic kidney disease (135)"),
      ("Neurological_Diseases", "Neurological (139)")]
MODELS = [("Age+sex", "Age and sex", GRAY["300"], "o"), ("Raw CpG, locked", "Raw CpGs (converged fit)", BLUE["300"], "s"),
          ("Raw CpG, GPU", "Raw CpGs (dense fit)", BLUE["500"], "s"),
          ("GrimAge2 + DunedinPACE", "GrimAge2 + DunedinPACE", GOLD["400"], "D"),
          ("MGPT-2 combined", "MethylGPT, equal weights", RUST["400"], "o"),
          ("MGPT-2 stacked", "MethylGPT, learned weights", RUST["600"], "^")]


def main():
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.96), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.05)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.80, 0.62])
    r1 = outer[0].subgridspec(1, 2, width_ratios=[1.25, 1.0])
    r2 = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1.0])
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1]), "c": fig.add_subplot(r2[0]),
           "d": fig.add_subplot(r2[1]), "e": fig.add_subplot(outer[2])}

    # a: C-index per endpoint, full coverage
    t = pd.read_csv(A / "full_coverage_cindex_table_2026-10-01.csv", index_col=0)
    st = json.loads((A / "stacked_2026-10-04.json").read_text())["endpoints"]
    t["MGPT-2 stacked"] = [st[e]["c_index"]["stacked"] for e in t.index]
    ax = axd["a"]
    yy = np.arange(len(EP))[::-1]
    offs = np.linspace(-0.28, 0.28, len(MODELS))
    for (col, lab, c, mk), off in zip(MODELS, offs):
        ax.scatter(t.loc[[e for e, _ in EP], col], yy + off, s=13, marker=mk, color=c, label=lab, edgecolors="none",
                   clip_on=False, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in EP])
    ax.set_xlabel("C-index (pooled within folds)")
    # legend in a band below the last endpoint row, clear of every data point
    ax.legend(title="Model", loc="lower left", ncol=2, frameon=False, fontsize=7, title_fontsize=7, columnspacing=1.0,
              handletextpad=0.3, borderaxespad=0.1, alignment="left")
    ax._ylim = (-2.9, 7.6)

    # b: paired differences, MethylGPT combined minus baseline (1,000 draws)
    pb = pd.read_csv(A / "paired_bootstrap_mgpt2_combined_2026-10-01_finite_p_2026-10-09.csv")
    ax = axd["b"]
    comps = [("raw_cpg_v12", "Raw CpGs (converged fit)", BLUE["300"]), ("raw_cpg_gpu", "Raw CpGs (dense fit)", BLUE["500"]),
             ("grimage2_dunedinpace", "GrimAge2 + DunedinPACE", GOLD["400"])]
    for (key, lab, c), off in zip(comps, [0.22, 0.0, -0.22]):
        sub = pb[pb.comparator == key].set_index("endpoint").loc[[e for e, _ in EP]]
        ax.errorbar(sub.difference, yy + off, xerr=[sub.difference - sub.ci95_low, sub.ci95_high - sub.difference],
                    fmt="o", ms=3, color=c, ecolor=c, label=lab, **ERRORBAR["forest"])
    ax.set_yticks(yy); ax.set_yticklabels([])
    ax.set_xlabel("MethylGPT minus comparator, C-index (95% CI)")

    # c: mean C-index under missing-CpG scenarios
    sc = pd.read_csv(A / "scenario_summary_combined_2026-10-01.csv")
    order = ["full", "random_30", "random_50", "random_70", "random_90"]
    m = sc.groupby("scenario").mean(numeric_only=True)
    ax = axd["c"]
    keys = [("raw_cpg_v12", "Raw CpGs (converged fit)", BLUE["300"], "s"), ("raw_cpg_gpu", "Raw CpGs (dense fit)", BLUE["500"], "s"),
            ("grimage2_dunedinpace", "GrimAge2 + DunedinPACE", GOLD["400"], "D"), ("mgpt2_combined", "MethylGPT", RUST["400"], "o")]
    removed = [0, 30, 50, 70, 90]
    X27 = 105   # slot for the 27K-array scenario (open symbols)
    for k, lab, c, mk in keys:
        ax.plot(removed, [m.loc[s, k] for s in order], "-", marker=mk, ms=3, lw=1.0, color=c, label=lab, clip_on=False)
    for s, lab in (("hm27", "27K"),):
        for k, _, c, mk in keys:
            ax.scatter([X27], [m.loc[s, k]],
                       marker=mk, s=16, facecolors="white", edgecolors=c, linewidths=0.8, zorder=4, clip_on=False)
    # dense raw-CpG fit with missing CpGs filled by MethylGPT's factor model (scenario_gs_raw_factor_imputed_2026-10-09)
    fi = {r["scenario"]: r["raw_cpg_gpu_factor_imputed"] for r in json.loads((A / "scenario_raw_factor_imputed_2026-10-09_2000draws.json").read_text())["mean_over_endpoints"]}
    ax.plot(removed, [fi[s_] for s_ in order], "--", marker="s", ms=3, lw=1.0, color=BLUE["500"], markerfacecolor="white", clip_on=False, zorder=3)
    ax.scatter([X27], [fi["hm27"]], marker="s", s=16, facecolors=BLUE["100"], edgecolors=BLUE["500"], linewidths=0.8, zorder=5, clip_on=False)
    ax.text(2, m.loc["random_90", "raw_cpg_v12"] + 0.0005, "dashed: dense fit with\nMethylGPT imputation", fontsize=6, ha="left", va="bottom", color=INK)
    ax.set_xticks(removed + [X27]); ax.set_xticklabels([str(r_) for r_ in removed] + ["27K"])
    ax.axvline(97.5, color=GRAY["200"], lw=0.6, zorder=0)
    ax.set_xlabel("CpGs removed (%)"); ax.set_ylabel("Mean C-index\n(CpGs removed)")

    # d: significant wins / losses per scenario (summary of the scenario bootstrap)
    # significance = Benjamini-Hochberg-adjusted P < 0.05 across all scenario x endpoint x comparator contrasts
    bsc = pd.read_csv(A / "bootstrap_scenarios_mgpt2_combined_2000draws_2026-10-09_finite_p_2026-10-09.csv")   # 2,000 draws, add-one P, BH over 384
    bs = {s: {c: {"better": int(((g.q_bh < 0.05) & (g.difference > 0)).sum()), "worse": int(((g.q_bh < 0.05) & (g.difference < 0)).sum())}
              for c, g in gs.groupby("comparator")} for s, gs in bsc.groupby("scenario")}
    ax = axd["d"]
    # complete-data contrasts are shown in b (1,000 draws); d shows the missing-CpG scenarios only
    scen = [("hm450", "450K"), ("epic_v2", "EPIC v2"), ("random_30", "30% removed"),
            ("random_50", "50% removed"), ("random_70", "70% removed"), ("hm27", "27K"), ("random_90", "90% removed")]
    cols = [("raw_cpg_v12", "Raw\n(converged)"), ("raw_cpg_gpu", "Raw\n(dense)"), ("grimage2", "Grim-\nAge2"), ("dunedinpace", "Dun.\nPACE"),
            ("grimage2_dunedinpace", "Both\nclocks")]
    M = np.array([[bs[s][c]["better"] - bs[s][c]["worse"] for c, _ in cols] for s, _ in scen], float)
    im = ax.imshow(M, cmap=CMAP_DIVERGE_CRISP, vmin=-8, vmax=8, aspect="auto")
    for i, (s, _) in enumerate(scen):
        for j, (c, _) in enumerate(cols):
            b, w = bs[s][c]["better"], bs[s][c]["worse"]
            ax.text(j, i, f"{b}/{w}", ha="center", va="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols])
    ax.set_yticks(range(len(scen))); ax.set_yticklabels([l for _, l in scen])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Endpoints better / worse", fontsize=7, fontweight="normal")
    add_colorbar(im, ax, label="Net", ticks=[-8, 0, 8])


    # e: the eight primary contrasts of the test-subset plan (filled) and the same contrasts in the validation subset (open)
    T = A.parent / "gs_test_2026-10-05"
    lab = {("complete", "mean"): "Complete profiles, mean", ("complete", "Endocrine_and_Metabolic_Diseases"): "Complete profiles, type 2 diabetes",
           ("hm27", "mean"): "27K-array CpGs, mean", ("random_90", "mean"): "90% of CpGs removed, mean"}
    cmp_ = {"raw_cpg_dense": ("Raw CpGs (dense fit)", BLUE["500"]), "grimage2_dunedinpace": ("GrimAge2 + DunedinPACE", GOLD["400"])}
    ax = axd["e"]
    prim = {n: json.loads((T / f"{f}_finite_p_2026-10-09.json").read_text())["primary"] for n, f in (("test", "confirmation"), ("validation", "validation_subset"))}
    rows = [(r["scenario"], r["endpoint"], r["comparator"]) for r in prim["test"]]
    ye = np.arange(len(rows))[::-1]
    for n, off, filled in (("test", 0.17, True), ("validation", -0.17, False)):
        by = {(r["scenario"], r["endpoint"], r["comparator"]): r for r in prim[n]}
        for y_, key in zip(ye, rows):
            r, c = by[key], cmp_[key[2]][1]
            ax.errorbar(r["difference"], y_ + off, xerr=[[r["difference"] - r["ci95"][0]], [r["ci95"][1] - r["difference"]]], fmt="o", ms=3.2,
                        color=c, ecolor=c, mfc=c if filled else "white", mew=0.8, **ERRORBAR["forest"])
    ax.set_yticks(ye); ax.set_yticklabels([f"{lab[(s_, e_)]}: vs {cmp_[c_][0]}" for s_, e_, c_ in rows])
    ax.set_xlabel("Held-out difference in C-index, MethylGPT minus comparator (95% CI)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, color=GRAY["500"], label="Test subset"),
                       Line2D([], [], marker="o", ls="", ms=3.2, color=GRAY["500"], mfc="white", mew=0.8, label="Validation subset")],
              loc="lower right", frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.2)

    trim_axes(fig, exclude=[axd["d"]])
    ref_line(axd["e"], x=0)
    axd["e"].axvline(-0.01, color=GRAY["400"], lw=0.55, ls=(0, (1, 2)), zorder=0)
    pad_categorical_axis(axd["e"], "y", pad=0.6)
    axd["c"].set_yticks([0.76, 0.77, 0.78, 0.79])
    set_domain(axd["c"], ylim=(0.755, 0.79))
    ref_line(axd["b"], x=0)
    pad_categorical_axis(axd["a"], "y", pad=0.5)
    axd["a"].set_ylim(*axd["a"]._ylim)   # keep the legend band below the endpoint rows
    pad_categorical_axis(axd["b"], "y", pad=0.5)
    set_domain(axd["b"], ylim=axd["a"].get_ylim())
    add_panel_labels(fig, axd)
    fig.savefig(str(Path(__file__).with_name("figure6_2026-10-02_draft.png")), dpi=150)
    save_figure(fig, str(Path(__file__).with_name("figure6_2026-10-02.pdf")), axes_dict=axd,
                review_png=str(Path(__file__).with_name("figure6_2026-10-02_review.png")),
                footer="data: Generation Scotland development subjects (aggregate C-indices); artifacts/gs_cox")
    plt.close(fig)


if __name__ == "__main__":
    main()
