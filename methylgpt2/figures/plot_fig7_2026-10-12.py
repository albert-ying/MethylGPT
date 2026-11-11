"""Figure 7: what pretraining contributes when labels or CpGs are scarce.

Inputs (run from this directory):
  ../artifacts/eval/zero_shot_init_2026-10-14.json                                 zero-shot reconstruction, pretrained and randomly initialized
  ../artifacts/eval/pretraining_label_efficiency_finetune_2026-10-12.csv          encoder frozen, heads trained on 100 to 10,000 labels
  ../artifacts/eval/rrbs_few_shot_adaptation_2026-10-12.{csv,json}                sequencing cohort, 10 to 88 labelled samples
  ../artifacts/eval/rrbs_few_shot_adaptation_2026-10-12.heldout_mean_predictions_n20.npz
  ../artifacts/gs_cox/few_labels_pretrained_vs_raw_2026-10-12.json                Generation Scotland, development participants
Run: MPLBACKEND=Agg python plot_fig7_2026-10-12.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, label_panel_group, nature_legend, set_domain,
                                 save_figure, RUST, GRAY, BLUE, GOLD, PLUM, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
# (method, label, colour, marker, filled, line style, label position and alignment)
RRBS = [("MethylGPT recalibrated", "MethylGPT, recalibrated", RUST["400"], "o", True, "-", (88, 5.72, "right", "bottom")),
        ("array elastic net with factor-model imputation, recalibrated", "Corpus elastic net, imputed, recalibrated", BLUE["600"], "D", False, "-", (24, 6.95, "left", "bottom")),
        ("elastic net, screened genome-wide CpGs", "Elastic net, screened CpGs", BLUE["400"], "s", False, "-", (14, 7.42, "left", "bottom")),
        ("ridge, genome-wide CpGs", "Ridge, 1.2 million CpGs", BLUE["300"], "^", False, "-", (88, 8.93, "right", "bottom")),
        ("mean age", "Mean age", GRAY["400"], None, False, (0, (3, 1.5)), (88, 9.9, "right", "bottom"))]
GSM = [("MethylGPT distilled outputs", "MethylGPT distilled outputs", RUST["400"], "o", True),
       ("GrimAge2 + DunedinPACE", "GrimAge2 + DunedinPACE", GOLD["500"], "D", False),
       ("raw CpG", "Raw CpGs", BLUE["400"], "s", False),
       ("age and sex", "Age and sex", GRAY["400"], "v", False)]


def series(ax, x, y, lo, hi, color, marker, filled, ls="-", label=None):
    ax.plot(x, y, ls=ls, lw=1.0, color=color, zorder=2, clip_on=False, label=label if marker is None else None)
    if marker is not None:
        if lo is not None:
            ax.vlines(x, lo, hi, color=color, lw=0.7, zorder=2, clip_on=False)
        ax.scatter(x, y, s=16, marker=marker, facecolors=color if filled else "white", edgecolors=color, linewidths=0.9, zorder=3, clip_on=False, label=label)


def main():
    ft = pd.read_csv(A / "eval/pretraining_label_efficiency_finetune_2026-10-12.csv")
    fr = ft[ft["mode"] == "frozen"]
    zs = json.loads((A / "eval/zero_shot_init_2026-10-14.json").read_text())["tasks"]
    rr = pd.read_csv(A / "eval/rrbs_few_shot_adaptation_2026-10-12.csv")
    sc = np.load(A / "eval/rrbs_few_shot_adaptation_2026-10-12.heldout_mean_predictions_n20.npz")
    gsp = A / "gs_cox/few_labels_pretrained_vs_raw_2026-10-12.json"
    if not gsp.exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {gsp}")
    gs = json.loads(gsp.read_text())

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.66), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.14, w_pad=0.05)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.05, 0.9, 1.1])
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1]), "c": fig.add_subplot(r1[2]),
           "d": fig.add_subplot(r2[0]), "e": fig.add_subplot(r2[1]), "f": fig.add_subplot(r2[2])}
    PRE, RND, MEAN = RUST["400"], PLUM["400"], GRAY["400"]

    # a: zero-shot reconstruction of held-out profiles (no labels, no training on these datasets)
    ax = axd["a"]
    tasks = [("random_30", "30% of CpGs\nmasked"), ("hm27_conversion", "27K-array\nCpGs only")]
    arms = [("pretrained", "Pretrained", PRE), ("random", "Randomly initialized", RND), ("training CpG means", "CpG means", MEAN)]
    # pooled MSE over all masked values (one value per arm and task); dot plot
    w = 0.26
    for k, (t, _) in enumerate(tasks):
        for j, (arm, lab, c) in enumerate(arms):
            v = zs[t][arm]["mse"] * 1e3
            ax.scatter(k + (j - 1) * w, v, s=22, marker="o", facecolors=c if arm == "pretrained" else "white", edgecolors=c,
                       linewidths=0.9, zorder=3, label=lab if k == 0 else None, clip_on=False)
            ax.text(k + (j - 1) * w + 0.05, v, f"{v:.1f}", ha="left", va="center", fontsize=6, color=INK)
    ax.set_ylim(0, 40); ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t[1] for t in tasks])
    ax.set_ylabel("Masked-value MSE (×10$^{-3}$)")
    ax.set_title("Zero-shot reconstruction, held-out datasets", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="center", bbox_to_anchor=(0.62, 0.42))

    # b, c: encoder frozen, output heads trained on 1,000-10,000 labelled profiles (100 and 300 in Supplementary Table 9)
    fr = fr[fr.n_labels >= 1000]

    def frozen_panel(ax, conds, title):
        for col, ls, filled in conds:
            for arm, c, mk in (("pretrained", PRE, "o"), ("none", RND, "D")):
                g = fr[fr.arm == arm].groupby("n_labels")[col].agg(["mean", "min", "max"])
                series(ax, g.index.to_numpy(), g["mean"].to_numpy(), g["min"].to_numpy(), g["max"].to_numpy(), c, mk, filled, ls=ls)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1000, 3000, 10000]); ax.set_xticklabels(["1,000", "3,000", "10,000"])
        ax.set_xlabel("Labelled profiles"); ax.set_title(title, fontsize=7, fontweight="normal")
    frozen_panel(axd["b"], [("altumage_test|all|medae", "-", True), ("heldout|all|medae", (0, (3, 1.5)), False)], "Encoder frozen: complete profiles")
    frozen_panel(axd["c"], [("altumage_test|random_90|medae", "-", True), ("altumage_test|hm27|medae", (0, (3, 1.5)), False)], "Encoder frozen: CpGs missing")
    axd["b"].set_ylabel("MedAE (years)")
    from matplotlib.lines import Line2D
    # colours (pretrained, randomly initialized) are keyed in panel a
    axd["b"].legend(handles=[Line2D([], [], color=INK, lw=1.0, label="AltumAge test set"),
                             Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="Held-out datasets")],
                    loc="upper right", fontsize=7, frameon=False, handlelength=2.2)
    axd["c"].legend(handles=[Line2D([], [], color=INK, lw=1.0, label="90% of CpGs removed"),
                             Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="27K-array CpGs only")],
                    loc="lower left", fontsize=7, frameon=False, handlelength=2.2)

    # d: sequencing cohort, pretrained model recalibrated against models trained on the cohort
    ax = axd["d"]
    for m, lab, c, mk, filled, ls, _ in RRBS:
        g = rr[rr.method == m].groupby("n_labelled").mae.mean()
        series(ax, g.index.to_numpy(), g.to_numpy(), None, None, c, mk, filled, ls=ls)
    ax.set_xlabel("Labelled samples of the sequencing cohort"); ax.set_ylabel("MAE, remaining samples (years)")
    ax.set_title("RRBS cohort: adapt MethylGPT or train on the cohort", fontsize=7, fontweight="normal")

    # e: each sample's mean held-out prediction with 20 labelled samples
    ax = axd["e"]
    ax.plot([15, 90], [15, 90], color=GRAY["300"], lw=0.6, zorder=0)
    for k, (a_, c, lab) in enumerate((("ridge, genome-wide CpGs", BLUE["300"], "Ridge, 1.2 million CpGs"), ("MethylGPT recalibrated", RUST["400"], "MethylGPT, recalibrated"))):
        ax.scatter(sc["age"], sc[a_], s=6, color=c, edgecolors="none", alpha=0.75, zorder=3 + k, clip_on=False)
        ax.text(0.98, 0.04 + 0.09 * k, f"{lab}, $r$ = {np.corrcoef(sc['age'], sc[a_])[0, 1]:.2f}", transform=ax.transAxes, fontsize=6, ha="right", va="bottom", color=c)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("Held-out prediction (years)")
    ax.set_title("RRBS cohort, 20 labelled samples", fontsize=7, fontweight="normal")

    # f: Generation Scotland, mean C-index over eight endpoints by number of fitting participants
    ax = axd["f"]
    tab = pd.DataFrame(gs["mean_c_by_fraction"]).set_index("fraction")
    nfit = {float(m["fraction"]): m["mean_fitting_participants_per_fold"] for m in gs["means"]}
    full = max(v for v in nfit.values() if v is not None) / 0.25 if nfit.get(1.0) is None else nfit[1.0]
    x = np.array([nfit[f] if nfit[f] is not None else full for f in tab.index])
    for m, lab, c, mk, filled in GSM:
        series(ax, x, tab[m].to_numpy(), None, None, c, mk, filled, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Participants used for fitting, per fold"); ax.set_ylabel("Mean C-index, eight endpoints")
    ax.set_title("Generation Scotland: frozen MethylGPT features", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="lower right")

    trim_axes(fig)
    axd["a"].set_xlim(-0.6, 1.75)
    set_domain(axd["a"], ylim=(0, 42))
    for k in ("b", "c"):
        axd[k].set_xticks([100, 1000, 10000]); axd[k].set_xticklabels(["100", "1,000", "10,000"])
        axd[k].set_yticks([2, 5, 10, 20]); axd[k].set_yticklabels(["2", "5", "10", "20"])
        set_domain(axd[k], xlim=(850, 12000), ylim=(1.9, 22))
        axd[k].set_xticks([1000, 3000, 10000]); axd[k].set_xticklabels(["1,000", "3,000", "10,000"])
    axd["c"].set_yticklabels([])
    axd["d"].set_xticks([10, 20, 40, 60, 88]); set_domain(axd["d"], xlim=(6, 92), ylim=(5.5, 11.2))
    for _, lab, c, _, _, _, (x_, y_, ha, va) in RRBS:
        axd["d"].text(x_, y_, lab, fontsize=6, ha=ha, va=va, color=INK if lab == "Mean age" else c)
    axd["e"].set_xticks([20, 50, 80]); axd["e"].set_yticks([20, 50, 80])
    set_domain(axd["e"], xlim=(15, 90), ylim=(15, 90))
    axd["f"].set_xticks([400, 800, 2000, 8000]); axd["f"].set_xticklabels(["400", "800", "2,000", "8,000"])
    set_domain(axd["f"], xlim=(340, 9200))
    add_panel_labels(fig, axd)
    fig.savefig(str(HERE / "figure7_2026-10-12_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "figure7_2026-10-12.pdf"), axes_dict=axd, review_png=str(HERE / "figure7_2026-10-12_review.png"),
                footer="data: MethylGPT test-split datasets, AltumAge test set, GSE194154, Generation Scotland development participants; artifacts/eval, artifacts/gs_cox")
    plt.close(fig)


if __name__ == "__main__":
    main()
