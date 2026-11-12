"""Figure 5: distilled clocks and sequencing data.

Inputs (run from this directory):
  ../artifacts/runs/multitask_bio_2026-10-01/results.json          held-out correlations of 25 distilled outputs
  ../artifacts/gs_features_2026-10-01/*_mt_bio.embedding.npz       distilled outputs, GS development subjects
  task_a_gs_clock_scores_fixed_grimage_20260629.csv (v12 server results)  BioLearn GrimAge2 from the full array
  ../artifacts/eval/rrbs_gse194154_2026-10-05.{csv,json}            RRBS predictions and summaries
  ../artifacts/eval/rrbs_factor_imputed_comparators_2026-10-09.{csv,json}  comparators with factor-model imputation
GS values are shown only as a binned density (no individual participants).
Run: MPLBACKEND=Agg python plot_fig5_2026-10-02.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain,
                                 save_figure, ref_line, add_colorbar, pad_categorical_axis, allow_dead_margin,
                                 RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_BLUE_LIGHT)

A = Path(__file__).resolve().parent.parent / "artifacts"
GS_CLOCKS = Path("/Entropy/Projects/multimodal/kying/A.Y/project/methylGPT/revision_server_20260622/server_results/task_a/"
                 "task_a_gs_clock_scores_fixed_grimage_20260629.csv")
SPLIT = "full18859_common_task_a_incident_positive_duration_20260629"
retro_style()
NAMES = {"Horvathv1": "Horvath multi-tissue", "Horvathv2": "Horvath skin and blood", "HRSInCHPhenoAge": "HRS PhenoAge",
         "YingCausAge": "CausAge", "YingDamAge": "DamAge", "YingAdaptAge": "AdaptAge", "GrimAge1": "GrimAge",
         "GrimAge2": "GrimAge2", "PACKYRS": "Pack-years", "logA1C": "log HbA1c", "logCRP": "log CRP",
         "Alcohol_McC": "Alcohol", "BMI_McC": "Body mass index", "Smoking_McC": "Smoking", "DNAmTL": "Telomere length",
         "CystatinC": "Cystatin C"}


def main():
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.64), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.05)
    outer = fig.add_gridspec(1, 2, width_ratios=[0.78, 2.3])
    right = outer[1].subgridspec(2, 2, height_ratios=[0.78, 1.22])
    axd = {"a": fig.add_subplot(outer[0]), "b": fig.add_subplot(right[0, 0]), "c": fig.add_subplot(right[0, 1]),
           "d": fig.add_subplot(right[1, :])}

    # a: distilled vs full-array correlation on unseen datasets, ranked
    r = json.loads((A / "runs/multitask_bio_2026-10-01/results.json").read_text())["biomarker_r2_heldout"]
    items = sorted(r.items(), key=lambda kv: kv[1]["pearson_r"])
    ax = axd["a"]
    yy = np.arange(len(items))
    hero = {"GrimAge2", "DunedinPACE", "PhenoAge"}
    ax.hlines(yy, 0.8, [v["pearson_r"] for _, v in items], color=GRAY["200"], lw=0.8)
    ax.scatter([v["pearson_r"] for _, v in items], yy, s=16, color=[RUST["500"] if k in hero else RUST["200"] for k, _ in items],
               edgecolors="none", zorder=3, clip_on=False)
    ax.set_yticks(yy); ax.set_yticklabels([NAMES.get(k, k) for k, _ in items])
    ax.set_xlabel("Pearson $r$, distilled vs full array")
    med = float(np.median([v["pearson_r"] for _, v in items]))
    ax.set_title(f"Unseen datasets (median $r$ = {med:.2f})", fontsize=7, fontweight="normal")

    # b: GS distilled GrimAge2 vs BioLearn GrimAge2 (density only)
    mt = torch.load(A / "runs/multitask_bio_2026-10-01/final.pt", map_location="cpu", weights_only=False)
    names = list(mt["bio_names"])
    rows = []
    for sp in ("train", "val"):
        z = np.load(A / f"gs_features_2026-10-01/COMMON_SAMPLE_SPLIT_{SPLIT}_{sp}_mt_bio.embedding.npz")
        rows.append(pd.DataFrame(z["embedding"] * np.asarray(mt["bio_scale"]) + np.asarray(mt["bio_median"]),
                                 index=z["dnam_id"].astype(str), columns=names))
    B = pd.concat(rows)
    sc = pd.read_csv(GS_CLOCKS)
    w = sc.pivot_table(index="sample_id", columns="model", values="score"); w.index = w.index.astype(str)
    j = B[["GrimAge2"]].join(w[["GrimAgeV2"]], how="inner").dropna()
    ax = axd["b"]
    hb = ax.hexbin(j.GrimAgeV2, j.GrimAge2, gridsize=45, bins="log", cmap=CMAP_BLUE_LIGHT, mincnt=1, linewidths=0)
    rr = float(np.corrcoef(j.GrimAgeV2, j.GrimAge2)[0, 1])
    ax.set_xlabel("GrimAge2, full array (years)"); ax.set_ylabel("GrimAge2, distilled (years)")
    ax.set_title(f"Generation Scotland, $r$ = {rr:.2f}", fontsize=7, fontweight="normal")
    cnt = hb.get_array()
    add_colorbar(hb, ax, label="Participants", ticks=[1, int(10 ** np.floor(np.log10(cnt.max())))])

    # c: RRBS predicted vs chronological age
    rb = pd.read_csv(A / "eval/rrbs_gse194154_2026-10-05.csv")
    rj = json.loads((A / "eval/rrbs_gse194154_2026-10-05.json").read_text())
    ax = axd["c"]
    fi = pd.read_csv(A / "eval/rrbs_factor_imputed_comparators_2026-10-09.csv").set_index("gsm").loc[rb.gsm]
    for col, lab, c, mk in (("ElasticNet (same labels)", "Elastic net, means", BLUE["300"], "s"),
                            ("MethylGPT", "MethylGPT", RUST["400"], "o")):
        ax.scatter(rb.age, rb[col], s=8, marker=mk, color=c, alpha=0.8, edgecolors="none", label=lab, clip_on=False)
    ax.scatter(rb.age, fi["ElasticNet (factor-model imputation)"].to_numpy(), s=9, marker="s", facecolors="none", edgecolors=BLUE["500"],
               linewidths=0.6, label="Elastic net, imputed", clip_on=False)
    ax.plot([18, 88], [18, 88], color=GRAY["400"], lw=0.55, ls=(0, (2, 2)))
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("Predicted age (years)")
    ax.set_title(f"Whole-blood RRBS ($n$ = {rj['n_samples']})", fontsize=7, fontweight="normal")
    ax._legend_top = True

    # d: RRBS, absolute error of each sample after fivefold linear recalibration (seed 20261002, as in the analysis scripts);
    # unmeasured CpGs filled with array means (open) or by MethylGPT imputation (filled); line, mean over samples (printed)
    from sklearn.model_selection import KFold

    def recalibrated(pred, y, seed=20261002):
        pred, y = np.asarray(pred, float), np.asarray(y, float)
        cal = np.empty(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(pred):
            b_, a_ = np.polyfit(pred[tr], y[tr], 1)
            cal[te] = a_ + b_ * pred[te]
        return cal

    fj = json.loads((A / "eval/rrbs_factor_imputed_comparators_2026-10-09.json").read_text())["age"]
    ca = pd.read_csv(A / "eval/rrbs_clocks_all_measured_cpgs_2026-10-09.csv").set_index("gsm").loc[rb.gsm]
    cj = json.loads((A / "eval/rrbs_clocks_all_measured_cpgs_2026-10-09.json").read_text())["age"]
    y = rb.age.to_numpy()
    groups = [("MethylGPT", RUST, [(None, rb["MethylGPT"].to_numpy(), rj["age"]["MethylGPT"])]),
              ("Elastic net", BLUE, [("means", rb["ElasticNet (same labels)"].to_numpy(), rj["age"]["ElasticNet (same labels)"]),
                                     ("imputed", fi["ElasticNet (factor-model imputation)"].to_numpy(), fj["ElasticNet (factor-model imputation)"])])]
    for k, lab in (("Horvath skin & blood", "Horvath skin and blood"), ("Hannum", "Hannum"), ("Horvath 2013", "Horvath multi-tissue"),
                   ("Lin", "Lin")):   # PhenoAge is not a chronological-age predictor; left out of the age comparison
        groups.append((lab, GOLD, [("means", ca[f"{k} (array means)"].to_numpy(), cj[f"{k} (array means)"]),
                                   ("imputed", ca[f"{k} (factor model)"].to_numpy(), cj[f"{k} (factor model)"])]))
    ax = axd["d"]
    rng = np.random.default_rng(0)
    ymax = 0.0
    for x_, (lab, fam, series_) in enumerate(groups):  # noqa: B007
        for kind, pred, ref in series_:
            err = np.abs(recalibrated(pred, y) - y)
            assert abs(err.mean() - ref["recalibrated_5fold"]["mae"]) < 1e-9, lab   # same value as the analysis output
            xo = x_ if kind is None else x_ + (-0.19 if kind == "means" else 0.19)
            filled = kind != "means"
            ax.scatter(xo + rng.uniform(-0.08, 0.08, len(err)), err, s=5, marker="o", linewidths=0.5, clip_on=False, zorder=2,
                       facecolors=fam["400"] if filled else "none", edgecolors=fam["500"] if filled else fam["300"], alpha=0.75)
            ax.plot([xo - 0.13, xo + 0.13], [err.mean()] * 2, color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.text(xo, 49.5, f"{err.mean():.1f}", ha="center", va="bottom", fontsize=6, color=INK)
            ymax = max(ymax, err.max())
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.5, mfc="none", mec=GRAY["500"], label="Means"),
                       Line2D([], [], marker="o", ls="none", ms=3.5, mfc=GRAY["400"], mec=GRAY["500"], label="MethylGPT imputation"),
                       Line2D([], [], color=INK, lw=1.2, label="Mean")],
              loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7, handlelength=1.4, borderaxespad=0.1, columnspacing=1.2)
    base = rj["age"]["fold-mean age (no methylation)"]["recalibrated_5fold"]["mae"]
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0].replace('Horvath skin and blood', 'Horvath skin\nand blood').replace('Horvath multi-tissue', 'Horvath\nmulti-tissue') for g in groups])
    ax.set_ylabel("Absolute error after\nrecalibration (years)")
    ax._ymax = 62.0
    ax._base = base
    ax._datamax = ymax

    trim_axes(fig, exclude=[axd["b"]])
    set_domain(axd["a"], xlim=(0.8, 1.0))
    set_domain(axd["c"], ylim=(0, 128))
    axd["c"].set_yticks([0, 20, 40, 60, 80, 100]); axd["c"].set_xticks([20, 50, 80])
    allow_dead_margin(axd["c"], "y", reason="room above the data for the legend")
    axd["c"].legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.3)
    set_domain(axd["d"], ylim=(-1.5, axd["d"]._ymax))
    axd["d"].set_yticks([0, 10, 20, 30, 40])
    ref_line(axd["d"], y=axd["d"]._base)
    allow_dead_margin(axd["d"], "y", reason="room for value labels")
    pad_categorical_axis(axd["d"], "x", pad=0.5)
    add_panel_labels(fig, axd)
    fig.savefig(str(Path(__file__).with_name("figure5_2026-10-02_draft.png")), dpi=150)
    save_figure(fig, str(Path(__file__).with_name("figure5_2026-10-02.pdf")), axes_dict=axd,
                review_png=str(Path(__file__).with_name("figure5_2026-10-02_review.png")),
                footer="data: MethylGPT corpus test split; Generation Scotland (binned); GSE194154")
    plt.close(fig)


if __name__ == "__main__":
    main()
