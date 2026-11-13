"""Figure 1: MethylGPT architecture, training and reconstruction of unseen datasets.

Every value is read from the evaluation outputs in ../artifacts (run from this directory):
  eval/heldout_cf_dense_step2000_2026-09-30.json   random masks, 2,000 profiles from 107 held-out datasets
  eval/heldout_structured_2026-10-01.json          held-out genes / chromosomes, error by beta bin, density pairs
  eval/heldout_platform_2026-10-01.json            array conversion (27K, 450K, EPIC v2 CpGs kept)
  eval/gse308517_cf_dense_step2000_2026-09-30.json fixed GSE308517 protocol (and methyLImp2 when present)
  ppca_core_meanfilled_2026-09-30.rank_selection_dev.json   core rank on development datasets
  logs/train_aging_heads_*.log, logs/train_multitask_bio_*.log   validation MedAE during training
Run: MPLBACKEND=Agg python plot_fig1_2026-10-01.py
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, add_colorbar,
                                 nature_legend, set_domain, save_figure, schematic_axes, module_box, flow_arrow,
                                 ref_line, pad_categorical_axis,
                                 RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK, CMAP_BLUE_LIGHT)

A = Path(__file__).resolve().parent.parent / "artifacts"
retro_style()
# figure-wide key: MethylGPT rust, raw-CpG methods blue, other published methods sage, mean baselines grey
C = {"MethylGPT": RUST["400"], "Nearest neighbours": BLUE["400"], "CpG means": GRAY["400"], "Original MethylGPT": SAGE["500"],
     "methyLImp2": SAGE["300"]}


def load(p):
    return json.loads((A / p).read_text())


def valid_curve(log):
    pts = [(int(m.group(1)), float(m.group(2))) for m in
           re.finditer(r"VALID step (\d+): AltumAge valid MedAE ([0-9.]+)", (A / "logs" / log).read_text())]
    return np.array(pts)


def schematic(ax):
    schematic_axes(ax)
    ax.set_xlim(0, 100); ax.set_ylim(0, 24.5)
    y, h = 10.0, 9.5
    boxes = [
        (0.5, 17.0, "Input profile", "any subset of the 49,156\nvocabulary CpGs", "input"),
        (20.7, 17.0, "Input layer", "probabilistic, 1,536 latent\ndimensions, 145,700 profiles", "representation"),
        (40.9, 17.0, "Encoder", "profile + latent scores in;\n8 attention blocks", "representation"),
        (61.1, 17.0, "Embedding", "2,048 dimensions", "representation"),
        (81.3, 18.2, "Predictions", "age (mean of 3 runs), sex,\ntissue, 25 clock outputs", "output"),
    ]
    fills = {"input": GOLD["100"], "representation": BLUE["100"], "output": SAGE["100"]}
    edges = {"input": GOLD["400"], "representation": BLUE["400"], "output": SAGE["400"]}
    for x, w, t, sub, kind in boxes:
        module_box(ax, (x, y), w, h, t, sub=sub, facecolor=fills[kind], edgecolor=edges[kind], fontsize=6.5)
    for (x0, w0, *_), (x1, *_r) in zip(boxes[:-1], boxes[1:]):
        flow_arrow(ax, (x0 + w0 + 0.3, y + h / 2), (x1 - 0.3, y + h / 2), kind="representation")
    # imputation comes from the factor model's reconstruction (posterior factor scores -> all CpGs)
    yi, hi = 0.2, 8.6
    module_box(ax, (40.9, yi), 17.0, hi, "Imputation", "missing CpGs reconstructed;\nobserved values kept",
               facecolor=SAGE["100"], edgecolor=SAGE["400"], fontsize=6.5)
    xf = 20.7 + 17.0 / 2
    ax.plot([xf, xf], [y - 0.3, yi + hi / 2], color=BLUE["400"], lw=1.0, zorder=2, solid_capstyle="butt")
    flow_arrow(ax, (xf, yi + hi / 2), (40.9 - 0.3, yi + hi / 2), kind="representation")
    # the encoder also receives the profile itself (values, residuals and observation mask)
    xin, xen, yt = 0.5 + 17.0 / 2, 40.9 + 17.0 / 2, 22.6
    ax.plot([xin, xin, xen], [y + h + 0.3, yt, yt], color=BLUE["400"], lw=1.0, zorder=2, solid_capstyle="butt")
    flow_arrow(ax, (xen, yt), (xen, y + h + 0.3), kind="representation")
    ax.text((xin + xen) / 2, yt + 0.5, "profile (values, residuals, observation mask)", ha="center", va="bottom", fontsize=6, color=INK)
    ax.text(61.1, yi + hi / 2, "One encoder pass per profile and training run, without recursion\n(age: mean of three runs); the encoder was trained on masked\nvalues, then on age, sex, tissue and clock targets",
            ha="left", va="center", fontsize=6, color=INK)


def main():
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.66), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.04)
    outer = fig.add_gridspec(3, 1, height_ratios=[0.50, 0.92, 0.95])
    r2 = outer[1].subgridspec(1, 3, width_ratios=[0.95, 1.05, 1.75])
    r3 = outer[2].subgridspec(1, 2, width_ratios=[2.9, 1.0])
    axd = {"a": fig.add_subplot(outer[0]), "b": fig.add_subplot(r2[0]), "c": fig.add_subplot(r2[1]),
           "d": fig.add_subplot(r2[2]), "e": fig.add_subplot(r3[0]), "f": fig.add_subplot(r3[1])}

    schematic(axd["a"])

    # b: core rank on development datasets
    rs = load("ppca_core_meanfilled_2026-09-30.rank_selection_dev.json")
    ranks = sorted(int(k) for k in rs["ranks"])
    ax = axd["b"]
    ax.plot(ranks, [rs["ranks"][str(r)]["mse"] * 1e3 for r in ranks], "-o", color=RUST["400"], ms=3, lw=1.0, clip_on=False)
    ax.set_xticks([256, 512, 1024, 1536])
    ax.set_xlabel("Latent dimensions"); ax.set_ylabel("Masked MSE (×10$^{-3}$)")
    ax.text(0.97, 0.97, f"Development data\n$n$ = {rs['n_dev_profiles']:,} profiles", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color=INK)

    # c: density of observed vs predicted, random 30% masks on held-out datasets
    ax = axd["c"]
    pz = np.load(A / "eval/heldout_structured_2026-10-01.random30_pairs.npz")
    st = load("eval/heldout_structured_2026-10-01.json")
    hb = ax.hexbin(pz["observed"], pz["methylgpt2"], gridsize=60, bins="log", cmap=CMAP_BLUE_LIGHT, mincnt=1,
                   extent=(0, 1, 0, 1), linewidths=0)
    ax.plot([0, 1], [0, 1], color=GRAY["400"], lw=0.55, ls=(0, (2, 2)))
    r30 = st["mask_types"]["random_30"]["methylgpt2"]

    ax.set_xlabel("Observed beta"); ax.set_ylabel("Predicted beta")
    ax.set_title(f"{st['n_test_groups']} unseen datasets\nMSE = {r30['mse']:.4f}, within-CpG $R^2$ = {r30['within_cpg_r2_pooled']:.2f}",
                 fontsize=7, fontweight="normal")
    cnt = hb.get_array()
    add_colorbar(hb, ax, label="Pairs", ticks=[1, int(10 ** np.floor(np.log10(cnt.max())))])

    # e: masked-value MSE by task (held-out datasets)
    ax = axd["e"]
    hr = load("eval/heldout_cf_dense_step2000_2026-09-30.json")["rates"]
    pl = load("eval/heldout_platform_2026-10-01.json")["rates"]
    rows = [(f"Random {int(float(k) * 100)}%", hr[k]) for k in ("0.15", "0.3", "0.5", "0.7")]
    rows += [("Held-out genes", st["mask_types"]["gene_fold"]), ("Held-out chromosome", st["mask_types"]["chromosome"])]
    rows += [(f"{lab} CpGs only", pl[k]) for lab, k in (("27K", "keep_hm27"), ("450K", "keep_hm450"), ("EPIC v2", "keep_epic_v2"))]
    xx = np.arange(len(rows))
    for name, key, off, mk in (("CpG means", "cpg_mean", -0.21, "s"), ("Nearest neighbours", "knn10", -0.07, "D"), ("MethylGPT", "methylgpt2", 0.07, "o")):
        ax.scatter(xx + off, [r[1][key]["mse"] for r in rows], s=14, marker=mk, color=C[name], label=name,
                   edgecolors="none", clip_on=False, zorder=3)
    jan = [(i, r[1]["january_mlm"]["mse"]) for i, r in enumerate(rows) if "january_mlm" in r[1]]
    ax.scatter([xx[i] + 0.21 for i, _ in jan], [v for _, v in jan], s=14, marker="^", color=C["Original MethylGPT"],
               label="Original MethylGPT ($n$ = 500)", edgecolors="none", clip_on=False, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 0.12)
    ax.set_yticks([3e-3, 1e-2, 3e-2]); ax.set_yticklabels(["0.003", "0.01", "0.03"])
    ax.set_xticks(xx); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("Masked-value MSE")
    nature_legend(ax, title="Method", ncol=4, loc="upper center")

    # f: MAE by observed beta
    ax = axd["f"]
    bb = st["mask_types"]["random_30"]["by_beta_bin"]
    edges = [(float(k.split(",")[0][1:]), float(k.split(",")[1][:-1])) for k in bb]
    mids = [(a + b) / 2 for a, b in edges]
    for name, key in (("CpG means", "cpg_mean"), ("Nearest neighbours", "knn10"), ("MethylGPT", "methylgpt2")):
        ax.plot(mids, [v[key]["mae"] for v in bb.values()], "-o", ms=3, lw=1.0, color=C[name], label=name, clip_on=False)
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.03, 0.1]); ax.set_yticklabels(["0.01", "0.03", "0.1"])
    ax.set_xlabel("Observed beta"); ax.set_ylabel("Mean absolute error")

    # d: GSE308517 fixed protocol, 30% masking: one dot per profile (MAE averaged over the three mask seeds); line, mean over profiles
    ax = axd["d"]
    gj = load("eval/gse308517_cf_dense_step2000_2026-09-30.json")
    seeds = sorted(k for k in gj["conditions"] if k.endswith("_rate30_per_sample"))
    per = {k: np.mean([gj["conditions"][s_][k] for s_ in seeds], axis=0) for k in ("MLM", "fold_mean", "donor_knn_k5", "methylgpt2", "methylgpt2_foldmean")}
    ml = A / "eval/gse308517_methylimp2_2026-10-01.json"
    rows = [("Original MethylGPT", per["MLM"], C["Original MethylGPT"], "^"), ("CpG means", per["fold_mean"], GRAY["400"], "s")]
    if ml.exists():
        mj = json.loads(ml.read_text())
        rows.append(("methyLImp2", np.mean([mj["conditions"][s_]["methylimp2"] for s_ in sorted(k for k in mj["conditions"] if k.endswith("_rate30_per_sample"))], axis=0),
                     C["methyLImp2"], "v"))
    rows += [("Nearest neighbours", per["donor_knn_k5"], C["Nearest neighbours"], "D"), ("MethylGPT", per["methylgpt2"], RUST["400"], "o"),
             ("MethylGPT, fold-mean centre", per["methylgpt2_foldmean"], RUST["600"], "o")]
    rng = np.random.default_rng(0)
    yy = np.arange(len(rows))[::-1]
    for y_, (name, v, col, mk) in zip(yy, rows):
        ax.scatter(v, y_ + rng.uniform(-0.22, 0.22, len(v)), s=5, marker=mk, facecolors="none", edgecolors=col, linewidths=0.5, alpha=0.7, clip_on=False, zorder=2)
        m_ = float(np.mean(v))
        ax.plot([m_, m_], [y_ - 0.34, y_ + 0.34], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
        ax.text(0.0465, y_, f"{m_:.4f}", va="center", ha="left", fontsize=6, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Mean absolute error per profile")
    ax.set_title(f"GSE308517, 30% masked, {len(per['methylgpt2'])} profiles", fontsize=7, fontweight="normal")
    ax._xmax = 0.053

    trim_axes(fig, exclude=[axd["a"], axd["c"]])
    set_domain(axd["c"], xlim=(0, 1), ylim=(0, 1))
    set_domain(axd["f"], ylim=(0.01, 0.2))
    axd["d"].set_xticks([0.02, 0.03, 0.04]); axd["d"].set_xticklabels(["0.02", "0.03", "0.04"])
    set_domain(axd["d"], xlim=(0.012, axd["d"]._xmax))
    pad_categorical_axis(axd["d"], "y", pad=0.5)
    pad_categorical_axis(axd["e"], "x", pad=0.5)
    for lbl in axd["e"].get_xticklabels():
        lbl.set_rotation(30); lbl.set_ha("right"); lbl.set_rotation_mode("anchor")
    from nature_figure_utils import allow_dead_margin
    set_domain(axd["e"], ylim=(2e-3, 0.12))
    allow_dead_margin(axd["e"], "y", reason="headroom reserved for the method legend")
    allow_dead_margin(axd["d"], "x", reason="room for the printed values")
    add_panel_labels(fig, axd)
    fig.savefig(str(Path(__file__).with_name("figure1_2026-10-01_draft.png")), dpi=150)
    save_figure(fig, str(Path(__file__).with_name("figure1_2026-10-01.pdf")), axes_dict=axd,
                review_png=str(Path(__file__).with_name("figure1_2026-10-01_review.png")),
                footer="data: held-out test-split datasets of the MethylGPT corpus and GSE308517; values from artifacts/eval")
    plt.close(fig)


if __name__ == "__main__":
    main()
