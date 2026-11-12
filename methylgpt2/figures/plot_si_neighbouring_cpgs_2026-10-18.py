"""Supplementary Figure: neighbouring CpGs (18 Oct 2026).

Every value is read from the aggregate outputs of ../si_neighbouring_cpgs_2026-10-18.py:
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.json        distances, correlation summaries, masking results
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.nn.npz      nearest-neighbour distance of each vocabulary CpG
  artifacts/eval/si_neighbouring_cpgs_2026-10-18.pairs.npz   per-pair correlations
Run: MPLBACKEND=Agg python plot_si_neighbouring_cpgs_2026-10-18.py
     NATURE_FIGURE_NOFOOTER=1 MPLBACKEND=Agg python plot_si_neighbouring_cpgs_2026-10-18.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, label_panel_group, add_colorbar, nature_legend,
                                 set_domain, save_figure, ref_line, pad_categorical_axis, ERRORBAR,
                                 RUST, GRAY, BLUE, INK, CMAP_BLUE_LIGHT)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
STEM = "supplementary_neighbouring_cpgs_2026-10-18"
retro_style()
SHORT = {"<50 bp": "<50 bp", "50-200 bp": "50–200 bp", "200 bp-1 kb": "0.2–1 kb", "1-10 kb": "1–10 kb",
         "10-100 kb": "10–100 kb", "100 kb-1 Mb": "0.1–1 Mb", "different chromosomes": "Other chromosome",
         ">=100 kb": "≥100 kb"}


def corr_panel(ax, rows, title):
    xx = np.arange(len(rows))
    for key, lab, col, light, mk, off in (("observed", "Raw beta values", BLUE["400"], BLUE["100"], "D", -0.13),
                                          ("reconstruction", "MethylGPT reconstruction", RUST["400"], RUST["100"], "o", 0.13)):
        med = np.array([r[f"median_r_{key}"] for r in rows])
        lo = np.array([r[f"median_r_{key}_lo"] for r in rows]); hi = np.array([r[f"median_r_{key}_hi"] for r in rows])
        q1 = np.array([r[f"q25_r_{key}"] for r in rows]); q3 = np.array([r[f"q75_r_{key}"] for r in rows])
        ax.vlines(xx + off, q1, q3, color=light, lw=2.6, zorder=1)
        ax.errorbar(xx + off, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK, **ERRORBAR["point"], zorder=2)
        ax.scatter(xx + off, med, s=16, marker=mk, color=col, edgecolors=INK, linewidths=0.4, zorder=3, clip_on=False,
                   label=lab)
    ax.set_xticks(xx)
    ax.set_xticklabels([SHORT[r["label"]] for r in rows])
    ax.set_ylabel("Correlation across profiles ($r$)")
    ax.set_title(title, fontsize=7, fontweight="normal", loc="left")
    return xx


def main():
    R = json.loads((A / "eval/si_neighbouring_cpgs_2026-10-18.json").read_text())
    nnz = np.load(A / "eval/si_neighbouring_cpgs_2026-10-18.nn.npz")
    pz = np.load(A / "eval/si_neighbouring_cpgs_2026-10-18.pairs.npz")
    d = nnz["nn_distance"]
    d = d[np.isfinite(d)]

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.62), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.03)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0])
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.0, 1.35, 1.35])
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.15, 1.6, 0.95])
    ax = {"a": fig.add_subplot(r1[0]), "b1": fig.add_subplot(r1[1]), "b2": fig.add_subplot(r1[2]),
          "c": fig.add_subplot(r2[0]), "d": fig.add_subplot(r2[1]), "e": fig.add_subplot(r2[2])}

    # a: nearest-neighbour distance, ECDF
    xs = np.sort(d) / 1000
    ax["a"].step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=BLUE["400"], lw=1.0,
                 solid_capstyle="butt", solid_joinstyle="miter")
    ax["a"].set_xscale("log")
    ax["a"].set_xlabel("Distance to nearest vocabulary CpG (kb)")
    ax["a"].set_ylabel("Fraction of CpGs")
    fw = R["nn_distance"]["frac_within"]
    ax["a"].set_title(f"{R['nn_distance']['n_with_neighbour']:,} vocabulary CpGs", fontsize=7, fontweight="normal", loc="left")
    f1k = fw["<=1000"]
    ax["a"].annotate(f"{f1k:.0%} within 1 kb", (1, f1k), xytext=(6, -2), textcoords="offset points", ha="left", va="top",
                     fontsize=6, color=INK)
    ax["a"].scatter([1], [f1k], s=12, color=INK, zorder=3, clip_on=False)

    # b: correlation by distance, corpus and whole blood (tissue-centred)
    cc = R["correlation"]
    corr_panel(ax["b1"], cc["corpus"]["tissue"], f"Corpus, {cc['corpus']['info']['n_profiles']:,} profiles, "
                                                 f"{cc['corpus']['info']['n_datasets']:,} datasets")
    corr_panel(ax["b2"], cc["whole_blood"]["tissue"], f"Whole blood, {cc['whole_blood']['info']['n_profiles']:,} profiles, "
                                                      f"{cc['whole_blood']['info']['n_datasets']} datasets")
    ax["b2"].set_ylabel("")

    # c: per-pair reconstruction vs observed correlation, pairs within 1 kb (corpus)
    near = pz["bin"] <= 2
    ro = pz["corpus_tissue_r_observed"][near].astype(float)
    rr = pz["corpus_tissue_r_reconstruction"][near].astype(float)
    ok = np.isfinite(ro) & np.isfinite(rr)
    hb = ax["c"].hexbin(ro[ok], rr[ok], gridsize=36, extent=(-0.4, 1, -0.4, 1), bins="log", cmap=CMAP_BLUE_LIGHT,
                        mincnt=1, linewidths=0)
    ax["c"].set_xlabel("Raw beta values ($r$)")
    ax["c"].set_ylabel("MethylGPT reconstruction ($r$)")
    ax["c"].set_title(f"Corpus pairs within 1 kb ($n$ = {ok.sum():,})", fontsize=7, fontweight="normal", loc="left")
    top = 10 ** int(np.floor(np.log10(hb.get_array().max())))
    cb = add_colorbar(hb, ax["c"], label="Pairs", ticks=[1, top])
    cb.ax.yaxis.set_minor_locator(NullLocator()); cb.set_ticklabels(["1", f"{top:,}"])

    # d: neighbour-masked minus random-masked MSE by nearest-neighbour distance
    rows = R["masking"]["nearest_neighbour"]["rows"]
    bins = [r for r in rows if r["subset"].startswith("nn ")]
    prim = [r for r in rows if r["subset"] == "nn_within_1kb"][0]
    allr = [r for r in rows if r["subset"] == "all_targets"][0]
    win = [r for r in R["masking"]["window_1kb"]["rows"] if r["subset"] == "all_targets"][0]
    groups = [(prim, "Neighbour ≤1 kb"), (allr, "All targets"), (win, "1 kb window")] + \
             [(r, SHORT[r["subset"][3:]]) for r in bins]
    xx = np.r_[np.arange(3), np.arange(len(bins)) + 3.7]
    for key, lab, col, mk, off in (("methylgpt2", "MethylGPT", RUST["400"], "o", 0.0),):
        v = np.array([g[f"diff_{key}"] for g, _ in groups]) * 1e4
        lo = np.array([g[f"diff_{key}_lo"] for g, _ in groups]) * 1e4
        hi = np.array([g[f"diff_{key}_hi"] for g, _ in groups]) * 1e4
        ax["d"].errorbar(xx + off, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, **ERRORBAR["point"], zorder=2)
        ax["d"].scatter(xx + off, v, s=16, marker=mk, color=col, edgecolors=INK, linewidths=0.4, zorder=3,
                        clip_on=False, label=lab)
    ax["d"].set_xticks(xx)
    ax["d"].set_xticklabels([lab for _, lab in groups])
    ax["d"].set_ylabel("MSE, neighbour − random mask (×10$^{-4}$)")
    ax["d"].set_title(f"Targets in {prim['n_profiles']:,} held-out profiles, {prim['n_datasets']} datasets",
                      fontsize=7, fontweight="normal", loc="left")

    # e: per-dataset difference for targets with a neighbour within 1 kb
    pdd = np.sort(np.array([v for v in prim["per_dataset_diff_methylgpt2"] if v is not None]) * 1e4)
    ax["e"].scatter(np.arange(1, len(pdd) + 1), pdd, s=8, color=RUST["300"], edgecolors=INK, linewidths=0.3,
                    clip_on=False, zorder=2)
    ax["e"].set_xlabel(f"Datasets, ranked ($n$ = {len(pdd)})")
    ax["e"].set_ylabel("MSE difference (×10$^{-4}$)")
    ax["e"].set_title(f"Neighbour ≤1 kb\n{(pdd > 0).sum()} of {len(pdd)} datasets > 0", fontsize=7, fontweight="normal", loc="left")

    trim_axes(fig, exclude=[ax["c"]])
    ax["a"].set_xticks([0.001, 0.1, 10, 1000]); ax["a"].set_xticklabels(["0.001", "0.1", "10", "1000"])
    set_domain(ax["a"], xlim=(0.001, 1000))
    ax["e"].set_xticks([1, 50, 100]); ax["e"].set_xticklabels(["1", "50", "100"]); set_domain(ax["e"], xlim=(1, len(pdd))); ax["e"].set_xlim(-3, len(pdd) + 4)
    ax["c"].set_xlim(-0.4, 1); ax["c"].set_ylim(-0.4, 1)
    ax["c"].plot([-0.4, 1], [-0.4, 1], color=GRAY["400"], lw=0.55, ls=(0, (2, 2)))
    for k in ("b1", "b2", "d"):
        pad_categorical_axis(ax[k], "x", pad=0.5)
        for lbl in ax[k].get_xticklabels():
            lbl.set_rotation(40); lbl.set_ha("right"); lbl.set_rotation_mode("anchor")
    for k in ("b1", "b2"):
        ax[k].set_yticks([0, 0.5, 1]); ax[k].set_yticklabels(["0", "0.5", "1"]); set_domain(ax[k], ylim=(-0.05, 1))
    ax["d"].text(3.7 + (len(bins) - 1) / 2, ax["d"].get_ylim()[1], "by distance to nearest neighbour", ha="center", va="top",
                 fontsize=6, color=INK)
    ref_line(ax["d"], y=0); ref_line(ax["e"], y=0)
    for k in ("b1", "b2"):
        ref_line(ax[k], y=0)
    nature_legend(ax["b2"], title="Values", loc="upper right")
    nature_legend(ax["d"], title="Model", loc="center right")
    label_panel_group(fig, [ax["a"]], "a")
    label_panel_group(fig, [ax["b1"], ax["b2"]], "b")
    label_panel_group(fig, [ax["c"]], "c")
    label_panel_group(fig, [ax["d"]], "d")
    label_panel_group(fig, [ax["e"]], "e")
    fig.savefig(str(HERE / f"{STEM}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{STEM}.pdf"), axes_dict=ax, review_png=str(HERE / f"{STEM}_review.png"),
                footer="data: vocabulary positions (hg19); training-split and held-out corpus profiles; "
                       "artifacts/eval/si_neighbouring_cpgs_2026-10-18.json")
    plt.close(fig)


if __name__ == "__main__":
    main()
