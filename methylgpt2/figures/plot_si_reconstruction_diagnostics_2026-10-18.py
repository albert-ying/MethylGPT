"""Supplementary Figure: reconstruction diagnostics on held-out datasets (18 Oct 2026).

Every value is read from the aggregate outputs of ../si_reconstruction_diagnostics_2026-10-18.py:
  artifacts/eval/si_recon_diagnostics_2026-10-18.json        metrics, intervals, binned residuals
  artifacts/eval/si_recon_diagnostics_2026-10-18.pairs.npz   value samples for the hexbins
Panels a-d: MethylGPT, nearest neighbours and CpG means on the 2,000 held-out profiles (structured 30% masks).
Panel e: the originally submitted model on the first 500 profiles (09-30 masks), its own sub-panel.
Run: MPLBACKEND=Agg python plot_si_reconstruction_diagnostics_2026-10-18.py
     NATURE_FIGURE_NOFOOTER=1 MPLBACKEND=Agg python plot_si_reconstruction_diagnostics_2026-10-18.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullLocator

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, label_panel_group, add_colorbar,
                                 nature_legend, set_domain, save_figure, ref_line, pad_categorical_axis, allow_dead_margin,
                                 ERRORBAR, RUST, GRAY, BLUE, SAGE, INK, CMAP_BLUE_LIGHT, CMAP_SAGE_LIGHT)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
STEM = "supplementary_reconstruction_diagnostics_2026-10-18"
retro_style()
M = {"methylgpt2": ("MethylGPT", RUST["400"], "o"), "knn10": ("Nearest neighbours", BLUE["400"], "D"),
     "cpg_mean": ("CpG means", GRAY["400"], "s"), "january_mlm": ("Originally submitted model", SAGE["500"], "^")}
ORDER = ("cpg_mean", "knn10", "methylgpt2")


def point_range(ax, methods, block, key, fmt, domain, log=False):
    yy = np.arange(len(methods))
    f = np.log10 if log else (lambda z: z)
    mid = (f(domain[0]) + f(domain[1])) / 2
    for y_, m in zip(yy, methods):
        lab, col, mk = M[m]
        v = block["methods"][m]
        lo, hi = v[f"{key}_ci95"]
        ax.errorbar(v[key], y_, xerr=[[v[key] - lo], [hi - v[key]]], fmt="none", ecolor=INK, **ERRORBAR["forest"], zorder=2)
        ax.scatter(v[key], y_, s=22, marker=mk, color=col, edgecolors=INK, linewidths=0.4, zorder=3, clip_on=False)
        right = f(v[key]) < mid
        ax.annotate(fmt.format(v[key]), (hi if right else lo, y_), xytext=(4 if right else -4, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=6, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([M[m][0] for m in methods])
    return yy


def hexbin_resid(ax, x, r, norm, cmap):
    hb = ax.hexbin(x, r, gridsize=(50, 40), extent=(0, 1, -0.4, 0.4), bins=None, norm=norm, cmap=cmap, mincnt=1,
                   linewidths=0)
    return hb


def mean_line(ax, blk, axis, edges, m, lw=1.0, ls="-"):
    mids = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 2
    mean = np.array([np.nan if v is None else v for v in blk["methods"][m][axis]["mean"]], float)
    n = np.asarray(blk["methods"][m][axis]["n"])
    ok = n >= 1000
    ax.plot(mids[ok], mean[ok], ls, color=M[m][1], lw=lw, label=M[m][0], zorder=4)


def beta_bin_panel(ax, rows, methods, sd_key=None, n_label=None):
    mids = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    for m in methods:
        lab, col, mk = M[m]
        v = np.array([r[f"mae_{m}"] for r in rows])
        lo = np.array([r[f"mae_{m}_lo"] for r in rows]); hi = np.array([r[f"mae_{m}_hi"] for r in rows])
        ax.plot(mids, v, "-", color=col, lw=1.0, zorder=2)
        ax.errorbar(mids, v, yerr=[v - lo, hi - v], fmt="none", ecolor=col, **ERRORBAR["point"], zorder=2)
        ax.scatter(mids, v, s=12, marker=mk, color=col, edgecolors="none", zorder=3, clip_on=False, label=lab)
    if sd_key:
        sd = np.array([r[f"median_sd_{sd_key}"] for r in rows])
        q1 = np.array([r[f"sd_{sd_key}_q25"] for r in rows]); q3 = np.array([r[f"sd_{sd_key}_q75"] for r in rows])
        ax.fill_between(mids, q1, q3, color=GRAY["100"], lw=0, zorder=1)
        ax.plot(mids, sd, ls=(0, (3, 1.5)), color=GRAY["600"], lw=1.0, zorder=1.5, label="Within-dataset CpG SD")
    ax.set_xlabel("Observed beta")
    return mids


def log_ticks(ax, axis, ticks):
    labels = [f"{t:g}" for t in ticks]
    if axis == "y":
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
    else:
        ax.set_xticks(ticks); ax.set_xticklabels(labels)


def main():
    R = json.loads((A / "eval/si_recon_diagnostics_2026-10-18.json").read_text())
    pz = np.load(A / "eval/si_recon_diagnostics_2026-10-18.pairs.npz")
    S, J = R["overall_S"], R["january500_T"]
    nprof, nds = S["n_profiles"], S["n_datasets"]

    fig = plt.figure(figsize=figsize_nature("double", aspect=0.98), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.09, w_pad=0.03)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.95, 0.95])
    r1 = outer[0].subgridspec(1, 3, width_ratios=[1.12, 1.0, 1.22])
    ra = r1[0].subgridspec(2, 1)
    r2 = outer[1].subgridspec(1, 3, width_ratios=[1.1, 1.15, 1.5])
    rc = r2[1].subgridspec(2, 1, height_ratios=[0.32, 1.0])
    r3 = outer[2].subgridspec(1, 3, width_ratios=[1.12, 1.0, 1.22])
    re_ = r3[0].subgridspec(2, 1)
    ax = {"a1": fig.add_subplot(ra[0]), "a2": fig.add_subplot(ra[1]), "b1": fig.add_subplot(r1[1]),
          "b2": fig.add_subplot(r1[2]), "cn": fig.add_subplot(rc[0])}
    ax["c1"] = fig.add_subplot(r2[0])
    ax["c2"] = fig.add_subplot(rc[1], sharex=ax["cn"])
    ax["d"] = fig.add_subplot(r2[2])
    ax["e1"], ax["e2"] = fig.add_subplot(re_[0]), fig.add_subplot(re_[1])
    ax["e3"], ax["e4"] = fig.add_subplot(r3[1]), fig.add_subplot(r3[2])

    # a: pooled r and within-CpG R2 (2,000 profiles)
    point_range(ax["a1"], ORDER, S, "pooled_r", "{:.3f}", (0.8, 1.0))
    ax["a1"].set_xlabel("Pooled Pearson $r$")
    ax["a1"].set_title(f"{nprof:,} profiles, {nds} datasets", fontsize=7, fontweight="normal", loc="left")
    point_range(ax["a2"], ORDER, S, "within_cpg_r2_pooled", "{:.2f}", (-0.5, 1.0))
    ax["a2"].set_xlabel("Within-CpG $R^2$")

    # b: residual hexbins for MethylGPT, with the binned mean residual of every method
    res = R["residuals_S"]
    edges = res["x_edges"]
    obs, mg = pz["observed"].astype(float), pz["methylgpt2"].astype(float)
    hb1 = hexbin_resid(ax["b1"], mg, mg - obs, LogNorm(), CMAP_BLUE_LIGHT)
    hb2 = hexbin_resid(ax["b2"], obs, mg - obs, LogNorm(), CMAP_BLUE_LIGHT)
    vmax = max(hb1.get_array().max(), hb2.get_array().max())
    for hb in (hb1, hb2):
        hb.set_clim(1, vmax)
    for k, axis in (("b1", "predicted"), ("b2", "observed")):
        for m in ("cpg_mean", "knn10", "methylgpt2"):
            mean_line(ax[k], res, axis, edges, m)
        ax[k].set_xlabel(f"{axis.capitalize()} beta")
        sl = res["methods"]["methylgpt2"][axis]["ols_slope"]
        ax[k].text(0.97, 0.04 if k == "b1" else 0.93, f"MethylGPT slope {sl:+.3f}", transform=ax[k].transAxes, fontsize=6,
                   color=INK, ha="right", va="bottom" if k == "b1" else "top")
        ax[k].set_ylabel("Residual (predicted − observed)")
    ax["b2"].set_ylabel("")
    top = 10 ** int(np.floor(np.log10(vmax)))
    cb = add_colorbar(hb2, ax["b2"], label="Masked values", ticks=[1, top])
    cb.ax.yaxis.set_minor_locator(NullLocator()); cb.set_ticklabels(["1", f"{top:,}"])

    # c: MAE by observed beta, n per bin, within-dataset CpG SD; MAE / SD
    rows = R["by_beta_bin_S"]
    mids = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    ax["cn"].scatter(mids, [r["n_targets"] / 1e6 for r in rows], s=10, color=INK, edgecolors="none", clip_on=False)
    ax["cn"].set_ylabel("Values (M)")
    ax["cn"].tick_params(labelbottom=False)
    beta_bin_panel(ax["c1"], rows, ORDER, sd_key="dataset")
    ax["c1"].set_yscale("log")
    ax["c1"].set_ylabel("MAE or SD")
    for m in ORDER:
        lab, col, mk = M[m]
        v = np.array([r[f"mae_over_sd_dataset_{m}"] for r in rows])
        lo = np.array([r[f"mae_over_sd_dataset_{m}_lo"] for r in rows]); hi = np.array([r[f"mae_over_sd_dataset_{m}_hi"] for r in rows])
        ax["c2"].plot(mids, v, "-", color=col, lw=1.0)
        ax["c2"].errorbar(mids, v, yerr=[v - lo, hi - v], fmt="none", ecolor=col, **ERRORBAR["point"])
        ax["c2"].scatter(mids, v, s=12, marker=mk, color=col, edgecolors="none", clip_on=False, label=lab)
    ax["c2"].set_yscale("log")
    ax["c2"].set_xlabel("Observed beta")
    ax["c2"].set_ylabel("MAE / CpG SD")

    # d: by tissue (tissues with >= 50 of the 2,000 profiles)
    tis = sorted(R["by_tissue_S"], key=lambda r: r["n_profiles"])
    yy = np.arange(len(tis))
    for m, off in (("cpg_mean", -0.22), ("knn10", 0.0), ("methylgpt2", 0.22)):
        lab, col, mk = M[m]
        v = np.array([r[f"mse_{m}"] for r in tis])
        lo = np.array([r[f"mse_{m}_lo"] for r in tis]); hi = np.array([r[f"mse_{m}_hi"] for r in tis])
        ax["d"].errorbar(v, yy + off, xerr=[v - lo, hi - v], fmt="none", ecolor=col, **ERRORBAR["forest"])
        ax["d"].scatter(v, yy + off, s=14, marker=mk, color=col, edgecolors="none", clip_on=False, label=lab, zorder=3)
    ax["d"].set_yticks(yy)
    ax["d"].set_yticklabels([f"{r['tissue']} ({r['n_profiles']:,}; {r['n_datasets']})" + (" *" if r["fewer_than_5_datasets"] else "")
                             for r in tis])
    ax["d"].set_xscale("log")
    ax["d"].set_xlabel("Masked-value MSE")
    ax["d"].set_title("Tissue (profiles; datasets)", fontsize=7, fontweight="normal", loc="left")

    # e: originally submitted model, first 500 profiles, its own sub-panel
    EM = ("cpg_mean", "january_mlm", "methylgpt2")
    point_range(ax["e1"], EM, J, "mse", "{:.4f}", (0.001, 0.05), log=True)
    ax["e1"].set_xscale("log")
    ax["e1"].set_xlabel("Masked-value MSE")
    ax["e1"].set_title(f"Originally submitted model\n{J['n_profiles']} profiles, {J['n_datasets']} datasets",
                       fontsize=7, fontweight="normal", loc="left")
    point_range(ax["e2"], EM, J, "within_cpg_r2_pooled", "{:.2f}", (-0.5, 1.0))
    ax["e2"].set_xlabel("Within-CpG $R^2$")
    for k in ("e1", "e2"):
        ax[k].set_yticklabels(["CpG means", "Originally\nsubmitted model", "MethylGPT"])
    beta_bin_panel(ax["e3"], J["by_beta_bin"], EM)
    ax["e3"].set_yscale("log")
    ax["e3"].set_ylabel("Mean absolute error")
    jo, jm = pz["j500_observed"].astype(float), pz["j500_january_mlm"].astype(float)
    hj = hexbin_resid(ax["e4"], jm, jm - jo, LogNorm(), CMAP_SAGE_LIGHT)
    hj.set_clim(1, hj.get_array().max())
    for m in ("cpg_mean", "methylgpt2", "january_mlm"):
        mean_line(ax["e4"], J["residuals"], "predicted", J["residuals"]["x_edges"], m)
    ax["e4"].set_xlabel("Predicted beta")
    ax["e4"].set_ylabel("Residual (predicted − observed)")
    top = 10 ** int(np.floor(np.log10(hj.get_array().max())))
    cb = add_colorbar(hj, ax["e4"], label="Masked values", ticks=[1, top])
    cb.ax.yaxis.set_minor_locator(NullLocator()); cb.set_ticklabels(["1", f"{top:,}"])

    trim_axes(fig, exclude=[ax["b1"], ax["b2"], ax["e4"]])
    for k in ("b1", "b2", "e4"):
        ax[k].set_xlim(0, 1); ax[k].set_ylim(-0.4, 0.4)
        ax[k].set_xticks([0, 0.5, 1]); ax[k].set_xticklabels(["0", "0.5", "1"])
        ax[k].set_yticks([-0.4, -0.2, 0, 0.2, 0.4]); ax[k].set_yticklabels(["−0.4", "−0.2", "0", "0.2", "0.4"])
        ref_line(ax[k], y=0)
    for k in ("a1", "a2", "e1", "e2"):
        pad_categorical_axis(ax[k], "y", pad=0.5)
    log_ticks(ax["e1"], "x", [0.001, 0.01]); set_domain(ax["e1"], xlim=(0.001, 0.05))
    allow_dead_margin(ax["e1"], "x", reason="same MSE domain as panel d")
    for k in ("a2", "e2"):
        ax[k].set_xticks([-0.5, 0, 0.5, 1]); ax[k].set_xticklabels(["−0.5", "0", "0.5", "1"]); set_domain(ax[k], xlim=(-0.5, 1))
    ax["a1"].set_xticks([0.8, 0.9, 1.0]); ax["a1"].set_xticklabels(["0.8", "0.9", "1"]); set_domain(ax["a1"], xlim=(0.8, 1.0))
    pad_categorical_axis(ax["d"], "y", pad=0.5)
    log_ticks(ax["d"], "x", [0.001, 0.003, 0.01, 0.03]); set_domain(ax["d"], xlim=(0.001, 0.05))
    allow_dead_margin(ax["d"], "x", reason="log axis closed at the 0.05 tick above the widest interval")
    log_ticks(ax["c1"], "y", [0.01, 0.1, 1]); set_domain(ax["c1"], ylim=(0.006, 1)); ax["c1"].set_ylim(0.006, 2.6)
    allow_dead_margin(ax["c1"], "y", reason="headroom for the method legend")
    log_ticks(ax["e3"], "y", [0.01, 0.1, 1]); set_domain(ax["e3"], ylim=(0.01, 1))
    allow_dead_margin(ax["e3"], "y", reason="headroom for the method legend")
    log_ticks(ax["c2"], "y", [0.5, 1, 2, 4, 8]); set_domain(ax["c2"], ylim=(0.45, 10))
    for k in ("c1", "c2", "e3"):
        ax[k].set_xticks([0, 0.5, 1]); ax[k].set_xticklabels(["0", "0.5", "1"]); set_domain(ax[k], xlim=(0, 1))
    ax["cn"].set_yticks([0, 2.5, 5]); ax["cn"].set_yticklabels(["0", "2.5", "5"]); ax["cn"].set_ylim(-0.3, 5.3)
    ref_line(ax["c2"], y=1)
    nature_legend(ax["b1"], title="Mean residual per bin", loc="upper left")
    nature_legend(ax["c1"], title=f"$n$ = {nprof:,} profiles", loc="upper left")
    nature_legend(ax["e3"], title=f"$n$ = {J['n_profiles']} profiles", loc="upper left")
    label_panel_group(fig, [ax["a1"], ax["a2"]], "a")
    label_panel_group(fig, [ax["b1"], ax["b2"]], "b")
    label_panel_group(fig, [ax["cn"], ax["c1"], ax["c2"]], "c")
    label_panel_group(fig, [ax["d"]], "d")
    label_panel_group(fig, [ax["e1"], ax["e2"], ax["e3"], ax["e4"]], "e")
    fig.savefig(str(HERE / f"{STEM}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{STEM}.pdf"), axes_dict=dict(ax), review_png=str(HERE / f"{STEM}_review.png"),
                footer="data: 2,000 profiles from 107 held-out datasets; artifacts/eval/si_recon_diagnostics_2026-10-18.json")
    plt.close(fig)


if __name__ == "__main__":
    main()
