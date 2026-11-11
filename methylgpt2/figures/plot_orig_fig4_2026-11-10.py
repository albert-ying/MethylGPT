"""Original Figure 4, final build: panels f-h from plot_orig_fig4_2026-10-23.py; a, b-d and e redrawn from the
tables of 10 November 2026 (export_figure_tables_2026-11-10.py on Chronos).

a  tissue composition (pie, from plot_orig_fig4_2026-10-23.py) and the age distribution of the 41,845 age-labelled
   training profiles: one-year bins smoothed with a Gaussian kernel (SD 2 years), drawn as a density
b  first two principal components of the encoder output (the 512 transformer dimensions of the embedding) of the
   5,811 held-out profiles after self-supervised pretraining, coloured by age
c  the same after the age, sex and tissue stage
d  the coordinates of c coloured by tissue
e  median absolute error on the AltumAge test set: bars, all scored profiles; points, datasets with at least 10
   scored profiles (points above 20 years drawn at the top of the axis, with their count)

Data: ../artifacts/figure_tables_2026-11-10/fig4a_training_age_bins_2026-11-10.csv,
      fig4bcd_heldout_embedding_2026-11-10.csv.gz, fig4bcd_pc_variance_age_2026-11-10.csv, fig4e_medae_by_dataset_2026-11-10.csv
Run: MPLBACKEND=Agg python plot_orig_fig4_2026-11-10.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("orig4", HERE / "plot_orig_fig4_2026-10-23.py")
o4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o4)
D = HERE.parent / "artifacts/figure_tables_2026-11-10"
PH_OUT = 836.0
TIS = {"Whole Blood": "#2ca02c", "Brain": "#d62728", "Muscle": "#ff7f0e", "Colon": "#9467bd", "Cells": "#8c564b",
       "Adipose": "#e377c2", "Other": "#1f77b4"}


def white(fig, bb):
    ax = fig.add_axes(o4.fig_rect(*bb), zorder=2)
    ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def pca_panel(fig, box, x, y, var, colours=None, age=None):
    ax = fig.add_axes(o4.fig_rect(*box), zorder=3)
    o = np.random.default_rng(0).permutation(len(x))
    ax.scatter(x, y, s=9, c="#1a1a1a", linewidths=0, rasterized=True, zorder=1)
    ax.scatter(x, y, s=5.5, c="white", linewidths=0, rasterized=True, zorder=2)
    if age is not None:
        sc = ax.scatter(x[o], y[o], s=3, c=age[o], cmap="coolwarm", vmin=0, vmax=100, linewidths=0, rasterized=True, zorder=3)
    else:
        sc = ax.scatter(x[o], y[o], s=3, c=np.asarray(colours, dtype=object)[o].tolist(), linewidths=0, rasterized=True, zorder=3)
    px, py = 0.04 * np.ptp(x), 0.04 * np.ptp(y)
    ax.set_xlim(x.min() - px, x.max() + px); ax.set_ylim(y.min() - py, y.max() + py)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% of variance)", labelpad=2)
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)", labelpad=2)
    o4.style(ax, 7.5, 8.5)
    o4.trim(ax)
    return ax, sc


def age_bar(fig, sc, x, y0, y1):
    cax = fig.add_axes(o4.fig_rect(x, y0, x + 6, y1), zorder=3)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([0, 25, 50, 75, 100]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    cb.set_label("Age (years)", fontsize=7.5, labelpad=3)


def panel_e_bars(fig):
    """MedAE per method on the AltumAge test set: bars over all scored profiles, points for datasets with at least
    10 scored profiles (the distribution the journal asks to see behind a bar)."""
    from matplotlib.ticker import MultipleLocator
    E = pd.read_csv(D / "fig4e_medae_by_dataset_2026-11-10.csv")
    # the same MLP trained on the 41,845 corpus age labels (Chronos commit 5d72547, fig4e_corpus_mlp_2026-11-10.py)
    E = pd.concat([E, pd.read_csv(D / "fig4e_corpus_mlp_by_dataset_2026-11-10.csv")], ignore_index=True)
    E = E[E.set == "AltumAge test"]
    names = [("MethylGPT", "MethylGPT"), ("Elastic net", "Elastic net"), ("MLP (corpus labels)", "MLP (corpus)"),
             ("MLP (AltumAge-style)", "MLP (AltumAge)"),
             ("Horvath multi-tissue", "Horvath"), ("Hannum", "Hannum"), ("Horvath skin and blood", "Horvath skin\n& blood"),
             ("Lin", "Lin"), ("Zhang 2019", "Zhang 2019"), ("AltumAge (pyaging)", "AltumAge\n(pyaging)")]
    n = len(names)
    YMAX = 20
    ax = o4.data_axes(fig, (-0.5, 74.7, n - 0.5, 274.3), (0, 584.2, YMAX, 446.0), (-0.6, n - 0.4), (0, YMAX))
    rng = np.random.default_rng(0)
    rows = []
    for x, (key, lab) in enumerate(names):
        allrow = E[(E.method == key) & (E.dataset == "all")]
        if len(allrow) != 1:
            raise SystemExit(f"Cannot proceed - row not found: {key}, all")
        v = float(allrow.medae.iloc[0])
        ds = E[(E.method == key) & (E.dataset != "all")].medae.values
        ax.bar(x, v, width=0.72, color="#73b3cf" if x == 0 else o4.E_TEST, edgecolor="#333333", linewidth=0.6, alpha=0.85, zorder=2)
        inside = ds[ds <= YMAX]
        ax.scatter(x + rng.uniform(-0.24, 0.24, len(inside)), inside, s=3.5, c="#222222", alpha=0.55, linewidths=0, zorder=3,
                   clip_on=False)
        over = int((ds > YMAX).sum())
        if over:
            ax.scatter([x], [YMAX + 0.55], marker="^", s=12, c="#222222", linewidths=0, zorder=3, clip_on=False)
            ax.text(x + 0.2, YMAX + 0.55, f"{over}", fontsize=7, ha="left", va="center", color="black")
        rows.append(dict(method=key, medae_all=v, n_datasets=len(ds), datasets_above_axis=over))
    ax.axhline(rows[0]["medae_all"], color="black", lw=0.9, ls=(0, (4, 2.5)), zorder=4)
    ax.set_xticks(np.arange(n))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_ylabel("Median absolute error (years)", labelpad=3)
    o4.style(ax, 7.3, 8.2, minor_x=False)
    ax.set_xticklabels([lab for _, lab in names], fontsize=7, linespacing=0.95)
    for t in ax.get_xticklabels():
        t.set_rotation(45); t.set_ha("right"); t.set_rotation_mode("anchor")
    ax.tick_params(axis="x", which="major", length=4)
    o4.trim(ax)
    return rows


def panel_a_density(fig, bb):
    """Age distribution of the 41,845 age-labelled training profiles from one-year bins (Gaussian smoothing, SD 2
    years), filled with a dark-to-teal gradient as in the submitted panel."""
    from matplotlib.colors import LinearSegmentedColormap
    B = pd.read_csv(D / "fig4a_training_age_bins_2026-11-10.csv")
    B = B[B.group == "all"].sort_values("age_bin_start")
    x = B.age_bin_start.values + 0.5
    n = B.n_profiles.values.astype(float)
    k = np.exp(-0.5 * ((np.arange(-8, 9)) / 2.0) ** 2); k /= k.sum()
    dens = np.convolve(n, k, mode="same") / n.sum()
    x0, y0, x1, y1 = bb
    ax = fig.add_axes(o4.fig_rect(x0 + 18, y0 + 4, x1 - 6, y1 - 22), zorder=3)
    grad = LinearSegmentedColormap.from_list("age", ["#2a0f3a", "#3b3f8c", "#2a7f9e", "#3fbfa3", "#b8e0c9"])
    ax.plot(x, dens, color="#2a0f3a", lw=0.9, zorder=3)
    poly = ax.fill_between(x, 0, dens, color="none", zorder=2)
    img = ax.imshow(np.linspace(0, 1, 256)[None, :], extent=(x.min(), x.max(), 0, dens.max() * 1.05), aspect="auto", cmap=grad,
                    zorder=1, origin="lower")
    img.set_clip_path(poly.get_paths()[0], transform=ax.transData)
    ax.set_xlim(0, 100); ax.set_ylim(0, dens.max() * 1.08)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 0.01, 0.02])
    ax.set_yticklabels(["0", "0.01", "0.02"])
    ax.set_xlabel("Age (years)", labelpad=1); ax.set_ylabel("Density", labelpad=2)
    o4.style(ax, 6.5, 7, minor_x=False, minor_y=False)
    o4.trim(ax)
    mark = getattr(ax, "set_label", None)
    return dict(n_profiles=int(n.sum()), peak_density=float(dens.max()), age_at_peak=float(x[dens.argmax()]))


def build():
    T = pd.read_csv(D / "fig4bcd_heldout_embedding_2026-11-10.csv.gz")
    V = pd.read_csv(D / "fig4bcd_pc_variance_age_2026-11-10.csv").query("part == 'transformer'").set_index(["encoder", "pc"])
    imgs = o4.original_images()
    fig = plt.figure(figsize=(o4.PW / 72, o4.PH / 72))
    fig.patch.set_facecolor("white")
    res = {"a": o4.panel_a(fig, imgs, False), "e": panel_e_bars(fig), "f": o4.panel_f(fig), "g": o4.panel_g(fig),
           "h": o4.panel_h(fig)}
    arr, bb = imgs["a"]
    dbb = o4.crop(imgs["a"], bb, (140, 885, 980, arr.shape[0]))[1]
    white(fig, dbb)                                                            # submitted density, earlier corpus
    res["a_density"] = panel_a_density(fig, dbb)
    age = T.age.values
    for enc, box, cbx, title, tx in (("ssl", (338, 34, 512, 180), (528, 50, 165), "Encoder output after pretraining", 425),
                                     ("age_stage", (62, 262, 236, 408), (252, 278, 393), "After the age stage", 149)):
        ax, sc = pca_panel(fig, box, T[f"pc1_transformer_{enc}"].values, T[f"pc2_transformer_{enc}"].values,
                           (V.loc[(enc, 1), "explained_variance_ratio"], V.loc[(enc, 2), "explained_variance_ratio"]), age=age)
        age_bar(fig, sc, *cbx)
        rho = V.loc[(enc, 1), "spearman_age_all"]
        ax.text(0.03, 0.97, f"PC1 and age, $\\rho$ = {rho:.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=7, va="top")
        o4.text_pt(fig, tx, 4.54 if enc == "ssl" else 236.0, title, 9)
        fig.texts[-1].set_ha("center")
        res[f"rho_pc1_{enc}"] = float(rho)
    tis = T.tissue.where(T.tissue.isin(list(TIS)[:-1]), "Other")
    pca_panel(fig, (346, 232, 496, 370), T.pc1_transformer_age_stage.values, T.pc2_transformer_age_stage.values,
              (V.loc[("age_stage", 1), "explained_variance_ratio"], V.loc[("age_stage", 2), "explained_variance_ratio"]),
              colours=tis.map(TIS).values)
    n = tis.value_counts()
    h = [Line2D([], [], ls="", marker="o", ms=3.5, mfc=c, mec="none", label=f"{t.replace('Whole Blood', 'Whole blood')} ({n[t]:,})") for t, c in TIS.items()]
    fig.legend(handles=h, title="Tissue", loc="upper left", bbox_to_anchor=(499 / o4.PW, 1 - 236 / o4.PH), fontsize=7,
               title_fontsize=7, frameon=False, handletextpad=0.2, labelspacing=0.35, alignment="left")
    for letter, x, y in (("a", 11.73, 4.62), ("b", 274.79, 4.77), ("c", 12.17, 221.76), ("d", 316.71, 218.84),
                         ("e", 12.67, 425.01), ("f", 317.21, 394.86), ("g", 13.05, 622.77), ("h", 308.42, 610.15)):
        o4.text_pt(fig, x, y, letter, 12)
    return fig, res


if __name__ == "__main__":
    fig, res = build()
    fig.savefig(o4.OUT / "fig4.pdf"); fig.savefig(o4.OUT / "fig4.png", dpi=300); plt.close(fig)
    for k, v in res.items():
        print(k, v if not isinstance(v, list) else v)
