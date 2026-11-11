"""Supplementary Figure: age with missing CpGs, more comparators and tissues.

Inputs (aggregate results of ../missing_cpgs_comparators_2026-10-18.py, run from this directory):
  ../artifacts/eval/missing_cpgs_comparators_2026-10-18.json            MedAE, paired differences, tissues,
                                                                         originally submitted model block
  ../artifacts/eval/missing_cpgs_comparators_2026-10-18_training.json   MLP validation histories
Panels
  a  MedAE on the AltumAge test set (n = 4,626) per scenario: MethylGPT, mean-filled elastic net, AltumAge-style MLP
     (5,461 AltumAge training profiles; three seeds as small dots, their mean as the marker), the same MLP trained on
     the 41,845 corpus profiles, and the elastic net applied after MethylGPT imputation; 95% intervals resample datasets.
  b  MedAE difference, MethylGPT minus each comparator, per scenario (paired, datasets resampled).
  c  Per tissue (>= 50 profiles): MethylGPT minus mean-filled elastic net, complete profiles and 27K CpGs.
  d  Same contrast with complete profiles, split by whether the dataset also gave training profiles.
  e  Originally submitted model on the same masks (single run).
  f  MLP validation MedAE by epoch for the three seeds of each training set.
The canvas carries labels only (10 Nov 2026): definitions, interval methods and caveats are in the legend.
Run: MPLBACKEND=Agg python run_nofooter_2026-11-10.py plot_si_missing_cpgs_comparators_2026-10-18.py
     (writes the gated PDF and the no-footer PDF/PNG that the SI uses as Supplementary Fig. 12)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain, ref_line,
                                 allow_dead_margin,
                                 save_figure, pad_categorical_axis, ERRORBAR, RUST, GRAY, BLUE, SAGE, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
NAME = "supplementary_missing_cpgs_comparators_2026-10-18"
retro_style()
SCEN = ["all", "hm450", "epic_v2", "hm27", "random_10", "random_30", "random_50", "random_70", "random_90"]
TICK = ["All", "450K", "EPIC v2", "27K", "10%", "30%", "50%", "70%", "90%"]
SERIES = [  # key in results, legend label, colour, marker, filled
    ("MethylGPT", "MethylGPT", RUST["400"], "o", True),
    ("Elastic net", "Elastic net, means filled", BLUE["400"], "s", False),
    ("MLP", "MLP, means filled, trained on 5,461 profiles", BLUE["500"], "^", True),
    ("MLP corpus labels", "MLP, means filled, trained on 41,845 profiles", BLUE["300"], "v", False),
    ("Elastic net, MethylGPT imputation", "Elastic net, MethylGPT imputation", RUST["300"], "s", False)]


def minus(s):
    return s.replace("-", "−")


def pointrange(ax, x, est, ci, color, marker, filled, s=16, label=None, z=3):
    ax.vlines(x, ci[0], ci[1], color=color, lw=ERRORBAR["point"]["elinewidth"], zorder=z - 1)
    ax.scatter([x], [est], s=s, marker=marker, facecolors=color if filled else "white", edgecolors=color,
               linewidths=0.8, zorder=z, clip_on=False, label=label)


def main():
    r = json.loads((A / "eval/missing_cpgs_comparators_2026-10-18.json").read_text())
    tr = json.loads((A / "eval/missing_cpgs_comparators_2026-10-18_training.json").read_text())
    ov, pdiff, tis = r["overall"], r["paired_differences"], r["tissues"]
    n, nd = r["n_profiles"], r["n_datasets"]
    orig = r.get("originally_submitted_model", {})
    has_orig = bool(orig.get("scenarios"))

    fig = plt.figure(figsize=figsize_nature("double", aspect=1.24), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.11, w_pad=0.05)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.0, 0.72, 1.3, 1.0])
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.3, 1.0])
    r4 = outer[3].subgridspec(1, 2, width_ratios=[1.3, 1.0])
    axd = {"a": fig.add_subplot(outer[0]), "b": fig.add_subplot(outer[1]), "c": fig.add_subplot(r3[0]),
           "d": fig.add_subplot(r3[1]), "e": fig.add_subplot(r4[0]), "f": fig.add_subplot(r4[1])}
    xs = np.arange(len(SCEN))

    # a: MedAE by scenario --------------------------------------------------------------------------------------------
    ax = axd["a"]
    off = np.linspace(-0.3, 0.3, len(SERIES))
    for (key, lab, c, mk, fl), o in zip(SERIES, off):
        for i, s in enumerate(SCEN):
            v = ov[s][key]
            pointrange(ax, i + o, v["medae"], v["medae_ci95_dataset_bootstrap"], c, mk, fl, label=lab if i == 0 else None)
            if key == "MLP":
                sv = [ov[s][f"MLP seed {k}"]["medae"] for k in (1, 2, 3)]
                ax.scatter([i + o + 0.07] * 3, sv, s=3, color=c, edgecolors="none", zorder=4, clip_on=False)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(TICK)
    ax.set_ylabel("MedAE (years)")
    ax.set_title(f"AltumAge test, $n$ = {n:,} profiles from {nd} datasets", fontsize=7, fontweight="normal")
    nature_legend(ax, title="Model", loc="upper left")

    # b: paired differences -------------------------------------------------------------------------------------------
    ax = axd["b"]
    comps = [("MethylGPT minus Elastic net", "Elastic net", BLUE["400"], "s", False, -0.2),
             ("MethylGPT minus MLP", "MLP, 5,461 training profiles", BLUE["500"], "^", True, 0.0),
             ("MethylGPT minus MLP corpus labels", "MLP, 41,845 training profiles", BLUE["300"], "v", False, 0.2)]
    for key, lab, c, mk, fl, o in comps:
        for j, s in enumerate(SCEN):
            d = pdiff[s][key]
            ax.vlines(j + o, d["ci95"][0], d["ci95"][1], color=c, lw=ERRORBAR["point"]["elinewidth"], zorder=2)
            ax.scatter([j + o], [d["difference"]], s=14, marker=mk, facecolors=c if fl else "white", edgecolors=c, linewidths=0.8,
                       zorder=3, clip_on=False, label=lab if j == 0 else None)
    ax.set_xticks(xs); ax.set_xticklabels(TICK)
    ax.set_xlabel("Array subset and CpGs removed at random")
    ax.set_ylabel("MethylGPT minus\ncomparator (years)")
    nature_legend(ax, title="Comparator", loc="lower left")

    # c: tissues, MethylGPT minus elastic net -------------------------------------------------------------------------
    ax = axd["c"]
    keys = sorted(tis, key=lambda t: -tis[t]["n_profiles"])
    yt = np.arange(len(keys))[::-1]
    for (s, lab, fl, o) in (("all", "All CpGs", True, 0.15), ("hm27", "27K CpGs", False, -0.15)):
        for j, (y_, t) in enumerate(zip(yt, keys)):
            d = tis[t]["scenarios"][s]["all profiles"]["MethylGPT minus Elastic net"]
            if d["ci95"][0] is not None:      # no interval for a tissue from a single dataset (Cervix)
                ax.hlines(y_ + o, d["ci95"][0], d["ci95"][1], color=RUST["400"], lw=ERRORBAR["forest"]["elinewidth"], zorder=2)
            ax.scatter([d["difference"]], [y_ + o], s=14, marker="o", facecolors=RUST["400"] if fl else "white", edgecolors=RUST["400"],
                       linewidths=0.8, zorder=3, clip_on=False, label=lab if j == 0 else None)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t.replace('Whole Blood', 'Whole blood')} ({tis[t]['n_profiles']:,}; {tis[t]['n_datasets']})"
                        + ("†" if tis[t]["fewer_than_5_datasets"] else "") for t in keys])
    ax.set_xlabel("MethylGPT minus elastic net,\nMedAE (years)")
    nature_legend(ax, title="Scenario", loc="lower left")

    # d: shared versus unshared training datasets, complete profiles --------------------------------------------------
    ax = axd["d"]
    for (sub, lab, mk, fl, o) in (("shared training dataset", f"Training-split datasets ({r['n_shared']:,} profiles)", "o", True, 0.15),
                                  ("unshared training dataset", f"Other datasets ({r['n_unshared']:,} profiles)", "D", False, -0.15)):
        first = True
        for y_, t in zip(yt, keys):
            c_ = tis[t]["scenarios"]["all"][sub]
            if not c_.get("n_profiles"):
                continue
            d = c_["MethylGPT minus Elastic net"]
            if d["ci95"][0] is not None:
                ax.hlines(y_ + o, d["ci95"][0], d["ci95"][1], color=RUST["500"], lw=ERRORBAR["forest"]["elinewidth"], zorder=2)
            ax.scatter([d["difference"]], [y_ + o], s=12, marker=mk, facecolors=RUST["500"] if fl else "white", edgecolors=RUST["500"],
                       linewidths=0.8, zorder=3, clip_on=False, label=lab if first else None)
            first = False
    ax.set_yticks(yt); ax.set_yticklabels([])
    ax.set_xlabel("MethylGPT minus elastic net,\nall CpGs, MedAE (years)")
    nature_legend(ax, title="Profiles from", loc="lower left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.1)

    # e: originally submitted model -----------------------------------------------------------------------------------
    ax = axd["e"]
    if has_orig:
        for i, s in enumerate(SCEN):
            v = orig["scenarios"][s]
            pointrange(ax, i, v["medae"], v["medae_ci95_dataset_bootstrap"], SAGE["500"], "^", False, s=18,
                       label="Originally submitted model" if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(xs); ax.set_xticklabels(TICK)
        ax.set_xlabel("Array subset and CpGs removed at random")
        ax.set_ylabel("MedAE (years)")
        nature_legend(ax, loc="upper left")
    # f: MLP validation curves ----------------------------------------------------------------------------------------
    ax = axd["f"]
    for fam, lab, c in (("mlp_altumage_train", "5,461 profiles", BLUE["500"]), ("mlp_corpus_labels", "41,845 profiles", BLUE["300"])):
        h = tr[fam]["seed_histories"]
        for k, (seed, hist) in enumerate(sorted(h.items())):
            ep = np.array([e["epoch"] for e in hist]) + 1
            m = np.array([e["valid_medae"] for e in hist])
            ax.plot(ep, m, "-", lw=0.8, color=c, label=f"{lab}, three seeds" if k == 0 else None, clip_on=False)
            b = int(np.argmin(m))
            ax.scatter([ep[b]], [m[b]], s=10, color=c, edgecolors=INK, linewidths=0.3, zorder=3, clip_on=False)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MLP training epoch"); ax.set_ylabel("Validation MedAE (years)")
    ax.set_title(f"AltumAge validation, $n$ = {tr['mlp_altumage_train']['n_valid']:,}", fontsize=7, fontweight="normal")
    nature_legend(ax, title="MLP training set", loc="upper right")

    if not has_orig:
        axd["e"].set_axis_off()
    trim_axes(fig)
    for k in ("a", "e") if has_orig else ("a",):
        axd[k].set_yticks([1, 2, 5, 10, 20]); axd[k].set_yticklabels(["1", "2", "5", "10", "20"])
        set_domain(axd[k], ylim=(1, 30))
    if has_orig:
        allow_dead_margin(axd["e"], "y", reason="same MedAE domain as panel a, so the two panels can be compared by eye")
    axd["f"].set_yticks([2, 5, 10, 20]); axd["f"].set_yticklabels(["2", "5", "10", "20"])
    axd["f"].set_xticks([1, 10, 100, 300]); axd["f"].set_xticklabels(["1", "10", "100", "300"])
    set_domain(axd["f"], xlim=(1, 300), ylim=(1.5, 30))
    axd["b"].set_yticks([-20, -15, -10, -5, 0]); axd["b"].set_yticklabels([minus(str(v)) for v in (-20, -15, -10, -5, 0)])
    set_domain(axd["b"], ylim=(-22, 1))
    axd["c"].set_xticks([-20, -15, -10, -5, 0, 5]); axd["c"].set_xticklabels([minus(str(v)) for v in (-20, -15, -10, -5, 0, 5)])
    set_domain(axd["c"], xlim=(-20, 5))
    axd["d"].set_xticks([-10, -5, 0, 5]); axd["d"].set_xticklabels([minus(str(v)) for v in (-10, -5, 0, 5)])
    set_domain(axd["d"], xlim=(-10, 5))
    for k in ("c", "d"):
        ref_line(axd[k], x=0)
    ref_line(axd["b"], y=0)
    for k in ("a", "b", "e") if has_orig else ("a", "b"):
        ref_line(axd[k], x=3.5)
    for k in ("c", "d"):
        pad_categorical_axis(axd[k], "y", pad=0.5)
    add_panel_labels(fig, axd)
    fig.savefig(str(HERE / f"{NAME}_draft.png"), dpi=150)
    save_figure(fig, str(HERE / f"{NAME}.pdf"), axes_dict=axd, review_png=str(HERE / f"{NAME}_review.png"),
                footer="data: AltumAge test set (4,626 profiles); artifacts/eval/missing_cpgs_comparators_2026-10-18.json")
    plt.close(fig)


if __name__ == "__main__":
    main()
