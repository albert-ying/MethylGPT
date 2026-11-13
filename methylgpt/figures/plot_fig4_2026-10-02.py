"""Figure 4: age prediction.

Inputs (run from this directory):
  ../artifacts/logs/train_aging_heads_*.log, train_multitask_bio_*.log      validation MedAE during training
  ../artifacts/eval/age_ensemble_2026-10-04.json (+ .predictions.npz)        released model (age = mean of three runs)
  ../artifacts/eval/age_enet_factor_imputed_2026-10-02.json                  elastic net with factor-model imputation
  ../artifacts/eval/reprogramming_gse54848_2026-10-01.csv                    GSE54848 replicates
Two values are published, not recomputed: AltumAge's MedAE on this test set (2.153 years, Table 1 of
de Lima Camillo et al. 2022) and the original MethylGPT's regenerated test MedAE (2.21 years, v12).
Run: MPLBACKEND=Agg python plot_fig4_2026-10-02.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain,
                                 save_figure, pad_categorical_axis, RUST, GRAY, BLUE, GOLD, SAGE, PLUM, INK)

HERE = Path(__file__).resolve().parent
A = HERE.parent / "artifacts"
retro_style()
PUBLISHED = {"AltumAge (reported)": 2.153, "Original MethylGPT": 2.21}
MG, EN = "MethylGPT", "ElasticNet (same labels)"


def curve(log):
    p = A / "logs" / log
    if not p.exists():
        return None
    pts = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"VALID step (\d+): AltumAge valid MedAE ([0-9.]+)", p.read_text())]
    return np.array(pts) if pts else None


def ensemble_res():
    """Results of the released MethylGPT (age = mean of three training runs) in the layout the panels read."""
    r = json.loads((A / "eval/age_ensemble_2026-10-04.json").read_text())
    res = {"altumage_test": {MG: {s: v["ensemble"] for s, v in r["b_missing_cpg_scenarios"].items()},
                             EN: {s: v["elastic net"] for s, v in r["b_missing_cpg_scenarios"].items()}},
           "clocks_on_altumage_test": {c: {"clock": v["clock"], MG: v["ensemble"]} for c, v in r["e_clocks_altumage_test"].items()
                                       if isinstance(v, dict) and "clock" in v},
           "by_tissue": {MG: {}, EN: {}}}
    # panel d: all 34 held-out datasets (age_tissue_all_datasets_2026-10-10.py); the analysis without flagged datasets is in Supplementary Table 4
    for t, v in json.loads((A / "eval/age_tissue_all_datasets_2026-10-10.json").read_text())["tissues"].items():
        if "all_datasets" in v:
            res["by_tissue"][MG][t] = {"medae": v["all_datasets"]["MethylGPT"], "n": v["all_datasets"]["n"]}
            res["by_tissue"][EN][t] = {"medae": v["all_datasets"]["elastic net"], "n": v["all_datasets"]["n"]}
    return res


def main():
    res = ensemble_res()
    pr = np.load(A / "eval/age_ensemble_2026-10-04.predictions.npz")
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.98), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.09, w_pad=0.05)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.88, 1.0])
    r1 = outer[0].subgridspec(1, 2, width_ratios=[1.6, 1.0])
    r2 = outer[1].subgridspec(1, 2, width_ratios=[1.2, 1.0])
    r3 = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.25])
    axd = {"a": fig.add_subplot(r1[0]), "b": fig.add_subplot(r1[1]), "c": fig.add_subplot(r2[0]),
           "d": fig.add_subplot(r2[1]), "e": fig.add_subplot(r3[0]), "f": fig.add_subplot(r3[1])}

    # a: validation MedAE during training (age, sex and tissue stage, then clock targets added at step 8,000)
    ax = axd["a"]
    dash = (0, (3, 1.5))   # second encoder learning rate (3e-4) of the control
    for log, c, lab, ls in (("train_aging_heads_clean_2026-10-01.log", RUST["300"], "Pretrained, three seeds", "-"),
                            ("train_aging_heads_seed20261002.log", RUST["300"], None, "-"), ("train_aging_heads_seed20261003.log", RUST["300"], None, "-"),
                            ("train_aging_heads_nopretrain_lr5e-5.log", PLUM["400"], "Randomly initialized", "-"),
                            ("train_aging_heads_nopretrain_lr3e-4.log", PLUM["400"], None, dash)):
        cv = curve(log)
        if cv is not None:
            ax.plot(cv[:, 0], cv[:, 1], ls=ls, lw=0.9, color=c, label=lab, clip_on=False)
    for log in ("train_multitask_bio_2026-10-01.log", "train_multitask_bio_seed20261002.log", "train_multitask_bio_seed20261003.log"):
        cv = curve(log)
        if cv is not None:
            ax.plot(cv[:, 0] + 8000, cv[:, 1], "-", lw=0.9, color=RUST["500"],
                    label="After adding clock targets" if log.endswith("2026-10-01.log") else None, clip_on=False)
    ax.set_xscale("log")
    ax.set_xlabel("Training step"); ax.set_ylabel("Validation MedAE (years)")
    nature_legend(ax, loc="upper right")

    # b: predicted vs chronological age, AltumAge test
    ax = axd["b"]
    y, p = pr["altumage_age"], pr["ensemble|all"]
    ax.scatter(y, p, s=2, color=RUST["400"], alpha=0.3, edgecolors="none", clip_on=False, rasterized=True)
    ax.plot([0, 115], [0, 115], color=GRAY["400"], lw=0.6, ls=(0, (2, 2)))
    s = res["altumage_test"][MG]["all"]
    ax.text(2, 112, f"AltumAge test, $n$ = {s['n']:,}\nMedAE {s['medae']:.2f} years\n$r$ = {s['pearson_r']:.3f}", fontsize=6,
            ha="left", va="top", color=INK)
    ax.set_xlabel("Chronological age (years)"); ax.set_ylabel("MethylGPT age (years)")

    # c: comparators and MethylGPT on the same profiles
    ax = axd["c"]
    ck, mg_all = res["clocks_on_altumage_test"], res["altumage_test"][MG]["all"]["medae"]
    z90 = "Zhang 2019 (>= 90% of its CpGs observed)"
    ra = json.loads((A / "eval/age_ensemble_2026-10-04.json").read_text())["a_altumage_test"]
    ci_mg, ci_en = ra["ensemble"]["medae_ci95_dataset_bootstrap"], ra["elastic net"]["medae_ci95_dataset_bootstrap"]
    # (label, comparator MedAE, MethylGPT MedAE on the same profiles, marker, comparator colour, comparator 95% CI or None)
    rows = [("Elastic net, same age labels", res["altumage_test"][EN]["all"]["medae"], mg_all, "s", BLUE["400"], ci_en),
            ("Original MethylGPT", PUBLISHED["Original MethylGPT"], mg_all, "^", SAGE["500"], None),
            ("AltumAge (reported)", PUBLISHED["AltumAge (reported)"], mg_all, "*", GOLD["500"], None),
            ("AltumAge (pyaging)", ck["AltumAge"]["clock"]["medae"], ck["AltumAge"][MG]["medae"], "D", GOLD["500"], None),
            ("Horvath multi-tissue", ck["Horvath 2013"]["clock"]["medae"], ck["Horvath 2013"][MG]["medae"], "D", GOLD["500"], None),
            ("Zhang 2019", ck[z90]["clock"]["medae"], ck[z90][MG]["medae"], "D", GOLD["500"], None),
            ("Horvath skin and blood", ck["Horvath skin & blood"]["clock"]["medae"], ck["Horvath skin & blood"][MG]["medae"], "D", GOLD["500"], None)]
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, cv, mv, mk, col, ci) in zip(yy, rows):
        ax.hlines(y_, mv, cv, color=GRAY["200"], lw=1.0)
        if ci is not None:   # 95% interval, datasets resampled
            ax.hlines(y_ + 0.16, ci[0], ci[1], color=col, lw=0.8, zorder=2)
        if ci is not None:
            ax.hlines(y_ - 0.16, ci_mg[0], ci_mg[1], color=RUST["400"], lw=0.8, zorder=2)
        ax.scatter(cv, y_, s=24 if mk != "*" else 40, marker=mk, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3, clip_on=False)
        ax.scatter(mv, y_, s=18, color=RUST["400"], zorder=4, clip_on=False)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("MedAE, AltumAge test (years)")
    ax.set_title("Comparator (open) and MethylGPT (filled), same profiles", fontsize=7, fontweight="normal")

    # d: by tissue, all held-out datasets
    ax = axd["d"]
    bt_m, bt_e = res["by_tissue"][MG], res["by_tissue"][EN]
    # organ tissues; samples annotated by cell type or cell line ("Cells") or as "Other" are reported in Supplementary Table 4
    keys = sorted([k for k in bt_m if k not in ("Cells", "Other")], key=lambda k: -bt_m[k]["n"])
    yy = np.arange(len(keys))[::-1]
    tj = json.loads((A / "eval/age_tissue_all_datasets_2026-10-10.json").read_text())["tissues"]
    for y_, k in zip(yy, keys):
        d_ = tj[k]["all_datasets"]["MethylGPT_minus_elastic_net"]   # difference in MedAE, 95% CI resampling profiles within each tissue
        ax.text(23, y_, f"{d_['difference']:+.1f} ({d_['ci95'][0]:.1f} to {d_['ci95'][1]:.1f})".replace("-", "\u2212"), va="center", ha="left", fontsize=6, color=INK)
        ax.hlines(y_, bt_m[k]["medae"], bt_e[k]["medae"], color=GRAY["200"], lw=1.0)
        ax.scatter(bt_e[k]["medae"], y_, s=18, marker="s", facecolors="white", edgecolors=BLUE["400"], linewidths=0.9, zorder=3, clip_on=False)
        ax.scatter(bt_m[k]["medae"], y_, s=18, color=RUST["400"], zorder=4, clip_on=False)
    ax.set_yticks(yy); ax.set_yticklabels([f"{k.replace('Whole Blood', 'Whole blood')} ({bt_m[k]['n']:,})" for k in keys])
    ax.set_xscale("log")
    ax.set_xlabel("MedAE, held-out datasets (years)")
    ax.set_title("MethylGPT (filled), elastic net (open); right, difference (95% CI)", fontsize=7, fontweight="normal")

    # e: missing CpGs
    ax = axd["e"]
    mg, en = res["altumage_test"][MG], res["altumage_test"][EN]
    rr = [("all", 0), ("random_30", 30), ("random_50", 50), ("random_70", 70), ("random_90", 90)]
    xs = [x for _, x in rr]
    ax.plot(xs, [mg[s]["medae"] for s, _ in rr], "-o", ms=3, lw=1.0, color=RUST["400"], clip_on=False)
    ax.plot(xs, [en[s]["medae"] for s, _ in rr], "-s", ms=3, lw=1.0, color=BLUE["400"], clip_on=False)
    labels = [(xs[-1], mg["random_90"]["medae"] / 1.25, "MethylGPT", RUST["500"]), (xs[-1], en["random_90"]["medae"], "Elastic net,\nmeans filled", BLUE["500"])]
    fi = A / "eval/age_enet_factor_imputed_2026-10-02.json"
    if fi.exists():
        fj = json.loads(fi.read_text())["scenarios"]
        if all(s in fj for s, _ in rr):
            v = [fj[s]["enet_factor_filled"]["medae"] for s, _ in rr]
            ax.plot(xs, v, "--s", ms=3, lw=1.0, color=BLUE["300"], clip_on=False)
            labels.append((xs[-1], v[-1] * 1.6, "Elastic net,\nMethylGPT\nimputation", BLUE["400"]))
    plat = {"hm450": 100 * (1 - 47826 / 49156), "epic_v2": 100 * (1 - 45668 / 49156), "hm27": 100 * (1 - 22231 / 49156)}
    for s, x in plat.items():
        ax.scatter([x], [mg[s]["medae"]], s=20, marker="o", facecolors="white", edgecolors=RUST["400"], linewidths=0.9, zorder=4, clip_on=False)
        ax.scatter([x], [en[s]["medae"]], s=20, marker="s", facecolors="white", edgecolors=BLUE["400"], linewidths=0.9, zorder=4, clip_on=False)
    ax.text(plat["hm27"], en["hm27"]["medae"] * 1.15, "27K CpGs (open)", fontsize=6, ha="center", va="bottom", color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("CpGs removed (%)"); ax.set_ylabel("MedAE, AltumAge test (years)")
    ax._labels = labels

    # f: reprogramming time course
    ax = axd["f"]
    rp = pd.read_csv(A / "eval/reprogramming_gse54848_2026-10-01.csv").dropna(subset=["day"])
    ens = {d["gsm"]: d["ensemble"] for d in json.loads((A / "eval/age_ensemble_2026-10-04.json").read_text())["i_reprogramming_gse54848"]["per_sample"]}
    rp["MethylGPT-2"] = rp.gsm.map(ens)   # MethylGPT age = mean of the three training runs
    for col, lab, c, mk in (("Horvath 2013", "Horvath multi-tissue", GOLD["300"], "D"), ("Horvath skin & blood", "Horvath skin and blood", GOLD["500"], "s"),
                            ("MethylGPT-2", "MethylGPT", RUST["400"], "o")):
        m = rp.groupby("day")[col].mean()
        ax.plot(m.index, m.values, "-", lw=0.9, color=c, label=lab, clip_on=False)
        ax.scatter(rp.day, rp[col], s=9, marker=mk, color=c, edgecolors="none", alpha=0.85, clip_on=False, zorder=3)
    ax.set_xlabel("Days of reprogramming (GSE54848)"); ax.set_ylabel("Predicted age, reprogramming (years)")
    nature_legend(ax, loc="upper right")

    trim_axes(fig)
    axd["a"].set_xticks([1000, 2000, 5000, 10000, 18000]); axd["a"].set_xticklabels(["1,000", "2,000", "5,000", "10,000", "18,000"])
    set_domain(axd["a"], xlim=(950, 18500))
    axd["a"].axvline(8000, color=GRAY["300"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    axd["a"].text(8400, 2.75, "clock targets\nadded", fontsize=6, ha="left", va="bottom", color=INK)
    axd["b"].set_xticks([0, 40, 80, 120]); axd["b"].set_yticks([0, 40, 80, 120])
    set_domain(axd["b"], xlim=(-3, 120), ylim=(-3, 120))
    axd["d"].set_xticks([1, 2, 5, 10, 20]); axd["d"].set_xticklabels(["1", "2", "5", "10", "20"])
    set_domain(axd["d"], xlim=(1, 20))
    axd["d"].set_xlim(1, 80)
    axd["e"].set_yticks([1, 2, 5, 10, 20]); axd["e"].set_yticklabels(["1", "2", "5", "10", "20"])
    set_domain(axd["e"], ylim=(1, 22))
    axd["f"].set_yticks([0, 20, 40, 60]); set_domain(axd["f"], ylim=(-3, 62))
    for x_, y_, t_, c_ in axd["e"]._labels:
        axd["e"].text(x_ + 3, y_, t_, fontsize=6, ha="left", va="center", color=c_)
    pad_categorical_axis(axd["c"], "y", pad=0.5)
    pad_categorical_axis(axd["d"], "y", pad=0.5)
    add_panel_labels(fig, axd)
    fig.savefig(str(HERE / "figure4_2026-10-02_draft.png"), dpi=150)
    save_figure(fig, str(HERE / "figure4_2026-10-02.pdf"), axes_dict=axd, review_png=str(HERE / "figure4_2026-10-02_review.png"),
                footer="data: AltumAge test set, MethylGPT test-split datasets, GSE54848; artifacts/eval")
    plt.close(fig)


if __name__ == "__main__":
    main()
