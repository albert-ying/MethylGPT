"""Supplementary Figure: masked-value pretraining curves of the released MethylGPT encoder.

Input (run from this directory): ../artifacts/eval/pretraining_curve_2026-10-18.json (pretraining_curve_2026-10-18.py),
which reads the original run (artifacts/runs/stopped_cf_dense_2026-09-30), its replay of steps 0-4,000 and its extension
from the 4,000-step checkpoint to 20,000 steps, all on the same 40,000-step schedule.
Run: MPLBACKEND=Agg python plot_supp_pretraining_curve_2026-10-18.py (and with NATURE_FIGURE_NOFOOTER=1)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain,
                                 save_figure, ref_line, tighten_to_data, RUST, GRAY, BLUE, INK)

HERE = Path(__file__).resolve().parent


def log_steps(ax, ticks):
    """Log-scaled training steps in thousands, with plain-number tick labels."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
A = HERE.parent / "artifacts"
retro_style()


def main():
    r = json.loads((A / "eval/pretraining_curve_2026-10-18.json").read_text())
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.275), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.5])
    axd = {"a": fig.add_subplot(gs[0]), "b": fig.add_subplot(gs[1]), "c": fig.add_subplot(gs[2])}

    # a: training batches (each point the mean of 100 steps): masked MSE of the encoder and of the cross-fitted input layer
    ax = axd["a"]
    tr = r["replay_train"] + [p for p in r["extension_train"] if p["step"] > 4000]
    st = np.array([p["step"] for p in tr]); mo = np.array([p["model"] for p in tr]); co = np.array([p["core"] for p in tr])
    ax.plot(st / 1e3, 1e3 * co, "-", lw=0.9, color=GRAY["400"], label="Input layer (cross-fitted)")
    ax.plot(st / 1e3, 1e3 * mo, "-", lw=0.9, color=RUST["400"], label="Encoder")
    ax.axvline(2, color=GRAY["300"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    log_steps(ax, [0.1, 0.3, 1, 2, 4, 10, 20])
    ax.set_xlabel("Training step (thousands, log scale)"); ax.set_ylabel("Masked MSE, training\nbatches (×10$^{-3}$)")
    ax.set_title("Training batches (10–70% masked)", fontsize=7, fontweight="normal")
    tail = st > 0.6 * st.max()
    ax.text(st.max() / 1e3, 1e3 * co[tail].max(), "Input layer (cross-fitted)", ha="right", va="bottom", fontsize=6, color=GRAY["500"])
    ax.text(st.max() / 1e3, 1e3 * mo[tail].min(), "Encoder", ha="right", va="top", fontsize=6, color=RUST["500"])

    # b: development set: encoder minus input layer, with 95% intervals resampling development datasets
    ax = axd["b"]
    for key, filled, lab in (("replay_dev", False, "Replay"), ("extension_dev", True, "Continuation")):
        pts = r[key]
        s_ = np.array([p["step"] for p in pts]); d = np.array([p["model_minus_core"] for p in pts])
        lo = np.array([p["model_minus_core_ci95"][0] for p in pts]); hi = np.array([p["model_minus_core_ci95"][1] for p in pts])
        ax.errorbar(s_ / 1e3, 1e6 * d, yerr=[1e6 * (d - lo), 1e6 * (hi - d)], fmt="o", ms=3, color=RUST["400"], ecolor=RUST["300"],
                    elinewidth=0.7, capsize=0, mfc=RUST["400"] if filled else "white", mec=RUST["400"], label=lab, clip_on=False, zorder=3)
    for s, v in ((p["step"], p) for p in r["original_dev"]):
        ax.scatter([s / 1e3], [1e6 * v["model_minus_core"]], marker="D", s=22, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4,
                   label="Original run" if s == 2000 else None, clip_on=False)
    ax.axvline(2, color=GRAY["300"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    log_steps(ax, [0.25, 1, 2, 4, 10, 20])
    ax.set_xlabel("Training step (thousands, log scale)"); ax.set_ylabel("Encoder minus input layer,\ndevelopment MSE (×10$^{-6}$)")
    n_ = r["replay_dev"][0]
    ax.set_title(f"Development ({n_['n_profiles']} profiles)", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="lower right")

    # c: held-out reconstruction (2,000 profiles, 107 test datasets): later checkpoints minus the released 2,000-step checkpoint
    ax = axd["c"]
    h = r["heldout"]
    rates = [k for k in h if k not in ("n_profiles", "n_datasets", "status", "missing")]
    x = np.arange(len(rates))
    for s_, off, col in (("4000", -0.12, RUST["300"]), ("20000", 0.12, RUST["500"])):
        if s_ not in h[rates[0]]:
            continue
        d = np.array([h[k][s_]["minus_2000"] for k in rates]); lo = np.array([h[k][s_]["minus_2000_ci95"][0] for k in rates])
        hi = np.array([h[k][s_]["minus_2000_ci95"][1] for k in rates])
        ax.errorbar(x + off, 1e6 * d, yerr=[1e6 * (d - lo), 1e6 * (hi - d)], fmt="o", ms=3, color=col, ecolor=col, elinewidth=0.7,
                    capsize=0, label=f"Step {int(s_):,}", clip_on=False, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(100 * float(k))}%" for k in rates])
    ax.set_xlabel("CpGs masked"); ax.set_ylabel("Held-out MSE minus\nstep-2,000 checkpoint (×10$^{-6}$)")
    ax.set_title(f"Held-out ({h['n_profiles']:,} profiles)", fontsize=7, fontweight="normal")
    nature_legend(ax, loc="upper left")

    trim_axes(fig)
    tighten_to_data(fig)
    set_domain(axd["c"], xlim=(-0.45, len(rates) - 0.55))
    for k in ("b", "c"):
        ref_line(axd[k], y=0)
    add_panel_labels(fig, axd)
    save_figure(fig, str(HERE / "supplementary_pretraining_curve_2026-10-18.pdf"), axes_dict=axd,
                review_png=str(HERE / "supplementary_pretraining_curve_2026-10-18_review.png"),
                footer="data: MethylGPT corpus development and test splits; artifacts/eval/pretraining_curve_2026-10-18.json")


if __name__ == "__main__":
    main()
