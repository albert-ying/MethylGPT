"""Supplementary Figure 6: length of masked-value pretraining of the released MethylGPT encoder (two panels).

Redraw of plot_supp_pretraining_curve_2026-10-18.py without its panel b (development MSE of the encoder minus that of
the input layer), which compared the two parts of the model. Panel a (training batches) is unchanged and the former
panel c (held-out MSE of later checkpoints minus the released checkpoint) becomes panel b, drawn from the same values.

Input (run from this directory): ../artifacts/eval/pretraining_curve_2026-10-18.json (pretraining_curve_2026-10-18.py),
which reads the original run (artifacts/runs/stopped_cf_dense_2026-09-30), its replay of steps 0-4,000 and its extension
from the 4,000-step checkpoint to 20,000 steps, all on the same 40,000-step schedule.
Output: supplementary_pretraining_curve_2026-11-12.pdf (gated by the nature-figure preflight, no footer) and a 300-dpi
PNG of the same figure. Pass --no-input-layer to write the variant whose panel a shows only the encoder's curve
(supplementary_pretraining_curve_2026-11-12_encoder_only.pdf/.png).
Run: MPLBACKEND=Agg /opt/homebrew/bin/python3 plot_supp_pretraining_curve_2026-11-12.py [--no-input-layer]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "nature-figure")   # the skill's current preflight
from nature_figure_utils import (retro_style, figsize_nature, trim_axes, add_panel_labels, nature_legend, set_domain,  # noqa: E402
                                 save_figure, ref_line, tighten_to_data, allow_device, RUST, GRAY)

A = HERE.parent / "artifacts"
SRC = A / "eval/pretraining_curve_2026-10-18.json"
retro_style()


def log_steps(ax, ticks):
    """Log-scaled training steps in thousands, with plain-number tick labels."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def main(with_input_layer=True):
    if not SRC.exists():
        sys.exit(f"Cannot proceed - required data file not found: {SRC}")
    r = json.loads(SRC.read_text())
    fig = plt.figure(figsize=figsize_nature("double", aspect=0.29), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.015)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    axd = {"a": fig.add_subplot(gs[0]), "b": fig.add_subplot(gs[1])}

    # a: training batches (each point the mean of 100 steps): masked MSE of the encoder and of the cross-fitted input layer
    ax = axd["a"]
    tr = r["replay_train"] + [p for p in r["extension_train"] if p["step"] > 4000]
    st = np.array([p["step"] for p in tr]); mo = np.array([p["model"] for p in tr]); co = np.array([p["core"] for p in tr])
    if with_input_layer:
        ax.plot(st / 1e3, 1e3 * co, "-", lw=0.9, color=GRAY["400"], label="Input layer (cross-fitted)")
    ax.plot(st / 1e3, 1e3 * mo, "-", lw=0.9, color=RUST["400"], label="Encoder")
    ax.axvline(2, color=GRAY["300"], lw=0.6, ls=(0, (2, 2)), zorder=0)
    log_steps(ax, [0.1, 0.3, 1, 2, 4, 10, 20])
    ax.set_xlabel("Training step (thousands, log scale)"); ax.set_ylabel("Masked MSE, training\nbatches (×10$^{-3}$)")
    ax.set_title("Training batches (10–70% masked)", fontsize=7, fontweight="normal")
    tail = st > 0.6 * st.max()
    if with_input_layer:
        ax.text(st.max() / 1e3, 1e3 * co[tail].max(), "Input layer (cross-fitted)", ha="right", va="bottom", fontsize=6,
                color=GRAY["500"])
        allow_device(ax, "grey-secondary-text",
                     reason="direct label of the grey input-layer curve in that curve's colour, as in the earlier figure")
        ax.text(st.max() / 1e3, 1e3 * mo[tail].min(), "Encoder", ha="right", va="top", fontsize=6, color=RUST["500"])

    # b (formerly c): held-out reconstruction (2,000 profiles, 107 test datasets): later checkpoints minus the released one
    ax = axd["b"]
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
    # right end close to the last points so the canvas carries no blank strip beyond the trimmed spine (the earlier
    # figure filled it with its provenance footer)
    set_domain(axd["b"], xlim=(-0.3, len(rates) - 0.75))
    ref_line(axd["b"], y=0)
    add_panel_labels(fig, axd)
    stem = "supplementary_pretraining_curve_2026-11-12" + ("" if with_input_layer else "_encoder_only")
    save_figure(fig, str(HERE / f"{stem}.pdf"), axes_dict=axd, review_png=str(HERE / f"{stem}_review.png"))
    fig.savefig(str(HERE / f"{stem}.png"), dpi=300, bbox_inches="tight")
    print("wrote", HERE / f"{stem}.pdf", "and .png")
    plt.close(fig)


if __name__ == "__main__":
    main(with_input_layer="--no-input-layer" not in sys.argv[1:])
