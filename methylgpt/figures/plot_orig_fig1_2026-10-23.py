"""Original Fig. 1 of the MethylGPT submission, rebuilt for the revision (23 Oct 2026).

The original Illustrator page (figures/original_submission/fig1.pdf) is kept as the vector
base. A transparent overlay of the same page size is merged on top of it:
  a  schematic: only the labels of rebuilt model components are covered with
     background-coloured patches and reset.
  b  masked-value MSE against training step (log x) for the rebuilt encoder.
  c  unchanged (original vector artwork).
  d  observed against predicted beta, 1,000,000 held-out masked values, with marginals.
  e  residual against observed beta for the same values, coloured by point density.
  f  absolute error by observed methylation level (10 bins of 0.1) as box plots.

Data (all local):
  artifacts/eval/pretraining_curve_2026-10-18.json
  artifacts/eval/heldout_structured_2026-10-01.random30_pairs.npz
  artifacts/eval/heldout_structured_2026-10-01.json
  artifacts/eval/si_recon_diagnostics_2026-10-18.json
Outputs: figures/original_rebuild/fig1.pdf, fig1.png (300 dpi), fig1_side_by_side.png
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter

Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
ORIG_PDF = ROOT / "figures" / "original_submission" / "fig1.pdf"
OUT = ROOT / "figures" / "original_rebuild"

plt.rcParams.update({
    "font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "xtick.color": "#444444", "ytick.color": "#444444",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black", "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
})


def load_json(name):
    try:
        return json.loads((EVAL / name).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Cannot proceed - required data file not readable: {EVAL / name} ({exc})")


# ---------------------------------------------------------------- page geometry
reader = PdfReader(str(ORIG_PDF))
page0 = reader.pages[0]
W = float(page0.mediabox.width)
H = float(page0.mediabox.height)
S = 600 / 72  # px per pt of the reference render

with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(ORIG_PDF), f"{td}/o"], check=True)
    REF = np.asarray(Image.open(f"{td}/o.png").convert("RGB")).astype(float) / 255


def ref_px(x, y):
    return REF[int(round(y * S)), int(round(x * S))]


def text_color(x0, y0, x1, y1):
    """Darkest pixel inside a label's box in the original render (its ink colour)."""
    reg = REF[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)].reshape(-1, 3)
    return tuple(reg[np.argmin(reg.sum(1))])


fig = plt.figure(figsize=(W / 72, H / 72))
bg = fig.add_axes([0, 0, 1, 1], zorder=0)
bg.set_xlim(0, W); bg.set_ylim(H, 0); bg.axis("off"); bg.patch.set_alpha(0)


def axes_pt(x0, y0, x1, y1, **kw):
    """Axes from page coordinates in pt (origin top left)."""
    return fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)


def _bg_sample(x, y, direction, offsets=(0.5, 1.0, 1.6)):
    """Lightest of a few samples just outside a patch edge (the artwork background there;
    lines and text are darker than the background)."""
    cands = [ref_px(x + direction * d, y) for d in offsets]
    return max(cands, key=lambda c: c.sum())


def cover(x0, y0, x1, y1, z=1, step=0.4, side="both", offsets=(0.5, 1.0, 1.6)):
    """Background-coloured vector patch made of small cells. Each horizontal strip runs from the
    background colour just left of the patch to the colour just right of it, so gradients in the
    artwork continue across the patch."""
    nseg = max(1, int(round((x1 - x0) / 3.0)))
    xs = np.linspace(x0, x1, nseg + 1)
    for y in np.arange(y0, y1, step):
        yc = min(y + step / 2, y1 - 0.01)
        lc, rc = _bg_sample(x0, yc, -1, offsets), _bg_sample(x1, yc, +1, offsets)
        if side == "left":
            rc = lc
        elif side == "right":
            lc = rc
        hgt = min(step * 1.6, y1 - y)
        for i in range(nseg):
            t = (i + 0.5) / nseg
            col = tuple(np.clip(lc * (1 - t) + rc * t, 0, 1))
            wdt = (xs[i + 1] - xs[i]) * (1.25 if i < nseg - 1 else 1.0)
            bg.add_patch(Rectangle((xs[i], y), wdt, hgt, facecolor=col, edgecolor="none", lw=0, zorder=z))


def white(x0, y0, x1, y1):
    bg.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white", edgecolor="none", zorder=1))


def label(x, y, s, size, color="black", weight="normal", ha="center", va="center", ls=1.15):
    bg.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, linespacing=ls, zorder=5)


# ---------------------------------------------------------------- data
pc = load_json("pretraining_curve_2026-10-18.json")
st = load_json("heldout_structured_2026-10-01.json")
rd = load_json("si_recon_diagnostics_2026-10-18.json")
try:
    pz = np.load(EVAL / "heldout_structured_2026-10-01.random30_pairs.npz")
    obs = pz["observed"].astype(np.float64)
    pred = pz["methylgpt2"].astype(np.float64)
    knn = pz["knn10"].astype(np.float64)
    cmean = pz["cpg_mean"].astype(np.float64)
except (OSError, KeyError) as exc:
    sys.exit(f"Cannot proceed - required data file not readable: heldout_structured_2026-10-01.random30_pairs.npz ({exc})")

resid = pred - obs
r30 = st["mask_types"]["random_30"]["methylgpt2"]
slopes = {m: rd["residuals_S"]["methods"][m]["observed"]["ols_slope"] for m in ("methylgpt2", "knn10", "cpg_mean")}
stats = {
    "n_pairs": int(obs.size),
    "npz_mae": float(np.abs(resid).mean()),
    "npz_pearson_r": float(np.corrcoef(obs, pred)[0, 1]),
    "resid_mean": float(resid.mean()),
    "resid_sd": float(resid.std(ddof=1)),
    "resid_slope_npz": float(np.polyfit(obs, resid, 1)[0]),
    "frac_pred_outside_0_1": float(((pred < 0) | (pred > 1)).mean()),
    "frac_resid_outside_pm1": float((np.abs(resid) > 1).mean()),
    "full_mae": r30["mae"], "full_within_cpg_r2": r30["within_cpg_r2_pooled"], "full_n_targets": r30["n_targets"],
    "slopes_json": slopes,
}

# ---------------------------------------------------------------- panel a: schematic label edits
# data box: profiles that passed quality control in the rebuilt corpus (Results, first paragraph)
cover(9.5, 21.5, 111, 32.5)
label(60.5, 27.0, "169,010 Human DNAm Profiles", 6.2, weight="bold")
# data box: plural wording, reset in Arial (both lines, so that the two lines share one face)
cover(21.0, 139.5, 96.6, 158.2)
label(58.6, 143.5, "Diverse sources of samples", 6.9)
label(58.6, 153.2, "20+ tissue types", 6.9)
# features box: the vocabulary is the union of CpGs with published trait associations (EWAS Atlas) and the
# most widely measured CpGs (Methods, Vocabulary), not a set chosen for physiological relevance
cover(127.0, 63.5, 221.0, 81.5)
label(173.7, 67.6, "49,156 CpGs, trait-associated", 6.7, weight="bold")
label(173.7, 77.0, "or widely measured", 6.7, weight="bold")
# circle: 7.6 billion training tokens -> training profiles
cover(141.0, 123, 202.5, 147, offsets=(0.3, 0.6))
label(171.6, 130.7, "145,700", 7.3, weight="bold")
label(171.6, 139.4, "Training Profiles", 7.3, weight="bold")

# pretraining: methyl embedding layer -> probabilistic input layer
cover(23, 272, 74.3, 304.8)
label(48.6, 283.5, "Probabilistic\ninput layer", 6.7, weight="bold", ls=1.05)
label(48.6, 297.2, "1,536 latent\ndimensions", 5.4, ls=1.05)
cover(86, 249, 119.5, 266.5, side="left")
label(103.2, 257.8, "Predicted\nvalues", 6.0, ls=1.2)
cover(125, 284, 163, 299)
label(144, 291.4, "Encoder\nblocks", 6.7, weight="bold", ls=1.05)
cover(179.9, 287.0, 204, 297.5)
label(180.2, 291.2, "8 blocks", 6.0, ha="left")
cover(89.3, 316, 119.4, 333, side="left")
label(104.4, 324.4, "Input\ntokens", 6.0, ls=1.2)
label(146.0, 335.0, "values, residuals, mask\n+ latent scores", 5.0, color="#333333", ls=1.1)

# finetuning: ResNet -> output heads, embeddings, targets
cover(285.5, 141.0, 316.5, 166.3, offsets=(0.3, 0.6))      # 'Pretrained methylGPT Model'
label(300.6, 149.4, "Pretrained", 5.8, weight="bold")
label(300.6, 156.4, "MethylGPT", 5.8, weight="bold")
cover(313, 120.5, 358, 129.5)
label(335.3, 124.9, "2,048-d embedding", 6.0)
cover(358, 141, 384, 160)
label(371.0, 150.6, "Output\nheads", 6.7, weight="bold", ls=1.05)
cover(343.5, 174.8, 396.6, 188.0)
bg.add_patch(FancyBboxPatch((328.4, 176.7), 62.4, 9.9, boxstyle="round,pad=0,rounding_size=4.6",
                            facecolor="white", edgecolor="#4d4d4d", lw=1.1, zorder=4))
label(359.6, 181.7, "Age, sex, tissue, 25 clocks", 5.0)

# downstream: imputation from the input layer; disease models are Cox models (nested CV)
label(302.8, 226.3, "(input-layer\nreconstruction)", 5.0, color="#333333", ls=1.1)
cover(354, 302.2, 389.5, 313.2)
label(372.0, 307.6, "(Cox, nested CV)", 6.3)

# model details: encoder block (pre-norm) and probabilistic input layer
hdr_col = text_color(481, 65, 535, 74)
cover(478, 64, 541, 75.5)
label(510.4, 70.0, "Encoder block (×8)", 7.7, color=hdr_col, weight="bold")
for (y0, y1), new in zip([(92.3, 99.3), (103.3, 116.2), (128.0, 135.0), (139.2, 152.0)],
                         ["Feed-forward", "Layer norm", "Multihead attention", "Layer norm"]):
    white(481.6, y0 + 0.7, 537.2, y1 - 0.7)
    label(509.4, (y0 + y1) / 2, new, 5.2)

cover(446, 186, 575, 259.2)
label(510.8, 192.6, "Probabilistic input layer", 7.7, color=hdr_col, weight="bold")
lab_col = text_color(447, 230, 482, 248)
label(464.5, 240.5, "PPCA fitted\nto 145,700\nprofiles", 6.3, color=lab_col, ls=1.25)
# encoder input bar
bg.add_patch(FancyBboxPatch((490, 204.5), 80, 14.5, boxstyle="round,pad=0,rounding_size=1.5",
                            facecolor="#e6def0", edgecolor="#222222", lw=0.6, zorder=4))
label(530, 211.8, "to encoder: values, residuals,\nobservation mask, latent scores", 4.9, ls=1.1)
bg.add_patch(FancyArrowPatch((531, 226.0), (531, 219.6), arrowstyle="simple,head_width=3.5,head_length=2.6",
                             color="#b3b3b3", lw=0, zorder=4))
green = tuple(ref_px(492.7, 240))
bg.add_patch(Rectangle((492.6, 226.2), 78.4, 29.2, facecolor="white", edgecolor=green, lw=1.0, zorder=4))
label(531.8, 231.4, "residual = value − reconstruction", 4.9)
bg.add_patch(Rectangle((494.6, 237.0), 30.4, 15.5, facecolor="#f6d3d6", edgecolor="#f0b9be", lw=0.8, zorder=4))
label(509.8, 244.8, "Latent scores\n(1,536)", 5.0, ls=1.1)
bg.add_patch(Rectangle((530.6, 237.0), 39.0, 15.5, facecolor="#cfd9ef", edgecolor="#9fb2df", lw=0.8, zorder=4))
label(550.1, 244.8, "Reconstruction\n(49,156 CpGs)", 5.0, ls=1.1)
bg.add_patch(FancyArrowPatch((525.4, 244.8), (530.4, 244.8), arrowstyle="-|>,head_width=1.2,head_length=1.8",
                             color="#222222", lw=0.6, zorder=5))
bg.add_patch(FancyArrowPatch((511.5, 262.6), (511.5, 255.9), arrowstyle="simple,head_width=3.5,head_length=2.6",
                             color="#b3b3b3", lw=0, zorder=4))
# token row: no <START> token; CpG tokens -> CpGs
tok_col = text_color(452, 293, 480, 300)
cover(451, 292, 482, 301.5)
label(466.0, 296.6, "CpGs", 6.8, color=tok_col)
cover(492.2, 293.0, 505.0, 299.6)

# ---------------------------------------------------------------- white-out of the replaced data panels
white(18, 366, 300, 511)        # b
white(0, 392, 18, 500)
white(18, 512, 329.5, 832)      # d
white(329.5, 695, 337, 832)     # d marginal (keeps the original f letter)
white(314.5, 518, 595, 670)     # e
white(336.5, 670, 595, 679.5)   # e x label (keeps the original f letter)
white(336.5, 680, 595, 832)     # f

AX = "#444444"


def style(ax, out=6, minor=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", out))
    ax.patch.set_alpha(0)
    if minor:
        ax.minorticks_on()


# ---------------------------------------------------------------- b: reconstruction error by masking scenario
# Replaces the training curve (10 Nov 2026). Masked-value MSE on the 2,000 held-out test profiles (107 datasets):
# random masks (heldout_cf_dense_step2000_2026-09-30.json; the original model on 500 of these profiles),
# masks of held-out genes and of a whole chromosome (heldout_structured_2026-10-01.json) and conversion from the
# CpGs of one array to the full vocabulary (heldout_platform_2026-10-01.json).
HC = load_json("heldout_cf_dense_step2000_2026-09-30.json")
HS = load_json("heldout_structured_2026-10-01.json")
HP = load_json("heldout_platform_2026-10-01.json")
SCEN = [("15%", HC["rates"]["0.15"]), ("30%", HC["rates"]["0.3"]), ("50%", HC["rates"]["0.5"]), ("70%", HC["rates"]["0.7"]),
        ("Genes", HS["mask_types"]["gene_fold"]), ("Chrom.", HS["mask_types"]["chromosome"]),
        ("27K", HP["rates"]["keep_hm27"]), ("450K", HP["rates"]["keep_hm450"]), ("EPIC v2", HP["rates"]["keep_epic_v2"])]
XS = np.array([0, 1, 2, 3, 4.55, 6.0, 7.6, 8.75, 9.95])
# Since 10 Nov 2026 the manuscript figure shows only the final model. `--with-original-model` adds the originally
# submitted model and writes a separate figure.
P2P = "--with-original-model" in sys.argv
METH = [("cpg_mean", "CpG means", dict(marker="^", ms=5.2, mfc="#bdbdbd", mec="#555555")),
        ("january_mlm", "Original MethylGPT", dict(marker="s", ms=4.8, mfc="white", mec="#555555")),
        ("knn10", "Nearest neighbours", dict(marker="D", ms=4.6, mfc="white", mec="#6d4a5e")),
        ("methylgpt2", "MethylGPT", dict(marker="o", ms=5.6, mfc="#c0507a", mec="#3a1f2c"))]
if not P2P:
    METH = [m for m in METH if m[0] != "january_mlm"]
ax = axes_pt(66, 393, 291, 461, zorder=2)
fig1b = []
for key, lab, kw in METH:
    xs = [x for x, (_, r) in zip(XS, SCEN) if key in r]
    ys = [r[key]["mse"] * 1e3 for _, r in SCEN if key in r]
    ax.plot(xs, ys, ls="", mew=0.7, label=lab, zorder=3, clip_on=False, **kw)
    fig1b += [dict(scenario=n, method=key, mse=r[key]["mse"], n_profiles=r[key].get("n_profiles")) for n, r in SCEN if key in r]
ax.set_yscale("log")
ax.set_ylim(2.6, 45)
ax.set_yticks([3, 5, 10, 20, 40]); ax.set_yticklabels(["3", "5", "10", "20", "40"])
ax.minorticks_off()
ax.set_xlim(-0.5, 10.4)
ax.set_ylabel("MSE (\u00d710$^{-3}$)", fontsize=9, labelpad=3)
style(ax, out=4)
ax.set_xticks(XS); ax.set_xticklabels([n for n, _ in SCEN], fontsize=7.5)
ax.tick_params(axis="y", labelsize=9, length=4)
ax.tick_params(axis="x", length=3, pad=1)
ax.spines["bottom"].set_bounds(XS[0], XS[-1])
for x0, x1, lab in ((XS[0], XS[3], "Random"), (XS[4], XS[5], "Held out"), (XS[6], XS[8], "From one array")):
    ax.text((x0 + x1) / 2, -0.27, lab, transform=ax.get_xaxis_transform(), fontsize=7.5, ha="center", va="top", color="black")
leg = ax.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2, handletextpad=0.1, columnspacing=0.8,
                labelspacing=0.2, borderaxespad=0.0, bbox_to_anchor=(0.0, 0.98))
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(fig1b).to_csv(OUT / ("fig1b_reconstruction_by_scenario_with_original_model.csv" if P2P else "fig1b_reconstruction_by_scenario.csv"),
                           index=False)

# ---------------------------------------------------------------- d: joint density with marginals
PINK, PINK_E = "#e7bcd5", "#6d4a5e"
X0, X1, Y0, Y1 = 71.6, 264.4, 579.9, 774.9
ax = axes_pt(X0, Y0, X1, Y1, zorder=2)
pc_d = np.clip(pred, -1, 2)
ax.scatter(obs, pc_d, s=0.15, color=PINK, alpha=0.35, lw=0, rasterized=True)
ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=(0, (1, 1.6)))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.tick_params(labelsize=9, length=4); ax.tick_params(which="minor", length=2.2)
ax.minorticks_on()
style(ax, out=24)
ax.set_xlabel("Ground Truth DNA Methylation", fontsize=9, labelpad=2)
ax.set_ylabel("Predicted DNA Methylation", fontsize=9, labelpad=4)
ax.text(-0.02, 0.86, f"MAE: {r30['mae']:.3f}\nWithin-CpG $R^2$: {r30['within_cpg_r2_pooled']:.2f}",
        fontsize=8, ha="left", va="top", linespacing=1.15, transform=ax.transData)
bins = np.linspace(0, 1, 101)
axt = axes_pt(X0, 512, X1, 552, zorder=2)
axt.hist(obs, bins=bins, color=PINK, edgecolor=PINK_E, lw=0.3)
axt.set_xlim(0, 1); axt.set_xticks(np.arange(0, 1.01, 0.2))
axt.tick_params(labelsize=9, length=4); axt.tick_params(which="minor", length=2.2)
axt.minorticks_on()
style(axt, out=6)
axt.spines["left"].set_position(("outward", 24))
axr = axes_pt(291.4, Y0, 330, Y1, zorder=2)
axr.hist(np.clip(pred, 0, 1), bins=bins, orientation="horizontal", color=PINK, edgecolor=PINK_E, lw=0.3)
axr.set_ylim(0, 1); axr.set_yticks(np.arange(0, 1.01, 0.2))
axr.tick_params(labelsize=9, length=4); axr.tick_params(which="minor", length=2.2)
axr.minorticks_on()
style(axr, out=24)
axr.spines["left"].set_position(("outward", 6))
for a_ in (axt,):
    a_.set_yticks([0, 40000]); a_.set_yticklabels(["0", "40000"])
axr.set_xticks([0, 40000]); axr.set_xticklabels(["0", "40000"])

# ---------------------------------------------------------------- e: residuals coloured by density
ax = axes_pt(352.3, 534.9, 584.6, 650.2, zorder=2)
H2, xe, ye = np.histogram2d(obs, resid, bins=(200, 400), range=((0, 1), (-1, 1)))
ix = np.clip(np.searchsorted(xe, obs) - 1, 0, 199)
iy = np.clip(np.searchsorted(ye, resid) - 1, 0, 399)
dens = np.log10(H2[ix, iy] + 1)
order = np.argsort(dens)
# dense core dark purple, sparse points towards teal and yellow-green, as in the original
cm_e = matplotlib.colors.ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 0.85, 256))[::-1])
ax.scatter(obs[order], resid[order], c=dens[order], cmap=cm_e, vmin=0, vmax=np.percentile(dens, 15),
           s=0.25, lw=0, rasterized=True)
ax.axhline(0, color="black", lw=0.8, ls=(0, (3, 2)))
ax.set_xlim(0, 1); ax.set_ylim(-1, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.tick_params(labelsize=8, length=4); ax.tick_params(which="minor", length=2.2)
ax.minorticks_on()
style(ax, out=5)
ax.set_xlabel("Ground Truth DNA Methylation", fontsize=8.5, labelpad=2)
ax.set_ylabel("Residual (Predicted - Actual)", fontsize=7.5, labelpad=2)
ax.text(0.015, 0.97,
        f"Mean Residual: {stats['resid_mean']:.4f}\nStd Dev Residual: {stats['resid_sd']:.3f}\n"
        f"Slope vs ground truth: {slopes['methylgpt2']:.3f} (nearest neighbours {slopes['knn10']:.3f}, "
        f"CpG means {slopes['cpg_mean']:.3f})",
        fontsize=5.5, ha="left", va="top", transform=ax.transAxes, linespacing=1.15)

# ---------------------------------------------------------------- f: absolute error by methylation level
ax = axes_pt(372, 690, 582, 799.3, zorder=2)
BAR_COLS = ["#f1b8c3", "#f1b8c3", "#e496b8", "#e496b8", "#cd78b2", "#cd78b2", "#ac5fac", "#ac5fac", "#824b9f", "#824b9f"]
ae = np.abs(resid)
b_ = np.minimum((obs * 10).astype(int), 9)
bxs, mae_b, knn_b, cm_b = [], [], [], []
for i in range(10):
    s_ = b_ == i
    q05, q25, q50, q75, q95 = np.percentile(ae[s_], [5, 25, 50, 75, 95])
    bxs.append(dict(med=q50, q1=q25, q3=q75, whislo=q05, whishi=q95, fliers=[], label=""))
    mae_b.append(ae[s_].mean())
    knn_b.append(np.abs(knn[s_] - obs[s_]).mean())
    cm_b.append(np.abs(cmean[s_] - obs[s_]).mean())
stats["f_bins"] = [dict(bin=f"[{i/10:.1f},{(i+1)/10:.1f})", n=int((b_ == i).sum()), mae=float(m), knn_mae=float(k),
                        cpg_mean_mae=float(c), q05=float(x["whislo"]), q25=float(x["q1"]), median=float(x["med"]),
                        q75=float(x["q3"]), q95=float(x["whishi"])) for i, (m, k, c, x) in enumerate(zip(mae_b, knn_b, cm_b, bxs))]
pos = np.arange(10) / 10 + 0.05
bp = ax.bxp(bxs, positions=pos, widths=0.075, showfliers=False, patch_artist=True,
            boxprops=dict(lw=0.6, edgecolor="black"), medianprops=dict(color="black", lw=0.8),
            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), manage_ticks=False)
for p_, c_ in zip(bp["boxes"], BAR_COLS):
    p_.set_facecolor(c_)
ax.scatter(pos, cm_b, marker="^", s=9, facecolor="#bdbdbd", edgecolor="#555555", lw=0.4, zorder=4, label="CpG means")
ax.scatter(pos, knn_b, marker="D", s=7, facecolor="white", edgecolor="#555555", lw=0.5, zorder=4, label="Nearest neighbours")
ax.scatter(pos, mae_b, marker="o", s=9, facecolor="black", edgecolor="black", lw=0, zorder=5, label="MethylGPT")
for x_, m_ in zip(pos, mae_b):
    ax.text(x_, 0.214, f"{m_:.3f}", fontsize=5.5, ha="center", va="bottom")
ax.set_xlim(0, 1); ax.set_ylim(0, 0.226)
ax.set_xticks(np.arange(0, 1.01, 0.1)); ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.1)])
ax.set_yticks([0, 0.05, 0.10, 0.15]); ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
ax.tick_params(labelsize=6.5, length=3.5)
style(ax, out=6)
ax.set_xlabel("Methylation Level", fontsize=7, labelpad=2)
ax.set_ylabel("Absolute Error", fontsize=7, labelpad=2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=5.5, frameon=False, handletextpad=0.1,
          columnspacing=0.9, borderaxespad=0.1)
ax.text(-0.04, 0.214, "MAE", fontsize=5.5, ha="right", va="bottom", transform=ax.transData)

# ---------------------------------------------------------------- merge onto the original page
OUT.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
fig.savefig(buf, format="pdf", transparent=True, dpi=600)
plt.close(fig)
buf.seek(0)
over = PdfReader(buf).pages[0]
base = PdfReader(str(ORIG_PDF)).pages[0]
base.merge_page(over)
w = PdfWriter()
w.add_page(base)
w.add_metadata({"/Title": "MethylGPT Figure 1 (revision rebuild of the original figure)"})
STEM = "fig1_with_original_model" if P2P else "fig1"
pdf_out = OUT / f"{STEM}.pdf"
with open(pdf_out, "wb") as fh:
    w.write(fh)
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf_out), str(OUT / STEM)], check=True)
if P2P:
    print("wrote the figure with the original model", pdf_out)
    sys.exit(0)

# side by side: original page (left) and rebuilt page (right), same height
with tempfile.TemporaryDirectory() as td:
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(ORIG_PDF), f"{td}/orig"], check=True)
    a = Image.open(f"{td}/orig.png").convert("RGB")
b2 = Image.open(OUT / "fig1.png").convert("RGB")
gap = 60
sbs = Image.new("RGB", (a.width + b2.width + gap, max(a.height, b2.height)), "white")
sbs.paste(a, (0, 0)); sbs.paste(b2, (a.width + gap, 0))
sbs.save(OUT / "fig1_side_by_side.png", dpi=(300, 300))

print(json.dumps({k: v for k, v in stats.items() if k != "f_bins"}, indent=1))
for r in stats["f_bins"]:
    print(r)
print("wrote", pdf_out, OUT / "fig1.png", OUT / "fig1_side_by_side.png")
