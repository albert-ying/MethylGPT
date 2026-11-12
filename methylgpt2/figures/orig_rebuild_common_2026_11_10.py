"""Shared helpers for the figures rebuilt on the submitted artwork (original Figs 2, 3, 5, 6; 10 November 2026).

The original PDF page is the template: retained regions are the original embedded images at their page
positions; replaced panels are redrawn in the look of the submitted panels (grey offset spines trimmed to
the tick range, Arial, outlined point clouds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz  # PyMuPDF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, FixedLocator
from PIL import Image

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
ORIG = HERE / "original_submission"
OUT = HERE / "original_rebuild"
PW = 595.276
INK = "#231f20"
SPINE = "#717171"

plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
                     "svg.fonttype": "none"})


def need(p):
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"Cannot proceed - required data file not found: {p}")
    return p


def read_csv(p, **kw):
    return pd.read_csv(need(p), **kw)


def original_images(n):
    """Embedded images of original figure n: list of (xref, array, bbox in page points)."""
    doc = fitz.open(need(ORIG / f"fig{n}.pdf"))
    out = []
    for info in doc[0].get_image_info(xrefs=True):
        pix = fitz.Pixmap(doc, info["xref"])
        if pix.n - pix.alpha > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
        out.append((info["xref"], arr, tuple(info["bbox"])))
    return out


class Page:
    """A figure whose coordinates are the original page points (top-left origin), cropped to height ph."""

    def __init__(self, ph):
        self.ph = ph
        self.fig = plt.figure(figsize=(PW / 72, ph / 72))

    def rect(self, x0, y0, x1, y1):
        return [x0 / PW, 1 - y1 / self.ph, (x1 - x0) / PW, (y1 - y0) / self.ph]

    def place(self, arr, bbox, z=1):
        ax = self.fig.add_axes(self.rect(*bbox), zorder=z)
        ax.imshow(arr, interpolation="none", aspect="auto", extent=(0, 1, 0, 1))
        ax.set_axis_off()
        return ax

    def axes(self, x0, y0, x1, y1, z=3):
        ax = self.fig.add_axes(self.rect(x0, y0, x1, y1), zorder=z)
        ax.patch.set_alpha(0)
        return ax

    def text(self, x, y_top, s, size, bold=False, color=INK, ha="left", **kw):
        return self.fig.text(x / PW, 1 - (y_top + 0.905 * size) / self.ph, s, fontsize=size,
                             fontweight="bold" if bold else "normal", color=color, ha=ha, va="baseline", **kw)

    def letter(self, x, y_top, s):
        self.text(x, y_top, s, 12, bold=True)

    def save(self, stem, orig_n=None):
        OUT.mkdir(exist_ok=True)
        self.fig.savefig(OUT / f"{stem}.pdf")
        self.fig.savefig(OUT / f"{stem}.png", dpi=300)
        plt.close(self.fig)
        print(f"wrote {OUT / stem}.pdf/.png", file=sys.stderr)
        if orig_n is not None:
            pix = fitz.open(ORIG / f"fig{orig_n}.pdf")[0].get_pixmap(dpi=300, clip=fitz.Rect(0, 0, PW, self.ph))
            left = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            right = Image.open(OUT / f"{stem}.png").convert("RGB")
            left = left.resize((round(left.width * right.height / left.height), right.height), Image.LANCZOS)
            sbs = Image.new("RGB", (left.width + 60 + right.width, right.height), "white")
            sbs.paste(left, (0, 0))
            sbs.paste(right, (left.width + 60, 0))
            sbs.save(OUT / f"{stem}_side_by_side.png")


def style(ax, fs_tick=7.5, fs_label=8.5, offset=4, minor=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", offset))
        ax.spines[s].set_color(SPINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(which="major", length=3.5, width=0.8, color=SPINE, labelsize=fs_tick, labelcolor="black", pad=2)
    ax.tick_params(which="minor", length=1.8, width=0.5, color=SPINE)
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    else:
        ax.minorticks_off()
    ax.xaxis.label.set_size(fs_label)
    ax.yaxis.label.set_size(fs_label)


def trim(ax, x=True, y=True):
    for axis, on, sp, lim in ((ax.xaxis, x, "bottom", ax.get_xlim()), (ax.yaxis, y, "left", ax.get_ylim())):
        if not on:
            continue
        lo, hi = min(lim), max(lim)
        t = [v for v in axis.get_majorticklocs() if lo - 1e-9 <= v <= hi + 1e-9]
        ax.spines[sp].set_bounds(min(t), max(t))
        axis.set_minor_locator(FixedLocator([v for v in axis.get_minorticklocs() if min(t) - 1e-9 <= v <= max(t) + 1e-9]))


def outlined_cloud(ax, x, y, colors, s=2.2, order=None, rng_seed=0):
    """Point cloud in the look of the submitted UMAP panels: dark outline, white body, coloured dots."""
    x, y = np.asarray(x), np.asarray(y)
    ax.scatter(x, y, s=s * 5.5, c="#1a1a1a", linewidths=0, rasterized=True, zorder=1)
    ax.scatter(x, y, s=s * 3.2, c="white", linewidths=0, rasterized=True, zorder=2)
    idx = np.random.default_rng(rng_seed).permutation(len(x)) if order is None else order
    ax.scatter(x[idx], y[idx], s=s, c=np.asarray(colors, dtype=object)[idx].tolist(), linewidths=0,
               rasterized=True, zorder=3)


def dot_legend(page, x, y_top, title, entries, size=7.5, dy=10.5, title_gap=11.5):
    """Legend in page points: title then coloured dots with labels, as in the submitted panels."""
    page.text(x, y_top, title, size)
    for i, (lab, col) in enumerate(entries):
        yy = y_top + title_gap + i * dy
        page.fig.add_artist(matplotlib.lines.Line2D([(x + 4) / PW], [1 - (yy + size * 0.55) / page.ph], marker="o",
                                                    ms=3.2, mfc=col, mec="none", ls="none",
                                                    transform=page.fig.transFigure))
        page.text(x + 13, yy, lab, size)
