"""Carry the relabelled Supplementary Fig. 10 labels into its PNG, which the Supplementary Information embeds.

Each new label is the leading part of the old one ("Tissue 10-NN balanced accuracy", "Sex 10-NN balanced accuracy",
"Adjusted Rand index", "Whole blood"), so it is cut from the PNG's own pixels: the label box is cleared to white and the
kept part is pasted back, recentred on the old label's centre. No pixel outside the four label boxes changes, and the
text keeps the exact weight of the rest of the figure. Each box must have a white border, so no marker or line is in
it; the cut is placed in a white gap between words, located from the PDF character positions.

Run: /opt/homebrew/bin/python3 figures/relabel_sf10_png_2026-11-11.py <old.png> <old.pdf> <out.png>
"""
import sys

import fitz
import numpy as np
from PIL import Image

old_png, old_pdf, out = sys.argv[1:4]
KEEP = {"Tissue 10-NN balanced accuracy (set A)": "Tissue 10-NN balanced accuracy",
        "Sex 10-NN balanced accuracy (set A)": "Sex 10-NN balanced accuracy",
        "Adjusted Rand index (set A)": "Adjusted Rand index",
        "Whole blood, set B": "Whole blood"}
S, PAD = 300 / 72, 5
img = np.asarray(Image.open(old_png).convert("RGB")).copy()
gray = img.mean(2)
res = img.copy()
found = 0
for block in fitz.open(old_pdf)[0].get_text("rawdict")["blocks"]:
    for line in block.get("lines", []):
        for sp in line["spans"]:
            text = "".join(c["c"] for c in sp["chars"])
            if text not in KEEP:
                continue
            found += 1
            keep = KEEP[text]
            assert text.startswith(keep)
            # PDF position of the gap between the kept part and the rest: end of the last kept glyph, start of the next
            # visible glyph after it
            chars = sp["chars"]
            last = chars[len(keep) - 1]["bbox"]
            nxt = next(c["bbox"] for c in chars[len(keep):] if c["c"].strip())
            x0, y0, x1, y1 = sp["bbox"]
            by0, by1, bx0, bx1 = int(y0 * S) - PAD, int(y1 * S) + PAD, max(int(x0 * S) - PAD, 0), int(x1 * S) + PAD
            box = gray[by0:by1, bx0:bx1]
            border = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
            if border.min() < 235:
                sys.exit(f"{text!r}: label box border is not white; stop")
            if line["dir"] == (1.0, 0.0):                       # horizontal; text runs left to right
                ink_cols = np.nonzero((box < 200).any(0))[0]
                lo, hi = ink_cols.min(), ink_cols.max()
                gap0, gap1 = int(last[2] * S) - bx0, int(nxt[0] * S) - bx0
                white = [c for c in range(gap0 - 3, gap1 + 4) if not (box[:, c] < 235).any()]
                if not white:
                    sys.exit(f"{text!r}: no white column between the kept text and the rest; stop")
                cut = white[0]                                  # first white column after the kept text
                kept = img[by0:by1, bx0 + lo:bx0 + cut]
                centre = (lo + hi) / 2
                new_lo = int(round(centre - kept.shape[1] / 2))
                res[by0:by1, bx0:bx1] = 255
                res[by0:by1, bx0 + new_lo:bx0 + new_lo + kept.shape[1]] = kept
            elif line["dir"] == (0.0, -1.0):                    # reads upwards; text runs bottom to top
                ink_rows = np.nonzero((box < 200).any(1))[0]
                lo, hi = ink_rows.min(), ink_rows.max()
                gap_bottom, gap_top = int(last[1] * S) - by0, int(nxt[3] * S) - by0   # last kept glyph top, next glyph bottom
                white = [r for r in range(gap_bottom + 3, gap_top - 4, -1) if not (box[r] < 235).any()]
                if not white:
                    sys.exit(f"{text!r}: no white row between the kept text and the rest; stop")
                cut = white[0]                                  # first white row above the kept text
                kept = img[by0 + cut + 1:by0 + hi + 1, bx0:bx1]
                centre = (lo + hi) / 2
                new_lo = int(round(centre - kept.shape[0] / 2))
                res[by0:by1, bx0:bx1] = 255
                res[by0 + new_lo:by0 + new_lo + kept.shape[0], bx0:bx1] = kept
            else:
                sys.exit(f"unexpected direction {line['dir']}")
            print(f"{text!r} -> {keep!r}: kept {kept.shape[1] if line['dir'] == (1.0, 0.0) else kept.shape[0]} px, recentred")
if found != len(KEEP):
    sys.exit(f"found {found} of {len(KEEP)} labels; stop")
mask = np.zeros(gray.shape, bool)
for block in fitz.open(old_pdf)[0].get_text("dict")["blocks"]:
    for line in block.get("lines", []):
        for sp in line["spans"]:
            if sp["text"] in KEEP:
                x0, y0, x1, y1 = sp["bbox"]
                mask[int(y0 * S) - PAD:int(y1 * S) + PAD, max(int(x0 * S) - PAD, 0):int(x1 * S) + PAD] = True
print("pixels changed outside the label boxes:", int(((res != img).any(2) & ~mask).sum()))
Image.fromarray(res).save(out, dpi=(300, 300))
print("written", out)
