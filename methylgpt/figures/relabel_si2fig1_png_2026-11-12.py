"""Carry the shortened Supplementary Information 2 Fig. 1a tick label into the PNG that the Supplementary Information embeds.

"Latent scores" is the leading part of the old label "Latent scores (input layer)", so it is cut from the PNG's own
pixels: the label box is cleared to white and the kept part is pasted back, shifted right so that it ends where the old
label ended, as matplotlib right-aligns y tick labels (the same shift as the vector PDF, relabel_si2fig1_pdf_2026-11-12.py:
the difference in Arial advance width between the old and the new label). No pixel outside the two label boxes changes,
and the text keeps the exact weight of the rest of the figure. Each box must have a white border, so no marker or line is
in it; the cut is placed in a white gap between words, located from the PDF character positions.

Run: /opt/homebrew/bin/python3 figures/relabel_si2fig1_png_2026-11-12.py figures/si2_embedding_designs_2026-10-19_nofooter.png \
         figures/si2_embedding_designs_2026-10-19_nofooter.pdf figures/si2_embedding_designs_2026-11-12_relabelled.png
(the inputs are identical to the SI2 Fig. 1 files installed before this change)
"""
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
OLD, NEW, COUNT = "Latent scores (input layer)", "Latent scores", 2
old_png, old_pdf, out = sys.argv[1:4]
for path in (old_png, old_pdf, ARIAL):
    if not Path(path).exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
S, PAD = 300 / 72, 5
font = fitz.Font(fontfile=ARIAL)
page = fitz.open(old_pdf)[0]
img = np.asarray(Image.open(old_png).convert("RGB")).copy()
if abs(img.shape[1] - page.rect.width * S) > 2 or abs(img.shape[0] - page.rect.height * S) > 2:
    sys.exit(f"PNG {img.shape[1]}x{img.shape[0]} does not match the PDF page at 300 dpi; stop")
gray = img.mean(2)
res = img.copy()
mask = np.zeros(gray.shape, bool)
found = 0
for block in page.get_text("rawdict")["blocks"]:
    for line in block.get("lines", []):
        for sp in line["spans"]:
            text = "".join(c["c"] for c in sp["chars"])
            if text != OLD:
                continue
            if line["dir"] != (1.0, 0.0):
                sys.exit(f"unexpected direction {line['dir']}")
            found += 1
            chars = sp["chars"]
            last = chars[len(NEW) - 1]["bbox"]                                  # end of "scores"
            nxt = next(c["bbox"] for c in chars[len(NEW):] if c["c"].strip())  # start of "("
            x0, y0, x1, y1 = sp["bbox"]
            by0, by1, bx0, bx1 = int(y0 * S) - PAD, int(y1 * S) + PAD, max(int(x0 * S) - PAD, 0), int(x1 * S) + PAD
            box = gray[by0:by1, bx0:bx1]
            border = np.concatenate([box[0], box[-1], box[:, 0], box[:, -1]])
            if border.min() < 235:
                sys.exit(f"label box at y {y0:.1f} pt: border is not white; stop")
            ink_cols = np.nonzero((box < 200).any(0))[0]
            lo = ink_cols.min()
            gap0, gap1 = int(last[2] * S) - bx0, int(nxt[0] * S) - bx0
            white = [c for c in range(gap0 - 3, gap1 + 4) if not (box[:, c] < 235).any()]
            if not white:
                sys.exit("no white column between 'scores' and '('; stop")
            cut = white[0]                                                      # first white column after "scores"
            kept = img[by0:by1, bx0 + lo:bx0 + cut]
            shift = int(round((font.text_length(OLD, fontsize=sp["size"]) - font.text_length(NEW, fontsize=sp["size"])) * S))
            new_lo = lo + shift
            if new_lo + kept.shape[1] > box.shape[1]:
                sys.exit("shifted label leaves its box; stop")
            res[by0:by1, bx0:bx1] = 255
            res[by0:by1, bx0 + new_lo:bx0 + new_lo + kept.shape[1]] = kept
            mask[by0:by1, bx0:bx1] = True
            old_right = bx0 + ink_cols.max()
            print(f"label at y {y0:.1f} pt: kept {kept.shape[1]} px, shifted right {shift} px; "
                  f"old ink right edge {old_right} px, new {bx0 + new_lo + kept.shape[1] - 1} px")
if found != COUNT:
    sys.exit(f"found {found} of {COUNT} labels; stop")
print("pixels changed outside the label boxes:", int(((res != img).any(2) & ~mask).sum()))
Image.fromarray(res).save(out, dpi=(300, 300))
print("written", out)
