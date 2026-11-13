"""Shorten the y tick label "Latent scores (input layer)" to "Latent scores" in Supplementary Information 2 Fig. 1a.

The label names the same representation as everywhere else in the paper (MethylGPT's latent scores), so the qualifier is
dropped without redrawing the figure. The label appears twice, once in each row of panel a. Each span is removed (text
only; no line art or image is touched) and the new label is written in the same font (Arial), size and colour on the same
baseline. These are y tick labels, which matplotlib right-aligns against the axis, so the new label ends where the old one
ended (same right edge of the advance box) instead of being centred on it. The PNG gets the same change from its own
pixels (relabel_si2fig1_png_2026-11-12.py).

Run: /opt/homebrew/bin/python3 figures/relabel_si2fig1_pdf_2026-11-12.py figures/si2_embedding_designs_2026-10-19_nofooter.pdf \
         figures/si2_embedding_designs_2026-11-12_relabelled.pdf
(the input is identical to the SI2 Fig. 1 PDF installed before this change)
"""
import sys
from pathlib import Path

import fitz

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
OLD, NEW, COUNT = "Latent scores (input layer)", "Latent scores", 2

src, dst = sys.argv[1], sys.argv[2]
for path in (src, ARIAL):
    if not Path(path).exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
doc = fitz.open(src)
page = doc[0]
font = fitz.Font(fontfile=ARIAL)
found = [(span, line["dir"]) for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])
         for span in line["spans"] if span["text"] == OLD]
if len(found) != COUNT or any(d != (1.0, 0.0) for _, d in found):
    sys.exit(f"expected {COUNT} horizontal spans {OLD!r}, found {len(found)}")

for span, _ in found:
    r = fitz.Rect(span["bbox"])
    page.add_redact_annot(fitz.Rect(r.x0 + 0.2, r.y0 + 0.6, r.x1 - 0.2, r.y1 - 0.6), fill=False)
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                      text=fitz.PDF_REDACT_TEXT_REMOVE)

page.insert_font(fontname="arialsi2", fontfile=ARIAL)
for span, _ in found:
    size, c = span["size"], span["color"]
    color = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
    ox, oy = span["origin"]
    x = ox + font.text_length(OLD, fontsize=size) - font.text_length(NEW, fontsize=size)   # same right edge
    page.insert_text(fitz.Point(x, oy), NEW, fontsize=size, fontname="arialsi2", color=color)
    print(f"{OLD!r} -> {NEW!r} at baseline y {oy:.2f}, x {ox:.2f} -> {x:.2f}")

doc.save(dst, garbage=3, deflate=True)
left = [s["text"] for b in fitz.open(dst)[0].get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
        if "input layer" in s["text"]]
if left:
    sys.exit(f"'input layer' still present: {left}")
print("written", dst)
