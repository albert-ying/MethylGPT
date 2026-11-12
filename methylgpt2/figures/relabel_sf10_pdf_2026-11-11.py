"""Replace the working labels "set A" and "set B" in Supplementary Fig. 10 without redrawing it.

The three inputs of plot_si_sample_embedding_2026-10-18.py are on Chronos only, so the published vector PDF is edited
in place: each of the four label spans is removed (text only; no line art or image is touched) and the new label is
written in the same font (Arial), size and colour, centred on the old label's centre. The panel legends in the
Supplementary Information already state that panels d and e are the six-tissue panel, so those axis labels drop the
qualifier, and the panel g title becomes "Whole blood". The PNG keeps the same labels (relabel_sf10_png_2026-11-11.py).

Run: /opt/homebrew/bin/python3 figures/relabel_sf10_pdf_2026-11-11.py <in.pdf> <out.pdf>
"""
import sys

import fitz

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
NEW = {
    "Tissue 10-NN balanced accuracy (set A)": "Tissue 10-NN balanced accuracy",
    "Sex 10-NN balanced accuracy (set A)": "Sex 10-NN balanced accuracy",
    "Adjusted Rand index (set A)": "Adjusted Rand index",
    "Whole blood, set B": "Whole blood",
}

src, dst = sys.argv[1], sys.argv[2]
doc = fitz.open(src)
page = doc[0]
font = fitz.Font(fontfile=ARIAL)
found = []
for block in page.get_text("dict")["blocks"]:
    for line in block.get("lines", []):
        for span in line["spans"]:
            if span["text"] in NEW:
                found.append((span, line["dir"]))
if sorted(s["text"] for s, _ in found) != sorted(NEW):
    sys.exit(f"expected the four labels once each, found {[s['text'] for s, _ in found]}")

for span, _ in found:
    r = fitz.Rect(span["bbox"])
    page.add_redact_annot(fitz.Rect(r.x0 + 0.2, r.y0 + 0.6, r.x1 - 0.2, r.y1 - 0.6), fill=False)
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                      text=fitz.PDF_REDACT_TEXT_REMOVE)

page.insert_font(fontname="arialsf10", fontfile=ARIAL)
for span, direction in found:
    text, size = NEW[span["text"]], span["size"]
    c = span["color"]
    color = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)
    width = font.text_length(text, fontsize=size)
    r = fitz.Rect(span["bbox"])
    ox, oy = span["origin"]
    if direction == (1.0, 0.0):                         # horizontal: same baseline, centred on the old label
        point, rotate = fitz.Point((r.x0 + r.x1) / 2 - width / 2, oy), 0
    elif direction == (0.0, -1.0):                      # reading upwards (y label): same baseline x, centred in y
        point, rotate = fitz.Point(ox, (r.y0 + r.y1) / 2 + width / 2), 90
    else:
        sys.exit(f"unexpected text direction {direction}")
    page.insert_text(point, text, fontsize=size, fontname="arialsf10", color=color, rotate=rotate)
    print(f"{span['text']!r} -> {text!r}")

doc.save(dst, garbage=3, deflate=True)
print("written", dst)
