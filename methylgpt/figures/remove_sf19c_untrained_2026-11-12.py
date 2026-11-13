"""Remove the "Untrained transformer" series from Supplementary Fig. 19c without redrawing the figure.

Panel c plots tissue silhouette against sex balanced accuracy in the six-tissue panel for six representations. The
"Untrained transformer" point (the embedding of a transformer at random weights on the fitted input layer) compares parts
of MethylGPT, so it is removed; the other five points (raw beta values, their 50 principal components, latent scores, the
MethylGPT embedding and the random projection) are kept as they are.

The figure cannot be redrawn here: the UMAP coordinates of panels a and b come from
artifacts/eval/representations_setA_2026-10-01.npz and the cache figures/figure3_umap_coords_2026-10-02.npz, neither of
which is on this machine. Panel c's values are local (artifacts/eval/representation_benchmark_2026-10-01.json), so the
script first checks every plotted panel-c marker of the installed vector PDF against those values, then edits the PDF:
the untrained-transformer marker, its legend marker and its legend label are deleted, and the "Random projection" legend
row is moved up one row (its label rewritten in the same font, size and colour, its marker redrawn with the same
geometry and fill). No other object is touched. The installed PNG was rendered from this PDF with pdftoppm at 300 dpi
(strip_provenance_footer_2026-11-10.py), so the new PNG is rendered the same way and compared pixel by pixel with the old.

Run (from the project folder; the inputs are the figure as installed before this change, kept as
figures/supplementary_figure19_installed_2026-11-10.*):
  /opt/homebrew/bin/python3 figures/remove_sf19c_untrained_2026-11-12.py figures/supplementary_figure19_installed_2026-11-10.pdf \
      figures/supplementary_figure19_installed_2026-11-10.png figures/supplementary_figure19_2026-11-12.pdf \
      figures/supplementary_figure19_2026-11-12.png
"""
import json
import subprocess
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "artifacts/eval/representation_benchmark_2026-10-01.json"
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
old_pdf, old_png, out_pdf, out_png = sys.argv[1:5]
for path in (SRC, old_pdf, old_png, ARIAL):
    if not Path(path).exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")

vals = json.loads(SRC.read_text())["natural"]["set_a"]
KEYS = ["raw", "raw_pca50", "factor_scores", "mgpt2_multitask", "untrained_transformer", "random_projection"]
doc = fitz.open(old_pdf)
page = doc[0]
spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]


def one(text):
    hits = [s for s in spans if s["text"] == text]
    if len(hits) != 1:
        sys.exit(f"expected one span {text!r}, found {len(hits)}")
    return hits[0]


# axis calibration from the panel-c tick labels (x: tissue silhouette, y: sex balanced accuracy)
cx = lambda s: (s["bbox"][0] + s["bbox"][2]) / 2
cy = lambda s: (s["bbox"][1] + s["bbox"][3]) / 2
x1, x2 = one("0.325"), one("0.450")
y1, y2 = one("0.87"), one("0.95")
fx = lambda v: cx(x1) + (v - 0.325) * (cx(x2) - cx(x1)) / 0.125
fy = lambda v: cy(y1) + (v - 0.87) * (cy(y2) - cy(y1)) / 0.08

# every filled marker inside panel c's data area, outside its legend box
leg = fitz.Rect(one("Representation")["bbox"]) | fitz.Rect(one("Random projection")["bbox"])
leg.x0 -= 12
area = fitz.Rect(cx(x1) - 10, cy(y2) - 10, 518, cy(y1) + 10)
marks = [d for d in page.get_drawings() if d.get("type") == "f" and 1 < d["rect"].width < 8 and d["rect"].height < 8
         and area.contains(d["rect"]) and not leg.intersects(d["rect"])]
if len(marks) != len(KEYS):
    sys.exit(f"expected {len(KEYS)} panel-c markers, found {len(marks)}")
target = None
for k in KEYS:
    px, py = fx(vals[k]["tissue_silhouette"]), fy(vals[k]["sex_knn_bacc"])
    d = min(marks, key=lambda m: abs((m["rect"].x0 + m["rect"].x1) / 2 - px) + abs((m["rect"].y0 + m["rect"].y1) / 2 - py))
    off = abs((d["rect"].x0 + d["rect"].x1) / 2 - px) + abs((d["rect"].y0 + d["rect"].y1) / 2 - py)
    print(f"{k:22s} silhouette {vals[k]['tissue_silhouette']:.4f} sex {vals[k]['sex_knn_bacc']:.4f}: marker seqno "
          f"{d['seqno']} ({len(d['items'])} edges), offset {off:.2f} pt")
    if off > 0.6:
        sys.exit(f"{k}: no marker at its value; stop")
    if k == "untrained_transformer":
        target = d
if len(target["items"]) != 3:
    sys.exit("the untrained-transformer marker is not a triangle; stop")

lab_u, lab_r = one("Untrained transformer"), one("Random projection")
leg_marks = [d for d in page.get_drawings() if d.get("type") == "f" and 1 < d["rect"].width < 8 and leg.intersects(d["rect"])]
mk_u = min(leg_marks, key=lambda d: abs((d["rect"].y0 + d["rect"].y1) / 2 - cy(lab_u)))
mk_r = min(leg_marks, key=lambda d: abs((d["rect"].y0 + d["rect"].y1) / 2 - cy(lab_r)))
if len(mk_u["items"]) != 3 or len(mk_r["items"]) != 4:
    sys.exit("legend markers are not the triangle and the diamond; stop")
dy = lab_u["origin"][1] - lab_r["origin"][1]            # one legend row up (negative)
diamond = [(it[1].x, it[1].y + dy) for it in mk_r["items"]] + [(mk_r["items"][-1][2].x, mk_r["items"][-1][2].y + dy)]
fill, size, c = mk_r["fill"], lab_r["size"], lab_r["color"]
ink = ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)

for d in (target, mk_u, mk_r):
    r = fitz.Rect(d["rect"])
    page.add_redact_annot(fitz.Rect(r.x0 - 0.15, r.y0 - 0.15, r.x1 + 0.15, r.y1 + 0.15), fill=False)
for s in (lab_u, lab_r):
    r = fitz.Rect(s["bbox"])
    page.add_redact_annot(fitz.Rect(r.x0 + 0.2, r.y0 + 0.6, r.x1 - 0.2, r.y1 - 0.6), fill=False)
page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                      text=fitz.PDF_REDACT_TEXT_REMOVE)

shape = page.new_shape()
shape.draw_polyline([fitz.Point(*p) for p in diamond])
shape.finish(fill=fill, color=None, closePath=True, width=0)
shape.commit()
page.insert_font(fontname="arialsf19", fontfile=ARIAL)
page.insert_text(fitz.Point(lab_r["origin"][0], lab_u["origin"][1]), "Random projection", fontsize=size,
                 fontname="arialsf19", color=ink)
doc.save(out_pdf, garbage=3, deflate=True)

# checks on the written PDF: one fewer marker, the kept markers unchanged, the label gone
new = fitz.open(out_pdf)[0]
texts = [s["text"].replace("\xa0", " ") for b in new.get_text("dict")["blocks"] for l in b.get("lines", [])
         for s in l["spans"]]   # PyMuPDF maps the inserted Arial space glyph to U+00A0 in text extraction
if "Untrained transformer" in texts or texts.count("Random projection") != 1:
    sys.exit("label check failed")
kept =sorted(tuple(round(v, 2) for v in m["rect"]) for m in marks if m is not target)
now = [d for d in new.get_drawings() if d.get("type") == "f" and 1 < d["rect"].width < 8 and d["rect"].height < 8
       and area.contains(d["rect"]) and not leg.intersects(d["rect"])]
if sorted(tuple(round(v, 2) for v in d["rect"]) for d in now) != kept:
    sys.exit("kept markers changed; stop")
print(f"removed the untrained-transformer marker and legend row; Random projection moved {dy:.2f} pt; written {out_pdf}")

stem = str(Path(out_png).with_suffix(""))
subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", out_pdf, stem], check=True)
a = np.asarray(Image.open(old_png).convert("RGB")).astype(int)
b = np.asarray(Image.open(out_png).convert("RGB")).astype(int)
if a.shape != b.shape:
    sys.exit(f"PNG size changed {a.shape} -> {b.shape}; stop")
diff = np.abs(a - b).max(2) > 0
ys, xs = np.nonzero(diff)
print(f"PNG: {diff.sum()} pixels differ, all within x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()} px "
      f"(panel c spans x >= {int(cx(x1) * 300 / 72) - 60} px, y <= {int(cy(y1) * 300 / 72) + 40} px); written {out_png}")
