"""Restore the panel letter "g" of Supplementary Fig. 10 in the vector PDF.

relabel_sf10_pdf_2026-11-11.py removed "(set A)" from the panel d x label with a text redaction. The redaction rectangle
of "Tissue 10-NN balanced accuracy (set A)" (x 148-291, y 423.6-431.3 pt) overlaps the font box of the panel letter "g"
(x 197.9-205.2, y 429.5-442.9 pt), so the redaction also deleted that letter. The PNG, the legend and the plotting
output (si_sample_representations_2026-10-18.pdf) all have it. This script writes the letter back with the plotting
output's own text operator: font F3 (the embedded CWKGXT+Arial-BoldMT subset, the same font file in both PDFs),
12 pt, black, at the plotting output's origin, shifted only by the 13.146174 pt the page was later cropped at the
bottom (the panel letter "f" sits at y 426.360444 in the plotting output and 413.21427 in the installed PDF). The
operator goes into the empty "q Q" that the redaction left after the "f" letter, in the same graphics state as "f".
Nothing else in the file changes.

Run: /opt/homebrew/bin/python3 figures/restore_panel_g_sf10_2026-11-12.py <in.pdf> <out.pdf>
"""
import sys
from pathlib import Path

import fitz

if len(sys.argv) != 3:
    sys.exit(__doc__)
src, dst = sys.argv[1], sys.argv[2]
REF = Path(__file__).resolve().parent / "si_sample_representations_2026-10-18.pdf"
for p in (src, REF):
    if not Path(p).exists():
        sys.exit(f"Cannot proceed - required data file not found: {p}")

ref = fitz.open(REF)
doc = fitz.open(src)
page = doc[0]
if any(s["text"] == "g" for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]):
    sys.exit("the input already has a panel letter g")

# the plotting output's operators for "f" and "g" (PDF y up), and the installed "f"
rs = b"".join(ref.xref_stream(x) for x in ref[0].get_contents()).decode("latin-1")
F_REF, G_REF = (426.360444, "1 0 -0 1 292.4225214703 426.360444 cm"), (163.992694, "1 0 -0 1 197.8902115992 163.992694 cm")
for _, op in (F_REF, G_REF):
    if op not in rs:
        sys.exit(f"operator {op!r} not found in {REF.name}")
if "[ (\x00g) ] TJ" not in rs.split(G_REF[1], 1)[1][:60]:
    sys.exit("panel letter g operator not found after its transform in the plotting output")

xrefs = page.get_contents()
main = xrefs[0]
s = doc.xref_stream(main).decode("latin-1")
F_OLD = "q 1 0 0 1 292.42253 413.21427 cm 0 g 0 G 1 w BT/F3 12 Tf 0 0 Td[<0066>]TJ ET Q q Q"
if s.count(F_OLD) != 1:
    sys.exit(f"expected the panel f operator followed by an empty q Q once, found {s.count(F_OLD)}")
dy = F_REF[0] - 413.21427
gy = G_REF[0] - dy
G_NEW = f"q 1 0 0 1 197.8902116 {gy:.6f} cm 0 g 0 G 1 w BT/F3 12 Tf 0 0 Td[<0067>]TJ ET Q"
s = s.replace(F_OLD, F_OLD[:-len("q Q")] + G_NEW, 1)
doc.update_stream(main, s.encode("latin-1"))
doc.save(dst, garbage=3, deflate=True)
print(f"crop offset {dy:.6f} pt; panel letter g written at (197.8902116, {gy:.6f}) in PDF coordinates")

# check: one new span "g" in Arial-BoldMT 12 pt at the plotting output's position; nothing else changed
out, before = fitz.open(dst)[0], fitz.open(src)[0]
spans = lambda p: [(s["text"], s["font"], round(s["size"], 2), tuple(round(v, 2) for v in s["bbox"]))
                   for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
ref_g = [x for x in spans(ref[0]) if x[0] == "g"]
new, lost = [x for x in spans(out) if x not in spans(before)], [x for x in spans(before) if x not in spans(out)]
print("plotting output g:", ref_g)
print("new spans:", new, "| spans lost:", lost)
if new != ref_g or lost:
    sys.exit("the restored g does not match the plotting output")
print("drawings:", len(before.get_drawings()), "->", len(out.get_drawings()))
