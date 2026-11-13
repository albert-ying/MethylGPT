"""Remove the grey provenance footer ("data: ...") from installed manuscript figure PDFs and trim the page to match.

Used for the former v13 main figures that are now Supplementary Figs 16 to 20, whose scripts need inputs that are
not all on this machine. The footer text is deleted from the content stream (redaction, graphics untouched) and the
page is cut just below the remaining content. The artwork is otherwise unchanged. A PNG at 300 dpi is written beside
each PDF.

Run: python strip_provenance_footer_2026-11-10.py <figure.pdf> [...]
"""
import subprocess
import sys
from pathlib import Path

import fitz

for arg in sys.argv[1:]:
    path = Path(arg)
    if not path.exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
    doc = fitz.open(path)
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    blocks = [b for b in page.get_text("blocks") if b[4].strip().startswith("data:")]
    if len(blocks) != 1:
        print(f"{path.name}: {len(blocks)} footer block(s) found; left unchanged")
        continue
    x0, y0, x1, y1, text = blocks[0][:5]
    if y0 < 0.9 * h:
        print(f"{path.name}: 'data:' block is not at the bottom (y0 {y0:.1f} of {h:.1f}); left unchanged")
        continue
    page.add_redact_annot(fitz.Rect(x0 - 1, y0 - 0.5, x1 + 1, y1 + 0.5))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)
    new_h = y0 + 2.0
    out = fitz.open()
    out.new_page(width=w, height=new_h).show_pdf_page(fitz.Rect(0, 0, w, new_h), doc, 0, clip=fitz.Rect(0, 0, w, new_h))
    tmp = path.with_suffix(".stripped.pdf")
    out.save(tmp, garbage=4, deflate=True)
    out.close()
    doc.close()
    tmp.replace(path)
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(path), str(path.with_suffix(""))], check=True)
    print(f"{path.name}: removed {' '.join(text.split())!r}; page {w:.1f} x {h:.1f} -> {w:.1f} x {new_h:.1f} pt")
