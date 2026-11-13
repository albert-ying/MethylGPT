"""Remove the "Untrained transformer" series from Supplementary Fig. 10d,e without redrawing the figure.

Panel d plots tissue and sex 10-nearest-neighbour balanced accuracy in the six-tissue panel, and panel e the adjusted
Rand index of Leiden and k-means clusters, for several representations. The "Untrained transformer" series (the
embedding of a transformer at random weights on the fitted input layer) compares parts of MethylGPT, so it is removed;
every other series is kept as drawn.

The figure cannot be redrawn here: plot_si_sample_embedding_2026-10-18.py also needs the set A and set B UMAP
coordinates and the corpus profile table, which are on Chronos only. The values of panels d and e are local
(artifacts/eval/si_sample_representations_2026-10-18.json), so the script first checks every marker, interval and
line vertex of panels d and e in the installed vector PDF against those values, then edits the PDF content stream:

  panel d  the untrained-transformer row label, and its two markers and two intervals in each subpanel, are deleted;
           the "Random projection" row above it (label, markers and intervals) is moved down by one row spacing of
           its subpanel (14.41 pt for tissue, 11.28 pt for sex), so the rows stay evenly spaced; the y tick of the
           vacated top slot (tissue: the old Random projection row; sex: the empty tick above it) is deleted and the
           left spine of each subpanel starts one row spacing lower, so each subpanel keeps its old margins.
  panel e  the two untrained-transformer lines (solid and dashed), their 20 markers and both k-means diamonds are
           deleted. The 20 marker XObjects that only those markers used are dropped from the page resources.

No other object is touched. The installed PNG is matplotlib's own 300 dpi raster (saved separately from the PDF, so it
sits up to a pixel off the PDF geometry), so it is edited in its own pixels: in panel d the band holding the Random
projection row is moved down by the same row spacing in pixels (60 and 47 px) and the vacated band set to white, which
removes the untrained row, the vacated tick and the top of each spine as in the PDF. In panel e the removed lines
cross kept lines, markers and diamonds, so no pixel edit can be clean. The installed PNG was drawn without
antialiasing and with heavier markers than its PDF, and its dashes sit at other phases, so neither a PDF render
(MuPDF and Poppler draw these marks about 19% lighter) nor a patch of new pixels along the removed dashed line (which
leaves broken dash chains) matches it. The plotting area of panel e (every data mark plus a white margin; no spine,
tick, label or title) is therefore rasterized again from the edited PDF's own vector marks (coordinates, colours,
widths, caps, joins and dashes read from the content stream, drawn in stream order) with matplotlib's Agg renderer
without antialiasing, with the offset and the stroke and marker multipliers fitted so that the same rasterization of
the installed PDF's marks reproduces the installed PNG; that comparison is printed as the check that the kept series
look as before. Everywhere else the PNG keeps its pixels.

Run (from the project folder; inputs are the figure as installed before this change, kept as
figures/supplementary_figure10_installed_2026-11-11.*):
  /opt/homebrew/bin/python3 figures/remove_untrained_sf10_2026-11-12.py figures/supplementary_figure10_installed_2026-11-11.pdf \
      figures/supplementary_figure10_installed_2026-11-11.png figures/supplementary_figure10_2026-11-12.pdf \
      figures/supplementary_figure10_2026-11-12.png
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFilter, PngImagePlugin

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "artifacts" / "eval" / "si_sample_representations_2026-10-18.json"
S = 300 / 72
if len(sys.argv) != 5:
    sys.exit(__doc__)
old_pdf, old_png, out_pdf, out_png = sys.argv[1:5]
for path in (SRC, old_pdf, old_png):
    if not Path(path).exists():
        sys.exit(f"Cannot proceed - required data file not found: {path}")
try:
    R = json.loads(SRC.read_text())
except (OSError, ValueError) as exc:
    sys.exit(f"Cannot read {SRC}: {exc}")
SAN, CL = R["set_a"]["natural"], R["set_a_clustering"]
KEYS_D = ["raw", "raw_pca50", "factor_scores", "mgpt2_selfsupervised", "mgpt2_multitask", "untrained_transformer",
          "random_projection"]                       # panel d rows, bottom to top (plot order)
KEYS_E = ["raw", "factor_scores", "mgpt2_multitask", "untrained_transformer", "random_projection"]
COL = {"raw": "#2D5688", "raw_pca50": "#8CADD2", "factor_scores": "#96412A", "mgpt2_selfsupervised": "#C2634A",
       "mgpt2_multitask": "#DA9A87", "untrained_transformer": "#C25FAF", "random_projection": "#8A827C"}
INK = "#181114"
GONE = "untrained_transformer"


def rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))


def same(c, h):
    return c is not None and h is not None and len(c) == 3 and max(abs(a - b) for a, b in zip(c, rgb(h))) < 0.6 / 255


def hexc(c):
    return None if c is None else "#%02X%02X%02X" % tuple(round(v * 255) for v in c)


# ---------------------------------------------------------------- content stream: tokens, operators, interpretation
WS, DL = " \t\r\n\f\0", "()<>[]{}/%"


def tokenize(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in WS:
            i += 1
        elif c == "%":
            j = s.find("\n", i)
            i = n if j < 0 else j
        elif c == "(":
            depth, j = 0, i
            while j < n:
                if s[j] == "\\":
                    j += 2
                    continue
                depth += (s[j] == "(") - (s[j] == ")")
                j += 1
                if depth == 0:
                    break
            toks.append(("str", s[i:j], i, j))
            i = j
        elif (c == "<" and s.startswith("<<", i)) or (c == ">" and s.startswith(">>", i)):
            toks.append(("dict", s[i:i + 2], i, i + 2))
            i += 2
        elif c == "<":
            j = s.index(">", i) + 1
            toks.append(("hex", s[i:j], i, j))
            i = j
        elif c in "[]":
            toks.append((c, c, i, i + 1))
            i += 1
        else:
            j = i + 1
            while j < n and s[j] not in WS and s[j] not in DL:
                j += 1
            w = s[i:j]
            kind = "name" if c == "/" else "num" if re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)", w) else "op"
            toks.append((kind, w, i, j))
            i = j
    return toks


def operators(toks):
    ops, args, start, depth = [], [], None, 0
    for t in toks:
        if start is None:
            start = t[2]
        if t[0] == "[":
            depth += 1
        elif t[0] == "]":
            depth -= 1
        if t[0] == "op" and depth == 0:
            if t[1] in ("BI", "ID", "EI"):
                sys.exit("inline image in the content stream; stop")
            ops.append({"op": t[1], "args": args, "start": start, "end": t[3]})
            args, start = [], None
        else:
            args.append(t)
    return ops


def mul(A, B):                                       # A applied first, then B
    a, b, c, d, e, f = A
    p, q, r, s_, t, u = B
    return (a * p + b * r, a * q + b * s_, c * p + d * r, c * q + d * s_, e * p + f * r + t, e * q + f * s_ + u)


PAINT = {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"}
STROKES, FILLS = {"S", "s", "B", "B*", "b", "b*"}, {"f", "F", "f*", "B", "B*", "b", "b*"}


def interpret(ops, H):
    """Painted objects (paths, XObject draws, text blocks) with their operator range, colours and page position."""
    st = {"ctm": (1, 0, 0, 1, 0, 0), "stroke": (0, 0, 0), "fill": (0, 0, 0), "lw": 1.0, "cap": 0, "join": 0,
          "dash": ((), 0.0)}
    stack, blocks, objs, unbalanced = [], {}, [], 0
    path0, pts, bt, texts = None, [], None, []
    num = lambda a: [float(x[1]) for x in a if x[0] == "num"]

    def page_pt(x, y, m):
        return (m[0] * x + m[2] * y + m[4], H - (m[1] * x + m[3] * y + m[5]))
    for k, o in enumerate(ops):
        op, a = o["op"], o["args"]
        if op == "q":
            stack.append((dict(st), k))
        elif op == "Q":
            if stack:
                prev, qk = stack.pop()
                blocks[qk] = k
                st = prev
            else:
                unbalanced += 1
        elif op == "cm":
            st["ctm"] = mul(tuple(num(a)), st["ctm"])
        elif op in ("RG", "rg"):
            st["stroke" if op == "RG" else "fill"] = tuple(num(a))
        elif op in ("G", "g"):
            st["stroke" if op == "G" else "fill"] = tuple(num(a)) * 3
        elif op in ("K", "k", "CS", "cs", "SC", "SCN", "sc", "scn"):
            st["stroke" if op.isupper() else "fill"] = None
        elif op == "w":
            st["lw"] = num(a)[0]
        elif op == "J":
            st["cap"] = int(num(a)[0])
        elif op == "j":
            st["join"] = int(num(a)[0])
        elif op == "d":
            v = num(a)
            st["dash"] = (tuple(v[:-1]), v[-1])
        elif op in ("m", "l", "c", "v", "y", "re", "h"):
            if path0 is None:
                path0, pts = k, []
            v = num(a)
            if op == "re":
                x, y, w, h = v
                pts += [page_pt(x, y, st["ctm"]), page_pt(x + w, y + h, st["ctm"])]
            else:
                pts += [page_pt(v[i], v[i + 1], st["ctm"]) for i in range(0, len(v), 2)]
        elif op in PAINT:
            if path0 is not None and op != "n":
                objs.append({"kind": "path", "i0": path0, "i1": k, "paint": op, "ctm": st["ctm"], "pts": pts,
                             "stroke": st["stroke"] if op in STROKES else None,
                             "fill": st["fill"] if op in FILLS else None, "q": stack[-1][1] if stack else None,
                             "lw": st["lw"], "cap": st["cap"], "join": st["join"], "dash": st["dash"],
                             "closed": any(ops[x]["op"] == "h" for x in range(path0, k))})
            path0 = None
        elif op == "Do":
            objs.append({"kind": "do", "i0": k, "i1": k, "name": a[0][1], "ctm": st["ctm"], "paint": "Do",
                         "pts": [page_pt(0, 0, st["ctm"])], "fill": st["fill"], "stroke": st["stroke"],
                         "q": stack[-1][1] if stack else None, "lw": st["lw"]})
        elif op == "BT":
            bt, texts = k, []
        elif op in ("Tj", "TJ", "'", '"'):
            texts += [bytes.fromhex(t[1][1:-1]).decode("utf-16-be", "replace") for t in a if t[0] == "hex"]
        elif op == "ET":
            objs.append({"kind": "text", "i0": bt, "i1": k, "text": "".join(texts), "ctm": st["ctm"], "pts": [],
                         "stroke": None, "fill": None, "q": stack[-1][1] if stack else None})
    return objs, blocks, unbalanced


def centre(o):
    xs, ys = [p[0] for p in o["pts"]], [p[1] for p in o["pts"]]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def size(o):
    xs, ys = [p[0] for p in o["pts"]], [p[1] for p in o["pts"]]
    return max(xs) - min(xs), max(ys) - min(ys)


def is_col(o, h):
    return same(o["stroke"], h) or same(o["fill"], h)


def inside(o, r):
    return bool(o["pts"]) and r.contains(fitz.Point(*centre(o)))


def main_stream(d, p):
    return max(p.get_contents(), key=lambda x: len(d.xref_stream(x)))


def render(p, dpi_matrix, clip=None, gray_=False):
    pix = p.get_pixmap(matrix=dpi_matrix, clip=clip, alpha=False, colorspace=fitz.csGRAY if gray_ else fitz.csRGB)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n).astype(int), pix.x, pix.y


def grow(m, n):
    return np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * n + 1))) > 0


# ---------------------------------------------------------------- read the installed PDF
doc = fitz.open(old_pdf)
page = doc[0]
H, W = page.rect.height, page.rect.width
main = main_stream(doc, page)
src = doc.xref_stream(main).decode("latin-1")
OPS = operators(tokenize(src))
OBJS, BLOCKS, unbal = interpret(OPS, H)
spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
print(f"content stream {main}: {len(OPS)} operators, {len(OBJS)} painted objects, {len(BLOCKS)} q/Q blocks "
      f"({unbal} unmatched Q, ignored as viewers do)")


def span(text):
    hits = [s for s in spans if s["text"] == text]
    if len(hits) != 1:
        sys.exit(f"expected one span {text!r}, found {len(hits)}")
    return hits[0]


# panel regions from the panel letters and axis labels
yd0, yd1 = span("d")["bbox"][1], span("Tissue\xa010\xadNN\xa0balanced\xa0accuracy")["bbox"][1]
ye0, ye1 = span("e")["bbox"][1], span("Leiden resolution")["bbox"][1]
x_g = span("Whole\xa0blood")["bbox"][0] - 20                      # panel g starts right of this
RD = fitz.Rect(0, yd0, W, yd1)
RE = fitz.Rect(0, ye0, x_g, ye1)
LEG = fitz.Rect(span("Cross-validation")["bbox"]) | fitz.Rect(span("Leave one dataset out")["bbox"])
LEG.x0 -= 12                                                      # panel d legend (its markers share the raw colour)

# every vector drawing in panels d and e must be one interpreted object of the same colour and position
cen = lambda r: fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
dr = [d for d in page.get_drawings() if RD.contains(cen(d["rect"])) or RE.contains(cen(d["rect"]))]
cand = [o for o in OBJS if o["kind"] in ("path", "do") and (inside(o, RD) or inside(o, RE))]
for d in dr:
    c = cen(d["rect"])
    col = d.get("color") if d.get("color") is not None else d.get("fill")
    hits =[o for o in cand if abs(centre(o)[0] - c.x) < 0.06 and abs(centre(o)[1] - c.y) < 0.06
            and (same(o["stroke"], hexc(col)) or same(o["fill"], hexc(col)))
            and (o["kind"] == "do" or max(abs(a - b) for a, b in zip(size(o), (d["rect"].width, d["rect"].height))) < 0.15)]
    if len(hits) != 1:
        sys.exit(f"drawing {d['seqno']} {hexc(col)} at {d['rect']} matches {len(hits)} interpreted objects; stop")
if len(dr) != len(cand):
    sys.exit(f"{len(dr)} drawings but {len(cand)} interpreted objects in panels d and e; stop")
print(f"interpreter check: all {len(dr)} drawings of panels d and e matched one-to-one")


# ---------------------------------------------------------------- axes (exact coordinates from the stream)
def seg(o):
    return o["kind"] == "path" and len(o["pts"]) == 2 and same(o["stroke"], INK) and o["paint"] != "f"


vert = lambda o: abs(o["pts"][0][0] - o["pts"][1][0]) < 1e-6
horiz = lambda o: abs(o["pts"][0][1] - o["pts"][1][1]) < 1e-6
length = lambda o: math.dist(*o["pts"])


def xticks(x0, x1, ybase):
    return sorted(o["pts"][0][0] for o in OBJS if seg(o) and vert(o) and 2.9 < length(o) < 3.1
                  and x0 < o["pts"][0][0] < x1 and abs(min(p[1] for p in o["pts"]) - ybase) < 0.05)


def yticks(xspine):
    return sorted((o for o in OBJS if seg(o) and horiz(o) and 2.9 < length(o) < 3.1
                   and abs(max(p[0] for p in o["pts"]) - xspine) < 0.05), key=lambda o: o["pts"][0][1])


def tick_values(xt, ybase, ylim):
    out = []
    for x in xt:
        lab = [s for s in spans if s["bbox"][1] > ybase and s["bbox"][3] < ylim
               and abs((s["bbox"][0] + s["bbox"][2]) / 2 - x) < 1.0 and re.fullmatch(r"[01]\.\d", s["text"])]
        if len(lab) != 1:
            sys.exit(f"tick at x {x:.2f}: {len(lab)} labels")
        out.append(float(lab[0]["text"]))
    return out


sp_d = sorted((o for o in OBJS if seg(o) and inside(o, RD) and vert(o) and length(o) > 50), key=lambda o: o["pts"][0][0])
if len(sp_d) != 2:
    sys.exit(f"expected two left spines in panel d, found {len(sp_d)}")
SUB = {}
for name, sp, nxt in (("tissue", sp_d[0], sp_d[1]["pts"][0][0] - 3.5), ("sex", sp_d[1], W)):
    xs = sp["pts"][0][0]
    ybase = max(p[1] for p in sp["pts"])
    xt = xticks(xs, nxt, ybase)
    yt = yticks(xs)
    ys = [o["pts"][0][1] for o in yt]
    gaps = np.diff(ys)
    if gaps.max() - gaps.min() > 0.005:
        sys.exit(f"{name}: y ticks not evenly spaced {gaps}")
    SUB[name] = {"spine": sp, "yticks": yt, "ys": ys, "row": float(gaps.mean()), "ybase": ybase, "x0": xs, "x1": nxt}
    vals = tick_values(xt, ybase, yd1 + 1)
    SUB[name]["fx"] = np.poly1d(np.polyfit(vals, xt, 1))
    raw_y = np.mean([centre(o)[1] for o in OBJS if is_col(o, COL["raw"]) and o["kind"] == "path"
                     and inside(o, RD) and not LEG.contains(fitz.Point(*centre(o))) and xs < centre(o)[0] < nxt])
    SUB[name]["y0"] = min(ys, key=lambda y: abs(y - raw_y))       # tick of the bottom row (raw beta values)
    print(f"panel d {name}: {len(ys)} y ticks, row spacing {SUB[name]['row']:.5f} pt; x ticks {vals}")

sp_e = [o for o in OBJS if seg(o) and inside(o, RE) and vert(o) and length(o) > 50]
bt_e = [o for o in OBJS if seg(o) and inside(o, RE) and horiz(o) and length(o) > 50]
if len(sp_e) != 1 or len(bt_e) != 1:
    sys.exit("panel e: expected one left and one bottom spine")
ex, ey = sp_e[0]["pts"][0][0], bt_e[0]["pts"][0][1]
e_yt = [o["pts"][0][1] for o in yticks(ex)][::-1]                   # bottom first
eyl = sorted((s for s in spans if s["bbox"][2] < ex and RE.contains(fitz.Rect(s["bbox"]))
              and re.fullmatch(r"0\.\d", s["text"])), key=lambda s: -s["bbox"][1])
e_xt = xticks(ex, x_g, ey)
if len(e_yt) != len(eyl) or len(e_xt) != 4:
    sys.exit(f"panel e: {len(e_yt)} y ticks / {len(eyl)} labels, {len(e_xt)} x ticks")
fy_e = np.poly1d(np.polyfit([float(s["text"]) for s in eyl], e_yt, 1))
fx_e = np.poly1d(np.polyfit([-2, -1, 0, 1], e_xt, 1))
print(f"panel e: x ticks {[round(x, 2) for x in e_xt]} (10^-2 to 10^1), y ticks {[s['text'] for s in eyl]}")


# ---------------------------------------------------------------- check the plotted values against the source data
def check_d(objs, rows_of, label):
    """rows_of: key -> row index (0 = bottom). Each key's markers and intervals must sit at its values in that row."""
    worst = 0.0
    for name, sub in SUB.items():
        for k, i in rows_of.items():
            mine = [o for o in objs if is_col(o, COL[k]) and inside(o, RD) and not LEG.contains(fitz.Point(*centre(o)))
                    and sub["x0"] < centre(o)[0] < sub["x1"]]
            if len(mine) != 4:
                sys.exit(f"{label} {name} {k}: {len(mine)} objects, expected 4")
            for j, sch in enumerate(("stratified5", "lodo")):
                e = SAN[k][f"{name}_knn_bacc|{sch}"]
                yexp = sub["y0"] - i * sub["row"] + (0.17 if j == 0 else -0.17) * sub["row"]
                mk = [o for o in mine if o["paint"] != "S" and abs(centre(o)[1] - yexp) < 0.05]
                ln = [o for o in mine if o["paint"] == "S" and abs(centre(o)[1] - yexp) < 0.05]
                if len(mk) != 1 or len(ln) != 1:
                    sys.exit(f"{label} {name} {k} {sch}: marker or interval not in row {i} (y {yexp:.2f})")
                lo, hi = sorted(p[0] for p in ln[0]["pts"])
                d = max(abs(centre(mk[0])[0] - sub["fx"](e["value"])), abs(lo - sub["fx"](e["ci95"][0])),
                        abs(hi - sub["fx"](e["ci95"][1])))
                worst = max(worst, d)
                if d > 0.05:
                    sys.exit(f"{label} {name} {k} {sch}: drawn {d:.3f} pt from its value; stop")
    return worst


def check_e(objs, keys, label):
    worst = 0.0
    res = sorted((kk for kk in CL["raw"] if kk.startswith("leiden_")), key=lambda kk: float(kk.split("_")[1]))
    xs = [fx_e(math.log10(float(kk.split("_")[1]))) for kk in res]
    for k in keys:
        mine = [o for o in objs if is_col(o, COL[k]) and inside(o, RE)]
        lines = [o for o in mine if o["paint"] == "S" and len(o["pts"]) == len(res)]
        dots = [o for o in mine if o["kind"] == "do"]
        dia = [o for o in mine if o["paint"] in ("B", "b", "f") and len(o["pts"]) >= 4]
        if len(lines) != 2 or len(dots) != 2 * len(res) or len(dia) != 2 or len(mine) != 4 + 2 * len(res):
            sys.exit(f"{label} panel e {k}: {len(lines)} lines, {len(dots)} markers, {len(dia)} diamonds")
        for tgt in ("tissue", "dataset"):
            ys = [fy_e(CL[k][kk][f"ari_{tgt}"]["value"]) for kk in res]
            ln = [o for o in lines if max(abs(p[0] - x) + abs(p[1] - y) for p, x, y in zip(o["pts"], xs, ys)) < 0.05]
            if len(ln) != 1:
                sys.exit(f"{label} panel e {k} {tgt}: no line at its values")
            for x, y in zip(xs, ys):
                dd = min(abs(o["pts"][0][0] - x) + abs(o["pts"][0][1] - y) for o in dots)
                worst = max(worst, dd)
                if dd > 0.05:
                    sys.exit(f"{label} panel e {k} {tgt}: no marker at ({x:.2f}, {y:.2f})")
            xk, yk = fx_e(math.log10(4.0)), fy_e(CL[k]["kmeans_6"][f"ari_{tgt}"]["value"])
            dd = min(abs(centre(o)[0] - xk) + abs(centre(o)[1] - yk) for o in dia)
            worst = max(worst, dd)
            if dd > 0.05:
                sys.exit(f"{label} panel e {k} {tgt}: no k-means diamond at its value")
    return worst


w = max(check_d(OBJS, {k: i for i, k in enumerate(KEYS_D)}, "installed"), check_e(OBJS, KEYS_E, "installed"))
print(f"installed PDF: all 7 panel-d series (both subpanels) and 5 panel-e series sit at their values in {SRC.name} "
      f"(largest offset {w:.4f} pt)")

# ---------------------------------------------------------------- plan the edits
edits = []                                            # (start, end, replacement)


def drop(o):
    edits.append((OPS[o["i0"]]["start"], OPS[o["i1"]]["end"], " "))


def drop_block(qk):
    edits.append((OPS[qk]["start"], OPS[BLOCKS[qk]]["end"], " "))


def identity(m):
    return max(abs(v - w_) for v, w_ in zip(m, (1, 0, 0, 1, 0, 0))) < 1e-9


def shift(o, dy):                                    # move a path object down the page by dy pt
    if not identity(o["ctm"]):
        sys.exit("unexpected transform at a moved object; stop")
    edits.append((OPS[o["i0"]]["start"], OPS[o["i0"]]["start"], f"q 1 0 0 1 0 {-dy:.6f} cm "))
    edits.append((OPS[o["i1"]]["end"], OPS[o["i1"]]["end"], " Q"))


gone_d = [o for o in OBJS if is_col(o, COL[GONE]) and inside(o, RD)]
moved_d = [o for o in OBJS if is_col(o, COL["random_projection"]) and inside(o, RD)]
if len(gone_d) != 8 or len(moved_d) != 8:
    sys.exit(f"panel d: {len(gone_d)} untrained and {len(moved_d)} random-projection objects, expected 8 and 8")
plan = {}
for name, sub in SUB.items():
    in_sub = lambda o: sub["x0"] < centre(o)[0] < sub["x1"]
    row_of = lambda o: int(np.argmin([abs(centre(o)[1] - y) for y in sub["ys"]]))
    g, m = [o for o in gone_d if in_sub(o)], [o for o in moved_d if in_sub(o)]
    rg, rm = {row_of(o) for o in g}, {row_of(o) for o in m}
    if len(g) != 4 or len(m) != 4 or len(rg) != 1 or len(rm) != 1 or min(rm) != min(rg) - 1:
        sys.exit(f"{name}: random projection is not the row directly above the untrained row")
    for o in g:
        drop(o)
    for o in m:
        shift(o, sub["row"])
    top = sub["yticks"][0]                            # topmost tick: the slot left empty by the move
    if row_of({"pts": top["pts"]}) != min(rm) and name == "tissue":
        sys.exit("tissue: the top tick is not the random projection row")
    drop(top)
    sp = sub["spine"]
    if not identity(sp["ctm"]):
        sys.exit(f"{name}: spine is transformed; stop")
    ytop = min(p[1] for p in sp["pts"])
    hit = [k for k in range(sp["i0"], sp["i1"]) if OPS[k]["op"] in ("m", "l")
           and abs(H - float(OPS[k]["args"][1][1]) - ytop) < 1e-6]
    if len(hit) != 1:
        sys.exit(f"{name}: spine top point not found")
    tok = OPS[hit[0]]["args"][1]
    edits.append((tok[2], tok[3], f"{float(tok[1]) - sub['row']:.5f}"))
    plan[name] = {"row": sub["row"], "tick_removed_y": top["pts"][0][1]}
    print(f"panel d {name}: delete {len(g)} untrained marks, move {len(m)} random-projection marks down "
          f"{sub['row']:.5f} pt, delete the tick at y {top['pts'][0][1]:.2f}, spine top {ytop:.2f} -> {ytop + sub['row']:.2f}")

# row labels: delete the untrained label block, move the random projection label block down one tissue row
tl = {o["text"]: o for o in OBJS if o["kind"] == "text" and o["text"] in ("Untrained transformer", "Random projection")}
if len(tl) != 2 or any(o["q"] is None for o in tl.values()):
    sys.exit("row labels not found as their own q/Q blocks")
for o in tl.values():
    if [x for x in OBJS if x is not o and o["q"] < x["i0"] < BLOCKS[o["q"]]] or OPS[o["q"] + 1]["op"] != "cm":
        sys.exit(f"label block of {o['text']!r} is not a plain positioned text block")
cm_rp, cm_ut = OPS[tl["Random projection"]["q"] + 1], OPS[tl["Untrained transformer"]["q"] + 1]
y_rp, y_ut = float(cm_rp["args"][5][1]), float(cm_ut["args"][5][1])
if abs((y_rp - y_ut) - SUB["tissue"]["row"]) > 1e-3 or abs(float(cm_rp["args"][4][1]) - float(cm_ut["args"][4][1])) < 1:
    sys.exit("label spacing differs from the tissue row spacing")
drop_block(tl["Untrained transformer"]["q"])
tok = cm_rp["args"][5]
edits.append((tok[2], tok[3], f"{y_rp - SUB['tissue']['row']:.5f}"))

# panel e: the two lines and two diamonds, and the two marker blocks of the untrained series
gone_e = [o for o in OBJS if is_col(o, COL[GONE]) and inside(o, RE)]
paths_e, dos_e = [o for o in gone_e if o["kind"] == "path"], [o for o in gone_e if o["kind"] == "do"]
qs = Counter(o["q"] for o in dos_e)
if len(paths_e) != 4 or len(dos_e) != 20 or len(qs) != 2 or set(qs.values()) != {10}:
    sys.exit(f"panel e: {len(paths_e)} paths, {len(dos_e)} markers in blocks {dict(qs)}; expected 4, and 20 in two of 10")
for qk in qs:
    if any(x not in dos_e for x in OBJS if qk < x["i0"] < BLOCKS[qk]):
        sys.exit("an untrained marker block holds other marks; stop")
    drop_block(qk)
for o in paths_e:
    drop(o)
fm_gone = sorted({o["name"] for o in dos_e})

# apply the edits, last first
edits.sort(key=lambda e: (e[0], e[1]))
for (a0, a1, _), (b0, b1, _) in zip(edits[:-1], edits[1:]):
    if a1 > b0 or (a0 == a1 == b0 == b1):
        sys.exit("overlapping edits; stop")
new = src
for a0, a1, rep in reversed(edits):
    new = new[:a0] + rep + new[a1:]
doc.update_stream(main, new.encode("latin-1"))
for name in fm_gone:
    if re.search(re.escape(name) + r"\s+Do", new):
        sys.exit(f"{name} still used; stop")
    doc.xref_set_key(page.xref, f"Resources/XObject/{name[1:]}", "null")
doc.save(out_pdf, garbage=3, deflate=True)
doc.close()
print(f"written {out_pdf}: {len(edits)} stream edits; {len(fm_gone)} marker XObjects dropped from the page resources")

# ---------------------------------------------------------------- checks on the written PDF
d_old, d_new = fitz.open(old_pdf), fitz.open(out_pdf)
p_old, p_new = d_old[0], d_new[0]
OBJS2, _, _ = interpret(operators(tokenize(d_new.xref_stream(main_stream(d_new, p_new)).decode("latin-1"))), H)
w = max(check_d(OBJS2, {k: i for i, k in enumerate(k for k in KEYS_D if k != GONE)}, "edited"),
        check_e(OBJS2, [k for k in KEYS_E if k != GONE], "edited"))
if any(is_col(o, COL[GONE]) for o in OBJS2 if inside(o, RD) or inside(o, RE)):
    sys.exit("untrained colour still drawn in panels d or e")
print(f"edited PDF: the 6 kept panel-d series sit at their values in rows 0-5 of both subpanels and the 4 kept panel-e "
      f"series at theirs (largest offset {w:.4f} pt); no untrained-series mark remains")


def sig(d, dy=0.0, dtop=0.0):
    r = d["rect"]
    return (d["type"], hexc(d.get("color")), hexc(d.get("fill")), round(d.get("width") or 0, 3), len(d["items"]),
            round(r.x0, 2), round(r.y0 + dy + dtop, 2), round(r.x1, 2), round(r.y1 + dy, 2))


exp, n_rm = Counter(), 0
for d in p_old.get_drawings():
    c = cen(d["rect"])
    inD, inE = RD.contains(c), RE.contains(c)
    sub = "tissue" if c.x < SUB["tissue"]["x1"] else "sex"
    cols = (hexc(d.get("color")), hexc(d.get("fill")))
    if (inD or inE) and COL[GONE] in cols:
        n_rm += 1
        continue
    if inD and abs(c.y - plan[sub]["tick_removed_y"]) < 0.01 and d["rect"].width > 2.9 and d["rect"].height < 0.01:
        continue
    if inD and COL["random_projection"] in cols:
        exp[sig(d, dy=plan[sub]["row"])] += 1
    elif inD and d["rect"].height > 50 and d["rect"].width < 0.01:
        exp[sig(d, dtop=plan[sub]["row"])] += 1
    else:
        exp[sig(d)] += 1
got = Counter(sig(d) for d in p_new.get_drawings())
if got != exp:
    print("missing:", list((exp - got).items())[:6])
    print("unexpected:", list((got - exp).items())[:6])
    sys.exit("drawings of the edited PDF differ from the planned edit; stop")
print(f"drawings: {len(p_old.get_drawings())} -> {len(p_new.get_drawings())}; {n_rm} untrained marks and 2 ticks removed, "
      f"8 random-projection marks moved one row down, 2 spines start one row lower, every other drawing identical")

key_ = lambda s: (s["text"], tuple(round(v, 2) for v in s["bbox"]))
tx_old = Counter(key_(s) for b in p_old.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"])
tx_new = Counter(key_(s) for b in p_new.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"])
exp_t = tx_old.copy()
exp_t[key_(span("Untrained transformer"))] -= 1
exp_t[key_(span("Random projection"))] -= 1
moved_rp = [k for k in tx_new if k[0] == "Random projection"]
if len(moved_rp) != 1 or abs(moved_rp[0][1][1] - span("Random projection")["bbox"][1] - SUB["tissue"]["row"]) > 0.02:
    sys.exit("Random projection label not moved by one row")
exp_t[moved_rp[0]] += 1
if +exp_t != tx_new:
    sys.exit(f"text changed beyond the two labels: {list((+exp_t - tx_new).items())[:4]} {list((tx_new - +exp_t).items())[:4]}")
print("text: 'Untrained transformer' removed, 'Random projection' moved down one row, every other span unchanged")

ra, _, _ = render(p_old, fitz.Matrix(S, S))
rb, _, _ = render(p_new, fitz.Matrix(S, S))
dm = np.abs(ra - rb).max(2) > 0
ys, xs = np.nonzero(dm)
okD = (ys >= RD.y0 * S - 2) & (ys <= RD.y1 * S + 2)
okE = (ys >= RE.y0 * S - 2) & (ys <= RE.y1 * S + 2) & (xs <= RE.x1 * S + 2)
print(f"PDF renders at 300 dpi: {int(dm.sum())} pixels differ, all within x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}; "
      f"{int((~(okD | okE)).sum())} outside panels d and e")
if (~(okD | okE)).any():
    sys.exit("PDF changed outside panels d and e; stop")

# ---------------------------------------------------------------- PNG
im0 = Image.open(old_png)
info = {k: v for k, v in im0.info.items() if isinstance(v, str)}
img = np.asarray(im0.convert("RGB")).copy()
gray = img.mean(2)
res = img.copy()
Hp, Wp = gray.shape
allowed = np.zeros(gray.shape, bool)
xgap = int((SUB["sex"]["x0"] - 5.5) * S)              # white column between the two subpanels
for name, sub in SUB.items():
    yb, yt = int(sub["ybase"] * S), int(min(p[1] for p in sub["spine"]["pts"]) * S)
    xs_px = int(round(sub["x0"] * S))
    xsp = min(range(xs_px - 4, xs_px + 5), key=lambda c: gray[yt + 100:yb - 100, c].mean())   # spine column
    tc = int(round((sub["x0"] - 1.5) * S))            # tick column
    rows = np.nonzero(gray[yt - 20:yb + 5, tc - 1:tc + 2].min(1) < 128)[0] + yt - 20
    trow = [int(round(g.mean())) for g in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1) if len(g)]
    step = round(sub["row"] * S)
    if len(trow) != len(sub["ys"]) or set(np.diff(trow)) != {step}:
        sys.exit(f"PNG {name}: tick rows {trow} are not {len(sub['ys'])} rows {step} px apart")
    i_rp = int(np.argmin([abs(y - sub["y0"] + 6 * sub["row"]) for y in sub["ys"]]))   # tick index of row 6
    r_rp = trow[i_rp]
    top_ink = int(np.nonzero(gray[yt - 20:yt + 20, xsp] < 128)[0][0]) + yt - 20
    top, split = top_ink - 4, r_rp + step // 2
    x0, x1 = (int(10 * S), xgap) if name == "tissue" else (xgap, Wp)
    cols = [c for c in range(x0, x1) if abs(c - xsp) > 2]
    for r in (top - 1, top, split - 1, split, split + step - 1, split + step):
        if gray[r, cols].min() < 250:
            sys.exit(f"PNG {name}: row {r} at a band edge is not white outside the spine; stop")
    for c in (x0, x1 - 1):
        if gray[top:split + step, c].min() < 250:
            sys.exit(f"PNG {name}: band side column {c} is not white; stop")
    res[top + step:split + step, x0:x1] = img[top:split, x0:x1]
    res[top:top + step, x0:x1] = 255
    allowed[top:split + step, x0:x1] = True
    print(f"PNG panel d {name}: rows {top}-{split - 1} (x {x0}-{x1 - 1}) moved down {step} px "
          f"(PDF row spacing {sub['row'] * S:.2f} px); tick rows {trow}; spine top row {top_ink} -> {top_ink + step}")

# panel e: the removed lines cross kept lines, markers and diamonds, so the plotting area of panel e (every data mark
# plus a white margin; no spine, tick, label or title) is rasterized again with Agg from the edited PDF's vector marks
def marks(objs):
    return [o for o in objs if o["kind"] in ("path", "do") and inside(o, RE)
            and not (same(o["stroke"], INK) or same(o["fill"], INK))]


ext = []
for o in marks(OBJS):
    xs_, ys_ = [q[0] for q in o["pts"]], [q[1] for q in o["pts"]]
    r_ = 2.0 if o["kind"] == "do" else 1.0           # marker radius 1.1 pt plus stroke; half line width plus cap
    ext.append((min(xs_) - r_, min(ys_) - r_, max(xs_) + r_, max(ys_) + r_))
boxE = fitz.Rect(min(e[0] for e in ext) - 1.5, min(e[1] for e in ext) - 1.5, max(e[2] for e in ext) + 1.5,
                 max(e[3] for e in ext) + 1.5)
if boxE.x0 < ex + 1 or boxE.y1 > ey - 1 or any(boxE.intersects(fitz.Rect(s["bbox"])) for s in spans):
    sys.exit(f"panel e plotting area {boxE} touches an axis or a label; stop")
gx0, gy0 = int(math.floor(boxE.x0 * S)), int(math.floor(boxE.y0 * S))
gx1, gy1 = int(math.ceil(boxE.x1 * S)), int(math.ceil(boxE.y1 * S))
wE, hE = gx1 - gx0, gy1 - gy0
old_blk = img[gy0:gy1, gx0:gx1].astype(int)
# marker shape: every panel-e marker XObject is the same circle; its radius is read from the XObject
fm_names = {o["name"][1:] for o in marks(OBJS) if o["kind"] == "do"}
fm_streams = {d_old.xref_stream(xo[0]) for xo in p_old.get_xobjects() if xo[1] in fm_names}
if len(fm_streams) != 1:
    sys.exit("panel e markers are not one shape; stop")
MR = max(abs(float(t[1])) for t in tokenize(fm_streams.pop().decode("latin-1")) if t[0] == "num" and t[1] != "1")


def mse_mu(tx, ty):                                  # coarse alignment: MuPDF render of the installed PDF vs the PNG
    a, x, y = render(p_old, fitz.Matrix(S, 0, 0, S, tx, ty), boxE, gray_=True)
    return float(np.mean((a[..., 0] - gray[y:y + a.shape[0], x:x + a.shape[1]]) ** 2))


_, TX0, TY0 = min((mse_mu(tx, ty), round(tx, 2), round(ty, 2)) for tx in np.arange(-1.0, 1.51, 0.1)
                  for ty in np.arange(-1.0, 1.51, 0.1))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402

CAP, JOIN = {0: "butt", 1: "round", 2: "projecting"}, {0: "miter", 1: "round", 2: "bevel"}


STYLE0 = {"lw": 1.0, "dash": 1.0, "ms": 1.0, "mew": 1.0, "ds": 1.0, "dlw": 1.0}   # multipliers of the PDF values
AA = False                                           # the installed PNG has no antialiasing (23 colours in panel d)


def agg(objs, tx, ty, black=False, st=None):
    """Rasterize the panel-e marks of objs, in stream order, into the plotting area at PNG offset (tx, ty) px.

    st scales the PDF's line widths, dash lengths, marker diameter and edge width, and diamond size and edge width
    (the installed PNG was drawn with heavier strokes than its PDF); black=True paints every mark black, fills
    included, to give the area the marks cover."""
    st = STYLE0 if st is None else st
    hexc_ = (lambda c: "#000000") if black else hexc
    plt.rcParams["lines.scale_dashes"] = False       # PDF dash arrays are already in points
    fig = plt.figure(figsize=(wE / 300, hE / 300), dpi=300)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, wE)
    ax.set_ylim(hE, 0)
    P = lambda q: (S * q[0] + tx - gx0, S * q[1] + ty - gy0)
    ms = marks(objs)
    i = 0
    while i < len(ms):
        o = ms[i]
        if o["kind"] == "do":                        # a line's markers: one draw_markers call, as matplotlib made it
            run = [o]
            while i + 1 < len(ms) and ms[i + 1]["kind"] == "do" and ms[i + 1]["q"] == o["q"]:
                i += 1
                run.append(ms[i])
            xy = [P(r["pts"][0]) for r in run]
            ax.add_line(Line2D([q[0] for q in xy], [q[1] for q in xy], linestyle="none", marker="o",
                               markersize=2 * MR * st["ms"], markeredgewidth=o["lw"] * st["mew"],
                               markerfacecolor=hexc_(o["fill"]),
                               markeredgecolor=hexc_(o["stroke"]), clip_on=False, zorder=i + 1, antialiased=AA))
        elif o["paint"] == "S":                      # a line
            xy = [P(q) for q in o["pts"]]
            dash, phase = o["dash"]
            ax.add_line(Line2D([q[0] for q in xy], [q[1] for q in xy], linewidth=o["lw"] * st["lw"],
                               color=hexc_(o["stroke"]),
                               linestyle=(phase, tuple(v * st["dash"] for v in dash)) if dash else "-",
                               solid_capstyle=CAP[o["cap"]],
                               dash_capstyle=CAP[o["cap"]], solid_joinstyle=JOIN[o["join"]],
                               dash_joinstyle=JOIN[o["join"]], clip_on=False, zorder=i + 1, antialiased=AA))
        elif o["paint"] == "B" and o["closed"] and len(o["pts"]) == 4:   # a k-means diamond (scatter marker "D")
            cx, cy = P(centre(o))
            side = size(o)[0] / math.sqrt(2)
            ax.scatter([cx], [cy], s=(side * st["ds"]) ** 2, marker="D", facecolors=hexc_(o["fill"]),
                       edgecolors=hexc_(o["stroke"]), linewidths=o["lw"] * st["dlw"], clip_on=False, zorder=i + 1,
                       antialiased=AA)
        else:
            sys.exit(f"panel e: unexpected mark {o['paint']} with {len(o['pts'])} points; stop")
        i += 1
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int).copy()
    plt.close(fig)
    if a.shape != (hE, wE, 3):
        sys.exit(f"Agg canvas {a.shape} is not the plotting area {(hE, wE)}; stop")
    return a


# fit the offset and the stroke and marker multipliers: the installed marks rasterized with Agg against the installed PNG
err = lambda tx, ty, st: float(np.mean((agg(OBJS, tx, ty, st=st) - old_blk) ** 2))
ST = dict(STYLE0)
best, TX, TY = min((err(tx, ty, ST), round(tx, 3), round(ty, 3)) for tx in np.arange(TX0 - 0.3, TX0 + 0.31, 0.1)
                   for ty in np.arange(TY0 - 0.3, TY0 + 0.31, 0.1))
e_pdf_widths = best
for step in (0.2, 0.1, 0.05, 0.02):
    improved = True
    while improved:
        improved = False
        for k in ST:
            for sgn in (-1, 1):
                trial = dict(ST)
                trial[k] = round(ST[k] + sgn * step, 4)
                if trial[k] < 0.5:
                    continue
                e = err(TX, TY, trial)
                if e < best - 1e-9:
                    best, ST, improved = e, trial, True
        for dx, dy in ((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)):
            e = err(TX + dx, TY + dy, ST)
            if e < best - 1e-9:
                best, TX, TY, improved = e, round(TX + dx, 3), round(TY + dy, 3), True
print(f"PNG panel e: Agg style fitted to the installed PNG: {ST}, offset ({TX:+.2f}, {TY:+.2f}) px; mean squared "
      f"difference {e_pdf_widths:.0f} at the PDF's widths -> {best:.0f}")
a_old, a_new = agg(OBJS, TX, TY, st=ST), agg(OBJS2, TX, TY, st=ST)
a_gone = agg([o for o in OBJS if is_col(o, COL[GONE])], TX, TY, black=True, st=ST)
frame = np.ones((hE, wE), bool)
frame[3:-3, 3:-3] = False
if old_blk.mean(2)[frame].min() < 250 or a_new.mean(2)[frame].min() < 250:
    sys.exit("panel e plotting area does not have a white border in the PNG and in the new raster; stop")
foot = grow(a_gone.min(2) < 255, 1)
if (np.abs(a_old - a_new).max(2)[~foot] > 0).any():
    sys.exit("kept marks rasterize differently once the untrained marks are removed; stop")
dk = np.abs(a_old - old_blk).max(2)
ink = (old_blk.mean(2) < 230) | (a_old.mean(2) < 230)
print(f"PNG panel e: plotting area x {gx0}-{gx1 - 1}, y {gy0}-{gy1 - 1} px rasterized with Agg (matplotlib "
      f"{matplotlib.__version__}) at offset ({TX:+.2f}, {TY:+.2f}) px; marker radius {MR} pt")
print(f"  Agg raster of the installed PDF's marks against the installed PNG: {100 * (dk == 0).mean():.2f}% of pixels "
      f"identical, {100 * (dk <= 8).mean():.2f}% within 8 of 255; on marks mean difference {dk[ink].mean():.1f}")
ratios = []
for k in [k for k in KEYS_E if k != GONE]:
    c = np.array([round(v * 255) for v in rgb(COL[k])])
    n_png, n_agg = int((np.abs(old_blk - c).max(2) <= 20).sum()), int((np.abs(a_old - c).max(2) <= 20).sum())
    ratios.append(n_agg / n_png)
    print(f"  {k:18s} {COL[k]}: core pixels in installed PNG {n_png}, in the Agg raster {n_agg} "
          f"(ratio {n_agg / n_png:.3f})")
if (dk == 0).mean() < 0.95 or min(ratios) < 0.85 or max(ratios) > 1.15:
    sys.exit("the Agg raster does not reproduce the installed PNG closely enough; stop")
print(f"  edited minus installed raster: {int((np.abs(a_old - a_new).max(2) > 0).sum())} px differ, all where the "
      f"untrained marks were drawn")
# the whole plotting area takes the new raster: the installed PNG's dashes sit at other phases than the PDF's, so a
# patch whose edge runs along a kept dashed line (as the removed dashed line did) would leave broken dash chains
regE = np.zeros(gray.shape, bool)
regE[gy0:gy1, gx0:gx1] = True
mag = lambda im: (im[..., 0] - im[..., 1] > 30) & (im[..., 2] - im[..., 1] > 30)
res[gy0:gy1, gx0:gx1] = a_new
allowed |= regE
print(f"  plotting area replaced ({wE} x {hE} px); {len(np.unique(a_new.reshape(-1, 3), axis=0))} colours in the new "
      f"raster, {len(np.unique(old_blk.reshape(-1, 3), axis=0))} in the installed area")

r16 = res.astype(int)
changed = np.abs(r16 - img.astype(int)).max(2) > 0
if (changed & ~allowed).any():
    sys.exit("PNG changed outside the panel d bands and the panel e replaced area; stop")
rowsD = slice(int(RD.y0 * S), int(RD.y1 * S))
magD = (int(mag(img.astype(int))[rowsD].sum()), int(mag(r16)[rowsD].sum()))
magE = (int((mag(img.astype(int)) & regE).sum()), int((mag(r16) & regE).sum()))
ys, xs = np.nonzero(changed)
print(f"PNG: {int(changed.sum())} pixels changed, all within x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()} px (panel d "
      f"spans y {rowsD.start}-{rowsD.stop}, panel e y {int(RE.y0 * S)}-{int(RE.y1 * S)}, x < {int(RE.x1 * S)}); "
      f"untrained-coloured pixels: panel d {magD[0]} -> {magD[1]}, panel e {magE[0]} -> {magE[1]}")
if magD[1] or magE[1]:
    sys.exit("untrained-coloured pixels remain; stop")
meta = PngImagePlugin.PngInfo()
for k, v in info.items():
    meta.add_text(k, v)
Image.fromarray(res).save(out_png, dpi=(300, 300), pnginfo=meta)
print(f"written {out_png} ({Wp} x {Hp} px, size unchanged)")
