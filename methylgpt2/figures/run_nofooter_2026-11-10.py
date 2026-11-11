"""Run a figure script and also write <stem>_nofooter.pdf / .png: the same gated figure without its provenance footer.

The manuscript uses the no-footer renders (see export_manuscript_pngs_2026-10-02.py). The copy of
nature_figure_utils.py in this folder has no NATURE_FIGURE_NOFOOTER switch, so this wrapper lets save_figure()
gate and save the figure as usual, then removes the footer text and saves the same figure again with the same
tight bounding box settings.

Run: MPLBACKEND=Agg python run_nofooter_2026-11-10.py <script.py> [script args]
"""
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nature_figure_utils as U  # noqa: E402

_save = U.save_figure


def save_both(fig, path, *args, **kw):
    res = _save(fig, path, *args, **kw)
    for t in list(fig.texts):
        if getattr(t, "_ying_footer", False):
            t.remove()
    p = Path(str(path))
    out = p.with_name(p.stem + "_nofooter" + p.suffix)
    fig.savefig(str(out), dpi=kw.get("dpi", 300), bbox_inches="tight")
    fig.savefig(str(out.with_suffix(".png")), dpi=300, bbox_inches="tight")
    print("wrote", out)
    return res


U.save_figure = save_both
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
